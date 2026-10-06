"""Exploration statistics of a robot team, logged per episode (design doc 3.7).

Measured the same way whatever the robots share, so that map sharing modes can be
compared:

- Coverage: a separate obstacle map that every robot's depth goes into (it is never
  used for decisions). Each robot's covered area is the union of what it has seen
  in it, by VLFM's own rule for "explored" (field of view, up to the first
  obstacle). Team coverage is the union over robots; repeated coverage is the
  fraction of team coverage that more than one robot has seen.
- Frontier overlap: among the joint steps in which every robot explores, the fraction
  in which two robots head for frontiers closer than a threshold.
"""

from typing import Any, Dict, List, Sequence

import numpy as np

from vlfm.mapping.obstacle_map import ObstacleMap

COVERAGE_INTERVAL = 10  # joint steps between two points of the coverage curve
COVERAGE_CURVE_LENGTH = 50  # covers a 500 joint step budget
FRONTIER_OVERLAP_THRESHOLDS = (0.5, 1.0, 2.0)  # meters


class TeamStats:
    def __init__(self, num_robots: int, obstacle_map_kwargs: Dict[str, Any]) -> None:
        """
        Args:
            num_robots: Number of robots in the team.
            obstacle_map_kwargs: Arguments of the coverage map's ObstacleMap, the same
                as the robots' maps.
        """
        self._num_robots = num_robots
        self._map = ObstacleMap(**obstacle_map_kwargs)
        self._pixel_area = 1.0 / self._map.pixels_per_meter**2
        self.reset()

    def reset(self) -> None:
        self._map.reset()
        self._seen = [np.zeros_like(self._map._map, dtype=bool) for _ in range(self._num_robots)]
        self._joint_steps = 0
        self._curve: List[float] = []
        self._explore_steps = 0
        self._overlap_steps = {threshold: 0 for threshold in FRONTIER_OVERLAP_THRESHOLDS}
        self._modes: List[Any] = [None] * self._num_robots
        self._frontiers: List[Any] = [None] * self._num_robots

    def add_view(
        self,
        robot: int,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
    ) -> None:
        """Adds what `robot` sees in this step (depth normalized to [0, 1])."""
        self._map.update_map(depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy, topdown_fov, explore=False)
        self._seen[robot] |= self._map.visible_area(tf_camera_to_episodic, max_depth, topdown_fov) > 0

    def add_decision(self, robot: int, mode: Any, frontier: Any) -> None:
        """Records the robot's mode in this step and, when exploring, its frontier.
        The joint step ends with the last robot's decision."""
        self._modes[robot] = mode
        self._frontiers[robot] = None if frontier is None else np.asarray(frontier, dtype=float)[:2]
        if robot != self._num_robots - 1:
            return
        self._joint_steps += 1
        if self._joint_steps % COVERAGE_INTERVAL == 0 and len(self._curve) < COVERAGE_CURVE_LENGTH:
            self._curve.append(self.coverage())
        if self._num_robots > 1 and all(m == "explore" for m in self._modes):
            self._explore_steps += 1
            distance = min_pairwise_distance([f for f in self._frontiers if f is not None])
            for threshold in FRONTIER_OVERLAP_THRESHOLDS:
                self._overlap_steps[threshold] += int(distance < threshold)

    def _covered(self) -> List[np.ndarray]:
        # Pixels later found to be obstacles are not floor that was covered
        navigable = self._map._navigable_map.astype(bool)
        return [seen & navigable for seen in self._seen]

    def coverage(self) -> float:
        """Area seen by at least one robot, in square meters."""
        return float(np.logical_or.reduce(self._covered()).sum() * self._pixel_area)

    def summary(self) -> Dict[str, Any]:
        """Scalars for the episode's JSON (the same keys in every step and mode)."""
        covered = self._covered()
        union = np.logical_or.reduce(covered)
        seen_by_several = np.sum(covered, axis=0) > 1
        stats: Dict[str, Any] = {
            "coverage_m2": float(union.sum() * self._pixel_area),
            "repeated_coverage": float(seen_by_several.sum() / union.sum()) if union.any() else 0.0,
            "all_explore_steps": self._explore_steps,
        }
        for robot, robot_covered in enumerate(covered):
            stats[f"coverage_r{robot + 1}_m2"] = float(robot_covered.sum() * self._pixel_area)
        for threshold, count in self._overlap_steps.items():
            key = f"frontier_overlap_{threshold:g}m".replace(".", "p")
            stats[key] = count / self._explore_steps if self._explore_steps else 0.0
        return stats

    def coverage_curve(self) -> List[float]:
        """Team coverage every COVERAGE_INTERVAL joint steps, with a fixed length: after
        the last sample, the current coverage is repeated (so once the episode has
        ended, its final coverage)."""
        return self._curve + [self.coverage()] * (COVERAGE_CURVE_LENGTH - len(self._curve))


def min_pairwise_distance(points: Sequence[np.ndarray]) -> float:
    """Smallest distance between two of the points; infinite with fewer than two."""
    best = np.inf
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = min(best, float(np.linalg.norm(points[i] - points[j])))
    return best
