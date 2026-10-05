"""Which robot goes to the target when the robots share their object map.

Design doc 3.5, option a: once the target is in the shared map, the robot closest to
it navigates there and calls STOP; the other robots keep exploring.
"""

from typing import Dict, Optional

import numpy as np

# Getting closer to the target by less than this does not count as progress, so that
# jitter in the target's point cloud does not keep a stuck robot's claim alive
MIN_PROGRESS = 0.05  # meters


class TargetClaim:
    def __init__(self, timeout_steps: int = 50) -> None:
        """
        Args:
            timeout_steps: The navigating robot loses the target after this many of its
                steps in a row without getting closer to it than it has been.
        """
        self.timeout_steps = timeout_steps
        self.reset()

    def reset(self) -> None:
        # Latest known (x, y) of each robot, in the shared frame
        self.positions: Dict[int, np.ndarray] = {}
        # The robot that navigates to the target, or None while nobody has claimed it
        self.robot: Optional[int] = None
        self.num_timeouts = 0
        self._best_distance = np.inf
        self._steps_without_progress = 0

    def release(self) -> None:
        """Called when the target is no longer in the map."""
        self.robot = None

    def may_navigate(self, robot: int, target_cloud: np.ndarray) -> bool:
        """Whether `robot` should navigate to the target.

        The first time a robot asks, the target goes to the robot that is closest to it
        (straight-line distance to the nearest target point), which may be a robot
        other than the one asking. The choice then holds, so that two robots at similar
        distances do not keep swapping roles, until the target is released or the
        navigating robot times out: then the target goes to the closest of the other
        robots.

        Args:
            robot: The robot that asks.
            target_cloud: The target's points in the shared frame, (N, >=2).
        """
        if robot not in self.positions:
            raise ValueError(f"The position of robot {robot} is unknown")
        if self.robot is None:
            self._assign(target_cloud)
        if self.robot == robot:
            self._track_progress(robot, target_cloud)
        return self.robot == robot

    def _distance(self, robot: int, target_cloud: np.ndarray) -> float:
        xy = np.asarray(self.positions[robot])[:2]
        return float(np.min(np.linalg.norm(target_cloud[:, :2] - xy, axis=1)))

    def _assign(self, target_cloud: np.ndarray, exclude: Optional[int] = None) -> None:
        candidates = [r for r in self.positions if r != exclude] or list(self.positions)
        distances = {r: self._distance(r, target_cloud) for r in candidates}
        # Ties go to the lower robot index
        self.robot = min(distances, key=lambda r: (distances[r], r))
        self._best_distance = np.inf
        self._steps_without_progress = 0

    def _track_progress(self, robot: int, target_cloud: np.ndarray) -> None:
        """Called on each step of the navigating robot."""
        distance = self._distance(robot, target_cloud)
        if distance < self._best_distance - MIN_PROGRESS:
            self._best_distance = distance
            self._steps_without_progress = 0
            return
        self._steps_without_progress += 1
        if self._steps_without_progress >= self.timeout_steps:
            self.num_timeouts += 1
            self._assign(target_cloud, exclude=robot)
