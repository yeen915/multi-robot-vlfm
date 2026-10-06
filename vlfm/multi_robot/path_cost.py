"""Path cost from a robot to frontiers, in actions (design doc 3.4).

Paths are planned on the shared obstacle map, through cells that are explored and
navigable: unknown space may hide walls, so a path through it would be optimistic.
The robot and every goal are first snapped to the nearest such cell.

    cost = forward actions + turn actions
         = path length / 0.25 m + (initial turn + turns at the path's corners) / 30°

The corners are those of the path simplified to a polyline, so that the zigzag of a
grid path does not count as many small turns. A goal that cannot be reached is
estimated as a straight line (initial turn + straight distance) times a penalty.
"""

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

FORWARD_STEP = 0.25  # meters per forward action
TURN_ANGLE = np.deg2rad(30.0)  # radians per turn action
SIMPLIFY_EPSILON_PX = 2.0  # polyline simplification tolerance (pixels)
CROP_MARGIN_M = 2.0  # margin around the explored area that paths are planned in


@dataclass
class PathCost:
    actions: float  # forward + turn actions (for an unreachable goal: the penalized estimate)
    reachable: bool
    forward: float  # forward actions
    turns: float  # turn actions


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def straight_line_cost(start_xy: np.ndarray, start_yaw: float, goal_xy: np.ndarray) -> PathCost:
    """Turn toward the goal, then drive straight to it."""
    delta = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    distance = float(np.linalg.norm(delta))
    turn = abs(wrap_angle(np.arctan2(delta[1], delta[0]) - start_yaw)) if distance > 0 else 0.0
    forward, turns = distance / FORWARD_STEP, turn / TURN_ANGLE
    return PathCost(forward + turns, True, forward, turns)


def polyline_turns(points_xy: np.ndarray, start_yaw: float) -> float:
    """Turn actions along a polyline: turning toward the first segment, then at each
    corner."""
    segments = np.diff(points_xy, axis=0)
    segments = segments[np.linalg.norm(segments, axis=1) > 1e-9]
    if len(segments) == 0:
        return 0.0
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    total = abs(wrap_angle(headings[0] - start_yaw))
    total += sum(abs(wrap_angle(b - a)) for a, b in zip(headings[:-1], headings[1:]))
    return total / TURN_ANGLE


class PathPlanner:
    """Plans on one snapshot of an obstacle map; reuse it for every robot and frontier
    of one decision."""

    def __init__(self, obstacle_map: object, unreachable_penalty: float = 3.0) -> None:
        """
        Args:
            obstacle_map: An ObstacleMap (explored_area, _navigable_map, _xy_to_px,
                _px_to_xy and pixels_per_meter are used).
            unreachable_penalty: Factor applied to the straight-line estimate of a goal
                that cannot be reached.
        """
        self._map = self.obstacle_map = obstacle_map
        self._penalty = unreachable_penalty
        self._ppm = obstacle_map.pixels_per_meter
        traversable = obstacle_map.explored_area.astype(bool) & obstacle_map._navigable_map.astype(bool)
        rows, cols = np.nonzero(traversable)
        if len(rows) == 0:
            self._traversable = None
            return
        margin = int(CROP_MARGIN_M * self._ppm)
        height, width = traversable.shape
        self._r0, self._c0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
        r1, c1 = min(rows.max() + margin + 1, height), min(cols.max() + margin + 1, width)
        self._traversable = traversable[self._r0 : r1, self._c0 : c1]
        # For every cell of the crop, the nearest traversable cell
        _, self._nearest = ndimage.distance_transform_edt(~self._traversable, return_indices=True)
        self._costs = np.where(self._traversable, 1.0, np.inf)

    def _to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(row, col) in the crop of the traversable cell nearest to `xy`."""
        px = self._map._xy_to_px(np.reshape(np.asarray(xy, dtype=float), (1, 2)))[0]
        row = int(np.clip(px[1] - self._r0, 0, self._traversable.shape[0] - 1))
        col = int(np.clip(px[0] - self._c0, 0, self._traversable.shape[1] - 1))
        return self._nearest[:, row, col]

    def _cells_to_xy(self, cells: np.ndarray) -> np.ndarray:
        px = np.stack([cells[:, 1] + self._c0, cells[:, 0] + self._r0], axis=1)
        return self._map._px_to_xy(px)

    def snap(self, xy: np.ndarray) -> np.ndarray:
        """The center of the traversable cell nearest to `xy` (unchanged if there is
        no traversable cell)."""
        if self._traversable is None:
            return np.asarray(xy, dtype=float)
        return self._cells_to_xy(self._to_cell(xy)[None])[0]

    def costs(self, start_xy: np.ndarray, start_yaw: float, goals_xy: Sequence[np.ndarray]) -> List[PathCost]:
        """Action cost from the robot pose to each goal."""
        if len(goals_xy) == 0:
            return []
        if self._traversable is None:
            return [self._unreachable(start_xy, start_yaw, goal) for goal in goals_xy]
        start = tuple(self._to_cell(start_xy))
        mcp = MCP_Geometric(self._costs, fully_connected=True)
        cumulative, _ = mcp.find_costs([start])
        results = []
        for goal_xy in goals_xy:
            goal = tuple(self._to_cell(goal_xy))
            length_px = cumulative[goal]
            if not np.isfinite(length_px):
                results.append(self._unreachable(start_xy, start_yaw, goal_xy))
                continue
            path = np.array(mcp.traceback(goal))
            corners = cv2.approxPolyDP(path[:, None, ::-1].astype(np.int32), SIMPLIFY_EPSILON_PX, closed=False)
            corners_xy = self._cells_to_xy(corners[:, 0, ::-1])
            forward = float(length_px) / self._ppm / FORWARD_STEP
            turns = polyline_turns(corners_xy, start_yaw)
            results.append(PathCost(forward + turns, True, forward, turns))
        return results

    def _unreachable(self, start_xy: np.ndarray, start_yaw: float, goal_xy: np.ndarray) -> PathCost:
        estimate = straight_line_cost(start_xy, start_yaw, goal_xy)
        return PathCost(estimate.actions * self._penalty, False, estimate.forward, estimate.turns)
