# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List

import numpy as np

from vlfm.mapping.traj_visualizer import TrajectoryVisualizer

# Trajectory color of each robot. The first one is VLFM's original green.
TRAJECTORY_COLORS = [(0, 255, 0), (255, 0, 255)]


class BaseMap:
    _map_dtype: np.dtype = np.dtype(np.float32)

    def __init__(self, size: int = 1000, pixels_per_meter: int = 20, *args: Any, **kwargs: Any):
        """
        Args:
            size: The size of the map in pixels.
        """
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self._map = np.zeros((size, size), dtype=self._map_dtype)
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self._reset_trajectories()

    def reset(self) -> None:
        self._map.fill(0)
        self._reset_trajectories()

    def _reset_trajectories(self) -> None:
        # One trajectory per robot; robots other than the first only appear when
        # several robots share the map
        self._camera_positions: Dict[int, List[np.ndarray]] = {}
        self._last_camera_yaw: Dict[int, float] = {}
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
        self._robot_traj_vis: Dict[int, TrajectoryVisualizer] = {0: self._traj_vis}

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float, robot_id: int = 0) -> None:
        self._camera_positions.setdefault(robot_id, []).append(robot_xy)
        self._last_camera_yaw[robot_id] = robot_heading

    def _draw_trajectories(self, img: np.ndarray) -> np.ndarray:
        """Draws each robot's trajectory and current pose on the image."""
        for robot_id in sorted(self._camera_positions):
            if robot_id not in self._robot_traj_vis:
                traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)
                traj_vis.path_color = TRAJECTORY_COLORS[robot_id % len(TRAJECTORY_COLORS)]
                self._robot_traj_vis[robot_id] = traj_vis
            self._robot_traj_vis[robot_id].draw_trajectory(
                img, self._camera_positions[robot_id], self._last_camera_yaw[robot_id]
            )
        return img

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]
