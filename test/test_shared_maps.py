"""Map behaviour that matters when several robots update the same map."""

import numpy as np

from vlfm.mapping.base_map import TRAJECTORY_COLORS, BaseMap
from vlfm.mapping.obstacle_map import keep_regions_with_robots


def two_rooms() -> np.ndarray:
    explored = np.zeros((100, 100), dtype=bool)
    explored[10:30, 10:30] = True  # room A
    explored[60:90, 60:90] = True  # room B
    return explored


def test_one_robot_keeps_only_its_own_region() -> None:
    kept = keep_regions_with_robots(two_rooms(), [np.array([20, 20])])
    assert kept[20, 20] and not kept[75, 75]


def test_teammate_region_is_kept() -> None:
    kept = keep_regions_with_robots(two_rooms(), [np.array([20, 20]), np.array([75, 75])])
    assert np.array_equal(kept, two_rooms())


def test_robot_outside_every_region_keeps_the_nearest() -> None:
    kept = keep_regions_with_robots(two_rooms(), [np.array([95, 95])])
    assert kept[75, 75] and not kept[20, 20]


def test_single_region_is_returned_unchanged() -> None:
    explored = np.zeros((50, 50), dtype=bool)
    explored[5:20, 5:20] = True
    assert keep_regions_with_robots(explored, [np.array([40, 40])]) is explored


def test_each_robot_has_its_own_trajectory() -> None:
    base_map = BaseMap(size=100)
    base_map.reset()
    base_map.update_agent_traj(np.array([0.0, 0.0]), 0.0)
    base_map.update_agent_traj(np.array([1.0, 0.0]), 0.5, robot_id=1)
    base_map.update_agent_traj(np.array([0.5, 0.0]), 0.1)
    assert [p.tolist() for p in base_map._camera_positions[0]] == [[0.0, 0.0], [0.5, 0.0]]
    assert [p.tolist() for p in base_map._camera_positions[1]] == [[1.0, 0.0]]
    assert base_map._last_camera_yaw == {0: 0.1, 1: 0.5}

    img = base_map._draw_trajectories(np.zeros((100, 100, 3), dtype=np.uint8))
    colors = {tuple(c) for c in img.reshape(-1, 3)}
    # Robot 0's path keeps VLFM's green; robot 1 has its own color
    assert TRAJECTORY_COLORS[0] in colors
    assert base_map._robot_traj_vis[1].path_color == TRAJECTORY_COLORS[1]

    base_map.reset()
    assert base_map._camera_positions == {}
