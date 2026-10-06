import numpy as np
import pytest
from test_shared_map_frames import (
    CAMERA_HEIGHT,
    FX,
    HFOV,
    MAX_DEPTH,
    MIN_DEPTH,
    R1_START,
    R1_YAW,
    R2_START,
    R2_YAW,
    habitat_gps_compass,
    render_depth,
)

from vlfm.multi_robot.team_stats import COVERAGE_CURVE_LENGTH, COVERAGE_INTERVAL, TeamStats, min_pairwise_distance
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix

MAP_KWARGS = {"min_height": 0.61, "max_height": 0.88, "agent_radius": 0.18, "area_thresh": 1.5}


def view(stats: TeamStats, robot: int, position: np.ndarray, yaw: float) -> None:
    """The robot's view of the test wall, both robots relative to R1's start."""
    gps, compass = habitat_gps_compass(position, yaw, R1_START, R1_YAW)
    tf = xyz_yaw_to_tf_matrix(np.array([gps[0], -gps[1], CAMERA_HEIGHT]), compass)
    depth = render_depth(position, yaw)[..., 0]
    stats.add_view(robot, depth, tf, MIN_DEPTH, MAX_DEPTH, FX, FX, np.deg2rad(HFOV))


def new_stats(num_robots: int = 2) -> TeamStats:
    stats = TeamStats(num_robots, MAP_KWARGS)
    return stats


def test_same_view_twice_is_fully_repeated() -> None:
    stats = new_stats()
    view(stats, 0, R1_START, R1_YAW)
    view(stats, 1, R1_START, R1_YAW)
    summary = stats.summary()
    assert summary["coverage_m2"] > 1.0
    assert summary["coverage_r1_m2"] == summary["coverage_r2_m2"] == summary["coverage_m2"]
    assert summary["repeated_coverage"] == pytest.approx(1.0)


def test_one_robot_covers_and_repeats_nothing() -> None:
    stats = new_stats()
    view(stats, 0, R1_START, R1_YAW)
    summary = stats.summary()
    assert summary["coverage_r2_m2"] == 0.0 and summary["repeated_coverage"] == 0.0


def test_two_views_of_the_same_wall_overlap_partly() -> None:
    stats = new_stats()
    view(stats, 0, R1_START, R1_YAW)
    view(stats, 1, R2_START, R2_YAW)
    summary = stats.summary()
    assert summary["coverage_m2"] > max(summary["coverage_r1_m2"], summary["coverage_r2_m2"])
    assert 0.0 < summary["repeated_coverage"] < 1.0
    # Team coverage counts the overlap once
    overlap = summary["repeated_coverage"] * summary["coverage_m2"]
    assert summary["coverage_r1_m2"] + summary["coverage_r2_m2"] - overlap == pytest.approx(summary["coverage_m2"])


def test_frontier_overlap_counts_joint_steps_where_both_explore() -> None:
    stats = new_stats()
    decisions = [
        (("explore", [0.0, 0.0]), ("explore", [0.3, 0.0])),  # 0.3 m apart
        (("explore", [0.0, 0.0]), ("explore", [1.5, 0.0])),  # 1.5 m
        (("explore", [0.0, 0.0]), ("explore", [5.0, 0.0])),  # 5 m
        (("explore", [0.0, 0.0]), ("navigate", None)),  # not both exploring
        (("initialize", None), ("initialize", None)),
    ]
    for r1, r2 in decisions:
        stats.add_decision(0, *r1)
        stats.add_decision(1, *r2)
    summary = stats.summary()
    assert summary["all_explore_steps"] == 3
    assert summary["frontier_overlap_0p5m"] == pytest.approx(1 / 3)
    assert summary["frontier_overlap_1m"] == pytest.approx(1 / 3)
    assert summary["frontier_overlap_2m"] == pytest.approx(2 / 3)


def test_single_robot_has_no_frontier_overlap() -> None:
    stats = new_stats(num_robots=1)
    for _ in range(3):
        stats.add_decision(0, "explore", [0.0, 0.0])
    summary = stats.summary()
    assert summary["all_explore_steps"] == 0 and summary["frontier_overlap_1m"] == 0.0


def test_coverage_curve_samples_joint_steps_and_has_a_fixed_length() -> None:
    stats = new_stats()
    for step in range(25):
        if step == 12:
            view(stats, 0, R1_START, R1_YAW)  # seen in joint step 13
        stats.add_decision(0, "explore", [0.0, 0.0])
        stats.add_decision(1, "explore", [9.0, 0.0])
    curve = stats.coverage_curve()
    assert len(curve) == COVERAGE_CURVE_LENGTH and COVERAGE_INTERVAL == 10
    assert curve[0] == 0.0  # joint step 10
    assert curve[1] > 0.0  # joint step 20
    assert all(value == curve[1] for value in curve[2:])  # after the end: final coverage


def test_reset_clears_everything() -> None:
    stats = new_stats()
    view(stats, 0, R1_START, R1_YAW)
    stats.add_decision(0, "explore", [0.0, 0.0])
    stats.add_decision(1, "explore", [0.0, 0.0])
    stats.reset()
    summary = stats.summary()
    assert summary["coverage_m2"] == 0.0 and summary["all_explore_steps"] == 0


def test_min_pairwise_distance() -> None:
    points = [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([0.0, 1.0])]
    assert min_pairwise_distance(points) == 1.0
    assert min_pairwise_distance(points[:1]) == np.inf
