import numpy as np
import pytest

from vlfm.multi_robot.team import RobotPose, RobotTeam, spl, team_spl, winner_spl

IDENTITY = [0.0, 0.0, 0.0, 1.0]


def pose(x: float, z: float = 0.0) -> RobotPose:
    return RobotPose(np.array([x, 0.0, z]), IDENTITY)


def two_robot_team() -> RobotTeam:
    return RobotTeam([pose(0.0), pose(5.0)], start_goal_distances=[8.0, 4.0])


def test_robots_alternate_and_joint_steps_count_rounds() -> None:
    team = two_robot_team()
    assert (team.active, team.joint_steps) == (0, 0)

    assert team.record_action(pose(0.25), stopped=False) == 1
    assert (team.active, team.joint_steps, team.num_actions) == (1, 1, [1, 0])

    assert team.record_action(pose(5.25), stopped=False) == 0
    assert (team.active, team.joint_steps, team.num_actions) == (0, 1, [1, 1])

    team.record_action(pose(0.5), stopped=False)
    assert (team.joint_steps, team.num_actions) == (2, [2, 1])


def test_each_robot_accumulates_its_own_path() -> None:
    team = two_robot_team()
    team.record_action(pose(0.25), stopped=False)  # R1 moves 0.25
    team.record_action(pose(5.0), stopped=False)  # R2 turns in place
    team.record_action(pose(0.5), stopped=False)  # R1 moves 0.25
    team.record_action(pose(5.0, 0.25), stopped=False)  # R2 moves 0.25
    assert team.path_lengths == pytest.approx([0.5, 0.25])
    assert team.poses[0].position == pytest.approx([0.5, 0.0, 0.0])
    assert team.poses[1].position == pytest.approx([5.0, 0.0, 0.25])


def test_stop_ends_episode_and_keeps_the_stopper_active() -> None:
    team = two_robot_team()
    team.record_action(pose(0.25), stopped=False)
    assert team.record_action(pose(5.0), stopped=True) is None
    assert (team.stopper, team.active, team.joint_steps) == (1, 1, 1)
    with pytest.raises(AssertionError):
        team.record_action(pose(0.5), stopped=False)


def test_single_robot_stays_active() -> None:
    team = RobotTeam([pose(0.0)], start_goal_distances=[3.0])
    assert team.record_action(pose(0.25), stopped=False) == 0
    assert team.record_action(pose(0.5), stopped=False) == 0
    assert (team.joint_steps, team.min_robot_distance, team.num_close_steps) == (2, float("inf"), 0)


def test_close_steps_are_counted() -> None:
    team = RobotTeam([pose(0.0), pose(0.5)], collision_distance=0.36)
    team.record_action(pose(0.25), stopped=False)  # 0.25 m apart
    team.record_action(pose(0.5), stopped=False)  # still 0.25 m apart
    team.record_action(pose(0.0), stopped=False)  # 0.5 m apart
    assert team.num_close_steps == 2
    assert team.min_robot_distance == pytest.approx(0.25)


def test_spl() -> None:
    assert spl(1.0, shortest=4.0, path_length=8.0) == pytest.approx(0.5)
    assert spl(1.0, shortest=4.0, path_length=2.0) == pytest.approx(1.0)  # capped at 1
    assert spl(0.0, shortest=4.0, path_length=8.0) == 0.0
    assert spl(1.0, shortest=0.0, path_length=0.0) == 1.0


def test_team_spl_variants() -> None:
    team = two_robot_team()  # L1 = 8, L2 = 4
    team.path_lengths = [6.0, 10.0]
    team.stopper = 1
    # L = min(L1, L2) = 4 over P1 + P2 = 16
    assert team_spl(1.0, team) == pytest.approx(4.0 / 16.0)
    # L = L1 = 8 over P1 + P2 = 16
    assert team_spl(1.0, team, use_r1_start=True) == pytest.approx(8.0 / 16.0)
    # The stopper's own L2 = 4 over its own P2 = 10
    assert winner_spl(1.0, team) == pytest.approx(4.0 / 10.0)
    assert team_spl(0.0, team) == 0.0


def test_winner_spl_without_stop_is_zero() -> None:
    team = two_robot_team()
    assert winner_spl(0.0, team) == 0.0


def test_single_robot_spl_variants_agree() -> None:
    team = RobotTeam([pose(0.0)], start_goal_distances=[5.0])
    team.path_lengths = [10.0]
    team.stopper = 0
    assert team_spl(1.0, team) == team_spl(1.0, team, use_r1_start=True) == winner_spl(1.0, team) == 0.5


def test_team_steps_count_time_and_total_actions_count_every_action() -> None:
    team = two_robot_team()
    for _ in range(6):  # three joint steps: R1, R2, R1, R2, R1, R2
        team.record_action(team.poses[team.active], stopped=False)
    team.record_action(team.poses[0], stopped=True)  # R1 starts a fourth and stops
    assert team.num_actions == [4, 3]
    assert team.joint_steps == 4
    assert team.total_actions == 7
