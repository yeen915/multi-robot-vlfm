"""The coordinator wired into the two-robot policy (assigner other than "vlfm")."""

from typing import Any, List, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_multi_robot_policy import FakeRobotPolicy

from vlfm.multi_robot.coordinator import CoordinatorConfig, UtilityScaleLog, values_in_order
from vlfm.multi_robot.path_cost import PathCost
from vlfm.policy.habitat_policies import TorchActionIDs
from vlfm.policy.multi_robot_policy import HabitatMultiRobotITMPolicy, TeamMemberITMPolicy

FORWARD = TorchActionIDs.MOVE_FORWARD


class ModeRobot(FakeRobotPolicy):
    """A fake robot that reports a mode and a goal after acting, and records the team
    status it was given."""

    def __init__(self, modes: List[str], goal: Optional[List[float]] = None) -> None:
        super().__init__([FORWARD] * len(modes))
        self._modes = list(modes)
        self._last_goal = np.array(goal or [0.0, 0.0])
        self._last_frontier = np.zeros(2)
        self.statuses: List[Any] = []

    def act(self, *args: Any) -> Any:
        self.statuses.append(getattr(self, "_team_status", None))
        action_data = super().act(*args)
        self._last_mode = self._modes.pop(0)
        action_data.policy_info = [{}]
        return action_data


def joint_policy(r1: Any, r2: Any) -> HabitatMultiRobotITMPolicy:
    return HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared", coordinator_config=CoordinatorConfig("joint"))


def act(policy: HabitatMultiRobotITMPolicy, robot: int, xy: List[float], yaw: float, new_episode: bool = False) -> Any:
    observations = {
        "active_robot_id": torch.tensor([[robot]]),
        "gps": torch.tensor([[xy[0], -xy[1]]]),  # Habitat's GPS has y pointing right
        "compass": torch.tensor([[yaw]]),
    }
    return policy.act(observations, None, None, torch.tensor([[not new_episode]]))


def test_coordinator_needs_shared_maps() -> None:
    with pytest.raises(ValueError):
        HabitatMultiRobotITMPolicy([FakeRobotPolicy([]), FakeRobotPolicy([])], coordinator_config=CoordinatorConfig())


def test_every_robot_gets_the_same_coordinator() -> None:
    r1, r2 = ModeRobot([]), ModeRobot([])
    policy = joint_policy(r1, r2)
    assert r1._coordinator is r2._coordinator is policy._coordinator


def test_poses_and_team_status_reach_the_coordinator() -> None:
    r1 = ModeRobot(["initialize", "explore", "explore"])
    r2 = ModeRobot(["explore", "navigate"], goal=[4.0, 1.0])
    policy = joint_policy(r1, r2)
    act(policy, 0, [0.0, 0.0], 0.1, new_episode=True)
    act(policy, 1, [1.0, 2.0], 0.2)
    act(policy, 0, [0.5, 0.0], 0.3)
    act(policy, 1, [1.0, 2.5], 0.4)
    act(policy, 0, [0.5, 0.5], 0.5)

    xy, yaw = policy._coordinator.poses[1]
    assert np.allclose(xy, [1.0, 2.5]) and yaw == pytest.approx(0.4)
    # R1's view of R2: not started, then exploring, then navigating to [4, 1]
    assert r1.statuses[0] == (set(), {})
    assert r1.statuses[1] == ({1}, {})
    exploring, occupied = r1.statuses[2]
    assert exploring == set() and np.allclose(occupied[1], [4.0, 1.0])
    # R2's view of R1: turning in place at the start counts as neither
    assert r2.statuses[0] == (set(), {})


def test_coordinator_is_reset_and_logged_per_episode() -> None:
    r1, r2 = ModeRobot(["explore"] * 3), ModeRobot(["explore"] * 2)
    policy = joint_policy(r1, r2)
    act(policy, 0, [0.0, 0.0], 0.0, new_episode=True)
    policy._coordinator.goals[0] = np.array([1.0, 1.0])
    policy._coordinator.stats["switches"] = 3
    info = act(policy, 1, [1.0, 0.0], 0.0).policy_info[0]
    assert info["coordinator"]["switches"] == 3
    act(policy, 0, [0.0, 0.0], 0.0, new_episode=True)
    assert policy._coordinator.goals == {} and policy._coordinator.stats["switches"] == 0


class FakeCoordinator:
    def __init__(self, goal: Optional[np.ndarray]) -> None:
        self.goal = goal
        self.calls: List[Any] = []

    def choose(self, robot: int, frontiers: Any, values: Any, obstacle_map: Any, exploring: Any, occupied: Any) -> Any:
        self.calls.append((robot, np.array(values), exploring, occupied))
        return self.goal


def make_member(goal: Optional[np.ndarray]) -> TeamMemberITMPolicy:
    member = object.__new__(TeamMemberITMPolicy)  # skips the model clients
    member._robot_id = 1
    member._coordinator = FakeCoordinator(goal)
    member._team_status = ({0}, {})
    member._obstacle_map = None
    member._observations_cache = {"frontier_sensor": np.array([[1.0, 0.0], [5.0, 5.0]])}
    # As ValueMap.sort_waypoints: sorted by value, best first
    member._sort_frontiers_by_value = lambda obs, f: (np.array([[5.0, 5.0], [1.0, 0.0]]), [0.3, 0.1])
    member.pointnav_goals = []
    member._pointnav = lambda goal, stop=False: member.pointnav_goals.append(goal) or FORWARD
    return member


def test_team_member_heads_for_its_assigned_frontier() -> None:
    member = make_member(np.array([1.0, 0.0]))
    assert member._explore(None) is FORWARD
    assert np.allclose(member.pointnav_goals[0], [1.0, 0.0])
    robot, values, exploring, _ = member._coordinator.calls[0]
    assert robot == 1 and exploring == {0, 1}
    assert np.allclose(values, [0.1, 0.3])  # back in the frontiers' order
    assert np.allclose(member._last_frontier, [1.0, 0.0]) and member._last_value == pytest.approx(0.1)


def test_team_member_turns_in_place_without_a_frontier() -> None:
    member = make_member(None)
    assert member._explore(None) is TorchActionIDs.TURN_LEFT
    assert member.pointnav_goals == []


def test_values_in_order_and_scale_log() -> None:
    frontiers = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values = values_in_order(np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]), [0.9, 0.5, 0.1], frontiers)
    assert np.allclose(values, [0.1, 0.9, 0.5])

    log = UtilityScaleLog()
    log.add(values, [PathCost(10.0, True, 8, 2), PathCost(30.0, True, 28, 2), PathCost(99.0, False, 30, 3)])
    summary = log.summary()
    assert summary["frontiers"] == 3 and summary["unreachable"] == 1
    assert summary["value_p50"] == pytest.approx(0.5) and summary["actions_p50"] == pytest.approx(20.0)
    # Only two reachable frontiers: too few for a spread
    assert summary["step_iqrs"] == []

    log.add([0.1, 0.2, 0.3, 0.4, 0.5], [PathCost(a, True, int(a), 0) for a in (10.0, 20.0, 30.0, 40.0, 50.0)])
    log.add([0.1, 0.2], [PathCost(10.0, True, 10, 0), PathCost(20.0, True, 20, 0)])  # too few frontiers
    frontiers, value_iqr, actions_iqr = log.summary()["step_iqrs"][0]
    assert len(log.summary()["step_iqrs"]) == 1
    assert frontiers == 5 and value_iqr == pytest.approx(0.2) and actions_iqr == pytest.approx(20.0)
    log.reset()
    assert log.summary()["step_iqrs"] == []


def test_ground_truth_detector_is_reserved() -> None:
    config = OmegaConf.create({"habitat_baselines": {"rl": {"policy": {"detector": "gt", "assigner": "vlfm"}}}})
    with pytest.raises(NotImplementedError):
        HabitatMultiRobotITMPolicy.from_config(config)


def test_scale_log_is_kept_for_any_assigner() -> None:
    policy = HabitatMultiRobotITMPolicy([FakeRobotPolicy([])], log_utility_scales=True)
    assert isinstance(policy._scale_log, UtilityScaleLog) and policy._coordinator is None




def test_previous_episode_modes_are_forgotten() -> None:
    r1 = ModeRobot(["explore", "explore"])
    r2 = ModeRobot(["navigate"], goal=[4.0, 1.0])
    policy = joint_policy(r1, r2)
    act(policy, 0, [0.0, 0.0], 0.0, new_episode=True)
    act(policy, 1, [1.0, 0.0], 0.0)  # R2 ends the episode navigating
    act(policy, 0, [0.0, 0.0], 0.0, new_episode=True)
    # In the new episode R2 has not acted yet: it is neither exploring nor navigating
    assert r1.statuses[-1] == (set(), {})
