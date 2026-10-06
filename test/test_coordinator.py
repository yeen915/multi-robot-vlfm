from typing import Any, Dict, Optional

import numpy as np
import pytest
from test_path_cost import make_map

from vlfm.multi_robot.coordinator import Coordinator, CoordinatorConfig

ROOM = make_map([(-10.0, 10.0, -10.0, 10.0)])  # open, everything reachable
A, A_NEAR, FAR = np.array([5.0, 0.0]), np.array([5.0, 1.0]), np.array([-5.0, 0.0])


def make(assigner: str, **kwargs: Any) -> Coordinator:
    # Values dominate unless a test says otherwise: one value point is worth 10 path actions
    settings = dict(lambda_path=1.0, d_min=2.0, hysteresis_tau=0.1, value_scale=1.0, path_scale=10.0)
    settings.update(kwargs)
    config = CoordinatorConfig(assigner=assigner, **settings)
    coordinator = Coordinator(config)
    coordinator.poses = {0: (np.array([0.0, 0.0]), 0.0), 1: (np.array([0.0, 0.5]), 0.0)}
    return coordinator


def ask(
    c: Coordinator,
    robot: int,
    frontiers: list,
    values: list,
    exploring: Optional[set] = None,
    occupied: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    return c.choose(robot, np.array(frontiers), np.array(values), ROOM, exploring or {0, 1}, occupied)


def near(goal: Optional[np.ndarray], point: np.ndarray) -> bool:
    return goal is not None and np.linalg.norm(goal - point) < 1e-6


# Spacing ----------------------------------------------------------------------------

def test_joint_sends_the_robots_apart_where_independent_utility_does_not() -> None:
    frontiers, values = [A, A_NEAR, FAR], [3.0, 2.9, 2.0]
    independent = make("independent_utility")
    assert near(ask(independent, 0, frontiers, values), A)
    assert near(ask(independent, 1, frontiers, values), A)  # both go to the best one

    joint = make("joint")
    ask(joint, 0, frontiers, values)
    goals = [joint.goals[0], joint.goals[1]]
    assert any(near(g, A) for g in goals) and any(near(g, FAR) for g in goals)  # A_NEAR is too close to A
    assert np.linalg.norm(goals[0] - goals[1]) >= 2.0


def test_greedy_takes_the_nearest_frontier_away_from_the_teammate() -> None:
    greedy = make("greedy")
    close, far_but_valuable = np.array([1.0, 0.0]), np.array([8.0, 0.0])
    assert near(ask(greedy, 0, [close, far_but_valuable], [0.0, 9.0]), close)  # values ignored
    # The teammate stays clear of R1's goal
    assert near(ask(greedy, 1, [close, far_but_valuable], [0.0, 9.0]), far_but_valuable)


def test_joint_keeps_away_from_the_navigating_robot() -> None:
    joint = make("joint")
    goal = ask(joint, 0, [A, FAR], [3.0, 2.0], exploring={0}, occupied={1: np.array([5.0, 0.5])})
    assert near(goal, FAR)


# Fallbacks ----------------------------------------------------------------------------

def test_one_frontier_goes_to_one_robot_and_the_other_turns_in_place() -> None:
    joint = make("joint")
    ask(joint, 0, [A], [3.0])
    assert sum(g is None for g in joint.goals.values()) == 1
    assert sum(near(g, A) for g in joint.goals.values()) == 1


def test_all_frontiers_too_close_the_second_robot_ignores_the_spacing() -> None:
    joint = make("joint")
    ask(joint, 0, [A, A_NEAR], [3.0, 2.0])
    assert {tuple(g) for g in joint.goals.values()} == {tuple(A), tuple(A_NEAR)}


def test_no_frontiers_means_no_goal() -> None:
    joint = make("joint")
    assert joint.choose(0, np.zeros((0, 2)), np.zeros(0), ROOM, {0, 1}) is None


def test_teammate_without_a_pose_is_left_out() -> None:
    joint = make("joint")
    joint.poses = {0: (np.array([0.0, 0.0]), 0.0)}
    assert near(ask(joint, 0, [A, FAR], [3.0, 2.0]), A)
    assert 1 not in joint.goals


# Hysteresis -----------------------------------------------------------------------------

@pytest.mark.parametrize("assigner", ["independent_utility", "greedy", "joint"])
def test_small_gains_do_not_switch_the_goal(assigner: str) -> None:
    c = make(assigner)
    first = ask(c, 0, [A, FAR], [3.0, 2.0], exploring={0})
    # The other frontier becomes slightly better: not by more than tau
    if assigner == "greedy":
        # Greedy scores by path cost: move the robot so that FAR is a little closer
        c.poses[0] = (np.array([-0.02, 0.0]), 0.0)
    gain = 0.05
    values = [3.0, 2.0] if assigner == "greedy" else ([3.0, 3.0 + gain] if near(first, A) else [3.0 + gain, 2.0])
    assert near(ask(c, 0, [A, FAR], values, exploring={0}), first)
    assert c.stats["switches"] == 0


@pytest.mark.parametrize("assigner", ["independent_utility", "joint"])
def test_large_gains_switch_the_goal(assigner: str) -> None:
    c = make(assigner)
    assert near(ask(c, 0, [A, FAR], [3.0, 2.0], exploring={0}), A)
    assert near(ask(c, 0, [A, FAR], [3.0, 4.0], exploring={0}), FAR)
    assert c.stats["switches"] == 1


def test_greedy_switches_when_another_frontier_becomes_much_closer() -> None:
    c = make("greedy")
    assert near(ask(c, 0, [A, FAR], [0.0, 0.0], exploring={0}), A)
    c.poses[0] = (np.array([-4.0, 0.0]), np.pi)  # now right next to FAR, facing it
    assert near(ask(c, 0, [A, FAR], [0.0, 0.0], exploring={0}), FAR)


def test_joint_pair_switches_only_beyond_tau() -> None:
    c = make("joint", path_scale=1e9)  # values only, so that the margins are exact
    B = np.array([0.0, 6.0])
    ask(c, 0, [A, FAR, B], [3.0, 2.0, 1.0])
    before = {r: tuple(g) for r, g in c.goals.items()}
    assert set(before.values()) == {tuple(A), tuple(FAR)}
    ask(c, 1, [A, FAR, B], [3.0, 2.0, 2.05])  # B now ties FAR within tau: no change
    assert {r: tuple(g) for r, g in c.goals.items()} == before
    ask(c, 0, [A, FAR, B], [3.0, 2.0, 5.0])  # B is now clearly better
    assert tuple(B) in {tuple(g) for g in c.goals.values()}


def test_moving_frontier_is_still_the_same_goal() -> None:
    c = make("independent_utility")
    ask(c, 0, [A, FAR], [3.0, 2.9], exploring={0})
    moved = A + np.array([0.3, 0.0])  # the frontier shifted as the map grew
    assert near(ask(c, 0, [moved, FAR], [3.0, 3.05], exploring={0}), moved)
    assert c.stats["switches"] == 0


def test_when_one_goal_vanishes_only_that_robot_is_reassigned() -> None:
    c = make("joint")
    B = np.array([0.0, 6.0])
    ask(c, 0, [A, FAR, B], [3.0, 2.0, 1.0])
    keeper = next(r for r, g in c.goals.items() if near(g, A))
    other = 1 - keeper
    # FAR disappears; B is now worth more than A, but the robot heading to A keeps it
    ask(c, 0, [A, B], [3.0, 9.0])
    assert near(c.goals[keeper], A) and near(c.goals[other], B)


# Utility terms and statistics ---------------------------------------------------------

def test_reserved_terms_are_checked() -> None:
    with pytest.raises(ValueError):
        make("joint", term_weights={"unknown": 1.0})
    c = make("joint", term_weights={"info_gain": 0.0})  # weight 0: never evaluated
    ask(c, 0, [A, FAR], [3.0, 2.0])
    c = make("joint", term_weights={"info_gain": 1.0})
    with pytest.raises(NotImplementedError):
        ask(c, 0, [A, FAR], [3.0, 2.0])


def test_unreachable_frontiers_are_penalized_and_counted() -> None:
    two_rooms = make_map([(-1.0, 1.0, -1.0, 1.0), (4.0, 6.0, -1.0, 1.0)])
    c = make("independent_utility", unreachable_penalty=3.0)
    behind_wall, reachable = np.array([5.0, 0.0]), np.array([-0.9, 0.0])
    # Same value; the unreachable one is nearer in a straight line but costs 3x its estimate
    goal = c.choose(0, np.array([behind_wall, reachable]), np.array([1.0, 1.0]), two_rooms, {0})
    assert near(goal, reachable)
    assert c.stats["unreachable_paths"] == 1 and c.stats["path_costs"] == 2


def test_unknown_assigner_is_rejected() -> None:
    with pytest.raises(ValueError):
        Coordinator(CoordinatorConfig(assigner="vlfm"))
