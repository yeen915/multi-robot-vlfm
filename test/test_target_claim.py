import numpy as np
import pytest

from vlfm.multi_robot.target_claim import TargetClaim

TARGET = np.array([[10.0, 0.0, 0.5], [10.0, 1.0, 0.5]])


def make_claim(r1_xy: list, r2_xy: list) -> TargetClaim:
    claim = TargetClaim()
    claim.positions = {0: np.array(r1_xy), 1: np.array(r2_xy)}
    return claim


def test_the_closer_robot_navigates_even_if_the_other_one_asks() -> None:
    claim = make_claim([0.0, 0.0], [8.0, 0.0])
    assert not claim.may_navigate(0, TARGET)
    assert claim.robot == 1
    assert claim.may_navigate(1, TARGET)


def test_distance_is_to_the_nearest_target_point() -> None:
    # R1 is 3 m from the first point; R2 is 4 m from the second and 4.1 m from the first
    claim = make_claim([7.0, 0.0], [10.0, 5.0])
    assert claim.may_navigate(0, TARGET)


def test_the_claim_holds_until_released() -> None:
    claim = make_claim([0.0, 0.0], [8.0, 0.0])
    claim.may_navigate(0, TARGET)
    claim.positions[0] = np.array([9.5, 0.0])  # R1 is now closer, but R2 keeps the target
    assert claim.may_navigate(1, TARGET) and not claim.may_navigate(0, TARGET)
    claim.release()
    assert claim.may_navigate(0, TARGET)


def test_ties_go_to_the_first_robot() -> None:
    claim = make_claim([9.0, 0.0], [11.0, 0.0])
    assert claim.may_navigate(1, TARGET) is False
    assert claim.robot == 0


def test_a_robot_whose_teammate_is_unknown_claims_the_target() -> None:
    claim = TargetClaim()
    claim.positions = {0: np.array([0.0, 0.0])}
    assert claim.may_navigate(0, TARGET)


def test_the_asking_robot_must_have_a_position() -> None:
    claim = TargetClaim()
    claim.positions = {1: np.array([0.0, 0.0])}
    with pytest.raises(ValueError):
        claim.may_navigate(0, TARGET)


def test_reset_forgets_positions_and_claim() -> None:
    claim = make_claim([0.0, 0.0], [8.0, 0.0])
    claim.may_navigate(0, TARGET)
    claim.reset()
    assert claim.robot is None and claim.positions == {}


def navigate_steps(claim: TargetClaim, robot: int, xys: list) -> list:
    """The robot moves through the given positions, asking at each one."""
    answers = []
    for xy in xys:
        claim.positions[robot] = np.array(xy)
        answers.append(claim.may_navigate(robot, TARGET))
    return answers


def test_a_stuck_navigator_hands_the_target_to_the_other_robot() -> None:
    claim = TargetClaim(timeout_steps=50)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    # R2 claims the target, makes progress once, then stays at the same distance
    answers = navigate_steps(claim, 1, [[8.0, 0.0], [8.5, 0.0]] + [[8.5, 0.0]] * 50)
    assert answers[:-1] == [True] * 51
    assert answers[-1] is False  # 50 steps without progress: it is R1's now
    assert claim.robot == 0 and claim.num_timeouts == 1


def test_steady_progress_keeps_the_claim() -> None:
    claim = TargetClaim(timeout_steps=5)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([2.0, 0.0])}
    # Turning in place between forward steps: the distance only drops every 4th step
    xys = []
    for i in range(1, 7):
        xys += [[2.0 + 0.25 * i, 0.0]] + [[2.0 + 0.25 * i, 0.0]] * 3
    assert all(navigate_steps(claim, 1, xys))
    assert claim.num_timeouts == 0


def test_tiny_progress_does_not_count() -> None:
    claim = TargetClaim(timeout_steps=3)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    # 1 cm per step: under the 5 cm needed, measured against the best distance so far
    answers = navigate_steps(claim, 1, [[8.0, 0.0], [8.01, 0.0], [8.02, 0.0], [8.03, 0.0]])
    assert answers == [True, True, True, False]


def test_moving_back_and_forth_is_not_progress() -> None:
    claim = TargetClaim(timeout_steps=4)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    answers = navigate_steps(claim, 1, [[8.0, 0.0], [7.5, 0.0], [8.0, 0.0], [7.5, 0.0], [8.0, 0.0]])
    assert answers == [True, True, True, True, False]


def test_after_a_second_timeout_the_target_goes_back() -> None:
    claim = TargetClaim(timeout_steps=2)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    navigate_steps(claim, 1, [[8.0, 0.0]] * 3)
    assert claim.robot == 0
    navigate_steps(claim, 0, [[0.0, 0.0]] * 3)
    assert claim.robot == 1 and claim.num_timeouts == 2


def test_a_lone_robot_keeps_trying_after_a_timeout() -> None:
    claim = TargetClaim(timeout_steps=2)
    claim.positions = {0: np.array([0.0, 0.0])}
    assert navigate_steps(claim, 0, [[0.0, 0.0]] * 3) == [True, True, True]
    assert claim.num_timeouts == 1


def test_reset_clears_the_timeout_count() -> None:
    claim = TargetClaim(timeout_steps=1)
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    navigate_steps(claim, 1, [[8.0, 0.0]] * 2)
    claim.reset()
    assert claim.num_timeouts == 0
