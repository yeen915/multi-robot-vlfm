from types import SimpleNamespace
from typing import Any, List

import torch
from habitat_baselines.rl.ppo.policy import PolicyActionData

from vlfm.policy.habitat_policies import TorchActionIDs
from vlfm.policy.multi_robot_policy import HabitatMultiRobotITMPolicy


class FakeRobotPolicy:
    """Returns scripted actions and records the masks it was called with."""

    def __init__(self, actions: List[torch.Tensor], called_stop: bool = False, hw_corruption: bool = False) -> None:
        self._actions = list(actions)
        self._called_stop = called_stop
        self._hw_corruption = hw_corruption
        self._object_map = SimpleNamespace(seeds=[])
        self._object_map.seed = self._object_map.seeds.append
        self.masks: List[int] = []

    def act(self, observations: Any, rnn_hidden_states: Any, prev_actions: Any, masks: torch.Tensor, *args: Any) -> Any:
        self.masks.append(int(masks[0].item()))
        return PolicyActionData(actions=self._actions.pop(0), rnn_hidden_states=rnn_hidden_states)


def step(policy: HabitatMultiRobotITMPolicy, robot: int, new_episode: bool = False) -> int:
    observations = {"active_robot_id": torch.tensor([[robot]])}
    masks = torch.tensor([[not new_episode]])
    return int(policy.act(observations, None, None, masks).actions.item())


FORWARD, STOP, LEFT = TorchActionIDs.MOVE_FORWARD, TorchActionIDs.STOP, TorchActionIDs.TURN_LEFT


def test_steps_are_routed_to_the_active_robot() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD, FORWARD]), FakeRobotPolicy([LEFT])
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    assert [step(policy, 0, new_episode=True), step(policy, 1), step(policy, 0)] == [1, 2, 1]
    assert (len(r1.masks), len(r2.masks)) == (2, 1)


def test_each_robot_is_reset_on_its_first_step_of_an_episode() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 4), FakeRobotPolicy([FORWARD] * 3)
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    step(policy, 0, new_episode=True)
    step(policy, 1)  # R2's first step: the env mask is 1, but R2 must still reset
    step(policy, 0)
    step(policy, 1)
    step(policy, 0, new_episode=True)
    step(policy, 1)
    assert r1.masks == [0, 1, 0]
    assert r2.masks == [0, 1, 0]


def test_giving_up_does_not_end_the_episode_while_the_other_robot_searches() -> None:
    r1, r2 = FakeRobotPolicy([STOP, STOP]), FakeRobotPolicy([FORWARD, STOP])
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    assert step(policy, 0, new_episode=True) == 2  # R1 gives up: turn in place instead
    assert step(policy, 1) == 1
    assert step(policy, 0) == 2
    assert step(policy, 1) == 0  # both gave up: the episode ends


def test_stop_at_the_goal_is_passed_through() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD]), FakeRobotPolicy([STOP], called_stop=True)
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    step(policy, 0, new_episode=True)
    assert step(policy, 1) == 0


def test_single_robot_stops_when_it_gives_up() -> None:
    policy = HabitatMultiRobotITMPolicy([FakeRobotPolicy([STOP])])
    observations = {"rgb": torch.zeros(1)}  # no active_robot_id: single-robot task
    assert int(policy.act(observations, None, None, torch.tensor([[False]])).actions.item()) == 0


def test_corrupted_step_ends_the_episode_instead_of_turning() -> None:
    r1, r2 = FakeRobotPolicy([STOP], hw_corruption=True), FakeRobotPolicy([FORWARD])
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    assert step(policy, 0, new_episode=True) == 0


def test_each_robots_object_map_is_seeded_per_episode() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 3), FakeRobotPolicy([FORWARD] * 2)
    policy = HabitatMultiRobotITMPolicy([r1, r2])

    def act(robot: int, seed: int, new_episode: bool) -> None:
        observations = {"active_robot_id": torch.tensor([[robot]]), "episode_seed": torch.tensor([[seed]])}
        policy.act(observations, None, None, torch.tensor([[not new_episode]]))

    act(0, 123, new_episode=True)
    act(1, 123, new_episode=False)
    act(0, 456, new_episode=True)
    # Both robots are seeded when the episode starts, each with its own stream
    assert r1._object_map.seeds == [[123, 0], [456, 0]]
    assert r2._object_map.seeds == [[123, 1], [456, 1]]
