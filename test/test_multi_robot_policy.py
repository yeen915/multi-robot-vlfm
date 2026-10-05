from typing import Any, List

import numpy as np
import pytest
import torch
from habitat_baselines.rl.ppo.policy import PolicyActionData

from vlfm.policy.habitat_policies import TorchActionIDs
from vlfm.multi_robot.target_claim import TargetClaim
from vlfm.policy.multi_robot_policy import HabitatMultiRobotITMPolicy, TeamMemberITMPolicy


class FakeMap:
    def __init__(self) -> None:
        self.num_resets = 0
        self.rngs: List[Any] = []
        self.teammate_positions: List[Any] = []

    def reset(self) -> None:
        self.num_resets += 1

    def use_rng(self, rng: Any) -> None:
        self.rngs.append(rng)


class FakeRobotPolicy:
    """Returns scripted actions and records the masks it was called with."""

    _compute_frontiers = True

    def __init__(self, actions: List[torch.Tensor], called_stop: bool = False, hw_corruption: bool = False) -> None:
        self._actions = list(actions)
        self._called_stop = called_stop
        self._hw_corruption = hw_corruption
        self._object_map, self._obstacle_map, self._value_map = FakeMap(), FakeMap(), FakeMap()
        self.masks: List[int] = []
        self.teammates_seen: List[Any] = []

    def act(self, observations: Any, rnn_hidden_states: Any, prev_actions: Any, masks: torch.Tensor, *args: Any) -> Any:
        self.masks.append(int(masks[0].item()))
        self.teammates_seen.append([xy.tolist() for xy in self._obstacle_map.teammate_positions])
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


def test_each_robot_draws_from_its_own_stream_seeded_per_episode() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 3), FakeRobotPolicy([FORWARD] * 2)
    policy = HabitatMultiRobotITMPolicy([r1, r2])

    def act(robot: int, seed: int, new_episode: bool) -> None:
        observations = {"active_robot_id": torch.tensor([[robot]]), "episode_seed": torch.tensor([[seed]])}
        policy.act(observations, None, None, torch.tensor([[not new_episode]]))

    act(0, 123, new_episode=True)
    act(1, 123, new_episode=False)
    act(0, 456, new_episode=True)

    def state(rng: np.random.RandomState) -> List[int]:
        return rng.get_state()[1].tolist()

    # Each robot's map gets its robot's generator, seeded from (episode, robot)
    assert [state(r) for r in r1._object_map.rngs] == [
        state(np.random.RandomState([123, 0])),
        state(np.random.RandomState([456, 0])),
    ]
    assert [state(r) for r in r2._object_map.rngs] == [state(np.random.RandomState([123, 1]))]


def shared_step(policy: HabitatMultiRobotITMPolicy, robot: int, gps: List[float], new_episode: bool = False) -> None:
    observations = {
        "active_robot_id": torch.tensor([[robot]]),
        "episode_seed": torch.tensor([[7]]),
        "gps": torch.tensor([gps]),
    }
    policy.act(observations, None, None, torch.tensor([[not new_episode]]))


def test_shared_maps_are_the_first_robots_and_cleared_once_per_episode() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 3), FakeRobotPolicy([FORWARD] * 2)
    first_maps = (r1._object_map, r1._obstacle_map, r1._value_map)
    policy = HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared")

    assert (r2._object_map, r2._obstacle_map, r2._value_map) == first_maps
    assert not r1._resets_maps and not r2._resets_maps
    assert r1._target_claim is r2._target_claim

    shared_step(policy, 0, [0.0, 0.0], new_episode=True)
    shared_step(policy, 1, [1.0, 0.0])
    shared_step(policy, 0, [0.0, 0.0])
    assert [m.num_resets for m in first_maps] == [1, 1, 1]


def test_shared_maps_still_use_one_stream_per_robot() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 2), FakeRobotPolicy([FORWARD])
    policy = HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared")
    shared_step(policy, 0, [0.0, 0.0], new_episode=True)
    shared_step(policy, 1, [1.0, 0.0])
    shared_step(policy, 0, [0.0, 0.0])
    rngs = r1._object_map.rngs  # the shared map
    assert rngs[0] is rngs[2] and rngs[0] is not rngs[1]


def test_obstacle_map_knows_where_the_teammate_is() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 2), FakeRobotPolicy([FORWARD])
    policy = HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared")
    shared_step(policy, 0, [0.0, 0.0], new_episode=True)  # R2 has not been seen yet
    shared_step(policy, 1, [1.0, 2.0])
    shared_step(policy, 0, [0.5, 0.0])
    # VLFM's y is minus Habitat's GPS y
    assert r1.teammates_seen == [[], [[1.0, -2.0]]]
    assert r2.teammates_seen == [[[0.0, 0.0]]]


def test_independent_maps_are_untouched() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 2), FakeRobotPolicy([FORWARD])
    policy = HabitatMultiRobotITMPolicy([r1, r2])
    shared_step(policy, 0, [0.0, 0.0], new_episode=True)
    shared_step(policy, 1, [1.0, 2.0])
    assert r2._object_map is not r1._object_map
    assert r1.teammates_seen == [[]] and r2.teammates_seen == [[]]
    assert r1._object_map.num_resets == 0  # each robot's own policy clears its maps


class FakeObjectMap:
    def __init__(self, cloud: Any) -> None:
        self.cloud = cloud

    def has_object(self, target: str) -> bool:
        return self.cloud is not None

    def get_target_cloud(self, target: str) -> Any:
        return self.cloud

    def get_best_object(self, target: str, position: Any) -> Any:
        return self.cloud[0, :2]


def make_member(robot_id: int, claim: Any, cloud: Any, done_initializing: bool = True) -> TeamMemberITMPolicy:
    member = object.__new__(TeamMemberITMPolicy)  # skips the model clients
    member._robot_id = robot_id
    member._target_claim = claim
    member._object_map = FakeObjectMap(cloud)
    member._target_object = "chair"
    member._done_initializing = done_initializing
    return member


def test_only_the_robot_that_claims_the_target_navigates_to_it() -> None:
    claim = TargetClaim()
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    cloud = np.array([[10.0, 0.0, 0.5]])
    r1, r2 = make_member(0, claim, cloud), make_member(1, claim, cloud)
    assert r1._get_target_object_location(np.array([0.0, 0.0])) is None  # R2 is closer
    assert r2._get_target_object_location(np.array([8.0, 0.0])).tolist() == [10.0, 0.0]


def test_no_claim_while_turning_in_place_and_release_when_the_target_is_gone() -> None:
    claim = TargetClaim()
    claim.positions = {0: np.array([0.0, 0.0]), 1: np.array([8.0, 0.0])}
    cloud = np.array([[10.0, 0.0, 0.5]])
    assert make_member(1, claim, cloud, done_initializing=False)._get_target_object_location(np.zeros(2)) is None
    assert claim.robot is None
    make_member(0, claim, cloud)._get_target_object_location(np.zeros(2))
    assert claim.robot == 1
    make_member(0, claim, None)._get_target_object_location(np.zeros(2))  # target removed from the map
    assert claim.robot is None


def test_without_a_claim_it_behaves_like_vlfm() -> None:
    member = make_member(1, None, np.array([[10.0, 0.0, 0.5]]), done_initializing=False)
    assert member._get_target_object_location(np.zeros(2)).tolist() == [10.0, 0.0]


def test_shared_maps_only_keeps_one_object_map_per_robot() -> None:
    r1, r2 = FakeRobotPolicy([FORWARD] * 2), FakeRobotPolicy([FORWARD])
    r1_object_map, r2_object_map = r1._object_map, r2._object_map
    policy = HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared_maps_only")
    assert r2._obstacle_map is r1._obstacle_map and r2._value_map is r1._value_map
    assert (r1._object_map, r2._object_map) == (r1_object_map, r2_object_map)
    assert not hasattr(r1, "_target_claim")  # each robot goes to what it sees itself

    shared_step(policy, 0, [0.0, 0.0], new_episode=True)
    shared_step(policy, 1, [1.0, 2.0])
    assert [r1._obstacle_map.num_resets, r1._value_map.num_resets] == [1, 1]
    assert [r1_object_map.num_resets, r2_object_map.num_resets] == [1, 1]
    assert r2.teammates_seen == [[[0.0, 0.0]]]  # the obstacle map is still shared
    # Each robot's own object map draws from that robot's stream
    assert len(r1_object_map.rngs) == 1 and len(r2_object_map.rngs) == 1
    assert r1_object_map.rngs[0] is not r2_object_map.rngs[0]


def test_unknown_map_sharing_is_rejected() -> None:
    with pytest.raises(ValueError):
        HabitatMultiRobotITMPolicy([FakeRobotPolicy([])], map_sharing="everything")


def test_target_claim_is_logged_with_a_shared_object_map() -> None:
    class InfoPolicy(FakeRobotPolicy):
        def act(self, *args: Any) -> Any:
            action_data = super().act(*args)
            action_data.policy_info = [{}]
            return action_data

    r1, r2 = InfoPolicy([FORWARD]), InfoPolicy([FORWARD])
    policy = HabitatMultiRobotITMPolicy([r1, r2], map_sharing="shared")
    observations = {"active_robot_id": torch.tensor([[0]]), "gps": torch.tensor([[0.0, 0.0]])}
    info = policy.act(observations, None, None, torch.tensor([[False]])).policy_info[0]
    assert info == {"target_claim_robot": -1, "target_claim_timeouts": 0}
