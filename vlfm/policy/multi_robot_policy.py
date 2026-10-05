"""Policy for the two-robot ObjectNav task (see vlfm.multi_robot.task).

The environment serves one robot per step and reports which one in the
`active_robot_id` observation. This policy keeps one single-robot VLFM policy per
robot and routes each step to the active robot's policy. With shared maps, all the
robots' policies use the first robot's maps.

Map sharing modes:
    independent: each robot has its own maps; the robots do not communicate.
    shared: one obstacle, value and object map for all; the robot closest to the
        target navigates to it, the others keep exploring (TargetClaim).
    shared_maps_only: one obstacle and value map for all; each robot keeps its own
        object map and goes to the target when it sees it itself.
"""

from typing import Any, List, Optional, Union

import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig
from torch import Tensor

from vlfm.multi_robot.target_claim import TargetClaim
from vlfm.multi_robot.task import ActiveRobotIdSensor, EpisodeSeedSensor
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.habitat_policies import HabitatITMPolicyV2, TorchActionIDs


MAP_SHARING_MODES = ("independent", "shared", "shared_maps_only")


class TeamMemberITMPolicy(HabitatITMPolicyV2):
    """A robot's VLFM policy when the robots share their maps. The only difference to
    the single-robot policy: if the object map is shared too, then once the target is
    in it, only the robot that claims it navigates there; the others keep exploring."""

    _target_claim: Optional[TargetClaim] = None  # set by HabitatMultiRobotITMPolicy

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._target_claim is None:
            return super()._get_target_object_location(position)
        if not self._object_map.has_object(self._target_object):
            self._target_claim.release()
            return None
        if not self._done_initializing:
            # The goal is not used while turning in place; claim the target only when
            # a robot is ready to go
            return None
        target_cloud = self._object_map.get_target_cloud(self._target_object)
        if not self._target_claim.may_navigate(self._robot_id, target_cloud):
            return None
        return super()._get_target_object_location(position)


@baseline_registry.register_policy
class HabitatMultiRobotITMPolicy(BasePolicy):
    def __init__(
        self, robot_policies: List[Any], map_sharing: str = "independent", target_claim_timeout: int = 50
    ) -> None:
        """
        Args:
            robot_policies: One single-robot policy per robot. With shared maps, they
                must be TeamMemberITMPolicy.
            map_sharing: "independent", "shared" or "shared_maps_only" (see above).
            target_claim_timeout: See TargetClaim.
        """
        super().__init__()
        if map_sharing not in MAP_SHARING_MODES:
            raise ValueError(f"Unknown map_sharing={map_sharing}")
        self._robot_policies = robot_policies
        self._shared_maps = map_sharing != "independent"
        self._shared_object_map = map_sharing == "shared"
        self._needs_reset = [False] * len(robot_policies)
        self._gave_up = [False] * len(robot_policies)
        # Each robot's generator for the object map's random sampling, set per episode
        self._rngs: Optional[List[np.random.RandomState]] = None
        self._target_claim = TargetClaim(target_claim_timeout)

        for robot_id, policy in enumerate(robot_policies):
            policy._robot_id = robot_id
            if self._shared_maps:
                policy._resets_maps = False  # cleared once per episode, in act()
                if robot_id > 0:
                    first = robot_policies[0]
                    policy._obstacle_map = first._obstacle_map
                    policy._value_map = first._value_map
            if self._shared_object_map:
                policy._target_claim = self._target_claim
                if robot_id > 0:
                    policy._object_map = robot_policies[0]._object_map

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMultiRobotITMPolicy":
        policy_config = config.habitat_baselines.rl.policy
        if policy_config.assigner != "vlfm":
            raise NotImplementedError(f"assigner={policy_config.assigner} is not implemented yet")
        robot_cls = TeamMemberITMPolicy if policy_config.map_sharing != "independent" else HabitatITMPolicyV2
        robot_policies = [robot_cls.from_config(config) for _ in range(policy_config.num_robots)]
        return cls(robot_policies, policy_config.map_sharing, policy_config.target_claim_timeout)

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        if ActiveRobotIdSensor.cls_uuid in observations:
            robot = int(observations[ActiveRobotIdSensor.cls_uuid][0].item())
        else:
            robot = 0

        if masks[0] == 0:  # new episode
            self._start_episode(observations)
        # A robot's policy resets itself when it sees a zero mask, so each robot gets
        # one on its own first step of the episode
        robot_masks = torch.zeros_like(masks) if self._needs_reset[robot] else torch.ones_like(masks)
        self._needs_reset[robot] = False

        policy = self._robot_policies[robot]
        if self._rngs is not None:
            policy._object_map.use_rng(self._rngs[robot])
        if "gps" in observations:
            # Habitat's GPS makes west negative; VLFM's frame has y pointing left
            x, y = observations["gps"][0].cpu().numpy()
            self._target_claim.positions[robot] = np.array([x, -y])
        if self._shared_maps and policy._compute_frontiers:
            policy._obstacle_map.teammate_positions = [
                xy for r, xy in sorted(self._target_claim.positions.items()) if r != robot
            ]

        action_data = policy.act(observations, rnn_hidden_states, prev_actions, robot_masks, deterministic)

        if getattr(policy, "_hw_corruption", False):
            # The step hit corrupted memory. Let the STOP through: the episode ends
            # flagged with hw_corruption and is rerun.
            return action_data

        # A single-robot policy also returns STOP when it has nowhere left to go (no
        # frontiers, or the edge of its map). That must not end the episode while the
        # other robot can still search, so the robot turns in place instead.
        stopped = bool((action_data.actions == TorchActionIDs.STOP.to(action_data.actions.device)).all())
        self._gave_up[robot] = stopped and not policy._called_stop
        if self._gave_up[robot] and not all(self._gave_up):
            action_data.actions = TorchActionIDs.TURN_LEFT.to(action_data.actions.device)

        if self._shared_object_map and action_data.policy_info:
            # Logged in the episode's JSON
            claim = self._target_claim
            action_data.policy_info[0]["target_claim_robot"] = -1 if claim.robot is None else claim.robot
            action_data.policy_info[0]["target_claim_timeouts"] = claim.num_timeouts
        return action_data

    def _start_episode(self, observations: TensorDict) -> None:
        num_robots = len(self._robot_policies)
        self._needs_reset = [True] * num_robots
        self._gave_up = [False] * num_robots
        self._target_claim.reset()
        if EpisodeSeedSensor.cls_uuid in observations:
            # Each robot draws from its own random stream, which depends only on the
            # episode and the robot, also when the robots share the object map
            episode_seed = int(observations[EpisodeSeedSensor.cls_uuid][0].item())
            self._rngs = [np.random.RandomState([episode_seed, robot_id]) for robot_id in range(num_robots)]
        if self._shared_maps:
            # The robots' policies leave this to us; each map is cleared once
            maps = {}
            for policy in self._robot_policies:
                robot_maps = [policy._object_map, policy._value_map]
                if policy._compute_frontiers:
                    robot_maps.append(policy._obstacle_map)
                maps.update({id(m): m for m in robot_maps})
            for robot_map in maps.values():
                robot_map.reset()
