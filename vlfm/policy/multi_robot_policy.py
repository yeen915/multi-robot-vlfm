"""Policy for the two-robot ObjectNav task (see vlfm.multi_robot.task).

The environment serves one robot per step and reports which one in the
`active_robot_id` observation. This policy keeps one single-robot VLFM policy per
robot and routes each step to the active robot's policy.
"""

from typing import Any, List

import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig
from torch import Tensor

from vlfm.multi_robot.task import ActiveRobotIdSensor, EpisodeSeedSensor
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.habitat_policies import HabitatITMPolicyV2, TorchActionIDs


@baseline_registry.register_policy
class HabitatMultiRobotITMPolicy(BasePolicy):
    def __init__(self, robot_policies: List[Any]) -> None:
        """
        Args:
            robot_policies: One single-robot policy per robot. With independent maps,
                each policy owns its maps and the robots do not communicate.
        """
        super().__init__()
        self._robot_policies = robot_policies
        self._needs_reset = [False] * len(robot_policies)
        self._gave_up = [False] * len(robot_policies)

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMultiRobotITMPolicy":
        policy_config = config.habitat_baselines.rl.policy
        if policy_config.map_sharing != "independent":
            raise NotImplementedError(f"map_sharing={policy_config.map_sharing} is not implemented yet")
        if policy_config.assigner != "vlfm":
            raise NotImplementedError(f"assigner={policy_config.assigner} is not implemented yet")
        return cls([HabitatITMPolicyV2.from_config(config) for _ in range(policy_config.num_robots)])

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
            if EpisodeSeedSensor.cls_uuid in observations:
                # Each robot's object map gets its own random stream, which depends
                # only on the episode and the robot
                episode_seed = int(observations[EpisodeSeedSensor.cls_uuid][0].item())
                for robot_id, robot_policy in enumerate(self._robot_policies):
                    robot_policy._object_map.seed([episode_seed, robot_id])
            self._needs_reset = [True] * len(self._robot_policies)
            self._gave_up = [False] * len(self._robot_policies)
        # A robot's policy resets itself when it sees a zero mask, so each robot gets
        # one on its own first step of the episode
        robot_masks = torch.zeros_like(masks) if self._needs_reset[robot] else torch.ones_like(masks)
        self._needs_reset[robot] = False

        policy = self._robot_policies[robot]
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

        return action_data
