"""Policy for the two-robot ObjectNav task (see vlfm.multi_robot.task).

The environment serves one robot per step and reports which one in the
`active_robot_id` observation. This policy keeps one single-robot VLFM policy per
robot and routes each step to the active robot's policy. With shared maps, all the
robots' policies use the first robot's maps.

Frontier choice: with assigner "vlfm", every robot picks its frontiers by VLFM's own
rule. With the other assigners (shared maps only), a Coordinator assigns them.

Map sharing modes:
    independent: each robot has its own maps; the robots do not communicate.
    shared: one obstacle, value and object map for all; the robot closest to the
        target navigates to it, the others keep exploring (TargetClaim).
    shared_maps_only: one obstacle and value map for all; each robot keeps its own
        object map and goes to the target when it sees it itself.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig
from torch import Tensor

from vlfm.multi_robot.coordinator import (
    ASSIGNERS,
    Coordinator,
    CoordinatorConfig,
    UtilityScaleLog,
    match_frontier,
    values_in_order,
)
from vlfm.multi_robot.path_cost import PathPlanner
from vlfm.multi_robot.target_claim import TargetClaim
from vlfm.multi_robot.task import ActiveRobotIdSensor, EpisodeSeedSensor
from vlfm.multi_robot.team_stats import TeamStats
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.habitat_policies import HabitatITMPolicyV2, TorchActionIDs
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.utils.hw_corruption import is_corrupted_index_error


MAP_SHARING_MODES = ("independent", "shared", "shared_maps_only")


class TeamMemberITMPolicy(HabitatITMPolicyV2):
    """A robot's VLFM policy when the robots share their maps. It differs from the
    single-robot policy in two ways:
    - If the object map is shared too, then once the target is in it, only the robot
      that claims it navigates there; the others keep exploring.
    - With a coordinator, the robot heads for the frontier the coordinator assigns it.
      VLFM's own rule (sticking to the last frontier, avoiding cycles) is not used; the
      coordinator's hysteresis takes its place.
    """

    # Set by HabitatMultiRobotITMPolicy
    _target_claim: Optional[TargetClaim] = None
    _coordinator: Optional[Coordinator] = None
    # Teammates exploring now, and goals of the teammates that are not (set each step)
    _team_status: Tuple[Set[int], Dict[int, np.ndarray]] = (set(), {})

    def _explore(self, observations: TensorDict) -> Tensor:
        if self._coordinator is None:
            return super()._explore(observations)
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        values = values_in_order(*self._sort_frontiers_by_value(observations, frontiers), frontiers)
        teammates_exploring, occupied = self._team_status
        goal = self._coordinator.choose(
            self._robot_id, frontiers, values, self._obstacle_map, teammates_exploring | {self._robot_id}, occupied
        )
        if goal is None:
            # The only frontier went to a teammate: look around meanwhile
            os.environ["DEBUG_INFO"] = "Only frontier taken by a teammate."
            return TorchActionIDs.TURN_LEFT
        index = match_frontier(goal, np.asarray(frontiers).reshape(-1, 2))
        self._last_frontier = goal
        self._last_value = float(values[index]) if index is not None else float("nan")
        os.environ["DEBUG_INFO"] = f"Assigned frontier, value: {self._last_value * 100:.2f}%"
        return self._pointnav(goal, stop=False)

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
        self,
        robot_policies: List[Any],
        map_sharing: str = "independent",
        target_claim_timeout: int = 50,
        stats_map_kwargs: Optional[Dict[str, Any]] = None,
        coordinator_config: Optional[CoordinatorConfig] = None,
        log_utility_scales: bool = False,
    ) -> None:
        """
        Args:
            robot_policies: One single-robot policy per robot. With shared maps, they
                must be TeamMemberITMPolicy.
            map_sharing: "independent", "shared" or "shared_maps_only" (see above).
            target_claim_timeout: See TargetClaim.
            stats_map_kwargs: ObstacleMap arguments for the team statistics (see
                TeamStats); without them, no statistics are kept.
            coordinator_config: Assigns frontiers when given (needs shared maps);
                without it, each robot uses VLFM's rule.
            log_utility_scales: Log the values and path costs of all frontiers at
                each exploration step (see UtilityScaleLog).
        """
        super().__init__()
        if map_sharing not in MAP_SHARING_MODES:
            raise ValueError(f"Unknown map_sharing={map_sharing}")
        self._robot_policies = robot_policies
        self._shared_maps = map_sharing != "independent"
        self._shared_object_map = map_sharing == "shared"
        self._needs_reset = [False] * len(robot_policies)
        self._env_step = 0
        self._gave_up = [False] * len(robot_policies)
        # Each robot's generator for the object map's random sampling, set per episode
        self._rngs: Optional[List[np.random.RandomState]] = None
        self._target_claim = TargetClaim(target_claim_timeout)
        self._team_stats = None if stats_map_kwargs is None else TeamStats(len(robot_policies), stats_map_kwargs)
        if coordinator_config is not None and not self._shared_maps:
            raise ValueError("A coordinator needs shared maps")
        self._coordinator = None if coordinator_config is None else Coordinator(coordinator_config)
        self._unreachable_penalty = 3.0 if coordinator_config is None else coordinator_config.unreachable_penalty
        self._scale_log = UtilityScaleLog() if log_utility_scales else None

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
            if self._coordinator is not None:
                policy._coordinator = self._coordinator

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMultiRobotITMPolicy":
        policy_config = config.habitat_baselines.rl.policy
        if policy_config.detector != "vlm":
            raise NotImplementedError(f"detector={policy_config.detector} is planned for phase 6")
        coordinator_config = None
        if policy_config.assigner != "vlfm":
            if policy_config.assigner not in ASSIGNERS:
                raise ValueError(f"Unknown assigner {policy_config.assigner!r}")
            coordinator_config = CoordinatorConfig(
                assigner=policy_config.assigner,
                lambda_path=policy_config.lambda_path,
                d_min=policy_config.d_min,
                hysteresis_tau=policy_config.hysteresis_tau,
                value_scale=policy_config.value_scale,
                path_scale=policy_config.path_scale,
                unreachable_penalty=policy_config.unreachable_penalty,
                term_weights={
                    "info_gain": policy_config.info_gain_weight,
                    "trajectory_reuse": policy_config.trajectory_reuse_weight,
                },
            )
        robot_cls = TeamMemberITMPolicy if policy_config.map_sharing != "independent" else HabitatITMPolicyV2
        robot_policies = [robot_cls.from_config(config) for _ in range(policy_config.num_robots)]
        stats_map_kwargs = {
            "min_height": policy_config.min_obstacle_height,
            "max_height": policy_config.max_obstacle_height,
            "agent_radius": policy_config.agent_radius,
            "area_thresh": policy_config.obstacle_map_area_threshold,
            "hole_area_thresh": policy_config.hole_area_thresh,
        }
        return cls(
            robot_policies,
            policy_config.map_sharing,
            policy_config.target_claim_timeout,
            stats_map_kwargs,
            coordinator_config,
            policy_config.log_utility_scales,
        )

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
        pose = None
        if "gps" in observations:
            # Habitat's GPS makes west negative; VLFM's frame has y pointing left
            x, y = observations["gps"][0].cpu().numpy()
            self._target_claim.positions[robot] = np.array([x, -y])
            if "compass" in observations:
                pose = (np.array([x, -y]), float(observations["compass"][0].cpu().item()))
        if self._coordinator is not None:
            self._coordinator.step = self._env_step
            if pose is not None:
                self._coordinator.poses[robot] = pose
            policy._team_status = self._team_status(robot)
        if self._shared_maps and policy._compute_frontiers:
            policy._obstacle_map.teammate_positions = [
                xy for r, xy in sorted(self._target_claim.positions.items()) if r != robot
            ]

        if self._team_stats is not None:
            self._add_view_to_stats(robot, policy, observations)
        policy._last_mode = None
        action_data = policy.act(observations, rnn_hidden_states, prev_actions, robot_masks, deterministic)
        self._env_step += 1

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

        if self._team_stats is not None:
            mode = policy._last_mode
            self._team_stats.add_decision(robot, mode, policy._last_frontier if mode == "explore" else None)
        if self._scale_log is not None and policy._last_mode == "explore" and pose is not None:
            self._log_utility_scales(policy, pose)
        if action_data.policy_info:
            # Logged in the episode's JSON
            info = action_data.policy_info[0]
            if self._shared_object_map:
                claim = self._target_claim
                info["target_claim_robot"] = -1 if claim.robot is None else claim.robot
                info["target_claim_timeouts"] = claim.num_timeouts
            if self._team_stats is not None:
                info["team_stats"] = self._team_stats.summary()
                info["coverage_curve"] = self._team_stats.coverage_curve()
            if self._coordinator is not None:
                info["coordinator"] = {k: list(v) if isinstance(v, list) else v for k, v in self._coordinator.stats.items()}
            if self._scale_log is not None:
                info["utility_scales"] = self._scale_log.summary()
        return action_data

    def _team_status(self, robot: int) -> Tuple[Set[int], Dict[int, np.ndarray]]:
        """The teammates that explored in their last step, and the goals of those that
        navigated to the target (which exploring robots keep away from). Teammates
        that are turning in place at the start are neither."""
        exploring, occupied = set(), {}
        for other, policy in enumerate(self._robot_policies):
            if other == robot:
                continue
            if policy._last_mode == "explore":
                exploring.add(other)
            elif policy._last_mode == "navigate":
                occupied[other] = np.asarray(policy._last_goal, dtype=float)
        return exploring, occupied

    def _log_utility_scales(self, policy: Any, pose: Tuple[np.ndarray, float]) -> None:
        """Values and path costs, from where the robot stood, of all the frontiers of
        its (updated) obstacle map."""
        frontiers = np.asarray(policy._obstacle_map.frontiers, dtype=float).reshape(-1, 2)
        if len(frontiers) == 0:
            return
        values = values_in_order(*policy._value_map.sort_waypoints(frontiers, 0.5), frontiers)
        costs = PathPlanner(policy._obstacle_map, self._unreachable_penalty).costs(pose[0], pose[1], list(frontiers))
        self._scale_log.add(values, costs)

    def _add_view_to_stats(self, robot: int, policy: Any, observations: TensorDict) -> None:
        """Adds the robot's current view to the team statistics, with the same depth
        processing and camera pose as the robot's own policy."""
        depth = observations["depth"][0].cpu().numpy()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        x, y = observations["gps"][0].cpu().numpy()
        yaw = observations["compass"][0].cpu().item()
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([x, -y, policy._camera_height]), yaw)
        try:
            self._team_stats.add_view(
                robot,
                depth,
                tf_camera_to_episodic,
                policy._min_depth,
                policy._max_depth,
                policy._fx,
                policy._fy,
                policy._camera_fov,
            )
        except IndexError as e:
            if is_corrupted_index_error(e):
                raise  # the run crashes and the episode is rerun
            # A view reaching past the map's edge: the robot's own policy stops there
            # too. Leave it out of the statistics.

    def _start_episode(self, observations: TensorDict) -> None:
        num_robots = len(self._robot_policies)
        self._needs_reset = [True] * num_robots
        self._gave_up = [False] * num_robots
        self._env_step = 0  # steps of the episode, both robots together (one video frame each)
        self._target_claim.reset()
        for policy in self._robot_policies:
            policy._last_mode = None  # not the previous episode's, for the teammates' view
        if self._coordinator is not None:
            self._coordinator.reset()
        if self._scale_log is not None:
            self._scale_log.reset()
        if self._team_stats is not None:
            self._team_stats.reset()
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
