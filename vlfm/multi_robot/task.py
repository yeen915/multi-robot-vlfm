"""ObjectNav with two robots that share the simulator's single agent.

Each environment step serves one robot (see vlfm.multi_robot.team). After a robot's
action, the agent is moved to the next robot's pose and that robot's observations are
returned. GPS and compass are computed from the agent's pose relative to the episode
start, so both robots report their pose in the same frame: R1's episodic frame.

The episode ends when either robot calls STOP. The agent is left at that robot's
pose, so Habitat's own `distance_to_goal` and `success` measures refer to the robot
that stopped. Habitat's `spl` and `soft_spl` accumulate the agent's displacement,
which includes the jumps between robots; they are not valid here. Use `team_spl`,
`team_spl_l1` and `winner_spl` instead.

With the active_robot_id sensor absent from the config, or num_robots set to 1, the
task behaves exactly like ObjectNav-v1.
"""

import gzip
import json
import os
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from gym import spaces
from habitat import registry
from habitat.config.default_structured_configs import (
    LabSensorConfig,
    MeasurementConfig,
)
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import Success
from habitat.tasks.nav.object_nav_task import ObjectNavigationTask
from habitat.utils.geometry_utils import quaternion_to_list
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from vlfm.multi_robot.team import RobotPose, RobotTeam, team_spl, winner_spl


def load_robot2_starts(path: str) -> Dict[str, Dict[str, Any]]:
    """Loads the file written by scripts/generate_robot2_starts.py."""
    with gzip.open(path, "rt") as f:
        return json.load(f)["starts"]


@registry.register_task(name="MultiRobotObjectNav-v1")
class MultiRobotObjectNavTask(ObjectNavigationTask):
    team: RobotTeam

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sensor_config = self._config.lab_sensors.get(ActiveRobotIdSensor.cls_uuid, None)
        self._num_robots: int = 1 if sensor_config is None else sensor_config.num_robots
        assert self._num_robots in (1, 2), "Only one or two robots are supported"
        self._collision_distance: float = 0.0
        self._robot2_starts: Dict[str, Dict[str, Any]] = {}
        if self._num_robots == 2:
            self._robot2_starts = load_robot2_starts(sensor_config.robot2_starts_path)
            agent_name = self._sim.habitat_config.agents_order[self._sim.habitat_config.default_agent_id]
            self._collision_distance = 2 * self._sim.habitat_config.agents[agent_name].radius

    def reset(self, episode: Episode) -> Any:
        # The team must exist before super().reset(), which queries the sensors
        start_poses = [RobotPose(np.array(episode.start_position), list(episode.start_rotation))]
        if self._num_robots == 2:
            start_poses.append(self._robot2_start_pose(episode))
        self.team = RobotTeam(start_poses, collision_distance=self._collision_distance)
        observations = super().reset(episode)
        self.team.start_goal_distances = [self._distance_to_goal(pose.position, episode) for pose in start_poses]
        return observations

    def step(self, action: Dict[str, Any], episode: Episode) -> Any:
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        observations: Any = {}
        # The agent is at the active robot's pose, so this moves the active robot
        self._step_single_action(observations, action["action"], action, episode)

        agent_state = self._sim.get_agent_state()
        stopped = bool(getattr(self, "is_stop_called", False))
        next_robot = self.team.record_action(
            RobotPose(np.array(agent_state.position), quaternion_to_list(agent_state.rotation)),
            stopped,
        )
        if next_robot is not None and self._num_robots > 1:
            pose = self.team.poses[next_robot]
            observations = self._sim.get_observations_at(
                position=[float(v) for v in pose.position],
                rotation=pose.rotation,
                keep_agent_at_new_pose=True,
            )
            assert observations is not None, f"Could not place the agent at robot {next_robot}'s pose"

        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )
        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )
        return observations

    def _robot2_start_pose(self, episode: Episode) -> RobotPose:
        scene = os.path.basename(episode.scene_id).split(".")[0]
        start = self._robot2_starts.get(scene, {}).get(str(episode.episode_id))
        if start is None:
            raise ValueError(
                f"Episode {episode.episode_id} in scene {scene} has no R2 start. Episodes without one must be"
                " excluded from two-robot runs."
            )
        return RobotPose(np.array(start["position"]), list(start["rotation"]))

    def _distance_to_goal(self, position: np.ndarray, episode: Episode) -> float:
        """Geodesic distance to the closest goal viewpoint, as in Habitat's
        DistanceToGoal measure with distance_to=VIEW_POINTS."""
        viewpoints = [
            view_point.agent_state.position for goal in episode.goals for view_point in goal.view_points
        ]
        # The episode is not passed on purpose: with it, the simulator would cache
        # this path as the episode's shortest path
        return float(self._sim.geodesic_distance(position, viewpoints))


@registry.register_sensor
class ActiveRobotIdSensor(Sensor):
    """Index of the robot that the current observations belong to, and that the next
    action will be applied to."""

    cls_uuid: str = "active_robot_id"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64)

    def get_observation(self, *args: Any, task: Any, **kwargs: Any) -> np.ndarray:
        team: Optional[RobotTeam] = getattr(task, "team", None)
        return np.array([0 if team is None else team.active], dtype=np.int64)


@registry.register_sensor
class EpisodeSeedSensor(Sensor):
    """A seed that depends only on the scene and the episode id.

    VLFM draws random numbers when it subsamples object point clouds. The policy
    reseeds each robot's generator from this value and the robot's index at the start
    of every episode, so an episode's result does not depend on which episodes ran
    before it in the same process.
    """

    cls_uuid: str = "episode_seed"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return spaces.Box(low=0, high=2**31 - 1, shape=(1,), dtype=np.int64)

    def get_observation(self, *args: Any, episode: Episode, **kwargs: Any) -> np.ndarray:
        return np.array([episode_seed(episode.scene_id, episode.episode_id)], dtype=np.int64)


def episode_seed(scene_id: str, episode_id: Any) -> int:
    scene = os.path.basename(scene_id).split(".")[0]
    return zlib.crc32(f"{scene}_{episode_id}".encode()) % 2**31


class _TeamSPLBase(Measure):
    """Shared code for the SPL variants; subclasses define cls_uuid and _compute."""

    cls_uuid: str = ""

    def __init__(self, sim: Any, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [Success.cls_uuid])
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: Any, **kwargs: Any) -> None:
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._metric = self._compute(success, task.team)

    def _compute(self, success: float, team: RobotTeam) -> float:
        raise NotImplementedError


@registry.register_measure
class TeamSPL(_TeamSPLBase):
    """success * L / max(P1 + P2, L), with L = min(L1, L2)."""

    cls_uuid: str = "team_spl"

    def _compute(self, success: float, team: RobotTeam) -> float:
        return team_spl(success, team)


@registry.register_measure
class TeamSPLL1(_TeamSPLBase):
    """success * L1 / max(P1 + P2, L1): the shortest distance is taken from the first
    robot's start, as in CoRelNav."""

    cls_uuid: str = "team_spl_l1"

    def _compute(self, success: float, team: RobotTeam) -> float:
        return team_spl(success, team, use_r1_start=True)


@registry.register_measure
class WinnerSPL(_TeamSPLBase):
    """SPL of the robot that stopped, from its own start and along its own path."""

    cls_uuid: str = "winner_spl"

    def _compute(self, success: float, team: RobotTeam) -> float:
        return winner_spl(success, team)


@registry.register_measure
class MultiRobotStats(Measure):
    """Per-episode team statistics. `joint_steps` of a successful episode is the
    number of joint steps it took to succeed."""

    cls_uuid: str = "multi_robot"

    def __init__(self, sim: Any, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args: Any, task: Any, **kwargs: Any) -> None:
        team: RobotTeam = task.team
        stats: Dict[str, float] = {
            "joint_steps": team.joint_steps,  # team steps: elapsed time
            "total_actions": team.total_actions,  # all robots' actions added up
            "stopper": -1 if team.stopper is None else team.stopper,
            "num_close_steps": team.num_close_steps,
        }
        if team.num_robots > 1:
            stats["min_robot_distance"] = team.min_robot_distance
        per_robot: Dict[str, List[float]] = {
            "path_length": team.path_lengths,
            "num_actions": team.num_actions,
            "start_goal_distance": team.start_goal_distances,
        }
        for name, values in per_robot.items():
            for robot, value in enumerate(values):
                stats[f"{name}_r{robot + 1}"] = value
        self._metric = stats


@dataclass
class ActiveRobotIdSensorConfig(LabSensorConfig):
    type: str = ActiveRobotIdSensor.__name__
    num_robots: int = 2
    robot2_starts_path: str = "data/multi_robot/hm3d_val_robot2_starts.json.gz"


@dataclass
class EpisodeSeedSensorConfig(LabSensorConfig):
    type: str = EpisodeSeedSensor.__name__


@dataclass
class TeamSPLMeasurementConfig(MeasurementConfig):
    type: str = TeamSPL.__name__


@dataclass
class TeamSPLL1MeasurementConfig(MeasurementConfig):
    type: str = TeamSPLL1.__name__


@dataclass
class WinnerSPLMeasurementConfig(MeasurementConfig):
    type: str = WinnerSPL.__name__


@dataclass
class MultiRobotStatsMeasurementConfig(MeasurementConfig):
    type: str = MultiRobotStats.__name__


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{ActiveRobotIdSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=ActiveRobotIdSensor.cls_uuid,
    node=ActiveRobotIdSensorConfig,
)
cs.store(
    package=f"habitat.task.lab_sensors.{EpisodeSeedSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=EpisodeSeedSensor.cls_uuid,
    node=EpisodeSeedSensorConfig,
)
for measure, measure_config in [
    (TeamSPL, TeamSPLMeasurementConfig),
    (TeamSPLL1, TeamSPLL1MeasurementConfig),
    (WinnerSPL, WinnerSPLMeasurementConfig),
    (MultiRobotStats, MultiRobotStatsMeasurementConfig),
]:
    cs.store(
        package=f"habitat.task.measurements.{measure.cls_uuid}",
        group="habitat/task/measurements",
        name=measure.cls_uuid,
        node=measure_config,
    )
