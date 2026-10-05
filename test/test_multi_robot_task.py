"""Tests MultiRobotObjectNavTask against a fake simulator (no Habitat scene needed)."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import quaternion

from vlfm.multi_robot.task import MultiRobotObjectNavTask, MultiRobotStats, TeamSPL, episode_seed

IDENTITY = [0.0, 0.0, 0.0, 1.0]
R1_START = [0.0, 0.0, 0.0]
R2_START = [5.0, 0.0, 0.0]
GOAL = [9.0, 0.0, 0.0]


class FakeSim:
    """One agent that moves along x; observations report where they were rendered."""

    def __init__(self) -> None:
        self.position = np.zeros(3)
        self.num_teleports = 0

    def _observe(self) -> Dict[str, Any]:
        return {"rendered_at": self.position.copy()}

    def reset(self) -> Dict[str, Any]:
        self.position = np.array(R1_START)
        return self._observe()

    def get_agent_state(self) -> SimpleNamespace:
        return SimpleNamespace(position=self.position.copy(), rotation=quaternion.quaternion(1, 0, 0, 0))

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Dict[str, Any]:
        if position is not None:
            assert keep_agent_at_new_pose
            self.position = np.array(position)
            self.num_teleports += 1
        return self._observe()

    def geodesic_distance(self, position: Any, goals: Any, episode: Any = None) -> float:
        return float(min(np.linalg.norm(np.array(position) - np.array(goal)) for goal in goals))


class FakeForward:
    def __init__(self, sim: FakeSim) -> None:
        self._sim = sim

    def reset(self, *args: Any, **kwargs: Any) -> None:
        pass

    def step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self._sim.position = self._sim.position + np.array([0.25, 0.0, 0.0])
        return self._sim._observe()


class FakeStop(FakeForward):
    def reset(self, *args: Any, task: Any, **kwargs: Any) -> None:
        task.is_stop_called = False

    def step(self, *args: Any, task: Any, **kwargs: Any) -> Dict[str, Any]:
        task.is_stop_called = True
        return self._sim.get_observations_at()


class FakeSensorSuite:
    def get_observations(self, *args: Any, task: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"active_robot_id": task.team.active}


def make_task(num_robots: int = 2) -> MultiRobotObjectNavTask:
    # Bypass __init__, which needs a full Habitat config
    task = object.__new__(MultiRobotObjectNavTask)
    sim = FakeSim()
    task._sim = sim
    task._num_robots = num_robots
    task._collision_distance = 0.36
    task._robot2_starts = {"scene": {"7": {"position": R2_START, "rotation": IDENTITY}}}
    task.sensor_suite = FakeSensorSuite()
    task.actions = {"stop": FakeStop(sim), "move_forward": FakeForward(sim)}
    return task


def make_episode(episode_id: str = "7") -> SimpleNamespace:
    goal = SimpleNamespace(view_points=[SimpleNamespace(agent_state=SimpleNamespace(position=GOAL))])
    return SimpleNamespace(
        episode_id=episode_id,
        scene_id="data/scene_datasets/hm3d/val/00000-scene/scene.basis.glb",
        start_position=R1_START,
        start_rotation=IDENTITY,
        goals=[goal],
    )


def forward() -> Dict[str, Any]:
    return {"action": "move_forward"}


def test_reset_returns_r1_observations_and_start_distances() -> None:
    task, episode = make_task(), make_episode()
    observations = task.reset(episode)
    assert observations["rendered_at"] == pytest.approx(R1_START)
    assert observations["active_robot_id"] == 0
    assert task.team.start_goal_distances == pytest.approx([9.0, 4.0])


def test_step_moves_the_active_robot_and_returns_the_next_robots_view() -> None:
    task, episode = make_task(), make_episode()
    task.reset(episode)

    observations = task.step(forward(), episode)  # R1 acts
    assert task.team.poses[0].position == pytest.approx([0.25, 0.0, 0.0])
    assert observations["rendered_at"] == pytest.approx(R2_START)
    assert observations["active_robot_id"] == 1

    observations = task.step(forward(), episode)  # R2 acts from its own pose
    assert task.team.poses[1].position == pytest.approx([5.25, 0.0, 0.0])
    assert observations["rendered_at"] == pytest.approx([0.25, 0.0, 0.0])
    assert observations["active_robot_id"] == 0

    assert task.team.path_lengths == pytest.approx([0.25, 0.25])
    assert task.team.joint_steps == 1
    assert task._is_episode_active


def test_stop_ends_the_episode_with_the_agent_at_the_stopper() -> None:
    task, episode = make_task(), make_episode()
    task.reset(episode)
    task.step(forward(), episode)  # R1
    observations = task.step({"action": "stop"}, episode)  # R2 stops

    assert task.team.stopper == 1
    assert not task._is_episode_active
    # The agent stays at R2, so Habitat's success measure looks at the robot that stopped
    assert task._sim.position == pytest.approx(R2_START)
    assert observations["rendered_at"] == pytest.approx(R2_START)
    assert task.team.path_lengths == pytest.approx([0.25, 0.0])


def test_single_robot_never_moves_the_agent_between_robots() -> None:
    task, episode = make_task(num_robots=1), make_episode()
    task.reset(episode)
    for _ in range(3):
        observations = task.step(forward(), episode)
    assert task._sim.num_teleports == 0
    assert observations["rendered_at"] == pytest.approx([0.75, 0.0, 0.0])
    assert observations["active_robot_id"] == 0
    assert task.team.start_goal_distances == pytest.approx([9.0])


def test_episode_without_r2_start_is_rejected() -> None:
    task = make_task()
    with pytest.raises(ValueError, match="no R2 start"):
        task.reset(make_episode(episode_id="8"))


def test_measures_read_the_team_state() -> None:
    task, episode = make_task(), make_episode()
    task.reset(episode)
    task.step(forward(), episode)  # R1 moves 0.25
    task.step({"action": "stop"}, episode)  # R2 stops
    task.measurements = SimpleNamespace(measures={"success": SimpleNamespace(get_metric=lambda: 1.0)})

    spl_measure = object.__new__(TeamSPL)
    spl_measure.update_metric(task=task)
    # L = min(9, 4) = 4 over P1 + P2 = 0.25, capped at 1
    assert spl_measure.get_metric() == pytest.approx(1.0)

    stats_measure = object.__new__(MultiRobotStats)
    stats_measure.update_metric(task=task)
    stats = stats_measure.get_metric()
    assert stats["joint_steps"] == 1
    assert stats["total_actions"] == 2  # R1 moved, R2 stopped
    assert stats["stopper"] == 1
    assert stats["path_length_r1"] == pytest.approx(0.25)
    assert stats["path_length_r2"] == pytest.approx(0.0)
    assert stats["start_goal_distance_r2"] == pytest.approx(4.0)
    assert stats["min_robot_distance"] == pytest.approx(4.75)


def test_episode_seed_depends_only_on_scene_and_episode() -> None:
    seed = episode_seed("data/scene_datasets/hm3d/val/00000-scene/scene.basis.glb", "7")
    assert seed == episode_seed("other/prefix/scene.basis.glb", 7)
    assert seed != episode_seed("other/prefix/scene.basis.glb", "8")
    assert seed != episode_seed("other/prefix/scene2.basis.glb", "7")
    assert 0 <= seed < 2**31
