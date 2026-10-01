"""Bookkeeping for a team of robots that share one simulator agent.

The simulator only has one agent. Each environment step serves one robot: the agent
is placed at that robot's pose, takes the robot's action, and the resulting pose is
stored here. A joint step is one action by every robot.

Nothing in this module depends on Habitat, so it can be tested on its own.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RobotPose:
    position: np.ndarray  # (3,), simulator world frame
    rotation: List[float]  # quaternion as [x, y, z, w]


@dataclass
class RobotTeam:
    start_poses: List[RobotPose]
    # Shortest geodesic distance from each robot's start to the goal
    start_goal_distances: List[float] = field(default_factory=list)
    # Robots closer than this are counted as "would have collided"
    collision_distance: float = 0.36

    def __post_init__(self) -> None:
        assert len(self.start_poses) >= 1
        self.poses = [RobotPose(np.array(p.position, dtype=np.float64), list(p.rotation)) for p in self.start_poses]
        self.active = 0
        self.stopper: Optional[int] = None
        self.path_lengths = [0.0] * self.num_robots
        self.num_actions = [0] * self.num_robots
        self.min_robot_distance = self._robot_distance()
        self.num_close_steps = 0

    @property
    def num_robots(self) -> int:
        return len(self.start_poses)

    @property
    def joint_steps(self) -> int:
        """Number of joint steps started so far (the first action starts step 1)."""
        return max(self.num_actions)

    def record_action(self, new_pose: RobotPose, stopped: bool) -> Optional[int]:
        """Stores the outcome of the active robot's action.

        Args:
            new_pose: The active robot's pose after its action.
            stopped: Whether the action was STOP, which ends the episode.

        Returns:
            The robot that acts next, or None if the episode ended.
        """
        assert self.stopper is None, "The episode already ended"
        robot = self.active
        position = np.array(new_pose.position, dtype=np.float64)
        self.path_lengths[robot] += float(np.linalg.norm(position - self.poses[robot].position))
        self.poses[robot] = RobotPose(position, list(new_pose.rotation))
        self.num_actions[robot] += 1

        distance = self._robot_distance()
        self.min_robot_distance = min(self.min_robot_distance, distance)
        self.num_close_steps += int(distance < self.collision_distance)

        if stopped:
            self.stopper = robot
            return None
        self.active = (robot + 1) % self.num_robots
        return self.active

    def _robot_distance(self) -> float:
        """Smallest distance between any two robots (inf for a single robot)."""
        distances = [
            float(np.linalg.norm(self.poses[i].position - self.poses[j].position))
            for i in range(self.num_robots)
            for j in range(i + 1, self.num_robots)
        ]
        return min(distances, default=float("inf"))


def spl(success: float, shortest: float, path_length: float) -> float:
    """Success weighted by path length: success * shortest / max(path_length, shortest)."""
    denominator = max(path_length, shortest)
    if denominator == 0:
        return float(success)
    return float(success) * shortest / denominator


def team_spl(success: float, team: RobotTeam, use_r1_start: bool = False) -> float:
    """SPL with the summed path length of all robots.

    Args:
        use_r1_start: If True, the shortest distance is taken from the first robot's
            start (the CoRelNav definition). Otherwise it is the smallest over all
            robots' starts.
    """
    shortest = team.start_goal_distances[0] if use_r1_start else min(team.start_goal_distances)
    return spl(success, shortest, sum(team.path_lengths))


def winner_spl(success: float, team: RobotTeam) -> float:
    """SPL of the robot that stopped, using only its own start and its own path."""
    if team.stopper is None:
        return 0.0
    return spl(success, team.start_goal_distances[team.stopper], team.path_lengths[team.stopper])
