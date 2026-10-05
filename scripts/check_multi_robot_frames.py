"""Checks, in the simulator, that both robots of the two-robot task report their
observations in the same frame (W, R1's episodic frame).

For a few episodes, both robots look around and move a little. At every step:
  1. Pose: the pose that VLFM derives from the GPS and compass observations must equal
     the active robot's stored simulator pose converted to W.
  2. Geometry: points at obstacle height in the active robot's depth image are mapped
     to W with that pose, then to the simulator's world frame. Obstacles must not lie
     on the navmesh, so the fraction that does should be near zero for both robots.
  3. Control: R2's points are also mapped as if its poses were relative to its own
     start (the mistake this check is meant to catch). That fraction should be large.

Needs a GPU for rendering, but none of the model servers.

Usage (from the repo root):
    python scripts/check_multi_robot_frames.py --scenes 5cdEh9F2hJL mv2HUxq3B53
"""

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo root, for vlfm

import habitat
import numpy as np
from habitat.config import read_write
from habitat.config.default import patch_config
from hydra import compose, initialize_config_dir

import vlfm.run  # noqa: F401  (registers the config search path, task, sensors and measures)
from vlfm.multi_robot.frames import shared_to_world, world_to_shared
from vlfm.utils.geometry_utils import get_point_cloud, transform_points, xyz_yaw_to_tf_matrix

ACTIONS = ["turn_left"] * 24 + ["move_forward"] * 12 + ["turn_left"] * 24


def angle_diff(a: float, b: float) -> float:
    return float(np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi))


def check_scene(scene: str, num_episodes: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("config")):
        cfg = compose(
            config_name="experiments/vlfm_objectnav_hm3d_multi",
            overrides=[f"habitat.dataset.content_scenes=[{scene}]", "habitat.dataset.split=val"],
        )
    cfg = patch_config(cfg)
    with read_write(cfg):
        cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor", None)

    depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera_height = depth_cfg.position[1]
    fx = fy = depth_cfg.width / (2 * np.tan(np.deg2rad(depth_cfg.hfov) / 2))
    policy_cfg = cfg.habitat_baselines.rl.policy

    results = []
    with habitat.Env(config=cfg) as env:
        attempts = 0
        while len(results) < num_episodes and attempts < 3 * num_episodes:
            attempts += 1
            try:
                observations = env.reset()
            except ValueError as e:  # episode without an R2 start
                print(f"  skipped: {e}")
                continue
            episode, team, pathfinder = env.current_episode, env.task.team, env.sim.pathfinder
            start_position, start_rotation = np.array(episode.start_position), list(episode.start_rotation)
            r2_start = team.start_poses[1]

            pose_errors = [0.0, 0.0]
            yaw_errors = [0.0, 0.0]
            on_navmesh: Dict[str, List[int]] = {"r1": [0, 0], "r2": [0, 0], "r2_wrong_frame": [0, 0]}

            def count_navigable(world_points: np.ndarray, key: str) -> None:
                world_points = world_points.copy()
                world_points[:, 1] = start_position[1]
                on_navmesh[key][0] += sum(bool(pathfinder.is_navigable(p, 0.5)) for p in world_points)
                on_navmesh[key][1] += len(world_points)

            for action in ACTIONS:
                robot = int(observations["active_robot_id"][0])
                # 1. Pose from the observations, as HabitatMixin._cache_observations does
                x, y = float(observations["gps"][0]), -float(observations["gps"][1])
                yaw = float(observations["compass"][0])
                expected_xyz, expected_yaw = world_to_shared(
                    team.poses[robot].position, team.poses[robot].rotation, start_position, start_rotation
                )
                pose_errors[robot] = max(pose_errors[robot], float(np.linalg.norm(expected_xyz[:2] - [x, y])))
                yaw_errors[robot] = max(yaw_errors[robot], angle_diff(expected_yaw, yaw))

                # 2. Obstacle-height points of the depth image, in W and then in the world
                depth = observations["depth"].squeeze(-1)
                scaled = depth * (depth_cfg.max_depth - depth_cfg.min_depth) + depth_cfg.min_depth
                cloud = get_point_cloud(scaled, (depth > 0) & (scaled < depth_cfg.max_depth), fx, fy)
                cloud = cloud[:: args.subsample]
                cloud_w = transform_points(xyz_yaw_to_tf_matrix(np.array([x, y, camera_height]), yaw), cloud)
                obstacles = cloud_w[
                    (cloud_w[:, 2] >= policy_cfg.min_obstacle_height) & (cloud_w[:, 2] <= policy_cfg.max_obstacle_height)
                ]
                if len(obstacles) > 0:
                    count_navigable(shared_to_world(obstacles, start_position, start_rotation), f"r{robot + 1}")
                    if robot == 1:
                        # 3. The wrong frame: as if W were centered on R2's own start
                        count_navigable(
                            shared_to_world(obstacles, r2_start.position, r2_start.rotation), "r2_wrong_frame"
                        )
                observations = env.step(action)

            result = {
                "scene": scene,
                "episode_id": episode.episode_id,
                "pose_error_m": pose_errors,
                "yaw_error_rad": yaw_errors,
                **{f"on_navmesh_{k}": (v[0] / v[1] if v[1] else float("nan")) for k, v in on_navmesh.items()},
                "num_points": {k: v[1] for k, v in on_navmesh.items()},
            }
            results.append(result)
            print(
                f"  ep {result['episode_id']:>3}: pose err R1 {pose_errors[0]:.1e} m, R2 {pose_errors[1]:.1e} m;"
                f" yaw err R1 {yaw_errors[0]:.1e}, R2 {yaw_errors[1]:.1e} rad;"
                f" obstacles on navmesh R1 {result['on_navmesh_r1']:.1%}, R2 {result['on_navmesh_r2']:.1%},"
                f" R2 wrong frame {result['on_navmesh_r2_wrong_frame']:.1%}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=50, help="use every n-th depth point")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for scene in args.scenes:
        print(scene)
        results += check_scene(scene, args.episodes_per_scene, args)

    pose = max(max(r["pose_error_m"]) for r in results)
    yaw = max(max(r["yaw_error_rad"]) for r in results)
    print(f"\n{len(results)} episodes. Max pose error {pose:.1e} m, max yaw error {yaw:.1e} rad.")
    for key in ("r1", "r2", "r2_wrong_frame"):
        print(f"Obstacle points on the navmesh, {key}: {np.nanmean([r[f'on_navmesh_{key}'] for r in results]):.1%}")


if __name__ == "__main__":
    main()
