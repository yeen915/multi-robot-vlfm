"""Samples a fixed start pose for the second robot (R2) of every ObjectNav episode.

R1 keeps the episode's own start pose. For each episode, R2's start is drawn on the
navmesh that the simulator uses at eval time (recomputed for the agent's radius and
height), such that:
  - it is on the same floor as R1's start (height difference <= --max-dy),
  - it is reachable from R1's start, at a geodesic distance drawn uniformly from
    [--min-dist, --max-dist] (matched within --dist-tol),
  - its geodesic distance to the goal is at least --min-goal-dist,
  - its yaw is uniform in [0, 2*pi).

Runs on CPU only (no renderer). The output is deterministic for a given seed: running
the script twice must produce byte-identical files.

Usage (from the repo root):
    python scripts/generate_robot2_starts.py
"""

import os

os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")

import argparse  # noqa: E402
import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import habitat_sim  # noqa: E402
import numpy as np  # noqa: E402


def make_sim(scene_id: str, scene_dataset_config: str, radius: float, height: float) -> habitat_sim.Simulator:
    """Creates a renderer-less simulator; its pathfinder holds the navmesh recomputed
    for the given agent size, exactly like the one used during evaluation."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = scene_id
    sim_cfg.create_renderer = False
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.radius = radius
    agent_cfg.height = height
    agent_cfg.sensor_specifications = []
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def shortest_path(
    pathfinder: habitat_sim.PathFinder, start: np.ndarray, ends: List[np.ndarray]
) -> Tuple[float, Optional[np.ndarray]]:
    """Geodesic distance from start to the closest of ends, and the point where that
    path ends. Returns (inf, None) if no end is reachable."""
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = start
    path.requested_ends = ends
    if not pathfinder.find_path(path):
        return float("inf"), None
    return float(path.geodesic_distance), np.array(path.points[-1])


def geodesic(pathfinder: habitat_sim.PathFinder, start: np.ndarray, ends: List[np.ndarray]) -> float:
    """Geodesic distance from start to the closest of ends (inf if unreachable)."""
    return shortest_path(pathfinder, start, ends)[0]


def yaw_to_quat_xyzw(yaw: float) -> List[float]:
    """Rotation about the up (y) axis, in the [x, y, z, w] order used by episodes."""
    return [0.0, float(np.sin(yaw / 2)), 0.0, float(np.cos(yaw / 2))]


def sample_episode(
    pathfinder: habitat_sim.PathFinder,
    episode: Dict[str, Any],
    viewpoints: List[np.ndarray],
    pool: np.ndarray,
    goal_dist_cache: Dict[int, float],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Returns (start record or None, diagnostics)."""
    r1_start = np.array(episode["start_position"], dtype=np.float32)
    # The goal counts as being on R1's floor if the closest goal viewpoint (the end of
    # R1's shortest path) is within --goal-floor-dy of R1's start height.
    l1, closest_viewpoint = shortest_path(pathfinder, r1_start, viewpoints)
    goal_height_diff = float(closest_viewpoint[1] - r1_start[1]) if closest_viewpoint is not None else None
    goal_same_floor = goal_height_diff is not None and abs(goal_height_diff) <= args.goal_floor_dy

    same_floor = np.flatnonzero(np.abs(pool[:, 1] - r1_start[1]) <= args.max_dy)
    r1_dists = np.array([geodesic(pathfinder, r1_start, [pool[i]]) for i in same_floor])
    in_range = (r1_dists >= args.min_dist - args.dist_tol) & (r1_dists <= args.max_dist + args.dist_tol)
    diagnostics = {
        "num_same_floor": int(len(same_floor)),
        "num_in_dist_range": int(in_range.sum()),
        "max_reachable_dist": float(np.max(r1_dists[np.isfinite(r1_dists)], initial=0.0)),
        "goal_same_floor": goal_same_floor,
        "goal_height_diff": goal_height_diff,
    }

    def goal_dist(pool_idx: int) -> float:
        if pool_idx not in goal_dist_cache:
            goal_dist_cache[pool_idx] = geodesic(pathfinder, pool[pool_idx], viewpoints)
        return goal_dist_cache[pool_idx]

    for draw in range(1, args.max_draws + 1):
        target_dist = rng.uniform(args.min_dist, args.max_dist)
        band = same_floor[np.abs(r1_dists - target_dist) <= args.dist_tol]
        band = np.array([i for i in band if goal_dist(i) >= args.min_goal_dist], dtype=int)
        if len(band) == 0:
            continue
        idx = int(rng.choice(band))
        yaw = float(rng.uniform(0.0, 2 * np.pi))
        position = pool[idx]
        record = {
            "position": [float(v) for v in position],
            "rotation": yaw_to_quat_xyzw(yaw),
            "yaw": yaw,
            "r1_r2_geodesic": float(r1_dists[np.flatnonzero(same_floor == idx)[0]]),
            "r1_r2_euclidean": float(np.linalg.norm(position - r1_start)),
            "height_diff": float(position[1] - r1_start[1]),
            "l1": l1,
            "l1_dataset": float(episode["info"]["geodesic_distance"]),
            "l2": goal_dist(idx),
            "goal_same_floor": goal_same_floor,
            "goal_height_diff": goal_height_diff,
            "object_category": episode["object_category"],
            "target_dist": float(target_dist),
            "num_draws": draw,
        }
        return record, diagnostics

    return None, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes-dir", default="data/datasets/objectnav/hm3d/v1/val/content")
    parser.add_argument("--scenes-dir", default="data/scene_datasets")
    parser.add_argument("--output", default="data/multi_robot/hm3d_val_robot2_starts.json.gz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-dist", type=float, default=2.0, help="min R1-R2 geodesic distance (m)")
    parser.add_argument("--max-dist", type=float, default=10.0, help="max R1-R2 geodesic distance (m)")
    parser.add_argument("--dist-tol", type=float, default=0.25, help="tolerance when matching the drawn distance (m)")
    parser.add_argument("--max-dy", type=float, default=0.2, help="max height difference to R1's start (m)")
    parser.add_argument("--min-goal-dist", type=float, default=3.0, help="min geodesic distance from R2 to goal (m)")
    parser.add_argument(
        "--goal-floor-dy", type=float, default=0.5, help="max height difference for the goal to count as on R1's floor"
    )
    parser.add_argument("--pool-size", type=int, default=3000, help="candidate navigable points per scene")
    parser.add_argument("--max-draws", type=int, default=50, help="distance draws before giving up on an episode")
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--agent-height", type=float, default=0.88)
    parser.add_argument("--scenes", nargs="*", default=None, help="only process these scene names")
    args = parser.parse_args()

    starts: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    max_l1_error = 0.0

    for episodes_file in sorted(glob.glob(os.path.join(args.episodes_dir, "*.json.gz"))):
        scene = os.path.basename(episodes_file).split(".")[0]
        if args.scenes is not None and scene not in args.scenes:
            continue
        with gzip.open(episodes_file, "rt") as f:
            data = json.load(f)
        episodes = data["episodes"]
        scene_seed = zlib.crc32(f"{args.seed}_{scene}".encode())

        sim = make_sim(
            os.path.join(args.scenes_dir, episodes[0]["scene_id"]),
            episodes[0]["scene_dataset_config"],
            args.agent_radius,
            args.agent_height,
        )
        pathfinder = sim.pathfinder
        pathfinder.seed(scene_seed)
        pool = np.array([pathfinder.get_random_navigable_point() for _ in range(args.pool_size)], dtype=np.float32)

        goal_dist_caches: Dict[str, Dict[int, float]] = {}
        starts[scene] = {}
        # Habitat's dataset loader overwrites episode_id with the episode's index in
        # its scene file, so the same ids are used here.
        for index, episode in enumerate(episodes):
            episode_id = str(index)
            goals_key = f"{os.path.basename(episode['scene_id'])}_{episode['object_category']}"
            viewpoints = [
                np.array(view_point["agent_state"]["position"], dtype=np.float32)
                for goal in data["goals_by_category"][goals_key]
                for view_point in goal["view_points"]
            ]
            rng = np.random.default_rng([args.seed, scene_seed, index])
            record, diagnostics = sample_episode(
                pathfinder,
                episode,
                viewpoints,
                pool,
                goal_dist_caches.setdefault(goals_key, {}),
                rng,
                args,
            )
            if record is None:
                failures.append({"scene": scene, "episode_id": episode_id, **diagnostics})
            else:
                starts[scene][episode_id] = record
                max_l1_error = max(max_l1_error, abs(record["l1"] - record["l1_dataset"]))
        num_failed = sum(f["scene"] == scene for f in failures)
        print(f"{scene}: {len(starts[scene])} sampled, {num_failed} failed, navigable area {pathfinder.navigable_area:.1f} m^2")
        sim.close()

    output = {
        "meta": {
            "seed": args.seed,
            "min_dist": args.min_dist,
            "max_dist": args.max_dist,
            "dist_tol": args.dist_tol,
            "max_dy": args.max_dy,
            "min_goal_dist": args.min_goal_dist,
            "goal_floor_dy": args.goal_floor_dy,
            "pool_size": args.pool_size,
            "max_draws": args.max_draws,
            "agent_radius": args.agent_radius,
            "agent_height": args.agent_height,
            "frame": "habitat world; rotation is [x, y, z, w]",
            "num_sampled": sum(len(v) for v in starts.values()),
            "num_failed": len(failures),
            "max_l1_error": max_l1_error,
        },
        "starts": starts,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # No filename or mtime in the gzip header, and sorted keys, keep the file
    # byte-identical across runs
    with open(args.output, "wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(output, sort_keys=True).encode())
    print(
        f"Wrote {args.output}: {output['meta']['num_sampled']} sampled, {len(failures)} failed, "
        f"max |L1 - dataset L1| = {max_l1_error:.2e} m"
    )


if __name__ == "__main__":
    main()
