"""Runs one method on a fixed episode split, with integrity checks.

The evaluation machine occasionally corrupts memory. This script therefore never
trusts a run's exit code or its console log. It works from the per-episode JSON files:

  - An episode counts as done only when its JSON file exists.
  - An episode whose JSON has hw_corruption = true is moved aside and rerun.
  - Episodes are run scene by scene, so an interrupted evaluation resumes where it
    stopped: rerunning the same command only runs what is still missing.
  - A fixed fraction of the episodes (20% by default) is run a second time and
    compared with the first run field by field. Runs are deterministic, so a mismatch
    means one of the two was silently corrupted; a third run then decides.

Results are written under outputs/eval/<method>/ (ignored by git):
    main/        one JSON per episode: the results to report
    dup/         second runs of the duplicated episodes
    tiebreak/    third runs, only for episodes whose first two runs disagree
    */corrupted/ JSON files of runs that were flagged or rejected
    */logs/      console log of every run
    integrity.json   what was rerun and why, and the outcome of the duplicate check

Usage (from the repo root):
    python scripts/eval_multi_robot.py --method independent
    python scripts/eval_multi_robot.py --method single --split tune
"""

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Episode = Tuple[str, str]  # (scene, episode_id)

SERVER_PORTS = [12181, 12182, 12183, 12184]
COMPARED_FIELDS = ["success", "distance_to_goal", "num_steps", "stop_called", "team_spl", "multi_robot"]

TWO_ROBOTS = ["habitat_baselines.rl.policy.num_robots=2"]
METHODS: Dict[str, List[str]] = {
    "single": [
        "habitat.task.lab_sensors.active_robot_id.num_robots=1",
        "habitat_baselines.rl.policy.num_robots=1",
        "habitat.environment.max_episode_steps=500",
    ],
    "independent": TWO_ROBOTS + ["habitat_baselines.rl.policy.map_sharing=independent"],
    "shared_independent_utility": TWO_ROBOTS
    + ["habitat_baselines.rl.policy.map_sharing=shared", "habitat_baselines.rl.policy.assigner=independent_utility"],
    "joint": TWO_ROBOTS + ["habitat_baselines.rl.policy.map_sharing=shared", "habitat_baselines.rl.policy.assigner=joint"],
    "greedy": TWO_ROBOTS + ["habitat_baselines.rl.policy.map_sharing=shared", "habitat_baselines.rl.policy.assigner=greedy"],
    "shared_vlfm": TWO_ROBOTS + ["habitat_baselines.rl.policy.map_sharing=shared", "habitat_baselines.rl.policy.assigner=vlfm"],
    # Diagnostic: shared obstacle and value maps, but each robot its own object map
    "shared_maps_only": TWO_ROBOTS
    + ["habitat_baselines.rl.policy.map_sharing=shared_maps_only", "habitat_baselines.rl.policy.assigner=vlfm"],
}


def result_path(directory: str, episode: Episode) -> str:
    scene, episode_id = episode
    return os.path.join(directory, f"{episode_id}_{scene}.json")


def load_result(directory: str, episode: Episode) -> Optional[Dict[str, Any]]:
    """The episode's result, or None if it is missing or unreadable."""
    try:
        with open(result_path(directory, episode)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pending_episodes(directory: str, episodes: List[Episode]) -> List[Episode]:
    return [episode for episode in episodes if load_result(directory, episode) is None]


def set_aside(directory: str, episode: Episode, reason: str) -> None:
    """Moves an episode's result out of the way so that it gets rerun."""
    corrupted_dir = os.path.join(directory, "corrupted")
    os.makedirs(corrupted_dir, exist_ok=True)
    scene, episode_id = episode
    count = len([name for name in os.listdir(corrupted_dir) if name.startswith(f"{episode_id}_{scene}.")])
    shutil.move(result_path(directory, episode), os.path.join(corrupted_dir, f"{episode_id}_{scene}.{reason}{count}.json"))


def set_aside_flagged(directory: str, episodes: List[Episode], log: Dict[str, Any], attempt: int) -> None:
    """Sets aside every result flagged with hw_corruption, so that the episode is rerun."""
    for episode in episodes:
        result = load_result(directory, episode)
        if result is not None and result.get("hw_corruption", False):
            set_aside(directory, episode, "hw_corruption")
            log["hw_corruption"].append({"pass": os.path.basename(directory), "episode": episode, "attempt": attempt})


def differing_fields(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    return [field for field in COMPARED_FIELDS if a.get(field) != b.get(field)]


def choose_duplicates(episodes: List[Episode], fraction: float, seed: int) -> List[Episode]:
    """A fixed random sample of the episodes to run twice."""
    count = min(len(episodes), math.ceil(fraction * len(episodes)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(episodes), size=count, replace=False)
    return [episodes[i] for i in sorted(chosen)]


def servers_up() -> bool:
    for port in SERVER_PORTS:
        with socket.socket() as s:
            s.settimeout(2)
            if s.connect_ex(("localhost", port)) != 0:
                return False
    return True


def stop_servers() -> None:
    sessions = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True).stdout.split()
    for session in sessions:
        if session.startswith("vlm_servers"):
            subprocess.run(["tmux", "kill-session", "-t", session])
    subprocess.run(["pkill", "-f", r"python -m vlfm\.vlm\.(grounding_dino|blip2itm|sam|yolov7)"])


def ensure_servers(log: Dict[str, Any]) -> None:
    """Restarts all model servers if any of them is not listening."""
    if servers_up():
        return
    log["server_restarts"] += 1
    stop_servers()
    time.sleep(5)
    subprocess.run(["./scripts/launch_vlm_servers.sh"], check=True, stdout=subprocess.DEVNULL)
    deadline = time.time() + 300
    while not servers_up():
        if time.time() > deadline:
            raise RuntimeError("The model servers did not come up within 5 minutes")
        time.sleep(5)


def run_episodes(directory: str, episodes: List[Episode], args: argparse.Namespace, tag: str) -> None:
    """Runs the given episodes (all from one scene) in one evaluation process."""
    os.makedirs(os.path.join(directory, "logs"), exist_ok=True)
    scene = episodes[0][0]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"scene": s, "episode_id": e} for s, e in episodes], f)
    command = [
        sys.executable,
        "-m",
        "vlfm.run",
        "--config-name",
        args.config_name,
        "habitat.dataset.type=ObjectNavSubset-v1",
        f"habitat.dataset.content_scenes=[{scene}]",
        f"habitat_baselines.test_episode_count={len(episodes)}",
        *METHODS[args.method],
        *args.overrides,
    ]
    env = {**os.environ, "VLFM_EPISODE_LIST": f.name, "ZSOS_LOG_DIR": os.path.abspath(directory)}
    try:
        with open(os.path.join(directory, "logs", f"{scene}.{tag}.log"), "w") as log_file:
            subprocess.run(
                command, env=env, stdout=log_file, stderr=subprocess.STDOUT, timeout=args.episode_timeout * len(episodes) + 600
            )
    except subprocess.TimeoutExpired:
        pass  # whatever is missing afterwards is rerun
    finally:
        os.unlink(f.name)


def run_pass(directory: str, episodes: List[Episode], args: argparse.Namespace, log: Dict[str, Any]) -> List[Episode]:
    """Runs episodes until each has an unflagged result. Returns those that still have
    none after the maximum number of attempts."""
    os.makedirs(directory, exist_ok=True)
    name = os.path.basename(directory)
    for attempt in range(1, args.max_attempts + 1):
        # Also catches flagged results left behind by an interrupted earlier run
        set_aside_flagged(directory, episodes, log, attempt)
        pending = pending_episodes(directory, episodes)
        if not pending:
            break
        print(f"[{name}] attempt {attempt}: {len(pending)} of {len(episodes)} episodes to run")
        for scene in sorted({scene for scene, _ in pending}):
            chunk = [episode for episode in pending if episode[0] == scene]
            ensure_servers(log)
            run_episodes(directory, chunk, args, f"attempt{attempt}")
            for episode in chunk:
                if load_result(directory, episode) is None:
                    log["missing_after_run"].append({"pass": name, "episode": episode, "attempt": attempt})
            set_aside_flagged(directory, chunk, log, attempt)
            done = len(episodes) - len(pending_episodes(directory, episodes))
            print(f"[{name}] {scene}: {done} of {len(episodes)} episodes done")
    return pending_episodes(directory, episodes)


def check_duplicates(out_dir: str, duplicates: List[Episode], args: argparse.Namespace, log: Dict[str, Any]) -> None:
    """Compares the two runs of each duplicated episode; a third run settles a mismatch."""
    main_dir, dup_dir, tiebreak_dir = (os.path.join(out_dir, name) for name in ("main", "dup", "tiebreak"))
    mismatched = []
    for episode in duplicates:
        first, second = load_result(main_dir, episode), load_result(dup_dir, episode)
        if first is None or second is None:
            continue
        fields = differing_fields(first, second)
        if fields:
            mismatched.append(episode)
            log["duplicate_mismatches"].append({"episode": episode, "fields": fields})
    log["duplicates_compared"] = len(duplicates)
    if not mismatched:
        return

    run_pass(tiebreak_dir, mismatched, args, log)
    for entry in log["duplicate_mismatches"]:
        episode = tuple(entry["episode"])
        first, second, third = (load_result(d, episode) for d in (main_dir, dup_dir, tiebreak_dir))
        if third is None:
            entry["resolution"] = "unresolved: no third run"
        elif not differing_fields(first, third):
            entry["resolution"] = "main kept: third run agrees with it"
        elif not differing_fields(second, third):
            set_aside(main_dir, episode, "duplicate_mismatch")
            shutil.copy(result_path(dup_dir, episode), result_path(main_dir, episode))
            entry["resolution"] = "main replaced by the second run: third run agrees with that one"
        else:
            entry["resolution"] = "unresolved: all three runs differ"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--split", default="eval", help="a split of the split file: eval, tune, dev, ...")
    parser.add_argument("--split-file", default="data/multi_robot/hm3d_val_splits.json")
    parser.add_argument("--config-name", default="experiments/vlfm_objectnav_hm3d_multi")
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--name", default=None, help="output directory name (default: <method>_<split>)")
    parser.add_argument("--dup-fraction", type=float, default=0.2, help="fraction of episodes to run twice")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the duplicated episodes")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--episode-timeout", type=int, default=1200, help="seconds allowed per episode")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N episodes of the split")
    parser.add_argument("--keep-servers", action="store_true", help="leave the model servers running at the end")
    parser.add_argument("overrides", nargs="*", help="extra config overrides passed to vlfm.run")
    args = parser.parse_args()

    with open(args.split_file) as f:
        splits = json.load(f)
    if not isinstance(splits.get(args.split), list):
        parser.error(f"{args.split_file} has no split {args.split!r}")
    episodes: List[Episode] = [(entry["scene"], str(entry["episode_id"])) for entry in splits[args.split]]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    out_dir = os.path.join(args.output_root, args.name or f"{args.method}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    log: Dict[str, Any] = {
        "method": args.method,
        "split": args.split,
        "num_episodes": len(episodes),
        "overrides": METHODS[args.method] + args.overrides,
        "server_restarts": 0,
        "hw_corruption": [],
        "missing_after_run": [],
        "duplicate_mismatches": [],
    }
    try:
        duplicates = choose_duplicates(episodes, args.dup_fraction, args.seed)
        log["duplicated_episodes"] = duplicates
        log["unfinished_main"] = run_pass(os.path.join(out_dir, "main"), episodes, args, log)
        log["unfinished_dup"] = run_pass(os.path.join(out_dir, "dup"), duplicates, args, log)
        check_duplicates(out_dir, duplicates, args, log)
    finally:
        with open(os.path.join(out_dir, "integrity.json"), "w") as f:
            json.dump(log, f, indent=1)
        if not args.keep_servers:
            stop_servers()

    print(
        f"\n{len(episodes) - len(log['unfinished_main'])} of {len(episodes)} episodes done."
        f" Rerun for hw_corruption: {len(log['hw_corruption'])}."
        f" Runs that left no result: {len(log['missing_after_run'])}."
        f" Duplicate check: {len(log['duplicate_mismatches'])} of {len(duplicates)} disagreed."
        f" Server restarts: {log['server_restarts']}."
    )
    unresolved = [m for m in log["duplicate_mismatches"] if m.get("resolution", "").startswith("unresolved")]
    if log["unfinished_main"] or log["unfinished_dup"] or unresolved:
        print("INCOMPLETE: see integrity.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
