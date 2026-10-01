"""Builds fixed, scene-disjoint tuning and evaluation splits for the two-robot runs.

Only episodes that have an R2 start (see generate_robot2_starts.py) are eligible.
Scenes are split at random into tuning scenes and evaluation scenes. Within each
split, episodes are spread evenly over the scenes, and within each scene they are
drawn round-robin from the R1-R2 distance bins so that every bin is represented.

The output is deterministic for a given seed.

Usage (from the repo root):
    python scripts/make_eval_splits.py
"""

import argparse
import gzip
import json
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

DIST_BINS = [2.0, 4.0, 6.0, 8.0, 10.0]


def dist_bin(r1_r2_geodesic: float) -> int:
    """Index of the R1-R2 distance bin; the matching tolerance used when sampling can
    put a distance slightly outside [2, 10], so the ends are clamped."""
    return int(np.clip(np.digitize(r1_r2_geodesic, DIST_BINS) - 1, 0, len(DIST_BINS) - 2))


def pick_from_scene(episodes: Dict[str, Any], count: int, rng: np.random.Generator) -> List[str]:
    """Picks `count` episode ids from one scene, cycling through the distance bins."""
    by_bin: List[List[str]] = [[] for _ in range(len(DIST_BINS) - 1)]
    for episode_id in sorted(episodes, key=int):
        by_bin[dist_bin(episodes[episode_id]["r1_r2_geodesic"])].append(episode_id)
    for ids in by_bin:
        rng.shuffle(ids)

    picked: List[str] = []
    bin_order = list(rng.permutation(len(by_bin)))
    while len(picked) < count and any(by_bin):
        for b in bin_order:
            if by_bin[b] and len(picked) < count:
                picked.append(by_bin[b].pop())
    return sorted(picked, key=int)


def make_split(
    starts: Dict[str, Dict[str, Any]], scenes: List[str], total: int, rng: np.random.Generator
) -> List[Tuple[str, str]]:
    """Spreads `total` episodes over `scenes` as evenly as possible."""
    quota = {scene: total // len(scenes) for scene in scenes}
    for scene in rng.permutation(scenes)[: total % len(scenes)]:
        quota[scene] += 1
    return [(scene, episode_id) for scene in scenes for episode_id in pick_from_scene(starts[scene], quota[scene], rng)]


def describe(starts: Dict[str, Dict[str, Any]], split: List[Tuple[str, str]]) -> Dict[str, Any]:
    records = [starts[scene][episode_id] for scene, episode_id in split]
    categories = Counter(r["object_category"] for r in records)
    return {
        "num_episodes": len(records),
        "num_scenes": len({scene for scene, _ in split}),
        "episodes_per_scene": dict(sorted(Counter(scene for scene, _ in split).items())),
        "r1_r2_dist_bins": [int(c) for c in np.bincount([dist_bin(r["r1_r2_geodesic"]) for r in records], minlength=4)],
        "object_category": dict(sorted(categories.items())),
        "goal_same_floor_frac": float(np.mean([r["goal_same_floor"] for r in records])),
        "r2_closer_frac": float(np.mean([r["l2"] < r["l1"] for r in records])),
        "l1_mean": float(np.mean([r["l1"] for r in records])),
        "l2_mean": float(np.mean([r["l2"] for r in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--starts", default="data/multi_robot/hm3d_val_robot2_starts.json.gz")
    parser.add_argument("--output", default="data/multi_robot/hm3d_val_splits.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tune-scenes", type=int, default=5)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--num-tune", type=int, default=50)
    args = parser.parse_args()

    with gzip.open(args.starts, "rt") as f:
        starts = json.load(f)["starts"]

    rng = np.random.default_rng(args.seed)
    scenes = sorted(starts)
    tune_scenes = sorted(rng.choice(scenes, size=args.num_tune_scenes, replace=False).tolist())
    eval_scenes = [scene for scene in scenes if scene not in tune_scenes]

    splits = {
        "tune": make_split(starts, tune_scenes, args.num_tune, rng),
        "eval": make_split(starts, eval_scenes, args.num_eval, rng),
    }
    everything = [(scene, episode_id) for scene in scenes for episode_id in starts[scene]]
    summary = {name: describe(starts, split) for name, split in {**splits, "all_with_r2_start": everything}.items()}

    output = {
        "meta": {
            "seed": args.seed,
            "starts_file": args.starts,
            "dist_bins": DIST_BINS,
            "tune_scenes": tune_scenes,
            "eval_scenes": eval_scenes,
        },
        "summary": summary,
        **{name: [{"scene": scene, "episode_id": episode_id} for scene, episode_id in split] for name, split in splits.items()},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=1, sort_keys=True)
        f.write("\n")

    for name, stats in summary.items():
        print(f"== {name}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
