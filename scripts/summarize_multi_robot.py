"""Summarizes the per-episode results of several methods on the same episodes.

Each run is a directory of per-episode JSONs (the main/ directory written by
eval_multi_robot.py). Prints a markdown table, then paired comparisons of every method
against the first one, and writes everything to a JSON file.

Usage (from the repo root):
    python scripts/summarize_multi_robot.py \\
        independent=outputs/eval/independent_dev/main \\
        shared_maps_only=outputs/eval/shared_maps_only_dev/main \\
        shared=outputs/eval/shared_vlfm_dev/main --output outputs/eval/phase3_dev.json
"""

import argparse
import glob
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

COVERAGE_STEPS = (50, 100, 250, 500)  # joint steps at which coverage is reported
COVERAGE_INTERVAL = 10  # see vlfm.multi_robot.team_stats
BOOTSTRAP_SAMPLES = 10000


def load_run(directory: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    results = {}
    for path in glob.glob(os.path.join(directory, "*.json")):
        with open(path) as f:
            result = json.load(f)
        results[(result["scene_id"], str(result["episode_id"]))] = result
    return results


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (center - half, center + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(differences: np.ndarray, seed: int = 0) -> Tuple[float, float]:
    """95% interval of the mean of paired differences."""
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, size=(BOOTSTRAP_SAMPLES, len(differences)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def team_stat(result: Dict[str, Any], key: str) -> float:
    return float(result.get("team_stats", {}).get(key, math.nan))


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    successes = [r["success"] for r in results]
    num_successes = int(sum(successes))
    summary: Dict[str, Any] = {
        "episodes": n,
        "success_rate": num_successes / n,
        "success_rate_ci95": wilson_interval(num_successes, n),
        "team_spl": float(np.mean([r["team_spl"] for r in results])),
        "team_spl_l1": float(np.mean([r["team_spl_l1"] for r in results])),
        "winner_spl": float(np.mean([r["winner_spl"] for r in results])),
        "joint_steps": float(np.mean([r["multi_robot"]["joint_steps"] for r in results])),
        "joint_steps_successes_median": float(
            np.median([r["multi_robot"]["joint_steps"] for r in results if r["success"]]) if num_successes else math.nan
        ),
        "total_actions": float(np.mean([r["multi_robot"]["total_actions"] for r in results])),
        "repeated_coverage": float(np.nanmean([team_stat(r, "repeated_coverage") for r in results])),
        "failure_causes": dict(Counter(r["failure_cause"] for r in results).most_common()),
    }
    for steps in COVERAGE_STEPS:
        index = steps // COVERAGE_INTERVAL - 1
        summary[f"coverage_m2_at_{steps}"] = float(np.mean([r["coverage_curve"][index] for r in results]))
    # Frontier overlap pooled over all joint steps in which every robot explored
    explore_steps = np.array([team_stat(r, "all_explore_steps") for r in results])
    for key in ("frontier_overlap_0p5m", "frontier_overlap_1m", "frontier_overlap_2m"):
        rates = np.array([team_stat(r, key) for r in results])
        total = np.nansum(explore_steps)
        summary[key] = float(np.nansum(rates * explore_steps) / total) if total > 0 else math.nan
    timeouts = [r["target_claim_timeouts"] for r in results if "target_claim_timeouts" in r]
    if timeouts:
        summary["target_claim_timeouts_mean"] = float(np.mean(timeouts))
        summary["target_claim_timeouts_hist"] = {
            label: sum(1 for t in timeouts if (t >= 3 if label == "3+" else t == int(label))) for label in ("0", "1", "2", "3+")
        }
    return summary


def compare(reference: List[Dict[str, Any]], other: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Paired comparison on the same episodes, in the same order."""
    ref_success = np.array([r["success"] for r in reference]) > 0
    other_success = np.array([r["success"] for r in other]) > 0
    b = int(np.sum(other_success & ~ref_success))  # only the other method succeeds
    c = int(np.sum(ref_success & ~other_success))
    spl_diff = np.array([o["team_spl"] - r["team_spl"] for r, o in zip(reference, other)])
    steps_diff = np.array([o["multi_robot"]["joint_steps"] - r["multi_robot"]["joint_steps"] for r, o in zip(reference, other)])
    return {
        "episodes": len(reference),
        "success_rate_diff": float(other_success.mean() - ref_success.mean()),
        "only_this_succeeds": b,
        "only_reference_succeeds": c,
        "mcnemar_p": mcnemar_exact_p(b, c),
        "team_spl_diff": float(spl_diff.mean()),
        "team_spl_diff_ci95": paired_bootstrap(spl_diff),
        "joint_steps_diff": float(steps_diff.mean()),
        "joint_steps_diff_ci95": paired_bootstrap(steps_diff.astype(float)),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "–" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="name=directory; the first one is the reference")
    parser.add_argument("--output", default=None, help="JSON file for the full summary")
    args = parser.parse_args()

    runs = {}
    for run in args.runs:
        name, directory = run.split("=", 1)
        runs[name] = load_run(directory)
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    for name, results in runs.items():
        if len(results) != len(common):
            print(f"Note: {name} has {len(results)} episodes; using the {len(common)} that all runs have")

    summaries = {name: summarize([results[e] for e in common]) for name, results in runs.items()}
    names = list(runs)
    rows = [
        ("Episodes", "episodes", 0),
        ("Team success rate", "success_rate", 3),
        ("Team SPL (L = min(L1, L2))", "team_spl", 3),
        ("Team SPL (L = L1)", "team_spl_l1", 3),
        ("Winner SPL", "winner_spl", 3),
        ("Joint steps (mean)", "joint_steps", 1),
        ("Joint steps of successes (median)", "joint_steps_successes_median", 1),
        ("Total actions (mean)", "total_actions", 1),
        *[(f"Coverage at {s} joint steps (m²)", f"coverage_m2_at_{s}", 1) for s in COVERAGE_STEPS],
        ("Repeated coverage", "repeated_coverage", 3),
        ("Same frontier (< 0.5 m)", "frontier_overlap_0p5m", 3),
        ("Same or close frontier (< 1 m)", "frontier_overlap_1m", 3),
        ("Frontiers < 2 m apart", "frontier_overlap_2m", 3),
        ("Claim timeouts per episode (mean)", "target_claim_timeouts_mean", 2),
    ]
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, key, digits in rows:
        print(f"| {label} | " + " | ".join(fmt(summaries[n].get(key, math.nan), digits) for n in names) + " |")
    for name in names:
        s = summaries[name]
        lo, hi = s["success_rate_ci95"]
        print(f"\n{name}: success rate 95% CI [{lo:.3f}, {hi:.3f}]; failure causes {s['failure_causes']}")
        if "target_claim_timeouts_hist" in s:
            print(f"  claim timeouts per episode (0/1/2/3+): {s['target_claim_timeouts_hist']}")

    reference = names[0]
    comparisons = {}
    for name in names[1:]:
        c = compare([runs[reference][e] for e in common], [runs[name][e] for e in common])
        comparisons[name] = c
        print(
            f"\n{name} vs {reference} (paired, {c['episodes']} episodes):"
            f" success rate {c['success_rate_diff']:+.3f}"
            f" (only {name}: {c['only_this_succeeds']}, only {reference}: {c['only_reference_succeeds']},"
            f" McNemar p = {c['mcnemar_p']:.3f});"
            f" team SPL {c['team_spl_diff']:+.3f} [{c['team_spl_diff_ci95'][0]:+.3f}, {c['team_spl_diff_ci95'][1]:+.3f}];"
            f" joint steps {c['joint_steps_diff']:+.1f}"
            f" [{c['joint_steps_diff_ci95'][0]:+.1f}, {c['joint_steps_diff_ci95'][1]:+.1f}]"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"episodes": [list(e) for e in common], "summaries": summaries, "comparisons": comparisons}, f, indent=1)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
