"""Coordinator behaviour of several runs, split by episode outcome (success / failure):
goal switches, flip-flops, and how often d_min changed the best pair. Complements
summarize_multi_robot.py (success rate, SPL, coverage, frontier overlap).

Usage:
    python scripts/summarize_coordinator.py lam1=outputs/eval/p4lam_1/main greedy=outputs/eval/p4lam_greedy/main
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coordinator_diagnostics import flip_flops  # noqa: E402


def outcome_rows(run_dir: str):
    groups = {"success": [], "failure": []}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        e = json.load(open(f))
        c = e.get("coordinator")
        if c:
            groups["success" if e["success"] else "failure"].append((e, c))
    return groups


def main() -> None:
    runs = [arg.split("=", 1) for arg in sys.argv[1:]]
    print("| run | outcome | episodes | joint steps (mean) | goal switches / episode | switches / 100 joint steps"
          " | flip-flops / episode | pair decisions | d_min changed the pair |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, run_dir in runs:
        for outcome, rows in outcome_rows(run_dir).items():
            if not rows:
                print(f"| {name} | {outcome} | 0 | | | | | | |")
                continue
            steps = np.array([e["multi_robot"]["joint_steps"] for e, _ in rows], dtype=float)
            switches = np.array([c["switches"] for _, c in rows], dtype=float)
            flips = np.array([len(flip_flops(c["goal_changes"], 30)) for _, c in rows], dtype=float)
            pairs = sum(c["pair_decisions"] for _, c in rows)
            binding = sum(c["spacing_binding"] for _, c in rows)
            rate = f"{binding / pairs:.0%} ({binding}/{pairs})" if pairs else "-"
            print(f"| {name} | {outcome} | {len(rows)} | {steps.mean():.1f} | {switches.mean():.2f} |"
                  f" {100 * switches.sum() / steps.sum():.2f} | {flips.mean():.2f} | {pairs} | {rate} |")


if __name__ == "__main__":
    main()
