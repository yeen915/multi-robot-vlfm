"""Diagnostics of the frontier coordinator from per-episode JSONs (assigner joint,
greedy or independent_utility), to judge d_min and the hysteresis tau.

For each episode: how often d_min changed the best pair and what that cost in path
actions, how close the robots and their goals were, and goal flip-flops (a robot going
back to a frontier it left within --window steps). Steps are environment steps, which
are also the video frame indices.

Usage:
    python scripts/coordinator_diagnostics.py outputs/eval/p4video_joint/main [--window 30]
"""

import argparse
import glob
import json
import os

import numpy as np

SAME = 0.5  # m, as SAME_FRONTIER_DISTANCE in the coordinator


def flip_flops(changes, window):
    """(step_left, step_back, robot, frontier) for each return to a frontier left within window steps."""
    found = []
    by_robot = {}
    for step, robot, x, y in changes:
        by_robot.setdefault(robot, []).append((step, np.array([x, y])))
    for robot, goals in by_robot.items():
        for k in range(2, len(goals)):
            step, goal = goals[k]
            for m in range(k - 2, -1, -1):
                if step - goals[m + 1][0] > window:
                    break
                if np.linalg.norm(goals[m][1] - goal) <= SAME:
                    found.append((goals[m + 1][0], step, robot, goal.round(2).tolist()))
                    break
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--window", type=int, default=30, help="steps within which a return counts as a flip-flop")
    parser.add_argument("--d-min", type=float, default=2.0)
    args = parser.parse_args()

    totals = {"pair": 0, "binding": 0, "extra": 0.0, "infeasible": 0, "single": 0, "flips": 0, "changes": 0}
    robot_d, goal_d = [], []
    print("| episode | success | steps | pair decisions | d_min binding | extra actions | infeasible | goal changes | flip-flops (left -> back, robot) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        e = json.load(open(f))
        c = e.get("coordinator")
        if not c:
            continue
        flips = flip_flops(c["goal_changes"], args.window)
        totals["pair"] += c["pair_decisions"]
        totals["binding"] += c["spacing_binding"]
        totals["extra"] += c["spacing_extra_actions"]
        totals["infeasible"] += c["spacing_infeasible"]
        totals["single"] += c["single_frontier"]
        totals["flips"] += len(flips)
        totals["changes"] += len(c["goal_changes"])
        robot_d += c["robot_distances"]
        goal_d += c["goal_distances"]
        name = os.path.basename(f)[:-5]
        flip_text = ", ".join(f"{a}->{b} R{r + 1}" for a, b, r, _ in flips) or "-"
        print(
            f"| {name} | {int(e['success'])} | {e['num_steps']} | {c['pair_decisions']} | {c['spacing_binding']} |"
            f" {c['spacing_extra_actions']:.1f} | {c['spacing_infeasible']} | {len(c['goal_changes'])} | {flip_text} |"
        )
    t = totals
    print(f"\nPair decisions {t['pair']}: d_min changed the best pair in {t['binding']}"
          f" ({t['binding'] / max(t['pair'], 1):.0%}), costing {t['extra'] / max(t['binding'], 1):.1f} extra path actions"
          f" each on average; all frontiers within d_min in {t['infeasible']}; one frontier in {t['single']}.")
    print(f"Goal changes {t['changes']}, flip-flops within {args.window} steps {t['flips']}.")
    for name, d in (("robot distance", robot_d), ("goal distance", goal_d)):
        if d:
            d = np.array(d)
            print(f"{name} at pair decisions (m): p10 {np.percentile(d, 10):.1f}, median {np.median(d):.1f},"
                  f" p90 {np.percentile(d, 90):.1f}; below 1 m {np.mean(d < 1):.0%}, below d_min {np.mean(d < args.d_min):.0%}")


if __name__ == "__main__":
    main()
