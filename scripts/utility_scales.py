"""Fixed utility scales s_V and s_A from a single-robot run with log_utility_scales on.

Method B (used): the interquartile range of the frontier values V and of the path costs
A (in actions) among the frontiers of each exploration step, over steps with at least
3 frontiers; s_V and s_A are the medians of these over all steps. Method A (the earlier
plan, kept for comparison): the median over episodes of each episode's median V and A.
See docs/multi_robot_design.md, section 3.4.

Usage:
    python scripts/utility_scales.py outputs/eval/p4scale_tune_iqr/main
"""

import glob
import json
import os
import sys

import numpy as np


def main(run_dir: str) -> None:
    episodes = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run_dir, "*.json")))]
    logs = [e["utility_scales"] for e in episodes if e.get("utility_scales", {}).get("frontiers", 0) > 0]
    steps = np.array([s for u in logs for s in u.get("step_iqrs", [])], dtype=float).reshape(-1, 3)
    if len(steps) == 0:
        sys.exit("no step_iqrs in these results: rerun with the current UtilityScaleLog")
    s_v_a = float(np.median([u["value_p50"] for u in logs]))
    s_a_a = float(np.median([u["actions_p50"] for u in logs]))
    s_v_b, s_a_b = float(np.median(steps[:, 1])), float(np.median(steps[:, 2]))

    print(f"{len(episodes)} episodes, {len(logs)} with exploration steps, {len(steps)} steps with >= 3 frontiers")
    print("\n| | s_V | s_A (actions) |\n|---|---|---|")
    print(f"| A: median of episode medians | {s_v_a:.4f} | {s_a_a:.2f} |")
    print(f"| B: median of within-step IQRs | {s_v_b:.4f} | {s_a_b:.2f} |")
    print("\nWithin-step distributions (p10 / p25 / p50 / p75 / p90):")
    for name, column in (("frontiers per step", 0), ("IQR of V", 1), ("IQR of A", 2)):
        print(f"  {name}: " + " / ".join(f"{q:.4g}" for q in np.percentile(steps[:, column], [10, 25, 50, 75, 90])))
    print("\nTypical within-step spread of the normalized terms (median IQR of V/s_V and of A/s_A):")
    for name, s_v, s_a in (("A", s_v_a, s_a_a), ("B", s_v_b, s_a_b)):
        print(f"  {name}: V/s_V {np.median(steps[:, 1] / s_v):.3f}, A/s_A {np.median(steps[:, 2] / s_a):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
