import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import summarize_multi_robot as sm  # noqa: E402


def result(success: float, spl: float, joint_steps: int, **extra: Any) -> Dict[str, Any]:
    return {
        "success": success,
        "team_spl": spl,
        "team_spl_l1": spl,
        "winner_spl": spl,
        "failure_cause": "did_not_fail" if success else "false_positive",
        "multi_robot": {"joint_steps": joint_steps, "total_actions": 2 * joint_steps},
        "team_stats": {
            "repeated_coverage": 0.5,
            "all_explore_steps": 10,
            "frontier_overlap_0p5m": 0.0,
            "frontier_overlap_1m": 0.2,
            "frontier_overlap_2m": 0.4,
        },
        "coverage_curve": [float(i) for i in range(50)],
        **extra,
    }


def test_mcnemar_exact() -> None:
    assert sm.mcnemar_exact_p(0, 0) == 1.0
    assert sm.mcnemar_exact_p(5, 5) == 1.0
    # 0 against 6: 2 * 0.5^6
    assert sm.mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)


def test_wilson_interval_contains_the_rate() -> None:
    lo, hi = sm.wilson_interval(30, 100)
    assert lo < 0.3 < hi and lo == pytest.approx(0.219, abs=1e-3) and hi == pytest.approx(0.396, abs=1e-3)


def test_summary_reads_curves_and_pools_frontier_overlap() -> None:
    results = [result(1.0, 0.5, 100), result(0.0, 0.0, 500, target_claim_timeouts=3)]
    results[1]["team_stats"]["all_explore_steps"] = 30
    results[1]["team_stats"]["frontier_overlap_1m"] = 0.6
    s = sm.summarize(results)
    assert s["success_rate"] == 0.5 and s["joint_steps"] == 300 and s["total_actions"] == 600
    assert s["coverage_m2_at_50"] == 4.0 and s["coverage_m2_at_500"] == 49.0
    # (0.2 * 10 + 0.6 * 30) / 40
    assert s["frontier_overlap_1m"] == pytest.approx(0.5)
    assert s["target_claim_timeouts_hist"] == {"0": 0, "1": 0, "2": 0, "3+": 1}


def test_paired_comparison_counts_discordant_episodes() -> None:
    reference = [result(1.0, 0.4, 100), result(0.0, 0.0, 500), result(1.0, 0.2, 200)]
    other = [result(1.0, 0.5, 80), result(1.0, 0.3, 300), result(0.0, 0.0, 500)]
    c = sm.compare(reference, other)
    assert (c["only_this_succeeds"], c["only_reference_succeeds"]) == (1, 1)
    assert c["success_rate_diff"] == 0.0
    assert c["team_spl_diff"] == pytest.approx((0.1 + 0.3 - 0.2) / 3)
    assert c["joint_steps_diff"] == pytest.approx((-20 - 200 + 300) / 3)
