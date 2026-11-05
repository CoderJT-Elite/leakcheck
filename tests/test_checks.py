# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
Tests for leakage_tool.leakcheck.checks:
- Validates 5 planted leak detectors across 100 seeds (requires >= 95% catch rate).
- Validates clean panel across 100 seeds (requires < 5% false positive rate).
- Validates CheckResult object behavior and edge-case error handling.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from leakage_tool.leakcheck.checks import (
        CheckResult,
        check_lag,
        check_relog,
        check_carryover,
        check_clock,
        check_sameday,
        run_all_checks,
        synth_planted_leak,
    )
except ModuleNotFoundError:
    from leakcheck.checks import (
        CheckResult,
        check_lag,
        check_relog,
        check_carryover,
        check_clock,
        check_sameday,
        run_all_checks,
        synth_planted_leak,
    )


def test_planted_lag_detector_100_seeds():
    """Requirement 2a: Planted lag detector flags >= 95 of 100 seeds."""
    catches = 0
    total_seeds = 100
    for seed in range(total_seeds):
        df = synth_planted_leak("lag", seed=seed)
        res = check_lag(df, feat_dates={"feat": "src"})
        assert isinstance(res, CheckResult)
        assert bool(res) is res.flag
        if res.flag:
            catches += 1

    catch_rate = catches / total_seeds
    print(f"\n[Planted Leak] Lag detector catch rate: {catches}/{total_seeds} ({catch_rate:.1%})")
    assert catches >= 95, f"Lag detector caught only {catches}/{total_seeds} (expected >= 95)"


def test_planted_relog_detector_100_seeds():
    """Requirement 2a: Planted relog detector flags >= 95 of 100 seeds."""
    catches = 0
    total_seeds = 100
    for seed in range(total_seeds):
        df = synth_planted_leak("relog", seed=seed)
        res = check_relog(df, gap=7, thresh=0.3)
        assert isinstance(res, CheckResult)
        if res.flag:
            catches += 1

    catch_rate = catches / total_seeds
    print(f"\n[Planted Leak] Relog detector catch rate: {catches}/{total_seeds} ({catch_rate:.1%})")
    assert catches >= 95, f"Relog detector caught only {catches}/{total_seeds} (expected >= 95)"


def test_planted_carryover_detector_100_seeds():
    """Requirement 2a: Planted carryover detector flags >= 95 of 100 seeds."""
    catches = 0
    total_seeds = 100
    for seed in range(total_seeds):
        df = synth_planted_leak("carryover", seed=seed)
        res = check_carryover(df, thresh=0.3)
        assert isinstance(res, CheckResult)
        if res.flag:
            catches += 1

    catch_rate = catches / total_seeds
    print(f"\n[Planted Leak] Carryover detector catch rate: {catches}/{total_seeds} ({catch_rate:.1%})")
    assert catches >= 95, f"Carryover detector caught only {catches}/{total_seeds} (expected >= 95)"


def test_planted_clock_detector_100_seeds():
    """Requirement 2a: Planted study clock detector flags >= 95 of 100 seeds."""
    catches = 0
    total_seeds = 100
    for seed in range(total_seeds):
        df = synth_planted_leak("clock", seed=seed)
        res = check_clock(df, thresh=0.65)
        assert isinstance(res, CheckResult)
        if res.flag:
            catches += 1

    catch_rate = catches / total_seeds
    print(f"\n[Planted Leak] Clock detector catch rate: {catches}/{total_seeds} ({catch_rate:.1%})")
    assert catches >= 95, f"Clock detector caught only {catches}/{total_seeds} (expected >= 95)"


def test_planted_sameday_detector_100_seeds():
    """Requirement 2a: Planted same-day feature detector flags >= 95 of 100 seeds."""
    catches = 0
    total_seeds = 100
    for seed in range(total_seeds):
        df = synth_planted_leak("sameday", seed=seed)
        res = check_sameday(df, feat="feat", thresh=0.65, margin=0.10)
        assert isinstance(res, CheckResult)
        if res.flag:
            catches += 1

    catch_rate = catches / total_seeds
    print(f"\n[Planted Leak] Sameday detector catch rate: {catches}/{total_seeds} ({catch_rate:.1%})")
    assert catches >= 95, f"Sameday detector caught only {catches}/{total_seeds} (expected >= 95)"


def test_clean_panel_false_positive_rate_100_seeds():
    """Requirement 2b: Clean panel is flagged by NONE of the checks in >= 95 of 100 seeds (< 5 false positives)."""
    false_positives = 0
    total_seeds = 100
    per_check_fp = {"lag": 0, "relog": 0, "carryover": 0, "clock": 0, "sameday": 0}

    for seed in range(total_seeds):
        df = synth_planted_leak("clean", seed=seed)
        res = run_all_checks(df, feat_dates={"feat": "src"}, sameday_feat="feat")
        flagged_any = False
        for name, r in res.items():
            if r.flag:
                per_check_fp[name] += 1
                flagged_any = True
        if flagged_any:
            false_positives += 1

    fp_rate = false_positives / total_seeds
    print(f"\n[Clean Panel] False positive rate across 100 seeds: {false_positives}/{total_seeds} ({fp_rate:.1%})")
    print(f"  Per-check false positives: {per_check_fp}")
    assert false_positives < 5, f"Clean panel had {false_positives}/100 false positives (expected < 5)"


def test_check_result_structure_and_edge_cases():
    """Validates CheckResult object fields, boolean conversion, and empty/single-class edge cases."""
    # Empty dataframe
    empty_df = pd.DataFrame()
    r_lag = check_lag(empty_df, feat_dates={})
    assert not r_lag.flag
    assert r_lag.metric_value == 0.0

    # Single class label
    single_class_df = pd.DataFrame({"player": [0, 0], "day": [1, 2], "y": [0, 0], "feat": [1.0, 2.0]})
    r_clock = check_clock(single_class_df)
    assert not r_clock.flag
    assert "Fewer than 2 classes" in r_clock.reason

    r_sameday = check_sameday(single_class_df, feat="feat")
    assert not r_sameday.flag

    # String representation
    assert "[CLEAN]" in str(r_clock)
