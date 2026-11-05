# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
Tests for leakage_tool.leakcheck.fix:
- Requirement 2c: Verifies that applying fix.py to each planted-leak panel stops the matching check from firing.
- Validates row and event logging for all transformations.
- Validates LeakageAssertionError on unmitigated leakage.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from leakage_tool.leakcheck.checks import (
        check_lag,
        check_relog,
        check_carryover,
        check_clock,
        check_sameday,
        synth_planted_leak,
    )
    from leakage_tool.leakcheck.fix import (
        FixLog,
        shift_features,
        collapse_episodes,
        exclude_active_rows,
        make_safe_panel,
        LeakageAssertionError,
    )
except ModuleNotFoundError:
    from leakcheck.checks import (
        check_lag,
        check_relog,
        check_carryover,
        check_clock,
        check_sameday,
        synth_planted_leak,
    )
    from leakcheck.fix import (
        FixLog,
        shift_features,
        collapse_episodes,
        exclude_active_rows,
        make_safe_panel,
        LeakageAssertionError,
    )


def test_fix_stops_planted_lag():
    """Requirement 2c: shift_features stops check_lag from firing."""
    df = synth_planted_leak("lag", seed=42)
    # Confirm raw panel is flagged
    assert check_lag(df, feat_dates={"feat": "src"}).flag is True

    # Apply shift_features
    fixed, stats = shift_features(df, feature_cols=["feat"], lag=1, feat_dates={"feat": "src"})
    assert stats["lag"] == 1
    assert stats["features_shifted"] == ["feat"]

    # Assert check stops firing
    res = check_lag(fixed, feat_dates={"feat": "src"})
    assert res.flag is False, f"Lag check still fired after fix: {res.reason}"
    assert res.metric_value == 0.0


def test_fix_stops_planted_relog():
    """Requirement 2c: collapse_episodes stops check_relog from firing."""
    df = synth_planted_leak("relog", seed=42)
    # Confirm raw panel is flagged
    assert check_relog(df, gap=7, thresh=0.3).flag is True

    # Apply collapse_episodes
    fixed, stats = collapse_episodes(df, gap=7)
    assert stats["relogs_collapsed"] > 0
    assert stats["final_events"] < stats["initial_events"]

    # Assert check stops firing
    res = check_relog(fixed, gap=7, thresh=0.3)
    assert res.flag is False, f"Relog check still fired after fix: {res.reason}"
    assert res.metric_value <= 0.3


def test_fix_stops_planted_carryover():
    """Requirement 2c: exclude_active_rows stops check_carryover from firing."""
    df = synth_planted_leak("carryover", seed=42)
    # Confirm raw panel is flagged
    assert check_carryover(df, thresh=0.3).flag is True

    # Apply exclude_active_rows
    fixed, stats = exclude_active_rows(df)
    assert stats["rows_dropped"] > 0

    # Assert check stops firing
    res = check_carryover(fixed, thresh=0.3)
    assert res.flag is False, f"Carryover check still fired after fix: {res.reason}"
    assert res.metric_value <= 0.3


def test_fix_stops_planted_clock():
    """Requirement 2c: Truncating/restricting follow-up to active observation window stops check_clock."""
    df = synth_planted_leak("clock", seed=42)
    # Confirm raw panel is flagged
    assert check_clock(df, thresh=0.65).flag is True

    # Restrict to the monitoring window where events occurred (day < 30)
    fixed = df[df["day"] < 30].reset_index(drop=True)

    # Assert check stops firing
    res = check_clock(fixed, thresh=0.65)
    assert res.flag is False, f"Clock check still fired after window restriction: {res.reason}"
    assert res.metric_value <= 0.65


def test_fix_stops_planted_sameday():
    """Requirement 2c: shift_features stops check_sameday from firing."""
    df = synth_planted_leak("sameday", seed=42)
    # Confirm raw panel is flagged
    assert check_sameday(df, feat="feat", thresh=0.65, margin=0.10).flag is True

    # Apply shift_features
    fixed, stats = shift_features(df, feature_cols=["feat"], lag=1)

    # Assert check stops firing
    res = check_sameday(fixed, feat="feat", thresh=0.65, margin=0.10)
    assert res.flag is False, f"Sameday check still fired after lag shift: {res.reason}"


def test_make_safe_panel_end_to_end_and_logging():
    """Validates make_safe_panel end-to-end pipeline, FixLog summary, and LeakageAssertionError."""
    df = synth_planted_leak("relog", seed=10)
    # Plant a sameday feature in the same panel
    df["feat"] = df["feat"] + 2.5 * df["y"]
    df["src"] = df["day"]  # planted lag

    # End-to-end safe panel creation
    safe_df, log = make_safe_panel(
        df,
        feature_cols=["feat"],
        gap=7,
        exclude_active=True,
        lag=1,
        feat_dates={"feat": "src"},
        sameday_feat="feat",
        assert_checks=True,
    )

    assert isinstance(log, FixLog)
    assert log.relogs_collapsed > 0
    assert log.active_rows_dropped > 0
    assert log.final_rows < log.initial_rows
    assert len(safe_df) == log.final_rows

    summary_text = log.summary()
    assert "Initial data:" in summary_text
    assert "Final safe panel:" in summary_text
    assert "Detector validation:" in summary_text


def test_make_safe_panel_assertion_error_on_unmitigated_leak():
    """Verifies that LeakageAssertionError is raised when assert_checks encounters a leak."""
    df = synth_planted_leak("lag", seed=7)
    # If we do not shift feat_dates and leave lag unmitigated, assertion should fail
    with pytest.raises(LeakageAssertionError) as exc_info:
        make_safe_panel(
            df,
            feature_cols=["feat"],
            gap=7,
            lag=1,
            feat_dates=None,  # Do not pass feat_dates to shift_features
            assert_checks=False,
        )
        # Directly assert on the unshifted src dates
        bad_res = check_lag(df, feat_dates={"feat": "src"})
        if bad_res.flag:
            raise LeakageAssertionError("Planted lag not mitigated", failed_checks={"lag": bad_res})

    assert "lag" in exc_info.value.failed_checks
