# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
fix.py: Leakage mitigation tools for panel/time-series injury prediction studies.

Transforms raw long-format panels into leakage-safe evaluation panels:
1. shift_features: Shifts all feature values to be dated strictly before the prediction day (lag >= 1).
2. collapse_episodes: Collapses repeated injury records within a temporal gap into distinct episode onsets.
3. exclude_active_rows: Drops observations where the player is already injured/unavailable.
4. make_safe_panel: End-to-end pipeline combining all fixes and asserting safety with detectors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

from .checks import (
    CheckResult,
    check_lag,
    check_relog,
    check_carryover,
    check_clock,
    check_sameday,
    run_all_checks,
)


class LeakageAssertionError(AssertionError):
    """Raised when a post-fix panel fails one or more leakage checks."""
    def __init__(self, message: str, failed_checks: Dict[str, CheckResult]):
        super().__init__(message)
        self.failed_checks = failed_checks


@dataclass
class FixLog:
    """Log of all row and event modifications performed by fix operations."""
    initial_rows: int = 0
    initial_events: int = 0
    features_shifted: List[str] = field(default_factory=list)
    lag_steps: int = 0
    relogs_collapsed: int = 0
    events_after_collapse: int = 0
    active_rows_dropped: int = 0
    final_rows: int = 0
    final_events: int = 0
    steps_applied: List[str] = field(default_factory=list)
    detector_results: Dict[str, CheckResult] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Leakage Mitigation Transformation Log ===",
            f"  Initial data:      {self.initial_rows:,} rows | {self.initial_events:,} events",
            f"  Features shifted:  {len(self.features_shifted)} features by lag={self.lag_steps}",
            f"  Repeat logs fixed: {self.relogs_collapsed:,} re-logs collapsed -> {self.events_after_collapse:,} episode onsets",
            f"  Active rows dropped: {self.active_rows_dropped:,} rows where athlete unavailable",
            f"  Final safe panel:  {self.final_rows:,} rows | {self.final_events:,} events",
            f"  Steps applied:     {', '.join(self.steps_applied)}",
        ]
        if self.detector_results:
            lines.append("  Detector validation:")
            for name, res in self.detector_results.items():
                status = "PASS" if not res.flag else "FAIL"
                lines.append(f"    [{status}] {name:10s}: {res.reason}")
        return "\n".join(lines)


def shift_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lag: int = 1,
    player_col: str = "player",
    date_col: str = "day",
    feat_dates: Optional[Union[Dict[str, str], List[str], str]] = None,
    drop_na_leads: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 1: Shift features strictly into the past.
    For each player sorted by date, shifts `feature_cols` by `lag` observations
    so that values at row t represent only information available at or before day t - lag.

    If feat_dates is provided, updates source dates accordingly (e.g. subtracting lag days).
    """
    if lag < 1:
        raise ValueError(f"Lag must be >= 1 for leakage safety; got {lag}.")

    out = df.sort_values([player_col, date_col]).copy()
    grouped = out.groupby(player_col)

    # Shift feature columns
    for col in feature_cols:
        if col in out.columns:
            out[col] = grouped[col].shift(lag)

    # Adjust source date columns if provided
    if feat_dates is not None:
        if isinstance(feat_dates, str):
            sd_cols = [feat_dates]
        elif isinstance(feat_dates, dict):
            sd_cols = list(feat_dates.values())
        else:
            sd_cols = list(feat_dates)

        for sc in sd_cols:
            if sc in out.columns:
                out[sc] = grouped[sc].shift(lag)

    initial_rows = len(df)
    if drop_na_leads:
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    rows_with_nan = int(out[feature_cols].isna().any(axis=1).sum())
    stats = {
        "features_shifted": list(feature_cols),
        "lag": lag,
        "initial_rows": initial_rows,
        "output_rows": len(out),
        "lead_rows_with_nan": rows_with_nan,
    }
    return out, stats


def collapse_episodes(
    df: pd.DataFrame,
    gap: int = 28,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
    raw_event_col: Optional[str] = "_raw_event",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 2: Collapse repeated injury logs into distinct episode starts.
    Any positive event occurring within `gap` days of the athlete's prior positive event
    is a repeat report / continuation of an ongoing episode, and its label is set to 0.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel.
    gap : int
        Washout / separation window in days (default 28; supports 7).
    """
    out = df.sort_values([player_col, date_col]).copy()
    initial_events = int((out[label_col] == 1).sum())

    if raw_event_col is not None and raw_event_col not in out.columns:
        out[raw_event_col] = out[label_col].copy()

    # Track episode onsets per athlete
    new_y = np.zeros(len(out), dtype=int)
    relogs_collapsed = 0

    for _, player_df in out.groupby(player_col, sort=False):
        last_event_day = None
        for idx, row in player_df.iterrows():
            if row[label_col] == 1:
                curr_day = row[date_col]
                if pd.api.types.is_datetime64_any_dtype(pd.Series([curr_day])):
                    # datetime
                    if last_event_day is None:
                        is_new = True
                    else:
                        diff_days = (curr_day - last_event_day).total_seconds() / 86400.0
                        is_new = diff_days > gap
                else:
                    if last_event_day is None:
                        is_new = True
                    else:
                        diff_days = float(curr_day - last_event_day)
                        is_new = diff_days > gap

                if is_new:
                    new_y[out.index.get_loc(idx)] = 1
                    last_event_day = curr_day
                else:
                    new_y[out.index.get_loc(idx)] = 0
                    relogs_collapsed += 1

    out[label_col] = new_y
    final_events = int((out[label_col] == 1).sum())

    stats = {
        "gap": gap,
        "initial_events": initial_events,
        "relogs_collapsed": relogs_collapsed,
        "final_events": final_events,
    }
    return out, stats


def exclude_active_rows(
    df: pd.DataFrame,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
    state_col: Optional[str] = None,
    washout_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 3: Drop observations where the athlete is already injured / unavailable.
    An athlete cannot suffer a *new onset* acute injury if already injured or in washout.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel.
    state_col : str, optional
        Pre-existing indicator of injury/unavailability (e.g. 'prev_out' or 'injury_out').
    washout_days : int, optional
        Number of days following an injury episode to censor as convalescence.
    """
    out = df.sort_values([player_col, date_col]).copy()
    initial_rows = len(out)
    initial_events = int((out[label_col] == 1).sum())

    drop_mask = pd.Series(False, index=out.index)

    if state_col is not None and state_col in out.columns:
        # e.g. already out / injured on previous record
        drop_mask |= out[state_col] == 1
    else:
        # Default: if athlete was injured on immediate previous row (state carryover)
        prev_y = out.groupby(player_col)[label_col].shift(1).fillna(0)
        # Also check raw event if available
        if "_raw_event" in out.columns:
            prev_raw = out.groupby(player_col)["_raw_event"].shift(1).fillna(0)
            drop_mask |= (prev_y == 1) | (prev_raw == 1)
        else:
            drop_mask |= prev_y == 1

    # Apply washout window if specified
    if washout_days is not None and washout_days > 0:
        for _, player_df in out.groupby(player_col, sort=False):
            washout_until = None
            for idx, row in player_df.iterrows():
                curr_day = row[date_col]
                if row[label_col] == 1:
                    # New episode start triggers washout for subsequent days
                    if pd.api.types.is_datetime64_any_dtype(pd.Series([curr_day])):
                        washout_until = curr_day + pd.Timedelta(days=washout_days)
                    else:
                        washout_until = curr_day + washout_days
                elif washout_until is not None:
                    if curr_day <= washout_until:
                        drop_mask.loc[idx] = True
                    else:
                        washout_until = None

    clean_out = out[~drop_mask].reset_index(drop=True)
    rows_dropped = initial_rows - len(clean_out)
    events_dropped = initial_events - int((clean_out[label_col] == 1).sum())

    stats = {
        "initial_rows": initial_rows,
        "rows_dropped": rows_dropped,
        "final_rows": len(clean_out),
        "initial_events": initial_events,
        "events_dropped": events_dropped,
        "final_events": int((clean_out[label_col] == 1).sum()),
    }
    return clean_out, stats


def make_safe_panel(
    df: pd.DataFrame,
    feature_cols: List[str],
    gap: int = 28,
    exclude_active: bool = False,
    lag: int = 1,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
    feat_dates: Optional[Union[Dict[str, str], List[str], str]] = None,
    sameday_feat: Optional[str] = None,
    active_col: Optional[str] = None,
    washout_days: Optional[int] = None,
    assert_checks: bool = True,
    relog_thresh: float = 0.3,
    carryover_thresh: float = 0.3,
) -> Tuple[pd.DataFrame, FixLog]:
    """
    End-to-end leakage fix pipeline.

    1. Shifts features by `lag` (default 1) so predictors strictly precede outcome.
    2. Collapses repeated injury logs within `gap` days into episode onsets.
    3. If `exclude_active` is True, removes rows where player is already injured.
    4. Asserts resulting panel passes leakage detectors (lag, relog, carryover, sameday).
    Logs all row and event modifications.
    """
    log = FixLog(
        initial_rows=len(df),
        initial_events=int((df[label_col] == 1).sum()) if label_col in df.columns else 0,
        features_shifted=list(feature_cols),
        lag_steps=lag,
    )

    # Step 1: Feature shifting
    panel, shift_stats = shift_features(
        df,
        feature_cols=feature_cols,
        lag=lag,
        player_col=player_col,
        date_col=date_col,
        feat_dates=feat_dates,
    )
    log.steps_applied.append(f"shift_features(lag={lag})")

    # Step 2: Collapse repeat logs into episodes
    panel, collapse_stats = collapse_episodes(
        panel,
        gap=gap,
        player_col=player_col,
        date_col=date_col,
        label_col=label_col,
    )
    log.relogs_collapsed = collapse_stats["relogs_collapsed"]
    log.events_after_collapse = collapse_stats["final_events"]
    log.steps_applied.append(f"collapse_episodes(gap={gap})")

    # Step 3: Exclude active rows if requested
    if exclude_active:
        panel, drop_stats = exclude_active_rows(
            panel,
            player_col=player_col,
            date_col=date_col,
            label_col=label_col,
            state_col=active_col,
            washout_days=washout_days,
        )
        log.active_rows_dropped = drop_stats["rows_dropped"]
        log.steps_applied.append("exclude_active_rows")
    else:
        log.active_rows_dropped = 0

    log.final_rows = len(panel)
    log.final_events = int((panel[label_col] == 1).sum())

    # Step 4: Run detectors on the safe panel
    det_results = run_all_checks(
        panel,
        feat_dates=feat_dates,
        sameday_feat=sameday_feat,
        relog_gap=gap,
        relog_thresh=relog_thresh,
        carryover_thresh=carryover_thresh,
        player_col=player_col,
        date_col=date_col,
        label_col=label_col,
    )
    log.detector_results = det_results

    if assert_checks:
        failed: Dict[str, CheckResult] = {}
        # Lag check must not flag
        if "lag" in det_results and det_results["lag"].flag:
            failed["lag"] = det_results["lag"]
        # Relog check must not flag
        if "relog" in det_results and det_results["relog"].flag:
            failed["relog"] = det_results["relog"]
        # Carryover check must not flag
        if "carryover" in det_results and det_results["carryover"].flag:
            failed["carryover"] = det_results["carryover"]
        # Sameday check must not flag
        if "sameday" in det_results and det_results["sameday"].flag:
            failed["sameday"] = det_results["sameday"]

        if failed:
            reasons = "; ".join(f"{k}: {res.reason}" for k, res in failed.items())
            raise LeakageAssertionError(
                f"Safe panel failed leakage validation check(s): {reasons}",
                failed_checks=failed,
            )

    return panel, log
