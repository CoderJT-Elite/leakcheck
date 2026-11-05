# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
checks.py: Five automated leakage detectors for sports injury prediction studies,
copied and cleaned up from leakage_ladder/audit_tool.py.

Detectors:
1. check_lag: Feature source timestamp >= label/prediction timestamp.
2. check_relog: High proportion of events occurring within `gap` days of athlete's prior event.
3. check_carryover: High proportion of events whose immediately prior observation is already injured.
4. check_clock: Days in study alone predicts injury with AUC > threshold (temporal confounding).
5. check_sameday: Current-day feature separates label substantially better than its 1-day lag.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class CheckResult:
    """Structured result from a leakage detector."""
    name: str
    flag: bool
    metric_value: float
    threshold: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.flag

    def __str__(self) -> str:
        status = "FLAGGED" if self.flag else "CLEAN"
        return f"[{status}] {self.name}: {self.reason} (value={self.metric_value:.4f}, thresh={self.threshold:.4f})"


def _ensure_days(series: pd.Series) -> pd.Series:
    """Helper to convert date series to numeric day if datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.min()).dt.days
    return series


def check_lag(
    df: pd.DataFrame,
    feat_dates: Union[Dict[str, str], List[str], str],
    player_col: str = "player",
    date_col: str = "day",
) -> CheckResult:
    """
    Check 1: Lag detector.
    Flags if any feature's recorded source date is on or after the observation date.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel.
    feat_dates : dict, list, or str
        Mapping of feature_name -> source_date_column, or list/str of source date columns.
    player_col : str
        Athlete identifier column.
    date_col : str
        Prediction / observation date column.
    """
    if len(df) == 0:
        return CheckResult(
            name="lag",
            flag=False,
            metric_value=0.0,
            threshold=0.0,
            reason="Panel is empty; no lag violations possible.",
        )

    if isinstance(feat_dates, str):
        date_cols = [feat_dates]
    elif isinstance(feat_dates, dict):
        date_cols = list(feat_dates.values())
    else:
        date_cols = list(feat_dates)

    total_rows = len(df)
    violating_rows = 0
    max_offsets = []

    for c in date_cols:
        if c not in df.columns:
            raise KeyError(f"Source date column '{c}' not found in DataFrame.")
        # offset = src - day (positive or zero means leak)
        src = df[c]
        day = df[date_col]
        if pd.api.types.is_datetime64_any_dtype(src) and pd.api.types.is_datetime64_any_dtype(day):
            diff = (src - day).dt.total_seconds() / 86400.0
        else:
            diff = (src - day).astype(float)
        violation_mask = diff >= 0
        v_count = int(violation_mask.sum())
        violating_rows = max(violating_rows, v_count)
        if v_count > 0:
            max_offsets.append(float(diff.max()))

    violation_rate = violating_rows / total_rows if total_rows > 0 else 0.0
    flag = violating_rows > 0
    max_offset_val = max(max_offsets) if max_offsets else 0.0

    if flag:
        reason = (
            f"Feature source date on or after prediction day on {violating_rows:,} rows "
            f"({violation_rate:.1%}; max offset: +{max_offset_val:.1f} days)."
        )
    else:
        reason = "All feature source dates precede prediction day strictly."

    return CheckResult(
        name="lag",
        flag=flag,
        metric_value=violation_rate,
        threshold=0.0,
        reason=reason,
        details={
            "violating_rows": violating_rows,
            "total_rows": total_rows,
            "max_offset_days": max_offset_val,
            "checked_columns": date_cols,
        },
    )


def check_relog(
    df: pd.DataFrame,
    gap: int = 7,
    thresh: float = 0.3,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> CheckResult:
    """
    Check 2: Relog detector.
    Flags if more than `thresh` of positive events occur within `gap` days
    of the same athlete's prior positive event.
    """
    if len(df) == 0 or label_col not in df.columns:
        return CheckResult("relog", False, 0.0, thresh, "Panel is empty or label column missing.")

    positives = df[df[label_col] == 1].sort_values([player_col, date_col]).copy()
    if len(positives) < 2:
        return CheckResult(
            name="relog",
            flag=False,
            metric_value=0.0,
            threshold=thresh,
            reason=f"Fewer than 2 positive events ({len(positives)}); cannot compute relog rate.",
            details={"positive_count": len(positives)},
        )

    day_series = positives[date_col]
    if pd.api.types.is_datetime64_any_dtype(day_series):
        diffs = positives.groupby(player_col)[date_col].diff().dt.total_seconds() / 86400.0
    else:
        diffs = positives.groupby(player_col)[date_col].diff().astype(float)

    valid_diffs = diffs.dropna()
    if len(valid_diffs) == 0:
        metric_value = 0.0
        relog_count = 0
    else:
        relog_count = int((valid_diffs <= gap).sum())
        metric_value = float((valid_diffs <= gap).mean())

    flag = metric_value > thresh
    if flag:
        reason = (
            f"{relog_count} of {len(valid_diffs)} consecutive events ({metric_value:.1%}) "
            f"occur within {gap} days of a prior event (threshold: {thresh:.1%}), indicating repeated logging."
        )
    else:
        reason = (
            f"Consecutive event relog rate is {metric_value:.1%} ({relog_count}/{len(valid_diffs)}) "
            f"within {gap} days, below threshold {thresh:.1%}."
        )

    return CheckResult(
        name="relog",
        flag=flag,
        metric_value=metric_value,
        threshold=thresh,
        reason=reason,
        details={
            "relog_count": relog_count,
            "total_consecutive_pairs": len(valid_diffs),
            "total_positives": len(positives),
            "gap": gap,
        },
    )


def check_carryover(
    df: pd.DataFrame,
    thresh: float = 0.3,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> CheckResult:
    """
    Check 3: Carryover detector.
    Flags if more than `thresh` of positive events had y==1 on the athlete's immediately
    preceding observation row (persistent unavailable state).
    """
    if len(df) == 0 or label_col not in df.columns:
        return CheckResult("carryover", False, 0.0, thresh, "Panel is empty or label column missing.")

    d = df.sort_values([player_col, date_col]).copy()
    d["_prev_y"] = d.groupby(player_col)[label_col].shift(1)
    positives = d[d[label_col] == 1]

    if len(positives) == 0:
        return CheckResult(
            name="carryover",
            flag=False,
            metric_value=0.0,
            threshold=thresh,
            reason="No positive events in panel.",
            details={"positive_count": 0},
        )

    carryover_count = int(positives["_prev_y"].fillna(0).eq(1).sum())
    carryover_rate = carryover_count / len(positives)
    flag = carryover_rate > thresh

    if flag:
        reason = (
            f"{carryover_count} of {len(positives)} positive events ({carryover_rate:.1%}) "
            f"had y==1 on the immediately preceding observation (threshold: {thresh:.1%}), "
            "indicating state carryover."
        )
    else:
        reason = (
            f"Carryover rate is {carryover_rate:.1%} ({carryover_count}/{len(positives)}), "
            f"below threshold {thresh:.1%}."
        )

    return CheckResult(
        name="carryover",
        flag=flag,
        metric_value=carryover_rate,
        threshold=thresh,
        reason=reason,
        details={
            "carryover_count": carryover_count,
            "total_positives": len(positives),
        },
    )


def check_clock(
    df: pd.DataFrame,
    thresh: float = 0.65,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> CheckResult:
    """
    Check 4: Study clock detector.
    Flags if days-in-study alone separates the label with AUC > `thresh` (or < 1 - thresh).
    """
    if len(df) == 0 or label_col not in df.columns:
        return CheckResult("clock", False, 0.5, thresh, "Panel is empty or label column missing.")

    d = df.copy()
    if d[label_col].nunique() < 2:
        return CheckResult(
            name="clock",
            flag=False,
            metric_value=0.5,
            threshold=thresh,
            reason="Fewer than 2 classes present in label; AUC undefined.",
            details={"unique_classes": d[label_col].unique().tolist()},
        )

    day_series = d[date_col]
    if pd.api.types.is_datetime64_any_dtype(day_series):
        min_date = d.groupby(player_col)[date_col].transform("min")
        d["_clock"] = (day_series - min_date).dt.total_seconds() / 86400.0
    else:
        min_day = d.groupby(player_col)[date_col].transform("min")
        d["_clock"] = (day_series - min_day).astype(float)

    raw_auc = float(roc_auc_score(d[label_col], d["_clock"]))
    metric_value = max(raw_auc, 1.0 - raw_auc)
    flag = metric_value > thresh

    if flag:
        direction = "late" if raw_auc > 0.5 else "early"
        reason = (
            f"Study clock alone yields AUC {metric_value:.3f} (raw {raw_auc:.3f}, events concentrate {direction}; "
            f"threshold: {thresh:.3f}), indicating temporal confounding."
        )
    else:
        reason = (
            f"Study clock AUC is {metric_value:.3f} (raw {raw_auc:.3f}), below threshold {thresh:.3f}."
        )

    return CheckResult(
        name="clock",
        flag=flag,
        metric_value=metric_value,
        threshold=thresh,
        reason=reason,
        details={
            "raw_auc": raw_auc,
            "effective_auc": metric_value,
        },
    )


def check_sameday(
    df: pd.DataFrame,
    feat: str,
    thresh: float = 0.65,
    margin: float = 0.10,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> CheckResult:
    """
    Check 5: Same-day feature detector.
    Flags if feature `feat` separates today's label with AUC > thresh AND
    its current-day AUC exceeds its 1-step lagged AUC by more than `margin`.
    """
    if len(df) == 0 or feat not in df.columns or label_col not in df.columns:
        return CheckResult(
            name="sameday",
            flag=False,
            metric_value=0.0,
            threshold=margin,
            reason=f"Column '{feat}' or '{label_col}' missing or panel empty.",
        )

    d = df.sort_values([player_col, date_col]).copy()
    d["_lagf"] = d.groupby(player_col)[feat].shift(1)
    d = d.dropna(subset=[feat, "_lagf", label_col])

    if d[label_col].nunique() < 2 or len(d) == 0:
        return CheckResult(
            name="sameday",
            flag=False,
            metric_value=0.0,
            threshold=margin,
            reason="Fewer than 2 classes present after dropping lag NaNs; AUC undefined.",
            details={"valid_rows": len(d)},
        )

    a0 = float(roc_auc_score(d[label_col], d[feat]))
    a1 = float(roc_auc_score(d[label_col], d["_lagf"]))
    m0 = max(a0, 1.0 - a0)
    m1 = max(a1, 1.0 - a1)
    delta = m0 - m1

    flag = (m0 > thresh) and (delta > margin)
    if flag:
        reason = (
            f"Feature '{feat}' same-day AUC is {m0:.3f} vs lag-1 AUC {m1:.3f} "
            f"(excess delta {delta:.3f} > {margin:.3f}, AUC > {thresh:.3f}), indicating same-day leakage."
        )
    else:
        reason = (
            f"Feature '{feat}' same-day AUC {m0:.3f} vs lag-1 AUC {m1:.3f} "
            f"(delta {delta:.3f} <= margin {margin:.3f} or AUC <= {thresh:.3f})."
        )

    return CheckResult(
        name="sameday",
        flag=flag,
        metric_value=delta,
        threshold=margin,
        reason=reason,
        details={
            "feature": feat,
            "same_day_auc": m0,
            "lag1_auc": m1,
            "raw_same_day_auc": a0,
            "raw_lag1_auc": a1,
            "delta": delta,
            "abs_thresh": thresh,
        },
    )


def run_all_checks(
    df: pd.DataFrame,
    feat_dates: Optional[Union[Dict[str, str], List[str], str]] = None,
    sameday_feat: Optional[str] = None,
    relog_gap: int = 7,
    relog_thresh: float = 0.3,
    carryover_thresh: float = 0.3,
    clock_thresh: float = 0.65,
    sameday_thresh: float = 0.65,
    sameday_margin: float = 0.10,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> Dict[str, CheckResult]:
    """
    Run all five leakage checks on a panel.
    Returns a dict mapping check_name -> CheckResult.
    """
    results: Dict[str, CheckResult] = {}

    # 1. Lag
    if feat_dates is not None:
        results["lag"] = check_lag(df, feat_dates, player_col=player_col, date_col=date_col)

    # 2. Relog
    results["relog"] = check_relog(
        df, gap=relog_gap, thresh=relog_thresh, player_col=player_col, date_col=date_col, label_col=label_col
    )

    # 3. Carryover
    results["carryover"] = check_carryover(
        df, thresh=carryover_thresh, player_col=player_col, date_col=date_col, label_col=label_col
    )

    # 4. Clock
    results["clock"] = check_clock(
        df, thresh=clock_thresh, player_col=player_col, date_col=date_col, label_col=label_col
    )

    # 5. Sameday
    if sameday_feat is not None and sameday_feat in df.columns:
        results["sameday"] = check_sameday(
            df,
            feat=sameday_feat,
            thresh=sameday_thresh,
            margin=sameday_margin,
            player_col=player_col,
            date_col=date_col,
            label_col=label_col,
        )

    return results


def synth_planted_leak(
    kind: str,
    seed: int = 0,
    n_players: int = 60,
    n_days: int = 200,
) -> pd.DataFrame:
    """
    Synthetic panel generator with planted leakage modes,
    copied from leakage_ladder/audit_tool.py.

    Kinds:
    - 'clean': No leakage.
    - 'lag': Feature source date equals prediction day (violation).
    - 'relog': Burst of repeated logs for positive events.
    - 'carryover': Injury status persists across multiple consecutive days.
    - 'clock': Events concentrate early in the player's study timeline.
    - 'sameday': Feature recorded on event day separates label strongly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        y = (rng.random(n_days) < 0.02).astype(int)
        base = rng.normal(size=n_days)

        if kind == "relog":
            for t in np.where(y == 1)[0]:
                y[t : t + 4] = 1

        if kind == "carryover":
            for t in np.where(y == 1)[0]:
                y[t : t + 10] = 1

        feat = base.copy()
        if kind == "sameday":
            feat = feat + 3.0 * y

        day = np.arange(n_days)
        if kind == "clock":
            y = ((rng.random(n_days) < 0.08) & (day < 30)).astype(int)

        src = day - 1
        if kind == "lag":
            src = day

        rows.append(pd.DataFrame({"player": p, "day": day, "y": y, "feat": feat, "src": src}))

    return pd.concat(rows, ignore_index=True)
