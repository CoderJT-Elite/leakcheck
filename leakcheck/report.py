# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
report.py: Leakage-safe evaluation reporting for injury prediction models.

Calculates:
1. Model AUC with player-cluster bootstrap 95% confidence intervals (>= 1,000 resamples).
2. Persistence baseline AUC (status on prior observation predicting today).
3. Study-clock baseline AUC (days in study alone predicting today).
4. Excess AUC: Model AUC minus the stronger of persistence or study clock.
5. Noise floor: 95th percentile AUC achievable by pure noise across K candidate models
   simulated through the identical event count (>= 500 repetitions).
6. One-page publication-ready markdown evaluation table.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class EvaluationReport:
    """Structured leakage-safe evaluation report."""
    model_name: str
    n_samples: int
    n_events: int
    n_players: int
    model_auc: float
    model_ci_lower: float
    model_ci_upper: float
    persistence_auc: float
    clock_auc: float
    better_baseline_auc: float
    better_baseline_name: str
    excess_auc: float
    noise_floor_mean: float
    noise_floor_q95: float
    n_models_tried: int
    n_boot: int
    n_noise_reps: int
    beats_baselines: bool
    beats_noise_floor: bool
    markdown_table: str = ""

    def summary(self) -> str:
        return self.markdown_table


def compute_auc(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Compute standard ROC-AUC with validation checks."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(float)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_arr, y_pred_arr))


def persistence_baseline(
    df: pd.DataFrame,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
    persistence_col: Optional[str] = None,
) -> float:
    """
    Compute persistence baseline AUC.
    Tests whether the athlete's state on the previous record predicts today's outcome.
    """
    if len(df) == 0 or label_col not in df.columns:
        return 0.5
    y = df[label_col].values
    if len(np.unique(y)) < 2:
        return 0.5

    if persistence_col is not None and persistence_col in df.columns:
        scores = df[persistence_col].fillna(0).astype(float).values
    else:
        d = df.sort_values([player_col, date_col]).copy()
        scores = d.groupby(player_col)[label_col].shift(1).fillna(0).astype(float).values

    if len(np.unique(scores)) < 2:
        return 0.5
    a = float(roc_auc_score(y, scores))
    return max(a, 1.0 - a)


def clock_baseline(
    df: pd.DataFrame,
    player_col: str = "player",
    date_col: str = "day",
    label_col: str = "y",
) -> float:
    """
    Compute study-clock baseline AUC.
    Tests whether days-in-study alone predicts the outcome (temporal artefact).
    """
    if len(df) == 0 or label_col not in df.columns:
        return 0.5
    y = df[label_col].values
    if len(np.unique(y)) < 2:
        return 0.5

    d = df.copy()
    day_series = d[date_col]
    if pd.api.types.is_datetime64_any_dtype(day_series):
        min_date = d.groupby(player_col)[date_col].transform("min")
        clock = (day_series - min_date).dt.total_seconds() / 86400.0
    else:
        min_day = d.groupby(player_col)[date_col].transform("min")
        clock = (day_series - min_day).astype(float)

    if len(np.unique(clock)) < 2:
        return 0.5
    a = float(roc_auc_score(y, clock))
    return max(a, 1.0 - a)


def compute_excess_auc(model_auc: float, persistence_auc: float, clock_auc: float) -> Tuple[float, float, str]:
    """
    Compute excess AUC = model_auc - max(persistence_auc, clock_auc).
    Returns (excess_auc, better_baseline_auc, better_baseline_name).
    """
    if persistence_auc >= clock_auc:
        better_auc = persistence_auc
        better_name = "Persistence"
    else:
        better_auc = clock_auc
        better_name = "Study Clock"
    excess = model_auc - better_auc
    return float(excess), float(better_auc), better_name


def simulate_noise_floor(
    n_events: int,
    n_samples: int,
    n_models_tried: int = 1,
    n_reps: int = 500,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Simulate the noise floor: maximum AUC achievable by pure random noise
    when selecting the best among `n_models_tried` candidate models on `n_events` events.

    Returns
    -------
    mean_best_auc : float
    q95_best_auc : float (95th percentile noise ceiling)
    best_aucs : np.ndarray (shape (n_reps,))
    """
    if n_events <= 0 or n_events >= n_samples:
        return 0.5, 0.5, np.full(n_reps, 0.5)

    rng = np.random.default_rng(seed)
    n1 = int(n_events)
    n0 = int(n_samples - n1)

    best_aucs = np.zeros(n_reps)
    for rep in range(n_reps):
        # Generate random uniform rankings for n_models_tried
        R = rng.random(size=(n_samples, n_models_tried))
        ranks = stats.rankdata(R, axis=0)
        # First n1 rows are positive
        pos_ranks = ranks[:n1, :]
        auc_candidates = (np.sum(pos_ranks, axis=0) - (n1 * (n1 + 1.0) / 2.0)) / (n1 * n0)
        # For non-directional models, take maximum discrimination
        auc_directional = np.maximum(auc_candidates, 1.0 - auc_candidates)
        best_aucs[rep] = np.max(auc_directional)

    mean_best = float(np.mean(best_aucs))
    q95_best = float(np.percentile(best_aucs, 95))
    return mean_best, q95_best, best_aucs


def player_cluster_bootstrap(
    df: pd.DataFrame,
    y_true: Union[str, np.ndarray, pd.Series],
    y_pred: Union[str, np.ndarray, pd.Series],
    player_col: str = "player",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Player-cluster bootstrap (resampling athletes/players with replacement).
    Preserves intra-athlete temporal correlation structure.

    Returns
    -------
    point_auc : float
    ci_lower : float (e.g. 2.5th percentile for 95% CI)
    ci_upper : float (e.g. 97.5th percentile for 95% CI)
    boot_aucs : np.ndarray
    """
    if isinstance(y_true, str):
        y_arr = df[y_true].values.astype(int)
    else:
        y_arr = np.asarray(y_true).astype(int)

    if isinstance(y_pred, str):
        pred_arr = df[y_pred].values.astype(float)
    else:
        pred_arr = np.asarray(y_pred).astype(float)

    point_auc = compute_auc(y_arr, pred_arr)

    players = np.unique(df[player_col].values)
    n_players = len(players)
    if n_players <= 1:
        # Cannot cluster-bootstrap a single player
        return point_auc, point_auc, point_auc, np.array([point_auc])

    # Index map: player -> row indices
    player_indices = {p: np.where(df[player_col].values == p)[0] for p in players}

    rng = np.random.default_rng(seed)
    boot_aucs = []
    max_attempts = n_boot * 5
    attempts = 0

    while len(boot_aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_players = rng.choice(players, size=n_players, replace=True)
        sampled_idx = np.concatenate([player_indices[p] for p in sampled_players])
        y_sample = y_arr[sampled_idx]
        if np.sum(y_sample == 1) > 0 and np.sum(y_sample == 0) > 0:
            pred_sample = pred_arr[sampled_idx]
            boot_aucs.append(float(roc_auc_score(y_sample, pred_sample)))

    if len(boot_aucs) == 0:
        return point_auc, point_auc, point_auc, np.array([point_auc])

    boot_aucs_arr = np.array(boot_aucs)
    ci_lower = float(np.percentile(boot_aucs_arr, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_aucs_arr, 100.0 * (1.0 - alpha / 2.0)))

    return point_auc, ci_lower, ci_upper, boot_aucs_arr


def evaluate_predictions(
    df: pd.DataFrame,
    y_pred: Union[str, np.ndarray, pd.Series],
    y_true: Union[str, np.ndarray, pd.Series] = "y",
    model_name: str = "Proposed Model",
    n_models_tried: int = 1,
    player_col: str = "player",
    date_col: str = "day",
    persistence_col: Optional[str] = None,
    n_boot: int = 1000,
    n_noise_reps: int = 500,
    seed: int = 42,
) -> EvaluationReport:
    """
    Produce a complete, leakage-safe evaluation report with baseline benchmarking,
    excess AUC, player-cluster bootstrap CI, and noise floor.
    """
    if isinstance(y_true, str):
        y_arr = df[y_true].values.astype(int)
        label_col = y_true
    else:
        y_arr = np.asarray(y_true).astype(int)
        label_col = "_y_eval"
        df = df.copy()
        df[label_col] = y_arr

    if isinstance(y_pred, str):
        pred_arr = df[y_pred].values.astype(float)
    else:
        pred_arr = np.asarray(y_pred).astype(float)

    n_samples = len(df)
    n_events = int(np.sum(y_arr == 1))
    n_players = int(df[player_col].nunique())

    # 1. Point AUC & cluster bootstrap CI
    model_auc, ci_lower, ci_upper, _ = player_cluster_bootstrap(
        df=df,
        y_true=y_arr,
        y_pred=pred_arr,
        player_col=player_col,
        n_boot=n_boot,
        seed=seed,
    )

    # 2. Persistence baseline
    p_auc = persistence_baseline(
        df=df,
        player_col=player_col,
        date_col=date_col,
        label_col=label_col,
        persistence_col=persistence_col,
    )

    # 3. Clock baseline
    c_auc = clock_baseline(
        df=df,
        player_col=player_col,
        date_col=date_col,
        label_col=label_col,
    )

    # 4. Excess AUC
    excess, better_base_auc, better_base_name = compute_excess_auc(model_auc, p_auc, c_auc)

    # 5. Noise floor
    mean_noise, q95_noise, _ = simulate_noise_floor(
        n_events=n_events,
        n_samples=n_samples,
        n_models_tried=n_models_tried,
        n_reps=n_noise_reps,
        seed=seed,
    )

    beats_baselines = excess > 0.0
    beats_noise = model_auc > q95_noise

    # Build markdown table
    md_lines = [
        f"### Leakage-Safe Evaluation: {model_name}",
        f"- **Cohort**: {n_samples:,} observations across {n_players:,} athletes ({n_events:,} acute events; incidence: {n_events / n_samples * 100:.2f}%)",
        f"- **Bootstrap Resamples**: {n_boot:,} athlete clusters | **Noise Floor Simulations**: {n_noise_reps:,} reps (K={n_models_tried} candidate models)",
        "",
        "| Evaluation Metric | Value | 95% CI / Benchmark | Interpretation |",
        "|---|---|---|---|",
        f"| **Model ROC-AUC** | **{model_auc:.3f}** | [{ci_lower:.3f}, {ci_upper:.3f}] | Out-of-fold discrimination |",
        f"| Baseline: Persistence | {p_auc:.3f} | Prior-observation status | Prior active state predicting today |",
        f"| Baseline: Study Clock | {c_auc:.3f} | Elapsed days in study | Reporting decay / study calendar artefact |",
        f"| Stronger Baseline | {better_base_auc:.3f} | {better_base_name} | Maximum uninformative reference |",
        f"| **EXCESS AUC** | **{excess:+.3f}** | Model minus {better_base_name} | {'Genuinely informative predictive signal' if beats_baselines else 'Artifactual / below simple baseline'} |",
        f"| Noise Floor (K={n_models_tried} models) | {mean_noise:.3f} (mean) | **{q95_noise:.3f} (95th %ile)** | Ceiling achievable by pure noise on {n_events} events |",
        "",
        f"> **Summary Verdict**: {'PASS — Model outperforms persistence/study-clock baselines and noise ceiling.' if (beats_baselines and beats_noise) else 'CAUTION — Model does not meaningfully exceed non-predictive baselines or random noise floor.'}",
    ]
    markdown_table = "\n".join(md_lines)

    return EvaluationReport(
        model_name=model_name,
        n_samples=n_samples,
        n_events=n_events,
        n_players=n_players,
        model_auc=model_auc,
        model_ci_lower=ci_lower,
        model_ci_upper=ci_upper,
        persistence_auc=p_auc,
        clock_auc=c_auc,
        better_baseline_auc=better_base_auc,
        better_baseline_name=better_base_name,
        excess_auc=excess,
        noise_floor_mean=mean_noise,
        noise_floor_q95=q95_noise,
        n_models_tried=n_models_tried,
        n_boot=n_boot,
        n_noise_reps=n_noise_reps,
        beats_baselines=beats_baselines,
        beats_noise_floor=beats_noise,
        markdown_table=markdown_table,
    )
