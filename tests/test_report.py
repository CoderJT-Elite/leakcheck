# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
Tests for leakage_tool.leakcheck.report:
- Requirement 2d: Verifies AUC and cluster bootstrap match sklearn on a hand-checkable tiny example.
- Verifies persistence baseline, study-clock baseline, and excess AUC arithmetic.
- Verifies noise floor simulation properties and markdown report generation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from leakage_tool.leakcheck.report import (
        compute_auc,
        persistence_baseline,
        clock_baseline,
        compute_excess_auc,
        simulate_noise_floor,
        player_cluster_bootstrap,
        evaluate_predictions,
        EvaluationReport,
    )
except ModuleNotFoundError:
    from leakcheck.report import (
        compute_auc,
        persistence_baseline,
        clock_baseline,
        compute_excess_auc,
        simulate_noise_floor,
        player_cluster_bootstrap,
        evaluate_predictions,
        EvaluationReport,
    )


def test_hand_checkable_tiny_example_matches_sklearn():
    """
    Requirement 2d: AUC matches sklearn.metrics.roc_auc_score on a hand-checkable tiny example.

    Hand calculation:
    y_true: [0, 0, 1, 1] (two negatives, two positives: 4 pairs total)
    y_pred: [0.1, 0.4, 0.35, 0.8]
    Pairs (pos, neg):
      pos=0.35 vs neg=0.1: 0.35 > 0.1 -> 1.0
      pos=0.35 vs neg=0.4: 0.35 < 0.4 -> 0.0
      pos=0.80 vs neg=0.1: 0.80 > 0.1 -> 1.0
      pos=0.80 vs neg=0.4: 0.80 > 0.4 -> 1.0
    Sum = 3.0 / 4.0 = 0.7500 exactly.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    hand_calculated_auc = 0.75
    sklearn_auc = roc_auc_score(y_true, y_pred)
    tool_auc = compute_auc(y_true, y_pred)

    assert abs(tool_auc - hand_calculated_auc) < 1e-6
    assert abs(tool_auc - sklearn_auc) < 1e-6


def test_player_cluster_bootstrap_on_hand_checkable_panel():
    """Requirement 2d: Verifies player-cluster bootstrap point estimate and CI behavior."""
    # 4 players, 2 observations each
    df = pd.DataFrame({
        "player": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
        "day": [1, 2, 1, 2, 1, 2, 1, 2],
        "y": [0, 0, 0, 1, 0, 1, 1, 1],
        "pred": [0.1, 0.2, 0.2, 0.7, 0.3, 0.8, 0.6, 0.9],
    })

    sklearn_point = roc_auc_score(df["y"], df["pred"])
    point_auc, ci_lo, ci_hi, boot_dist = player_cluster_bootstrap(
        df=df,
        y_true="y",
        y_pred="pred",
        player_col="player",
        n_boot=1000,
        seed=42,
    )

    assert abs(point_auc - sklearn_point) < 1e-6
    assert 0.0 <= ci_lo <= point_auc <= ci_hi <= 1.0
    assert len(boot_dist) == 1000


def test_persistence_and_clock_baselines():
    """Verifies persistence baseline and study clock baseline arithmetic."""
    # Hand-checkable panel: Player 0 has an injury at t=2 and t=3 (persisting state)
    df = pd.DataFrame({
        "player": [0, 0, 0, 1, 1, 1],
        "day": [1, 2, 3, 1, 2, 3],
        "y": [0, 1, 1, 0, 0, 0],
    })

    # Shifted label per player:
    # Player 0: [0, 0, 1]
    # Player 1: [0, 0, 0]
    # Combined prev_y: [0, 0, 1, 0, 0, 0]
    # Positives are indices 1 and 2 (y=1). Negatives are 0, 3, 4, 5 (y=0).
    # Scores for positives: [0, 1]
    # Scores for negatives: [0, 0, 0, 0]
    p_auc = persistence_baseline(df, player_col="player", date_col="day", label_col="y")
    assert p_auc > 0.5, "Persistence should exceed chance on continuing state"

    # Study clock
    c_auc = clock_baseline(df, player_col="player", date_col="day", label_col="y")
    assert 0.5 <= c_auc <= 1.0

    # Excess AUC
    model_auc = 0.85
    excess, better_auc, better_name = compute_excess_auc(model_auc, p_auc, c_auc)
    assert abs(excess - (model_auc - max(p_auc, c_auc))) < 1e-6
    assert better_auc == max(p_auc, c_auc)


def test_simulate_noise_floor():
    """Verifies noise floor simulation outputs and statistical monotonicity."""
    n_events = 20
    n_samples = 1000
    mean_1, q95_1, _ = simulate_noise_floor(n_events, n_samples, n_models_tried=1, n_reps=500, seed=42)
    mean_10, q95_10, _ = simulate_noise_floor(n_events, n_samples, n_models_tried=10, n_reps=500, seed=42)

    # Pure noise for K=1 should have mean near 0.5
    assert 0.45 <= mean_1 <= 0.60
    # 95th percentile must be >= mean
    assert q95_1 >= mean_1
    # Selecting the best of 10 random models must yield a higher noise floor than 1 model
    assert mean_10 > mean_1
    assert q95_10 > q95_1


def test_evaluate_predictions_markdown_output():
    """Verifies evaluate_predictions returns complete EvaluationReport and valid markdown."""
    df = pd.DataFrame({
        "player": ["P1"] * 50 + ["P2"] * 50,
        "day": list(range(50)) + list(range(50)),
        "y": [1 if i in [5, 10, 15] else 0 for i in range(100)],
    })
    rng = np.random.default_rng(123)
    df["pred"] = rng.random(100) + 0.5 * df["y"]

    report = evaluate_predictions(
        df=df,
        y_pred="pred",
        y_true="y",
        model_name="Unit Test Model",
        n_models_tried=3,
        n_boot=1000,
        n_noise_reps=500,
        seed=42,
    )

    assert isinstance(report, EvaluationReport)
    assert report.n_samples == 100
    assert report.n_events == 3
    assert report.n_players == 2
    assert report.model_auc >= 0.5

    md = report.summary()
    assert "### Leakage-Safe Evaluation: Unit Test Model" in md
    assert "| **Model ROC-AUC** |" in md
    assert "| Baseline: Persistence |" in md
    assert "| Baseline: Study Clock |" in md
    assert "| **EXCESS AUC** |" in md
    assert "| Noise Floor (K=3 models) |" in md
