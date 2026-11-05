# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
leakcheck: A package for detecting and fixing data leakage in injury-prediction and sports medicine models.
"""

from .checks import (
    CheckResult,
    check_lag,
    check_relog,
    check_carryover,
    check_clock,
    check_sameday,
    run_all_checks,
    synth_planted_leak,
)

from .fix import (
    FixLog,
    shift_features,
    collapse_episodes,
    exclude_active_rows,
    make_safe_panel,
    LeakageAssertionError,
)

from .report import (
    EvaluationReport,
    compute_auc,
    persistence_baseline,
    clock_baseline,
    compute_excess_auc,
    simulate_noise_floor,
    player_cluster_bootstrap,
    evaluate_predictions,
)

__all__ = [
    "CheckResult",
    "check_lag",
    "check_relog",
    "check_carryover",
    "check_clock",
    "check_sameday",
    "run_all_checks",
    "synth_planted_leak",
    "FixLog",
    "shift_features",
    "collapse_episodes",
    "exclude_active_rows",
    "make_safe_panel",
    "LeakageAssertionError",
    "EvaluationReport",
    "compute_auc",
    "persistence_baseline",
    "clock_baseline",
    "compute_excess_auc",
    "simulate_noise_floor",
    "player_cluster_bootstrap",
    "evaluate_predictions",
]
