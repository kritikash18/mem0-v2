"""
Evaluation Metrics for Cognee Evaluation.

Reuses the same metric implementations from audio_eval for fair comparison.
Imports all metrics from audio_eval.metrics to ensure identical scoring.
"""

import sys
import os

# Add parent directory to path so we can import from audio_eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_eval.metrics import (
    normalize_text,
    exact_match,
    f1_score,
    bleu_score,
    llm_judge,
    compute_metrics,
    aggregate_results,
    compute_score_distribution,
)

__all__ = [
    "normalize_text",
    "exact_match",
    "f1_score",
    "bleu_score",
    "llm_judge",
    "compute_metrics",
    "aggregate_results",
    "compute_score_distribution",
]
