"""
Audio Memory Evaluation Package.

A framework for evaluating audio-based memory retrieval using the open-source mem0 library.

Modules:
    config: Configuration settings (ASR, LLM, embedder, memory settings)
    prompts: Prompt templates for answer generation and LLM judge
    metrics: Evaluation metrics (EM, F1, BLEU, LLM Judge) and aggregation
    evaluator: Main evaluation pipeline
    generate_scores: Score aggregation, comparison, and export

Usage:
    # Run evaluation
    python -m audio_eval.evaluator --num_samples 100

    # Compare results
    python -m audio_eval.generate_scores results/audio_eval/*.json

Example:
    from audio_eval import run_evaluation
    results = run_evaluation(num_samples=10, experiment_name="test")
"""

from .evaluator import run_evaluation, evaluate_sample, load_evaluation_dataset
from .metrics import (
    exact_match,
    f1_score,
    bleu_score,
    llm_judge,
    compute_metrics,
    aggregate_results,
)
from .config import get_mem0_config

__all__ = [
    # Evaluation
    "run_evaluation",
    "evaluate_sample",
    "load_evaluation_dataset",
    # Metrics
    "exact_match",
    "f1_score",
    "bleu_score",
    "llm_judge",
    "compute_metrics",
    "aggregate_results",
    # Config
    "get_mem0_config",
]
