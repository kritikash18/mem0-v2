"""
Cognee Evaluation Package.

Evaluation framework for testing Cognee (a competing AI memory framework) using
the same Spoken SQuAD dataset and metrics as the mem0 audio evaluation.

This enables direct comparison between mem0 and Cognee on audio-based
memory retrieval tasks.

Pipeline:
    Audio → ASR Transcription → Cognee add() → Cognee cognify() → Cognee search() → Answer Generation → Evaluation

Usage:
    # Run evaluation
    python -m cognee_eval.evaluator --num_samples 10

    # Compare with mem0 results
    python -m cognee_eval.generate_scores results/cognee_eval/*_final.json

Example:
    from cognee_eval import run_evaluation
    results = run_evaluation(num_samples=10, experiment_name="cognee_test")
"""

from .evaluator import run_evaluation
from .config import get_cognee_config
