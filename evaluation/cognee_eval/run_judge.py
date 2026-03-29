"""
Run (or re-run) batched LLM judge scoring on an existing results JSON file.

Use this when you evaluated with --skip-judge and want to add LLM scores
later in one go, or when you want to re-score with a different judge model.

Usage:
    # Score with default judge (gpt-4o-mini, batch size 10)
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json

    # Use a different judge model
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json --judge-model gpt-4o

    # Use a local Ollama judge
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json \\
        --judge-provider ollama --judge-model qwen2.5

    # Larger batch size to save even more API calls
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json --batch-size 20

    # Only re-score entries that don't already have an llm score
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json --fill-missing

    # Write output to a new file instead of overwriting
    python -m cognee_eval.run_judge results/cognee_eval/my_run_final.json \\
        -o results/cognee_eval/my_run_judged.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import config
from ..audio_eval.metrics import aggregate_results, batch_llm_judge, compute_score_distribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_client(provider: Optional[str], model: Optional[str]) -> tuple:
    """Return (client, model_name) for the requested provider."""
    provider = provider or config.LLM_JUDGE_PROVIDER or "openai"
    model = model or config.LLM_JUDGE_MODEL or "gpt-4o-mini"

    if provider == "ollama":
        client = OpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
    else:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    return client, model


def run_judge(
    input_path: str,
    output_path: Optional[str] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    batch_size: int = 10,
    fill_missing: bool = False,
):
    """
    Load results, run batched LLM judge, write updated file.

    Args:
        input_path: Path to an existing results JSON file
        output_path: Where to write updated results (defaults to input_path)
        judge_provider: LLM provider for judging (default: from config)
        judge_model: Model name for judging (default: gpt-4o-mini)
        batch_size: Entries per LLM call
        fill_missing: If True, only score entries that have llm=None; skip the rest
    """
    p = Path(input_path)
    if not p.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    with open(p) as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = data.get("results", [])
    if not results:
        logger.warning("No results found in file — nothing to judge.")
        return

    client, model = _build_client(judge_provider, judge_model)
    logger.info(f"Judge: {judge_provider or 'openai'}/{model}, batch_size={batch_size}")

    # Decide which entries to score
    if fill_missing:
        to_score = [r for r in results if r.get("llm") is None]
        logger.info(f"Scoring {len(to_score)} entries that are missing llm score")
    else:
        to_score = results
        logger.info(f"Re-scoring all {len(to_score)} entries")

    if not to_score:
        logger.info("All entries already have llm scores — nothing to do. Use without --fill-missing to re-score.")
        return

    # Build sample list for batch judge (keep original list index for writing back)
    samples = [
        {
            "question": r.get("question", ""),
            "ground_truth": r.get("ground_truth", ""),
            "prediction": r.get("prediction", ""),
        }
        for r in to_score
    ]

    labels = batch_llm_judge(
        samples=samples,
        model=model,
        client=client,
        batch_size=batch_size,
    )

    # Write labels back into the original results list
    for entry, label in zip(to_score, labels):
        entry["llm"] = label

    # Recompute summary statistics
    data["summary"] = aggregate_results(results)
    data["distributions"] = {
        "f1": compute_score_distribution(results, "f1"),
        "bleu": compute_score_distribution(results, "bleu"),
    }
    if "config" in data:
        data["config"]["judge"] = f"{judge_provider or 'openai'}/{model}"
        data["config"]["judge_batch_size"] = batch_size

    out = Path(output_path or input_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Judged results saved to: {out}")
    _print_summary(data["summary"], len(results))


def _print_summary(summary: Dict[str, Any], n: int):
    print("\n" + "=" * 55)
    print(f"JUDGE SUMMARY  ({n} samples)")
    print("=" * 55)
    for metric in ["em", "f1", "bleu", "llm"]:
        mean = summary.get(f"{metric}_mean")
        if mean is None:
            continue
        print(f"  {metric.upper():<6} mean={mean:.4f}")
    if "em_pct" in summary:
        print(f"\n  Exact Match: {summary['em_correct']}/{n} ({summary['em_pct']}%)")
    if "llm_pct" in summary:
        print(f"  LLM Judge:   {summary['llm_correct']}/{n} ({summary['llm_pct']}%)")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch LLM judge scoring for existing evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_file",
        metavar="FILE",
        help="Results JSON file to score",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="OUTPUT_FILE",
        help="Output path (default: overwrite input file)",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=["openai", "ollama", "anthropic", "groq", "together"],
        help="LLM provider for judging (default: from config / openai)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for judging (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples per LLM call (default: 10, max recommended: 20)",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        default=False,
        help="Only score entries that have llm=None; skip already-scored entries",
    )

    args = parser.parse_args()

    run_judge(
        input_path=args.input_file,
        output_path=args.output,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        fill_missing=args.fill_missing,
    )


if __name__ == "__main__":
    main()
