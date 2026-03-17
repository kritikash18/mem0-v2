"""
Combine multiple incremental evaluation result files into one.

Use this after running the evaluator in range-based batches to merge the
partial JSONs into a single file with recomputed summary statistics.

Usage:
    # Merge two incremental runs
    python -m audio_eval.combine_results \\
        results/audio_eval/run_0_100_final.json \\
        results/audio_eval/run_100_500_final.json \\
        -o results/audio_eval/merged_0_500_final.json

    # Merge three runs into one
    python -m audio_eval.combine_results \\
        results/audio_eval/run_0_100_final.json \\
        results/audio_eval/run_100_500_final.json \\
        results/audio_eval/run_500_1000_final.json \\
        -o results/audio_eval/full_1000_final.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .metrics import aggregate_results, compute_score_distribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_result_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(p) as f:
        return json.load(f)


def merge_files(input_paths: List[str], output_path: str):
    """
    Merge result files and write combined output.

    Duplicate sample indices are resolved by keeping the entry from the
    *latest* file in the argument list (rightmost wins). A warning is printed
    so you know which samples were overwritten.
    """
    if len(input_paths) < 2:
        logger.error("Provide at least two input files to merge.")
        sys.exit(1)

    all_data = [load_result_file(p) for p in input_paths]

    # Collect results, latest file wins on duplicate idx
    by_idx: Dict[int, Dict[str, Any]] = {}
    for file_path, data in zip(input_paths, all_data):
        file_results = data.get("results", [])
        for r in file_results:
            idx = r.get("idx")
            if idx in by_idx:
                logger.warning(
                    f"Duplicate index {idx} — overwriting entry from earlier file "
                    f"with one from {file_path}"
                )
            by_idx[idx] = r

    merged_results = sorted(by_idx.values(), key=lambda r: r.get("idx", 0))
    logger.info(f"Total unique samples after merge: {len(merged_results)}")

    # Recompute summary using merged results
    summary = aggregate_results(merged_results)
    distributions = {
        "f1": compute_score_distribution(merged_results, "f1"),
        "bleu": compute_score_distribution(merged_results, "bleu"),
    }

    # Inherit config metadata from the first file; annotate it
    base_config = all_data[0].get("config", {})
    base_config["merged_from"] = input_paths
    base_config["merged_sample_count"] = len(merged_results)

    output_data = {
        "config": base_config,
        "summary": summary,
        "distributions": distributions,
        "results": merged_results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Merged results saved to: {output_path}")
    _print_summary(summary, len(merged_results))


def _print_summary(summary: Dict[str, Any], n: int):
    print("\n" + "=" * 55)
    print(f"MERGED SUMMARY  ({n} samples)")
    print("=" * 55)
    print(f"  {'Metric':<8} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 28)
    for metric in ["em", "f1", "bleu", "llm"]:
        mean = summary.get(f"{metric}_mean")
        std = summary.get(f"{metric}_std")
        if mean is None:
            continue
        print(f"  {metric.upper():<8} {mean:>8.4f} {std if std is not None else 0.0:>8.4f}")
    if "em_pct" in summary:
        print(f"\n  Exact Match: {summary['em_correct']}/{n} ({summary['em_pct']}%)")
    if "llm_pct" in summary:
        print(f"  LLM Judge:   {summary['llm_correct']}/{n} ({summary['llm_pct']}%)")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge incremental audio_eval result JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="FILE",
        help="Two or more result JSON files to merge (in order; later files win on duplicate indices)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="OUTPUT_FILE",
        help="Path to write the merged JSON file",
    )
    args = parser.parse_args()
    merge_files(args.input_files, args.output)


if __name__ == "__main__":
    main()
