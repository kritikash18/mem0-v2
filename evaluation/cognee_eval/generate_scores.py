"""
Generate and Compare Scores from Cognee Evaluation Results.

Supports:
- Single experiment detailed summary
- Multiple Cognee experiment comparison
- Cross-framework comparison (Cognee vs mem0)

Usage:
    # Single Cognee result
    python -m cognee_eval.generate_scores results/cognee_eval/default_final.json

    # Compare Cognee experiments
    python -m cognee_eval.generate_scores results/cognee_eval/*_final.json

    # Cross-framework comparison (Cognee vs mem0)
    python -m cognee_eval.generate_scores results/cognee_eval/*_final.json results/audio_eval/*_final.json
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def print_detailed_summary(filepath: str, data: Dict[str, Any]):
    """Print comprehensive summary for a single experiment."""
    config_data = data.get("config", {})
    results = data.get("results", [])

    if not results:
        print(f"No results found in {filepath}")
        return

    df = results_to_dataframe(results)
    framework = config_data.get("framework", "unknown")

    print("\n" + "=" * 80)
    print(f"DETAILED EVALUATION SUMMARY ({framework.upper()})")
    print("=" * 80)

    print("\n[CONFIGURATION]")
    print("-" * 40)
    print(f"  Framework:      {framework}")
    print(f"  Experiment:     {config_data.get('experiment', 'N/A')}")
    print(f"  Dataset:        {config_data.get('dataset', 'N/A')}")
    print(f"  ASR:            {config_data.get('asr', 'N/A')}")

    if framework == "cognee":
        print(f"  Cognee LLM:     {config_data.get('cognee_llm', 'N/A')}")
        print(f"  Search Type:    {config_data.get('search_type', 'N/A')}")
        print(f"  Answer LLM:     {config_data.get('answer_llm', 'N/A')}")
    else:
        print(f"  LLM:            {config_data.get('llm', 'N/A')}")
        print(f"  Embedder:       {config_data.get('embedder', 'N/A')}")
        print(f"  Top-K:          {config_data.get('top_k', 'N/A')}")

    print(f"  Total Samples:  {len(results)}")

    # Accuracy
    accuracy_cols = [c for c in ["em", "f1", "bleu", "llm"] if c in df.columns]
    if accuracy_cols:
        print("\n[ACCURACY METRICS]")
        print("-" * 80)
        stats = df[accuracy_cols].describe()
        stats.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        print(stats.round(4).to_string())

    print("\n[OVERALL MEAN SCORES]")
    print("-" * 40)
    for col in accuracy_cols:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col.upper():6} : {mean:.4f} (±{std:.4f})")

    # Distributions
    print("\n[SCORE DISTRIBUTION]")
    print("-" * 40)
    if "em" in df.columns:
        em_correct = (df["em"] == 1).sum()
        print(f"  Exact Match:  {em_correct}/{len(df)} ({100*em_correct/len(df):.1f}%)")
    if "llm" in df.columns:
        llm_correct = (df["llm"] == 1).sum()
        print(f"  LLM Correct:  {llm_correct}/{len(df)} ({100*llm_correct/len(df):.1f}%)")
    if "f1" in df.columns:
        f1_perfect = (df["f1"] == 1.0).sum()
        f1_above_50 = (df["f1"] >= 0.5).sum()
        print(f"  F1 = 1.0:     {f1_perfect}/{len(df)} ({100*f1_perfect/len(df):.1f}%)")
        print(f"  F1 >= 0.5:    {f1_above_50}/{len(df)} ({100*f1_above_50/len(df):.1f}%)")

    # Latency
    latency_cols = [c for c in ["total_time", "asr_time", "cognify_time", "add_time", "search_time", "answer_time"] if c in df.columns]
    if latency_cols:
        print("\n[LATENCY METRICS (seconds)]")
        print("-" * 80)
        print(df[latency_cols].describe().round(3).to_string())

        if "total_time" in df.columns:
            total = df["total_time"].sum()
            print(f"\n  Total evaluation time: {total:.2f}s ({total/60:.2f} min)")

    print("\n" + "=" * 80 + "\n")


def compare_experiments(all_data: List[Dict[str, Any]]):
    """Print cross-framework comparison table."""
    print("\n" + "=" * 110)
    print("FRAMEWORK COMPARISON")
    print("=" * 110)

    rows = []
    for data in all_data:
        cfg = data.get("config", {})
        results = data.get("results", [])
        if not results:
            continue

        df = results_to_dataframe(results)
        framework = cfg.get("framework", "mem0")

        row = {
            "Framework": framework,
            "Experiment": cfg.get("experiment", "N/A")[:20],
            "ASR": cfg.get("asr", "N/A").split("/")[-1][:15] if cfg.get("asr") else "N/A",
            "N": len(results),
        }

        for metric in ["em", "f1", "bleu", "llm"]:
            if metric in df.columns:
                row[metric.upper()] = df[metric].mean()

        if "total_time" in df.columns:
            row["Time(s)"] = df["total_time"].mean()

        rows.append(row)

    comp_df = pd.DataFrame(rows)

    print("\n[COMPARISON TABLE]")
    print("-" * 110)
    print(comp_df.round(4).to_string(index=False))

    # Best per framework
    for fw in comp_df["Framework"].unique():
        fw_df = comp_df[comp_df["Framework"] == fw]
        print(f"\n[BEST SCORES - {fw.upper()}]")
        print("-" * 40)
        for metric in ["EM", "F1", "BLEU", "LLM"]:
            if metric in fw_df.columns:
                best_idx = fw_df[metric].idxmax()
                print(f"  Best {metric}: {fw_df.loc[best_idx, 'Experiment']} ({fw_df.loc[best_idx, metric]:.4f})")

    print("\n" + "=" * 110 + "\n")


def save_to_csv(all_data: List[Dict], output_path: str):
    """Save comparison to CSV."""
    rows = []
    for data in all_data:
        cfg = data.get("config", {})
        results = data.get("results", [])
        if not results:
            continue

        df = results_to_dataframe(results)

        row = {
            "framework": cfg.get("framework", "mem0"),
            "experiment": cfg.get("experiment", ""),
            "asr": cfg.get("asr", ""),
            "samples": len(results),
        }

        for metric in ["em", "f1", "bleu", "llm", "total_time", "asr_time", "cognify_time", "add_time", "search_time", "answer_time"]:
            if metric in df.columns:
                row[f"{metric}_mean"] = df[metric].mean()
                row[f"{metric}_std"] = df[metric].std()

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scores and cross-framework comparisons"
    )
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--output", "-o", type=str, default=None, help="CSV output path")

    args = parser.parse_args()

    filepaths = []
    for pattern in args.files:
        if "*" in pattern:
            filepaths.extend(glob.glob(pattern))
        else:
            filepaths.append(pattern)

    filepaths = [f for f in filepaths if os.path.exists(f)]

    if not filepaths:
        print("No valid result files found")
        return

    all_data = []
    for fp in filepaths:
        try:
            data = load_results(fp)
            data["_filepath"] = fp
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    if not all_data:
        print("No results loaded")
        return

    if len(all_data) == 1:
        print_detailed_summary(all_data[0]["_filepath"], all_data[0])
    else:
        compare_experiments(all_data)
        for data in all_data:
            print_detailed_summary(data["_filepath"], data)

    if args.output:
        save_to_csv(all_data, args.output)


if __name__ == "__main__":
    main()
