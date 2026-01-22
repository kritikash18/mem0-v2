"""
Generate and Compare Scores from Audio Evaluation Results.

Generates detailed summaries with:
- Per-sample detailed metrics
- Overall aggregate statistics
- Descriptive statistics (mean, std, min, max, percentiles)
- Multiple experiment comparison tables

Usage:
    # Single result file - detailed summary
    python -m audio_eval.generate_scores results/audio_eval/default_final.json

    # Compare multiple experiments
    python -m audio_eval.generate_scores results/audio_eval/*_final.json

    # Export comparison to CSV
    python -m audio_eval.generate_scores results/audio_eval/*.json --output comparison.csv
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
    """Convert results list to pandas DataFrame."""
    df = pd.DataFrame(results)
    return df


def print_detailed_summary(filepath: str, data: Dict[str, Any]):
    """
    Print comprehensive detailed summary for a single experiment.
    
    Includes:
    - Configuration overview
    - Descriptive statistics for all metrics
    - Score distributions
    - Sample-level breakdown
    """
    config = data.get("config", {})
    results = data.get("results", [])
    
    if not results:
        print(f"No results found in {filepath}")
        return
    
    df = results_to_dataframe(results)
    
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION SUMMARY")
    print("=" * 80)
    
    # Configuration
    print("\n[CONFIGURATION]")
    print("-" * 40)
    print(f"  Experiment:     {config.get('experiment', 'N/A')}")
    print(f"  Dataset:        {config.get('dataset', 'N/A')}")
    print(f"  ASR:            {config.get('asr', 'N/A')}")
    print(f"  LLM:            {config.get('llm', 'N/A')}")
    print(f"  Embedder:       {config.get('embedder', 'N/A')}")
    print(f"  Top-K:          {config.get('top_k', 'N/A')}")
    print(f"  Infer Memories: {config.get('infer_memories', 'N/A')}")
    print(f"  Total Samples:  {len(results)}")
    
    # Accuracy Metrics - Descriptive Statistics
    print("\n[ACCURACY METRICS - DESCRIPTIVE STATISTICS]")
    print("-" * 80)
    
    accuracy_cols = ["em", "f1", "bleu", "llm"]
    available_cols = [c for c in accuracy_cols if c in df.columns]
    
    if available_cols:
        stats = df[available_cols].describe()
        stats.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        print(stats.round(4).to_string())
    
    # Overall Mean Scores (like original generate_scores.py)
    print("\n[OVERALL MEAN SCORES]")
    print("-" * 40)
    for col in available_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col.upper():6} : {mean_val:.4f} (±{std_val:.4f})")
    
    # Score Distribution
    print("\n[SCORE DISTRIBUTION]")
    print("-" * 40)
    if "em" in df.columns:
        em_correct = (df["em"] == 1).sum()
        em_total = len(df)
        print(f"  Exact Match:  {em_correct}/{em_total} ({100*em_correct/em_total:.1f}%)")
    
    if "llm" in df.columns:
        llm_correct = (df["llm"] == 1).sum()
        llm_total = len(df)
        print(f"  LLM Correct:  {llm_correct}/{llm_total} ({100*llm_correct/llm_total:.1f}%)")
    
    if "f1" in df.columns:
        f1_perfect = (df["f1"] == 1.0).sum()
        f1_above_50 = (df["f1"] >= 0.5).sum()
        print(f"  F1 = 1.0:     {f1_perfect}/{len(df)} ({100*f1_perfect/len(df):.1f}%)")
        print(f"  F1 >= 0.5:    {f1_above_50}/{len(df)} ({100*f1_above_50/len(df):.1f}%)")
    
    # Latency Metrics
    print("\n[LATENCY METRICS (seconds)]")
    print("-" * 80)
    
    latency_cols = ["total_time", "add_time", "search_time", "answer_time"]
    available_latency = [c for c in latency_cols if c in df.columns]
    
    if available_latency:
        latency_stats = df[available_latency].describe()
        latency_stats.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        print(latency_stats.round(3).to_string())
    
    # Total time
    if "total_time" in df.columns:
        total_eval_time = df["total_time"].sum()
        print(f"\n  Total evaluation time: {total_eval_time:.2f}s ({total_eval_time/60:.2f} min)")
    
    # Error Summary
    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if len(errors) > 0:
            print("\n[ERRORS]")
            print("-" * 40)
            print(f"  Total errors: {len(errors)}")
            for _, row in errors.head(5).iterrows():
                print(f"    Sample {row.get('idx', 'N/A')}: {row['error'][:60]}...")
    
    # Sample Results Preview
    print("\n[SAMPLE RESULTS PREVIEW]")
    print("-" * 80)
    
    preview_cols = ["idx", "question", "ground_truth", "prediction", "em", "f1", "llm"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    
    # Show first 5 samples
    preview = df[preview_cols].head(5).copy()
    if "question" in preview.columns:
        preview["question"] = preview["question"].str[:40] + "..."
    if "ground_truth" in preview.columns:
        preview["ground_truth"] = preview["ground_truth"].str[:20] + "..."
    if "prediction" in preview.columns:
        preview["prediction"] = preview["prediction"].str[:20] + "..."
    
    print(preview.to_string(index=False))
    
    print("\n" + "=" * 80 + "\n")


def compare_experiments(all_data: List[Dict[str, Any]]):
    """
    Print comparison table for multiple experiments.
    
    Similar to original generate_scores.py but for multiple files.
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    
    # Build comparison dataframe
    comparison_rows = []
    
    for data in all_data:
        config = data.get("config", {})
        results = data.get("results", [])
        
        if not results:
            continue
        
        df = results_to_dataframe(results)
        
        row = {
            "Experiment": config.get("experiment", "N/A")[:20],
            "ASR": config.get("asr", "N/A").split("/")[-1][:15] if config.get("asr") else "N/A",
            "LLM": config.get("llm", "N/A").split("/")[-1][:15] if config.get("llm") else "N/A",
            "N": len(results),
        }
        
        # Add metrics
        for metric in ["em", "f1", "bleu", "llm"]:
            if metric in df.columns:
                row[metric.upper()] = df[metric].mean()
        
        if "total_time" in df.columns:
            row["Time(s)"] = df["total_time"].mean()
        
        comparison_rows.append(row)
    
    # Create and sort comparison DataFrame
    comp_df = pd.DataFrame(comparison_rows)
    
    if "LLM" in comp_df.columns and comp_df["LLM"].dtype in ["float64", "int64"]:
        comp_df = comp_df.sort_values("LLM", ascending=False)
    
    print("\n[COMPARISON TABLE]")
    print("-" * 100)
    print(comp_df.round(4).to_string(index=False))
    
    # Best scores
    print("\n[BEST SCORES]")
    print("-" * 40)
    
    metric_cols = ["EM", "F1", "BLEU", "LLM"]
    for metric in metric_cols:
        if metric in comp_df.columns:
            best_idx = comp_df[metric].idxmax()
            best_exp = comp_df.loc[best_idx, "Experiment"]
            best_val = comp_df.loc[best_idx, metric]
            print(f"  Best {metric}:   {best_exp} ({best_val:.4f})")
    
    if "Time(s)" in comp_df.columns:
        fastest_idx = comp_df["Time(s)"].idxmin()
        fastest_exp = comp_df.loc[fastest_idx, "Experiment"]
        fastest_val = comp_df.loc[fastest_idx, "Time(s)"]
        print(f"  Fastest:   {fastest_exp} ({fastest_val:.2f}s)")
    
    print("\n" + "=" * 100 + "\n")


def save_to_csv(all_data: List[Dict[str, Any]], output_path: str):
    """Save detailed comparison to CSV."""
    rows = []
    
    for data in all_data:
        config = data.get("config", {})
        results = data.get("results", [])
        
        if not results:
            continue
        
        df = results_to_dataframe(results)
        
        row = {
            "experiment": config.get("experiment", ""),
            "dataset": config.get("dataset", ""),
            "asr": config.get("asr", ""),
            "llm": config.get("llm", ""),
            "embedder": config.get("embedder", ""),
            "top_k": config.get("top_k", ""),
            "samples": len(results),
        }
        
        # Metrics with mean and std
        for metric in ["em", "f1", "bleu", "llm", "total_time", "add_time", "search_time", "answer_time"]:
            if metric in df.columns:
                row[f"{metric}_mean"] = df[metric].mean()
                row[f"{metric}_std"] = df[metric].std()
                row[f"{metric}_min"] = df[metric].min()
                row[f"{metric}_max"] = df[metric].max()
        
        rows.append(row)
    
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")


def save_detailed_csv(data: Dict[str, Any], output_path: str):
    """Save detailed per-sample results to CSV."""
    results = data.get("results", [])
    if not results:
        print("No results to save")
        return
    
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed scores and comparisons from audio evaluation results"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result JSON files (supports glob patterns like results/*.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for comparison (CSV format)"
    )
    parser.add_argument(
        "--detailed-csv",
        type=str,
        default=None,
        help="Save detailed per-sample results to CSV (only for single file)"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    filepaths = []
    for pattern in args.files:
        if "*" in pattern:
            filepaths.extend(glob.glob(pattern))
        else:
            filepaths.append(pattern)
    
    # Filter to existing files
    filepaths = [f for f in filepaths if os.path.exists(f)]
    
    if not filepaths:
        print("No valid result files found")
        return
    
    # Load all results
    all_data = []
    for filepath in filepaths:
        try:
            data = load_results(filepath)
            data["_filepath"] = filepath
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not all_data:
        print("No results loaded")
        return
    
    # Display results
    if len(all_data) == 1:
        # Single file - show detailed summary
        print_detailed_summary(all_data[0]["_filepath"], all_data[0])
        
        # Optionally save detailed CSV
        if args.detailed_csv:
            save_detailed_csv(all_data[0], args.detailed_csv)
    else:
        # Multiple files - show comparison first
        compare_experiments(all_data)
        
        # Then show detailed summary for each
        for data in all_data:
            print_detailed_summary(data["_filepath"], data)
    
    # Save comparison CSV if requested
    if args.output:
        save_to_csv(all_data, args.output)


if __name__ == "__main__":
    main()
