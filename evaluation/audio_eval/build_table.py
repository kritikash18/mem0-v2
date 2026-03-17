"""
Build a Markdown results table from cumulative evaluation checkpoint files.

Reads the summary block from each checkpoint JSON and produces two Markdown
tables — Accuracy (EM, F1, BLEU, LLM Judge) and Latency (Total, Add, Search,
Answer Gen) — written to a single output file.

Usage:
    # Default: reads results/audio_eval/whisper_gpt_* checkpoints,
    #          writes results/audio_eval/results_table.md
    python -m audio_eval.build_table

    # Custom results directory and output path
    python -m audio_eval.build_table \
        --results-dir results/audio_eval \
        --output results/audio_eval/results_table.md

    # Custom experiment prefix (e.g. for a different ASR/LLM combo)
    python -m audio_eval.build_table --prefix whisper_gpt

    # Override individual checkpoint files
    python -m audio_eval.build_table \
        --checkpoints 50:whisper_gpt_0_50_final.json \
                      100:whisper_gpt_cum_100.json \
                      500:whisper_gpt_cum_500.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def _load_summary(results_dir: str, fname: str):
    """Return the summary dict for a checkpoint file, or None if missing."""
    path = Path(results_dir) / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("summary", {})


def _load_config(results_dir: str, fname: str):
    """Return the config dict for a checkpoint file, or {} if missing."""
    path = Path(results_dir) / fname
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("config", {})


def build_tables(
    checkpoints: List[Tuple[str, str]],
    results_dir: str,
    output_path: str,
    experiment_label: Optional[str] = None,
):
    """
    Build accuracy and latency Markdown tables from checkpoint files.

    Args:
        checkpoints: List of (N_label, filename) pairs in display order
        results_dir: Directory containing the result JSON files
        output_path: Path to write the output Markdown file
        experiment_label: Optional label for the experiment header
    """
    lines = []

    # Header
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = experiment_label or "mem0 Evaluation Results"
    lines.append(f"# {label}")
    lines.append(f"*Generated: {ts}*\n")

    # Pull config from the largest available checkpoint for the metadata block
    cfg = {}
    for _, fname in reversed(checkpoints):
        cfg = _load_config(results_dir, fname)
        if cfg:
            break
    if cfg:
        lines.append("## Configuration\n")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        for key in ["framework", "asr", "llm", "embedder", "judge", "top_k"]:
            val = cfg.get(key)
            if val is not None:
                lines.append(f"| {key} | `{val}` |")
        lines.append("")

    # ── Accuracy table ──────────────────────────────────────────────────────
    lines.append("## Accuracy\n")
    lines.append("| N | EM (%) | F1 | BLEU | LLM Judge (%) |")
    lines.append("|---|--------|----|------|---------------|")

    for label_n, fname in checkpoints:
        s = _load_summary(results_dir, fname)
        if s is None:
            lines.append(f"| {label_n:<4} | *(pending)* | | | |")
            continue
        em   = s.get("em_pct",  "—")
        f1   = round(s.get("f1_mean",   0), 3)
        bleu = round(s.get("bleu_mean", 0), 3)
        llm  = s.get("llm_pct", "—")
        em_str  = f"{em}%" if isinstance(em, (int, float)) else str(em)
        llm_str = f"{llm}%" if isinstance(llm, (int, float)) else str(llm)
        lines.append(f"| {label_n:<4} | {em_str} | {f1} | {bleu} | {llm_str} |")

    lines.append("")

    # ── Latency table ───────────────────────────────────────────────────────
    lines.append("## Latency (mean per sample)\n")
    lines.append("| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |")
    lines.append("|---|-----------|---------|------------|----------------|")

    for label_n, fname in checkpoints:
        s = _load_summary(results_dir, fname)
        if s is None:
            lines.append(f"| {label_n:<4} | *(pending)* | | | |")
            continue
        tot    = round(s.get("total_time_mean",  0), 2)
        add    = round(s.get("add_time_mean",    0), 2)
        search = round(s.get("search_time_mean", 0), 2)
        ans    = round(s.get("answer_time_mean", 0), 2)
        lines.append(f"| {label_n:<4} | {tot} | {add} | {search} | {ans} |")

    lines.append("")

    # ── Score distributions (F1 > 0.8, F1 > 0.5) ──────────────────────────
    lines.append("## F1 Score Distribution\n")
    lines.append("| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |")
    lines.append("|---|---------------|-----------|-----------|------|")

    for label_n, fname in checkpoints:
        path = Path(results_dir) / fname
        if not path.exists():
            lines.append(f"| {label_n:<4} | *(pending)* | | | |")
            continue
        with open(path) as f:
            data = json.load(f)
        dist = data.get("distributions", {}).get("f1", {})
        if not dist:
            lines.append(f"| {label_n:<4} | — | — | — | — |")
            continue
        total   = dist.get("total", 1)
        perfect = dist.get("perfect", 0)
        a80     = dist.get("above_80", 0)
        a50     = dist.get("above_50", 0)
        zero    = dist.get("zero", 0)
        lines.append(
            f"| {label_n:<4} "
            f"| {perfect} ({round(100*perfect/total, 1)}%) "
            f"| {a80} ({round(100*a80/total, 1)}%) "
            f"| {a50} ({round(100*a50/total, 1)}%) "
            f"| {zero} ({round(100*zero/total, 1)}%) |"
        )

    lines.append("")

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Results table written to: {output_path}")


def _default_checkpoints(prefix: str) -> List[Tuple[str, str]]:
    return [
        ("50",   f"{prefix}_0_50_final.json"),
        ("100",  f"{prefix}_cum_100.json"),
        ("500",  f"{prefix}_cum_500.json"),
        ("1000", f"{prefix}_cum_1000.json"),
        ("2000", f"{prefix}_cum_2000.json"),
        ("3000", f"{prefix}_cum_3000.json"),
        ("4704", f"{prefix}_cum_4704.json"),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build Markdown results table from evaluation checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", "-d",
        default="results/audio_eval",
        help="Directory containing result JSON files (default: results/audio_eval)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output Markdown file path (default: <results-dir>/results_table.md)",
    )
    parser.add_argument(
        "--prefix", "-p",
        default="whisper_gpt",
        help="Experiment filename prefix used to locate checkpoint files (default: whisper_gpt)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="N:FILENAME",
        help=(
            "Override checkpoint list as space-separated N:filename pairs, e.g. "
            "--checkpoints 50:run_0_50_final.json 100:cum_100.json"
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Title label for the output document (default: 'mem0 Evaluation Results')",
    )

    args = parser.parse_args()

    # Resolve checkpoint list
    if args.checkpoints:
        checkpoints = []
        for item in args.checkpoints:
            parts = item.split(":", 1)
            if len(parts) != 2:
                print(f"Error: invalid checkpoint format '{item}'. Use N:filename.", file=sys.stderr)
                sys.exit(1)
            checkpoints.append((parts[0], parts[1]))
    else:
        checkpoints = _default_checkpoints(args.prefix)

    output_path = args.output or str(Path(args.results_dir) / "results_table.md")

    build_tables(
        checkpoints=checkpoints,
        results_dir=args.results_dir,
        output_path=output_path,
        experiment_label=args.label,
    )


if __name__ == "__main__":
    main()
