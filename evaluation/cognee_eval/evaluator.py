"""
Cognee Memory Evaluator.

Evaluation pipeline for testing Cognee (knowledge graph memory framework) on
the same audio-based memory retrieval task used for mem0 evaluation.

Cognee natively supports audio ingestion (.wav, .mp3, .flac) — audio is passed
directly to cognee.add() as a file, and Cognee handles transcription internally.
This gives a true independent framework comparison on the same raw audio input.

Pipeline:
1. Load audio samples from HuggingFace dataset
2. Write audio array to a temporary .wav file
3. Add audio file to Cognee (cognee.add) — Cognee transcribes internally
4. Build knowledge graph (cognee.cognify)
5. Search with question (cognee.search)
6. Generate answer from search results
7. Evaluate against ground truth

Usage:
    python -m cognee_eval.evaluator --num_samples 10
    python -m cognee_eval.evaluator --num_samples 50 --experiment_name cognee_vs_mem0
    python -m cognee_eval.evaluator -n 10 --search-type RAG_COMPLETION
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from . import config
from .metrics import aggregate_results, compute_metrics, compute_score_distribution
from .prompts import ANSWER_PROMPT

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# AUDIO FILE UTILITIES
# =============================================================================

def write_audio_to_tempfile(audio_data: dict) -> str:
    """
    Write a decoded HuggingFace audio array to a temporary .wav file.

    Cognee's add() accepts file paths, not in-memory numpy arrays, so we
    write to a named temp file. The caller is responsible for deleting it.

    Args:
        audio_data: Dict with "array" (numpy float32) and "sampling_rate" keys

    Returns:
        Path to the temporary .wav file
    """
    array = audio_data["array"]
    sr = audio_data["sampling_rate"]

    # soundfile expects float32 or int16; HuggingFace audio is already float32
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, array, sr)
    return tmp.name


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_evaluation_dataset(num_samples: Optional[int] = None):
    """Load dataset from HuggingFace (same as audio_eval)."""
    logger.info(f"Loading dataset: {config.DATASET_NAME}")

    ds = load_dataset(
        config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        token=config.HF_TOKEN
    )
    ds = ds.cast_column(config.AUDIO_COLUMN, Audio(decode=True))

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    logger.info(f"Loaded {len(ds)} samples")
    return ds


# =============================================================================
# COGNEE OPERATIONS
# =============================================================================

async def cognee_reset():
    """Reset Cognee state for sample isolation."""
    import cognee
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception as e:
        logger.warning(f"Cognee reset warning: {e}")


async def cognee_add_and_cognify(audio_path: str, dataset_name: str = "eval_doc") -> float:
    """
    Add an audio file to Cognee and build the knowledge graph.

    Cognee handles audio transcription internally when given a .wav file path.

    Args:
        audio_path: Path to a .wav file
        dataset_name: Dataset name for Cognee

    Returns:
        Time taken for add + cognify (seconds)
    """
    import cognee

    start = time.time()
    await cognee.add(audio_path, dataset_name)
    await cognee.cognify()
    return time.time() - start


async def cognee_search(query: str, search_type: str = None) -> tuple:
    """
    Search Cognee knowledge graph.

    Args:
        query: Question to search for
        search_type: Cognee search type (defaults to config)

    Returns:
        Tuple of (results_list, search_time)
    """
    import cognee

    search_type = search_type or config.COGNEE_SEARCH_TYPE

    # Try the current import path first; fall back to the legacy path
    try:
        from cognee import SearchType
    except ImportError:
        from cognee.api.v1.search import SearchType

    type_map = {
        "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
        "RAG_COMPLETION": SearchType.RAG_COMPLETION,
        "CHUNKS": SearchType.CHUNKS,
        "SUMMARIES": SearchType.SUMMARIES,
        "GRAPH_SUMMARY_COMPLETION": SearchType.GRAPH_SUMMARY_COMPLETION,
        "GRAPH_COMPLETION_COT": SearchType.GRAPH_COMPLETION_COT,
    }

    cognee_type = type_map.get(search_type, SearchType.GRAPH_COMPLETION)

    start = time.time()
    results = await cognee.search(query_text=query, query_type=cognee_type)
    search_time = time.time() - start

    return results, search_time


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def format_cognee_results(results) -> str:
    """
    Format Cognee search results for the answer prompt.

    Cognee returns different formats depending on search type.
    This normalizes them into a readable string.
    """
    if not results:
        return "No relevant information found."

    formatted = []
    for i, result in enumerate(results, 1):
        if isinstance(result, str):
            formatted.append(f"{i}. {result}")
        elif isinstance(result, dict):
            text = result.get("text", result.get("content", result.get("description", str(result))))
            formatted.append(f"{i}. {text}")
        elif hasattr(result, "text"):
            formatted.append(f"{i}. {result.text}")
        elif hasattr(result, "content"):
            formatted.append(f"{i}. {result.content}")
        else:
            formatted.append(f"{i}. {str(result)}")

    return "\n".join(formatted)


def generate_answer(
    question: str,
    cognee_results,
    openai_client: OpenAI,
    model: str = None,
) -> tuple:
    """Generate answer from Cognee search results using LLM."""
    if not cognee_results:
        return "I don't know", 0.0

    model = model or config.LLM_MODEL
    results_text = format_cognee_results(cognee_results)

    template = Template(ANSWER_PROMPT)
    prompt = template.render(results=results_text, question=question)

    start = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
    )
    gen_time = time.time() - start

    answer = response.choices[0].message.content.strip()
    return answer, gen_time


# =============================================================================
# SINGLE SAMPLE EVALUATION
# =============================================================================

async def evaluate_sample(
    sample: Dict,
    idx: int,
    openai_client: OpenAI,
) -> Dict[str, Any]:
    """
    Evaluate a single sample through the Cognee pipeline.

    Steps:
    1. Write audio array to a temp .wav file
    2. Reset Cognee state (for isolation)
    3. Add audio file to Cognee + cognify (Cognee transcribes internally)
    4. Search with question
    5. Generate answer
    6. Compute metrics
    """
    audio_raw = sample[config.AUDIO_COLUMN]
    try:
        audio = {
            "array": audio_raw["array"],
            "sampling_rate": audio_raw["sampling_rate"],
        }
    except (TypeError, KeyError) as e:
        raise ValueError(f"Cannot extract audio from {type(audio_raw)}: {e}")

    question = sample[config.QUESTION_COLUMN]
    ground_truth = sample[config.ANSWER_COLUMN]

    result = {
        "idx": idx,
        "question": question,
        "ground_truth": ground_truth,
    }

    audio_path = None
    try:
        # Step 1: Write audio to temp file for Cognee ingestion
        audio_path = write_audio_to_tempfile(audio)
        logger.debug(f"[Sample {idx}] Audio written to {audio_path}")

        # Step 2: Reset Cognee for isolation
        if config.RESET_BETWEEN_SAMPLES:
            await cognee_reset()

        # Step 3: Add audio file to Cognee + cognify
        # Cognee handles transcription internally from the .wav file
        cognify_time = await cognee_add_and_cognify(audio_path)

        # Step 4: Search
        search_results, search_time = await cognee_search(question)

        num_results = len(search_results) if search_results else 0
        logger.debug(f"[Sample {idx}] Cognee returned {num_results} results")

        # Step 5: Generate answer
        prediction, answer_time = generate_answer(
            question=question,
            cognee_results=search_results,
            openai_client=openai_client,
        )

        # Step 6: Compute metrics
        metrics = compute_metrics(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            include_llm_judge=True,
            openai_client=openai_client,
        )

        result.update({
            "prediction": prediction,
            "num_results": num_results,
            "search_results_preview": format_cognee_results(
                search_results[:3] if search_results else []
            ),
            "cognify_time": round(cognify_time, 3),
            "search_time": round(search_time, 3),
            "answer_time": round(answer_time, 3),
            "total_time": round(cognify_time + search_time + answer_time, 3),
            **metrics,
        })

    except Exception as e:
        logger.error(f"Error evaluating sample {idx}: {e}")
        result.update({
            "prediction": "",
            "error": str(e),
            "em": 0,
            "f1": 0.0,
            "bleu": 0.0,
            "llm": 0,
            "total_time": 0,
        })

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    return result


# =============================================================================
# COGNEE ENVIRONMENT SETUP
# =============================================================================

def setup_cognee_env():
    """
    Configure Cognee's environment variables before import.

    Cognee reads configuration from environment variables,
    so we set them based on our config before any cognee imports.
    """
    os.environ["LLM_API_KEY"] = config.LLM_API_KEY or ""
    os.environ["LLM_PROVIDER"] = config.COGNEE_LLM_PROVIDER
    os.environ["LLM_MODEL"] = config.COGNEE_LLM_MODEL

    os.environ["EMBEDDING_PROVIDER"] = config.COGNEE_EMBEDDING_PROVIDER
    os.environ["EMBEDDING_MODEL"] = config.COGNEE_EMBEDDING_MODEL
    os.environ.setdefault("EMBEDDING_API_KEY", config.LLM_API_KEY or "")
    os.environ["EMBEDDING_DIMENSIONS"] = str(config.COGNEE_EMBEDDING_DIMENSIONS)


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

async def _run_evaluation_async(
    num_samples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    search_type: Optional[str] = None,
    no_reset: bool = False,
) -> List[Dict[str, Any]]:
    """Async implementation of the evaluation runner."""

    if search_type:
        config.COGNEE_SEARCH_TYPE = search_type
    if no_reset:
        config.RESET_BETWEEN_SAMPLES = False

    config.validate_config()

    experiment_name = experiment_name or config.EXPERIMENT_NAME

    # Setup Cognee environment
    setup_cognee_env()

    # Load dataset and resolve which indices to evaluate
    if sample_indices:
        dataset = load_evaluation_dataset(None)
        max_idx = max(sample_indices)
        if max_idx >= len(dataset):
            raise ValueError(f"Index {max_idx} out of range. Dataset has {len(dataset)} samples.")
        indices_to_evaluate = sample_indices
        logger.info(f"Evaluating {len(indices_to_evaluate)} explicit indices")
    elif start is not None or end is not None:
        dataset = load_evaluation_dataset(None)
        _start = start if start is not None else 0
        _end = min(end, len(dataset)) if end is not None else len(dataset)
        if _start >= len(dataset):
            raise ValueError(f"--start {_start} is out of range. Dataset has {len(dataset)} samples.")
        indices_to_evaluate = list(range(_start, _end))
        logger.info(f"Evaluating range [{_start}, {_end}) = {len(indices_to_evaluate)} samples")
    else:
        dataset = load_evaluation_dataset(num_samples)
        indices_to_evaluate = list(range(len(dataset)))

    # Initialize OpenAI client for answer generation and LLM judge
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    results = []
    total = len(indices_to_evaluate)

    logger.info(f"Starting Cognee evaluation: {total} samples")
    logger.info(f"Cognee LLM: {config.COGNEE_LLM_MODEL}")
    logger.info(f"Search Type: {config.COGNEE_SEARCH_TYPE}")

    for i, idx in enumerate(tqdm(indices_to_evaluate, desc="Evaluating (Cognee)")):
        sample = dataset[idx]

        result = await evaluate_sample(
            sample=sample,
            idx=idx,
            openai_client=openai_client,
        )
        results.append(result)

        if result.get("error"):
            logger.warning(f"[Sample {idx}] ERROR: {result['error'][:80]}")
        else:
            logger.debug(
                f"[Sample {idx}] EM={result['em']}, F1={result['f1']:.2f}, "
                f"LLM={result.get('llm', 'N/A')}, Time={result['total_time']:.1f}s"
            )

        # Save intermediate results every 10 samples
        if (i + 1) % 10 == 0:
            save_results(results, experiment_name, "intermediate")

    # Save final results
    save_results(results, experiment_name, "final")
    print_summary(results, experiment_name)

    return results


def run_evaluation(
    num_samples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    search_type: Optional[str] = None,
    no_reset: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full Cognee evaluation pipeline (sync wrapper).

    Audio is passed directly to Cognee as a .wav file — Cognee handles
    transcription internally, giving a true independent framework comparison.

    Args:
        num_samples: Number of samples to evaluate
        experiment_name: Name for output files
        sample_indices: Specific indices to evaluate
        search_type: Cognee search type override
        no_reset: Disable prune between samples (memories accumulate across samples)

    Returns:
        List of result dictionaries
    """
    return asyncio.run(_run_evaluation_async(
        num_samples=num_samples,
        experiment_name=experiment_name,
        sample_indices=sample_indices,
        start=start,
        end=end,
        search_type=search_type,
        no_reset=no_reset,
    ))


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(results: List[Dict], experiment_name: str, suffix: str):
    """Save results to JSON file."""
    output_path = config.get_output_path(experiment_name, suffix)

    summary = aggregate_results(results)

    data = {
        "config": {
            "framework": "cognee",
            "experiment": experiment_name,
            "dataset": config.DATASET_NAME,
            "cognee_llm": f"{config.COGNEE_LLM_PROVIDER}/{config.COGNEE_LLM_MODEL}",
            "cognee_embedding": f"{config.COGNEE_EMBEDDING_PROVIDER}/{config.COGNEE_EMBEDDING_MODEL}",
            "search_type": config.COGNEE_SEARCH_TYPE,
            "answer_llm": f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            "top_k": config.TOP_K,
            "reset_between_samples": config.RESET_BETWEEN_SAMPLES,
        },
        "summary": summary,
        "distributions": {
            "f1": compute_score_distribution(results, "f1"),
            "bleu": compute_score_distribution(results, "bleu"),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Results saved: {output_path}")


# =============================================================================
# SUMMARY DISPLAY
# =============================================================================

def print_summary(results: List[Dict], experiment_name: str):
    """Print evaluation summary to console."""
    summary = aggregate_results(results)
    n = summary.get("count", 0)

    print("\n" + "=" * 60)
    print(f"COGNEE EVALUATION SUMMARY: {experiment_name}")
    print("=" * 60)

    print(f"\nFramework: Cognee")
    print(f"Samples evaluated: {n}")
    print(f"Configuration:")
    print(f"  Cognee LLM: {config.COGNEE_LLM_MODEL}")
    print(f"  Search Type: {config.COGNEE_SEARCH_TYPE}")
    print(f"  Answer LLM: {config.LLM_MODEL}")

    print("\n--- Accuracy Metrics ---")
    print(f"  {'Metric':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 40)

    for metric in ["em", "f1", "bleu", "llm"]:
        mean = summary.get(f"{metric}_mean", 0)
        std = summary.get(f"{metric}_std", 0)
        min_val = summary.get(f"{metric}_min", 0)
        max_val = summary.get(f"{metric}_max", 0)
        print(f"  {metric.upper():<8} {mean:>8.4f} {std:>8.4f} {min_val:>8.4f} {max_val:>8.4f}")

    print("\n--- Score Percentages ---")
    if "em_pct" in summary:
        print(f"  Exact Match: {summary['em_correct']}/{n} ({summary['em_pct']}%)")
    if "llm_pct" in summary:
        print(f"  LLM Correct: {summary['llm_correct']}/{n} ({summary['llm_pct']}%)")

    print("\n--- Latency (seconds) ---")
    print(f"  {'Component':<15} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 32)

    for metric, label in [
        ("total_time", "Total"),
        ("cognify_time", "Add+Cognify"),
        ("search_time", "Search"),
        ("answer_time", "Answer Gen"),
    ]:
        mean = summary.get(f"{metric}_mean", 0)
        std = summary.get(f"{metric}_std", 0)
        print(f"  {label:<15} {mean:>8.2f} {std:>8.2f}")

    if "total_time_mean" in summary:
        total_time = summary["total_time_mean"] * n
        print(f"\n  Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Cognee Memory Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--experiment_name", "-e",
        type=str,
        default=None,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        action="append",
        help="Evaluate specific sample index (can use multiple times)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated list of sample indices",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start of index range (inclusive). Use with --end for incremental runs.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End of index range (exclusive). Use with --start for incremental runs.",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default=None,
        choices=[
            "GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS",
            "SUMMARIES", "GRAPH_SUMMARY_COMPLETION", "GRAPH_COMPLETION_COT",
        ],
        help="Cognee search type to use",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        default=False,
        help="Disable prune between samples — Cognee knowledge graph accumulates across samples",
    )

    args = parser.parse_args()

    # Parse indices
    sample_indices = None
    if args.index:
        sample_indices = args.index
    elif args.indices:
        try:
            sample_indices = [int(x.strip()) for x in args.indices.split(",")]
        except ValueError:
            parser.error("--indices must be comma-separated integers")

    run_evaluation(
        num_samples=args.num_samples,
        experiment_name=args.experiment_name,
        sample_indices=sample_indices,
        start=args.start,
        end=args.end,
        search_type=args.search_type,
        no_reset=args.no_reset,
    )


if __name__ == "__main__":
    main()
