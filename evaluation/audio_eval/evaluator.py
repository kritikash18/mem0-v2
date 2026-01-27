"""
Audio Memory Evaluator.

Main evaluation pipeline for testing audio-based memory retrieval:
1. Load audio samples from HuggingFace dataset
2. Add audio to memory (ASR transcription + fact extraction)
3. Search memories with question
4. Generate answer from retrieved memories
5. Evaluate against ground truth

Usage:
    # Evaluate first 100 samples
    python -m audio_eval.evaluator --num_samples 100 --experiment_name my_test
    
    # Evaluate specific indices (multiple -i flags)
    python -m audio_eval.evaluator -i 0 -i 5 -i 10 --experiment_name debug_test
    
    # Evaluate specific indices (comma-separated)
    python -m audio_eval.evaluator --indices "0,5,10,15,20" --experiment_name selected_samples
    
    # Evaluate single sample
    python -m audio_eval.evaluator -i 42 --experiment_name sample_42
"""

import argparse
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

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
# DATASET LOADING
# =============================================================================

def load_evaluation_dataset(num_samples: Optional[int] = None):
    """
    Load dataset from HuggingFace.
    
    Args:
        num_samples: Number of samples to load (None for all)
        
    Returns:
        HuggingFace dataset with decoded audio
    """
    logger.info(f"Loading dataset: {config.DATASET_NAME}")
    
    ds = load_dataset(
        config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        token=config.HF_TOKEN
    )
    
    # Decode audio column
    ds = ds.cast_column(config.AUDIO_COLUMN, Audio(decode=True))
    
    # Limit samples if specified
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    
    logger.info(f"Loaded {len(ds)} samples")
    
    # Debug: Check audio format
    first_audio = ds[0][config.AUDIO_COLUMN]
    logger.info(f"Audio type: {type(first_audio)}")
    if hasattr(first_audio, 'array'):
        logger.info(f"Audio array shape: {first_audio.array.shape}")
        logger.info(f"Audio sampling rate: {first_audio.sampling_rate}")
    
    return ds


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def format_memories_for_prompt(memories: List[Dict]) -> str:
    """
    Format retrieved memories for the answer generation prompt.
    
    Args:
        memories: List of memory dictionaries from mem0 search
        
    Returns:
        Formatted string of memories
    """
    if not memories:
        return "No relevant memories found."
    
    formatted = []
    for i, mem in enumerate(memories, 1):
        memory_text = mem.get("memory", str(mem))
        score = mem.get("score", "N/A")
        formatted.append(f"{i}. {memory_text} (relevance: {score})")
    
    return "\n".join(formatted)


def generate_answer(
    question: str,
    memories: List[Dict],
    openai_client: OpenAI,
    model: str = None
) -> tuple:
    """
    Generate answer from retrieved memories using LLM.
    
    Args:
        question: The question to answer
        memories: List of retrieved memories
        openai_client: OpenAI client instance
        model: Model to use (defaults to config.LLM_MODEL)
        
    Returns:
        Tuple of (answer_text, generation_time_seconds)
    """
    if not memories:
        return "I don't know", 0.0
    
    model = model or config.LLM_MODEL
    
    # Format memories for prompt
    memories_text = format_memories_for_prompt(memories)
    
    # Render prompt with Jinja2
    template = Template(ANSWER_PROMPT)
    prompt = template.render(memories=memories_text, question=question)
    
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
    )
    
    generation_time = time.time() - start_time
    logger.info(f"[Answer Generation] Response: {response}")
    answer = response.choices[0].message.content.strip()
    return answer, generation_time


# =============================================================================
# SINGLE SAMPLE EVALUATION
# =============================================================================

def evaluate_sample(
    sample: Dict,
    idx: int,
    memory,
    openai_client: OpenAI
) -> Dict[str, Any]:
    """
    Evaluate a single audio sample through the full pipeline.
    
    Pipeline steps:
    1. Add audio to memory (triggers ASR + fact extraction)
    2. Search memories with question
    3. Generate answer from memories
    4. Compute evaluation metrics
    
    Args:
        sample: Dataset sample with audio, question, answer
        idx: Sample index
        memory: mem0 Memory instance
        openai_client: OpenAI client for answer generation
        
    Returns:
        Result dictionary with prediction, metrics, and timing info
    """
    # Create unique user_id to isolate this sample's memories
    user_id = f"eval_{idx}_{uuid.uuid4().hex[:8]}"
    
    # Extract data from sample (original columns)
    audio_raw = sample["context"]
    
    # Convert to dict format that mem0 expects: {"array": [...], "sampling_rate": int}
    # HuggingFace Audio uses dict-style access (even for AudioDecoder objects)
    try:
        audio = {
            "array": audio_raw["array"],
            "sampling_rate": audio_raw["sampling_rate"]
        }
    except (TypeError, KeyError) as e:
        raise ValueError(f"Cannot extract audio data from {type(audio_raw)}: {e}")
    
    question = sample[config.QUESTION_COLUMN]
    ground_truth = sample[config.ANSWER_COLUMN]
    
    # Debug: Verify audio format
    logger.debug(f"[Sample {idx}] Audio format: dict with keys {audio.keys()}")
    logger.debug(f"[Sample {idx}] Audio array shape: {audio['array'].shape}")
    logger.debug(f"[Sample {idx}] Question: {question[:50]}...")
    
    result = {
        "idx": idx,
        "user_id": user_id,
        "question": question,
        "ground_truth": ground_truth,
    }
    
    try:
        # Step 1: Add audio to memory
        # mem0 handles ASR transcription automatically via configured ASR provider
        add_start = time.time()
        add_result = memory.add(
            messages=[{"role": "user", "content": audio}],
            user_id=user_id,
            infer=config.INFER_MEMORIES,
        )
        add_time = time.time() - add_start
        
        # Extract added memories info if available
        num_memories_added = len(add_result.get("results", [])) if isinstance(add_result, dict) else 0
        
        # Step 2: Search memories with question
        search_start = time.time()
        search_result = memory.search(
            query=question,
            user_id=user_id,
            limit=config.TOP_K,
        )
        search_time = time.time() - search_start
        
        # Extract memories from search result
        if isinstance(search_result, dict):
            memories = search_result.get("results", [])
        else:
            memories = search_result if search_result else []
        
        logger.info(f"Memories: {memories}")
        # Step 3: Generate answer
        prediction, answer_time = generate_answer(
            question=question,
            memories=memories,
            openai_client=openai_client,
        )
        
        # Step 4: Compute metrics
        metrics = compute_metrics(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            include_llm_judge=True,
            openai_client=openai_client,
        )
        
        # Compile result
        result.update({
            "prediction": prediction,
            "num_memories_added": num_memories_added,
            "num_memories_retrieved": len(memories),
            "retrieved_memories": [
                {"memory": m.get("memory", str(m)), "score": m.get("score")}
                for m in memories
            ],
            "add_time": round(add_time, 3),
            "search_time": round(search_time, 3),
            "answer_time": round(answer_time, 3),
            "total_time": round(add_time + search_time + answer_time, 3),
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
        # Cleanup: delete memories for this user to avoid interference
        try:
            memory.delete_all(user_id=user_id)
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error for {user_id}: {cleanup_error}")
    
    return result


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

def run_evaluation(
    num_samples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
    asr_provider: Optional[str] = None,
    asr_model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run full evaluation pipeline.
    
    Args:
        num_samples: Number of samples to evaluate (None for all)
        experiment_name: Name for output files
        sample_indices: Specific sample indices to evaluate (overrides num_samples if provided)
        asr_provider: ASR provider to use (overrides config.ASR_PROVIDER)
        asr_model: ASR model to use (overrides config.ASR_MODEL)
        
    Returns:
        List of result dictionaries
    """
    from mem0 import Memory
    
    # Override config with runtime parameters
    if asr_provider:
        logger.info(f"Overriding ASR provider: {config.ASR_PROVIDER} → {asr_provider}")
        config.ASR_PROVIDER = asr_provider
    if asr_model:
        logger.info(f"Overriding ASR model: {config.ASR_MODEL} → {asr_model}")
        config.ASR_MODEL = asr_model
    
    # Validate configuration before starting
    config.validate_config()
    
    experiment_name = experiment_name or config.EXPERIMENT_NAME
    
    # Load dataset
    if sample_indices:
        # Load full dataset to access specific indices
        dataset = load_evaluation_dataset(None)
        # Validate indices
        max_idx = max(sample_indices)
        if max_idx >= len(dataset):
            raise ValueError(f"Index {max_idx} out of range. Dataset has {len(dataset)} samples.")
        indices_to_evaluate = sample_indices
        logger.info(f"Evaluating specific indices: {sample_indices}")
    else:
        dataset = load_evaluation_dataset(num_samples)
        indices_to_evaluate = list(range(len(dataset)))
    
    # Initialize mem0 Memory with configured components
    logger.info("Initializing mem0 Memory...")
    mem0_config = config.get_mem0_config()
    memory = Memory.from_config(mem0_config)
    
    # Initialize OpenAI client for answer generation (uses same API key)
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Run evaluation
    results = []
    total = len(indices_to_evaluate)
    
    logger.info(f"Starting evaluation: {total} samples")
    logger.info(f"Configuration: ASR={config.ASR_PROVIDER}/{config.ASR_MODEL}, LLM={config.LLM_MODEL}")
    
    for i, idx in enumerate(tqdm(indices_to_evaluate, desc="Evaluating")):
        sample = dataset[idx]
        
        result = evaluate_sample(
            sample=sample,
            idx=idx,
            memory=memory,
            openai_client=openai_client,
        )
        results.append(result)
        
        # Log progress
        if result.get("error"):
            logger.warning(f"[Sample {idx}] ERROR: {result['error'][:50]}")
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
    
    # Print summary
    print_summary(results, experiment_name)
    
    return results


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(
    results: List[Dict],
    experiment_name: str,
    suffix: str
):
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        experiment_name: Experiment name for filename
        suffix: File suffix (e.g., 'intermediate', 'final')
    """
    output_path = config.get_output_path(experiment_name, suffix)
    
    # Compute summary statistics
    summary = aggregate_results(results)
    
    # Build output data
    data = {
        "config": {
            "experiment": experiment_name,
            "dataset": config.DATASET_NAME,
            "asr": f"{config.ASR_PROVIDER}/{config.ASR_MODEL}",
            "llm": f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            "embedder": f"{config.EMBEDDER_PROVIDER}/{config.EMBEDDER_MODEL}",
            "top_k": config.TOP_K,
            "infer_memories": config.INFER_MEMORIES,
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
    """
    Print evaluation summary to console.
    
    Args:
        results: List of result dictionaries
        experiment_name: Name of the experiment
    """
    summary = aggregate_results(results)
    n = summary.get("count", 0)
    
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {experiment_name}")
    print("=" * 60)
    
    print(f"\nSamples evaluated: {n}")
    print(f"Configuration:")
    print(f"  ASR: {config.ASR_PROVIDER}/{config.ASR_MODEL}")
    print(f"  LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    print(f"  Top-K: {config.TOP_K}")
    
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
        ("add_time", "Memory Add"),
        ("search_time", "Search"),
        ("answer_time", "Answer Gen")
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
        description="Run Audio Memory Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--experiment_name", "-e",
        type=str,
        default=None,
        help="Name for this experiment run"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        action="append",
        help="Evaluate specific sample index (can be used multiple times, e.g., -i 5 -i 10 -i 15)"
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Evaluate specific sample indices as comma-separated list (e.g., '0,5,10,15')"
    )
    parser.add_argument(
        "--asr-provider",
        type=str,
        default=None,
        choices=["openai_whisper", "speech_recognition_google", "assemblyai", "google_stt", "local"],
        help="ASR provider to use (default: from config.py, usually openai_whisper)"
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help="ASR model to use (e.g., 'whisper-1' for OpenAI, 'best' or 'nano' for AssemblyAI)"
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
            parser.error("--indices must be a comma-separated list of integers")
    
    run_evaluation(
        num_samples=args.num_samples,
        experiment_name=args.experiment_name,
        sample_indices=sample_indices,
        asr_provider=args.asr_provider,
        asr_model=args.asr_model
    )


if __name__ == "__main__":
    main()
