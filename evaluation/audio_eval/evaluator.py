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

    # Incremental runs — run 0-99, then build on it with 100-499, etc.
    python -m audio_eval.evaluator --start 0   --end 100  --experiment_name run_0_100
    python -m audio_eval.evaluator --start 100 --end 500  --experiment_name run_100_500
    # Then merge: python -m audio_eval.combine_results results/run_0_100_final.json results/run_100_500_final.json -o results/merged.json

    # Skip LLM judge during evaluation, score later in one batch
    python -m audio_eval.evaluator -n 100 --skip-judge --experiment_name no_judge_run
    python -m audio_eval.run_judge results/audio_eval/no_judge_run_final.json

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

    When USE_AUDIO_QUERY is enabled, loads from AUDIO_QUERY_DATASET (v2) which
    contains the spoken instruction_v2 column, and decodes both audio columns.

    Args:
        num_samples: Number of samples to load (None for all)

    Returns:
        HuggingFace dataset with decoded audio
    """
    dataset_name = (
        config.AUDIO_QUERY_DATASET if config.USE_AUDIO_QUERY
        else config.DATASET_NAME
    )
    logger.info(f"Loading dataset: {dataset_name}")

    ds = load_dataset(
        dataset_name,
        split=config.DATASET_SPLIT,
        token=config.HF_TOKEN
    )

    # Always decode the context audio column
    ds = ds.cast_column(config.AUDIO_COLUMN, Audio(decode=True))

    # Decode the spoken query column when audio queries are enabled
    if config.USE_AUDIO_QUERY:
        ds = ds.cast_column(config.AUDIO_QUERY_COLUMN, Audio(decode=True))
        logger.info(f"Audio query mode: using '{config.AUDIO_QUERY_COLUMN}' column for search")

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
    openai_client: OpenAI,
    judge_client: OpenAI = None,
    judge_model: str = None,
    include_llm_judge: bool = True,
    use_audio_query: bool = False,
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
        openai_client: OpenAI-compatible client for answer generation
        judge_client: OpenAI-compatible client for LLM judge (defaults to openai_client)
        judge_model: Model name for LLM judge (defaults to gpt-4o-mini)
        
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

    # Build the search query — either spoken audio or plain text
    if use_audio_query:
        query_raw = sample[config.AUDIO_QUERY_COLUMN]
        search_query = {
            "array": query_raw["array"],
            "sampling_rate": query_raw["sampling_rate"],
        }
        logger.debug(f"[Sample {idx}] Query mode: audio ({config.AUDIO_QUERY_COLUMN})")
    else:
        search_query = question

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
        
        # Step 2: Search memories — text question or spoken audio query
        search_start = time.time()
        search_result = memory.search(
            query=search_query,
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
            include_llm_judge=include_llm_judge,
            openai_client=judge_client or openai_client,
            judge_model=judge_model,
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
        if config.CLEANUP_AFTER_SAMPLE:
            try:
                memory.delete_all(user_id=user_id)
            except Exception as cleanup_error:
                logger.debug(f"Cleanup error for {user_id}: {cleanup_error}")
        else:
            logger.debug(f"Skipping cleanup for {user_id} (CLEANUP_AFTER_SAMPLE=False)")
    
    return result


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

def run_evaluation(
    num_samples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    asr_provider: Optional[str] = None,
    asr_model: Optional[str] = None,
    asr_model_type: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    cleanup_after_sample: Optional[bool] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    skip_judge: bool = False,
    qdrant_path: Optional[str] = None,
    audio_query: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full evaluation pipeline.

    Index selection priority (highest to lowest):
        sample_indices > start/end range > num_samples (first N) > all

    Args:
        num_samples: Evaluate first N samples
        experiment_name: Name for output files
        sample_indices: Explicit list of dataset indices to evaluate
        start: Start of index range (inclusive); use with end for incremental runs
        end: End of index range (exclusive); use with start for incremental runs
        asr_provider: ASR provider override
        asr_model: ASR model override
        asr_model_type: Model architecture for local ASR
        llm_provider: LLM provider override
        llm_model: LLM model override
        cleanup_after_sample: Per-sample memory cleanup toggle
        judge_provider: LLM judge provider override
        judge_model: LLM judge model override
        skip_judge: Skip LLM judge (set llm=None); run judge later with run_judge.py
        qdrant_path: Override Qdrant local storage path; required when running two
                     experiments in parallel to avoid the file lock conflict
        audio_query: Use spoken audio (instruction_v2) as search query instead of
                     text; loads from AUDIO_QUERY_DATASET automatically

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
    if asr_model_type:
        logger.info(f"Overriding ASR model type: {config.ASR_MODEL_TYPE} → {asr_model_type}")
        config.ASR_MODEL_TYPE = asr_model_type
    if llm_provider:
        logger.info(f"Overriding LLM provider: {config.LLM_PROVIDER} → {llm_provider}")
        config.LLM_PROVIDER = llm_provider
    if llm_model:
        logger.info(f"Overriding LLM model: {config.LLM_MODEL} → {llm_model}")
        config.LLM_MODEL = llm_model
    if cleanup_after_sample is not None:
        logger.info(f"Overriding cleanup: {config.CLEANUP_AFTER_SAMPLE} → {cleanup_after_sample}")
        config.CLEANUP_AFTER_SAMPLE = cleanup_after_sample
    if judge_provider:
        logger.info(f"Overriding judge provider: {config.LLM_JUDGE_PROVIDER} → {judge_provider}")
        config.LLM_JUDGE_PROVIDER = judge_provider
    if judge_model:
        logger.info(f"Overriding judge model: {config.LLM_JUDGE_MODEL} → {judge_model}")
        config.LLM_JUDGE_MODEL = judge_model
    if qdrant_path:
        logger.info(f"Overriding Qdrant path: {config.QDRANT_PATH} → {qdrant_path}")
        config.QDRANT_PATH = qdrant_path
    if audio_query:
        config.USE_AUDIO_QUERY = True

    # Validate configuration before starting
    config.validate_config()
    
    experiment_name = experiment_name or config.EXPERIMENT_NAME
    
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
    
    # Initialize mem0 Memory with configured components
    logger.info("Initializing mem0 Memory...")
    mem0_config = config.get_mem0_config()
    memory = Memory.from_config(mem0_config)
    
    # Initialize LLM client for answer generation
    # Ollama exposes an OpenAI-compatible API at /v1, so we reuse the OpenAI client
    if config.LLM_PROVIDER == "ollama":
        openai_client = OpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
    else:
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # LLM Judge client — can use a different provider/model from answer generation
    judge_provider = config.LLM_JUDGE_PROVIDER or config.LLM_PROVIDER
    judge_model = config.LLM_JUDGE_MODEL or "gpt-4o-mini"
    
    if judge_provider == "ollama":
        judge_client = OpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
    elif judge_provider == config.LLM_PROVIDER and config.LLM_PROVIDER != "ollama":
        judge_client = openai_client
    else:
        judge_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    logger.info(f"Judge: {judge_provider}/{judge_model}")
    
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
            judge_client=judge_client,
            judge_model=judge_model,
            include_llm_judge=not skip_judge,
            use_audio_query=config.USE_AUDIO_QUERY,
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
            "framework": "mem0",
            "experiment": experiment_name,
            "dataset": config.DATASET_NAME,
            "asr": f"{config.ASR_PROVIDER}/{config.ASR_MODEL}",
            "llm": f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            "embedder": f"{config.EMBEDDER_PROVIDER}/{config.EMBEDDER_MODEL}",
            "judge": f"{config.LLM_JUDGE_PROVIDER or config.LLM_PROVIDER}/{config.LLM_JUDGE_MODEL or 'gpt-4o-mini'}",
            "top_k": config.TOP_K,
            "infer_memories": config.INFER_MEMORIES,
            "cleanup_after_sample": config.CLEANUP_AFTER_SAMPLE,
            "query_mode": "audio" if config.USE_AUDIO_QUERY else "text",
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
        "--start",
        type=int,
        default=None,
        help="Start of index range (inclusive). Use with --end for incremental runs, e.g. --start 100 --end 500"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End of index range (exclusive). Use with --start for incremental runs."
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        default=False,
        help="Skip LLM judge during evaluation (saves cost). Run scoring later with run_judge.py"
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
        help="ASR model to use (e.g., 'whisper-1' for OpenAI, 'best' or 'nano' for AssemblyAI, 'facebook/wav2vec2-base-960h' for local)"
    )
    parser.add_argument(
        "--asr-model-type",
        type=str,
        default=None,
        choices=["wav2vec2", "hubert", "whisper"],
        help="Model architecture type for local ASR (e.g., 'wav2vec2', 'hubert', 'whisper'). Only used with --asr-provider local"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "ollama", "groq", "together"],
        help="LLM provider to use (default: from config.py, usually openai)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use (e.g., 'gpt-4o-mini', 'llama3.2', 'claude-3-5-sonnet-20241022')"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Disable per-sample memory cleanup (memories accumulate across samples)"
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["openai", "ollama", "anthropic", "groq", "together"],
        help="LLM provider for the judge (default: same as --llm-provider)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM model for the judge (default: gpt-4o-mini). E.g., 'qwen2.5', 'llama3.2'"
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help=(
            "Override Qdrant local storage path (default: ./qdrant_data). "
            "Set a different path per terminal when running two experiments in parallel "
            "to avoid the Qdrant file lock, e.g. --qdrant-path ./qdrant_data_gpt4o"
        )
    )
    parser.add_argument(
        "--audio-query",
        action="store_true",
        default=False,
        help=(
            "Use spoken audio (instruction_v2) as the search query instead of text. "
            "Automatically loads from byteCode18/spoken-squad-memory-eval-v2. "
            "mem0 transcribes the audio query via the configured ASR before searching."
        )
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
        start=args.start,
        end=args.end,
        asr_provider=args.asr_provider,
        asr_model=args.asr_model,
        asr_model_type=args.asr_model_type,
        llm_provider=args.llm_provider,
        qdrant_path=args.qdrant_path,
        llm_model=args.llm_model,
        cleanup_after_sample=not args.no_cleanup,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        skip_judge=args.skip_judge,
        audio_query=args.audio_query,
    )


if __name__ == "__main__":
    main()
