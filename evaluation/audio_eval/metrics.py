"""
Evaluation Metrics for Audio Memory Evaluation.

Metrics implemented:
- Exact Match (EM): Binary match after normalization
- F1 Score: Token-level precision/recall
- BLEU Score: N-gram overlap (BLEU-1)
- LLM Judge: GPT-based semantic correctness evaluation

Also includes aggregation utilities for computing summary statistics.
"""

import json
import logging
import re
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

from .prompts import BATCH_LLM_JUDGE_PROMPT, LLM_JUDGE_PROMPT

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Applies:
    - Lowercase conversion
    - Punctuation removal
    - Article removal (a, an, the)
    - Whitespace normalization
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    return " ".join(text.split()).strip()


# =============================================================================
# EXACT MATCH
# =============================================================================

def exact_match(prediction: str, ground_truth: str) -> int:
    """
    Compute Exact Match score.
    
    Returns 1 if normalized prediction equals normalized ground truth, else 0.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1 if exact match, 0 otherwise
    """
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)
    return int(norm_pred == norm_gt)


# =============================================================================
# F1 SCORE
# =============================================================================

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    # Handle edge cases
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # Count common tokens
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = sum((pred_counter & gt_counter).values())
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# BLEU SCORE
# =============================================================================

def bleu_score(prediction: str, ground_truth: str) -> float:
    """
    Compute BLEU-1 score (unigram precision with smoothing).
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        BLEU-1 score between 0.0 and 1.0
    """
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(ground_truth.lower())]
        
        if not pred_tokens or not ref_tokens[0]:
            return 0.0
        
        # BLEU-1: only unigrams, with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoothing
        )
        return score
        
    except ImportError:
        logger.warning("NLTK not available, using simple BLEU approximation")
        return _simple_bleu(prediction, ground_truth)
    except Exception as e:
        logger.warning(f"BLEU calculation error: {e}")
        return 0.0


def _simple_bleu(prediction: str, ground_truth: str) -> float:
    """Simple BLEU-1 approximation without NLTK."""
    pred_tokens = prediction.lower().split()
    ref_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens:
        return 0.0
    
    matches = sum(1 for t in pred_tokens if t in ref_tokens)
    return matches / len(pred_tokens)


# =============================================================================
# LLM JUDGE
# =============================================================================

def llm_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    model: str = "gpt-4o-mini",
    client: Optional[Any] = None
) -> int:
    """
    Evaluate answer correctness using LLM as judge.
    
    Uses GPT to determine if the prediction semantically matches the ground truth,
    with generous grading for paraphrases and format differences.
    
    Args:
        question: The question that was asked
        ground_truth: Expected correct answer
        prediction: Model's predicted answer
        model: OpenAI model to use for judging
        client: Optional pre-initialized OpenAI client
        
    Returns:
        1 if CORRECT, 0 if WRONG
    """
    if not prediction or not prediction.strip():
        return 0
    
    try:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        
        prompt = LLM_JUDGE_PROMPT.format(
            question=question,
            gold_answer=ground_truth,
            generated_answer=prediction
        )
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        
        # response_format is supported by OpenAI and Ollama but may fail on
        # some providers/models; fall back to raw parsing if it errors
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response — handle cases where model wraps it in markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        label = result.get("label", "").upper()
        return 1 if label == "CORRECT" else 0
        
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        return 0


# =============================================================================
# BATCH LLM JUDGE
# =============================================================================

def batch_llm_judge(
    samples: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    client: Optional[Any] = None,
    batch_size: int = 10,
) -> List[int]:
    """
    Judge a list of samples using batched LLM calls.

    Sends up to `batch_size` entries per API call instead of one call per
    sample, which reduces cost and latency significantly for large runs.

    Args:
        samples: List of dicts with keys "question", "ground_truth", "prediction".
                 Entries may also include an "idx" key for logging purposes.
        model: OpenAI-compatible model name
        client: Optional pre-initialised OpenAI client
        batch_size: Maximum number of entries per LLM call (10-20 recommended)

    Returns:
        List of integers (1=CORRECT, 0=WRONG) in the same order as ``samples``.
    """
    if not samples:
        return []

    if client is None:
        from openai import OpenAI
        client = OpenAI()

    labels: List[int] = [0] * len(samples)

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        # Format entries block
        entry_lines = []
        for local_idx, s in enumerate(batch, start=1):
            entry_lines.append(
                f"[{local_idx}]\n"
                f"Question: {s.get('question', '')}\n"
                f"Gold Answer: {s.get('ground_truth', '')}\n"
                f"Generated Answer: {s.get('prediction', '')}"
            )
        entries_block = "\n\n".join(entry_lines)

        prompt = BATCH_LLM_JUDGE_PROMPT.format(
            n=len(batch),
            entries=entries_block,
        )

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        try:
            try:
                kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content)
            batch_results = parsed.get("results", [])

            for item in batch_results:
                # "idx" in the response is 1-based within the batch
                local_idx = int(item.get("idx", 0)) - 1
                if 0 <= local_idx < len(batch):
                    global_idx = batch_start + local_idx
                    labels[global_idx] = 1 if str(item.get("label", "")).upper() == "CORRECT" else 0

        except Exception as e:
            logger.warning(
                f"Batch LLM judge error (batch starting at {batch_start}): {e}. "
                "Falling back to 0 for all entries in this batch."
            )

    return labels


# =============================================================================
# COMBINED METRICS
# =============================================================================

def compute_metrics(
    question: str,
    prediction: str,
    ground_truth: str,
    include_llm_judge: bool = True,
    openai_client: Optional[Any] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for a single prediction.
    
    Args:
        question: The question asked
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        include_llm_judge: Whether to run LLM judge (slower, costs API credits)
        openai_client: Pre-initialized OpenAI client for judge
        judge_model: Model name for LLM judge (defaults to gpt-4o-mini)
        
    Returns:
        Dictionary with em, f1, bleu, and optionally llm scores
    """
    metrics = {
        "em": exact_match(prediction, ground_truth),
        "f1": round(f1_score(prediction, ground_truth), 4),
        "bleu": round(bleu_score(prediction, ground_truth), 4),
    }
    
    if include_llm_judge:
        metrics["llm"] = llm_judge(
            question, ground_truth, prediction,
            model=judge_model or "gpt-4o-mini",
            client=openai_client,
        )
    
    return metrics


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across all evaluation results.
    
    Computes mean, std, min, max for each metric.
    
    Args:
        results: List of result dictionaries from evaluation
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {"count": 0}
    
    n = len(results)
    aggregated = {"count": n}
    
    # Metrics to aggregate
    metric_keys = [
        "em", "f1", "bleu", "llm",
        "total_time", "add_time", "search_time", "answer_time"
    ]
    
    for key in metric_keys:
        values = [r.get(key) for r in results if r.get(key) is not None]
        
        if not values:
            continue
            
        aggregated[f"{key}_mean"] = round(statistics.mean(values), 4)
        
        if len(values) > 1:
            aggregated[f"{key}_std"] = round(statistics.stdev(values), 4)
            aggregated[f"{key}_min"] = round(min(values), 4)
            aggregated[f"{key}_max"] = round(max(values), 4)
        else:
            aggregated[f"{key}_std"] = 0.0
            aggregated[f"{key}_min"] = values[0]
            aggregated[f"{key}_max"] = values[0]
    
    # Compute percentages for binary metrics
    if "em_mean" in aggregated:
        em_correct = sum(1 for r in results if r.get("em") == 1)
        aggregated["em_correct"] = em_correct
        aggregated["em_pct"] = round(100 * em_correct / n, 2)
    
    if "llm_mean" in aggregated:
        llm_correct = sum(1 for r in results if r.get("llm") == 1)
        aggregated["llm_correct"] = llm_correct
        aggregated["llm_pct"] = round(100 * llm_correct / n, 2)
    
    return aggregated


def compute_score_distribution(results: List[Dict[str, Any]], metric: str) -> Dict[str, int]:
    """
    Compute distribution of scores for a given metric.
    
    Args:
        results: List of result dictionaries
        metric: Metric name (e.g., 'f1', 'bleu')
        
    Returns:
        Dictionary with distribution statistics
    """
    values = [r.get(metric) for r in results if r.get(metric) is not None]
    
    if not values:
        return {}
    
    distribution = {
        "total": len(values),
        "zero": sum(1 for v in values if v == 0),
        "perfect": sum(1 for v in values if v == 1.0),
        "above_50": sum(1 for v in values if v >= 0.5),
        "above_80": sum(1 for v in values if v >= 0.8),
    }
    
    return distribution
