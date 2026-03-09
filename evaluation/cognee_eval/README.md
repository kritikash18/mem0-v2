# Cognee Evaluation (Competitor Comparison)

Evaluation framework for testing [Cognee](https://docs.cognee.ai/) — a competing AI memory framework — on the same audio-based memory retrieval task used for mem0.

## Why Compare?

Cognee organizes data into a knowledge graph with embeddings, combining vector similarity with graph traversal for retrieval. This evaluation enables a direct, apples-to-apples comparison between two independent frameworks on the same raw audio input:

| | **mem0** | **Cognee** |
|---|---------|-----------|
| **Architecture** | Vector store + fact extraction | Knowledge graph + vector store + graph DB |
| **Memory Model** | Atomic facts with embeddings | Entities, relationships, summaries |
| **Retrieval** | Vector similarity + modality weighting | Graph traversal + vector similarity + LLM |
| **Search Modes** | Single (top-k vector search) | Multiple (GRAPH_COMPLETION, RAG, CHUNKS, etc.) |
| **Audio Handling** | Native ASR via mem0 ASR pipeline | Native audio ingestion via Cognee loaders |

## Pipeline

Each framework receives the same raw audio from the dataset and handles it through its own native pipeline:

```
Audio (.wav) ──→ cognee.add(wav_path) ──→ cognee.cognify()
                  (Cognee transcribes             ↓
                   internally)         Question ──→ cognee.search() ──→ LLM answer ──→ Metrics
```

Audio arrays from the HuggingFace dataset are written to a temporary `.wav` file, passed directly to `cognee.add()`, and deleted after ingestion. Cognee's built-in audio loader handles transcription natively — no external ASR is involved.

## Dataset

Same dataset as mem0 evaluation for fair comparison:
- **[Spoken SQuAD](https://huggingface.co/datasets/byteCode18/spoken-squad-1k-memory-eval)**: 1,000 samples
- Audio paragraphs (≤60s) with questions and short answers (≤10 tokens)

## Metrics

Same metrics as mem0 evaluation:

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| **EM** | Exact Match | 0 or 1 | Normalized string match |
| **F1** | Token-level | 0-1 | Precision/Recall harmonic mean |
| **BLEU** | N-gram | 0-1 | Unigram overlap (BLEU-1) |
| **LLM** | Semantic | 0 or 1 | GPT-4o-mini correctness judge |
| **Latency** | Time | seconds | Per-component breakdown |

### Latency Breakdown

Cognee evaluation tracks:
- **cognify_time**: `cognee.add()` + `cognee.cognify()` (includes Cognee's internal audio transcription + knowledge graph construction)
- **search_time**: `cognee.search()` (retrieval)
- **answer_time**: LLM answer generation
- **total_time**: End-to-end

## Prerequisites

```bash
# Install Cognee
pip install cognee

# Install shared dependencies
pip install datasets soundfile nltk openai tqdm jinja2

# Set API keys
export OPENAI_API_KEY="your-openai-key"
# Cognee reads LLM_API_KEY from env (defaults to OPENAI_API_KEY)
```

## Quick Start

```bash
# Quick test (10 samples)
python -m cognee_eval.evaluator --num_samples 10

# Standard run (100 samples)
python -m cognee_eval.evaluator --num_samples 100 --experiment_name cognee_100

# Test different search types
python -m cognee_eval.evaluator -n 50 -e cognee_graph --search-type GRAPH_COMPLETION
python -m cognee_eval.evaluator -n 50 -e cognee_rag --search-type RAG_COMPLETION
python -m cognee_eval.evaluator -n 50 -e cognee_chunks --search-type CHUNKS
```

## Cross-Framework Comparison

After running both mem0 and Cognee evaluations on the same sample count:

```bash
# Run mem0 evaluation
python -m audio_eval.evaluator -n 100 -e mem0_100

# Run Cognee evaluation
python -m cognee_eval.evaluator -n 100 -e cognee_100

# Compare results side-by-side
python -m cognee_eval.generate_scores \
  results/cognee_eval/cognee_100_final.json \
  results/audio_eval/mem0_100_final.json

# Export to CSV
python -m cognee_eval.generate_scores \
  results/cognee_eval/*_final.json \
  results/audio_eval/*_final.json \
  --output framework_comparison.csv
```

Expected comparison output:

```
==============================================================================
FRAMEWORK COMPARISON
==============================================================================

[COMPARISON TABLE]
------------------------------------------------------------------------------
 Framework        Experiment       N      EM      F1    BLEU     LLM  Time(s)
     cognee       cognee_100     100  0.4200  0.5900  0.5500  0.6800    8.50
      mem0          mem0_100     100  0.4500  0.6200  0.5800  0.7100    3.20
```

## Cognee Search Types

| Search Type | Description | When to Use |
|------------|-------------|-------------|
| `GRAPH_COMPLETION` | Graph-aware QA (default) | Best for relationship-based questions |
| `RAG_COMPLETION` | Standard RAG over chunks | Traditional retrieval-augmented generation |
| `CHUNKS` | Raw chunk retrieval | When you want direct text snippets |
| `SUMMARIES` | Summary-based search | For concise, high-level answers |
| `GRAPH_SUMMARY_COMPLETION` | Graph + summary | Tighter, summary-first responses |
| `GRAPH_COMPLETION_COT` | Chain-of-thought over graph | Complex multi-hop questions |

## Configuration

Edit `config.py` to change default settings:

```python
# Cognee settings
COGNEE_LLM_MODEL = "gpt-4o-mini"
COGNEE_SEARCH_TYPE = "GRAPH_COMPLETION"
```

## Output Format

Results are saved to `results/cognee_eval/{experiment}_final.json` with the same structure as mem0 results, enabling direct comparison.

## Project Structure

```
cognee_eval/
├── __init__.py         # Package exports
├── config.py           # Configuration (Cognee and LLM settings)
├── evaluator.py        # Main evaluation pipeline (native audio → Cognee)
├── prompts.py          # Answer generation and LLM judge prompts
├── metrics.py          # Reuses audio_eval metrics for identical scoring
├── generate_scores.py  # Score analysis + cross-framework comparison
├── Makefile            # Convenience commands
└── README.md
```
