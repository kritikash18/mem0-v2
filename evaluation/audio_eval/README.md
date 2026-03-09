# Audio Memory Evaluation

Evaluation framework for testing audio-based memory retrieval systems using the open-source mem0 library.

## Overview

This evaluation tests the pipeline:
```
Audio → Memory Ingestion → ASR Transcription → Transcription Cleanup → Memory Extraction → Memory Storage
                                    ↓
Question → Memory Search → Answer Generation → Evaluation Metrics
```

## Dataset

We use the [Spoken SQuAD](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test) dataset, specifically a curated subset:
- **1,000 samples** with audio paragraphs, questions, and answers
- Audio duration: ≤60 seconds per sample
- Answer length: ≤10 tokens (for objective evaluation)
- Dataset: `byteCode18/spoken-squad-1k-memory-eval`

### Dataset Structure

```
{
  "context": {          # Audio in HuggingFace format
    "array": [...],     # Audio samples as numpy array
    "sampling_rate": 16000
  },
  "instruction": "...", # Question text
  "answer": "..."       # Ground truth answer
}
```

## Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| **EM** | Exact Match | 0 or 1 | Normalized string match |
| **F1** | Token-level | 0-1 | Precision/Recall harmonic mean |
| **BLEU** | N-gram | 0-1 | Unigram overlap (BLEU-1) |
| **LLM** | Semantic | 0 or 1 | Configurable LLM correctness judge (default: GPT-4o-mini) |
| **Latency** | Time | seconds | Per-component and total |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install datasets soundfile nltk openai

# Quick test (10 samples, default: OpenAI Whisper)
python -m audio_eval.evaluator --num_samples 10

# Full evaluation (100 samples)
python -m audio_eval.evaluator --num_samples 100 --experiment_name my_experiment

# Use different ASR at runtime (overrides config.py)
python -m audio_eval.evaluator --num_samples 10 --asr-provider assemblyai --asr-model best

# Use Google Cloud Speech-to-Text v1
python -m audio_eval.evaluator -n 10 --asr-provider google_stt --asr-model default

# Use Ollama with Llama 3.2
python -m audio_eval.evaluator -n 10 --llm-provider ollama --llm-model llama3.2

# Use Ollama for everything (answer gen + judge)
python -m audio_eval.evaluator -n 10 \
  --llm-provider ollama --llm-model llama3.2 \
  --judge-provider ollama --judge-model qwen2.5

# Persistent memory mode (no per-sample cleanup)
python -m audio_eval.evaluator -n 50 --no-cleanup -e persistent_test

# Compare ASR providers
python -m audio_eval.evaluator -n 50 -e whisper_test --asr-provider openai_whisper
python -m audio_eval.evaluator -n 50 -e assembly_test --asr-provider assemblyai --asr-model best
python -m audio_eval.evaluator -n 50 -e google_test --asr-provider google_stt --asr-model default

# Compare LLM providers
python -m audio_eval.evaluator -n 50 -e gpt4_test --llm-provider openai --llm-model gpt-4o-mini
python -m audio_eval.evaluator -n 50 -e llama_test --llm-provider ollama --llm-model llama3.2
```

### ASR Runtime Selection

You can specify the ASR provider at runtime without editing `config.py`:

```bash
# Use AssemblyAI instead of default Whisper
python -m audio_eval.evaluator --asr-provider assemblyai --asr-model best -n 10

# Use local Whisper model
python -m audio_eval.evaluator --asr-provider local --asr-model openai/whisper-large-v3
```

**Supported ASR Providers:**
- `openai_whisper` (default) - OpenAI Whisper API
  - Models: `whisper-1`
  - Requires: `OPENAI_API_KEY`
- `speech_recognition_google` - Free Google Speech Recognition (via `speech_recognition` library)
  - **No API key required!**
  - Simple, easy to set up
  - Limited to ~50 requests/day per IP
  - Good for testing/prototyping
- `assemblyai` - AssemblyAI transcription service
  - Models: `best`, `nano`
  - Requires: `ASSEMBLYAI_API_KEY`
- `google_stt` - Google Cloud Speech-to-Text (v1 API)
  - Models: `default`, `phone_call`, `video`, `command_and_search`
  - Requires: Google Cloud credentials
  - **Note**: Requires explicit language code (e.g., "en-US"), defaults to "en-US" if `ASR_LANGUAGE=None`
- `local` - Local Whisper models via Hugging Face
  - Models: `openai/whisper-base`, `openai/whisper-large-v3`
  - No API key needed

**API Keys:**
```bash
# OpenAI (for Whisper, LLM, embeddings)
export OPENAI_API_KEY="your-openai-key"

# AssemblyAI (optional, only if using AssemblyAI)
export ASSEMBLYAI_API_KEY="your-assemblyai-key"

# Google Cloud (optional, only if using Google STT)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

## Configuration

Edit `config.py` to change defaults, or use CLI flags to override at runtime.

```python
# ASR
ASR_PROVIDER = "openai_whisper"  # or "local", "assemblyai", "google_stt"
ASR_MODEL = "whisper-1"
ASR_MODEL_TYPE = "wav2vec2"      # For local ASR: "wav2vec2", "hubert", "whisper"

# LLM (used for ASR cleanup, fact extraction, and answer generation)
LLM_PROVIDER = "openai"          # or "ollama", "anthropic", "groq", "together"
LLM_MODEL = "gpt-4o-mini"
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM Judge (separate from the main LLM — controls evaluation scoring)
LLM_JUDGE_PROVIDER = None        # None = same as LLM_PROVIDER
LLM_JUDGE_MODEL = None           # None = "gpt-4o-mini". E.g., "qwen2.5", "llama3.2"

# Embedder
EMBEDDER_PROVIDER = "openai"
EMBEDDER_MODEL = "text-embedding-3-small"

# Memory
TOP_K = 10
INFER_MEMORIES = True                     # Extract facts vs raw transcript
USE_READING_COMPREHENSION_PROMPT = True   # Custom RC prompt vs default
CLEANUP_AFTER_SAMPLE = True               # True = isolated, False = persistent memory
```

### CLI Reference

All `config.py` settings can be overridden at the command line:

```
usage: python -m audio_eval.evaluator [-h] [-n NUM_SAMPLES] [-e EXPERIMENT_NAME]
                                      [-i INDEX] [--indices INDICES]
                                      [--asr-provider {openai_whisper,speech_recognition_google,assemblyai,google_stt,local}]
                                      [--asr-model ASR_MODEL]
                                      [--asr-model-type {wav2vec2,hubert,whisper}]
                                      [--llm-provider {openai,anthropic,ollama,groq,together}]
                                      [--llm-model LLM_MODEL]
                                      [--judge-provider {openai,ollama,anthropic,groq,together}]
                                      [--judge-model JUDGE_MODEL]
                                      [--no-cleanup]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-n`, `--num_samples` | Number of samples to evaluate | All |
| `-e`, `--experiment_name` | Name for output files | `"default"` |
| `-i`, `--index` | Evaluate a specific sample index (repeatable) | — |
| `--indices` | Comma-separated sample indices (e.g., `"0,5,10"`) | — |
| `--asr-provider` | ASR provider | `openai_whisper` |
| `--asr-model` | ASR model name | `whisper-1` |
| `--asr-model-type` | Architecture for local ASR | `wav2vec2` |
| `--llm-provider` | LLM provider for cleanup, extraction, and answer generation | `openai` |
| `--llm-model` | LLM model name | `gpt-4o-mini` |
| `--judge-provider` | LLM provider for the evaluation judge | Same as `--llm-provider` |
| `--judge-model` | LLM model for the evaluation judge | `gpt-4o-mini` |
| `--no-cleanup` | Disable per-sample memory deletion (persistent mode) | Off (isolated) |

### Custom Memory Extraction Prompt

By default, we use a **custom reading comprehension prompt** optimized for extracting factual information from audio passages (instead of mem0's default conversational prompt).

**Why?** Our dataset is reading comprehension style (like SQuAD), not conversational. The custom prompt:
- Extracts definitions, entities, dates, and relationships
- Breaks content into atomic, searchable facts
- Preserves numbers, names, and specific details
- Focuses on content rather than "user preferences"
- **Auto-consolidates to max 30 facts** for long audio (merges related facts, prevents memory explosion)

**Example:**
```
Audio: "The UMC was founded in 1968..."
Custom: ["UMC stands for United Methodist Church", "UMC was founded in 1968"]
Default: ["User talking about UMC", "Mentioned founding"]
```

See [`CUSTOM_PROMPT_GUIDE.md`](CUSTOM_PROMPT_GUIDE.md) for detailed comparison and usage.

**Test the difference:**
```bash
python -m audio_eval.test_prompt_comparison
```

## Score Generation & Analysis

### Single Experiment - Detailed Summary

```bash
python -m audio_eval.generate_scores results/audio_eval/default_final.json
```

Output includes descriptive statistics, score distributions, and sample previews:

```
================================================================================
DETAILED EVALUATION SUMMARY
================================================================================

[CONFIGURATION]
----------------------------------------
  Experiment:     openai_whisper_eval
  Dataset:        byteCode18/spoken-squad-1k-memory-eval
  ASR:            openai_whisper/whisper-1
  LLM:            openai/gpt-4o-mini
  Top-K:          5
  Total Samples:  100

[ACCURACY METRICS - DESCRIPTIVE STATISTICS]
--------------------------------------------------------------------------------
            em       f1     bleu      llm
count   100.0   100.00   100.00   100.00
mean      0.45     0.62     0.58     0.71
std       0.50     0.28     0.31     0.45
min       0.00     0.00     0.00     0.00
25%       0.00     0.42     0.35     0.00
50%       0.00     0.67     0.62     1.00
75%       1.00     0.85     0.82     1.00
max       1.00     1.00     1.00     1.00

[OVERALL MEAN SCORES]
----------------------------------------
  EM     : 0.4500 (±0.5000)
  F1     : 0.6200 (±0.2800)
  BLEU   : 0.5800 (±0.3100)
  LLM    : 0.7100 (±0.4500)

[SCORE DISTRIBUTION]
----------------------------------------
  Exact Match:  45/100 (45.0%)
  LLM Correct:  71/100 (71.0%)
  F1 = 1.0:     32/100 (32.0%)
  F1 >= 0.5:    68/100 (68.0%)

[LATENCY METRICS (seconds)]
--------------------------------------------------------------------------------
       total_time  add_time  search_time  answer_time
count     100.000   100.000      100.000      100.000
mean        3.200     2.100        0.450        0.650
std         0.800     0.600        0.120        0.200
min         1.800     1.200        0.280        0.350
max         5.200     3.800        0.850        1.200

  Total evaluation time: 320.00s (5.33 min)

[SAMPLE RESULTS PREVIEW]
--------------------------------------------------------------------------------
idx                                  question        ground_truth          prediction  em    f1  llm
  0  What did the main character study in co...  computer science       computer sc...   1  1.00    1
  1  Where was the conference held last year...          New York           New York...   1  1.00    1
  2  How many people attended the event acco...               500         around 500...   0  0.67    1
  3  What is the name of the professor menti...       Dr. Johnson        Dr. Johnson...   1  1.00    1
  4  When did the company first start operat...              1995                2005...   0  0.00    0

================================================================================
```

### Compare Multiple Experiments

```bash
python -m audio_eval.generate_scores results/audio_eval/*_final.json
```

```
====================================================================================================
EXPERIMENT COMPARISON
====================================================================================================

[COMPARISON TABLE]
----------------------------------------------------------------------------------------------------
          Experiment             ASR             LLM    N      EM      F1    BLEU     LLM  Time(s)
    openai_whisper_eval       whisper-1      gpt-4o-mini  100  0.4500  0.6200  0.5800  0.7100     3.20
     local_whisper_eval     whisper-base     gpt-4o-mini  100  0.4200  0.5800  0.5400  0.6800     2.10
       assemblyai_eval            best      gpt-4o-mini  100  0.4300  0.6000  0.5600  0.6900     4.50

[BEST SCORES]
----------------------------------------
  Best EM:   openai_whisper_eval (0.4500)
  Best F1:   openai_whisper_eval (0.6200)
  Best LLM:  openai_whisper_eval (0.7100)
  Fastest:   local_whisper_eval (2.10s)

====================================================================================================
```

### Export to CSV

```bash
# Export comparison summary
python -m audio_eval.generate_scores results/audio_eval/*.json --output comparison.csv

# Export detailed per-sample results (single file only)
python -m audio_eval.generate_scores results/audio_eval/default_final.json --detailed-csv detailed_results.csv
```

## Output Format

Results saved to `results/audio_eval/{experiment}_final.json`:

```json
{
  "config": {
    "framework": "mem0",
    "experiment": "my_experiment",
    "dataset": "byteCode18/spoken-squad-1k-memory-eval",
    "asr": "openai_whisper/whisper-1",
    "llm": "openai/gpt-4o-mini",
    "judge": "ollama/qwen2.5",
    "embedder": "openai/text-embedding-3-small",
    "top_k": 10,
    "infer_memories": true,
    "cleanup_after_sample": true
  },
  "summary": {
    "count": 100,
    "em_mean": 0.4500,
    "em_std": 0.5000,
    "f1_mean": 0.6200,
    "f1_std": 0.2800,
    "bleu_mean": 0.5800,
    "bleu_std": 0.3100,
    "llm_mean": 0.7100,
    "llm_std": 0.4500,
    "total_time_mean": 3.20,
    "add_time_mean": 2.10,
    "search_time_mean": 0.45,
    "answer_time_mean": 0.65
  },
  "results": [
    {
      "idx": 0,
      "question": "What did the author study?",
      "ground_truth": "computer science",
      "prediction": "computer science",
      "num_memories": 3,
      "em": 1,
      "f1": 1.0,
      "bleu": 1.0,
      "llm": 1,
      "total_time": 3.45,
      "add_time": 2.30,
      "search_time": 0.42,
      "answer_time": 0.73
    }
  ]
}
```

## Project Structure

```
audio_eval/
├── config.py          # Configuration settings
├── metrics.py         # Evaluation metrics (EM, F1, BLEU, LLM Judge)
├── evaluator.py       # Main evaluation pipeline
├── generate_scores.py # Score aggregation and comparison
├── __init__.py
├── Makefile
└── README.md
```

## File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | All configuration in one place - ASR, LLM, embedder, output settings |
| `metrics.py` | Metric functions - exact_match, f1_score, bleu_score, llm_judge |
| `evaluator.py` | Main evaluation logic - load data, run pipeline, save results |
| `generate_scores.py` | Compare experiments, generate tables, export CSV |
