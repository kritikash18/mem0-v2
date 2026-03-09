# Local ASR Model Hosting

Setup and benchmarking utilities for running local ASR models (Wav2Vec2, HuBERT, Whisper) with the mem0 audio evaluation pipeline.

## Why Local Models?

- **No API costs** — Run unlimited evaluations without per-request charges
- **Reproducibility** — Fixed model weights ensure deterministic results
- **Latency control** — No network round-trips; inference time depends only on hardware
- **Privacy** — Audio data never leaves the machine

## Supported Models

| Model | Type | Size | Language | Description |
|-------|------|------|----------|-------------|
| `facebook/wav2vec2-base-960h` | wav2vec2 | ~360MB | en | Wav2Vec2 Base, LibriSpeech 960h |
| `facebook/wav2vec2-large-960h` | wav2vec2 | ~1.2GB | en | Wav2Vec2 Large, LibriSpeech 960h |
| `facebook/wav2vec2-large-960h-lv60-self` | wav2vec2 | ~1.2GB | en | Wav2Vec2 Large + LibriVox 60k self-training |
| `facebook/wav2vec2-large-robust-ft-libri-960h` | wav2vec2 | ~1.2GB | en | Wav2Vec2 Large Robust |
| `facebook/hubert-large-ls960-ft` | hubert | ~1.2GB | en | HuBERT Large, LibriSpeech 960h |
| `openai/whisper-tiny` | whisper | ~150MB | multi | Whisper Tiny (fastest) |
| `openai/whisper-base` | whisper | ~290MB | multi | Whisper Base |
| `openai/whisper-small` | whisper | ~960MB | multi | Whisper Small |
| `openai/whisper-medium` | whisper | ~3GB | multi | Whisper Medium |
| `openai/whisper-large-v3` | whisper | ~6GB | multi | Whisper Large V3 (best accuracy) |

## Prerequisites

```bash
pip install transformers torch librosa numpy
```

For GPU acceleration:
```bash
# CUDA (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon) — included by default with torch on macOS
```

## Quick Start

### 1. Download a Model

```bash
# Download Wav2Vec2 base (default)
python -m local_asr.setup_model

# Download a specific model
python -m local_asr.setup_model --model facebook/wav2vec2-large-960h

# Download and verify it works
python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --verify
```

### 2. Run Evaluation with Local Model

```bash
# Wav2Vec2 Base
python -m audio_eval.evaluator -n 10 \
  --asr-provider local \
  --asr-model facebook/wav2vec2-base-960h \
  --asr-model-type wav2vec2 \
  -e wav2vec2_base_test

# Wav2Vec2 Large
python -m audio_eval.evaluator -n 10 \
  --asr-provider local \
  --asr-model facebook/wav2vec2-large-960h \
  --asr-model-type wav2vec2 \
  -e wav2vec2_large_test

# HuBERT Large
python -m audio_eval.evaluator -n 10 \
  --asr-provider local \
  --asr-model facebook/hubert-large-ls960-ft \
  --asr-model-type hubert \
  -e hubert_test

# Local Whisper (compare against OpenAI API Whisper)
python -m audio_eval.evaluator -n 10 \
  --asr-provider local \
  --asr-model openai/whisper-base \
  --asr-model-type whisper \
  -e local_whisper_test
```

### 3. Compare Local vs API Models

```bash
# Run all configurations
python -m audio_eval.evaluator -n 50 -e whisper_api --asr-provider openai_whisper
python -m audio_eval.evaluator -n 50 -e wav2vec2_base --asr-provider local --asr-model facebook/wav2vec2-base-960h --asr-model-type wav2vec2
python -m audio_eval.evaluator -n 50 -e wav2vec2_large --asr-provider local --asr-model facebook/wav2vec2-large-960h --asr-model-type wav2vec2
python -m audio_eval.evaluator -n 50 -e whisper_local --asr-provider local --asr-model openai/whisper-base --asr-model-type whisper

# Compare results
python -m audio_eval.generate_scores results/audio_eval/*_final.json
```

## Utilities

### List Supported Models

```bash
python -m local_asr.setup_model --list
```

### Check Dependencies

```bash
python -m local_asr.setup_model --check-deps
```

### Benchmark Model Speed

```bash
python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --benchmark
```

## Model Type Reference

The `--asr-model-type` flag tells the local ASR provider which architecture to use:

| Type | Architecture | Models | Notes |
|------|-------------|--------|-------|
| `wav2vec2` | CTC (Connectionist Temporal Classification) | `facebook/wav2vec2-*` | No language model, direct character prediction |
| `hubert` | CTC | `facebook/hubert-*` | Similar to Wav2Vec2, different pre-training |
| `whisper` | Seq2Seq (Encoder-Decoder) | `openai/whisper-*` | Includes language model, better for noisy audio |

## Using via Makefile

From the `evaluation/` directory:

```bash
cd evaluation

# Quick local model tests
make -f audio_eval/Makefile run-wav2vec2-base
make -f audio_eval/Makefile run-wav2vec2-large
make -f audio_eval/Makefile run-hubert
make -f audio_eval/Makefile run-local-whisper
```
