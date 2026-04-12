# mem0 Evaluation Results
*Generated: 2026-03-22 21:52*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `ollama/llama3.2` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 48.0% | 0.589 | 0.551 | 70.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 35.99 | 34.72 | 0.45 | 0.83 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 96 (48.0%) | 99 (49.5%) | 126 (63.0%) | 63 (31.5%) |
