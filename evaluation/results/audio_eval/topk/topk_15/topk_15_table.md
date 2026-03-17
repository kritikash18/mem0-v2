# mem0 Evaluation Results
*Generated: 2026-03-15 10:09*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `15` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 100  | 59.0% | 0.74 | 0.698 | 85.0% |
| 200  | 58.0% | 0.742 | 0.695 | 85.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 100  | 21.78 | 20.57 | 0.33 | 0.88 |
| 200  | 21.45 | 20.32 | 0.32 | 0.81 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 100  | 59 (59.0%) | 64 (64.0%) | 80 (80.0%) | 16 (16.0%) |
| 200  | 116 (58.0%) | 127 (63.5%) | 160 (80.0%) | 31 (15.5%) |
