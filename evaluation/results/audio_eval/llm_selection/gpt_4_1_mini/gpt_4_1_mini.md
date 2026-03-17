# mem0 Evaluation Results
*Generated: 2026-03-16 08:15*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4.1-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 57.0% | 0.749 | 0.701 | 89.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 19.64 | 18.51 | 0.33 | 0.8 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 114 (57.0%) | 126 (63.0%) | 158 (79.0%) | 23 (11.5%) |
