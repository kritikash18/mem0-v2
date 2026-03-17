# mem0 Evaluation Results
*Generated: 2026-03-15 09:23*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 100  | 61.0% | 0.741 | 0.709 | 84.0% |
| 200  | 59.5% | 0.74 | 0.702 | 84.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 100  | 21.02 | 19.78 | 0.35 | 0.89 |
| 200  | 20.53 | 19.38 | 0.33 | 0.82 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 100  | 61 (61.0%) | 66 (66.0%) | 79 (79.0%) | 17 (17.0%) |
| 200  | 119 (59.5%) | 128 (64.0%) | 158 (79.0%) | 32 (16.0%) |
