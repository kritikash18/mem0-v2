# mem0 Evaluation Results
*Generated: 2026-03-24 17:36*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `google_stt/latest_long` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 55.0% | 0.71 | 0.675 | 84.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 33.15 | 31.92 | 0.37 | 0.86 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 110 (55.0%) | 120 (60.0%) | 150 (75.0%) | 35 (17.5%) |
