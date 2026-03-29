# mem0 Evaluation Results
*Generated: 2026-03-24 17:37*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `google_stt/phone_call` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 56.0% | 0.716 | 0.682 | 83.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 33.24 | 32.02 | 0.35 | 0.87 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 112 (56.0%) | 124 (62.0%) | 151 (75.5%) | 36 (18.0%) |
