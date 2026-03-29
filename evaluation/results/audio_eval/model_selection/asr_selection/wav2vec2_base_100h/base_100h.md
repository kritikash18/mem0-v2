# mem0 Evaluation Results
*Generated: 2026-03-19 18:56*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `local/facebook/wav2vec2-base-100h` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 52.5% | 0.694 | 0.649 | 79.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 20.85 | 19.62 | 0.41 | 0.82 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 105 (52.5%) | 115 (57.5%) | 149 (74.5%) | 38 (19.0%) |
