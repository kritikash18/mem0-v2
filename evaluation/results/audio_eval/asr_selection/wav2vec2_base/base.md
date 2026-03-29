# mem0 Evaluation Results
*Generated: 2026-03-19 18:57*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `local/facebook/wav2vec2-base-960h` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 57.5% | 0.735 | 0.696 | 82.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 20.18 | 19.0 | 0.37 | 0.81 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 115 (57.5%) | 124 (62.0%) | 155 (77.5%) | 29 (14.5%) |
