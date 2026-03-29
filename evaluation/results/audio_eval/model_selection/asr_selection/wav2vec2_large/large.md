# mem0 Evaluation Results
*Generated: 2026-03-19 18:58*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `local/facebook/wav2vec2-large-960h` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 55.0% | 0.71 | 0.674 | 82.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 23.13 | 21.89 | 0.42 | 0.82 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 110 (55.0%) | 121 (60.5%) | 151 (75.5%) | 35 (17.5%) |
