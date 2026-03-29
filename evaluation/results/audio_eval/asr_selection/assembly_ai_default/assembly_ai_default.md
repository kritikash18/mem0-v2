# mem0 Evaluation Results
*Generated: 2026-03-19 09:09*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `assemblyai/default` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 57.5% | 0.735 | 0.694 | 87.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 28.1 | 26.76 | 0.45 | 0.89 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 115 (57.5%) | 123 (61.5%) | 159 (79.5%) | 30 (15.0%) |
