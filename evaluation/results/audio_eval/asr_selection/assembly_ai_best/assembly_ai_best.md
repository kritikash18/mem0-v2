# mem0 Evaluation Results
*Generated: 2026-03-19 09:09*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `assemblyai/best` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 58.5% | 0.74 | 0.699 | 86.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 26.95 | 25.77 | 0.36 | 0.82 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 117 (58.5%) | 125 (62.5%) | 158 (79.0%) | 29 (14.5%) |
