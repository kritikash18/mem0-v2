# mem0 Evaluation Results
*Generated: 2026-03-19 09:10*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `assemblyai/nano` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 58.5% | 0.743 | 0.704 | 87.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 28.04 | 26.78 | 0.4 | 0.86 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 117 (58.5%) | 125 (62.5%) | 160 (80.0%) | 29 (14.5%) |
