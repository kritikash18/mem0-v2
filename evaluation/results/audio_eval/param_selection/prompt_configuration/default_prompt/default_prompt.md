# mem0 Evaluation Results
*Generated: 2026-03-15 11:18*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| prompt | `default` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 3.5% | 0.042 | 0.04 | 5.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 9.24 | 8.85 | 0.32 | 0.06 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 7 (3.5%) | 7 (3.5%) | 8 (4.0%) | 190 (95.0%) |
