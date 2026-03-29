# mem0 Evaluation Results
*Generated: 2026-03-15 12:25*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4o-mini` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| cleanup | `yes` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 100  | 59.0% | 0.731 | 0.691 | 85.0% |
| 200  | 58.5% | 0.735 | 0.693 | 85.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 100  | 20.96 | 19.77 | 0.31 | 0.88 |
| 200  | 20.43 | 19.28 | 0.3 | 0.85 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 100  | 59 (59.0%) | 63 (63.0%) | 79 (79.0%) | 17 (17.0%) |
| 200  | 117 (58.5%) | 127 (63.5%) | 158 (79.0%) | 33 (16.5%) |
