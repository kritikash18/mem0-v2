# mem0 Evaluation Results
*Generated: 2026-03-19 07:39*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `openai/gpt-4o-mini` |
| embedder | `huggingface/sentence-transformers/all-MiniLM-L6-v2` |
| judge | `openai/gpt-4o-mini` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 57.5% | 0.722 | 0.683 | 83.0% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 20.32 | 19.45 | 0.09 | 0.79 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 115 (57.5%) | 124 (62.0%) | 152 (76.0%) | 34 (17.0%) |
