# mem0 Evaluation Results
*Generated: 2026-03-20 07:01*

## Configuration

| Parameter | Value |
|-----------|-------|
| framework | `mem0` |
| asr | `openai_whisper/whisper-1` |
| llm | `ollama/llama3.2:1b-instruct-q4_1` |
| embedder | `openai/text-embedding-3-small` |
| judge | `openai/gpt-4o-mini` |
| top_k | `10` |

## Accuracy

| N | EM (%) | F1 | BLEU | LLM Judge (%) |
|---|--------|----|------|---------------|
| 200  | 1.5% | 0.04 | 0.024 | 6.5% |

## Latency (mean per sample)

| N | Total (s) | Add (s) | Search (s) | Answer Gen (s) |
|---|-----------|---------|------------|----------------|
| 200  | 32.67 | 32.17 | 0.37 | 0.13 |

## F1 Score Distribution

| N | Perfect (F1=1) | Above 0.8 | Above 0.5 | Zero |
|---|---------------|-----------|-----------|------|
| 200  | 3 (1.5%) | 5 (2.5%) | 9 (4.5%) | 185 (92.5%) |
