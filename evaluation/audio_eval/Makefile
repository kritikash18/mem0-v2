# Audio Memory Evaluation Makefile

.PHONY: help install quick run full compare clean

help:
	@echo "Audio Memory Evaluation Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo ""
	@echo "Evaluation:"
	@echo "  make quick         Quick test (10 samples)"
	@echo "  make run           Standard run (100 samples)"
	@echo "  make full          Full evaluation (all samples)"
	@echo ""
	@echo "Analysis:"
	@echo "  make compare       Compare all experiments"
	@echo "  make compare-csv   Export comparison to CSV"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove result files"

install:
	pip install datasets soundfile nltk openai tqdm

# Evaluation runs
quick:
	python -m audio_eval.evaluator --num_samples 10 --experiment_name quick_test

run:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name eval_100

full:
	python -m audio_eval.evaluator --experiment_name full_eval

# Different ASR configurations
run-openai-whisper:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name openai_whisper

run-local-whisper:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name local_whisper \
		--asr-provider local --asr-model openai/whisper-base --asr-model-type whisper

run-wav2vec2-base:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name wav2vec2_base \
		--asr-provider local --asr-model facebook/wav2vec2-base-960h --asr-model-type wav2vec2

run-wav2vec2-large:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name wav2vec2_large \
		--asr-provider local --asr-model facebook/wav2vec2-large-960h --asr-model-type wav2vec2

run-hubert:
	python -m audio_eval.evaluator --num_samples 100 --experiment_name hubert_large \
		--asr-provider local --asr-model facebook/hubert-large-ls960-ft --asr-model-type hubert

# Analysis
compare:
	python -m audio_eval.generate_scores results/audio_eval/*_final.json

compare-csv:
	python -m audio_eval.generate_scores results/audio_eval/*_final.json --output comparison.csv

# Show single result
show:
	@echo "Usage: python -m audio_eval.generate_scores results/audio_eval/<experiment>_final.json"

# Cleanup
clean:
	rm -rf results/audio_eval/*.json
	rm -f comparison.csv
