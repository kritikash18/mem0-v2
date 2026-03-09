# Cognee Memory Evaluation Makefile

.PHONY: help install quick run full compare clean

help:
	@echo "Cognee Memory Evaluation Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo ""
	@echo "Evaluation:"
	@echo "  make quick            Quick test (10 samples)"
	@echo "  make run              Standard run (100 samples)"
	@echo "  make full             Full evaluation (all samples)"
	@echo ""
	@echo "Search Type Comparisons:"
	@echo "  make run-graph        GRAPH_COMPLETION search"
	@echo "  make run-rag          RAG_COMPLETION search"
	@echo "  make run-chunks       CHUNKS search"
	@echo ""
	@echo "Cross-Framework:"
	@echo "  make compare-all      Compare Cognee + mem0 results"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove result files"

install:
	pip install cognee datasets soundfile nltk openai tqdm jinja2

# Evaluation runs
quick:
	python -m cognee_eval.evaluator --num_samples 10 --experiment_name cognee_quick

run:
	python -m cognee_eval.evaluator --num_samples 100 --experiment_name cognee_100

full:
	python -m cognee_eval.evaluator --experiment_name cognee_full

# Search type comparisons
run-graph:
	python -m cognee_eval.evaluator -n 50 -e cognee_graph --search-type GRAPH_COMPLETION

run-rag:
	python -m cognee_eval.evaluator -n 50 -e cognee_rag --search-type RAG_COMPLETION

run-chunks:
	python -m cognee_eval.evaluator -n 50 -e cognee_chunks --search-type CHUNKS

# Local ASR with Cognee
run-wav2vec2:
	python -m cognee_eval.evaluator -n 50 -e cognee_wav2vec2 \
		--asr-provider local --asr-model facebook/wav2vec2-base-960h --asr-model-type wav2vec2

# Compare Cognee experiments
compare:
	python -m cognee_eval.generate_scores results/cognee_eval/*_final.json

# Cross-framework comparison (Cognee vs mem0)
compare-all:
	python -m cognee_eval.generate_scores results/cognee_eval/*_final.json results/audio_eval/*_final.json

compare-csv:
	python -m cognee_eval.generate_scores results/cognee_eval/*_final.json results/audio_eval/*_final.json --output framework_comparison.csv

# Cleanup
clean:
	rm -rf results/cognee_eval/*.json
	rm -f framework_comparison.csv
