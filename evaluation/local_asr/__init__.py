"""
Local ASR Model Hosting for Audio Memory Evaluation.

Provides utilities to download, verify, and benchmark local ASR models
(Wav2Vec2, HuBERT, Whisper) for use with the mem0 evaluation pipeline.

Usage:
    # Download and verify a model
    python -m local_asr.setup_model --model facebook/wav2vec2-base-960h

    # Run evaluation with local Wav2Vec2
    python -m audio_eval.evaluator -n 10 --asr-provider local --asr-model facebook/wav2vec2-base-960h --asr-model-type wav2vec2
"""

from .setup_model import download_model, verify_model, list_models, SUPPORTED_MODELS
