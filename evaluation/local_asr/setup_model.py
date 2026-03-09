"""
Setup and verify local ASR models for evaluation.

Downloads HuggingFace models and runs a quick sanity check to ensure
they can transcribe audio before running the full evaluation pipeline.

Usage:
    # Download default Wav2Vec2 model
    python -m local_asr.setup_model

    # Download a specific model
    python -m local_asr.setup_model --model facebook/wav2vec2-large-960h

    # List supported models
    python -m local_asr.setup_model --list

    # Verify a model with a test transcription
    python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --verify

    # Benchmark a model on a short audio sample
    python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --benchmark
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: Dict[str, Dict] = {
    # Wav2Vec2 family (CTC-based)
    "facebook/wav2vec2-base-960h": {
        "type": "wav2vec2",
        "size": "~360MB",
        "description": "Wav2Vec2 Base, trained on LibriSpeech 960h",
        "language": "en",
        "pipeline": True,
    },
    "facebook/wav2vec2-large-960h": {
        "type": "wav2vec2",
        "size": "~1.2GB",
        "description": "Wav2Vec2 Large, trained on LibriSpeech 960h",
        "language": "en",
        "pipeline": True,
    },
    "facebook/wav2vec2-large-960h-lv60-self": {
        "type": "wav2vec2",
        "size": "~1.2GB",
        "description": "Wav2Vec2 Large with self-training on LibriVox 60k hours",
        "language": "en",
        "pipeline": True,
    },
    "facebook/wav2vec2-large-robust-ft-libri-960h": {
        "type": "wav2vec2",
        "size": "~1.2GB",
        "description": "Wav2Vec2 Large Robust, fine-tuned on LibriSpeech 960h",
        "language": "en",
        "pipeline": True,
    },
    # HuBERT family (CTC-based)
    "facebook/hubert-large-ls960-ft": {
        "type": "hubert",
        "size": "~1.2GB",
        "description": "HuBERT Large, fine-tuned on LibriSpeech 960h",
        "language": "en",
        "pipeline": True,
    },
    # Local Whisper models (Seq2Seq)
    "openai/whisper-tiny": {
        "type": "whisper",
        "size": "~150MB",
        "description": "Whisper Tiny - smallest, fastest local Whisper",
        "language": "multilingual",
        "pipeline": True,
    },
    "openai/whisper-base": {
        "type": "whisper",
        "size": "~290MB",
        "description": "Whisper Base - good speed/accuracy tradeoff",
        "language": "multilingual",
        "pipeline": True,
    },
    "openai/whisper-small": {
        "type": "whisper",
        "size": "~960MB",
        "description": "Whisper Small - better accuracy",
        "language": "multilingual",
        "pipeline": True,
    },
    "openai/whisper-medium": {
        "type": "whisper",
        "size": "~3GB",
        "description": "Whisper Medium - high accuracy",
        "language": "multilingual",
        "pipeline": True,
    },
    "openai/whisper-large-v3": {
        "type": "whisper",
        "size": "~6GB",
        "description": "Whisper Large V3 - highest accuracy",
        "language": "multilingual",
        "pipeline": True,
    },
}


def _generate_test_audio(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


def download_model(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Download a HuggingFace ASR model to local cache.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Optional custom cache directory

    Returns:
        True if download successful
    """
    try:
        from transformers import AutoProcessor, pipeline as hf_pipeline
    except ImportError:
        print("ERROR: 'transformers' library required. Install with: pip install transformers torch")
        return False

    model_info = SUPPORTED_MODELS.get(model_id, {})
    model_type = model_info.get("type", "wav2vec2")

    print(f"\nDownloading model: {model_id}")
    print(f"  Type: {model_type}")
    print(f"  Size: {model_info.get('size', 'unknown')}")
    print(f"  Description: {model_info.get('description', 'Custom model')}")
    print()

    try:
        start = time.time()

        print("  [1/2] Downloading processor...")
        AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

        print("  [2/2] Downloading model (this may take a while)...")
        hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=-1,
            cache_dir=cache_dir,
        )

        elapsed = time.time() - start
        print(f"\n  Download complete in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"\n  ERROR: Failed to download {model_id}: {e}")
        return False


def verify_model(model_id: str, cache_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verify a model can transcribe audio by running a quick inference.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Optional custom cache directory

    Returns:
        Tuple of (success, transcription_or_error)
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return False, "transformers library not installed"

    model_info = SUPPORTED_MODELS.get(model_id, {})

    print(f"\nVerifying model: {model_id}")

    try:
        print("  Loading model...")
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=-1,
            cache_dir=cache_dir,
        )

        print("  Running test transcription...")
        test_audio = _generate_test_audio(duration=2.0, sample_rate=16000)
        audio_input = {"array": test_audio, "sampling_rate": 16000}

        start = time.time()
        result = pipe(audio_input)
        elapsed = time.time() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Transcription: '{text}'")
        print(f"  Inference time: {elapsed:.3f}s")
        print("  Model verified OK")
        return True, text

    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: Verification failed: {error_msg}")
        return False, error_msg


def benchmark_model(
    model_id: str,
    num_runs: int = 5,
    audio_duration: float = 5.0,
    cache_dir: Optional[str] = None,
) -> Dict:
    """
    Benchmark a model's inference speed.

    Args:
        model_id: HuggingFace model identifier
        num_runs: Number of inference runs
        audio_duration: Test audio duration in seconds
        cache_dir: Optional custom cache directory

    Returns:
        Dict with benchmark results
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return {"error": "transformers library not installed"}

    print(f"\nBenchmarking model: {model_id}")
    print(f"  Runs: {num_runs}, Audio duration: {audio_duration}s")

    try:
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=-1,
            cache_dir=cache_dir,
        )

        test_audio = _generate_test_audio(duration=audio_duration, sample_rate=16000)
        audio_input = {"array": test_audio, "sampling_rate": 16000}

        # Warm-up run
        pipe(audio_input)

        times = []
        for i in range(num_runs):
            start = time.time()
            pipe(audio_input)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

        results = {
            "model": model_id,
            "num_runs": num_runs,
            "audio_duration_s": audio_duration,
            "mean_time_s": np.mean(times),
            "std_time_s": np.std(times),
            "min_time_s": np.min(times),
            "max_time_s": np.max(times),
            "rtf": np.mean(times) / audio_duration,  # Real-time factor
        }

        print(f"\n  Mean: {results['mean_time_s']:.3f}s (±{results['std_time_s']:.3f}s)")
        print(f"  Real-Time Factor: {results['rtf']:.2f}x")

        return results

    except Exception as e:
        return {"error": str(e)}


def list_models():
    """Print all supported models with their details."""
    print("\n" + "=" * 90)
    print("SUPPORTED LOCAL ASR MODELS")
    print("=" * 90)

    by_type = {}
    for model_id, info in SUPPORTED_MODELS.items():
        model_type = info["type"]
        by_type.setdefault(model_type, []).append((model_id, info))

    for model_type, models in by_type.items():
        print(f"\n--- {model_type.upper()} ---")
        for model_id, info in models:
            print(f"\n  {model_id}")
            print(f"    Size:     {info['size']}")
            print(f"    Language: {info['language']}")
            print(f"    Desc:     {info['description']}")

    print("\n" + "=" * 90)
    print("\nUsage with audio_eval:")
    print("  python -m audio_eval.evaluator -n 10 \\")
    print("    --asr-provider local \\")
    print("    --asr-model facebook/wav2vec2-base-960h \\")
    print("    --asr-model-type wav2vec2")
    print()


def check_dependencies() -> List[str]:
    """Check which required packages are installed."""
    missing = []
    packages = {
        "transformers": "transformers",
        "torch": "torch",
        "numpy": "numpy",
        "librosa": "librosa",
    }

    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Setup and verify local ASR models for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all supported models
  python -m local_asr.setup_model --list

  # Download Wav2Vec2 base
  python -m local_asr.setup_model --model facebook/wav2vec2-base-960h

  # Download and verify
  python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --verify

  # Benchmark model speed
  python -m local_asr.setup_model --model facebook/wav2vec2-base-960h --benchmark
        """,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace model ID to download (default: facebook/wav2vec2-base-960h)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all supported local ASR models",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the model after downloading",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Benchmark model inference speed",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for models",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check required dependencies",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.check_deps:
        missing = check_dependencies()
        if missing:
            print(f"Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            sys.exit(1)
        else:
            print("All dependencies installed!")
        return

    # Check dependencies first
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    # Download
    success = download_model(args.model, cache_dir=args.cache_dir)
    if not success:
        sys.exit(1)

    # Verify
    if args.verify:
        ok, result = verify_model(args.model, cache_dir=args.cache_dir)
        if not ok:
            print(f"Verification failed: {result}")
            sys.exit(1)

    # Benchmark
    if args.benchmark:
        benchmark_model(
            args.model,
            num_runs=args.benchmark_runs,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
