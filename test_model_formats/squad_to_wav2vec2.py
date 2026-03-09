"""
Test local Wav2Vec2 ASR on the Spoken SQuAD dataset.

Downloads a fine-tuned Wav2Vec2 model and transcribes samples from the
evaluation dataset. Tests multiple model variants to compare output quality.

Key distinction:
  - facebook/wav2vec2-base          → pre-trained only, NO CTC head, CANNOT transcribe
  - facebook/wav2vec2-base-960h     → fine-tuned on LibriSpeech 960h, HAS CTC head, works for ASR
  - facebook/wav2vec2-large-960h    → larger fine-tuned variant

Usage:
    python test_model_formats/squad_to_wav2vec2.py
    python test_model_formats/squad_to_wav2vec2.py --model facebook/wav2vec2-large-960h
    python test_model_formats/squad_to_wav2vec2.py --idx 5 --num_samples 3
    python test_model_formats/squad_to_wav2vec2.py --compare
"""

import argparse
import os
import time

import numpy as np
from datasets import Audio, load_dataset


FINE_TUNED_MODELS = {
    "facebook/wav2vec2-base-960h": "wav2vec2",
    "facebook/wav2vec2-large-960h": "wav2vec2",
    "facebook/wav2vec2-large-960h-lv60-self": "wav2vec2",
    "facebook/hubert-large-ls960-ft": "hubert",
}

# This is a pre-trained-only model — it will FAIL to transcribe
PRETRAINED_ONLY = "facebook/wav2vec2-base"


def load_dataset_samples(hf_token, idx, num_samples):
    ds = load_dataset(
        "byteCode18/spoken-squad-1k-memory-eval",
        split="test",
        token=hf_token,
    ).cast_column("context", Audio(decode=True))

    indices = list(range(idx, min(idx + num_samples, len(ds))))
    return ds, indices


def transcribe_with_pipeline(model_id, audio_array, sampling_rate):
    """Transcribe using HuggingFace pipeline (recommended path)."""
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,  # CPU for testing
    )

    result = pipe({"array": audio_array, "sampling_rate": sampling_rate})
    return result["text"] if isinstance(result, dict) else str(result)


def transcribe_with_manual_ctc(model_id, audio_array, sampling_rate):
    """Transcribe using manual model + processor loading (CTC decode path)."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)

    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]


def transcribe_with_mem0_local(model_id, model_type, audio_dict):
    """Transcribe using mem0's LocalASR provider (the actual eval path)."""
    from mem0.asr.local import LocalASR
    from mem0.configs.asr.local import LocalAsrConfig

    config = LocalAsrConfig(
        model=model_id,
        model_type=model_type,
        use_pipeline=True,
        device="cpu",
    )
    asr = LocalASR(config)
    return asr.transcribe(audio_dict)


def test_pretrained_only_fails():
    """Show that a pre-trained-only model (no CTC head) cannot transcribe."""
    print("\n" + "=" * 70)
    print("TEST: Pre-trained model (no fine-tuning) should FAIL")
    print(f"Model: {PRETRAINED_ONLY}")
    print("=" * 70)

    try:
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=PRETRAINED_ONLY,
            device=-1,
        )
        dummy = np.zeros(16000, dtype=np.float32)
        result = pipe({"array": dummy, "sampling_rate": 16000})
        print(f"  Unexpected success: {result}")
    except Exception as e:
        error_type = type(e).__name__
        print(f"  Expected failure: {error_type}")
        print(f"  Message: {str(e)[:200]}")
        print("\n  This confirms: raw pre-trained Wav2Vec2 has NO CTC head.")
        print("  You MUST use a fine-tuned variant like wav2vec2-base-960h.")


def main(hf_token, model_id, model_type, idx, num_samples, compare, show_fail):
    ds, indices = load_dataset_samples(hf_token, idx, num_samples)

    if show_fail:
        test_pretrained_only_fails()
        print()

    models_to_test = (
        list(FINE_TUNED_MODELS.items()) if compare
        else [(model_id, model_type)]
    )

    for mid, mtype in models_to_test:
        print("\n" + "=" * 70)
        print(f"MODEL: {mid}  (type: {mtype})")
        print("=" * 70)

        for sample_idx in indices:
            sample = ds[sample_idx]
            audio = sample["context"]
            waveform = audio["array"]
            sr = audio["sampling_rate"]
            question = sample["instruction"]
            answer = sample["answer"]

            print(f"\n--- Sample {sample_idx} ---")
            print(f"  Question: {question}")
            print(f"  Answer:   {answer}")
            print(f"  Audio:    {waveform.shape[0]/sr:.1f}s, {sr}Hz, {waveform.dtype}")

            # Method 1: HuggingFace pipeline
            print("\n  [Pipeline]")
            t0 = time.time()
            try:
                text_pipe = transcribe_with_pipeline(mid, waveform, sr)
                print(f"    Transcript: {text_pipe}")
                print(f"    Time: {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"    ERROR: {e}")

            # Method 2: Manual CTC (only for CTC models)
            if mtype in ("wav2vec2", "hubert"):
                print("\n  [Manual CTC]")
                t0 = time.time()
                try:
                    text_manual = transcribe_with_manual_ctc(mid, waveform, sr)
                    print(f"    Transcript: {text_manual}")
                    print(f"    Time: {time.time()-t0:.2f}s")
                except Exception as e:
                    print(f"    ERROR: {e}")

            # Method 3: mem0 LocalASR
            print("\n  [mem0 LocalASR]")
            t0 = time.time()
            try:
                audio_dict = {"array": waveform, "sampling_rate": sr}
                text_mem0 = transcribe_with_mem0_local(mid, mtype, audio_dict)
                print(f"    Transcript: {text_mem0}")
                print(f"    Time: {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"    ERROR: {e}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Wav2Vec2 local ASR on Spoken SQuAD"
    )
    parser.add_argument(
        "--hf_token", type=str, default=os.getenv("HF_TOKEN"),
        help="HuggingFace token",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/wav2vec2-base-960h",
        help="HuggingFace model ID (must be fine-tuned for ASR)",
    )
    parser.add_argument(
        "--model_type", type=str, default="wav2vec2",
        choices=["wav2vec2", "hubert", "whisper"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--idx", type=int, default=0,
        help="Starting sample index",
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=1,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Test ALL fine-tuned models on the same samples",
    )
    parser.add_argument(
        "--show_fail", action="store_true",
        help="Also demonstrate that pre-trained-only model fails",
    )
    args = parser.parse_args()

    main(
        args.hf_token, args.model, args.model_type,
        args.idx, args.num_samples, args.compare, args.show_fail,
    )
