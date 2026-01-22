import os
import argparse
import tempfile
import soundfile as sf
import assemblyai as aai
from datasets import load_dataset, Audio


def main(hf_token: str | None):
    ds = load_dataset(
        "byteCode18/spoken-squad-1k-memory-eval",
        split="test",
        token=hf_token
    ).cast_column("context", Audio(decode=True))

    sample = ds[0]
    audio = sample["context"]

    waveform = audio["array"]
    sampling_rate = audio["sampling_rate"]

    # Write temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sampling_rate)
        audio_path = tmp.name

    # AssemblyAI
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    print(sample["instruction"])
    print(sample["answer"])
    print("ASR output:")
    print(transcript.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()

    main(args.hf_token)
