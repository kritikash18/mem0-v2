import os
import argparse
import tempfile
import soundfile as sf
from datasets import load_dataset, Audio

from google.cloud import speech


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

    # ------------------
    # Google STT v1
    # ------------------
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        content = f.read()

    audio_request = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sampling_rate,
        language_code="en-US",
    )

    response = client.recognize(
        config=config,
        audio=audio_request
    )

    transcript_text = " ".join(
        result.alternatives[0].transcript
        for result in response.results
    )

    print(sample["instruction"])
    print(sample["answer"])
    print("ASR output:")
    print(transcript_text)


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
