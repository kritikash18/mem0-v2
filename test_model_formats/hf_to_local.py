import os
import argparse
import soundfile as sf
from datasets import load_dataset, Audio

def main(hf_token: str | None, idx: int):
    ds = load_dataset(
        "byteCode18/spoken-squad-1k-memory-eval",
        split="test",
        token=hf_token
    ).cast_column("context", Audio(decode=True))

    sample = ds[idx]
    audio = sample["context"]

    waveform = audio["array"]
    sampling_rate = audio["sampling_rate"]

    output_path = f"sample_{idx}.wav"
    sf.write(output_path, waveform, sampling_rate)

    print(f"Saved audio to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    main(args.hf_token, args.idx)
