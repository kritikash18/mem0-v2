import argparse
from datasets import load_dataset, Audio
from huggingface_hub import HfApi

def build_audio_url(repo_id, path, revision="main"):
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"

def main(args):
    ds = load_dataset(
        args.dataset,
        split=args.split,
        token=args.hf_token if args.hf_token else None,
    )

    ds = ds.cast_column("context", Audio(decode=False))

    def add_url(example):
        audio = example["context"]
        if audio is None or "path" not in audio:
            example["audio_url"] = None
        else:
            example["audio_url"] = build_audio_url(
                args.dataset, audio["path"], args.revision
            )
        return example

#     ds = ds.map(add_url)

#     ds = ds.filter(lambda x: x["audio_url"] is not None)

#     ds.to_parquet(args.output)
#     print(f"Saved dataset with URLs → {args.output}")
    exs = add_url(ds[0])
    print(exs['context']['path'])
    print(exs['audio_url'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="assembly_ready.parquet")
    parser.add_argument("--hf-token", default=None)

    main(parser.parse_args())
