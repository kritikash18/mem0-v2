from datasets import load_dataset, Audio

ds = load_dataset(
    "AudioLLMs/spoken_squad_test",
    split="test",
    token="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"
).cast_column("context", Audio(decode=True))

sample = ds[0]
audio = sample["context"]

print(audio["array"].shape, audio["sampling_rate"])
