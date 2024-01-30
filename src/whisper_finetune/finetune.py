from datasets import load_dataset
from whisper.tokenizer import get_tokenizer
from data.data_loader import Record, get_dataloader
from torch.utils.data import DataLoader
from typing import Iterator

def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

def main() -> None:
    hf_dataset = load_dataset("i4ds/STT_SRG_DEBUG", split="train")
    hf_dataset = hf_dataset.with_format(type="torch")

    tokenizer = get_tokenizer(multilingual=True, task="transcribe")

    max_prompt_length = 256

    records = []
    for sample in hf_dataset:
        records.append(Record(audio_array=sample["audio"]["array"], text=sample["text"], language=sample["language"], prompt=sample["prompt"]))

    debug_loader = get_dataloader(
        records=records,
        tokenizer=tokenizer,
        batch_size=1,
        fp16=True,
        no_timestamps_training=False,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=0.5,
        no_timestamps_rate=0.5,
        shuffle=True,
        num_workers=1,
        spec_augment=True,
    )

    it = iter(debug_loader)

    for i in range(5):
        sample = next(it)
        print(sample)


if __name__ == "__main__":
    main()