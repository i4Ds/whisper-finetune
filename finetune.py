from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer

from whisper_finetune.data.data_loader import AudioDataset, Record, get_dataloader


def dataloader():
    print("In main.")
    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    tokenizer = get_tokenizer(multilingual=True, task="transcribe")

    max_prompt_length = 256

    debug_loader = get_dataloader(
        hu_dataset=hf_dataset,
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
    print("Dataset loaded.")
    # Time to iterate over the dataset.
    for i, batch in enumerate(tqdm(debug_loader)):
        assert len(batch) == 3


def audiodataset():
    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    tokenizer = get_tokenizer(multilingual=True, task="transcribe", language="de")

    dataset = AudioDataset(
        hf_dataset,
        tokenizer,
        fp16=True,
        no_timestamps_training=False,
        max_prompt_length=223,
        prompt_use_rate=1,
        no_timestamps_rate=0.5,
        spec_augment=False,
    )
    for i in range(5, 9):
        t_prompt = hf_dataset[i]["prompt"]
        print(t_prompt)
        encoded = dataset._get_prompt_tokens(t_prompt, True)
        print(encoded)

        print(tokenizer.decode_with_timestamps(encoded))
        print(tokenizer.decode_with_timestamps([879, 482]))


def example_w():
    import whisper

    hf_dataset = load_dataset("i4ds/stt4sg-350_train_all_fold_4", split="train")
    hf_dataset = hf_dataset.with_format(type="torch")
    model = whisper.load_model("large-v3")
    print(hf_dataset[8])
    print(hf_dataset[8]["text"])
    result = model.transcribe(hf_dataset[8]["audio"]["array"])
    tokens = [x["tokens"] for x in result["segments"]]
    tokens = [item for sublist in tokens for item in sublist]
    tokenizer = get_tokenizer(multilingual=True, task="transcribe")
    print(tokenizer.decode_with_timestamps(tokens))


if __name__ == "__main__":
    audiodataset()
