#!/usr/bin/env python3
"""
Batch script to:
- Download Whisper checkpoints (.pt) from Weights & Biases runs
- Convert them to Hugging Face format
- Convert to CTranslate2 (float16)
- Upload the CTranslate2 folders to the Hugging Face Hub

All parameters are defined in the __main__ section. It supports batching via lists
for names, W&B run paths, and HF repo ids, which are zipped together and processed.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import wandb
from ctranslate2.converters import TransformersConverter
from huggingface_hub import HfApi

# Uses local helper from this repo
from whisper_finetune.scripts.convert_openai_to_hf import (
    convert_openai_whisper_to_tfms,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_model_from_wandb(
    run_path: str,
    save_dir: Path,
    file_basename: str = "last_model.pt",
) -> Path:
    """Download a model file from Weights & Biases.

    This searches the run's files for one whose name ends with `file_basename`
    (default: "last_model.pt"). The file is downloaded preserving its relative
    path under `save_dir`. Returns the local file path.
    """
    api = wandb.Api()
    run = api.run(run_path)

    # Find a matching file (first match wins)
    candidate = None
    for f in run.files():
        # f.name is the run-internal path (e.g. "40569234_output/last_model.pt")
        if f.name.endswith(file_basename):
            candidate = f
            break

    if candidate is None:
        available = "\n".join(sorted(f.name for f in run.files()))
        raise FileNotFoundError(
            f"No file ending with '{file_basename}' found in run {run_path}.\n" f"Available files:\n{available}"
        )

    ensure_dir(save_dir)
    # Download preserves subfolders under save_dir
    local_path = candidate.download(root=str(save_dir), replace=True)

    # The API returns the local path in newer versions; if not, reconstruct
    if isinstance(local_path, str):
        return Path(local_path)
    else:
        return save_dir / candidate.name


def convert_to_hf_and_ct2(
    *,
    checkpoint_path: Path,
    hf_model_folder: Path,
    tokenizer_source_dir: Path,
    ct2_output_dir: Path,
    ct2_quantization: str = "float16",
    load_as_float16: bool = True,
    readme_text: Optional[str] = None,
) -> Path:
    """Convert OpenAI Whisper checkpoint to HF, then to CTranslate2.

    Returns the path to the ct2 output directory.
    """
    ensure_dir(hf_model_folder)
    ensure_dir(ct2_output_dir)

    model, _is_multilingual, _num_languages = convert_openai_whisper_to_tfms(str(checkpoint_path), str(hf_model_folder))

    # Save README
    if readme_text:
        (hf_model_folder / "README.md").write_text(readme_text)

    # Save model weights/config
    model.save_pretrained(hf_model_folder)

    # Copy tokenizer/config from provided source directory
    tok_json = tokenizer_source_dir / "tokenizer.json"
    cfg_json = tokenizer_source_dir / "config.json"
    if tok_json.exists():
        shutil.copyfile(tok_json, hf_model_folder / "tokenizer.json")
    else:
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_source_dir}")

    if cfg_json.exists():
        shutil.copyfile(cfg_json, hf_model_folder / "config.json")
    else:
        raise FileNotFoundError(f"Missing config.json in {tokenizer_source_dir}")

    # Convert to CTranslate2
    converter = TransformersConverter(
        str(hf_model_folder),
        copy_files=["tokenizer.json", "README.md"],
        load_as_float16=load_as_float16,
    )
    converter.convert(output_dir=str(ct2_output_dir), quantization=ct2_quantization, force=True)

    return ct2_output_dir


def upload_ct2_to_hub(ct2_folder: Path, repo_id: str, private: bool = True) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ct2_folder),
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    # Caches (override here if desired)
    os.environ.setdefault("HF_HOME", "./cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "./cache/huggingface/t_cache")

    wandb_runs: Sequence[str] = [
        # Example: "i4ds/whisper4sg/runs/w2u8dihi"
        "i4ds/whisper4sg/runs/y109yl61",
        "i4ds/whisper4sg/runs/3g251dpa",
        "i4ds/whisper4sg/runs/4wwugkss",
        "i4ds/whisper4sg/runs/x9aebznp",
    ]
    hf_repo_ids: Sequence[str] = [
        # Example: "swissnlp/daily-brook-134"  # aka HU model path
        "i4ds/smart-galaxy-136",
        "i4ds/colorful-darkness-137",
        "i4ds/astral-star-138",
        "i4ds/clear-armadillo-139",
    ]

    # Optional: if your last checkpoint file in W&B has a different basename
    file_basename = "last_model.pt"

    # IO locations
    download_dir = Path("./downloaded_models")
    hf_models_root = Path("./hf_models")  # will create subfolders per name
    ct2_root = Path("./ct2_output")  # will create subfolders per name

    # Tokenizer/config source directory (choose one that matches your model)
    # Options present in this repo include: "whisper_v3_turbo_utils" or "whisper_v3_utils"
    tokenizer_source_dir = Path("whisper_v3_turbo_utils")

    # Conversion options
    ct2_quantization = "float16"
    load_as_float16 = True

    # Hub upload options
    make_private = True

    # README template (edit as needed)
    base_readme = (
        "# Model Information\n\n"
        "This folder contains a converted model using CTranslate2.\n\n"
        "## Conversion Details\n"
        "The model (large-v3) was converted to CTranslate2 format with float16 quantization.\n\n"
        "## Data\n"
        "Model was trained on stt4sg, SRG v3 PL, SRV v3 Real, SPC_R and SDS-200 data.\n"
    )

    if not (len(wandb_runs) == len(hf_repo_ids)):
        raise ValueError(f"List lengths must match: wandb_runs={len(wandb_runs)}, hf_repo_ids={len(hf_repo_ids)}")

    for run_path, repo in zip(wandb_runs, hf_repo_ids):
        print(f"Processing:  run={run_path}, repo={repo}")

        # 1) Download from W&B
        local_ckpt = download_model_from_wandb(
            run_path=run_path,
            save_dir=download_dir,
            file_basename=file_basename,
        )
        print(f"Downloaded checkpoint: {local_ckpt}")

        # 2) Convert to HF + CT2
        name = repo.split("/")[-1]
        hf_model_folder = hf_models_root / f"hf_model_{name}"
        ct2_output_dir = ct2_root / name

        # Add run link into README
        readme_text = base_readme + "\n## Weights & Biases Run\n" + f"https://wandb.ai/{run_path}\n"

        ct2_folder = convert_to_hf_and_ct2(
            checkpoint_path=local_ckpt,
            hf_model_folder=hf_model_folder,
            tokenizer_source_dir=tokenizer_source_dir,
            ct2_output_dir=ct2_output_dir,
            ct2_quantization=ct2_quantization,
            load_as_float16=load_as_float16,
            readme_text=readme_text,
        )
        print(f"CTranslate2 folder ready: {ct2_folder}")

        # 3) Upload to Hub
        upload_ct2_to_hub(ct2_folder=ct2_folder, repo_id=repo, private=make_private)
        print(f"Uploaded {ct2_folder} to {repo}")
