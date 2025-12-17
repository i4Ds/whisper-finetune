#!/usr/bin/env python3
"""
Unified script to upload Whisper models to Hugging Face Hub.

This script can:
1. Upload original .pt checkpoint files directly to HF
2. Convert to CTranslate2/faster-whisper format and upload
3. Do both - upload .pt AND faster-whisper format to the same repo

Supports both:
- Local model files (direct path to .pt file)
- W&B run paths (downloads from Weights & Biases)

Usage examples:
    # Upload from local file with both formats
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --both
    
    # Upload from W&B with both formats
    python upload_model_to_hub.py --wandb-run i4ds/whisper4sg/runs/abc123 --repo i4ds/my-model --both
    
    # Upload only the .pt file
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --pt-only
    
    # Upload only faster-whisper format
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --ct2-only
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import wandb
from ctranslate2.converters import TransformersConverter
from huggingface_hub import HfApi

from whisper_finetune.scripts.convert_openai_to_hf import (
    convert_openai_whisper_to_tfms,
)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_model_from_wandb(
    run_path: str,
    save_dir: Path,
    file_basename: str = "last_model.pt",
) -> Path:
    """Download a model file from Weights & Biases.

    Args:
        run_path: W&B run path (e.g., "i4ds/whisper4sg/runs/abc123")
        save_dir: Directory to save the downloaded file
        file_basename: Name of the file to download (default: "last_model.pt")

    Returns:
        Path to the downloaded file
    """
    api = wandb.Api()
    run = api.run(run_path)

    # Find a matching file
    candidate = None
    for f in run.files():
        if f.name.endswith(file_basename): 
            candidate = f
            break

    if candidate is None:
        available = "\n".join(sorted(f.name for f in run.files()))
        raise FileNotFoundError(
            f"No file ending with '{file_basename}' found in run {run_path}.\n"
            f"Available files:\n{available}"
        )

    ensure_dir(save_dir)
    local_path = candidate.download(root=str(save_dir), replace=True)

    if isinstance(local_path, str):
        return Path(local_path)
    else:
        return save_dir / candidate.name


def convert_to_ct2(
    *,
    checkpoint_path: Path,
    hf_model_folder: Path,
    tokenizer_source_dir: Path,
    ct2_output_dir: Path,
    ct2_quantization: str = "float16",
    load_as_float16: bool = True,
    readme_text: Optional[str] = None,
) -> Path:
    """Convert OpenAI Whisper checkpoint to CTranslate2/faster-whisper format.

    Args:
        checkpoint_path: Path to the .pt checkpoint
        hf_model_folder: Temporary folder for HF intermediate format
        tokenizer_source_dir: Directory containing tokenizer.json and config.json
        ct2_output_dir: Output directory for CTranslate2 model
        ct2_quantization: Quantization type (e.g., "float16", "int8")
        load_as_float16: Whether to load model in float16
        readme_text: Optional README content

    Returns:
        Path to the CTranslate2 output directory
    """
    ensure_dir(hf_model_folder)
    ensure_dir(ct2_output_dir)

    # Convert to HF format first
    model, _is_multilingual, _num_languages = convert_openai_whisper_to_tfms(
        str(checkpoint_path), str(hf_model_folder)
    )

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
    # Only copy README if it exists
    copy_files = ["tokenizer.json"]
    if (hf_model_folder / "README.md").exists():
        copy_files.append("README.md")
    
    converter = TransformersConverter(
        str(hf_model_folder),
        copy_files=copy_files,
        load_as_float16=load_as_float16,
    )
    converter.convert(output_dir=str(ct2_output_dir), quantization=ct2_quantization, force=True)

    return ct2_output_dir


def upload_to_hub(
    repo_id: str,
    pt_path: Optional[Path] = None,
    ct2_folder: Optional[Path] = None,
    private: bool = True,
    readme_text: Optional[str] = None,
) -> None:
    """Upload model files to Hugging Face Hub.

    Args:
        repo_id: HF repository ID (e.g., "i4ds/my-model")
        pt_path: Path to the .pt checkpoint (optional)
        ct2_folder: Path to CTranslate2 folder (optional)
        private: Whether to make the repo private
        readme_text: Optional README content for the repo
    """
    api = HfApi(token='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')
    
    # Create repo
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    
    # Upload .pt file if provided
    if pt_path is not None and pt_path.exists():
        print(f"Uploading {pt_path.name} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(pt_path),
            path_in_repo=pt_path.name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Uploaded {pt_path.name}")
    
    # Upload CTranslate2/faster-whisper folder if provided
    # Upload to root of repo so faster-whisper can load it directly with repo_id
    if ct2_folder is not None and ct2_folder.exists():
        print(f"Uploading faster-whisper model files to {repo_id} (root)...")
        api.upload_folder(
            folder_path=str(ct2_folder),
            path_in_repo=".",  # Upload to root, not a subfolder
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Uploaded faster-whisper files to repo root")
    
    # Upload/update README if provided
    if readme_text is not None:
        readme_path = Path("/tmp/README.md")
        readme_path.write_text(readme_text)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Updated README.md")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Whisper models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload from local file with both formats
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --both
    
    # Upload from W&B with both formats  
    python upload_model_to_hub.py --wandb-run i4ds/whisper4sg/runs/abc123 --repo i4ds/my-model --both
    
    # Upload only the .pt file
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --pt-only
    
    # Upload only faster-whisper format
    python upload_model_to_hub.py --local-path ./model.pt --repo i4ds/my-model --ct2-only
        """
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--local-path",
        type=Path,
        help="Local path to .pt checkpoint file"
    )
    source_group.add_argument(
        "--wandb-run",
        type=str,
        help="W&B run path (e.g., 'i4ds/whisper4sg/runs/abc123')"
    )
    
    # Required
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'i4ds/my-model')"
    )
    
    # Upload mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--both",
        action="store_true",
        help="Upload both .pt and faster-whisper format"
    )
    mode_group.add_argument(
        "--pt-only",
        action="store_true",
        help="Upload only the .pt checkpoint"
    )
    mode_group.add_argument(
        "--ct2-only",
        action="store_true",
        help="Upload only the faster-whisper/CTranslate2 format"
    )
    
    # Optional
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("whisper_v3_turbo_utils"),
        help="Directory containing tokenizer.json and config.json (default: whisper_v3_turbo_utils)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="float16",
        choices=["float16", "int8", "int8_float16", "int8_bfloat16"],
        help="CTranslate2 quantization type (default: float16)"
    )
    parser.add_argument(
        "--file-basename",
        type=str,
        default="last_model.pt",
        help="Basename of the model file in W&B (default: last_model.pt)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public (default: private)"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./upload_work"),
        help="Working directory for temporary files (default: ./upload_work)"
    )
    parser.add_argument(
        "--readme",
        type=str,
        help="Custom README content (or path to README file)"
    )
    parser.add_argument(
        "--wandb-url",
        type=str,
        help="W&B run URL for README (use with --local-path when you have a local file but want to link to W&B)"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    os.environ.setdefault("HF_HOME", str(args.work_dir / "cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.work_dir / "cache" / "transformers"))
    
    # Get checkpoint path
    if args.local_path:
        checkpoint_path = args.local_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        wandb_run_url = args.wandb_url  # Use provided URL if available
    else:
        print(f"Downloading from W&B: {args.wandb_run}")
        download_dir = args.work_dir / "downloaded"
        checkpoint_path = download_model_from_wandb(
            run_path=args.wandb_run,
            save_dir=download_dir,
            file_basename=args.file_basename,
        )
        wandb_run_url = f"https://wandb.ai/{args.wandb_run}"
        print(f"Downloaded checkpoint: {checkpoint_path}")
    
    # Prepare README
    if args.readme:
        readme_path = Path(args.readme)
        if readme_path.exists():
            readme_text = readme_path.read_text()
        else:
            readme_text = args.readme
    else:
        # Default README
        readme_text = f"""# {args.repo.split('/')[-1]}

This repository contains a fine-tuned Whisper model.

## Contents

"""
        if args.both or args.pt_only:
            readme_text += f"- `{args.file_basename}`: Original OpenAI Whisper format checkpoint\n"
        if args.both or args.ct2_only:
            readme_text += "- CTranslate2/faster-whisper model files (at repo root)\n"
        
        if args.both or args.ct2_only:
            readme_text += f"""
## Usage with faster-whisper

```python
from faster_whisper import WhisperModel

# Load directly from HuggingFace Hub
model = WhisperModel("{args.repo}", device="cuda", compute_type="{args.quantization}")
segments, info = model.transcribe("audio.mp3", language="de")

for segment in segments:
    print(f"[{{segment.start:.2f}}s -> {{segment.end:.2f}}s] {{segment.text}}")
```
"""
        
        if wandb_run_url:
            readme_text += f"\n## Training\n\nW&B Run: {wandb_run_url}\n"
    
    # Determine what to upload
    pt_to_upload = None
    ct2_folder = None
    
    if args.both or args.pt_only:
        pt_to_upload = checkpoint_path
    
    if args.both or args.ct2_only:
        print("Converting to CTranslate2/faster-whisper format...")
        repo_name = args.repo.split("/")[-1]
        hf_model_folder = args.work_dir / "hf_models" / f"hf_{repo_name}"
        ct2_output_dir = args.work_dir / "ct2_output" / repo_name
        
        ct2_folder = convert_to_ct2(
            checkpoint_path=checkpoint_path,
            hf_model_folder=hf_model_folder,
            tokenizer_source_dir=args.tokenizer_dir,
            ct2_output_dir=ct2_output_dir,
            ct2_quantization=args.quantization,
            load_as_float16=args.quantization in ("float16", "int8_float16"),
            readme_text=readme_text,  # Include README in faster-whisper folder
        )
        print(f"CTranslate2 conversion complete: {ct2_folder}")
    
    # Upload to Hub
    print(f"\nUploading to {args.repo}...")
    upload_to_hub(
        repo_id=args.repo,
        pt_path=pt_to_upload,
        ct2_folder=ct2_folder,
        private=not args.public,
        readme_text=readme_text,
    )
    
    print(f"\n✓ Done! Model available at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
