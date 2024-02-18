import argparse
import shutil
from pathlib import Path

from ctranslate2.converters import TransformersConverter
from transformers.models.whisper.convert_openai_to_hf import (
    convert_openai_whisper_to_tfms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert_to_ctranslate2.py", description="Converts model to ctranslate2 for use with faster-whisper."
    )
    parser.add_argument("--model", type=str, help="Path to the Whisper model .pt file to convert")
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory to save converted model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        help="Optional quantization",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=str,
        required=True,
        help="Tokenizer file for huggingface based model",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    hf_model_folder = Path(args.out_dir, "hf")
    ctranslate2_model_folder = Path(args.out_dir, "ct2")

    # Convert to Huggingface Model
    convert_openai_whisper_to_tfms(args.model, hf_model_folder)
    shutil.copyfile(args.tokenizer_json_path, Path(hf_model_folder, "tokenizer.json"))

    # Convert to ctranslate2
    converter = TransformersConverter(
        hf_model_folder,
        copy_files=["tokenizer.json"],
        load_as_float16=args.quantization in ("float16", "int8_float16"),
    )

    converter.convert(output_dir=ctranslate2_model_folder, quantization=args.quantization)


if __name__ == "__main__":
    main()
