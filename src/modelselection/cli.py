from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from .io import write_outputs
from .transcribe import transcribe_diarize


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _read_prompt(prompt: str | None, prompt_file: Path | None) -> str | None:
    if prompt_file is None:
        return prompt
    return prompt_file.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio using gpt-4o-transcribe-diarize and save JSON output.",
    )
    _ = parser.add_argument(
        "--audio", required=True, type=Path, help="Path to the input audio file."
    )
    _ = parser.add_argument(
        "--model",
        default="gpt-4o-transcribe-diarize",
        help="OpenAI transcription model name.",
    )
    _ = parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Base directory for output JSON files.",
    )
    _ = parser.add_argument(
        "--language-hint",
        default="",
        help="Language hint used in the prompt (stored in meta).",
    )
    _ = parser.add_argument(
        "--prompt", default=None, help="Prompt text passed to the model."
    )
    _ = parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a prompt file (overrides --prompt).",
    )
    _ = parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists.",
    )
    return parser


def main() -> None:
    _ = load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. Please set it in .env file."
        )

    parser = build_parser()
    args = parser.parse_args()

    audio_path: Path = cast(Path, args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = cast(str, args.model)
    output_dir = cast(Path, args.output_dir)
    overwrite = cast(bool, args.overwrite)

    model_dir = output_dir / _safe_filename(model)
    base_path = model_dir / _safe_filename(audio_path.name)

    raw_path = base_path.parent / f"{base_path.name}.raw.json"
    normalized_path = base_path.parent / f"{base_path.name}.normalized.json"
    if (raw_path.exists() or normalized_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {raw_path} or {normalized_path}"
        )

    prompt = _read_prompt(
        cast(str | None, args.prompt), cast(Path | None, args.prompt_file)
    )
    output = transcribe_diarize(
        audio_path=audio_path,
        model=model,
        language_hint=cast(str, args.language_hint),
        prompt=prompt,
    )
    raw_path, normalized_path = write_outputs(output, base_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {normalized_path}")
