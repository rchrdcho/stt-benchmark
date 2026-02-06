from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import TranscriptionOutput


def write_outputs(output: TranscriptionOutput, base_path: Path) -> tuple[Path, Path]:
    raw_path = base_path.parent / f"{base_path.name}.raw.json"
    normalized_path = base_path.parent / f"{base_path.name}.normalized.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)

    _ = raw_path.write_text(
        json.dumps(output.raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    normalized_payload = {
        "meta": output.normalized.meta,
        "segments": [asdict(segment) for segment in output.normalized.segments],
    }
    _ = normalized_path.write_text(
        json.dumps(normalized_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return (raw_path, normalized_path)
