from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class NormalizedMeta(TypedDict):
    model: str
    audio_id: str
    created_at: str
    duration: float
    language_hint: str


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    speaker: str | None


@dataclass(frozen=True)
class NormalizedOutput:
    meta: NormalizedMeta
    segments: list[Segment]


@dataclass(frozen=True)
class TranscriptionOutput:
    raw: dict[str, object]
    normalized: NormalizedOutput
