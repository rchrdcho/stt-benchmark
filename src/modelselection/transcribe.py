from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from langchain_core.runnables import RunnableLambda  # pyright: ignore[reportMissingImports]
from openai import OpenAI

from .types import NormalizedMeta, NormalizedOutput, Segment, TranscriptionOutput


class _TranscriptSegment(Protocol):
    text: str
    start: float
    end: float
    speaker: str | None


class _Transcript(Protocol):
    segments: Sequence[_TranscriptSegment]
    duration: float | None


def _transcribe_call(
    inputs: Mapping[str, object],
    config: Mapping[str, object],
) -> TranscriptionOutput:
    audio_path = cast(Path, inputs["audio_path"])
    cfg = cast(Mapping[str, object], config.get("configurable", {}))
    model = cast(str, cfg.get("model", "gpt-4o-transcribe-diarize"))
    language_hint = cast(str, cfg.get("language_hint", ""))
    prompt = cast(str | None, cfg.get("prompt"))

    client = OpenAI()
    with audio_path.open("rb") as audio_file:
        if prompt is not None:
            transcript = cast(
                _Transcript,
                client.audio.transcriptions.create(  # pyright: ignore[reportCallIssue,reportArgumentType,reportInvalidCast]
                    model=model,
                    file=audio_file,
                    response_format="diarized_json",  # pyright: ignore[reportArgumentType]
                    chunking_strategy="auto",
                    prompt=prompt,
                ),
            )
        else:
            transcript = cast(
                _Transcript,
                client.audio.transcriptions.create(  # pyright: ignore[reportCallIssue,reportArgumentType,reportInvalidCast]
                    model=model,
                    file=audio_file,
                    response_format="diarized_json",  # pyright: ignore[reportArgumentType]
                    chunking_strategy="auto",
                ),
            )

    # Minimal runtime guards: fail fast if API response structure is unexpected
    if not hasattr(transcript, "segments"):
        raise ValueError(
            "API response missing 'segments' field. Expected diarized_json response structure."
        )

    if not isinstance(transcript.segments, Sequence):
        raise ValueError(
            f"API response 'segments' is not iterable. Got type: {type(transcript.segments).__name__}"
        )

    # Validate each segment has required fields
    for i, segment in enumerate(transcript.segments):
        if not hasattr(segment, "text"):
            raise ValueError(
                f"Segment {i} missing required 'text' field. Got attributes: {dir(segment)}"
            )
        if not hasattr(segment, "start"):
            raise ValueError(
                f"Segment {i} missing required 'start' field. Got attributes: {dir(segment)}"
            )
        if not hasattr(segment, "end"):
            raise ValueError(
                f"Segment {i} missing required 'end' field. Got attributes: {dir(segment)}"
            )

    raw: dict[str, object]
    try:
        raw = transcript.model_dump()  # type: ignore[attr-defined]
    except AttributeError:
        try:
            raw = dict(transcript)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            raw = transcript.__dict__  # type: ignore[attr-defined]

    segments = [
        Segment(
            text=segment.text,
            start=segment.start,
            end=segment.end,
            speaker=getattr(segment, "speaker", None),
        )
        for segment in transcript.segments
    ]

    meta: NormalizedMeta = {
        "model": model,
        "audio_id": audio_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration": getattr(transcript, "duration", 0.0),
        "language_hint": language_hint,
    }

    normalized = NormalizedOutput(meta=meta, segments=segments)
    return TranscriptionOutput(raw=raw, normalized=normalized)


def build_transcribe_runnable() -> RunnableLambda:
    return RunnableLambda(_transcribe_call)


def transcribe_diarize(
    audio_path: Path,
    model: str,
    language_hint: str,
    prompt: str | None = None,
) -> TranscriptionOutput:
    runnable = build_transcribe_runnable()
    output = runnable.invoke(
        {"audio_path": audio_path},
        config={
            "configurable": {
                "model": model,
                "language_hint": language_hint,
                "prompt": prompt,
            }
        },
    )
    return cast(TranscriptionOutput, output)
