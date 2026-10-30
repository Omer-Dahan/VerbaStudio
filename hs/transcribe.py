import gc
from pathlib import Path
from typing import Any

from hs.models import Word


DEFAULT_MODEL = "Systran/faster-whisper-large-v3"


def transcribe(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "cuda",
    compute_type: str = "int8_float16",
) -> tuple[list[Word], str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        words = [word for segment in segments for word in _segment_words(segment)]
        detected_language = language or info.language
        return words, detected_language
    finally:
        del model
        gc.collect()
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.cuda.empty_cache()


def _segment_words(segment: Any) -> list[Word]:
    return [
        Word(
            start=float(word.start),
            end=float(word.end),
            text=word.word.strip(),
            probability=float(word.probability),
        )
        for word in (segment.words or [])
        if word.start is not None and word.end is not None and word.word.strip()
    ]
