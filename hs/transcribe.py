import gc
import re
from pathlib import Path
from typing import Any

from hs.models import Word


DEFAULT_MODEL = "Systran/faster-whisper-large-v3"
TRAILING_PUNCTUATION = re.compile(r"[.!?…,]+$")


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
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
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
    words = [
        Word(
            start=float(word.start),
            end=float(word.end),
            text=word.word.strip(),
            probability=float(word.probability),
        )
        for word in (segment.words or [])
        if word.start is not None and word.end is not None and word.word.strip()
    ]
    if words and not any(TRAILING_PUNCTUATION.search(word.text) for word in words):
        return _restore_segment_punctuation(words, getattr(segment, "text", "") or "")
    return words


def _restore_segment_punctuation(words: list[Word], segment_text: str) -> list[Word]:
    restored = list(words)
    word_index = 0

    for token in segment_text.split():
        match = TRAILING_PUNCTUATION.search(token)
        token_text = TRAILING_PUNCTUATION.sub("", token).casefold()
        while word_index < len(restored):
            word_text = TRAILING_PUNCTUATION.sub("", restored[word_index].text).casefold()
            if word_text == token_text:
                if match:
                    word = restored[word_index]
                    restored[word_index] = Word(
                        word.start,
                        word.end,
                        f"{word.text}{match.group()}",
                        word.probability,
                    )
                word_index += 1
                break
            word_index += 1

    return restored
