import sys
from types import SimpleNamespace

from hs.models import Word
from hs.transcribe import _restore_segment_punctuation, _segment_words, transcribe


def test_segment_words_preserves_timestamps_and_low_confidence_words() -> None:
    segment = SimpleNamespace(
        text="hello world",
        words=[
            SimpleNamespace(start=1.1, end=1.5, word=" hello", probability=0.94),
            SimpleNamespace(start=1.5, end=1.9, word=" world", probability=0.2),
        ]
    )

    words = _segment_words(segment)

    assert [word.text for word in words] == ["hello", "world"]
    assert [(word.start, word.end) for word in words] == [(1.1, 1.5), (1.5, 1.9)]
    assert words[1].probability == 0.2


def test_restore_segment_punctuation_matches_marks_to_words_without_changing_timing() -> None:
    segment = SimpleNamespace(
        text="Hello, again. women?",
        words=[
            SimpleNamespace(start=0.0, end=0.4, word=" Hello", probability=0.9),
            SimpleNamespace(start=0.5, end=0.9, word=" again", probability=0.8),
            SimpleNamespace(start=1.0, end=1.4, word=" women", probability=0.7),
        ],
    )

    restored = _segment_words(segment)

    assert [word.text for word in restored] == ["Hello,", "again.", "women?"]
    assert [(word.start, word.end) for word in restored] == [
        (0.0, 0.4),
        (0.5, 0.9),
        (1.0, 1.4),
    ]
    assert [word.probability for word in restored] == [0.9, 0.8, 0.7]


def test_restore_segment_punctuation_leaves_unpunctuated_text_untouched() -> None:
    words = [Word(0.0, 0.4, "Nothing"), Word(0.5, 0.9, "changes")]

    assert _restore_segment_punctuation(words, "Nothing changes") == words


def test_transcribe_disables_vad_and_uses_explicit_silence_thresholds(
    tmp_path, monkeypatch
) -> None:
    calls = []

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **kwargs):
            calls.append(kwargs)
            return [], SimpleNamespace(language="en")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    assert transcribe(tmp_path / "audio.wav", device="cpu") == ([], "en")
    assert calls == [
        {
            "language": None,
            "word_timestamps": True,
            "vad_filter": False,
            "beam_size": 5,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
        }
    ]
