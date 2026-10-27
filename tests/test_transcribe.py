from types import SimpleNamespace

from hs.transcribe import _segment_words


def test_segment_words_preserves_timestamps_and_low_confidence_words() -> None:
    segment = SimpleNamespace(
        words=[
            SimpleNamespace(start=1.1, end=1.5, word=" hello", probability=0.94),
            SimpleNamespace(start=1.5, end=1.9, word=" world", probability=0.2),
        ]
    )

    words = _segment_words(segment)

    assert [word.text for word in words] == ["hello", "world"]
    assert [(word.start, word.end) for word in words] == [(1.1, 1.5), (1.5, 1.9)]
    assert words[1].probability == 0.2
