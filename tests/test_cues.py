import pytest

from hs.cues import characters_per_second, enforce_reading_speed
from hs.models import Cue, Word


def test_fast_cue_splits_on_real_word_boundary() -> None:
    words = [
        Word(index * 0.5, index * 0.5 + 0.4, f"word{index}")
        for index in range(12)
    ]
    cue = Cue(
        words[0].start,
        words[-1].end,
        " ".join(word.text for word in words),
        " ".join(["כתובית"] * 18),
        words,
    )

    split = enforce_reading_speed([cue])

    assert len(split) > 1
    assert [word for part in split for word in part.words] == words
    assert all(part.end - part.start >= 0.8 for part in split)
    assert all(left.end <= right.start for left, right in zip(split, split[1:]))


def test_cue_without_word_metadata_uses_proportional_timing() -> None:
    cue = Cue(0.0, 4.0, "one two three four", "א ב ג ד ה ו ז ח ט")

    parts = enforce_reading_speed([cue], max_cps=4.1)

    assert [part.source for part in parts] == ["one two", "three", "four"]
    assert [(part.start, part.end) for part in parts] == [(0.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def test_short_dense_cue_is_split_and_marked_when_still_too_fast() -> None:
    cue = Cue(0.0, 1.7, "one two three four", "טקסט ארוך מאוד שלא ניתן לקרוא בזמן")

    split = enforce_reading_speed([cue], max_cps=4.0)

    assert len(split) == 2
    assert all(part.end - part.start >= 0.8 for part in split)
    assert all(part.cps_forced for part in split)


def test_short_trailing_cue_merges_after_speed_splitting() -> None:
    cues = [
        Cue(0.0, 1.6, "a complete cue", "כתובית מלאה"),
        Cue(1.7, 2.4, "short ending", "סוף קצר"),
    ]

    merged = enforce_reading_speed(cues)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 2.4
    assert merged[0].display_text == "כתובית מלאה סוף קצר"


def test_available_gaps_are_redistributed_to_remove_cps_violations() -> None:
    cues = [
        Cue(0.0, 1.0, "first", "א" * 20),
        Cue(3.0, 4.0, "second", "ב" * 20),
    ]

    adjusted = enforce_reading_speed(cues, max_cps=10.0)

    assert [(cue.start, cue.end) for cue in adjusted] == [(0.0, 2.0), (2.0, 4.0)]
    assert all(characters_per_second(cue) <= 10.0 for cue in adjusted)


def test_cue_borrows_leading_silence_and_following_gap() -> None:
    cues = [
        Cue(1.0, 2.0, "first cue", "א" * 20),
        Cue(4.0, 5.0, "second cue", "ב" * 5),
    ]

    adjusted = enforce_reading_speed(cues, max_cps=10.0)

    assert adjusted[0].start == pytest.approx(0.5)
    assert adjusted[0].end == pytest.approx(2.5)
    assert adjusted[0].cps_forced is False
