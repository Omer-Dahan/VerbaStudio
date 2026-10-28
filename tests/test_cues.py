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
    assert all(part.start == part.words[0].start for part in split)
    assert all(part.end == part.words[-1].end for part in split)
    assert all(part.end - part.start >= 1.0 for part in split)
    assert all(left.end <= right.start for left, right in zip(split, split[1:]))


def test_cue_without_word_metadata_uses_proportional_timing() -> None:
    cue = Cue(0.0, 4.0, "one two three four", "א ב ג ד ה ו ז ח ט")

    left, right = enforce_reading_speed([cue], max_cps=4.1)

    assert left.end == pytest.approx(4.0 * 4 / 9)
    assert right.start == left.end
    assert left.end - left.start >= 1.0
    assert right.end - right.start >= 1.0


def test_short_cue_is_not_split_below_minimum_duration() -> None:
    cue = Cue(0.0, 1.5, "short", "טקסט ארוך מאוד שלא ניתן לפצל בבטחה")

    assert enforce_reading_speed([cue], max_cps=4.0) == [cue]
    assert characters_per_second(cue) > 4.0


def test_short_trailing_cue_merges_after_speed_splitting() -> None:
    cues = [
        Cue(0.0, 1.6, "a complete cue", "כתובית מלאה"),
        Cue(1.7, 2.6, "short ending", "סוף קצר"),
    ]

    merged = enforce_reading_speed(cues)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 2.6
    assert merged[0].display_text == "כתובית מלאה סוף קצר"


def test_available_gaps_are_redistributed_to_remove_cps_violations() -> None:
    cues = [
        Cue(0.0, 1.0, "first", "א" * 20),
        Cue(3.0, 4.0, "second", "ב" * 20),
    ]

    adjusted = enforce_reading_speed(cues, max_cps=10.0)

    assert [(cue.start, cue.end) for cue in adjusted] == [(0.0, 2.0), (2.0, 4.0)]
    assert all(characters_per_second(cue) <= 10.0 for cue in adjusted)
