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
