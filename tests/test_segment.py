from hs.models import Word
from hs.segment import segment_words


def make_words(texts: list[str], step: float = 0.45, gap_after: dict[int, float] | None = None):
    words = []
    cursor = 0.0
    for index, text in enumerate(texts):
        words.append(Word(cursor, cursor + 0.35, text, 0.9))
        cursor += step + (gap_after or {}).get(index, 0.0)
    return words


def test_sentence_punctuation_creates_word_aligned_boundary() -> None:
    words = make_words(["This", "is", "one.", "This", "is", "two."])
    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["This is one.", "This is two."]
    assert cues[0].start == words[0].start
    assert cues[0].end == words[2].end
    assert cues[1].start == words[3].start


def test_long_word_is_never_cut() -> None:
    long_word = "x" * 100
    cues = segment_words(make_words([long_word, "after"]))

    assert cues[0].source == long_word
    assert "".join(cue.source.replace(" ", "") for cue in cues) == long_word + "after"


def test_three_second_gap_forces_boundary() -> None:
    words = make_words(["before", "pause", "after", "pause"], gap_after={1: 3.0})
    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["before pause", "after pause"]
    assert cues[1].start - cues[0].end >= 3.0


def test_two_hundred_words_respect_duration_and_word_boundaries() -> None:
    words = make_words([f"word{i}" for i in range(200)])
    cues = segment_words(words)

    flattened = [word for cue in cues for word in cue.words]
    assert flattened == words
    assert all(cue.end - cue.start <= 7.0 for cue in cues)
    assert all(cue.start == cue.words[0].start and cue.end == cue.words[-1].end for cue in cues)
