from hs.models import Word
from hs.segment import segment_sentences, segment_words


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

    assert [cue.source for cue in cues] == ["This is one. This is two."]
    assert cues[0].start == words[0].start
    assert cues[0].end == words[-1].end


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


def test_segmenter_preserves_every_source_word_when_line_limits_split() -> None:
    words = make_words(
        "but the case feels like it is going to fall apart in a week without warning".split()
    )

    cues = segment_words(words, max_chars=18, max_lines=2)

    assert " ".join(cue.source for cue in cues) == " ".join(word.text for word in words)
    assert [word for cue in cues for word in cue.words] == words
    assert len(cues) > 1


def test_character_limit_is_applied_to_each_line() -> None:
    words = make_words(["1234567890", "abcdefghij", "klmnopqrst"])

    cues = segment_words(words, max_chars=10, max_lines=2)

    assert [cue.source for cue in cues] == ["1234567890 abcdefghij", "klmnopqrst"]


def test_decimal_tokens_are_joined_without_an_internal_space() -> None:
    words = make_words(["version", "26", ".3", "ships"])

    cues = segment_words(words)

    assert " ".join(cue.source for cue in cues) == "version 26.3 ships"


def test_short_nonverbal_fragment_merges_with_following_sentence() -> None:
    words = make_words(["Mojang's", "code.", "It", "powers", "the", "game."])

    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["Mojang's code. It powers the game."]


def test_lowercase_and_unpunctuated_fragments_merge_before_translation() -> None:
    words = make_words(
        ["In", "the", "tests,", "developers", "found", "it", "worked."]
    )

    cues = segment_words(words, max_chars=40)

    assert [cue.source for cue in cues] == [
        "In the tests, developers found it worked."
    ]


def test_lowercase_fragment_merges_with_preceding_sentence() -> None:
    words = make_words(
        ["This", "ends.", "and", "this", "continues."]
    )

    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["This ends. and this continues."]


def test_lowercase_fragment_with_product_name_merges_with_previous_cue() -> None:
    words = make_words(
        ["They", "removed", "it.", "your", "old", "Java", "mods"], step=0.65
    )

    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["They removed it. your old Java mods"]


def test_bare_name_remains_a_separate_cue() -> None:
    words = make_words(
        ["Install", "the", "game", "on", "your", "computer.", "Minecraft"],
        step=0.65,
    )

    cues = segment_words(words)

    assert [cue.source for cue in cues] == [
        "Install the game on your computer.",
        "Minecraft",
    ]


def test_short_clause_with_subject_and_verb_remains_separate() -> None:
    words = make_words(
        ["It", "works.", "Developers", "test", "every", "build."], step=0.65
    )

    cues = segment_words(words)

    assert [cue.source for cue in cues] == ["It works. Developers test every build."]


def test_merged_fragment_splits_at_word_boundary_to_fit_line_budget() -> None:
    words = make_words(
        ["Mojang's", "code.", "Developers", "use", "this", "system", "every", "day."]
    )

    cues = segment_words(words, max_chars=12, max_lines=2)

    assert [word for cue in cues for word in cue.words] == words
    assert all(
        all(len(line) <= 12 for line in _wrap_source(cue.source, 12))
        for cue in cues
    )


def test_sentence_segmenter_closes_on_terminal_punctuation_once_minimum_duration_reached() -> None:
    words = make_words(["This", "is", "one", "sentence.", "This", "is", "two."])

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["This is one sentence.", "This is two."]
    assert cues[0].start == words[0].start
    assert cues[0].end == words[3].end
    assert cues[0].end - cues[0].start >= 1.2
    assert cues[1].start == words[4].start
    assert cues[1].end == words[-1].end


def test_sentence_segmenter_closes_on_long_pause_without_punctuation() -> None:
    words = make_words(
        ["one", "two", "three", "four", "five", "six"], gap_after={2: 0.45}
    )

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["one two three", "four five six"]
    assert cues[1].start - cues[0].end >= 0.40


def test_sentence_shorter_than_minimum_merges_into_following_sentence() -> None:
    words = make_words(["Ok.", "This", "is", "fine."])

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["Ok. This is fine."]
    assert cues[0].end - cues[0].start >= 1.2


def test_sentence_segmenter_preserves_every_word_across_boundaries() -> None:
    words = make_words(
        ["This", "ends.", "And", "this", "one", "keeps", "going", "without", "stopping."]
    )

    cues = segment_sentences(words)

    assert [word for cue in cues for word in cue.words] == words
    assert " ".join(cue.source for cue in cues) == " ".join(word.text for word in words)


def test_long_sentence_without_pauses_is_capped_at_the_largest_internal_gap() -> None:
    words = make_words([f"word{i}" for i in range(40)], gap_after={20: 0.25})

    cues = segment_sentences(words)

    assert len(cues) == 2
    assert cues[0].words[-1] is words[20]
    assert cues[1].words[0] is words[21]
    assert all(cue.end - cue.start <= 14.0 for cue in cues)
    assert [word for cue in cues for word in cue.words] == words


def test_long_sentence_with_no_internal_pause_is_capped_at_the_duration_limit() -> None:
    words = [Word(index * 0.3, index * 0.3 + 0.3, f"w{index}") for index in range(50)]

    cues = segment_sentences(words)

    assert len(cues) == 2
    assert all(cue.end - cue.start <= 14.0 for cue in cues)
    assert [word for cue in cues for word in cue.words] == words


def test_segment_sentences_returns_empty_for_no_words() -> None:
    assert segment_sentences([]) == []


def _wrap_source(text: str, max_chars: int) -> list[str]:
    lines = [""]
    for word in text.split():
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and len(candidate) > max_chars:
            lines.append(word)
        else:
            lines[-1] = candidate
    return lines
