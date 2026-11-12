from hs.models import Cue, Word
from hs.segment import (
    segment_clauses,
    segment_sentences,
    segment_words,
    merge_cues_below_min_words,
    count_cues_below_min_words,
    split_clause_by_word_limit,
)


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


def test_max_cue_words_splits_at_pauses_without_losing_words() -> None:
    words = make_words(
        [f"word{i}" for i in range(24)],
        gap_after={5: 0.8, 14: 1.2, 19: 0.6},
    )
    cue = Cue(
        words[0].start,
        words[-1].end,
        " ".join(word.text for word in words),
        words=words,
    )

    pieces = split_clause_by_word_limit(cue, 10)

    assert [len(piece.words) for piece in pieces] == [6, 9, 9]
    assert [word for piece in pieces for word in piece.words] == words
    assert all(piece.start == piece.words[0].start for piece in pieces)
    assert all(piece.end == piece.words[-1].end for piece in pieces)


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
    words = make_words(["Try", "again.", "Ask", "the", "women?"], step=0.85)

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["Try again.", "Ask the women?"]
    assert cues[0].start == words[0].start
    assert cues[0].end == words[1].end
    assert cues[0].end - cues[0].start >= 1.2
    assert cues[1].start == words[2].start
    assert cues[1].end == words[-1].end


def test_sentence_segmenter_does_not_close_on_pause_without_punctuation() -> None:
    words = make_words(
        ["one", "two", "three", "four", "five", "six"], gap_after={2: 0.45}
    )

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["one two three four five six"]


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


def test_long_sentence_splits_at_largest_pause_and_tail_keeps_terminator() -> None:
    words = make_words([*[f"word{i}" for i in range(39)], "last?"], gap_after={20: 0.25})

    cues = segment_sentences(words)

    assert len(cues) == 2
    assert cues[0].words[-1] is words[20]
    assert cues[1].words[0] is words[21]
    assert cues[-1].words[-1].text == "last?"
    assert cues[-1].source.endswith("last?")
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


def test_sentence_segmenter_never_emits_punctuation_only_cue() -> None:
    words = make_words(["This", "ends", ".", "Another", "ends", "?"])

    cues = segment_sentences(words)

    assert [cue.source for cue in cues] == ["This ends.", "Another ends?"]
    assert all(any(character.isalpha() for character in cue.source) for cue in cues)


def test_clause_segmenter_splits_on_commas_and_pauses() -> None:
    words = make_words(
        "First useful opening clause here, then another useful phrase with a pause and final ending.".split(),
        gap_after={8: 0.45},
    )

    cues = segment_clauses(words, max_chars=100)

    assert cues[0].source.endswith(",")
    assert any(right.start - left.end >= 0.35 for left, right in zip(cues, cues[1:]))
    assert [word for cue in cues for word in cue.words] == words


def test_clause_segmenter_targets_four_to_seven_words_without_singletons() -> None:
    words = make_words("one two three four five six seven eight nine ten eleven twelve thirteen.".split())

    cues = segment_clauses(words)

    assert all(2 <= len(cue.words) <= 7 for cue in cues)
    assert all(cue.words[-1].text.casefold() not in {"and", "the", "to", "of"} for cue in cues[:-1])


def test_one_word_remainder_clause_merges_forward_when_it_fits() -> None:
    words = make_words(["one", "two", "three", "four", "five"])
    cues = [
        Cue(words[0].start, words[0].end, "one", "וזה", [words[0]]),
        Cue(words[1].start, words[-1].end, "two three four five", "מתאים קדימה", words[1:]),
    ]

    merged = merge_cues_below_min_words(cues, min_words=2, max_chars=100)

    assert len(merged) == 1
    assert merged[0].display_text == "וזה מתאים קדימה"
    assert merged[0].words == words


def test_one_word_remainder_clause_merges_backward_when_forward_would_overflow() -> None:
    words = make_words(["one", "two", "three", "four", "five"], gap_after={2: 7.0})
    cues = [
        Cue(words[0].start, words[1].end, "one two", "אחורה מתאים", words[:2]),
        Cue(words[2].start, words[2].end, "three", "וזה", [words[2]]),
        Cue(words[3].start, words[-1].end, "four five", "קדימה רחוק", words[3:]),
    ]

    merged = merge_cues_below_min_words(cues, min_words=2, max_chars=100)

    assert [cue.display_text for cue in merged] == ["אחורה מתאים וזה", "קדימה רחוק"]
    assert [word for cue in merged for word in cue.words] == words


def test_one_word_sentence_stays_as_one_clause() -> None:
    word = Word(0, 0.5, "Go.")
    cue = Cue(word.start, word.end, word.text, "לך", [word])

    cues = merge_cues_below_min_words([cue], min_words=2)

    assert len(cues) == 1
    assert cues[0] is cue


def test_cues_below_min_words_counts_display_words_and_zero_disables() -> None:
    cues = [Cue(0, 1, "two source", "מילה"), Cue(1, 2, "one", "שתי מילים")]

    assert count_cues_below_min_words(cues, 2) == 1
    assert count_cues_below_min_words(cues, 0) == 0


def test_oversized_clause_splits_at_its_largest_internal_pause() -> None:
    words = make_words(
        ["longword"] * 8 + ["ending."],
        gap_after={3: 0.3},
    )

    cues = segment_clauses(words, max_chars=12)

    assert any(cue.words[-1] is words[3] for cue in cues[:-1])
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))
    assert all(cue.start >= words[0].start and cue.end <= words[-1].end for cue in cues)


def _wrap_source(text: str, max_chars: int) -> list[str]:
    lines = [""]
    for word in text.split():
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and len(candidate) > max_chars:
            lines.append(word)
        else:
            lines[-1] = candidate
    return lines
