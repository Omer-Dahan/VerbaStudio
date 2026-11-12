import random
import re

import pytest

from hs.ass import _ass_text
from hs.cues import (
    characters_per_second,
    clamp_sentence_cps,
    enforce_reading_speed,
    extend_clause_durations,
    normalize_cues,
    normalize_clause_cues,
    word_sequence_is_complete,
)
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


def test_clause_duration_borrows_only_following_silence() -> None:
    cues = [Cue(0.0, 0.8, "one two"), Cue(2.0, 8.8, "three four")]

    extended = extend_clause_durations(cues, 1.5)

    assert extended[0].end == 1.5
    assert extended[0].end <= extended[1].start
    assert all(cue.end - cue.start <= 7.0 for cue in extended)


def test_cue_without_word_metadata_uses_proportional_timing() -> None:
    cue = Cue(0.0, 4.0, "one two three four five six", "א ב ג ד ה ו ז ח ט")

    parts = enforce_reading_speed([cue], max_cps=4.1)

    assert [part.source for part in parts] == ["one two three", "four five six"]
    assert [(part.start, part.end) for part in parts] == [(0.0, 2.0), (2.0, 4.0)]


def test_short_dense_cue_is_split_and_marked_when_still_too_fast() -> None:
    cue = Cue(0.0, 1.7, "one two three four", "טקסט ארוך מאוד שלא ניתן לקרוא בזמן")

    split = enforce_reading_speed([cue], max_cps=4.0)

    assert len(split) == 1
    assert split[0].cps_forced is True
    assert len(split[0].source.split()) >= 3


def test_short_cue_is_absorbed_by_its_predecessor() -> None:
    cues = [
        Cue(0.0, 1.6, "a complete cue", "כתובית מלאה"),
        Cue(1.7, 2.0, "short ending", "סוף קצר"),
    ]

    merged = enforce_reading_speed(cues)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0
    assert merged[0].display_text == "כתובית מלאה סוף קצר"


def test_dense_cue_uses_backward_merge_by_default() -> None:
    cues = [
        Cue(0.0, 2.0, "A complete thought.", "משפט שלם."),
        Cue(2.0, 2.5, "Dense continuation!", "המשך צפוף מאוד!"),
    ]

    merged = enforce_reading_speed(cues, max_cps=17.0, max_chars=30)

    assert len(merged) == 1
    assert merged[0].source == "A complete thought. Dense continuation!"
    assert merged[0].display_text == "משפט שלם. המשך צפוף מאוד!"
    assert merged[0].end == 2.5


def test_over_budget_backward_merge_splits_combined_text() -> None:
    cues = [
        Cue(0.0, 2.0, "one two three four", "אחד שניים שלושה ארבעה"),
        Cue(2.0, 2.3, "five six seven eight", "חמישה שישה שבעה שמונה"),
    ]

    split = enforce_reading_speed(cues, max_cps=17.0, max_chars=12)

    assert len(split) > 1
    assert " ".join(cue.source for cue in split) == "one two three four five six seven eight"
    assert " ".join(cue.display_text for cue in split) == (
        "אחד שניים שלושה ארבעה חמישה שישה שבעה שמונה"
    )
    assert split[0].start == 0.0
    assert split[-1].end == 2.3
    assert all(left.end == right.start for left, right in zip(split, split[1:]))


def test_backward_merge_only_concatenates_and_reduces_peak_cps() -> None:
    cues = [
        Cue(0.0, 2.8, "Wait—keep this punctuation.", "רגע—שמור את הפיסוק."),
        Cue(2.8, 3.2, "Nothing disappears!", "שום דבר לא נעלם!"),
    ]
    original_peak = max(characters_per_second(cue) for cue in cues)

    merged = enforce_reading_speed(cues, max_cps=19.0, max_chars=40)

    assert [cue.source for cue in merged] == [
        "Wait—keep this punctuation. Nothing disappears!"
    ]
    assert [cue.display_text for cue in merged] == [
        "רגע—שמור את הפיסוק. שום דבר לא נעלם!"
    ]
    assert characters_per_second(merged[0]) < original_peak


def test_very_fast_speech_preserves_every_transcript_word_once() -> None:
    words = [
        Word(index * 0.08, index * 0.08 + 0.07, f"word{index}")
        for index in range(24)
    ]
    cues = [
        Cue(
            words[index].start,
            words[index + 3].end,
            " ".join(word.text for word in words[index:index + 4]),
            " ".join(f"תרגום{offset}" for offset in range(index, index + 4)),
            words[index:index + 4],
        )
        for index in range(0, len(words), 4)
    ]

    normalized = enforce_reading_speed(
        normalize_cues(cues, min_cue_duration=0.40),
        max_cps=19.0,
        min_cue_duration=0.40,
        max_chars=18,
    )

    assert word_sequence_is_complete(normalized, words)
    assert not word_sequence_is_complete(normalized, [*words, words[-1]])


def test_available_gaps_are_redistributed_to_remove_cps_violations() -> None:
    cues = [
        Cue(0.0, 1.0, "first", "א" * 20),
        Cue(3.0, 4.0, "second", "ב" * 20),
    ]

    adjusted = enforce_reading_speed(cues, max_cps=10.0)

    assert [(cue.start, cue.end) for cue in adjusted] == [(0.0, 4.0)]
    assert characters_per_second(adjusted[0]) == pytest.approx(10.25)


def test_cue_borrows_leading_silence_and_following_gap() -> None:
    cues = [
        Cue(1.0, 2.0, "first cue", "א" * 20),
        Cue(4.0, 5.0, "second cue", "ב" * 5),
    ]

    adjusted = enforce_reading_speed(cues, max_cps=10.0)

    assert adjusted[0].start == pytest.approx(0.5)
    assert adjusted[0].end == pytest.approx(2.5)
    assert adjusted[0].cps_forced is False


def test_split_preserves_decimal_and_latin_name_tokens() -> None:
    cue = Cue(
        0.0,
        4.0,
        "one two three four five six",
        'פתיח "מחר בבוקר" גרסה 26.3 עבור Product Name ממשיכה עכשיו',
    )

    parts = enforce_reading_speed([cue], max_cps=12.0)

    text_parts = [part.display_text for part in parts]
    assert any("מחר בבוקר" in text for text in text_parts)
    assert all('"' not in text for text in text_parts)
    assert any("26.3" in text for text in text_parts)
    assert any("Product Name" in text for text in text_parts)
    assert all("26 .3" not in text for text in text_parts)


def test_reading_speed_split_repairs_broken_numeric_token() -> None:
    cue = Cue(
        0.0,
        4.0,
        "one two three four five six",
        "ללא שינוי של '26. .3', בנויה מחדש לגמרי עכשיו",
    )

    parts = enforce_reading_speed([cue], max_cps=8.0)

    assert " ".join(part.display_text for part in parts).count("26.3") == 1
    assert all("26." not in part.display_text or "26.3" in part.display_text for part in parts)


@pytest.mark.parametrize("seed", range(40))
def test_full_pipeline_keeps_generated_numeric_and_latin_tokens_atomic(seed: int) -> None:
    rng = random.Random(seed)
    hebrew = ["הוועדה", "בנויה", "מחדש", "ללא", "שינוי", "ממשיכה"]
    english = ["with", "stable", "release", "for", "users", "today"]
    numeric_tokens = ["26.3", "1,024", "2026-10-30", "2:30", "5%"]
    product_names = ["Java Platform", "Open AI", "Product Name"]
    rng.shuffle(hebrew)
    rng.shuffle(english)
    rng.shuffle(numeric_tokens)
    text = " ".join([
        *hebrew[:3], product_names[seed % len(product_names)], numeric_tokens[0] + ",",
        *english[:3], numeric_tokens[1], hebrew[3], numeric_tokens[2], numeric_tokens[3],
        numeric_tokens[4], *hebrew[4:], *english[3:],
    ])
    split_token = rng.choice([token for token in numeric_tokens if re.search(r"[.,:/-]", token)])
    numeric_start = text.index(split_token)
    separators = [
        offset
        for offset, character in enumerate(split_token, start=1)
        if character in ".,:/-"
    ]
    split_at = numeric_start + rng.choice(separators)
    fragments = [text[:split_at], text[split_at:]]
    cues = [
        Cue(index * 2.0, (index + 1) * 2.0, "source fragment", fragment)
        for index, fragment in enumerate(fragments)
    ]

    output = enforce_reading_speed(
        normalize_cues(cues), max_cps=10.0, max_chars=rng.randrange(12, 25)
    )
    rendered = " ".join(cue.display_text for cue in output)

    assert re.search(r"\d+[.,:/-]\s+\d", rendered) is None
    assert re.search(r"\d\s+[.,:;%]", rendered) is None
    for token in numeric_tokens:
        assert token in rendered
    for cue in output:
        assert not any(
            left.endswith(tuple(".,:/-")) and right[:1].isdigit()
            for left, right in zip(cue.display_text.split(), cue.display_text.split()[1:])
        )


def test_real_java_decimal_fragment_survives_full_pipeline() -> None:
    cues = [
        Cue(0.0, 2.0, "first source fragment", "הוועדה ממשיכה עם Java 26."),
        Cue(2.0, 4.0, "second source fragment", ".3, בנויה מחדש ללא שינוי"),
    ]

    output = enforce_reading_speed(normalize_cues(cues), max_cps=9.0, max_chars=16)
    rendered = " ".join(cue.display_text for cue in output)

    assert "Java 26.3, בנויה מחדש" in rendered
    assert "26. .3" not in rendered
    assert "26.3 ," not in rendered


def test_duplicate_translation_tail_is_removed_from_following_cue() -> None:
    cues = [
        Cue(0.0, 2.0, "one two three", "היום נצא לדרך."),
        Cue(2.0, 4.0, "four five six", "היום נצא לדרך. ואז נמשיך"),
    ]

    normalized = normalize_cues(cues)

    assert normalized[1].display_text == "ואז נמשיך"


def test_source_cue_with_fewer_than_three_words_merges_into_neighbour() -> None:
    cues = [
        Cue(0.0, 0.7, "one two", "אחד שניים"),
        Cue(0.8, 2.5, "three four five", "שלוש ארבע חמש"),
    ]

    normalized = normalize_cues(cues)

    assert len(normalized) == 1
    assert normalized[0].source == "one two three four five"
    assert normalized[0].display_text == "אחד שניים שלוש ארבע חמש"


def test_short_cue_is_never_dropped_by_normalizer() -> None:
    cue = Cue(1.0, 1.1, "brief", "קצר")

    assert normalize_cues([cue], min_cue_duration=0.40) == [cue]


def test_short_clause_extends_into_following_gap() -> None:
    cues = [
        Cue(0.0, 0.4, "one two three", "אחת שתיים שלוש"),
        Cue(2.4, 4.0, "four five six", "ארבע חמש שש"),
    ]

    normalized = normalize_clause_cues(cues, minimum=1.5, max_chars=20)

    assert len(normalized) == 2
    assert normalized[0].end == pytest.approx(1.5)
    assert normalized[1].start == pytest.approx(2.4)


def test_short_clause_stays_word_aligned_when_minimum_is_disabled() -> None:
    cue = Cue(1.0, 1.4, "one two", "אחת שתיים")

    normalized = normalize_clause_cues([cue], minimum=0, max_chars=20)

    assert normalized == [cue]
    assert normalized[0].end - normalized[0].start == pytest.approx(0.4)


def test_short_clause_without_gap_merges_forward() -> None:
    cues = [
        Cue(0.0, 0.4, "one two three", "אחת שתיים שלוש"),
        Cue(0.4, 2.0, "four five six", "ארבע חמש שש"),
    ]

    normalized = normalize_clause_cues(cues, minimum=1.5, max_chars=20)

    assert len(normalized) == 1
    assert normalized[0].start == 0.0
    assert normalized[0].end == 2.0


def test_short_clause_merges_forward_even_when_text_exceeds_two_lines() -> None:
    cues = [
        Cue(0.0, 1.5, "one two three", "פתיח קצר"),
        Cue(1.5, 1.9, "four five six", "קטע קצר"),
        Cue(1.9, 3.9, "seven eight nine", "מילים ארוכות מאוד שלא ייכנסו קדימה"),
    ]

    normalized = normalize_clause_cues(cues, minimum=1.5, max_chars=10)

    assert len(normalized) == 2
    assert normalized[0].display_text == "פתיח קצר"
    assert normalized[1].display_text == "קטע קצר מילים ארוכות מאוד שלא ייכנסו קדימה"
    assert normalized[1].end == 3.9


def test_long_translated_cue_is_split_to_at_most_two_lines() -> None:
    translated = " ".join(f"מילה{index}" for index in range(24))
    source = " ".join(f"word{index}" for index in range(24))
    cue = Cue(0.0, 12.0, source, translated)

    output = enforce_reading_speed([cue], max_cps=100.0, max_chars=18)
    line_counts = [_ass_text(part.display_text, 18).count(r"\N") + 1 for part in output]

    assert " ".join(part.display_text for part in output) == translated
    assert max(line_counts) <= 2


def test_clamp_sentence_cps_borrows_from_slack_neighbor_to_fix_tiny_clause() -> None:
    cues = [
        Cue(0.0, 3.0, "a long calm first clause", "א" * 5),
        Cue(3.0, 3.42, "ok", "א" * 14),
        Cue(3.42, 6.42, "a long calm third clause", "א" * 5),
    ]

    clamped = clamp_sentence_cps(cues, cps_target=17.0)

    assert clamped[0].start == 0.0
    assert clamped[-1].end == 6.42
    assert clamped[0].end <= clamped[1].start
    assert clamped[1].end <= clamped[2].start
    assert characters_per_second(clamped[1]) <= 17.0 * 1.6 + 1e-6
    assert clamped[1].end - clamped[1].start > 0.42
    assert clamped[0].end - clamped[0].start >= 1.0
    assert clamped[2].end - clamped[2].start >= 1.0


def test_clamp_sentence_cps_leaves_cue_untouched_when_neighbors_have_no_slack() -> None:
    cues = [
        Cue(0.0, 1.0, "tight first clause", "א" * 3),
        Cue(1.0, 1.42, "ok", "א" * 14),
        Cue(1.42, 2.42, "tight third clause", "א" * 3),
    ]

    clamped = clamp_sentence_cps(cues, cps_target=17.0)

    assert clamped == cues


def test_clamp_sentence_cps_never_widens_past_the_sentence_boundary() -> None:
    cues = [
        Cue(0.0, 0.42, "ok", "א" * 14),
        Cue(0.42, 3.42, "a long calm second clause", "א" * 5),
    ]

    clamped = clamp_sentence_cps(cues, cps_target=17.0)

    assert clamped[0].start == 0.0
    assert clamped[-1].end == 3.42
    assert clamped[0].end <= clamped[1].start
    assert clamped[0].end - clamped[0].start > 0.42
    assert characters_per_second(clamped[0]) <= 17.0 * 1.6 + 1e-6


def test_clamp_sentence_cps_on_last_cue_borrows_without_moving_sentence_end() -> None:
    cues = [
        Cue(0.0, 3.0, "a long calm first clause", "א" * 5),
        Cue(3.0, 3.42, "ok", "א" * 14),
    ]

    clamped = clamp_sentence_cps(cues, cps_target=17.0)

    assert clamped[0].start == 0.0
    assert clamped[-1].end == 3.42
    assert clamped[0].end <= clamped[1].start
    assert clamped[1].end - clamped[1].start > 0.42
    assert characters_per_second(clamped[1]) <= 17.0 * 1.6 + 1e-6
