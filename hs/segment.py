import re

from hs.models import Cue, Word


SENTENCE_END = re.compile(r"[.?!:׃][\"'”’)]*$")
COMMA_END = re.compile(r"[,;،][\"'”’)]*$")


def segment_words(
    words: list[Word],
    max_chars: int = 42,
    max_lines: int = 2,
    max_duration: float = 7.0,
    min_duration: float = 1.0,
    merge_gap: float = 0.35,
) -> list[Cue]:
    if not words:
        return []

    groups: list[list[Word]] = []
    current: list[Word] = []

    for word in words:
        if current and _must_break_before(
            current, word, max_chars, max_lines, max_duration, merge_gap
        ):
            groups.append(current)
            current = []

        current.append(word)
        duration = current[-1].end - current[0].start
        text_length = len(_join_words(current))
        if duration >= min_duration and (
            SENTENCE_END.search(word.text)
            or (COMMA_END.search(word.text) and (text_length >= max_chars or duration >= 3.0))
        ):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    groups = _merge_short_groups(
        groups, min_duration, max_duration, max_chars, max_lines, merge_gap
    )
    return [
        Cue(start=group[0].start, end=group[-1].end, source=_join_words(group), words=group)
        for group in groups
    ]


def _must_break_before(
    current: list[Word],
    following: Word,
    max_chars: int,
    max_lines: int,
    max_duration: float,
    merge_gap: float,
) -> bool:
    if following.start - current[-1].end >= merge_gap:
        return True
    candidate = [*current, following]
    return (
        not _fits_lines(candidate, max_chars, max_lines)
        or candidate[-1].end - candidate[0].start > max_duration
    )


def _merge_short_groups(
    groups: list[list[Word]],
    min_duration: float,
    max_duration: float,
    max_chars: int,
    max_lines: int,
    merge_gap: float,
) -> list[list[Word]]:
    merged: list[list[Word]] = []
    for group in groups:
        if merged and group[-1].end - group[0].start < min_duration:
            candidate = [*merged[-1], *group]
            gap = group[0].start - merged[-1][-1].end
            if (
                gap < merge_gap
                and candidate[-1].end - candidate[0].start <= max_duration
                and _fits_lines(candidate, max_chars, max_lines)
            ):
                merged[-1] = candidate
                continue
        merged.append(group)

    if len(merged) > 1 and merged[0][-1].end - merged[0][0].start < min_duration:
        candidate = [*merged[0], *merged[1]]
        if (
            merged[1][0].start - merged[0][-1].end < merge_gap
            and candidate[-1].end - candidate[0].start <= max_duration
            and _fits_lines(candidate, max_chars, max_lines)
        ):
            merged[:2] = [candidate]
    return merged


def _join_words(words: list[Word]) -> str:
    text = " ".join(word.text.strip() for word in words).strip()
    text = re.sub(r"(?<=\d)\s+([.,])\s*(?=\d)", r"\1", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _fits_lines(words: list[Word], max_chars: int, max_lines: int) -> bool:
    text_words = [item.text.strip() for item in words if item.text.strip()]
    if len(text_words) > 1 and any(len(word) > max_chars for word in text_words):
        return False

    lines = 1
    line_length = 0
    for word in text_words:
        separator = 1 if line_length else 0
        if line_length and line_length + separator + len(word) > max_chars:
            lines += 1
            line_length = len(word)
        else:
            line_length += separator + len(word)
    return lines <= max_lines
