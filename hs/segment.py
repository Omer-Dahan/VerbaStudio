import re

from hs.models import Cue, Word


SENTENCE_END = re.compile(r"[.?!:׃][\"'”’)]*$")
COMMA_END = re.compile(r"[,;،][\"'”’)]*$")
TERMINAL_PUNCTUATION = re.compile(r"[.!?][\"'”’)]*$")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
URL = re.compile(r"(?:https?://|www\.)\S+$", re.IGNORECASE)
ENGLISH_VERB = re.compile(
    r"^(?:am|are|be|been|being|can|could|did|do|does|had|has|have|is|may|might|"
    r"must|shall|should|was|were|will|would|\w+(?:ed|ing))$",
    re.IGNORECASE,
)
SUBJECT_PRONOUNS = {"he", "i", "it", "she", "they", "we", "you"}

SENTENCE_TERMINATOR = re.compile(r"[.!?…][\"'”’)]*$")
MIN_SENTENCE_DURATION = 1.2
SENTENCE_PAUSE_GAP = 0.40
MAX_SENTENCE_DURATION = 14.0
MAX_SENTENCE_CHARS = 180


def segment_sentences(words: list[Word]) -> list[Cue]:
    if not words:
        return []

    groups = _split_into_sentences(words)
    groups = [part for group in groups for part in _split_oversized_sentence(group)]
    groups = _merge_short_sentences(groups)
    return [
        Cue(start=group[0].start, end=group[-1].end, source=_join_words(group), words=group)
        for group in groups
    ]


def _split_into_sentences(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for index, word in enumerate(words):
        current.append(word)
        duration = current[-1].end - current[0].start
        if duration < MIN_SENTENCE_DURATION:
            continue
        is_last = index == len(words) - 1
        gap = None if is_last else words[index + 1].start - word.end
        if SENTENCE_TERMINATOR.search(word.text) or (gap is not None and gap >= SENTENCE_PAUSE_GAP):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _split_oversized_sentence(words: list[Word]) -> list[list[Word]]:
    duration = words[-1].end - words[0].start
    text_length = len(_join_words(words))
    if duration <= MAX_SENTENCE_DURATION and text_length <= MAX_SENTENCE_CHARS:
        return [words]
    if len(words) < 2:
        return [words]

    gaps = [(words[index + 1].start - words[index].end, index) for index in range(len(words) - 1)]
    max_gap, split_index = max(gaps, key=lambda item: item[0])
    if max_gap <= 0:
        split_index = _cap_boundary_index(words)

    left = words[: split_index + 1]
    right = words[split_index + 1 :]
    if not left or not right:
        return [words]
    return [*_split_oversized_sentence(left), *_split_oversized_sentence(right)]


def _cap_boundary_index(words: list[Word]) -> int:
    start = words[0].start
    length = 0
    for index, word in enumerate(words):
        length += len(word.text) + 1
        if word.end - start >= MAX_SENTENCE_DURATION or length >= MAX_SENTENCE_CHARS:
            return max(0, index - 1) if index > 0 else 0
    return len(words) - 2


def _merge_short_sentences(groups: list[list[Word]]) -> list[list[Word]]:
    merged: list[list[Word]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        duration = group[-1].end - group[0].start
        if duration < MIN_SENTENCE_DURATION:
            if index + 1 < len(groups):
                groups[index + 1] = [*group, *groups[index + 1]]
                index += 1
                continue
            if merged:
                merged[-1] = [*merged[-1], *group]
                index += 1
                continue
        merged.append(group)
        index += 1
    return merged


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
    groups = _merge_fragment_groups(
        groups, max_chars, max_lines, max_duration, merge_gap
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
        if (
            merged
            and group[-1].end - group[0].start < min_duration
            and not _is_bare_passthrough(_join_words(group))
        ):
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

    if (
        len(merged) > 1
        and merged[0][-1].end - merged[0][0].start < min_duration
        and not _is_bare_passthrough(_join_words(merged[0]))
    ):
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


def _merge_fragment_groups(
    groups: list[list[Word]],
    max_chars: int,
    max_lines: int,
    max_duration: float,
    merge_gap: float,
) -> list[list[Word]]:
    merged: list[list[Word]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        if (
            _is_sentence_fragment(group)
            and index + 1 < len(groups)
            and _groups_can_merge(group, groups[index + 1], max_duration, merge_gap)
        ):
            merged.extend(
                _split_group_to_budget([*group, *groups[index + 1]], max_chars, max_lines)
            )
            index += 2
            continue
        if (
            _is_sentence_fragment(group)
            and merged
            and _groups_can_merge(merged[-1], group, max_duration, merge_gap)
        ):
            previous = merged.pop()
            merged.extend(_split_group_to_budget([*previous, *group], max_chars, max_lines))
        else:
            merged.append(group)
        index += 1
    return merged


def _groups_can_merge(
    left: list[Word], right: list[Word], max_duration: float, merge_gap: float
) -> bool:
    return (
        right[0].start - left[-1].end < merge_gap
        and right[-1].end - left[0].start <= max_duration
    )


def _is_sentence_fragment(words: list[Word]) -> bool:
    text = _join_words(words)
    source_words = text.split()
    if _is_bare_passthrough(text):
        return False
    first_letter = next((character for character in text if character.isalpha()), "")
    starts_lowercase = bool(first_letter and first_letter.islower())
    has_verb = (
        len(source_words) >= 2
        and source_words[0].strip(".,;:!?\"'()[]").casefold() in SUBJECT_PRONOUNS
    ) or any(
        ENGLISH_VERB.fullmatch(word.strip(".,;:!?\"'()[]")) for word in source_words
    )
    return (
        starts_lowercase
        or TERMINAL_PUNCTUATION.search(text) is None
        or len(source_words) < 5
    )


def _is_bare_passthrough(text: str) -> bool:
    cleaned = text.strip()
    return (
        not LETTER.search(cleaned)
        or URL.fullmatch(cleaned) is not None
        or len(cleaned.split()) == 1
    )


def _split_group_to_budget(
    words: list[Word], max_chars: int, max_lines: int
) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    remaining = words
    while remaining:
        split_at = len(remaining)
        while split_at > 1 and not _fits_lines(
            remaining[:split_at], max_chars, max_lines
        ):
            split_at -= 1
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks
