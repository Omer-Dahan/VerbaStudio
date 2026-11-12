from dataclasses import replace
import re

from hs.models import Cue, Word
from hs.text import normalize_subtitle_text


MIN_PIECE_DURATION = 0.8
MAX_CUE_DURATION = 7.0
SENTENCE_BOUNDARY = re.compile(r'[.!?;:]["”’)]?$')
COMMA_BOUNDARY = re.compile(r'[,،]["”’)]?$')
LATIN_NAME = re.compile(r"[A-Z][A-Za-z0-9'’-]*$")


def enforce_reading_speed(
    cues: list[Cue],
    max_cps: float = 17.0,
    min_cue_duration: float = 0.40,
    max_chars: int = 42,
) -> list[Cue]:
    adjusted: list[Cue] = []
    for cue in cues:
        if cue.text is not None:
            cue = replace(cue, text=normalize_subtitle_text(cue.text))
        if adjusted and _needs_more_time(cue, max_cps, min_cue_duration):
            candidate = _merge_pair(adjusted[-1], cue)
            if candidate.end - candidate.start <= MAX_CUE_DURATION:
                if _fits_line_budget(candidate.display_text, max_chars):
                    adjusted[-1] = candidate
                else:
                    adjusted[-1:] = _split_to_line_budget(candidate, max_chars)
                continue
        if cue.end - cue.start < min_cue_duration:
            adjusted.append(cue)
        else:
            adjusted.extend(_split_until_readable(cue, max_cps))
    retimed = _borrow_available_time(adjusted, max_cps)
    line_bounded = [
        part
        for cue in retimed
        for part in _split_to_line_budget(cue, max_chars)
    ]
    return [
        replace(cue, cps_forced=characters_per_second(cue) > max_cps)
        for cue in line_bounded
    ]


def _needs_more_time(cue: Cue, max_cps: float, min_cue_duration: float) -> bool:
    return (
        cue.end - cue.start < min_cue_duration
        or characters_per_second(cue) > max_cps
    )


def _fits_line_budget(text: str, max_chars: int, max_lines: int = 2) -> bool:
    lines = 1
    line_length = 0
    for word in text.replace("\n", " ").split():
        if len(word) > max_chars:
            return False
        separator = 1 if line_length else 0
        if line_length and line_length + separator + len(word) > max_chars:
            lines += 1
            line_length = len(word)
        else:
            line_length += separator + len(word)
    return lines <= max_lines


def _split_to_line_budget(cue: Cue, max_chars: int) -> list[Cue]:
    display_tokens = _text_tokens(cue.display_text)
    if len(display_tokens) < 2:
        return [cue]

    chunks: list[list[str]] = []
    current: list[str] = []
    for token in display_tokens:
        candidate = [*current, token]
        if current and not _fits_line_budget(" ".join(candidate), max_chars):
            chunks.append(current)
            current = [token]
        else:
            current = candidate
    chunks.append(current)
    if len(chunks) == 1:
        return [cue]

    source_tokens = cue.source.split()
    duration = cue.end - cue.start
    parts: list[Cue] = []
    display_offset = 0
    source_offset = 0
    word_offset = 0
    for index, chunk in enumerate(chunks):
        display_end = display_offset + len(chunk)
        end_ratio = display_end / len(display_tokens)
        source_end = (
            len(source_tokens)
            if index == len(chunks) - 1
            else max(source_offset + 1, round(len(source_tokens) * end_ratio))
        )
        word_end = (
            len(cue.words)
            if index == len(chunks) - 1
            else max(word_offset, round(len(cue.words) * end_ratio))
        )
        start = cue.start + duration * (display_offset / len(display_tokens))
        end = cue.start + duration * end_ratio
        parts.append(
            Cue(
                start=start,
                end=end,
                source=" ".join(source_tokens[source_offset:source_end]),
                text=" ".join(chunk),
                words=cue.words[word_offset:word_end],
                translation_failed=cue.translation_failed,
                translation_passthrough=cue.translation_passthrough,
            )
        )
        display_offset = display_end
        source_offset = source_end
        word_offset = word_end
    return parts


def characters_per_second(cue: Cue) -> float:
    duration = cue.end - cue.start
    return len(cue.display_text.replace("\n", "")) / duration if duration > 0 else float("inf")


def adaptive_cps_target(cues: list[Cue], duration: float) -> float:
    source_length = len(" ".join(cue.source.strip() for cue in cues).strip())
    source_cps = source_length / duration if duration > 0 else 0.0
    return min(24.0, max(17.0, source_cps * 1.05))


def word_sequence_is_complete(cues: list[Cue], transcript: list[Word]) -> bool:
    return [word for cue in cues for word in cue.words] == transcript


def normalize_sentence_cues(cues: list[Cue]) -> list[Cue]:
    return [
        replace(cue, text=normalize_subtitle_text(cue.text)) if cue.text is not None else cue
        for cue in cues
    ]


def extend_clause_durations(cues: list[Cue], minimum: float = 1.5) -> list[Cue]:
    extended: list[Cue] = []
    for index, cue in enumerate(cues):
        next_start = cues[index + 1].start if index + 1 < len(cues) else cue.end
        end = min(next_start, max(cue.end, cue.start + minimum))
        extended.append(replace(cue, end=end))
    return extended


def normalize_cues(cues: list[Cue], min_cue_duration: float = 0.40) -> list[Cue]:
    deduplicated: list[Cue] = []
    for cue in cues:
        if deduplicated and cue.text:
            cue = replace(cue, text=_strip_duplicate_tail(deduplicated[-1].display_text, cue.text))
        deduplicated.append(cue)
    return _merge_source_orphans(deduplicated)


def _split_until_readable(cue: Cue, max_cps: float) -> list[Cue]:
    if characters_per_second(cue) <= max_cps:
        return [cue]

    split = _split_cue(cue)
    if split is None:
        return [cue]
    left, right = split
    return [
        *_split_until_readable(left, max_cps),
        *_split_until_readable(right, max_cps),
    ]


def _split_cue(cue: Cue) -> tuple[Cue, Cue] | None:
    display_words = _text_tokens(cue.display_text)
    if len(display_words) < 2:
        return None

    duration = cue.end - cue.start
    if duration < MIN_PIECE_DURATION * 2:
        return None

    source_words = cue.source.split()
    if len(source_words) < 2:
        return None
    source_index = _balanced_index(len(source_words), duration)
    if source_index is None:
        return None
    boundary_ratio = source_index / len(source_words)
    boundary = cue.start + duration * boundary_ratio

    if cue.words and len(cue.words) >= 2:
        word_index = min(len(cue.words) - 1, max(1, round(len(cue.words) * boundary_ratio)))
        left_words = cue.words[:word_index]
        right_words = cue.words[word_index:]
        display_index = _preferred_text_boundary(display_words, boundary_ratio)
        return (
            Cue(
                start=cue.start,
                end=boundary,
                source=" ".join(source_words[:source_index]),
                text=" ".join(display_words[:display_index]),
                words=left_words,
                translation_failed=cue.translation_failed,
                translation_passthrough=cue.translation_passthrough,
            ),
            Cue(
                start=boundary,
                end=cue.end,
                source=" ".join(source_words[source_index:]),
                text=" ".join(display_words[display_index:]),
                words=right_words,
                translation_failed=cue.translation_failed,
                translation_passthrough=cue.translation_passthrough,
            ),
        )

    display_index = _preferred_text_boundary(display_words, boundary_ratio)
    return (
        Cue(
            cue.start,
            boundary,
            " ".join(source_words[:source_index]),
            " ".join(display_words[:display_index]),
            translation_failed=cue.translation_failed,
            translation_passthrough=cue.translation_passthrough,
        ),
        Cue(
            boundary,
            cue.end,
            " ".join(source_words[source_index:]),
            " ".join(display_words[display_index:]),
            translation_failed=cue.translation_failed,
            translation_passthrough=cue.translation_passthrough,
        ),
    )


def _balanced_index(word_count: int, duration: float) -> int | None:
    candidates = [index for index in range(3, word_count - 2) if duration * index / word_count >= MIN_PIECE_DURATION and duration * (word_count - index) / word_count >= MIN_PIECE_DURATION]
    if not candidates:
        return None
    midpoint = word_count / 2
    return min(candidates, key=lambda index: abs(index - midpoint))


def _strip_duplicate_tail(previous: str, current: str) -> str:
    previous_words = previous.split()
    current_words = current.split()
    overlap = 0
    limit = min(len(previous_words), len(current_words))
    for length in range(3, limit + 1):
        left = [_comparison_word(word) for word in previous_words[-length:]]
        right = [_comparison_word(word) for word in current_words[:length]]
        if left == right:
            overlap = length
    return " ".join(current_words[overlap:]).strip()


def _comparison_word(word: str) -> str:
    return word.casefold().strip('.,!?;:"“”‘’()[]')


def _merge_source_orphans(cues: list[Cue]) -> list[Cue]:
    merged: list[Cue] = []
    pending: Cue | None = None
    for cue in cues:
        if len(cue.source.split()) >= 3:
            if pending is not None:
                cue = _merge_pair(pending, cue)
                pending = None
            merged.append(cue)
        elif merged:
            merged[-1] = _merge_pair(merged[-1], cue)
        elif pending is None:
            pending = cue
        else:
            pending = _merge_pair(pending, cue)
    if pending is not None:
        merged.append(pending)
    return merged


def _merge_pair(left: Cue, right: Cue) -> Cue:
    return Cue(
        start=left.start,
        end=right.end,
        source=f"{left.source} {right.source}".strip(),
        text=_join_text(left.display_text, right.display_text),
        words=[*left.words, *right.words],
        translation_failed=left.translation_failed or right.translation_failed,
        translation_passthrough=(
            left.translation_passthrough and right.translation_passthrough
        ),
        cps_forced=left.cps_forced or right.cps_forced,
    )


def _preferred_text_boundary(tokens: list[str], ratio: float) -> int:
    target = len(tokens) * ratio
    candidates = [
        index
        for index in range(1, len(tokens))
        if not (LATIN_NAME.fullmatch(tokens[index - 1]) and LATIN_NAME.fullmatch(tokens[index]))
    ]
    if not candidates:
        candidates = list(range(1, len(tokens)))

    def rank(index: int) -> tuple[int, float]:
        previous = tokens[index - 1]
        priority = 0 if SENTENCE_BOUNDARY.search(previous) else 1 if COMMA_BOUNDARY.search(previous) else 2
        return priority, abs(index - target)

    return min(candidates, key=rank)


def _text_tokens(text: str) -> list[str]:
    tokens = normalize_subtitle_text(text).split()
    atomic: list[str] = []
    for token in tokens:
        if atomic and LATIN_NAME.fullmatch(atomic[-1].split()[-1]) and LATIN_NAME.fullmatch(token):
            atomic[-1] = f"{atomic[-1]} {token}"
        else:
            atomic.append(token)
    return atomic


def _join_text(left: str, right: str) -> str:
    return normalize_subtitle_text(f"{left} {right}".strip())


def _merge_short_cues(cues: list[Cue]) -> list[Cue]:
    merged: list[Cue] = []
    for cue in cues:
        if merged and cue.end - cue.start < MIN_PIECE_DURATION:
            previous = merged[-1]
            merged[-1] = Cue(
                start=previous.start,
                end=cue.end,
                source=f"{previous.source} {cue.source}".strip(),
                text=_join_text(previous.display_text, cue.display_text),
                words=[*previous.words, *cue.words],
                translation_failed=previous.translation_failed or cue.translation_failed,
                translation_passthrough=(
                    previous.translation_passthrough and cue.translation_passthrough
                ),
            )
        else:
            merged.append(cue)

    if len(merged) > 1 and merged[0].end - merged[0].start < MIN_PIECE_DURATION:
        first, following = merged[:2]
        merged[:2] = [
            Cue(
                start=first.start,
                end=following.end,
                source=f"{first.source} {following.source}".strip(),
                text=_join_text(first.display_text, following.display_text),
                words=[*first.words, *following.words],
                translation_failed=first.translation_failed or following.translation_failed,
                translation_passthrough=(
                    first.translation_passthrough and following.translation_passthrough
                ),
            )
        ]
    return merged


def _borrow_available_time(cues: list[Cue], max_cps: float) -> list[Cue]:
    retimed: list[Cue] = []
    for index, cue in enumerate(cues):
        earliest = retimed[-1].end if retimed else 0.0
        latest = cues[index + 1].start if index + 1 < len(cues) else cue.end
        required = len(cue.display_text.replace("\n", "")) / max_cps
        available = latest - earliest
        if required <= available and required > cue.end - cue.start:
            extra = required - (cue.end - cue.start)
            leading = min(cue.start - earliest, extra / 2)
            start = cue.start - leading
            end = min(latest, start + required)
            start = max(earliest, end - required)
            retimed.append(replace(cue, start=start, end=end))
        else:
            retimed.append(cue)
    return retimed
