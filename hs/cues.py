from dataclasses import replace

from hs.models import Cue, Word


MIN_PIECE_DURATION = 0.8


def enforce_reading_speed(cues: list[Cue], max_cps: float = 17.0) -> list[Cue]:
    adjusted: list[Cue] = []
    for cue in cues:
        adjusted.extend(_split_until_readable(cue, max_cps))
    merged = _merge_short_cues(adjusted)
    retimed = _borrow_available_time(merged, max_cps)
    return [
        replace(cue, cps_forced=characters_per_second(cue) > max_cps)
        for cue in retimed
    ]


def characters_per_second(cue: Cue) -> float:
    duration = cue.end - cue.start
    return len(cue.display_text.replace("\n", "")) / duration if duration > 0 else float("inf")


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
    display_words = cue.display_text.split()
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
        display_index = min(
            len(display_words) - 1,
            max(1, round(len(display_words) * boundary_ratio)),
        )
        return (
            Cue(
                start=cue.start,
                end=boundary,
                source=" ".join(source_words[:source_index]),
                text=" ".join(display_words[:display_index]),
                words=left_words,
                translation_failed=cue.translation_failed,
            ),
            Cue(
                start=boundary,
                end=cue.end,
                source=" ".join(source_words[source_index:]),
                text=" ".join(display_words[display_index:]),
                words=right_words,
                translation_failed=cue.translation_failed,
            ),
        )

    display_index = min(len(display_words) - 1, max(1, round(len(display_words) * boundary_ratio)))
    return (
        Cue(
            cue.start,
            boundary,
            " ".join(source_words[:source_index]),
            " ".join(display_words[:display_index]),
            translation_failed=cue.translation_failed,
        ),
        Cue(
            boundary,
            cue.end,
            " ".join(source_words[source_index:]),
            " ".join(display_words[display_index:]),
            translation_failed=cue.translation_failed,
        ),
    )


def _balanced_index(word_count: int, duration: float) -> int | None:
    candidates = [index for index in range(1, word_count) if duration * index / word_count >= MIN_PIECE_DURATION and duration * (word_count - index) / word_count >= MIN_PIECE_DURATION]
    if not candidates:
        return None
    midpoint = word_count / 2
    return min(candidates, key=lambda index: abs(index - midpoint))


def _merge_short_cues(cues: list[Cue]) -> list[Cue]:
    merged: list[Cue] = []
    for cue in cues:
        if merged and cue.end - cue.start < MIN_PIECE_DURATION:
            previous = merged[-1]
            merged[-1] = Cue(
                start=previous.start,
                end=cue.end,
                source=f"{previous.source} {cue.source}".strip(),
                text=f"{previous.display_text} {cue.display_text}".strip(),
                words=[*previous.words, *cue.words],
                translation_failed=previous.translation_failed or cue.translation_failed,
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
                text=f"{first.display_text} {following.display_text}".strip(),
                words=[*first.words, *following.words],
                translation_failed=first.translation_failed or following.translation_failed,
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
