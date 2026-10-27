from hs.models import Cue, Word


def enforce_reading_speed(cues: list[Cue], max_cps: float = 17.0) -> list[Cue]:
    adjusted: list[Cue] = []
    for cue in cues:
        adjusted.extend(_split_until_readable(cue, max_cps))
    return adjusted


def characters_per_second(cue: Cue) -> float:
    duration = cue.end - cue.start
    return len(cue.display_text.replace("\n", "")) / duration if duration > 0 else float("inf")


def _split_until_readable(cue: Cue, max_cps: float) -> list[Cue]:
    if characters_per_second(cue) <= max_cps or cue.end - cue.start < 2.0:
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

    if cue.words and len(cue.words) >= 2:
        split_index = _best_word_boundary(cue.words)
        if split_index is None:
            return None
        left_words = cue.words[:split_index]
        right_words = cue.words[split_index:]
        boundary_ratio = split_index / len(cue.words)
        display_index = min(
            len(display_words) - 1,
            max(1, round(len(display_words) * boundary_ratio)),
        )
        source_words = cue.source.split()
        source_index = min(
            len(source_words) - 1,
            max(1, round(len(source_words) * boundary_ratio)),
        )
        return (
            Cue(
                start=left_words[0].start,
                end=left_words[-1].end,
                source=" ".join(source_words[:source_index]),
                text=" ".join(display_words[:display_index]),
                words=left_words,
                translation_failed=cue.translation_failed,
            ),
            Cue(
                start=right_words[0].start,
                end=right_words[-1].end,
                source=" ".join(source_words[source_index:]),
                text=" ".join(display_words[display_index:]),
                words=right_words,
                translation_failed=cue.translation_failed,
            ),
        )

    display_index = len(display_words) // 2
    ratio = display_index / len(display_words)
    boundary = cue.start + (cue.end - cue.start) * ratio
    if boundary - cue.start < 1.0 or cue.end - boundary < 1.0:
        return None
    source_words = cue.source.split()
    source_index = min(len(source_words) - 1, max(1, round(len(source_words) * ratio)))
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


def _best_word_boundary(words: list[Word]) -> int | None:
    candidates = [
        index
        for index in range(1, len(words))
        if words[index - 1].end - words[0].start >= 1.0
        and words[-1].end - words[index].start >= 1.0
    ]
    if not candidates:
        return None
    midpoint = len(words) / 2
    return min(candidates, key=lambda index: abs(index - midpoint))

