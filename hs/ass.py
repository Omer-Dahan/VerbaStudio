from pathlib import Path
import math
import re

from hs.models import Cue
from hs.segment import split_sentence_clauses
from hs.text import normalize_subtitle_text
from hs.translate import split_translation


DEFAULT_FONT = "Ploni DL 1.1 AAA"
LATIN_RUN = re.compile(
    r"(?=[A-Za-z0-9.,:%$+\-/& ]*[A-Za-z])"
    r"[A-Za-z0-9](?:[A-Za-z0-9]|[.,:%$+\-/&](?=[A-Za-z0-9])| (?=[A-Za-z0-9]))*"
)
LRI = "\u2066"
PDI = "\u2069"
RLM = "\u200f"
HEBREW_LETTER = re.compile(r"[\u05d0-\u05ea]")


def subtitle_geometry(
    width: int, height: int, size_scale: float = 1.20
) -> tuple[int, int, int]:
    base_font_size = min(64, max(28, round(height * 0.045)))
    font_size = max(1, round(base_font_size * size_scale))
    margin = max(24, round(width * 0.055))
    max_chars = max(1, math.floor((width - 2 * margin) / (0.52 * font_size)))
    return font_size, margin, max_chars


def write_ass(
    cues: list[Cue],
    destination: Path,
    width: int,
    height: int,
    font: str = DEFAULT_FONT,
    size_scale: float = 1.20,
    box_alpha: float = 0.55,
    outline: float = 2.0,
    caption_band: list[int] | None = None,
    position: str = "auto",
    margin_v_override: int | None = None,
    cue_style: str = "sentences",
    caption_style: str = "box",
    font_explicit: bool = False,
) -> Path:
    alignment, margin_v = subtitle_layout(
        height, caption_band, position, margin_v_override
    )
    font_size, margin_h, max_chars = subtitle_geometry(width, height, size_scale)
    if caption_style == "box":
        boxed = box_alpha > 0
        border_style = 3 if boxed else 1
        style_outline = 8 if boxed else outline
        back_colour = f"&H{round((1 - box_alpha) * 255):02X}000000" if boxed else "&H80000000"
        primary_colour = "&H00FFFFFF"
        secondary_colour = "&H00808080" if cue_style == "reveal" else "&H000000FF"
        outline_colour = "&H00000000"
        shadow = 0
        bold = 0
    else:
        base_font_size = min(64, max(28, round(height * 0.045)))
        presets = {
            "capcut": (
                "Heebo Black", 1.35, 0, 1, "&H00FFFFFF", "&H0000FFFF",
                "&H00000000", "&H80000000", 0.11, 0.05, 0.28,
            ),
            "card": (
                "Assistant", 1.10, 0, 3, "&H00FFFFFF", "&H000000FF",
                "&H00000000", "&H00000000", 0.62, 0, 0.10,
            ),
            "broadcast": (
                "Heebo", 1.25, 1, 3, "&H00000000", "&H00FFFFFF",
                "&H0000FFFF", "&H0000FFFF", 0.45, 0, 0.09,
            ),
        }
        (
            preset_font, scale, bold, border_style, primary_colour,
            secondary_colour, outline_colour, back_colour, outline_scale,
            shadow_scale, margin_scale,
        ) = presets[caption_style]
        if not font_explicit and font == DEFAULT_FONT:
            font = preset_font
        font_size = round(base_font_size * scale)
        style_outline = max(3 if caption_style == "capcut" else 6, round(font_size * outline_scale))
        shadow = max(1, round(font_size * shadow_scale)) if caption_style == "capcut" else 0
        alignment = 2
        margin_v = round(height * margin_scale)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary_colour},{secondary_colour},{outline_colour},{back_colour},{bold},0,0,0,100,100,0,0,{border_style},{style_outline:g},{shadow:g},{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    if caption_style in {"capcut", "broadcast"}:
        events = [
            event
            for cue in cues
            for event in _highlight_events(cue, max_chars, font_size, caption_style)
        ]
    elif caption_style == "card":
        events = [
            _dialogue(cue.start, cue.end, r"{\fad(120,120)}" + _ass_text(cue.display_text, max_chars, font_size))
            for cue in cues
        ]
    elif cue_style == "reveal":
        events = [
            _dialogue(start, end, _karaoke_text(text, end - start, max_chars, font_size))
            for cue in cues
            for start, end, text in _reveal_groups(cue, max_chars)
        ]
    else:
        events = [
            _dialogue(
                cue.start,
                cue.end,
                _ass_text(cue.display_text, max_chars, font_size),
            )
            for cue in cues
        ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return destination


def _dialogue(start: float, end: float, text: str) -> str:
    return (
        f"Dialogue: 0,{format_timestamp(start)},{format_timestamp(end)},"
        f"Default,,0,0,0,,{{\\q2}}{text}"
    )


def _highlight_events(
    cue: Cue, max_chars: int, font_size: int, caption_style: str
) -> list[str]:
    words = (
        normalize_subtitle_text(cue.display_text)
        .replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
        .split()
    )
    if not words:
        return []

    start_cs = round(cue.start * 100)
    duration_cs = max(0, round(cue.end * 100) - start_cs)
    boundaries = _highlight_boundaries(cue, words, duration_cs)
    highlight = "&H0000FFFF&" if caption_style == "capcut" else "&H00FFFFFF&"
    primary = "&H00FFFFFF&" if caption_style == "capcut" else "&H00000000&"
    pop = (
        r"{\fscx80\fscy80\t(0,80,\fscx108\fscy108)\t(80,150,\fscx100\fscy100)}"
        if caption_style == "capcut"
        else ""
    )
    events: list[str] = []
    for group_start in range(0, len(words), 3):
        group = words[group_start : group_start + 3]
        rtl = _line_is_rtl(group)
        for active_index in range(len(group)):
            rendered = []
            for index, word in enumerate(group):
                isolated = LATIN_RUN.sub(lambda match: f"{LRI}{match.group()}{PDI}", word)
                if index == active_index:
                    isolated = f"{{\\c{highlight}}}{isolated}{{\\c{primary}}}"
                rendered.append(isolated)
            if rtl:
                rendered.reverse()
            text = pop + " ".join(rendered) + RLM
            word_index = group_start + active_index
            events.append(
                _dialogue(
                    (start_cs + boundaries[word_index]) / 100,
                    (start_cs + boundaries[word_index + 1]) / 100,
                    text,
                )
            )
    return events


def _line_is_rtl(words: list[str]) -> bool:
    letters = [character for word in words for character in word if character.isalpha()]
    hebrew_count = sum(bool(HEBREW_LETTER.fullmatch(character)) for character in letters)
    return hebrew_count > 0 and hebrew_count * 2 >= len(letters)


def _highlight_boundaries(cue: Cue, words: list[str], duration_cs: int) -> list[int]:
    if not cue.words or len(cue.words) < len(words):
        return _weighted_boundaries(words, duration_cs)

    cue_start_cs = round(cue.start * 100)
    source_count = len(cue.words)
    display_count = len(words)
    boundaries = []
    for index in range(display_count):
        source_index = min(source_count - 1, round(index * source_count / display_count))
        offset = round(cue.words[source_index].start * 100) - cue_start_cs
        boundaries.append(min(duration_cs, max(0, offset)))

    minimum_window = 4 if duration_cs >= 4 * display_count else 2
    if duration_cs >= minimum_window * display_count:
        for index in range(display_count):
            earliest = boundaries[index - 1] + minimum_window if index else 0
            latest = duration_cs - minimum_window * (display_count - index)
            boundaries[index] = min(latest, max(earliest, boundaries[index]))
    return [*boundaries, duration_cs]


def _weighted_boundaries(words: list[str], duration_cs: int) -> list[int]:
    character_total = sum(len(word) for word in words)
    boundaries = [
        round(duration_cs * sum(len(word) for word in words[:index]) / character_total)
        for index in range(len(words) + 1)
    ]
    if duration_cs >= 4 * len(words):
        for index in range(1, len(boundaries)):
            boundaries[index] = max(boundaries[index], boundaries[index - 1] + 4)
        for index in range(len(boundaries) - 2, -1, -1):
            boundaries[index] = min(boundaries[index], boundaries[index + 1] - 4)
    return boundaries


def _reveal_groups(cue: Cue, max_chars: int) -> list[tuple[float, float, str]]:
    text = " ".join(cue.display_text.split())
    words = text.split()
    if not words:
        return []

    source_clause_count = len(split_sentence_clauses(cue, max_chars)) if cue.words else 1
    clause_pieces = split_translation(text, source_clause_count)
    preferred_boundaries: set[int] = set()
    if clause_pieces is not None:
        offset = 0
        for piece in clause_pieces[:-1]:
            offset += len(piece.split())
            preferred_boundaries.add(offset)

    duration_cs = max(0, round((cue.end - cue.start) * 100))
    minimum_groups = max(math.ceil(len(words) / 3), math.ceil(duration_cs / 250))
    maximum_groups = max(1, len(words) // 2)
    group_count = min(maximum_groups, max(1, minimum_groups))
    sizes = _group_sizes(len(words), group_count, preferred_boundaries)
    groups: list[str] = []
    offset = 0
    for size in sizes:
        group = " ".join(words[offset : offset + size])
        offset += size
        lines, _ = layout_cue_lines(group, max_chars)
        if len(lines) <= 2:
            groups.append(group)
        else:
            groups.extend(words[offset - size : offset])

    weights = [sum(len(word) for word in group.split()) for group in groups]
    boundaries = [round(duration_cs * sum(weights[:index]) / sum(weights)) for index in range(len(groups) + 1)]
    start_cs = round(cue.start * 100)
    return [
        ((start_cs + boundaries[index]) / 100, (start_cs + boundaries[index + 1]) / 100, group)
        for index, group in enumerate(groups)
    ]


def _group_sizes(word_count: int, group_count: int, preferred: set[int]) -> list[int]:
    if group_count == 1:
        return [word_count]

    sizes: list[int] = []
    offset = 0
    for index in range(group_count - 1):
        groups_left = group_count - index - 1
        low = max(offset + 2, word_count - 3 * groups_left)
        high = min(offset + 3, word_count - 2 * groups_left)
        target = round(word_count * (index + 1) / group_count)
        choices = range(low, high + 1)
        boundary = min(
            choices,
            key=lambda candidate: (candidate not in preferred, abs(candidate - target)),
        )
        sizes.append(boundary - offset)
        offset = boundary
    sizes.append(word_count - offset)
    return sizes


def subtitle_margin_v(height: int, caption_band: list[int] | None = None) -> int:
    return subtitle_layout(height, caption_band)[1]


def subtitle_layout(
    height: int,
    caption_band: list[int] | None = None,
    position: str = "auto",
    margin_v_override: int | None = None,
) -> tuple[int, int]:
    if position == "top":
        alignment, margin_v = 8, round(height * 0.08)
    elif position in {"auto", "above"} and caption_band:
        subtitle_bottom_pct = max(8, caption_band[0] - 2)
        alignment, margin_v = 2, round(height * (100 - subtitle_bottom_pct) / 100)
    else:
        alignment, margin_v = 2, round(height * 0.06)
    return alignment, margin_v if margin_v_override is None else margin_v_override


def format_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


REDUCED_FONT_SCALE = 0.9


def _wrap_tokens(tokens: list[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = token
        else:
            current = candidate
    lines.append(current)
    return lines


def layout_cue_lines(
    text: str, max_chars: int, font_size: int | None = None
) -> tuple[list[str], int | None]:
    """Wrap text to lines, falling back to a smaller font for a third line instead of truncating."""
    safe = normalize_subtitle_text(text).replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    tokens = safe.replace("\n", " ").split()
    if not tokens:
        return [""], font_size

    lines = _wrap_tokens(tokens, max_chars)
    if len(lines) <= 2 or font_size is None:
        return lines, font_size

    reduced_font = max(1, round(font_size * REDUCED_FONT_SCALE))
    reduced_chars = math.ceil(max_chars / REDUCED_FONT_SCALE)
    return _wrap_tokens(tokens, reduced_chars), reduced_font


def _ass_text(text: str, max_chars: int, font_size: int | None = None) -> str:
    lines, used_font = layout_cue_lines(text, max_chars, font_size)
    body = r"\N".join(_bidi_line(line) for line in lines)
    if font_size is not None and used_font != font_size:
        return f"{{\\fs{used_font}}}{body}"
    return body


def _karaoke_text(
    text: str, duration: float, max_chars: int, font_size: int | None = None
) -> str:
    lines, used_font = layout_cue_lines(text, max_chars, font_size)
    tokens = [token for line in lines for token in line.split()]
    if not tokens:
        return ""

    total = max(0, round(duration * 100))
    character_total = sum(len(token) for token in tokens)
    timings = [round(total * len(token) / character_total) for token in tokens[:-1]]
    timings.append(total - sum(timings))

    rendered_lines: list[str] = []
    timing_index = 0
    for line in lines:
        rendered_words = []
        for token in line.split():
            isolated = LATIN_RUN.sub(lambda match: f"{LRI}{match.group()}{PDI}", token)
            rendered_words.append(f"{{\\k{timings[timing_index]}}}{isolated}")
            timing_index += 1
        rendered_lines.append(" ".join(rendered_words) + RLM)
    body = r"\N".join(rendered_lines)
    if font_size is not None and used_font != font_size:
        return f"{{\\fs{used_font}}}{body}"
    return body


def _bidi_line(text: str) -> str:
    isolated = LATIN_RUN.sub(lambda match: f"{LRI}{match.group()}{PDI}", text)
    return isolated + RLM
