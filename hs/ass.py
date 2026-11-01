from pathlib import Path
import math
import re

from hs.models import Cue
from hs.text import normalize_subtitle_text


DEFAULT_FONT = "Ploni DL 1.1 AAA"
LATIN_RUN = re.compile(
    r"(?=[A-Za-z0-9.,:%$+\-/& ]*[A-Za-z])"
    r"[A-Za-z0-9](?:[A-Za-z0-9]|[.,:%$+\-/&](?=[A-Za-z0-9])| (?=[A-Za-z0-9]))*"
)
LRI = "\u2066"
PDI = "\u2069"
RLM = "\u200f"


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
) -> Path:
    alignment, margin_v = subtitle_layout(
        height, caption_band, position, margin_v_override
    )
    font_size, margin_h, max_chars = subtitle_geometry(width, height, size_scale)
    boxed = box_alpha > 0
    border_style = 3 if boxed else 1
    style_outline = 8 if boxed else outline
    back_colour = f"&H{round((1 - box_alpha) * 255):02X}000000" if boxed else "&H80000000"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{style_outline:g},0,{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = [
        f"Dialogue: 0,{format_timestamp(cue.start)},{format_timestamp(cue.end)},Default,,0,0,0,,{{\\q2}}{_ass_text(cue.display_text, max_chars, font_size)}"
        for cue in cues
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return destination


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


def _bidi_line(text: str) -> str:
    isolated = LATIN_RUN.sub(lambda match: f"{LRI}{match.group()}{PDI}", text)
    return isolated + RLM
