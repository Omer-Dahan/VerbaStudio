from pathlib import Path
import math
import re

from hs.models import Cue


DEFAULT_FONT = "Ploni DL 1.1 AAA"
LATIN_RUN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9]|[.'\-&](?=[A-Za-z0-9])| (?=[A-Za-z0-9]))*"
)
RLI = "\u2067"
PDI = "\u2069"
RLM = "\u200f"


def subtitle_geometry(width: int, height: int) -> tuple[int, int, int]:
    font_size = min(64, max(28, round(height * 0.045)))
    margin = max(24, round(width * 0.055))
    max_chars = max(1, math.floor((width - 2 * margin) / (0.52 * font_size)))
    return font_size, margin, max_chars


def write_ass(
    cues: list[Cue],
    destination: Path,
    width: int,
    height: int,
    font: str = DEFAULT_FONT,
) -> Path:
    margin_v = round(height * 0.06)
    font_size, margin_h, max_chars = subtitle_geometry(width, height)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = [
        f"Dialogue: 0,{format_timestamp(cue.start)},{format_timestamp(cue.end)},Default,,0,0,0,,{{\\q2}}{_ass_text(cue.display_text, max_chars)}"
        for cue in cues
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return destination


def format_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def _ass_text(text: str, max_chars: int) -> str:
    safe = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    words = safe.replace("\n", " ").split()
    if len(safe) <= max_chars or len(words) < 2:
        return _bidi_line(safe)

    best_index = min(
        range(1, len(words)),
        key=lambda index: abs(len(" ".join(words[:index])) - len(" ".join(words[index:]))),
    )
    return _bidi_line(" ".join(words[:best_index])) + r"\N" + _bidi_line(
        " ".join(words[best_index:])
    )


def _bidi_line(text: str) -> str:
    isolated = LATIN_RUN.sub(lambda match: f"{RLI}{match.group()}{PDI}", text)
    return isolated + RLM
