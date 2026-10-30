import subprocess
from pathlib import Path

import pytest

from hs.ass import _ass_text, format_timestamp, subtitle_geometry, write_ass
from hs.models import Cue


def render_black_frame_margins(
    ass_path: Path, width: int, height: int
) -> dict[str, int]:
    fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
    ass_filter = f"ass=filename='{ass_path.resolve()}':fontsdir='{fonts_dir}'"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:d=1:r=1",
        "-vf",
        ass_filter,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    frame = subprocess.run(command, check=True, capture_output=True).stdout
    assert len(frame) == width * height
    visible = [index for index, value in enumerate(frame) if value]
    assert visible, "ASS render did not contain any visible subtitle pixels"
    xs = [index % width for index in visible]
    ys = [index // width for index in visible]
    return {
        "left": min(xs),
        "right": width - 1 - max(xs),
        "top": min(ys),
        "bottom": height - 1 - max(ys),
    }


def test_ass_uses_video_resolution_style_and_rtl_override(tmp_path: Path) -> None:
    destination = tmp_path / "cues.ass"
    write_ass([Cue(1.234, 4.567, "source", "שלום עולם")], destination, 720, 1280)
    content = destination.read_text(encoding="utf-8")

    assert "PlayResX: 720" in content
    assert "PlayResY: 1280" in content
    assert "WrapStyle: 0" in content
    assert "ScaledBorderAndShadow: yes" in content
    assert "Style: Default,Ploni DL 1.1 AAA" in content
    assert ",2,40,40,77,1" in content
    assert "Dialogue: 0,0:00:01.23,0:00:04.57" in content
    assert "{\\q2}שלום עולם\u200f" in content


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "הגרסה נקראת 'Horizon Minecraft Java 26.3'.",
            "הגרסה נקראת '\u2067Horizon Minecraft Java 26.3\u2069'.\u200f",
        ),
        ("Mojang's code.", "\u2067Mojang's code\u2069.\u200f"),
        ("שוחרר ב-2026.", "שוחרר ב-\u20672026\u2069.\u200f"),
        ("(GPU) ו-VRAM", "(\u2067GPU\u2069) ו-\u2067VRAM\u2069\u200f"),
        ("עברית בלבד", "עברית בלבד\u200f"),
    ],
)
def test_ass_text_isolates_latin_runs_and_marks_rtl_lines(
    text: str, expected: str
) -> None:
    assert _ass_text(text, max_chars=100) == expected


def test_wrapped_ass_text_marks_both_lines_as_rtl() -> None:
    rendered = _ass_text("עברית אחת English words עברית שנייה", max_chars=18)

    assert rendered.count(r"\N") == 1
    assert all(line.endswith("\u200f") for line in rendered.split(r"\N"))


def test_long_text_wraps_without_reordering_hebrew(tmp_path: Path) -> None:
    text = " ".join(["מילה"] * 20)
    destination = tmp_path / "cues.ass"
    write_ass([Cue(0, 5, text, text)], destination, 1920, 1080)

    dialogue = next(
        line for line in destination.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")
    )
    assert r"\N" in dialogue
    assert dialogue.count("מילה") == 20


def test_timestamp_rolls_over_at_hour_boundary() -> None:
    assert format_timestamp(3661.999) == "1:01:02.00"


def test_vertical_frame_geometry_scales_font_and_line_budget() -> None:
    font_size, margin, max_chars = subtitle_geometry(576, 1024)

    assert font_size == 46
    assert margin == 32
    assert max_chars == 21


def test_font_size_is_clamped_for_extreme_frame_heights() -> None:
    assert subtitle_geometry(320, 480)[0] == 28
    assert subtitle_geometry(3840, 2160)[0] == 64


def test_vertical_ass_has_clear_margins_on_black_frame(tmp_path: Path) -> None:
    destination = tmp_path / "vertical.ass"
    text = " ".join(["מילה"] * 8)
    write_ass([Cue(0.0, 1.0, text, text)], destination, 576, 1024)

    margins = render_black_frame_margins(destination, 576, 1024)

    assert all(margin >= 24 for margin in margins.values()), margins
