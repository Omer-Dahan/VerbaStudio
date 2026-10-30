import subprocess
from pathlib import Path

import pytest

from hs.ass import _ass_text, format_timestamp, subtitle_geometry, write_ass
from hs.models import Cue
from hs.text import QUOTE_CHARACTERS


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
    assert ",3,8,0,2,40,40,77,1" in content
    assert "&H73000000" in content
    assert "Dialogue: 0,0:00:01.23,0:00:04.57" in content
    assert "{\\q2}שלום עולם\u200f" in content


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "הגרסה נקראת 'Horizon Minecraft Java 26.3'.",
            "הגרסה נקראת \u2066Horizon Minecraft Java 26.3\u2069.\u200f",
        ),
        ("Mojang's code.", "\u2066Mojangs code\u2069.\u200f"),
        ("שוחרר ב-2026.", "שוחרר ב-2026.\u200f"),
        ("(GPU) ו-VRAM", "(\u2066GPU\u2069) ו-\u2066VRAM\u2069\u200f"),
        ("עברית בלבד", "עברית בלבד\u200f"),
    ],
)
def test_ass_text_isolates_latin_runs_and_marks_rtl_lines(
    text: str, expected: str
) -> None:
    assert _ass_text(text, max_chars=100) == expected


def test_ass_text_uses_lri_only_for_latin_runs() -> None:
    rendered = _ass_text(
        "נקרא `Horizon Minecraft Java 26.3` ועוד 2026-10-30 שוב",
        max_chars=100,
    )

    assert "\u2067" not in rendered
    assert rendered == (
        "נקרא \u2066Horizon Minecraft Java 26.3\u2069 ועוד 2026-10-30 שוב\u200f"
    )
    assert rendered.count("\u200f") == 1


def test_ass_text_removes_quotes_in_every_position() -> None:
    rendered = _ass_text(
        f"{QUOTE_CHARACTERS}תחילה א{QUOTE_CHARACTERS}מצע סוף{QUOTE_CHARACTERS}",
        max_chars=100,
    )

    assert not set(QUOTE_CHARACTERS) & set(rendered)
    assert rendered == "תחילה אמצע סוף\u200f"


def test_write_ass_never_emits_ascii_quotes(tmp_path: Path) -> None:
    destination = tmp_path / "quotes.ass"
    write_ass([Cue(0, 1, "source", "מוצר 'quoted' ו־\"double\"")], destination, 720, 1280)

    dialogue = destination.read_text(encoding="utf-8").split("[Events]\n", 1)[1]
    assert not {"'", '"', "`"} & set(dialogue)


def test_wrapped_ass_text_marks_both_lines_as_rtl() -> None:
    rendered = _ass_text("עברית אחת English words עברית שנייה", max_chars=18)

    assert rendered.count(r"\N") == 1
    assert all(line.endswith("\u200f") for line in rendered.split(r"\N"))


def test_ass_text_repairs_and_keeps_split_decimal_atomic() -> None:
    rendered = _ass_text("ללא שינוי של 26'. .3, בנויה מחדש", max_chars=16)

    assert "26.3" in rendered
    assert "26\\N.3" not in rendered


def test_oversized_token_stays_on_its_own_line() -> None:
    token = "Horizon-Minecraft-Java-26.3"
    rendered = _ass_text(f"לפני {token} אחרי", max_chars=10)

    assert rendered.split(r"\N") == [
        "לפני\u200f",
        f"\u2066{token}\u2069\u200f",
        "אחרי\u200f",
    ]


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

    assert font_size == 55
    assert margin == 32
    assert max_chars == 17


def test_font_size_is_clamped_for_extreme_frame_heights() -> None:
    assert subtitle_geometry(320, 480)[0] == 34
    assert subtitle_geometry(3840, 2160)[0] == 77


def test_zero_box_alpha_uses_configured_outline_style(tmp_path: Path) -> None:
    destination = tmp_path / "outlined.ass"

    write_ass(
        [Cue(0, 1, "source", "עברית")],
        destination,
        576,
        1024,
        box_alpha=0,
        outline=4,
    )

    style = next(
        line for line in destination.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style:")
    )
    assert ",1,4,0,2," in style
    assert "&H80000000" in style


def test_vertical_ass_has_clear_margins_on_black_frame(tmp_path: Path) -> None:
    destination = tmp_path / "vertical.ass"
    text = " ".join(["מילה"] * 8)
    write_ass([Cue(0.0, 1.0, text, text)], destination, 576, 1024)

    margins = render_black_frame_margins(destination, 576, 1024)

    assert all(margin >= 24 for margin in margins.values()), margins
