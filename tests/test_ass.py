from pathlib import Path

from hs.ass import format_timestamp, write_ass
from hs.models import Cue


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
    assert r"{\q2}שלום עולם" in content


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

