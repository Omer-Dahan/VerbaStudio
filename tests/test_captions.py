import subprocess
from pathlib import Path

from hs.ass import subtitle_layout
from hs.captions import detect_existing_captions
from hs.models import Cue
from hs.qa import build_report


WIDTH = 576
HEIGHT = 1024
DURATION = 12


def make_source(
    path: Path,
    with_captions: bool,
    caption_x: str = "(w-text_w)/2",
    caption_text: str = "EXISTING CAPTION",
) -> None:
    source = f"color=c=black:s={WIDTH}x{HEIGHT}:d={DURATION}:r=1"
    filters = []
    if with_captions:
        font = Path(__file__).resolve().parent.parent / "fonts/Assistant/Assistant-Bold.ttf"
        filters.append(
            "drawtext="
            f"fontfile='{font}':text='{caption_text}':"
            f"fontcolor=white:fontsize=72:x={caption_x}:y=h*0.85:"
            "enable='lt(mod(t,2),1)'"
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        source,
    ]
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    subprocess.run(command, check=True)


def test_captioned_source_is_detected_and_auto_layout_stays_above_it(
    tmp_path: Path,
) -> None:
    video = tmp_path / "captioned.mp4"
    make_source(video, with_captions=True)

    captions = detect_existing_captions(video, WIDTH, HEIGHT, DURATION)
    alignment, margin_v = subtitle_layout(HEIGHT, captions["band"], "auto")
    text_bottom_pct = (HEIGHT - margin_v) * 100 / HEIGHT

    assert captions["detected"] is True
    assert captions["band"][0] >= 85
    assert alignment == 2
    assert text_bottom_pct <= 83.1
    assert captions["band"][0] - text_bottom_pct >= 1.9


def test_clean_source_keeps_default_bottom_placement(tmp_path: Path) -> None:
    video = tmp_path / "clean.mp4"
    make_source(video, with_captions=False)

    captions = detect_existing_captions(video, WIDTH, HEIGHT, DURATION)

    assert captions == {"detected": False, "band": []}
    assert subtitle_layout(HEIGHT, None, "auto") == (2, round(HEIGHT * 0.06))


def test_left_aligned_bottom_hud_is_not_treated_as_captions(tmp_path: Path) -> None:
    video = tmp_path / "left-hud.mp4"
    make_source(video, with_captions=True, caption_x="0", caption_text="CHAT")

    assert detect_existing_captions(video, WIDTH, HEIGHT, DURATION) == {
        "detected": False,
        "band": [],
    }


def test_forced_bottom_position_reports_layout_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "captioned.mp4"
    make_source(video, with_captions=True)
    captions = detect_existing_captions(video, WIDTH, HEIGHT, DURATION)
    monkeypatch.setattr("hs.qa.detect_silences", lambda *_args: [])

    report = build_report(
        [Cue(0, 2, "source", "subtitle")],
        video,
        DURATION,
        width=WIDTH,
        height=HEIGHT,
        caption_band=captions["band"],
        position="bottom",
    )

    assert report["layout_conflicts"]
    conflict = report["layout_conflicts"][0]
    assert conflict["text_band"][1] > conflict["caption_band"][0]
