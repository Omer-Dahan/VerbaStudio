import subprocess
from pathlib import Path
from unittest.mock import patch

from hs.burn import burn_subtitles


def test_burn_prefers_nvenc() -> None:
    with patch("hs.burn.subprocess.run") as run:
        codec = burn_subtitles(Path("input.mp4"), Path("cues.ass"), Path("output.mp4"))

    assert codec == "h264_nvenc"
    command = run.call_args.args[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "+faststart" in command


def test_burn_falls_back_to_libx264() -> None:
    failure = subprocess.CalledProcessError(1, ["ffmpeg"])
    with patch("hs.burn.subprocess.run", side_effect=[failure, None]) as run:
        codec = burn_subtitles(Path("input.mp4"), Path("cues.ass"), Path("output.mp4"))

    assert codec == "libx264"
    fallback = run.call_args_list[1].args[0]
    assert fallback[fallback.index("-c:v") + 1] == "libx264"
    assert fallback[fallback.index("-crf") + 1] == "20"
