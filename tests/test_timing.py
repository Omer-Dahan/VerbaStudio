import os
import re
import subprocess
from pathlib import Path

import pytest


DIALOGUE = re.compile(r"^Dialogue: [^,]*,([^,]+),([^,]+),")


def parse_ass_time(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def test_generated_ass_has_valid_non_overlapping_timing() -> None:
    ass_value = os.environ.get("HS_TIMING_ASS")
    video_value = os.environ.get("HS_TIMING_VIDEO")
    if not ass_value or not video_value:
        pytest.skip("set HS_TIMING_ASS and HS_TIMING_VIDEO for the real-video acceptance test")

    ass_path = Path(ass_value)
    assert ass_path.is_file(), f"generated ASS not found: {ass_path}"
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_value,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(completed.stdout.strip())
    timings = [
        (parse_ass_time(match.group(1)), parse_ass_time(match.group(2)))
        for line in ass_path.read_text(encoding="utf-8").splitlines()
        if (match := DIALOGUE.match(line))
    ]

    assert timings
    assert all(start >= 0 and end > start and end <= duration for start, end in timings)
    assert all(left_end <= right_start for (_, left_end), (right_start, _) in zip(timings, timings[1:]))
