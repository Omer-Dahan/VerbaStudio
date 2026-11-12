#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np


PLAY_RESOLUTION = re.compile(r"^PlayRes([XY]):\s*(\d+)$", re.MULTILINE)


def _pixel_stats(mask: np.ndarray) -> tuple[int, int, float] | None:
    _, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return int(xs.min()), int(xs.max()), float(xs.mean())


def _format_stats(name: str, stats: tuple[int, int, float] | None) -> str:
    if stats is None:
        return f"{name}_x=none {name}_centroid=none"
    return f"{name}_x={stats[0]}-{stats[1]} {name}_centroid={stats[2]:.2f}"


def _render(
    ass_path: Path,
    fonts_dir: Path,
    timestamp: float,
    width: int,
    height: int,
) -> np.ndarray:
    escaped_ass = str(ass_path.resolve()).replace("'", r"\'")
    escaped_fonts = str(fonts_dir.resolve()).replace("'", r"\'")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=100",
        "-vf", f"ass=filename='{escaped_ass}':fontsdir='{escaped_fonts}'",
        "-ss", f"{timestamp:.3f}", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    frame = subprocess.run(command, check=True, capture_output=True).stdout
    expected = width * height * 3
    if len(frame) != expected:
        raise RuntimeError(f"ffmpeg returned {len(frame)} bytes; expected {expected}")
    return np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure highlighted and base caption pixel positions."
    )
    parser.add_argument("ass", type=Path)
    parser.add_argument("fontdir", type=Path)
    parser.add_argument("timestamps", type=float, nargs="+")
    args = parser.parse_args()

    content = args.ass.read_text(encoding="utf-8-sig")
    resolution = dict(PLAY_RESOLUTION.findall(content))
    if resolution.keys() != {"X", "Y"}:
        parser.error("ASS file must contain PlayResX and PlayResY")

    first_yellow = None
    first_white = None
    for timestamp in args.timestamps:
        frame = _render(
            args.ass,
            args.fontdir,
            timestamp,
            int(resolution["X"]),
            int(resolution["Y"]),
        )
        red, green, blue = (frame[:, :, channel] for channel in range(3))
        yellow = (red >= 160) & (green >= 140) & (blue <= 120)
        white = (red >= 175) & (green >= 175) & (blue >= 175)
        yellow_stats = _pixel_stats(yellow)
        white_stats = _pixel_stats(white)
        print(
            f"t={timestamp:.2f} "
            f"{_format_stats('yellow', yellow_stats)} "
            f"{_format_stats('white', white_stats)}"
        )
        if first_yellow is None and yellow_stats is not None and white_stats is not None:
            first_yellow, first_white = yellow_stats[2], white_stats[2]

    return int(first_yellow is not None and first_white is not None and first_yellow <= first_white)


if __name__ == "__main__":
    raise SystemExit(main())
