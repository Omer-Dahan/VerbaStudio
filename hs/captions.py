import subprocess
from pathlib import Path


SAMPLE_WIDTH = 288
SAMPLE_COUNT = 12


def detect_existing_captions(
    video_path: Path, width: int, height: int, duration: float
) -> dict[str, bool | list[int]]:
    sample_height = max(2, round(height * SAMPLE_WIDTH / width / 2) * 2)
    interval = duration / SAMPLE_COUNT
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval:.6f},scale={SAMPLE_WIDTH}:{sample_height}",
        "-frames:v",
        str(SAMPLE_COUNT),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        return {"detected": False, "band": []}
    frames = completed.stdout
    frame_size = SAMPLE_WIDTH * sample_height * 3
    frame_count = len(frames) // frame_size
    if frame_count < 2:
        return {"detected": False, "band": []}

    row_counts = _unstable_white_rows(frames, frame_count, SAMPLE_WIDTH, sample_height)
    bottom_start = round(sample_height * 0.66)
    bottom_total = sum(row_counts[bottom_start:])
    window_height = max(1, round(sample_height * 0.20))
    peak_start = max(
        range(bottom_start, sample_height - window_height + 1),
        key=lambda row: sum(row_counts[row : row + window_height]),
    )
    peak_end = peak_start + window_height
    peak_total = sum(row_counts[peak_start:peak_end])
    detected = bottom_total >= 300 and peak_total >= bottom_total * 0.60
    if not detected:
        return {"detected": False, "band": []}

    peak_row_count = max(row_counts[peak_start:peak_end])
    meaningful = max(3, round(peak_row_count * 0.40))
    occupied_rows = [
        row for row in range(peak_start, peak_end) if row_counts[row] >= meaningful
    ]
    top = round(occupied_rows[0] * 100 / sample_height)
    bottom = round((occupied_rows[-1] + 1) * 100 / sample_height)
    return {"detected": True, "band": [top, bottom]}


def _unstable_white_rows(
    frames: bytes, frame_count: int, width: int, height: int
) -> list[int]:
    frame_size = width * height * 3
    rows = [0] * height
    for pixel in range(width * height):
        white_frames = 0
        offset = pixel * 3
        for frame_index in range(frame_count):
            base = frame_index * frame_size + offset
            if min(frames[base : base + 3]) >= 220:
                white_frames += 1
        if 0 < white_frames < frame_count:
            rows[pixel // width] += 1
    return rows
