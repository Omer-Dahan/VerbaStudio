import json
import subprocess
from pathlib import Path


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    """Extract a mono, 16 kHz PCM track suitable for Whisper."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(command, check=True)
    return wav_path


def probe_video(video_path: Path) -> dict[str, float | int | str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration:format_tags=hebsub,hebsub_font",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise ValueError(f"No video stream found in {video_path}")

    stream = payload["streams"][0]
    tags = payload["format"].get("tags", {})
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(payload["format"]["duration"]),
        "hebsub": tags.get("hebsub", ""),
        "hebsub_font": tags.get("hebsub_font", ""),
    }


def has_audio_stream(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return bool(json.loads(completed.stdout).get("streams"))
