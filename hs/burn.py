import json
import subprocess
import sys
from pathlib import Path


AAC_ENCODE_OPTIONS = [
    "-c:a",
    "aac",
    "-profile:a",
    "aac_low",
    "-b:a",
    "128k",
    "-ar",
    "48000",
    "-ac",
    "2",
]


def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_value = f"ass=filename='{_filter_path(ass_path.resolve())}'"
    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_value,
        "-pix_fmt",
        "yuv420p",
    ]
    audio_options = _audio_options(video_path)

    try:
        subprocess.run(_nvenc_command(common, audio_options, output_path), check=True)
        return "h264_nvenc"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(_software_command(common, audio_options, output_path), check=True)
            return "libx264"
        except subprocess.CalledProcessError:
            if audio_options != AAC_ENCODE_OPTIONS:
                raise
            print(
                "AAC-LC audio encoding failed; falling back to copying the source audio",
                file=sys.stderr,
            )
            subprocess.run(_software_command(common, ["-c:a", "copy"], output_path), check=True)
            return "libx264"


def _audio_options(video_path: Path) -> list[str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,profile,sample_rate,channels,bit_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        return []

    audio = streams[0]
    if (
        audio.get("codec_name") == "aac"
        and audio.get("profile") == "LC"
        and audio.get("sample_rate") == "48000"
        and audio.get("channels") == 2
        and _bit_rate(audio) <= 160_000
    ):
        return ["-c:a", "copy"]
    return AAC_ENCODE_OPTIONS


def _bit_rate(audio: dict) -> int:
    try:
        return int(audio["bit_rate"])
    except (KeyError, TypeError, ValueError):
        return 160_001


def _nvenc_command(common: list[str], audio: list[str], output_path: Path) -> list[str]:
    return [
        *common,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "21",
        *audio,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _software_command(common: list[str], audio: list[str], output_path: Path) -> list[str]:
    return [
        *common,
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        *audio,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", r"\\").replace(":", r"\:").replace("'", r"'\''")
