import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AAC_ENCODE_OPTIONS = [
    "-c:a",
    "aac",
    "-profile:a",
    "aac_low",
    "-b:a",
    "96k",
    "-ar",
    "48000",
    "-ac",
    "2",
]


@dataclass(frozen=True)
class BurnResult:
    codec: str
    downscaled: bool
    quality_floor_hit: bool


def encoding_decisions(
    duration_s: float, max_size_mb: float, has_audio: bool
) -> tuple[int, bool, bool]:
    audio_kbps = 96 if has_audio else 0
    video_kbps = int((max_size_mb * 8192 / duration_s - audio_kbps) * 0.95)
    return video_kbps, video_kbps < 700, video_kbps < 500


def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    duration_s: float,
    max_size_mb: float = 9.0,
) -> BurnResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_options = _audio_options(video_path)
    video_kbps, downscaled, quality_floor_hit = encoding_decisions(
        duration_s, max_size_mb, bool(audio_options)
    )
    filters = []
    if downscaled:
        filters.append("scale=trunc(iw*0.75/2)*2:trunc(ih*0.75/2)*2")
        print(
            f"Video budget is {video_kbps} kbps; scaling output dimensions down by 25%",
            file=sys.stderr,
        )
    if quality_floor_hit:
        print(
            f"Warning: video budget is below the 500 kbps quality floor ({video_kbps} kbps)",
            file=sys.stderr,
        )
    filters.append(f"ass=filename='{_filter_path(ass_path.resolve())}'")
    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        ",".join(filters),
        "-pix_fmt",
        "yuv420p",
    ]
    try:
        subprocess.run(
            _nvenc_command(common, audio_options, output_path, video_kbps), check=True
        )
        return BurnResult("h264_nvenc", downscaled, quality_floor_hit)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                _software_command(common, audio_options, output_path, video_kbps), check=True
            )
            return BurnResult("libx264", downscaled, quality_floor_hit)
        except subprocess.CalledProcessError:
            if audio_options != AAC_ENCODE_OPTIONS:
                raise
            print(
                "AAC-LC audio encoding failed; falling back to copying the source audio",
                file=sys.stderr,
            )
            subprocess.run(
                _software_command(common, ["-c:a", "copy"], output_path, video_kbps),
                check=True,
            )
            return BurnResult("libx264", downscaled, quality_floor_hit)


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

    return AAC_ENCODE_OPTIONS


def _nvenc_command(
    common: list[str], audio: list[str], output_path: Path, video_kbps: int
) -> list[str]:
    return [
        *common,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{int(video_kbps * 1.2)}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        *audio,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _software_command(
    common: list[str], audio: list[str], output_path: Path, video_kbps: int
) -> list[str]:
    return [
        *common,
        "-c:v",
        "libx264",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{int(video_kbps * 1.2)}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        "-preset",
        "medium",
        *audio,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", r"\\").replace(":", r"\:").replace("'", r"'\''")
