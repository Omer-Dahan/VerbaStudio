import subprocess
from pathlib import Path


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
    ]
    nvenc = [
        *common,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "21",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(nvenc, check=True)
        return "h264_nvenc"
    except subprocess.CalledProcessError:
        software = [
            *common,
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(software, check=True)
        return "libx264"


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", r"\\").replace(":", r"\:").replace("'", r"'\''")

