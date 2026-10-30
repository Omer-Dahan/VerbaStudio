import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

from hs.burn import BurnResult, burn_subtitles, encoding_decisions


def probe_result(**overrides) -> subprocess.CompletedProcess:
    stream = {
        "codec_name": "aac",
        "profile": "LC",
        "sample_rate": "48000",
        "channels": 2,
        "bit_rate": "128000",
        **overrides,
    }
    return subprocess.CompletedProcess([], 0, json.dumps({"streams": [stream]}), "")


def test_burn_prefers_nvenc_and_targets_size_budget() -> None:
    with patch("hs.burn.subprocess.run", side_effect=[probe_result(), None]) as run:
        result = burn_subtitles(
            Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0
        )

    assert result.codec == "h264_nvenc"
    command = run.call_args_list[1].args[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-b:v") + 1] == "1751k"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "+faststart+use_metadata_tags" in command
    assert command[command.index("-metadata") + 1] == "hebsub=1"


def test_burn_falls_back_to_libx264() -> None:
    failure = subprocess.CalledProcessError(1, ["ffmpeg"])
    with patch("hs.burn.subprocess.run", side_effect=[probe_result(), failure, None]) as run:
        result = burn_subtitles(
            Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0, 0
        )

    assert result.codec == "libx264"
    fallback = run.call_args_list[2].args[0]
    assert fallback[fallback.index("-c:v") + 1] == "libx264"
    assert fallback[fallback.index("-crf") + 1] == "20"


def test_burn_reencodes_he_aac_as_aac_lc() -> None:
    he_aac = probe_result(profile="HE-AACv2", sample_rate="44100", bit_rate="32000")
    with patch("hs.burn.subprocess.run", side_effect=[he_aac, None]) as run:
        burn_subtitles(Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0)

    command = run.call_args_list[1].args[0]
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-profile:a") + 1] == "aac_low"
    assert command[command.index("-b:a") + 1] == "96k"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"


def test_audio_encode_failure_falls_back_to_copy(capsys) -> None:
    he_aac = probe_result(profile="HE-AACv2")
    failure = subprocess.CalledProcessError(1, ["ffmpeg"])
    with patch(
        "hs.burn.subprocess.run",
        side_effect=[he_aac, failure, failure, None],
    ) as run:
        result = burn_subtitles(
            Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0
        )

    assert result.codec == "libx264"
    fallback = run.call_args_list[3].args[0]
    assert fallback[fallback.index("-c:a") + 1] == "copy"
    assert "falling back to copying" in capsys.readouterr().err


def test_burn_keeps_video_only_input_video_only() -> None:
    no_audio = subprocess.CompletedProcess([], 0, '{"streams": []}', "")
    with patch("hs.burn.subprocess.run", side_effect=[no_audio, None]) as run:
        burn_subtitles(Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0)

    command = run.call_args_list[1].args[0]
    assert "-c:a" not in command


def test_38_second_fixture_meets_streaming_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subtitles = tmp_path / "empty.ass"
    output = tmp_path / "output.mp4"
    subtitles.write_text(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 320\n"
        "PlayResY: 180\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, "
        "Bold, Italic, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H00000000,0,0,2,10,10,10,1\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Text\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=10:d=38",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=44100:duration=38",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )

    burn_subtitles(source, subtitles, output, 38.0)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,profile,sample_rate,channels",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    assert streams[0]["codec_name"] == "h264"
    assert streams[1] == {
        "codec_name": "aac",
        "profile": "LC",
        "codec_type": "audio",
        "sample_rate": "48000",
        "channels": 2,
    }
    assert output.stat().st_size / 1024**2 <= 9.5
    assert _atoms(output)[:2] == ["ftyp", "moov"]


def test_twenty_minute_silent_video_hits_quality_floor() -> None:
    video_kbps, downscaled, quality_floor_hit = encoding_decisions(20 * 60, 9.0, False)

    assert video_kbps == 58
    assert downscaled is True
    assert quality_floor_hit is True


def test_zero_size_cap_keeps_nvenc_constant_quality_mode() -> None:
    no_audio = subprocess.CompletedProcess([], 0, '{"streams": []}', "")
    with patch("hs.burn.subprocess.run", side_effect=[no_audio, None]) as run:
        result = burn_subtitles(
            Path("input.mp4"), Path("cues.ass"), Path("output.mp4"), 38.0, 0
        )

    command = run.call_args_list[1].args[0]
    assert result == BurnResult("h264_nvenc", False, False)
    assert command[command.index("-cq") + 1] == "21"
    assert "-b:v" not in command


def _atoms(path: Path) -> list[str]:
    atoms = []
    with path.open("rb") as handle:
        while header := handle.read(8):
            size, kind = struct.unpack(">I4s", header)
            atoms.append(kind.decode("ascii"))
            header_size = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header_size = 16
            if size == 0:
                break
            handle.seek(size - header_size, 1)
    return atoms
