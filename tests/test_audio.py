import json
from pathlib import Path
from unittest.mock import Mock, patch

from hs.audio import extract_audio, has_audio_stream, probe_video


def test_extract_audio_uses_required_pcm_format(tmp_path: Path) -> None:
    destination = tmp_path / "audio.wav"
    with patch("hs.audio.subprocess.run") as run:
        assert extract_audio(Path("clip.mp4"), destination) == destination

    command = run.call_args.args[0]
    assert command[-1] == str(destination)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    run.assert_called_once_with(command, check=True)


def test_probe_video_returns_dimensions_and_duration() -> None:
    response = Mock(
        stdout=json.dumps(
            {"streams": [{"width": 720, "height": 1280}], "format": {"duration": "87.2"}}
        )
    )
    with patch("hs.audio.subprocess.run", return_value=response):
        assert probe_video(Path("clip.mp4")) == {
            "width": 720,
            "height": 1280,
            "duration": 87.2,
        }


def test_audio_stream_probe_reports_missing_audio() -> None:
    response = Mock(stdout=json.dumps({"streams": []}))
    with patch("hs.audio.subprocess.run", return_value=response):
        assert has_audio_stream(Path("silent.mp4")) is False
