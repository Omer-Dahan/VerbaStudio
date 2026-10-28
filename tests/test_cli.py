import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hs.cli import main, resolve_output_path
from hs.models import Word


def test_output_directory_uses_hebrew_suffixed_filename(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    destination = tmp_path / "rendered"
    destination.mkdir()

    assert resolve_output_path(video, str(destination)) == destination / "source_he.mp4"


def test_explicit_mp4_output_is_used_exactly(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    destination = tmp_path / "chosen.mp4"

    assert resolve_output_path(video, str(destination)) == destination


def test_invalid_output_path_is_rejected_with_usage_exit_code(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"

    with pytest.raises(ValueError, match="existing directory"):
        resolve_output_path(video, str(tmp_path / "not-a-video"))


def test_json_failure_is_parseable_and_human_error_uses_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(tmp_path / "missing.mp4"), "--json"])

    captured = capsys.readouterr()
    manifest = json.loads(captured.out)
    assert exit_code == 2
    assert manifest["ok"] is False
    assert manifest["out"] is None
    assert manifest["exit"] == 2
    assert "Video not found" in captured.err


def test_json_success_is_parseable_and_has_required_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    fake_report = {
        "cps_violations": [],
        "silence_overlaps": [],
        "translation_failures": [],
    }
    with (
        patch("hs.cli.probe_video", return_value={"width": 1920, "height": 1080, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([Word(0, 1, "hello")], "en")),
        patch("hs.cli.TranslationClient.translate_cues"),
        patch("hs.cli.write_ass"),
        patch("hs.cli.build_report", return_value=fake_report),
        patch("hs.cli.write_report"),
    ):
        exit_code = main([str(video), "--json", "--dry-run"])

    manifest = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert manifest["ok"] is True
    assert set(manifest) == {
        "ok", "out", "srt", "ass", "lang", "engine", "model", "compute_type",
        "cue_count", "cps_max", "qa", "secs", "exit",
    }


def test_no_audio_stream_returns_exit_three(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.touch()
    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=False),
    ):
        assert main([str(video)]) == 3


def test_translation_failure_returns_exit_five(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    word = Word(0, 1, "hello")

    def fail_translation(cues, *_args, **_kwargs):
        cues[0].translation_failed = True

    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([word], "en")),
        patch("hs.cli.TranslationClient.translate_cues", side_effect=fail_translation),
    ):
        assert main([str(video)]) == 5
