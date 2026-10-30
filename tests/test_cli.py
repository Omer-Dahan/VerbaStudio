import json
from pathlib import Path
from unittest.mock import call, patch

import pytest

from hs.cli import build_parser, main, resolve_output_path, wait_for_translation_server
from hs.models import Cue, Word


@pytest.fixture(autouse=True)
def healthy_translation_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hs.cli.server_is_healthy", lambda _url: True)


def test_output_directory_uses_hebrew_suffixed_filename(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    destination = tmp_path / "rendered"
    destination.mkdir()

    assert resolve_output_path(video, str(destination)) == destination / "source_he.mp4"


def test_defaults_fit_sixteen_gigabyte_gpu() -> None:
    args = build_parser().parse_args(["source.mp4"])

    assert args.model == "Systran/faster-whisper-large-v3"
    assert args.compute_type == "int8_float16"
    assert "fits 16GB" in " ".join(build_parser().format_help().split())


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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    fake_report = {
        "cps_violations": [],
        "forced_cues": [],
        "silence_overlaps": [],
        "translation_failures": [],
    }
    with (
        patch("hs.cli.probe_video", return_value={"width": 1920, "height": 1080, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([Word(0, 1, "hello")], "en")),
        patch("hs.cli.TranslationClient.translate_cues"),
        patch(
            "hs.cli.write_ass",
            side_effect=lambda _cues, path, *_args: path.write_text("ass", encoding="utf-8"),
        ),
        patch("hs.cli.build_report", return_value=fake_report),
        patch(
            "hs.cli.write_report",
            side_effect=lambda _report, json_path, markdown_path: (
                json_path.write_text("{}", encoding="utf-8"),
                markdown_path.write_text("qa", encoding="utf-8"),
            ),
        ),
    ):
        exit_code = main([str(video), "--json", "--dry-run"])

    manifest = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert manifest["ok"] is True
    assert set(manifest) == {
        "ok", "out", "srt", "ass", "lang", "engine", "model", "compute_type",
        "cue_count", "cps_max", "cps_target", "qa", "secs", "exit",
    }


def test_no_audio_stream_returns_exit_three(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.touch()
    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=False),
    ):
        assert main([str(video)]) == 3


def test_one_translation_failure_is_reported_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    word = Word(0, 1, "hello")

    def fail_translation(cues, *_args, **_kwargs):
        cues[0].translation_failed = True

    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([word], "en")),
        patch("hs.cli.TranslationClient.translate_cues", side_effect=fail_translation),
        patch("hs.cli.build_report", return_value={
            "cps_violations": [], "forced_cues": [], "silence_overlaps": [],
            "translation_failures": [1], "boundary_issues": [],
        }),
        patch(
            "hs.cli.write_report",
            side_effect=lambda _report, json_path, markdown_path: (
                json_path.write_text("{}", encoding="utf-8"),
                markdown_path.write_text("qa", encoding="utf-8"),
            ),
        ),
    ):
        assert main([str(video), "--dry-run", "--json"]) == 0

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["qa"]["translation_failures"] == 1


def test_translation_failures_over_threshold_return_exit_five(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    words = [Word(index, index + 1, f"word {index}") for index in range(4)]

    def fail_translation(cues, *_args, **_kwargs):
        for cue in cues:
            cue.translation_failed = True

    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 4.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=(words, "en")),
        patch("hs.cli.segment_words", return_value=[Cue(index, index + 1, "source") for index in range(4)]),
        patch("hs.cli.TranslationClient.translate_cues", side_effect=fail_translation),
    ):
        assert main([str(video)]) == 5


def test_unavailable_translation_server_returns_exit_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    monkeypatch.setattr("hs.cli.server_is_healthy", lambda _url: False)
    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([Word(0, 1, "hello")], "en")),
    ):
        assert main(
            [str(video), "--api-base", "http://127.0.0.1:8189", "--server-wait", "0"]
        ) == 5

    error = capsys.readouterr().err
    assert "http://127.0.0.1:8189/health" in error
    assert "systemctl --user start llama-translategemma" in error


def test_server_preflight_polls_every_five_seconds_until_healthy() -> None:
    clock = iter([0.0, 0.0, 5.0, 10.0])
    health = iter([False, False, True])

    with (
        patch("hs.cli.time.monotonic", side_effect=lambda: next(clock)),
        patch("hs.cli.time.sleep") as sleep,
        patch("hs.cli.server_is_healthy", side_effect=lambda _url: next(health)),
    ):
        assert wait_for_translation_server("http://127.0.0.1:8189", 300.0) is True

    assert sleep.call_args_list == [call(5.0), call(5.0)]


def test_repeat_run_uses_cache_and_force_bypasses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "result.mp4"
    cache_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: cache_home)
    word = Word(0, 2, "hello")
    report = {
        "cps_violations": [],
        "forced_cues": [],
        "silence_overlaps": [],
        "translation_failures": [],
        "boundary_issues": [],
    }

    def write_report_files(_report, json_path, markdown_path):
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("qa", encoding="utf-8")

    def write_ass_file(_cues, path, *_args):
        path.write_text("ass", encoding="utf-8")

    def write_report_files(_report, json_path, markdown_path):
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("qa", encoding="utf-8")

    def burn_file(_video, _ass, path):
        path.write_bytes(b"rendered")
        return "test"

    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([word], "en")) as transcribe_mock,
        patch("hs.cli.TranslationClient.translate_cues"),
        patch("hs.cli.write_ass", side_effect=write_ass_file),
        patch("hs.cli.build_report", return_value=report),
        patch("hs.cli.write_report", side_effect=write_report_files),
        patch("hs.cli.burn_subtitles", side_effect=burn_file),
    ):
        argv = [str(video), "-o", str(output)]
        assert main(argv) == 0
        assert main(argv) == 0
        assert transcribe_mock.call_count == 1
        assert main([*argv, "--force"]) == 0
        assert transcribe_mock.call_count == 2


def test_srt_only_writes_subtitles_and_jsonl_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "result.mp4"
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    report = {
        "cps_violations": [],
        "forced_cues": [],
        "silence_overlaps": [],
        "translation_failures": [],
        "boundary_issues": [],
    }

    def write_report_files(_report, json_path, markdown_path):
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("qa", encoding="utf-8")

    with (
        patch("hs.cli.probe_video", return_value={"width": 1, "height": 1, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([Word(0, 2, "hello")], "en")),
        patch("hs.cli.TranslationClient.translate_cues"),
        patch("hs.cli.build_report", return_value=report),
        patch("hs.cli.write_report", side_effect=write_report_files),
        patch("hs.cli.burn_subtitles") as burn_mock,
    ):
        assert main(
            [str(video), "-o", str(output), "--srt-only", "--progress", str(progress)]
        ) == 0

    assert output.with_suffix(".ass").is_file()
    assert output.with_suffix(".srt").is_file()
    burn_mock.assert_not_called()
    stages = [json.loads(line)["stage"] for line in progress.read_text().splitlines()]
    assert stages == ["start", "probe", "asr", "translate", "qa", "done"]


def test_qa_is_soft_by_default_and_strict_returns_seven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    report = {
        "cps_violations": [{"cue": 1, "cps": 21.0}],
        "forced_cues": [{"cue": 1, "cps": 21.0}],
        "silence_overlaps": [],
        "translation_failures": [],
        "boundary_issues": [],
    }

    def write_qa_files(_report, json_path, markdown_path):
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("qa", encoding="utf-8")

    with (
        patch("hs.cli.probe_video", return_value={"width": 1920, "height": 1080, "duration": 2.0}),
        patch("hs.cli.has_audio_stream", return_value=True),
        patch("hs.cli.extract_audio"),
        patch("hs.cli.transcribe", return_value=([Word(0, 2, "one two three")], "en")),
        patch("hs.cli.TranslationClient.translate_cues"),
        patch("hs.cli.build_report", return_value=report),
        patch("hs.cli.write_report", side_effect=write_qa_files),
        patch("hs.cli.burn_subtitles", return_value="test") as burn_mock,
    ):
        assert main([str(video), "--force"]) == 0
        assert main([str(video), "--force", "--strict"]) == 7

    burn_mock.assert_called_once()
