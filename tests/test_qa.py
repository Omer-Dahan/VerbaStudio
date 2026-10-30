from hs.models import Cue, Word
from hs.cues import adaptive_cps_target
from hs.qa import build_report, parse_silences, validate_boundaries


def test_parse_silences_including_open_ended_interval() -> None:
    output = """
[silencedetect] silence_start: 1.25
[silencedetect] silence_end: 2.75 | silence_duration: 1.5
[silencedetect] silence_start: 9.5
"""
    assert parse_silences(output, 10.0) == [(1.25, 2.75), (9.5, 10.0)]


def test_boundary_validation_reports_overlap_and_video_overrun() -> None:
    cues = [Cue(0.0, 2.0, "one"), Cue(1.9, 5.1, "two")]
    issues = validate_boundaries(cues, duration=5.0)

    assert "Cue 2 overlaps the previous cue" in issues
    assert "Cue 2 ends after the video" in issues


def test_low_confidence_word_is_preserved_by_model() -> None:
    word = Word(0.0, 0.4, "uncertain", 0.2)
    cue = Cue(0.0, 1.2, "uncertain", words=[word])
    assert cue.words == [word]


def test_report_lists_forced_cues(tmp_path, monkeypatch) -> None:
    cue = Cue(0.0, 1.0, "one two three", "טקסט ארוך", cps_forced=True)
    monkeypatch.setattr("hs.qa.detect_silences", lambda *_args: [])

    report = build_report([cue], tmp_path / "video.mp4", 1.0)

    assert report["forced_cues"] == [{"cue": 1, "cps": 9.0}]


def test_report_separates_failures_from_passthrough_cues(tmp_path, monkeypatch) -> None:
    failed = Cue(0.0, 1.0, "Real words", translation_failed=True)
    name = Cue(1.0, 2.0, "Minecraft", translation_passthrough=True)
    monkeypatch.setattr("hs.qa.detect_silences", lambda *_args: [])

    report = build_report([failed, name], tmp_path / "video.mp4", 2.0)

    assert report["translation_failures"] == [1]
    assert report["passthrough"] == 1


def test_adaptive_cps_target_stays_at_floor_for_normal_speech() -> None:
    cues = [Cue(0.0, 2.0, "A normal sentence")]

    assert adaptive_cps_target(cues, 2.0) == 17.0


def test_fast_speech_raises_target_and_only_reports_excess(tmp_path, monkeypatch) -> None:
    cues = [Cue(0.0, 2.0, "x" * 40, "א" * 44, cps_forced=True)]
    target = adaptive_cps_target(cues, 2.0)
    monkeypatch.setattr("hs.qa.detect_silences", lambda *_args: [])

    report = build_report(cues, tmp_path / "video.mp4", 2.0, target)

    assert target == 21.0
    assert report["cps_target"] == 21.0
    assert report["cps_violations"] == [{"cue": 1, "cps": 22.0}]
    assert report["forced_cues"] == [{"cue": 1, "cps": 22.0}]
