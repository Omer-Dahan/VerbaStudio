from hs.models import Cue, Word
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
