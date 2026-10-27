from hs.models import Cue, Word
from hs.qa import parse_silences, validate_boundaries


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
