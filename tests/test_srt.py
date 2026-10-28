from pathlib import Path

from hs.models import Cue
from hs.srt import write_srt


def test_write_srt_formats_numbered_cues(tmp_path: Path) -> None:
    destination = tmp_path / "captions.srt"

    write_srt([Cue(1.25, 3.5, "hello", "שלום")], destination)

    assert destination.read_text(encoding="utf-8") == (
        "1\n00:00:01,250 --> 00:00:03,500\nשלום\n"
    )
