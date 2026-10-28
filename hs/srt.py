from pathlib import Path

from hs.models import Cue


def write_srt(cues: list[Cue], destination: Path) -> Path:
    blocks = [
        f"{index}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.display_text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(blocks), encoding="utf-8")
    return destination


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{fraction:03d}"
