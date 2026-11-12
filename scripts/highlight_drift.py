#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from pathlib import Path


HIGHLIGHT = re.compile(r"\\c&H(?:0000FFFF|00FFFFFF)&")
DIALOGUE = re.compile(r"^Dialogue: 0,([^,]+),([^,]+),.*?,,(.*)$")
SRT_TIMING = re.compile(r"^(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})$")


def _timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _srt_cue_ranges(path: Path) -> list[tuple[float, float]]:
    if not path.is_file():
        return []
    ranges = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = SRT_TIMING.match(line)
        if match:
            ranges.append(
                tuple(_timestamp(value.replace(",", ".")) for value in match.groups())
            )
    return ranges


def _highlight_cues(path: Path) -> list[list[tuple[float, float]]]:
    events = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = DIALOGUE.match(line)
        if match and HIGHLIGHT.search(match.group(3)):
            events.append((_timestamp(match.group(1)), _timestamp(match.group(2))))

    cue_ranges = _srt_cue_ranges(path.with_suffix(".srt"))
    if cue_ranges:
        return [
            [event for event in events if start - 0.011 <= event[0] < end + 0.011]
            for start, end in cue_ranges
        ]

    cues: list[list[tuple[float, float]]] = []
    for event in events:
        if not cues or abs(cues[-1][-1][1] - event[0]) > 0.011:
            cues.append([])
        cues[-1].append(event)
    return cues


def _load_words(path: Path) -> list[dict]:
    words = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(words, list):
        raise ValueError("words JSON must be a list")
    return words


def _transcribe(video: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "Systran/faster-whisper-large-v3",
        device="cuda",
        compute_type="int8_float16",
    )
    segments, _ = model.transcribe(
        str(video),
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    return [
        {"start": float(word.start), "end": float(word.end), "text": word.word.strip()}
        for segment in segments
        for word in (segment.words or [])
        if word.start is not None and word.end is not None and word.word.strip()
    ]


def _percentile(values: list[float], percentile: float) -> float:
    position = (len(values) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure ASS per-word highlight drift against ASR word starts."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("ass", type=Path)
    parser.add_argument("--words", type=Path, help="JSON list of ASR words; skips transcription")
    args = parser.parse_args()

    cues = _highlight_cues(args.ass)
    words = _load_words(args.words) if args.words else _transcribe(args.video)
    if not cues or not words:
        parser.error("the ASS highlight events and words list must both be non-empty")

    matched = []
    fallback_cues = set()
    for cue_index, events in enumerate(cues, 1):
        if not events:
            continue
        cue_words = [
            word
            for word in words
            if events[0][0] - 0.011 <= float(word["start"]) <= events[-1][1] + 0.011
        ]
        if not cue_words:
            continue
        matched_word_indexes = []
        for event_index, (event_start, _) in enumerate(events):
            word_index = min(
                range(len(cue_words)),
                key=lambda index: abs(float(cue_words[index]["start"]) - event_start),
            )
            matched_word_indexes.append(word_index)
            word = cue_words[word_index]
            drift = event_start - float(word["start"])
            matched.append((abs(drift), drift, cue_index, event_start, word))
        if len(set(matched_word_indexes)) < len(events):
            fallback_cues.add(cue_index)

    if not matched:
        parser.error("no ASR words overlap the ASS highlight cues")

    absolute = sorted(item[0] for item in matched)
    signed = [item[1] for item in matched]
    print(f"mean_abs_drift_s={statistics.fmean(absolute):.3f}")
    print(f"median={statistics.median(absolute):.3f}")
    print(f"p90={_percentile(absolute, 0.90):.3f}")
    print(f"max={max(absolute):.3f}")
    print(f"signed_mean={statistics.fmean(signed):.3f}")
    worst_by_cue = {}
    for item in matched:
        cue_index = item[2]
        if cue_index not in worst_by_cue or item[0] > worst_by_cue[cue_index][0]:
            worst_by_cue[cue_index] = item
    print("five_worst_cues:")
    for absolute_drift, drift, cue_index, event_start, word in sorted(
        worst_by_cue.values(), reverse=True
    )[:5]:
        print(
            f"  cue={cue_index} event_start={event_start:.2f} "
            f"word_start={float(word['start']):.2f} drift={drift:+.3f} "
            f"abs={absolute_drift:.3f} word={word['text']}"
            + (" fallback=display_words_gt_english_words" if cue_index in fallback_cues else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
