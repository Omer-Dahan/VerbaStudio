import json
import re
import subprocess
from pathlib import Path

from hs.ass import layout_cue_lines, subtitle_geometry, subtitle_layout
from hs.cues import characters_per_second
from hs.models import Cue


SILENCE_START = re.compile(r"silence_start: ([0-9.]+)")
SILENCE_END = re.compile(r"silence_end: ([0-9.]+)")


def build_report(
    cues: list[Cue],
    video_path: Path,
    duration: float,
    cps_target: float = 17.0,
    width: int | None = None,
    height: int | None = None,
    size_scale: float = 1.20,
    caption_band: list[int] | None = None,
    position: str = "auto",
    margin_v: int | None = None,
    cue_style: str = "sentences",
    clause_split_fallback: int = 0,
    min_cue_duration: float = 0.4,
) -> dict:
    silences = detect_silences(video_path, duration)
    boundary_issues = validate_boundaries(cues, duration)
    durations = [cue.end - cue.start for cue in cues]
    low_confidence = [
        {
            "cue": index,
            "word": word.text,
            "start": word.start,
            "probability": word.probability,
        }
        for index, cue in enumerate(cues, start=1)
        for word in cue.words
        if word.probability < 0.35
    ]
    silence_overlaps = []
    for index, cue in enumerate(cues, start=1):
        for start, end in silences:
            overlap = max(0.0, min(cue.end, end) - max(cue.start, start))
            if overlap:
                silence_overlaps.append(
                    {
                        "cue": index,
                        "silence_start": start,
                        "silence_end": end,
                        "overlap_seconds": round(overlap, 3),
                        "fully_silent": start <= cue.start and end >= cue.end,
                    }
                )

    layout_bands, layout_conflicts = analyze_layout(
        cues, width, height, size_scale, caption_band, position, margin_v
    )
    three_line_cues = _count_three_line_cues(cues, width, height, size_scale)
    return {
        "cue_count": len(cues),
        "cue_style": cue_style,
        "clauses": len(cues) if cue_style == "clauses" else 0,
        "clause_split_fallback": clause_split_fallback,
        "cues_below_min": sum(value < min_cue_duration for value in durations),
        "sentences": len(cues),
        "mean_sentence_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "max_sentence_s": round(max(durations), 3) if durations else 0.0,
        "three_line_cues": three_line_cues,
        "average_duration": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "under_one_second": [index for index, value in enumerate(durations, 1) if value < 1.0],
        "cps_target": round(cps_target, 2),
        "cps_violations": [
            {"cue": index, "cps": round(characters_per_second(cue), 2)}
            for index, cue in enumerate(cues, 1)
            if characters_per_second(cue) > cps_target
        ],
        "forced_cues": [
            {"cue": index, "cps": round(characters_per_second(cue), 2)}
            for index, cue in enumerate(cues, 1)
            if cue.cps_forced
        ],
        "translation_failures": [
            index for index, cue in enumerate(cues, 1) if cue.translation_failed
        ],
        "passthrough": sum(cue.translation_passthrough for cue in cues),
        "low_confidence_words": low_confidence,
        "boundary_issues": boundary_issues,
        "silences": [{"start": start, "end": end} for start, end in silences],
        "silence_overlaps": silence_overlaps,
        "layout_bands": layout_bands,
        "layout_conflicts": layout_conflicts,
    }


def _count_three_line_cues(
    cues: list[Cue], width: int | None, height: int | None, size_scale: float
) -> int:
    if width is None or height is None:
        return 0
    font_size, _, max_chars = subtitle_geometry(width, height, size_scale)
    count = 0
    for cue in cues:
        _, used_font = layout_cue_lines(cue.display_text, max_chars, font_size)
        if used_font != font_size:
            count += 1
    return count


def write_report(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Subtitle QA",
        "",
        f"- Cue style: {report.get('cue_style', 'sentences')}",
        f"- Clauses: {report.get('clauses', 0)}",
        f"- Clause split fallbacks: {report.get('clause_split_fallback', 0)}",
        f"- Cues below minimum: {report.get('cues_below_min', 0)}",
        f"- Cues: {report['cue_count']}",
        f"- Sentences: {report.get('sentences', report['cue_count'])}",
        f"- Mean sentence duration: {report.get('mean_sentence_s', 0.0):.3f}s",
        f"- Max sentence duration: {report.get('max_sentence_s', 0.0):.3f}s",
        f"- Three-line cues: {report.get('three_line_cues', 0)}",
        f"- Average duration: {report['average_duration']:.3f}s",
        f"- Under one second: {len(report['under_one_second'])}",
        f"- CPS target: {report['cps_target']:.2f}",
        f"- CPS violations: {len(report['cps_violations'])}",
        f"- Forced cues: {len(report['forced_cues'])}",
        f"- Translation failures: {len(report['translation_failures'])}",
        f"- Passthrough cues: {report['passthrough']}",
        f"- Low-confidence words: {len(report['low_confidence_words'])}",
        f"- Boundary issues: {len(report['boundary_issues'])}",
        f"- Silence overlaps: {len(report['silence_overlaps'])}",
        f"- Layout conflicts: {len(report.get('layout_conflicts', []))}",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_layout(
    cues: list[Cue],
    width: int | None,
    height: int | None,
    size_scale: float,
    caption_band: list[int] | None,
    position: str,
    margin_v: int | None,
) -> tuple[list[dict], list[dict]]:
    if width is None or height is None or not caption_band:
        return [], []

    font_size, _, max_chars = subtitle_geometry(width, height, size_scale)
    alignment, margin = subtitle_layout(height, caption_band, position, margin_v)
    caption_top = round(height * caption_band[0] / 100)
    caption_bottom = round(height * caption_band[1] / 100)
    bands = []
    conflicts = []
    for index, cue in enumerate(cues, start=1):
        lines, used_font = layout_cue_lines(cue.display_text, max_chars, font_size)
        line_count = len(lines)
        text_height = used_font * line_count
        if alignment == 8:
            text_top = margin
            text_bottom = margin + text_height
        else:
            text_bottom = height - margin
            text_top = text_bottom - text_height
        band = {
            "cue": index,
            "font_size": used_font,
            "line_count": line_count,
            "margin_v": margin,
            "text_band": [text_top, text_bottom],
            "text_band_pct": [
                round(text_top * 100 / height, 2),
                round(text_bottom * 100 / height, 2),
            ],
            "caption_band": [caption_top, caption_bottom],
            "caption_band_pct": caption_band,
            "gap": caption_top - text_bottom,
        }
        bands.append(band)
        if text_bottom > caption_top and text_top < caption_bottom:
            conflicts.append(band)
    return bands, conflicts


def validate_boundaries(cues: list[Cue], duration: float) -> list[str]:
    issues: list[str] = []
    previous_end = 0.0
    for index, cue in enumerate(cues, start=1):
        if cue.start < 0:
            issues.append(f"Cue {index} starts before zero")
        if cue.end <= cue.start:
            issues.append(f"Cue {index} has a non-positive duration")
        if cue.end > duration + 0.01:
            issues.append(f"Cue {index} ends after the video")
        if cue.start < previous_end - 0.001:
            issues.append(f"Cue {index} overlaps the previous cue")
        previous_end = max(previous_end, cue.end)
    return issues


def detect_silences(
    video_path: Path,
    duration: float,
    noise: str = "-35dB",
    minimum: float = 0.5,
) -> list[tuple[float, float]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-af",
        f"silencedetect=noise={noise}:d={minimum}",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, stderr=completed.stderr)
    return parse_silences(completed.stderr, duration)


def parse_silences(output: str, duration: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    open_start: float | None = None
    for line in output.splitlines():
        if match := SILENCE_START.search(line):
            open_start = float(match.group(1))
        elif match := SILENCE_END.search(line):
            if open_start is not None:
                intervals.append((open_start, float(match.group(1))))
                open_start = None
    if open_start is not None:
        intervals.append((open_start, duration))
    return intervals
