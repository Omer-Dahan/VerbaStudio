import argparse
import json
import os
import sys
import tempfile
import time
import subprocess
from pathlib import Path

from hs.ass import DEFAULT_FONT, write_ass
from hs.audio import extract_audio, has_audio_stream, probe_video
from hs.burn import burn_subtitles
from hs.cues import characters_per_second, enforce_reading_speed
from hs.qa import build_report, write_report
from hs.segment import segment_words
from hs.transcribe import DEFAULT_MODEL, transcribe
from hs.translate import DEFAULT_BASE_URL, TranslationClient


class PipelineError(RuntimeError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hebrew-subtitles",
        description="Burn locally generated Hebrew subtitles into a video.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", choices=("en", "he"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--translate-mode", choices=("chat", "completion", "api"), default="chat")
    parser.add_argument("--api-base", default=DEFAULT_BASE_URL)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--gender", choices=("m", "f"))
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--dry-run", action="store_true", help="Write ASS and QA files without burning video")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable run manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    manifest = _empty_manifest(args)
    try:
        exit_code = _run(args, manifest, started)
    except PipelineError as error:
        print(f"Error: {error}", file=sys.stderr)
        manifest["secs"]["total"] = round(time.perf_counter() - started, 3)
        manifest["exit"] = error.exit_code
        if args.json:
            print(json.dumps(manifest, ensure_ascii=False))
        return error.exit_code

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False))
    return exit_code


def _run(args: argparse.Namespace, manifest: dict, started: float) -> int:
    video = args.video.resolve()
    if not video.is_file():
        raise PipelineError(f"Video not found: {video}", 2)
    if args.glossary and not args.glossary.is_file():
        raise PipelineError(f"Glossary not found: {args.glossary}", 2)

    try:
        output = resolve_output_path(video, args.output)
    except ValueError as error:
        raise PipelineError(str(error), 2) from error
    artifact_base = output.with_suffix("")
    ass_path = artifact_base.with_suffix(".ass")
    qa_json = artifact_base.with_suffix(".qa.json")
    qa_markdown = artifact_base.with_suffix(".qa.md")
    try:
        metadata = probe_video(video)
        if not has_audio_stream(video):
            raise PipelineError(f"No audio stream found in {video}", 3)
    except PipelineError:
        raise
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"Cannot read input video: {error}", 2) from error

    asr_started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="hebrew-subtitles-") as temp_dir:
            audio_path = extract_audio(video, Path(temp_dir) / "audio.wav")
            words, detected_language = transcribe(
                audio_path,
                model_name=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
            )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"ASR failed: {error}", 4) from error
    manifest["secs"]["asr"] = round(time.perf_counter() - asr_started, 3)

    cues = segment_words(words)
    if not cues:
        raise PipelineError("ASR failed: Whisper produced no timestamped words", 4)
    source_language = args.language or detected_language
    manifest["lang"] = source_language
    translate_started = time.perf_counter()
    if not args.no_translate:
        try:
            TranslationClient(args.api_base, args.translate_mode).translate_cues(
                cues, source_language, args.glossary, args.gender
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise PipelineError(f"Translation failed: {error}", 5) from error
        if any(cue.translation_failed for cue in cues):
            raise PipelineError("Translation failed after all retry attempts", 5)
    manifest["secs"]["translate"] = round(time.perf_counter() - translate_started, 3)
    cues = enforce_reading_speed(cues)
    write_ass(cues, ass_path, int(metadata["width"]), int(metadata["height"]), args.font)
    report = build_report(cues, video, float(metadata["duration"]))
    write_report(report, qa_json, qa_markdown)

    if report["cps_violations"] or report.get("boundary_issues"):
        raise PipelineError("QA failed: subtitle timing or reading-speed violations remain", 7)

    manifest["ass"] = str(ass_path)
    manifest["cue_count"] = len(cues)
    manifest["cps_max"] = round(max((characters_per_second(cue) for cue in cues), default=0.0), 2)
    manifest["qa"] = {
        "cps_violations": len(report["cps_violations"]),
        "silence_overlaps": len(report["silence_overlaps"]),
        "translation_failures": len(report["translation_failures"]),
    }
    print(f"ASS: {ass_path}", file=sys.stderr)
    print(f"QA: {qa_json}", file=sys.stderr)
    if args.dry_run:
        manifest["ok"] = True
        manifest["secs"]["total"] = round(time.perf_counter() - started, 3)
        return 0
    burn_started = time.perf_counter()
    try:
        codec = burn_subtitles(video, ass_path, output)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"Burn failed: {error}", 6) from error
    manifest["secs"]["burn"] = round(time.perf_counter() - burn_started, 3)
    manifest["out"] = str(output)
    manifest["ok"] = True
    manifest["secs"]["total"] = round(time.perf_counter() - started, 3)
    print(f"Video: {output} ({codec})", file=sys.stderr)
    return 0


def _empty_manifest(args: argparse.Namespace) -> dict:
    return {
        "ok": False,
        "out": None,
        "srt": None,
        "ass": "",
        "lang": args.language or "",
        "engine": "none" if args.no_translate else "llama.cpp",
        "model": args.model,
        "compute_type": args.compute_type,
        "cue_count": 0,
        "cps_max": 0.0,
        "qa": {"cps_violations": 0, "silence_overlaps": 0, "translation_failures": 0},
        "secs": {"asr": 0.0, "translate": 0.0, "burn": 0.0, "total": 0.0},
        "exit": 0,
    }


def resolve_output_path(video: Path, output: str | None) -> Path:
    if output is None:
        return video.with_name(f"{video.stem}.subtitled.mp4").resolve()

    requested = Path(output).expanduser()
    if requested.is_dir() or output.endswith((os.sep, "/")):
        return (requested / f"{video.stem}_he.mp4").resolve()
    if requested.suffix.casefold() == ".mp4":
        return requested.resolve()
    raise ValueError(
        "output must be an existing directory, a path ending in '/', or an .mp4 file"
    )


if __name__ == "__main__":
    sys.exit(main())
