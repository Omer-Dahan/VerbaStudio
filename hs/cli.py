import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import subprocess
from pathlib import Path

from hs.ass import DEFAULT_FONT, subtitle_geometry, write_ass
from hs.audio import extract_audio, has_audio_stream, probe_video
from hs.burn import burn_subtitles
from hs.cache import cache_key, restore_cache, store_cache
from hs.cues import (
    adaptive_cps_target,
    characters_per_second,
    enforce_reading_speed,
    normalize_cues,
    word_sequence_is_complete,
)
from hs.qa import build_report, write_report
from hs.segment import segment_words
from hs.srt import write_srt
from hs.transcribe import DEFAULT_MODEL, transcribe
from hs.translate import (
    DEFAULT_BASE_URL,
    DEFAULT_GLOSSARY_PATH,
    TranslationClient,
    server_is_healthy,
)


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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ASR model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--language", choices=("en", "he"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-type",
        default="int8_float16",
        help="CTranslate2 compute type (default: int8_float16; fits 16GB with the translation server resident)",
    )
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--translate-mode", choices=("chat", "completion", "api"), default="chat")
    parser.add_argument("--api-base", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--server-wait",
        type=float,
        default=300.0,
        help="Seconds to wait for the translation server to become healthy (default: 300)",
    )
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--gender", choices=("m", "f"))
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument(
        "--size-scale",
        type=_positive_float,
        default=1.20,
        help="Scale the frame-derived subtitle size (default: 1.20)",
    )
    parser.add_argument(
        "--box-alpha",
        type=_unit_float,
        default=0.55,
        help="Subtitle box opacity from 0 to 1; 0 uses the outline style (default: 0.55)",
    )
    parser.add_argument(
        "--outline",
        type=_nonnegative_float,
        default=2.0,
        help="Outline width when --box-alpha is 0 (default: 2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write ASS and QA files without burning video")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable run manifest")
    parser.add_argument("--force", action="store_true", help="Ignore cached transcription and translation")
    parser.add_argument("--srt-only", action="store_true", help="Write SRT and ASS files without burning")
    parser.add_argument("--progress", type=Path, help="Append stage updates as JSON lines")
    parser.add_argument("--strict", action="store_true", help="Exit 7 when subtitle QA violations remain")
    parser.add_argument(
        "--min-cue-duration",
        type=_positive_float,
        default=0.40,
        help="Minimum cue duration in seconds (default: 0.40)",
    )
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
    _progress(args.progress, "start")
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
    srt_path = artifact_base.with_suffix(".srt")
    qa_json = artifact_base.with_suffix(".qa.json")
    qa_markdown = artifact_base.with_suffix(".qa.md")
    cache_settings = _cache_settings(args)
    cache_dir = Path.home() / ".cache" / "hebrew-subs" / cache_key(video, cache_settings)
    cache_artifacts = {
        "subtitles.ass": ass_path,
        "subtitles.srt": srt_path,
        "qa.json": qa_json,
        "qa.md": qa_markdown,
    }
    if not args.force:
        cached = restore_cache(cache_dir, cache_artifacts)
        if cached is not None and (args.dry_run or args.srt_only or output.is_file()):
            manifest.update(cached["manifest"])
            manifest["out"] = None if args.dry_run or args.srt_only else str(output)
            manifest["srt"] = str(srt_path)
            manifest["ok"] = True
            manifest["exit"] = 0
            manifest["secs"] = {
                "asr": 0.0,
                "translate": 0.0,
                "burn": 0.0,
                "total": round(time.perf_counter() - started, 3),
            }
            print(f"Cache: {cache_dir}", file=sys.stderr)
            _progress(args.progress, "cache")
            _progress(args.progress, "done")
            return 0

    try:
        metadata = probe_video(video)
        if not has_audio_stream(video):
            raise PipelineError(f"No audio stream found in {video}", 3)
    except PipelineError:
        raise
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"Cannot read input video: {error}", 2) from error
    _progress(args.progress, "probe")

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
    _progress(args.progress, "asr")

    _, _, max_chars = subtitle_geometry(
        int(metadata["width"]), int(metadata["height"]), args.size_scale
    )
    cues = segment_words(words, max_chars=max_chars)
    if not cues:
        raise PipelineError("ASR failed: Whisper produced no timestamped words", 4)
    source_language = args.language or detected_language
    manifest["lang"] = source_language
    translate_started = time.perf_counter()
    if not args.no_translate:
        if not wait_for_translation_server(args.api_base, args.server_wait):
            health_url = f"{args.api_base.rstrip('/')}/health"
            raise PipelineError(
                f"Translation server {health_url} is unavailable after {args.server_wait:g}s; "
                "run: systemctl --user start llama-translategemma",
                5,
            )
        try:
            TranslationClient(args.api_base, args.translate_mode).translate_cues(
                cues, source_language, args.glossary, args.gender
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise PipelineError(f"Translation failed: {error}", 5) from error
        failure_count = sum(cue.translation_failed for cue in cues)
        failure_limit = max(3, len(cues) * 0.1)
        if failure_count > failure_limit:
            raise PipelineError(
                f"Translation failed for {failure_count} of {len(cues)} cues; "
                f"abort threshold is more than max(3, 10% of cues) ({failure_limit:g})",
                5,
            )
    manifest["secs"]["translate"] = round(time.perf_counter() - translate_started, 3)
    _progress(args.progress, "translate")
    cps_target = adaptive_cps_target(cues, float(metadata["duration"]))
    cues = enforce_reading_speed(
        normalize_cues(cues, min_cue_duration=args.min_cue_duration),
        max_cps=cps_target,
        min_cue_duration=args.min_cue_duration,
        max_chars=max_chars,
    )
    manifest["no_word_loss"] = word_sequence_is_complete(cues, words)
    if not manifest["no_word_loss"]:
        raise PipelineError("Cue normalization lost or duplicated transcript words", 7)
    write_ass(
        cues,
        ass_path,
        int(metadata["width"]),
        int(metadata["height"]),
        args.font,
        args.size_scale,
        args.box_alpha,
        args.outline,
    )
    write_srt(cues, srt_path)
    report = build_report(cues, video, float(metadata["duration"]), cps_target)
    write_report(report, qa_json, qa_markdown)

    if args.strict and (report["cps_violations"] or report.get("boundary_issues")):
        raise PipelineError("QA failed: subtitle timing or reading-speed violations remain", 7)
    _progress(args.progress, "qa")

    manifest["ass"] = str(ass_path)
    manifest["srt"] = str(srt_path)
    manifest["cue_count"] = len(cues)
    manifest["cps_max"] = round(max((characters_per_second(cue) for cue in cues), default=0.0), 2)
    manifest["cps_target"] = round(cps_target, 2)
    manifest["qa"] = {
        "cps_violations": report["cps_violations"],
        "forced_cues": report["forced_cues"],
        "silence_overlaps": report["silence_overlaps"],
        "translation_failures": report["translation_failures"],
        "passthrough": report["passthrough"],
    }
    print(f"ASS: {ass_path}", file=sys.stderr)
    print(f"QA: {qa_json}", file=sys.stderr)
    store_cache(
        cache_dir,
        cache_artifacts,
        {
            "manifest": {
                "ass": str(ass_path),
                "lang": manifest["lang"],
                "cue_count": manifest["cue_count"],
                "cps_max": manifest["cps_max"],
                "cps_target": manifest["cps_target"],
                "no_word_loss": manifest["no_word_loss"],
                "qa": manifest["qa"],
            },
            "transcript": " ".join(word.text for cue in cues for word in cue.words),
            "cue_source": " ".join(cue.source for cue in cues),
        },
    )
    if args.dry_run or args.srt_only:
        manifest["ok"] = True
        manifest["secs"]["total"] = round(time.perf_counter() - started, 3)
        _progress(args.progress, "done")
        return 0
    burn_started = time.perf_counter()
    try:
        codec = burn_subtitles(video, ass_path, output)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"Burn failed: {error}", 6) from error
    manifest["secs"]["burn"] = round(time.perf_counter() - burn_started, 3)
    _progress(args.progress, "burn")
    manifest["out"] = str(output)
    manifest["ok"] = True
    manifest["secs"]["total"] = round(time.perf_counter() - started, 3)
    print(f"Video: {output} ({codec})", file=sys.stderr)
    _progress(args.progress, "done")
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
        "cps_target": 17.0,
        "no_word_loss": False,
        "qa": {
            "cps_violations": [],
            "forced_cues": [],
            "silence_overlaps": [],
            "translation_failures": [],
            "passthrough": 0,
        },
        "secs": {"asr": 0.0, "translate": 0.0, "burn": 0.0, "total": 0.0},
        "exit": 0,
    }


def _cache_settings(args: argparse.Namespace) -> dict:
    glossary_digest = hashlib.sha256(DEFAULT_GLOSSARY_PATH.read_bytes())
    if args.glossary:
        glossary_digest.update(args.glossary.read_bytes())
    return {
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "translate": not args.no_translate,
        "translate_mode": args.translate_mode,
        "api_base": args.api_base,
        "server_wait": args.server_wait,
        "glossary": glossary_digest.hexdigest(),
        "font": args.font,
        "size_scale": args.size_scale,
        "box_alpha": args.box_alpha,
        "outline": args.outline,
        "gender": args.gender,
        "strict": args.strict,
        "min_cue_duration": args.min_cue_duration,
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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def wait_for_translation_server(base_url: str, wait_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        if server_is_healthy(base_url):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(5.0, remaining))


def _progress(path: Path | None, stage: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
