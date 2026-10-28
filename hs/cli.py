import argparse
import os
import sys
import tempfile
from pathlib import Path

from hs.ass import DEFAULT_FONT, write_ass
from hs.audio import extract_audio, probe_video
from hs.burn import burn_subtitles
from hs.cues import enforce_reading_speed
from hs.qa import build_report, write_report
from hs.segment import segment_words
from hs.transcribe import DEFAULT_MODEL, transcribe
from hs.translate import DEFAULT_BASE_URL, TranslationClient


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
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--dry-run", action="store_true", help="Write ASS and QA files without burning video")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    if args.glossary and not args.glossary.is_file():
        raise SystemExit(f"Glossary not found: {args.glossary}")

    try:
        output = resolve_output_path(video, args.output)
    except ValueError as error:
        build_parser().error(str(error))
    artifact_base = output.with_suffix("")
    ass_path = artifact_base.with_suffix(".ass")
    qa_json = artifact_base.with_suffix(".qa.json")
    qa_markdown = artifact_base.with_suffix(".qa.md")
    metadata = probe_video(video)

    with tempfile.TemporaryDirectory(prefix="hebrew-subtitles-") as temp_dir:
        audio_path = extract_audio(video, Path(temp_dir) / "audio.wav")
        words, detected_language = transcribe(
            audio_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
        )

    cues = segment_words(words)
    if not cues:
        raise RuntimeError("Whisper produced no timestamped words")
    source_language = args.language or detected_language
    if not args.no_translate:
        TranslationClient(args.api_base, args.translate_mode).translate_cues(
            cues, source_language, args.glossary
        )
    cues = enforce_reading_speed(cues)
    write_ass(cues, ass_path, int(metadata["width"]), int(metadata["height"]), args.font)
    report = build_report(cues, video, float(metadata["duration"]))
    write_report(report, qa_json, qa_markdown)

    print(f"ASS: {ass_path}")
    print(f"QA: {qa_json}")
    if args.dry_run:
        return 0
    codec = burn_subtitles(video, ass_path, output)
    print(f"Video: {output} ({codec})")
    return 0


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
