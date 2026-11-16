import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from hs.ass import (
    LRI,
    PDI,
    RLM,
    _bidi_line,
    _render_token,
    _ass_text,
    _rtl_highlight_segments,
    format_timestamp,
    subtitle_geometry,
    write_ass,
)
from hs.models import Cue, Word
from hs.text import QUOTE_CHARACTERS


def render_black_frame_margins(
    ass_path: Path, width: int, height: int
) -> dict[str, int]:
    fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
    ass_filter = f"ass=filename='{ass_path.resolve()}':fontsdir='{fonts_dir}'"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:d=1:r=1",
        "-vf",
        ass_filter,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    frame = subprocess.run(command, check=True, capture_output=True).stdout
    assert len(frame) == width * height
    visible = [index for index, value in enumerate(frame) if value]
    assert visible, "ASS render did not contain any visible subtitle pixels"
    xs = [index % width for index in visible]
    ys = [index // width for index in visible]
    return {
        "left": min(xs),
        "right": width - 1 - max(xs),
        "top": min(ys),
        "bottom": height - 1 - max(ys),
    }


def test_ass_uses_video_resolution_style_and_rtl_override(tmp_path: Path) -> None:
    destination = tmp_path / "cues.ass"
    write_ass([Cue(1.234, 4.567, "source", "שלום עולם")], destination, 720, 1280)
    content = destination.read_text(encoding="utf-8")

    assert "PlayResX: 720" in content
    assert "PlayResY: 1280" in content
    assert "WrapStyle: 0" in content
    assert "ScaledBorderAndShadow: yes" in content
    assert "Style: Default,Heebo," in content
    assert ",3,8,0,2,40,40,77,1" in content
    assert "&H73000000" in content
    assert "Dialogue: 0,0:00:01.23,0:00:04.57" in content
    assert "{\\q2}שלום עולם\u200f" in content


def test_reveal_ass_emits_short_groups_that_tile_the_sentence(tmp_path: Path) -> None:
    destination = tmp_path / "reveal.ass"
    write_ass(
        [Cue(0, 6.03, "one two three four five six", "אחד שני שלושה ארבעה חמישה שישה")],
        destination,
        720,
        1280,
        cue_style="reveal",
    )

    content = destination.read_text(encoding="utf-8")
    dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]

    assert len(dialogues) == 3
    assert len(dialogues) == len(set(dialogues))
    times = [re.match(r"Dialogue: 0,([^,]+),([^,]+),", line).groups() for line in dialogues]
    assert times[0][0] == "0:00:00.00"
    assert all(left[1] == right[0] for left, right in zip(times, times[1:]))
    assert times[-1][1] == "0:00:06.03"
    for dialogue, (start, end) in zip(dialogues, times):
        plain = re.sub(r"\{[^}]+\}|[\u200e\u200f\u2066\u2069]", "", dialogue.split(",,", 1)[1])
        assert len(plain.split()) <= 4
        assert dialogue.count(r"\N") <= 1
        start_cs = _ass_centiseconds(start)
        end_cs = _ass_centiseconds(end)
        assert end_cs - start_cs <= 250
        assert sum(int(value) for value in re.findall(r"\\k(\d+)", dialogue)) == end_cs - start_cs
    assert "Style: Default,Heebo," in content
    assert "&H00808080" in content


def test_reveal_ass_has_no_duplicate_events_for_two_sentences(tmp_path: Path) -> None:
    destination = tmp_path / "reveal.ass"
    cues = [
        Cue(0, 2, "one two three four", "אחד שני שלושה ארבעה"),
        Cue(2, 4, "five six seven eight", "חמישה שישה שבעה שמונה"),
    ]

    write_ass(cues, destination, 720, 1280, cue_style="reveal")

    events = [line for line in destination.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")]
    assert len(events) == len(set(events))


def _ass_centiseconds(timestamp: str) -> int:
    hours, minutes, seconds = timestamp.split(":")
    whole, fraction = seconds.split(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(whole)) * 100 + int(fraction)


def _style_and_dialogues(content: str) -> tuple[str, list[str]]:
    lines = content.splitlines()
    return (
        next(line for line in lines if line.startswith("Style:")),
        [line for line in lines if line.startswith("Dialogue:")],
    )


def _dialogue_windows(dialogues: list[str]) -> list[tuple[int, int]]:
    timestamps = [
        re.match(r"Dialogue: 0,([^,]+),([^,]+),", dialogue).groups()
        for dialogue in dialogues
    ]
    return [(_ass_centiseconds(start), _ass_centiseconds(end)) for start, end in timestamps]


def test_box_caption_style_is_byte_for_byte_default(tmp_path: Path) -> None:
    cues = [Cue(0, 1, "one", "שלום"), Cue(1, 2, "two", "עולם")]
    default = tmp_path / "default.ass"
    explicit = tmp_path / "box.ass"

    write_ass(cues, default, 720, 1280)
    write_ass(cues, explicit, 720, 1280, caption_style="box")

    assert default.read_bytes() == explicit.read_bytes()
    style, _ = _style_and_dialogues(explicit.read_text(encoding="utf-8"))
    assert style.split(",")[1] == "Heebo"
    assert "Style: Default,Heebo," in default.read_text(encoding="utf-8")


def test_box_rtl_output_matches_caption_style_baseline(tmp_path: Path) -> None:
    destination = tmp_path / "box.ass"
    write_ass(
        [Cue(0, 1, "one two three", "ראשון שני שלישי")],
        destination,
        720,
        1280,
        caption_style="box",
    )

    assert hashlib.sha256(destination.read_bytes()).hexdigest() == (
        "878b07bc4630e34507f9e202e04b50b16973894859b1b3670ff730a2ca48d7fa"
    )


def test_capcut_caption_style_highlights_ordered_words(tmp_path: Path) -> None:
    cue = Cue(0, 1.2, "one two three four", "אחד שני three ארבעה")
    destination = tmp_path / "capcut.ass"
    write_ass([cue], destination, 720, 1280, caption_style="capcut")
    style, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))

    fields = style.split(",")
    assert fields[1] == "Heebo Black"
    assert fields[15] == "1"
    assert float(fields[16]) > 3
    assert all(r"\fscx97" in event and r"\c&H0000FFFF&" in event for event in dialogues)
    windows = [re.match(r"Dialogue: 0,([^,]+),([^,]+),", event).groups() for event in dialogues]
    assert all(_ass_centiseconds(left[1]) <= _ass_centiseconds(right[0]) for left, right in zip(windows, windows[1:]))
    active_words = [
        re.search(r"\{\\c&H0000FFFF&[^}]*\}([^{} ]+)", event).group(1).replace("\u2066", "").replace("\u2069", "")
        for event in dialogues
    ]
    assert active_words == [
        RLM + word if word == "three" else word for word in cue.display_text.split()
    ]


def test_capcut_caption_style_uses_margin_override(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"

    write_ass(
        [Cue(0, 1, "one", "אחד")],
        destination,
        720,
        1024,
        caption_style="capcut",
        margin_v_override=120,
    )

    style, _ = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert style.split(",")[21] == "120"


def test_capcut_caption_style_keeps_preset_margin_by_default(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"

    write_ass(
        [Cue(0, 1, "one", "אחד")],
        destination,
        720,
        1024,
        caption_style="capcut",
    )

    style, _ = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert style.split(",")[21] == "287"


def test_capcut_top_position_wins_over_margin_override(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"

    write_ass(
        [Cue(0, 1, "one", "אחד")],
        destination,
        720,
        1024,
        caption_style="capcut",
        position="top",
        margin_v_override=120,
    )

    style, _ = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    fields = style.split(",")
    assert fields[18] == "8"
    assert fields[21] == "82"


@pytest.mark.parametrize("caption_style", ["card", "broadcast"])
def test_card_and_broadcast_clear_the_caption_band_like_box(
    tmp_path: Path, caption_style: str
) -> None:
    caption_band = [85, 97]

    box_destination = tmp_path / "box.ass"
    write_ass(
        [Cue(0, 1, "one", "אחד")],
        box_destination,
        1920,
        1080,
        caption_style="box",
        caption_band=caption_band,
    )
    box_style, _ = _style_and_dialogues(box_destination.read_text(encoding="utf-8"))
    box_margin_v = int(box_style.split(",")[21])

    destination = tmp_path / f"{caption_style}.ass"
    write_ass(
        [Cue(0, 1, "one", "אחד")],
        destination,
        1920,
        1080,
        caption_style=caption_style,
        caption_band=caption_band,
    )
    style, _ = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    margin_v = int(style.split(",")[21])

    assert margin_v == box_margin_v
    band_top_px = round(1080 * caption_band[0] / 100)
    assert 1080 - margin_v <= band_top_px


@pytest.mark.parametrize("caption_style", ["box", "card", "broadcast"])
def test_margin_v_override_wins_over_caption_band_for_every_style(
    tmp_path: Path, caption_style: str
) -> None:
    destination = tmp_path / f"{caption_style}.ass"
    write_ass(
        [Cue(0, 1, "one", "אחד")],
        destination,
        1920,
        1080,
        caption_style=caption_style,
        caption_band=[85, 97],
        margin_v_override=120,
    )
    style, _ = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert style.split(",")[21] == "120"


def _plain_event_tokens(event: str) -> list[str]:
    text = event.rsplit(",,", 1)[1]
    text = re.sub(r"\{[^}]+\}|[\u200e\u200f\u2066\u2069]", "", text)
    return text.split()


def test_capcut_rtl_events_emit_highlight_runs_in_visual_order(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"
    write_ass(
        [Cue(0, 1, "one two three", "אני רוצה בעל.")],
        destination,
        720,
        1280,
        caption_style="capcut",
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    event_texts = [
        event.rsplit(r"{\q2}", 1)[1].removesuffix("\u200f")
        for event in dialogues
    ]
    opening = (
        r"{\c&H0000FFFF&\fscx97\fscy97"
        r"\t(0,90,\fscx101.5\fscy101.5)\t(90,140,\fscx100\fscy100)}"
    )
    closing = r"{\c&H00FFFFFF&\fscx100\fscy100}"
    assert event_texts == [
        "רוצה בעל.\u200f " + opening + "אני" + closing,
        "בעל.\u200f " + opening + "רוצה" + closing + " אני",
        opening + "בעל.\u200f" + closing + " אני רוצה",
    ]


@pytest.mark.parametrize(
    ("caption_style", "cue_style"),
    [
        ("box", "sentences"),
        ("card", "sentences"),
        ("capcut", "sentences"),
        ("broadcast", "sentences"),
        ("box", "reveal"),
    ],
)
def test_trailing_punctuation_stays_inside_its_style_run(
    tmp_path: Path, caption_style: str, cue_style: str
) -> None:
    destination = tmp_path / f"{caption_style}-{cue_style}.ass"
    write_ass(
        [Cue(0, 1, "source", "אחד שלך. הצרכים, אחרון")],
        destination,
        720,
        1280,
        caption_style=caption_style,
        cue_style=cue_style,
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    event_texts = [dialogue.rsplit(r"{\q2}", 1)[1] for dialogue in dialogues]
    assert any("שלך.\u200f" in text for text in event_texts)
    assert any("הצרכים,\u200f" in text for text in event_texts)
    assert all("שלך." not in text or "שלך.\u200f" in text for text in event_texts)
    assert all("הצרכים," not in text or "הצרכים,\u200f" in text for text in event_texts)
    assert not any(re.search(r"[.,:;!?]\{\\", text) for text in event_texts)
    assert not any("\u200f\u200f" in text for text in event_texts)


def test_rtl_highlight_segments_keep_words_logical_within_each_run() -> None:
    rendered = ["כנראה", "אני", r"{\c&H0000FFFF&}לא{\c&H00FFFFFF&}", "רוצה", "בעל"]

    assert " ".join(_rtl_highlight_segments(rendered, 2)) == (
        r"רוצה בעל {\c&H0000FFFF&}לא{\c&H00FFFFFF&} כנראה אני"
    )


def test_rtl_highlight_segments_preserve_every_logical_word_at_each_position() -> None:
    words = ["אחד", "שניים", "שלושה", "ארבעה"]
    for active_index in range(len(words)):
        rendered = words.copy()
        rendered[active_index] = rf"{{\c&H0000FFFF&}}{words[active_index]}{{\c&H00FFFFFF&}}"
        visual = " ".join(_rtl_highlight_segments(rendered, active_index))
        plain = re.sub(r"\{[^}]+\}", "", visual).split()
        assert sorted(plain) == sorted(words)


def test_capcut_ltr_events_keep_logical_token_order(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"
    write_ass(
        [Cue(0, 1, "one two three", "one two three")],
        destination,
        720,
        1280,
        caption_style="capcut",
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert all(_plain_event_tokens(event) == ["one", "two", "three"] for event in dialogues)


def test_capcut_mixed_rtl_events_preserve_latin_isolation(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"
    write_ass(
        [Cue(0, 1, "hello world good", "שלום world טוב")],
        destination,
        720,
        1280,
        caption_style="capcut",
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert [_plain_event_tokens(event) for event in dialogues] == [
        ["world", "טוב", "שלום"],
        ["טוב", "world", "שלום"],
        ["טוב", "שלום", "world"],
    ]
    assert all(f"{LRI}world{PDI}" in event for event in dialogues)


def test_capcut_rtl_reordering_keeps_each_word_and_overrides(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"
    words = ["ראשון", "שני", "שלישי"]
    write_ass(
        [Cue(0, 1, "one two three", " ".join(words))],
        destination,
        720,
        1280,
        caption_style="capcut",
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    for event in dialogues:
        assert sorted(_plain_event_tokens(event)) == sorted(words)
        assert event.count(r"\fscx97") == 1
        assert event.count(r"\c&H0000FFFF&") == 1
        assert event.count(r"\c&H00FFFFFF&") == 1


def test_capcut_highlight_follows_uneven_asr_word_starts(tmp_path: Path) -> None:
    cue = Cue(
        1,
        2.2,
        "one two three",
        "אחד שני שלושה",
        words=[Word(1, 1.1, "one"), Word(1.1, 2, "two"), Word(2, 2.2, "three")],
    )
    destination = tmp_path / "capcut.ass"

    write_ass([cue], destination, 720, 1280, caption_style="capcut")

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    windows = _dialogue_windows(dialogues)
    assert [start for start, _ in windows] == [100, 110, 200]
    assert all(start < end for start, end in windows)
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))
    assert all(100 <= start < end <= 220 for start, end in windows)


def test_capcut_without_asr_words_keeps_character_weighted_windows(tmp_path: Path) -> None:
    destination = tmp_path / "capcut.ass"
    write_ass(
        [Cue(0, 1.2, "one two three", "א בbbb גג")],
        destination,
        720,
        1280,
        caption_style="capcut",
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    assert _dialogue_windows(dialogues) == [(0, 17), (17, 86), (86, 120)]


def test_capcut_with_fewer_asr_words_still_emits_every_display_word(tmp_path: Path) -> None:
    cue = Cue(
        0,
        1,
        "one two",
        "אחד שני שלושה ארבעה",
        words=[Word(0, 0.2, "one"), Word(0.2, 1, "two")],
    )
    destination = tmp_path / "capcut.ass"

    write_ass([cue], destination, 720, 1280, caption_style="capcut")

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    windows = _dialogue_windows(dialogues)
    assert len(windows) == 4
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))


def test_capcut_and_broadcast_use_the_same_asr_windows(tmp_path: Path) -> None:
    cue = Cue(
        0,
        1.2,
        "one two three",
        "אחד שני שלושה",
        words=[Word(0, 0.1, "one"), Word(0.1, 1, "two"), Word(1, 1.2, "three")],
    )
    windows_by_style = {}
    for caption_style in ("capcut", "broadcast"):
        destination = tmp_path / f"{caption_style}.ass"
        write_ass([cue], destination, 720, 1280, caption_style=caption_style)
        _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
        windows_by_style[caption_style] = _dialogue_windows(dialogues)

    assert windows_by_style["capcut"] == windows_by_style["broadcast"]


@pytest.mark.parametrize(
    ("caption_pop", "entrance", "closing", "excluded"),
    [
        ("off", "", r"{\c&H00FFFFFF&}", (r"\t", r"\fscx", r"\alpha")),
        (
            "subtle",
            r"\fscx97\fscy97\t(0,90,\fscx101.5\fscy101.5)\t(90,140,\fscx100\fscy100)",
            r"{\c&H00FFFFFF&\fscx100\fscy100}",
            (),
        ),
        (
            "fade",
            r"\alpha&HFF&\t(0,90,\alpha&H00&)",
            r"{\c&H00FFFFFF&\alpha&H00&}",
            (r"\fscx",),
        ),
        (
            "normal",
            r"\fscx80\fscy80\t(0,80,\fscx108\fscy108)\t(80,150,\fscx100\fscy100)",
            r"{\c&H00FFFFFF&\fscx100\fscy100}",
            (),
        ),
    ],
)
def test_capcut_caption_pop_modes(
    tmp_path: Path,
    caption_pop: str,
    entrance: str,
    closing: str,
    excluded: tuple[str, ...],
) -> None:
    destination = tmp_path / f"{caption_pop}.ass"
    write_ass(
        [Cue(0, 1, "one two", "אחד שני")],
        destination,
        720,
        1280,
        caption_style="capcut",
        caption_pop=caption_pop,
    )

    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    highlighted_opening = rf"{{\c&H0000FFFF&{entrance}}}"
    assert all(highlighted_opening in event for event in dialogues)
    assert all(closing in event for event in dialogues)
    assert not any(
        event.rsplit(",,", 1)[1].removeprefix(r"{\q2}").startswith(
            (r"{\fscx", r"{\alpha")
        )
        for event in dialogues
    )
    assert all(marker not in event for marker in excluded for event in dialogues)
    if caption_pop == "subtle":
        scales = [float(value) for event in dialogues for value in re.findall(r"\\fsc[xy]([\d.]+)", event)]
        assert min(scales) >= 96
        assert max(scales) <= 101.5


def test_capcut_default_pop_is_subtle_and_modes_keep_event_decomposition(
    tmp_path: Path,
) -> None:
    cue = Cue(
        1,
        2.2,
        "one two three four",
        "אחד שני three ארבעה",
        words=[
            Word(1, 1.1, "one"),
            Word(1.1, 1.8, "two"),
            Word(1.8, 2, "three"),
            Word(2, 2.2, "four"),
        ],
    )
    dialogues_by_mode = {}
    for caption_pop in (None, "off", "subtle", "fade", "normal"):
        label = caption_pop or "default"
        destination = tmp_path / f"{label}.ass"
        kwargs = {} if caption_pop is None else {"caption_pop": caption_pop}
        write_ass([cue], destination, 720, 1280, caption_style="capcut", **kwargs)
        _, dialogues_by_mode[label] = _style_and_dialogues(
            destination.read_text(encoding="utf-8")
        )

    assert dialogues_by_mode["default"] == dialogues_by_mode["subtle"]
    baseline = dialogues_by_mode["off"]
    for dialogues in dialogues_by_mode.values():
        assert len(dialogues) == len(baseline)
        assert _dialogue_windows(dialogues) == _dialogue_windows(baseline)
        assert [_plain_event_tokens(event) for event in dialogues] == [
            _plain_event_tokens(event) for event in baseline
        ]


def test_card_caption_style_has_one_fading_event_per_cue(tmp_path: Path) -> None:
    cues = [Cue(0, 1, "one", "שלום"), Cue(1, 2, "two", "עולם")]
    destination = tmp_path / "card.ass"
    write_ass(cues, destination, 720, 1280, caption_style="card")
    style, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))

    fields = style.split(",")
    assert len(dialogues) == len(cues)
    assert fields[15] == "3"
    assert fields[6] == "&H00000000"
    assert all(r"\fad(120,120)" in event for event in dialogues)
    assert not any(r"\c&H0000FFFF&" in event for event in dialogues)


def test_broadcast_caption_style_has_static_white_highlight(tmp_path: Path) -> None:
    destination = tmp_path / "broadcast.ass"
    write_ass([Cue(0, 1, "one two", "אחד שני")], destination, 720, 1280, caption_style="broadcast")
    style, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    fields = style.split(",")

    assert fields[3] == "&H00000000"
    assert fields[6] == "&H0000FFFF"
    assert all(r"\t(" not in event and r"\fad(" not in event for event in dialogues)


@pytest.mark.parametrize("caption_style", ["box", "capcut", "card", "broadcast"])
def test_caption_styles_keep_q2_and_latin_isolation(tmp_path: Path, caption_style: str) -> None:
    destination = tmp_path / f"{caption_style}.ass"
    write_ass([Cue(0, 1, "source", "שלום English עולם")], destination, 720, 1280, caption_style=caption_style)
    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))

    assert all(r"{\q2}" in event for event in dialogues)
    assert all(LRI in event and PDI in event for event in dialogues)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "הגרסה נקראת 'Horizon Minecraft Java 26.3'.",
            "הגרסה נקראת \u2066Horizon Minecraft Java 26.3\u2069.\u200f",
        ),
        ("Mojang's code.", "\u2066Mojangs code\u2069.\u200f"),
        ("שוחרר ב-2026.", "שוחרר ב-2026.\u200f"),
        ("(GPU) ו-VRAM", "(\u2066GPU\u2069) ו-\u2066VRAM\u2069\u200f"),
        ("עברית בלבד", "עברית בלבד\u200f"),
    ],
)
def test_ass_text_isolates_latin_runs_and_marks_rtl_lines(
    text: str, expected: str
) -> None:
    assert _ass_text(text, max_chars=100) == expected


def test_ass_text_uses_lri_only_for_latin_runs() -> None:
    rendered = _ass_text(
        "נקרא `Horizon Minecraft Java 26.3` ועוד 2026-10-30 שוב",
        max_chars=100,
    )

    assert "\u2067" not in rendered
    assert rendered == (
        "נקרא \u2066Horizon Minecraft Java 26.3\u2069 ועוד 2026-10-30 שוב\u200f"
    )
    assert rendered.count("\u200f") == 1


def test_ass_text_removes_quotes_in_every_position() -> None:
    rendered = _ass_text(
        f"{QUOTE_CHARACTERS}תחילה א{QUOTE_CHARACTERS}מצע סוף{QUOTE_CHARACTERS}",
        max_chars=100,
    )

    assert not set(QUOTE_CHARACTERS) & set(rendered)
    assert rendered == "תחילה אמצע סוף\u200f"


def test_write_ass_never_emits_ascii_quotes(tmp_path: Path) -> None:
    destination = tmp_path / "quotes.ass"
    write_ass([Cue(0, 1, "source", "מוצר 'quoted' ו־\"double\"")], destination, 720, 1280)

    dialogue = destination.read_text(encoding="utf-8").split("[Events]\n", 1)[1]
    assert not {"'", '"', "`"} & set(dialogue)


def test_wrapped_ass_text_marks_both_lines_as_rtl() -> None:
    rendered = _ass_text("עברית אחת English words עברית שנייה", max_chars=18)

    assert rendered.count(r"\N") == 1
    assert all(line.endswith("\u200f") for line in rendered.split(r"\N"))


def test_ass_text_repairs_and_keeps_split_decimal_atomic() -> None:
    rendered = _ass_text("ללא שינוי של 26'. .3, בנויה מחדש", max_chars=16)

    assert "26.3" in rendered
    assert "26\\N.3" not in rendered


def test_three_line_fallback_switches_to_smaller_font_and_keeps_every_word() -> None:
    words = [f"word{i}" for i in range(9)]
    text = " ".join(words)

    rendered = _ass_text(text, max_chars=17, font_size=40)

    assert "\\fs36" in rendered
    assert rendered.count(r"\N") == 2
    for word in words:
        assert word in rendered


def test_two_line_text_keeps_the_base_font_size() -> None:
    rendered = _ass_text("word0 word1 word2 word3", max_chars=11, font_size=40)

    assert "\\fs" not in rendered


def test_oversized_token_stays_on_its_own_line() -> None:
    token = "Horizon-Minecraft-Java-26.3"
    rendered = _ass_text(f"לפני {token} אחרי", max_chars=10)

    assert rendered.split(r"\N") == [
        "לפני\u200f",
        f"\u2066{token}\u2069\u200f",
        "אחרי\u200f",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "ללא שינוי של '26.3', בנויה מחדש",
        "הגרסה 26.3 נבנתה מחדש",
        "המשחק ב-Horizon Minecraft Java 26.3'",
    ],
)
def test_reported_numeric_bidi_regressions(text: str) -> None:
    rendered = _ass_text(text, max_chars=18)
    plain = rendered.replace("\u2066", "").replace("\u2069", "").replace("\u200f", "")

    assert "26.3" in rendered
    assert "26\\N.3" not in rendered
    assert not set(QUOTE_CHARACTERS) & set(rendered)
    assert "\u200e" not in rendered
    lines = rendered.split(r"\N")
    assert rendered.count(RLM) == len(lines) + sum(line.startswith(RLM) for line in lines)
    assert rendered.count("\u2066") == rendered.count("\u2069")
    assert plain.count("26.3") == 1
    for isolated in re.findall("\u2066(.*?)\u2069", rendered):
        assert re.search(r"[A-Za-z]", isolated)


def test_long_text_wraps_without_reordering_hebrew(tmp_path: Path) -> None:
    text = " ".join(["מילה"] * 20)
    destination = tmp_path / "cues.ass"
    write_ass([Cue(0, 5, text, text)], destination, 1920, 1080)

    dialogue = next(
        line for line in destination.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")
    )
    assert r"\N" in dialogue
    assert dialogue.count("מילה") == 20


def test_timestamp_rolls_over_at_hour_boundary() -> None:
    assert format_timestamp(3661.999) == "1:01:02.00"


def test_vertical_frame_geometry_scales_font_and_line_budget() -> None:
    font_size, margin, max_chars = subtitle_geometry(576, 1024)

    assert font_size == 55
    assert margin == 32
    assert max_chars == 17


def test_font_size_is_clamped_for_extreme_frame_heights() -> None:
    assert subtitle_geometry(320, 480)[0] == 34
    assert subtitle_geometry(3840, 2160)[0] == 77


def test_zero_box_alpha_uses_configured_outline_style(tmp_path: Path) -> None:
    destination = tmp_path / "outlined.ass"

    write_ass(
        [Cue(0, 1, "source", "עברית")],
        destination,
        576,
        1024,
        box_alpha=0,
        outline=4,
    )

    style = next(
        line for line in destination.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style:")
    )
    assert ",1,4,0,2," in style
    assert "&H80000000" in style


def test_vertical_ass_has_clear_margins_on_black_frame(tmp_path: Path) -> None:
    destination = tmp_path / "vertical.ass"
    text = " ".join(["מילה"] * 8)
    write_ass([Cue(0.0, 1.0, text, text)], destination, 576, 1024)

    margins = render_black_frame_margins(destination, 576, 1024)

    assert all(margin >= 24 for margin in margins.values()), margins


@pytest.mark.parametrize("caption_style", ["capcut", "broadcast"])
@pytest.mark.parametrize("text", ["עובדים 1,500", "בקרב 1,500 עובדים,"])
def test_number_led_highlight_run_starts_with_one_rlm(
    tmp_path: Path, caption_style: str, text: str
) -> None:
    destination = tmp_path / "number.ass"
    write_ass(
        [Cue(0, 1, "source", text)], destination, 720, 1280,
        caption_style=caption_style,
    )
    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    event = dialogues[0]
    body = re.sub(r"\{[^}]+\}", "", event.split(",", 9)[9])

    assert body.startswith(RLM + "1,500")
    assert len(body) - len(body.lstrip(RLM)) == 1
    assert body.endswith(RLM)
    expected = "1,500 עובדים" if text == "עובדים 1,500" else "1,500 עובדים," + RLM + " בקרב"
    assert body == RLM + expected + RLM


@pytest.mark.parametrize(
    ("caption_style", "cue_style"),
    [("box", "sentences"), ("box", "words"), ("card", "sentences")],
)
def test_number_led_single_run_starts_with_one_rlm(
    tmp_path: Path, caption_style: str, cue_style: str
) -> None:
    destination = tmp_path / "number.ass"
    write_ass(
        [Cue(0, 1, "source", "1,500 עובדים בקרב")], destination, 720, 1280,
        caption_style=caption_style, cue_style=cue_style,
    )
    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    body = re.sub(r"\{[^}]+\}", "", dialogues[0].split(",", 9)[9])

    assert body == RLM + "1,500 עובדים בקרב" + RLM


@pytest.mark.parametrize("render", [_render_token, _bidi_line])
@pytest.mark.parametrize("first", ["1,500", "Latin", "$", "%", "("])
def test_non_hebrew_start_gets_one_rlm_before_isolation(render, first: str) -> None:
    kwargs = {"rtl": True} if render is _render_token else {}
    rendered = render(first + " עובדים", **kwargs)
    assert rendered.startswith(RLM)
    assert not rendered.startswith(RLM * 2)
    if first == "Latin":
        assert rendered.startswith(RLM + LRI + "Latin")
    assert render(RLM + first + " עובדים", **kwargs) == rendered


@pytest.mark.parametrize("render", [_render_token, _bidi_line])
def test_hebrew_led_run_with_internal_number_keeps_rlm_count(render) -> None:
    kwargs = {"rtl": True} if render is _render_token else {}
    before = render("בקרב הרבה עובדים,", **kwargs)
    after = render("בקרב 1,500 עובדים,", **kwargs)

    assert before.count(RLM) == after.count(RLM) == 1
    assert after == "בקרב 1,500 עובדים," + RLM


@pytest.mark.parametrize("caption_style", ["box", "capcut", "card", "broadcast"])
def test_latin_cue_has_no_leading_rlm(tmp_path: Path, caption_style: str) -> None:
    destination = tmp_path / "latin.ass"
    write_ass(
        [Cue(0, 1, "this test", "this test")], destination, 720, 1280,
        caption_style=caption_style, caption_pop="off",
    )
    _, dialogues = _style_and_dialogues(destination.read_text(encoding="utf-8"))
    bodies = [event.split(",", 9)[9] for event in dialogues]
    for body in bodies:
        plain = re.sub(r"\{[^}]+\}", "", body)
        assert not any(token.startswith(RLM) for token in plain.split())
        assert plain.replace(LRI, "").replace(PDI, "").replace(RLM, "") == "this test"
    if caption_style == "capcut":
        assert bodies == [
            r"{\q2}{\c&H0000FFFF&}" + LRI + "this" + PDI
            + r"{\c&H00FFFFFF&} " + LRI + "test" + PDI + RLM,
            r"{\q2}" + LRI + "this" + PDI + r" {\c&H0000FFFF&}"
            + LRI + "test" + PDI + r"{\c&H00FFFFFF&}" + RLM,
        ]
