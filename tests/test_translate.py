from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Thread
import time
from unittest.mock import Mock, patch

import pytest

from hs.models import Cue
from hs.translate import (
    PROMPT_TEMPLATE,
    TranslationClient,
    _language_name,
    apply_glossary,
    build_prompt,
    is_sane_translation,
    load_glossary,
    server_is_healthy,
    should_passthrough,
    split_translation,
)


def test_official_prompt_has_quoted_text_after_two_newlines() -> None:
    prompt = build_prompt("How are you?", "English", "en")

    assert prompt == PROMPT_TEMPLATE.format(
        source_lang="English",
        source_code="en",
        target_lang="Hebrew",
        target_code="he-IL",
        text='"How are you?"',
    )
    assert prompt.endswith('\n\n"How are you?"')


def test_arabic_source_language_name_and_prompt() -> None:
    language = _language_name("ar")

    assert language == "Arabic"
    prompt = build_prompt("مرحبا", language, "ar")
    assert "Arabic (ar) to Hebrew (he-IL)" in prompt
    assert prompt.endswith('"مرحبا"')


def test_unknown_source_language_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported source language: xx"):
        _language_name("xx")


def test_hebrew_clause_split_preserves_every_character_and_punctuation() -> None:
    sentence = "זה חלק ראשון, וזה החלק השני; ולבסוף מגיע החלק השלישי."

    pieces = split_translation(sentence, 3)

    assert pieces is not None
    assert " ".join(pieces) == sentence
    assert pieces[0].endswith(",")
    assert pieces[1].endswith(";")


def test_hebrew_clause_split_uses_nearest_proportional_space() -> None:
    sentence = "אחד שני שלושה ארבעה חמישה שישה שבעה שמונה"

    pieces = split_translation(sentence, 2)

    assert pieces == ["אחד שני שלושה ארבעה", "חמישה שישה שבעה שמונה"]


def test_hebrew_clause_split_avoids_weak_end_tokens_when_possible() -> None:
    sentence = "למה לעזאזל בכלל התחלתי לצאת שוב עם האדם הזה היום"

    pieces = split_translation(sentence, 3)

    assert pieces is not None
    assert " ".join(pieces) == sentence
    assert all(piece.split()[-1] not in {"אל", "על", "עם", "של", "את", "לי", "לו", "לה", "מן", "כי", "אם", "אשר", "כמו", "או", "גם", "רק", "לא"} for piece in pieces[:-1])


def test_hebrew_clause_split_avoids_separate_prefix_at_piece_start() -> None:
    sentence = "זה משפט ארוך ב מקרה שבו קיימת חלופה טובה יותר"

    pieces = split_translation(sentence, 2)

    assert pieces is not None
    assert pieces[1].split()[0] not in {"ב", "ל", "מ", "ה", "ו", "כ", "ש"}
    assert " ".join(pieces) == sentence


def test_hebrew_clause_split_returns_none_when_minimum_words_are_impossible() -> None:
    assert split_translation("מעט מילים", 2) is None


def test_weighted_split_places_boundaries_near_weight_proportional_positions() -> None:
    sentence = " ".join(f"מ{index}" for index in range(1, 31))

    pieces = split_translation(sentence, 3, weights=[1.0, 2.0, 3.0])

    assert pieces is not None
    assert " ".join(pieces) == sentence
    assert len(pieces[0].split()) == 5
    assert len(pieces[1].split()) == 10
    assert len(pieces[2].split()) == 15


def test_weighted_split_does_not_overload_the_small_weight_piece() -> None:
    sentence = " ".join(f"מ{index}" for index in range(1, 21))

    pieces = split_translation(sentence, 2, weights=[0.1, 0.9])

    assert pieces is not None
    assert " ".join(pieces) == sentence
    assert len(pieces[0].split()) < 7


def test_distant_hebrew_comma_is_ignored_outside_the_tolerance_window() -> None:
    words = ["מ1", "מ2", "מ3,", *(f"מ{index}" for index in range(4, 21))]
    sentence = " ".join(words)

    pieces = split_translation(sentence, 2)

    assert pieces is not None
    assert " ".join(pieces) == sentence
    assert not pieces[0].endswith(",")


def test_weighted_split_join_invariant_holds_with_uneven_weights() -> None:
    sentence = " ".join(f"מ{index}" for index in range(1, 23))

    pieces = split_translation(sentence, 4, weights=[0.3, 1.1, 0.4, 2.2])

    assert pieces is not None
    assert len(pieces) == 4
    assert " ".join(pieces) == sentence


def test_prompt_template_instructs_to_keep_ellipses() -> None:
    assert "ellipsis" in PROMPT_TEMPLATE
    assert "..." in PROMPT_TEMPLATE


def test_ellipsis_survives_normalize_and_passthrough() -> None:
    from hs.text import normalize_subtitle_text

    source = "רגע, חכה…"
    assert not should_passthrough(source)
    assert normalize_subtitle_text(source) == source
    assert is_sane_translation(source, source)


def test_client_retries_prompt_echo_and_accepts_hebrew() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "אתה מתרגם מקצועי ללא הסברים"}}]},
            {"choices": [{"message": {"content": "מה שלומך היום?"}}]},
        ]
    )
    calls = []

    def request(endpoint, payload, timeout):
        calls.append((endpoint, payload, timeout))
        return next(responses)

    client = TranslationClient(request=request)
    assert client.translate_text("How are you today?", "en") == "מה שלומך היום?"
    assert len(calls) == 2
    assert calls[0][1]["temperature"] == 0.0
    assert calls[0][1]["messages"][0]["content"] == '"How are you today?"'
    assert calls[1][1]["messages"][0]["content"] == "How are you today?"
    assert calls[0][1]["chat_template_kwargs"] == {
        "source_lang_code": "en",
        "target_lang_code": "he-IL",
    }


def test_failed_translation_preserves_source_and_marks_cue() -> None:
    def request(endpoint, payload, timeout):
        return {"choices": [{"message": {"content": "English only."}}]}

    cue = Cue(0.0, 2.0, "English only.")
    TranslationClient(request=request).translate_cues([cue], "en")

    assert cue.text == cue.source
    assert cue.translation_failed


def test_rejected_fragment_retries_with_neighbor_context() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "your old Java mods"}}]},
            {"choices": [{"message": {"content": "your old Java mods"}}]},
            {"choices": [{"message": {"content": "your old Java mods"}}]},
            {"choices": [{"message": {"content": "your old Java mods"}}]},
            {"choices": [{"message": {"content": "הסרנו אותם אתמול"}}]},
            {"choices": [{"message": {"content": "המודים הישנים שלכם ל-Java"}}]},
        ]
    )
    calls = []

    def request(_endpoint, payload, _timeout):
        calls.append(payload)
        return next(responses)

    fragment = Cue(0.0, 1.0, "your old Java mods")
    neighbor = Cue(1.0, 3.0, "We removed them yesterday.")

    TranslationClient(request=request).translate_cues([fragment, neighbor], "en")

    assert fragment.text == "המודים הישנים שלכם ל-Java"
    assert fragment.translation_outcome == "translated"
    assert not fragment.translation_failed
    assert calls[-1]["messages"][:2] == [
        {"role": "user", "content": "We removed them yesterday."},
        {"role": "assistant", "content": "הסרנו אותם אתמול"},
    ]


def test_atomic_cues_pass_through_without_translation() -> None:
    request = Mock()
    cues = [Cue(0.0, 1.0, "Minecraft"), Cue(1.0, 2.0, "26.3")]

    TranslationClient(request=request).translate_cues(cues, "en")

    request.assert_not_called()
    assert [cue.text for cue in cues] == ["Minecraft", "26.3"]
    assert all(cue.translation_passthrough for cue in cues)
    assert not any(cue.translation_failed for cue in cues)
    assert all(cue.translation_outcome == "passthrough" for cue in cues)


def test_stub_server_retries_503_then_translates() -> None:
    statuses = iter([503, 200])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status = next(statuses)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if status == 200:
                body = {"choices": [{"message": {"content": "שלום לכולם"}}]}
                self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        cue = Cue(0.0, 1.0, "Hello everyone")
        client = TranslationClient(
            f"http://127.0.0.1:{server.server_port}", sleep=lambda _delay: None
        )
        client.translate_cues([cue], "en")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert cue.text == "שלום לכולם"
    assert cue.translation_outcome == "translated"
    assert not cue.translation_failed
    assert cue.translation_retried
    assert cue.translation_attempts == 2


def test_stub_server_503_forever_fails_with_url() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        cue = Cue(0.0, 1.0, "Hello everyone")
        client = TranslationClient(base_url, sleep=lambda _delay: None, recovery_seconds=0)
        with pytest.raises(RuntimeError, match=base_url):
            client.translate_cues([cue], "en")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert cue.translation_outcome == "failed"
    assert cue.translation_attempts == 10


def test_translation_survives_server_reload_after_three_seconds() -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"choices": [{"message": {"content": "שלום לכולם"}}]}
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            pass

    server: ThreadingHTTPServer | None = None

    def start_after_reload() -> None:
        nonlocal server
        time.sleep(3.0)
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.serve_forever()

    thread = Thread(target=start_after_reload)
    thread.start()
    try:
        cue = Cue(0.0, 1.0, "Hello everyone")
        client = TranslationClient(
            f"http://127.0.0.1:{port}", jitter=lambda: 0.0, recovery_seconds=10.0
        )
        translated = client.translate_cues([cue], "en")
    finally:
        if server is not None:
            server.shutdown()
        thread.join()
        if server is not None:
            server.server_close()

    assert translated[0].text == "שלום לכולם"
    assert translated[0].translation_failed is False


def test_mixed_sentence_is_translated() -> None:
    request = Mock(
        return_value={"choices": [{"message": {"content": "שחקו Minecraft עכשיו"}}]}
    )
    cue = Cue(0.0, 2.0, "Play Minecraft now")

    TranslationClient(request=request).translate_cues([cue], "en")

    request.assert_called_once()
    assert cue.text == "שחקו Minecraft עכשיו"
    assert not cue.translation_passthrough


def test_passthrough_classification_covers_atomic_content() -> None:
    assert should_passthrough("Minecraft")
    assert should_passthrough("26.3")
    assert should_passthrough("https://example.test/path")
    assert should_passthrough("?!")
    assert not should_passthrough("Play Minecraft now")


def test_sanity_checks_reject_echo_latin_sentence_and_repetition() -> None:
    assert is_sane_translation("Good morning", "בוקר טוב")
    assert not is_sane_translation("Translate this", "אנא תרגמו את הטקסט")
    assert not is_sane_translation("A normal source sentence", "זה תרגום but this sentence stayed here.")
    assert not is_sane_translation("One two three four", "אחד שתיים אחד שתיים")


def test_sanity_checks_allow_wider_ratio_and_source_latin_names() -> None:
    assert is_sane_translation("Use Rust for this service", "השתמשו ב-Rust עבור השירות הזה בבקשה")
    assert is_sane_translation("A longer source text", "טקסט קצר")
    assert not is_sane_translation("Use this tool", "השתמשו ב-Rust")


def test_glossary_is_deterministic_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text('"ניו יורק": "נ״י"\n"יורק": "יורק סיטי"\n', encoding="utf-8")

    glossary = load_glossary(path)
    assert apply_glossary("טיסה לניו יורק", glossary) == "טיסה לנ״י"


def test_default_glossary_preserves_rust_and_minecraft() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "כתבו את השירות ב-Rust."}}]},
            {"choices": [{"message": {"content": "שחקו Minecraft יחד."}}]},
        ]
    )
    client = TranslationClient(request=lambda *_args: next(responses))
    cues = [
        Cue(0.0, 2.0, "Write the service in Rust."),
        Cue(2.0, 4.0, "Play Minecraft together."),
    ]

    client.translate_cues(cues, "en")

    assert "Rust" in cues[0].display_text
    assert "חלודה" not in cues[0].display_text
    assert "Minecraft" in cues[1].display_text


def test_missing_presubstituted_term_retries_once_without_substitution() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "השתמשו בממשק"}}]},
            {"choices": [{"message": {"content": "השתמשו בממשק"}}]},
            {"choices": [{"message": {"content": "השתמשו בממשק"}}]},
            {"choices": [{"message": {"content": "השתמשו בממשק"}}]},
        ]
    )
    calls = []

    def request(_endpoint, payload, _timeout):
        calls.append(payload)
        return next(responses)

    output = TranslationClient(request=request).translate_text(
        "Use the interface", "en", glossary={"interface": "API"}
    )

    assert output == "השתמשו בממשק"
    assert len(calls) == 4
    assert calls[-1]["messages"][-1]["content"] == '"Use the interface"'


def test_identity_glossary_term_also_gets_unprotected_retry() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "כתבו את השירות"}}]},
            {"choices": [{"message": {"content": "כתבו את השירות"}}]},
            {"choices": [{"message": {"content": "כתבו את השירות"}}]},
            {"choices": [{"message": {"content": "כתבו את השירות"}}]},
        ]
    )

    output = TranslationClient(request=lambda *_args: next(responses)).translate_text(
        "Write the service in Rust", "en", glossary={"Rust": "Rust"}
    )

    assert output == "כתבו את השירות"


def test_gender_hint_follows_source_and_never_appears_in_output() -> None:
    calls = []

    def request(endpoint, payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "מה שלומך?"}}]}

    client = TranslationClient(request=request)
    output = client.translate_text("How are you?", "en", gender="m")

    hint = "(The speaker is a man; use masculine forms.)"
    assert output == "מה שלומך?"
    assert hint not in output
    assert calls[0]["messages"][0]["content"] == f'"How are you?"\n\n{hint}'
    assert build_prompt("How are you?", "English", "en", gender="m").endswith(
        f'"How are you?"\n\n{hint}'
    )


def test_context_uses_separate_turns_and_is_capped_at_two_pairs() -> None:
    calls = []

    def request(endpoint, payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "שורה נוכחית"}}]}

    history = [("old", "ישן"), ("previous", "קודם"), ("recent", "אחרון")]
    client = TranslationClient(request=request)
    assert client.translate_text("current", "en", history=history) == "שורה נוכחית"

    assert calls[0]["messages"] == [
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "קודם"},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "אחרון"},
        {"role": "user", "content": '"current"'},
    ]


def test_server_health_requires_http_200() -> None:
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch("hs.translate.urllib.request.urlopen", return_value=response) as urlopen:
        assert server_is_healthy("http://127.0.0.1:8189/") is True

    assert urlopen.call_args.args[0].full_url == "http://127.0.0.1:8189/health"
