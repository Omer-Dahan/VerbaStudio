from pathlib import Path
from unittest.mock import Mock, patch

from hs.models import Cue
from hs.translate import (
    PROMPT_TEMPLATE,
    TranslationClient,
    apply_glossary,
    build_prompt,
    is_sane_translation,
    load_glossary,
    server_is_healthy,
    should_passthrough,
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


def test_atomic_cues_pass_through_without_translation() -> None:
    request = Mock()
    cues = [Cue(0.0, 1.0, "Minecraft"), Cue(1.0, 2.0, "26.3")]

    TranslationClient(request=request).translate_cues(cues, "en")

    request.assert_not_called()
    assert [cue.text for cue in cues] == ["Minecraft", "26.3"]
    assert all(cue.translation_passthrough for cue in cues)
    assert not any(cue.translation_failed for cue in cues)


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


def test_glossary_is_deterministic_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text('"ניו יורק": "נ״י"\n"יורק": "יורק סיטי"\n', encoding="utf-8")

    glossary = load_glossary(path)
    assert apply_glossary("טיסה לניו יורק", glossary) == "טיסה לנ״י"


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
