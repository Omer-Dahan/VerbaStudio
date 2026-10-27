import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import yaml

from hs.models import Cue


DEFAULT_BASE_URL = "http://127.0.0.1:8189"
HEBREW = re.compile(r"[\u0590-\u05ff]")
LATIN_SENTENCE = re.compile(r"\b[A-Za-z]+(?:[ ,'-]+[A-Za-z]+){3,}[.!?]")
PROMPT_ECHOES = (
    "מתרגם מקצועי",
    "אנא תרגמו",
    "ללא הסברים",
    "you are a professional",
)
PROMPT_TEMPLATE = """You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following {source_lang} text into {target_lang}:

{text}"""


class TranslationClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        mode: str = "chat",
        timeout: float = 60.0,
        request: Callable[[str, dict, float], dict] | None = None,
    ) -> None:
        if mode not in {"chat", "completion", "api"}:
            raise ValueError(f"Unsupported translation mode: {mode}")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self._request_override = request

    def translate_cues(
        self,
        cues: list[Cue],
        source_code: str,
        glossary_path: Path | None = None,
    ) -> list[Cue]:
        glossary = load_glossary(glossary_path) if glossary_path else {}
        for cue in cues:
            translation = self.translate_text(cue.source, source_code)
            if translation is None:
                cue.text = cue.source
                cue.translation_failed = True
            else:
                cue.text = apply_glossary(translation, glossary)
        return cues

    def translate_text(self, source: str, source_code: str) -> str | None:
        source_lang = _language_name(source_code)
        attempts = ((True, 0.0), (False, 0.0), (True, 0.3))
        for quoted, temperature in attempts:
            prompt = build_prompt(source, source_lang, source_code, quoted=quoted)
            try:
                output = self._complete(prompt, source, temperature).strip()
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if is_sane_translation(source, output):
                return output.strip('"').strip()
        return None

    def _complete(self, prompt: str, source: str, temperature: float) -> str:
        max_tokens = min(400, max(1, 3 * len(source.split())))
        if self.mode == "completion":
            endpoint = f"{self.base_url}/v1/completions"
            wrapped = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            payload = {
                "prompt": wrapped,
                "temperature": temperature,
                "top_p": 1,
                "max_tokens": max_tokens,
                "stop": ["<end_of_turn>"],
            }
        else:
            endpoint = f"{self.base_url}/v1/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": 1,
                "max_tokens": max_tokens,
                "stop": ["<end_of_turn>"],
            }

        response = (
            self._request_override(endpoint, payload, self.timeout)
            if self._request_override
            else _post_json(endpoint, payload, self.timeout)
        )
        choice = response["choices"][0]
        if self.mode == "completion":
            return choice["text"]
        return choice["message"]["content"]


def build_prompt(
    source: str,
    source_lang: str,
    source_code: str,
    quoted: bool = True,
) -> str:
    text = f'"{source}"' if quoted else source
    return PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_code=source_code,
        target_lang="Hebrew",
        target_code="he-IL",
        text=text,
    )


def is_sane_translation(source: str, translation: str) -> bool:
    cleaned = translation.strip().strip('"').strip()
    if not cleaned or not HEBREW.search(cleaned):
        return False
    if any(marker in cleaned.casefold() for marker in PROMPT_ECHOES):
        return False
    ratio = len(cleaned) / max(1, len(source.strip()))
    if not 0.5 <= ratio <= 1.8 or LATIN_SENTENCE.search(cleaned):
        return False
    return not _is_repeated(cleaned)


def load_glossary(path: Path) -> dict[str, str]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
    ):
        raise ValueError("Glossary must be a YAML mapping of text replacements")
    return loaded


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    for original in sorted(glossary, key=len, reverse=True):
        text = text.replace(original, glossary[original])
    return text


def _language_name(code: str) -> str:
    try:
        return {"en": "English", "he": "Hebrew"}[code]
    except KeyError as error:
        raise ValueError(f"Unsupported source language: {code}") from error


def _is_repeated(text: str) -> bool:
    words = text.split()
    if len(words) < 4:
        return False
    midpoint = len(words) // 2
    return len(words) % 2 == 0 and words[:midpoint] == words[midpoint:]


def _post_json(endpoint: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

