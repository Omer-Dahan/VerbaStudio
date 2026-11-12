import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import yaml

from hs.models import Cue


DEFAULT_BASE_URL = "http://127.0.0.1:8189"
DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.yml"
HEBREW = re.compile(r"[\u0590-\u05ff]")
LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9.+#'’-]*")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
URL = re.compile(r"(?:https?://|www\.)\S+$", re.IGNORECASE)
PROMPT_ECHOES = (
    "מתרגם מקצועי",
    "אנא תרגמו",
    "ללא הסברים",
    "you are a professional",
)
PROMPT_TEMPLATE = """You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following {source_lang} text into {target_lang}:
Write natural, warm spoken {target_lang} that a person would actually say out loud - not literal, not robotic. Keep every ellipsis (...) and every question or exclamation mark that appears in the source; never drop, shorten or merge them.

{text}"""
GENDER_HINTS = {
    "m": "(The speaker is a man; use masculine forms.)",
    "f": "(The speaker is a woman; use feminine forms.)",
}
RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)
RELOAD_RECOVERY_SECONDS = 180.0
HEALTH_POLL_SECONDS = 5.0
HEBREW_BOUNDARY = re.compile(r"[,;:?!\.…][\"'”’)]*$")
WEAK_END_TOKENS = {
    "אל", "על", "עם", "של", "את", "לי", "לו", "לה", "מן", "כי",
    "אם", "אשר", "כמו", "או", "גם", "רק", "לא",
}
SEPARATE_PREFIXES = {"ב", "ל", "מ", "ה", "ו", "כ", "ש"}


class PermanentTranslationError(RuntimeError):
    pass


PUNCTUATION_TOLERANCE_RATIO = 0.25


def split_translation(
    text: str, piece_count: int, weights: list[float] | None = None
) -> list[str] | None:
    normalized = " ".join(text.split())
    if normalized != text or piece_count < 1:
        return None
    if piece_count == 1:
        return [text]

    words = text.split(" ")
    if len(words) < piece_count + 1:
        return None

    if weights is None:
        weights = [1.0] * piece_count
    total_weight = sum(weights) or float(piece_count)
    tolerance = PUNCTUATION_TOLERANCE_RATIO * len(words)

    boundaries: list[int] = []
    previous = 0
    cumulative_weight = 0.0
    for piece_index in range(1, piece_count):
        pieces_left = piece_count - piece_index
        low = previous + 1
        high = len(words) - pieces_left
        cumulative_weight += weights[piece_index - 1]
        target = len(words) * cumulative_weight / total_weight
        punctuation = [
            index
            for index in range(low, high + 1)
            if HEBREW_BOUNDARY.search(words[index - 1])
            and abs(index - target) <= tolerance
        ]
        strong_spaces = [
            index
            for index in range(low, high + 1)
            if words[index - 1] not in WEAK_END_TOKENS
            and words[index] not in SEPARATE_PREFIXES
        ]
        candidates = punctuation or strong_spaces or list(range(low, high + 1))
        boundary = min(candidates, key=lambda index: abs(index - target))
        boundaries.append(boundary)
        previous = boundary

    pieces = []
    start = 0
    for end in [*boundaries, len(words)]:
        pieces.append(" ".join(words[start:end]))
        start = end
    assert " ".join(pieces) == text
    return pieces


class TranslationClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        mode: str = "chat",
        timeout: float = 60.0,
        request: Callable[[str, dict, float], dict] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        recovery_seconds: float = RELOAD_RECOVERY_SECONDS,
        health_check: Callable[[str], bool] | None = None,
    ) -> None:
        if mode not in {"chat", "completion", "api"}:
            raise ValueError(f"Unsupported translation mode: {mode}")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self._request_override = request
        self._sleep = sleep
        self._jitter = jitter
        self._monotonic = monotonic
        self._recovery_seconds = recovery_seconds
        self._health_check = health_check or server_is_healthy
        self._recovery_deadline: float | None = None
        self._request_attempts = 0
        self._transport_failures = 0

    def translate_cues(
        self,
        cues: list[Cue],
        source_code: str,
        glossary_path: Path | None = None,
        gender: str | None = None,
    ) -> list[Cue]:
        self._recovery_deadline = self._monotonic() + max(0.0, self._recovery_seconds)
        glossary = load_glossary(DEFAULT_GLOSSARY_PATH)
        if glossary_path:
            glossary.update(load_glossary(glossary_path))
        history = list(glossary.items())[-2:]
        pending: list[tuple[int, Cue]] = []
        attempted = 0
        transport_only_failures = 0
        for index, cue in enumerate(cues):
            if should_passthrough(cue.source):
                cue.text = cue.source
                cue.translation_passthrough = True
                cue.translation_outcome = "passthrough"
                continue
            attempted += 1
            history, transport_only = self._translate_cue(
                cue, source_code, gender, history, glossary
            )
            if cue.translation_failed:
                pending.append((index, cue))
                transport_only_failures += transport_only

        for index, cue in pending:
            cue.translation_failed = False
            retry_history = self._neighbor_history(cues, index, history)
            history, transport_only = self._translate_cue(
                cue, source_code, gender, retry_history, glossary
            )
            if not cue.translation_failed:
                transport_only_failures -= 1
            elif not transport_only:
                transport_only_failures -= 1

        if attempted and transport_only_failures == attempted:
            raise RuntimeError(
                f"Translation server {self.base_url} was unreachable for the full retry budget"
            )
        return cues

    @staticmethod
    def _neighbor_history(
        cues: list[Cue],
        index: int,
        fallback: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(cues):
                neighbor = cues[neighbor_index]
                if neighbor.translation_outcome == "translated":
                    return [(neighbor.source, neighbor.text)]
        return fallback

    def _translate_cue(
        self,
        cue: Cue,
        source_code: str,
        gender: str | None,
        history: list[tuple[str, str]],
        glossary: dict[str, str],
    ) -> tuple[list[tuple[str, str]], bool]:
        self._request_attempts = 0
        self._transport_failures = 0
        try:
            translation = self.translate_text(
                cue.source,
                source_code,
                gender=gender,
                history=history,
                glossary=glossary,
            )
        except PermanentTranslationError:
            translation = None
        except (urllib.error.URLError, TimeoutError, OSError):
            translation = None
        cue.translation_attempts += self._request_attempts
        cue.translation_retried |= self._request_attempts > 1
        transport_only = bool(self._request_attempts) and (
            self._transport_failures == self._request_attempts
        )
        if translation is None:
            cue.text = cue.source
            cue.translation_failed = True
            cue.translation_outcome = "failed"
            return history, transport_only

        cue.text = apply_glossary(translation, glossary)
        cue.translation_failed = False
        cue.translation_outcome = "translated"
        history.append((cue.source, cue.text))
        return history[-2:], False

    def translate_text(
        self,
        source: str,
        source_code: str,
        gender: str | None = None,
        history: list[tuple[str, str]] | None = None,
        glossary: dict[str, str] | None = None,
    ) -> str | None:
        source_lang = _language_name(source_code)
        glossary = glossary or {}
        substituted = apply_glossary(source, glossary)
        protected = {
            replacement
            for original, replacement in glossary.items()
            if original in source and replacement in substituted
        }
        attempts = ((True, 0.0), (False, 0.0), (True, 0.3))
        for quoted, temperature in attempts:
            prompt = build_prompt(
                substituted, source_lang, source_code, quoted=quoted, gender=gender
            )
            try:
                output = self._complete_with_retries(
                    prompt,
                    substituted,
                    source_code,
                    temperature,
                    quoted=quoted,
                    gender=gender,
                    history=(history or [])[-2:],
                ).strip()
            except PermanentTranslationError:
                raise
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
            if is_sane_translation(substituted, output) and all(
                token.casefold() in output.casefold() for token in protected
            ):
                return output.strip('"').strip()
        if protected:
            prompt = build_prompt(source, source_lang, source_code, gender=gender)
            try:
                output = self._complete_with_retries(
                    prompt,
                    source,
                    source_code,
                    0.0,
                    quoted=True,
                    gender=gender,
                    history=(history or [])[-2:],
                ).strip()
            except PermanentTranslationError:
                raise
            except (ValueError, KeyError, json.JSONDecodeError):
                return None
            if is_sane_translation(source, output):
                return output.strip('"').strip()
        return None

    def _complete_with_retries(self, *args, **kwargs) -> str:
        while True:
            for attempt in range(5):
                self._request_attempts += 1
                try:
                    return self._complete(*args, **kwargs)
                except urllib.error.HTTPError as error:
                    if 400 <= error.code < 500 and error.code != 429:
                        raise PermanentTranslationError(
                            f"Translation request was rejected with HTTP {error.code}"
                        ) from error
                    self._transport_failures += 1
                    last_error = error
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    self._transport_failures += 1
                    last_error = error
                if attempt < 4:
                    delay = RETRY_DELAYS[attempt]
                    self._sleep(delay + self._jitter() * delay * 0.25)
            if not self._wait_for_server_reload():
                raise last_error

    def _wait_for_server_reload(self) -> bool:
        if self._recovery_deadline is None:
            return False
        while True:
            remaining = self._recovery_deadline - self._monotonic()
            if remaining <= 0:
                return False
            if self._health_check(self.base_url):
                return True
            self._sleep(min(HEALTH_POLL_SECONDS, remaining))

    def _complete(
        self,
        prompt: str,
        source: str,
        source_code: str,
        temperature: float,
        quoted: bool,
        gender: str | None,
        history: list[tuple[str, str]],
    ) -> str:
        source_tokens = max(len(source.split()), math.ceil(len(source) / 4))
        max_tokens = min(400, max(1, 3 * source_tokens))
        if self.mode == "completion":
            endpoint = f"{self.base_url}/v1/completions"
            prior_turns = "".join(
                f"<start_of_turn>user\n{user}<end_of_turn>\n"
                f"<start_of_turn>model\n{assistant}<end_of_turn>\n"
                for user, assistant in history
            )
            wrapped = (
                f"{prior_turns}<start_of_turn>user\n{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            payload = {
                "prompt": wrapped,
                "temperature": temperature,
                "top_p": 1,
                "max_tokens": max_tokens,
                "stop": ["<end_of_turn>"],
            }
        elif self.mode == "chat":
            endpoint = f"{self.base_url}/v1/chat/completions"
            content = f'"{source}"' if quoted else source
            if gender:
                content += f"\n\n{GENDER_HINTS[gender]}"
            messages = _history_messages(history)
            messages.append({"role": "user", "content": content})
            payload = {
                "messages": messages,
                "chat_template_kwargs": {
                    "source_lang_code": source_code,
                    "target_lang_code": "he-IL",
                },
                "temperature": temperature,
                "top_p": 1,
                "max_tokens": max_tokens,
                "stop": ["<end_of_turn>"],
            }
        else:
            endpoint = f"{self.base_url}/v1/chat/completions"
            messages = _history_messages(history)
            messages.append({"role": "user", "content": prompt})
            payload = {
                "messages": messages,
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
    gender: str | None = None,
) -> str:
    text = f'"{source}"' if quoted else source
    if gender:
        text += f"\n\n{GENDER_HINTS[gender]}"
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
    if not 0.4 <= ratio <= 2.2 or _has_unexpected_latin(cleaned, source):
        return False
    return not _is_repeated(cleaned)


def should_passthrough(source: str) -> bool:
    cleaned = source.strip()
    return (
        not LETTER.search(cleaned)
        or URL.fullmatch(cleaned) is not None
        or len(cleaned.split()) == 1
    )


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


def _has_unexpected_latin(translation: str, source: str) -> bool:
    source_runs = {match.group().casefold() for match in LATIN_RUN.finditer(source)}
    return any(
        match.group().casefold() not in source_runs
        for match in LATIN_RUN.finditer(translation)
    )


def _post_json(endpoint: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_is_healthy(base_url: str, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _history_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": role, "content": content}
        for source, translation in history[-2:]
        for role, content in (("user", source), ("assistant", translation))
    ]
