import re


QUOTE_CHARACTERS = "'\"`´’‘“”"
QUOTE_TRANSLATION = str.maketrans("", "", QUOTE_CHARACTERS)
_DUPLICATE_NUMERIC_SEPARATOR = re.compile(
    rf"(?<=\d)([.,:/%\-])[{re.escape(QUOTE_CHARACTERS)}]*\s+([.,:/%\-])(?=\d)"
)
_SPACED_NUMERIC_SEPARATOR = re.compile(r"(?<=\d)\s+([.,:/%\-])(?=\d)")


def normalize_subtitle_text(text: str) -> str:
    text = text.translate(QUOTE_TRANSLATION)
    text = _DUPLICATE_NUMERIC_SEPARATOR.sub(_keep_one_separator, text)
    return _SPACED_NUMERIC_SEPARATOR.sub(r"\1", text)


def _keep_one_separator(match: re.Match[str]) -> str:
    left, right = match.groups()
    return left if left == right else right
