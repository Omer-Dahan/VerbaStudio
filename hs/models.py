from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Cue:
    start: float
    end: float
    source: str
    text: str | None = None
    words: list[Word] = field(default_factory=list)
    translation_failed: bool = False
    cps_forced: bool = False

    @property
    def display_text(self) -> str:
        return self.text if self.text is not None else self.source
