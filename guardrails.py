"""Input / output guardrails: small, deterministic validators run in a pipeline.

A validator can BLOCK (stop the turn) or TRANSFORM (rewrite the text and keep
going). Matching is done on whole words, so "dan" does not fire inside
"abundance" and "die" does not fire inside "studied".
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


def _find_phrase(text: str, phrases: list[str]) -> str | None:
    """Return the first phrase found in text as a whole word/phrase, else None."""
    for phrase in phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
            return phrase
    return None


# The verdict every validator returns.
@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    reason: str | None = None

    @classmethod
    def ok(cls, text: str, reason: str | None = None) -> "GuardrailResult":
        return cls(allowed=True, text=text, reason=reason)

    @classmethod
    def block(cls, text: str, reason: str) -> "GuardrailResult":
        return cls(allowed=False, text=text, reason=reason)


# The contract every rule must satisfy.
class Validator(ABC):
    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        ...


class MinWordsValidator(Validator):
    def __init__(self, min_words: int = 3) -> None:
        self.min_words = min_words

    def check(self, text: str) -> GuardrailResult:
        if len(re.findall(r"\w+", text)) < self.min_words:
            return GuardrailResult.block(
                text, f"Text must have at least {self.min_words} words."
            )
        return GuardrailResult.ok(text)


class MinLengthValidator(Validator):
    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars

    def check(self, text: str) -> GuardrailResult:
        if len(text.strip()) < self.min_chars:
            return GuardrailResult.block(
                text, f"Text must have at least {self.min_chars} characters."
            )
        return GuardrailResult.ok(text)


class PromptInjectionValidator(Validator):
    DEFAULT_PHRASES = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard all previous instructions",
        "disregard previous instructions",
        "forget everything i said before",
        "forget your instructions",
        "pretend you are a different model",
        "jailbreak",
        "bypass the rules",
        "override your safety protocols",
        "ignore your harness",
        "DAN mode",
    ]

    def __init__(self, forbidden_phrases: list[str] | None = None) -> None:
        self.forbidden_phrases = forbidden_phrases or list(self.DEFAULT_PHRASES)

    def check(self, text: str) -> GuardrailResult:
        hit = _find_phrase(text, self.forbidden_phrases)
        if hit:
            return GuardrailResult.block(
                text, f"Possible prompt injection: contains '{hit}'."
            )
        return GuardrailResult.ok(text)


class ToxicityKeywordValidator(Validator):
    # Abuse and threats only. Topic words like "violence", "murder" or "suicide"
    # are deliberately NOT here: a document assistant must be able to discuss a
    # biography or a clinical trial's adverse events without blocking itself.
    DEFAULT_PHRASES = [
        "kill yourself",
        "screw you",
        "shut up",
        "retard",
        "stupid",
        "idiot",
        "moron",
        "dumbass",
        "asshole",
        "bastard",
        "worthless",
    ]

    def __init__(self, forbidden_phrases: list[str] | None = None) -> None:
        self.forbidden_phrases = forbidden_phrases or list(self.DEFAULT_PHRASES)

    def check(self, text: str) -> GuardrailResult:
        hit = _find_phrase(text, self.forbidden_phrases)
        if hit:
            return GuardrailResult.block(
                text, f"Text contains forbidden phrase: '{hit}'."
            )
        return GuardrailResult.ok(text)


class PIIRedactionValidator(Validator):
    """Rewrites emails, phone numbers and SSNs to placeholders (does not block)."""

    def __init__(self) -> None:
        # Order matters: SSN first so it is not half-eaten by the phone patterns.
        self.patterns: list[tuple[str, re.Pattern[str]]] = [
            ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
            # Indian mobile numbers, e.g. +91 98765 43210
            ("PHONE", re.compile(r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\w)")),
            # US / international style, e.g. +1 (415) 555-0132. Needs 3-3-4
            # digit groups, so dates like 2024-01-15 are left alone.
            (
                "PHONE",
                re.compile(
                    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
                ),
            ),
        ]

    def check(self, text: str) -> GuardrailResult:
        found: list[str] = []
        for label, pattern in self.patterns:
            text, count = pattern.subn(f"[REDACTED_{label}]", text)
            if count and label not in found:
                found.append(label)
        reason = f"Redacted personal data: {', '.join(found).lower()}." if found else None
        return GuardrailResult.ok(text, reason)


# ---------------------------------------------------------------------------
# Optional third-party validator
# ---------------------------------------------------------------------------
class GuardrailsAIValidator(Validator):
    """Wraps Guardrails AI (https://github.com/guardrails-ai) if importable.

    NOTE: this project file is itself named guardrails.py, so Python resolves
    `import guardrails` to THIS file, not the pip package. Until this module is
    renamed, the library can never be imported and available() stays False,
    which is why it is off by default (USE_GUARDRAILS_AI=false).
    """

    def __init__(self) -> None:
        self._guard = None
        try:
            from guardrails import Guard  # type: ignore

            self._guard = Guard()
        except Exception:
            self._guard = None

    def available(self) -> bool:
        return self._guard is not None

    def check(self, text: str) -> GuardrailResult:
        if self._guard is None:
            return GuardrailResult.ok(text)
        try:
            self._guard.validate(text)
            return GuardrailResult.ok(text)
        except Exception as exc:
            return GuardrailResult.block(text, f"Blocked by Guardrails AI: {exc}")


# ---------------------------------------------------------------------------
# The engine: runs a pipeline of validators
# ---------------------------------------------------------------------------
class GuardrailEngine:
    """Runs input validators before the LLM and output validators after it.

    The engine threads the (possibly rewritten) text through the whole pipeline,
    short-circuits on the first block, and collects the notes of validators that
    changed something so callers can show them.
    """

    def __init__(
        self,
        input_validators: list[Validator],
        output_validators: list[Validator],
    ) -> None:
        self.input_validators = input_validators
        self.output_validators = output_validators

    def _run(self, validators: list[Validator], text: str) -> GuardrailResult:
        current = text
        notes: list[str] = []
        for validator in validators:
            result = validator.check(current)
            if not result.allowed:
                return result  # blocked: stop immediately
            current = result.text  # carry forward any transformation
            if result.reason:
                notes.append(result.reason)
        return GuardrailResult.ok(current, " ".join(notes) or None)

    def check_input(self, text: str) -> GuardrailResult:
        return self._run(self.input_validators, text)

    def check_output(self, text: str) -> GuardrailResult:
        return self._run(self.output_validators, text)

    @classmethod
    def default(cls, use_guardrails_ai: bool = False) -> "GuardrailEngine":
        input_validators: list[Validator] = [
            MinWordsValidator(min_words=3),
            PromptInjectionValidator(),
            PIIRedactionValidator(),
            ToxicityKeywordValidator(),
        ]
        output_validators: list[Validator] = [
            MinLengthValidator(min_chars=5),
            ToxicityKeywordValidator(),
        ]

        if use_guardrails_ai:
            extra = GuardrailsAIValidator()
            if extra.available():
                print("[guardrails] Guardrails AI detected - enabled.")
                input_validators.append(extra)
                output_validators.append(extra)

        return cls(input_validators, output_validators)
