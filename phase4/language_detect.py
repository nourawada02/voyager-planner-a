"""Deterministic Arabic-script detection (Hybrid Chat C.1 §6). Small,
regex/range-based -- never an LLM call, matching this project's existing
`phase4.guards` "deterministic, regex/allowlist-based" precedent. No such
utility existed anywhere in this codebase before this checkpoint (only a
prompt-string instruction telling Qwen to recognize multilingual
continuation phrases, `phase4.graph._build_capability_classification_prompt`
-- never deterministic detection code); this is a new, narrow addition,
not a modification of that prompt.
"""

from __future__ import annotations

import re

# Arabic, Arabic Supplement, Arabic Extended-A, and Arabic Presentation
# Forms-A/B unicode blocks, built from explicit numeric code points
# (never a literal right-to-left character typed inline in source --
# easier to review and immune to editor/encoding mangling). Covers
# standard Arabic-script text (Persian/Urdu loan characters included,
# deliberately broad rather than Arabic-only, since this project only
# ever needs "is this Arabic-script text", not a language-family
# classifier).
_ARABIC_UNICODE_BLOCKS = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)
_ARABIC_CHAR_PATTERN = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _ARABIC_UNICODE_BLOCKS) + "]"
)
_LETTER_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)

_MIN_MEANINGFUL_ARABIC_CHARS = 8


def is_predominantly_arabic(text: str) -> bool:
    """True when more than half of the letter characters in `text` are
    Arabic-script. Returns False for empty/whitespace-only/no-letter text
    -- never a division by zero, never a guess."""
    letters = _LETTER_PATTERN.findall(text or "")
    if not letters:
        return False
    arabic_letters = _ARABIC_CHAR_PATTERN.findall(text or "")
    return len(arabic_letters) / len(letters) > 0.5


def contains_meaningful_arabic(text: str, min_chars: int = _MIN_MEANINGFUL_ARABIC_CHARS) -> bool:
    """True when `text` contains at least `min_chars` Arabic-script
    characters -- the language-correctness check for a response that was
    supposed to be generated in Arabic (Hybrid Chat C.1 §6: 'validate the
    Arabic response contains meaningful Arabic text'). A handful of
    Arabic proper nouns embedded in an otherwise-English reply does not
    pass; a genuine Arabic-language reply does."""
    return len(_ARABIC_CHAR_PATTERN.findall(text or "")) >= min_chars
