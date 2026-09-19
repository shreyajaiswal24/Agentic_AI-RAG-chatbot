"""Text normalisation for the extracted PDF text.

The ebook's extracted text contains predictable noise that hurts both
embeddings and the LLM's reading of the context:
  * a running header "AGENTIC AI FOR EXECUTIVES" on most pages
  * standalone page-number lines
  * bullet glyphs that pypdf cannot decode (U+FFFD replacement char)
  * ragged whitespace from PDF line wrapping
Rules are deliberately conservative: we never rewrite words, only remove
layout artefacts.
"""

import re

_RUNNING_HEADER = re.compile(r"^\s*AGENTIC AI FOR EXECUTIVES\s*$", re.IGNORECASE | re.MULTILINE)
_BARE_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")

_CHAR_MAP = str.maketrans({
    "�": "•",  # undecodable bullet glyph
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    " ": " ",
})


def clean_text(text: str) -> str:
    """Return normalised text. Paragraph breaks are preserved for the chunker."""
    text = text.translate(_CHAR_MAP)
    text = _RUNNING_HEADER.sub("", text)
    text = _BARE_PAGE_NUMBER.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
