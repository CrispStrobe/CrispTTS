# chunking.py — Smart text chunking for long-form TTS synthesis.
#
# Splits text at sentence boundaries so each chunk can be synthesized
# independently and concatenated. Prevents mid-word cuts and keeps
# chunks within reasonable length limits for TTS backends.

import re


def split_sentences(text: str, max_chars: int = 500) -> list[str]:
    """Split text into sentences, merging short ones up to max_chars.

    Splits at sentence-ending punctuation (. ! ? ; :) followed by
    whitespace. Very short sentences are merged with the next one
    to avoid tiny audio fragments.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk (soft limit — won't
            break mid-sentence).

    Returns:
        List of text chunks, each roughly max_chars or less.
    """
    if not text or len(text) <= max_chars:
        return [text] if text else []

    # Split at sentence boundaries: punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?;:])\s+', text.strip())

    chunks = []
    current = ""

    for part in parts:
        if not part.strip():
            continue
        # If adding this part would exceed max_chars and we already have content
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = f"{current} {part}".strip() if current else part

    if current.strip():
        chunks.append(current.strip())

    return chunks
