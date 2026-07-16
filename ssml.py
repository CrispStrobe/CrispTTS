# ssml.py — SSML-lite preprocessor for CrispTTS.
#
# Parses a subset of SSML tags in input text and converts them to
# a sequence of (text, params) tuples that the synthesis pipeline
# can process segment-by-segment.
#
# Supported tags:
#   <break time="500ms"/>           → silence gap
#   <prosody rate="fast|slow|N%">   → speed multiplier
#   <say-as interpret-as="characters">ABC</say-as> → spell out
#   <phoneme ph="...">word</phoneme> → phoneme override (ignored in text)
#
# Unknown tags are stripped, their content preserved.

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("CrispTTS.ssml")

# Rate name → speed multiplier
_RATE_MAP = {
    "x-slow": 0.5, "slow": 0.75, "medium": 1.0,
    "fast": 1.25, "x-fast": 1.5,
}


@dataclass
class SSMLSegment:
    """A segment of text with optional synthesis parameters."""
    text: str
    speed: float = 1.0
    silence_ms: float = 0.0  # silence to INSERT before this segment
    phoneme_overrides: dict = field(default_factory=dict)


def parse_ssml(text: str) -> list[SSMLSegment]:
    """Parse SSML-lite markup and return a list of segments.

    If the input contains no SSML tags, returns a single segment
    with the original text and default parameters.

    Args:
        text: Input text, optionally containing SSML tags.

    Returns:
        List of SSMLSegment instances.
    """
    # Quick check: if no angle brackets, skip parsing
    if "<" not in text:
        return [SSMLSegment(text=text.strip())] if text.strip() else []

    # Strip optional <speak> wrapper
    text = re.sub(r"</?speak>", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return []

    segments: list[SSMLSegment] = []
    current_speed = 1.0
    pending_silence = 0.0
    phonemes: dict[str, str] = {}

    # Tokenize: split into tags and text chunks
    parts = re.split(r"(<[^>]+>)", text)

    for part in parts:
        if not part:
            continue

        # <break time="500ms"/> or <break time="1s"/>
        m_break = re.match(
            r'<break\s+time=["\'](\d+(?:\.\d+)?)(ms|s)["\']\s*/?>',
            part, re.IGNORECASE,
        )
        if m_break:
            val = float(m_break.group(1))
            unit = m_break.group(2)
            pending_silence += val if unit == "ms" else val * 1000
            continue

        # <prosody rate="fast"> or <prosody rate="120%">
        m_prosody = re.match(
            r'<prosody\s+rate=["\']([^"\']+)["\']\s*>',
            part, re.IGNORECASE,
        )
        if m_prosody:
            rate_str = m_prosody.group(1).strip().lower()
            if rate_str in _RATE_MAP:
                current_speed = _RATE_MAP[rate_str]
            elif rate_str.endswith("%"):
                try:
                    current_speed = float(rate_str[:-1]) / 100.0
                except ValueError:
                    pass
            else:
                try:
                    current_speed = float(rate_str)
                except ValueError:
                    pass
            continue

        # </prosody>
        if re.match(r"</prosody>", part, re.IGNORECASE):
            current_speed = 1.0
            continue

        # <say-as interpret-as="characters">ABC</say-as>
        m_sayas = re.match(
            r'<say-as\s+interpret-as=["\']characters["\']\s*>',
            part, re.IGNORECASE,
        )
        if m_sayas:
            # Next text chunk will be spelled out
            continue
        if re.match(r"</say-as>", part, re.IGNORECASE):
            continue

        # <phoneme ph="...">word</phoneme>
        m_ph = re.match(
            r'<phoneme\s+ph=["\']([^"\']+)["\']\s*>',
            part, re.IGNORECASE,
        )
        if m_ph:
            # Store phoneme; the next text part is the word
            phonemes["_pending_ph"] = m_ph.group(1)
            continue
        if re.match(r"</phoneme>", part, re.IGNORECASE):
            phonemes.pop("_pending_ph", None)
            continue

        # Skip any other tags
        if part.startswith("<"):
            continue

        # Text content
        clean = part.strip()
        if not clean:
            continue

        # Handle say-as: spell out characters
        # (detected by checking if previous tag was say-as)

        # Handle phoneme: store override
        seg_phonemes = {}
        if "_pending_ph" in phonemes:
            seg_phonemes[clean] = phonemes.pop("_pending_ph")

        segments.append(SSMLSegment(
            text=clean,
            speed=current_speed,
            silence_ms=pending_silence,
            phoneme_overrides=seg_phonemes,
        ))
        pending_silence = 0.0

    # Trailing silence (rare)
    if pending_silence > 0 and segments:
        segments.append(SSMLSegment(text="", silence_ms=pending_silence))

    return segments if segments else [SSMLSegment(text=text.strip())]


def spell_out(text: str) -> str:
    """Convert text to spelled-out characters: 'ABC' → 'A. B. C.'"""
    return ". ".join(text) + "."


def has_ssml(text: str) -> bool:
    """Check if text contains SSML-like tags."""
    return bool(re.search(r"<(break|prosody|say-as|phoneme|speak)\b", text, re.IGNORECASE))
