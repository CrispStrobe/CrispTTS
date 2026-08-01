#!/usr/bin/env python
"""Render the bundled spoken AI-disclosure clips in ``crisptts_assets/``.

These are the last-resort disclosure source: they let CrispTTS disclose
voice-cloned output on a machine with no TTS backend and no network, which
is otherwise the one configuration where Art. 50(4) disclosure fails and the
output has to be discarded.

Run from the repository root after changing ``DISCLAIMER_TEXTS``:

    python scripts/make_disclosure_assets.py

Requires edge-tts (``pip install edge-tts``) and soundfile. Output is mono
16 kHz FLAC — intelligible for speech, and roughly a third the size of the
equivalent WAV, which matters for something shipped in every wheel.
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from watermark import (  # noqa: E402
    _DISCLAIMER_EDGE_VOICES,
    DISCLAIMER_TEXTS,
    _resample_linear,
)

ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crisptts_assets")
TARGET_RATE = 16000


async def _render(text: str, voice: str) -> np.ndarray:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        await communicate.save(tmp)
        data, rate = sf.read(tmp, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if rate != TARGET_RATE:
            data = _resample_linear(data, rate, TARGET_RATE)
        return data
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    loop = asyncio.new_event_loop()
    try:
        for lang, text in sorted(DISCLAIMER_TEXTS.items()):
            voice = _DISCLAIMER_EDGE_VOICES[lang]
            pcm = loop.run_until_complete(_render(text, voice))
            # Trim near-silence at both ends so the clip starts promptly.
            loud = np.nonzero(np.abs(pcm) > 0.01)[0]
            if len(loud):
                pcm = pcm[max(0, loud[0] - 800):loud[-1] + 800]
            path = os.path.join(ASSET_DIR, f"disclosure_{lang}.flac")
            sf.write(path, pcm, TARGET_RATE)
            print(f"{lang}: {len(pcm) / TARGET_RATE:5.2f}s  "
                  f"{os.path.getsize(path) / 1024:6.1f} KiB  {voice}")
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
