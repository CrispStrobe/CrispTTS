"""Pre-rendered spoken AI-disclosure clips shipped with CrispTTS.

``disclosure_<lang>.flac`` — the sentence in :data:`watermark.DISCLAIMER_TEXTS`
for that language, mono 16 kHz.

These exist so that prepending the Art. 50(4) spoken disclosure to voice-cloned
audio never depends on a TTS backend, a model download or a network call. Every
other source can be unavailable — no CrispASR binary, no edge-tts, no
connectivity — and CrispTTS can still disclose, in the language of the audio,
rather than discarding the output.

Regenerate with ``python scripts/make_disclosure_assets.py``.
"""
