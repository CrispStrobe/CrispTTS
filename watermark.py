# watermark.py — AI-generated audio watermarking for CrispTTS.
#
# Implements a multi-layered provenance system ported from CrispASR:
#
#   1. Spread-spectrum watermark (pure Python/numpy, always available)
#      Embeds an imperceptible pseudorandom pattern in the frequency domain.
#
#   2. AudioSeal neural watermark (optional, multiple backends):
#      a) Direct Python AudioSeal package (pip install audioseal)
#      b) CrispASR C binding with GGUF model (pip install crispasr)
#      More robust against adversarial removal, lossy compression, etc.
#
#   3. WAV LIST/INFO and MP3 ID3v2 metadata declaring AI-generated origin.
#
#   4. C2PA content credentials (optional, pip install c2pa-python)
#      Cryptographically signed provenance manifests.
#
# The dispatcher tries AudioSeal (Python or crispasr) first, then falls
# back to the built-in spread-spectrum.

import logging
import os
import struct

import numpy as np

logger = logging.getLogger("CrispTTS.watermark")

# ---------------------------------------------------------------------------
# Constants — must match CrispASR's crispasr_watermark.h for cross-compat
# ---------------------------------------------------------------------------
WATERMARK_KEY = 0x437269737041535F   # "CrispASR" in hex-ish
WATERMARK_NBINS = 32
_FFT_SIZE = 1024
_HOP = _FFT_SIZE // 2


# ---------------------------------------------------------------------------
# xoshiro128+ PRNG (matches CrispASR's crispasr_wm::prng exactly)
# ---------------------------------------------------------------------------
_U64 = 0xFFFFFFFFFFFFFFFF  # mask to 64-bit


def _splitmix64(x: int) -> tuple[int, int]:
    x = (x + 0x9E3779B97F4A7C15) & _U64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return x, (z ^ (z >> 31)) & _U64


class _Prng:
    __slots__ = ("s0", "s1")

    def __init__(self, seed: int):
        # Must match C++ semantics: splitmix takes arg by reference.
        # prng(seed): s[0] = splitmix(seed); s[1] = splitmix(s[0]);
        # The second call MUTATES s[0] (pass-by-ref), so s[0] ends up
        # as the intermediate state (original_s0 + K), not the hash.
        _, s0_initial = _splitmix64(seed)
        self.s0, self.s1 = _splitmix64(s0_initial)  # s0 = state after K added, s1 = hash

    def next(self) -> int:
        s0, s1 = self.s0, self.s1
        result = (s0 + s1) & _U64
        s1 ^= s0
        self.s0 = (((s0 << 55) | (s0 >> 9)) & _U64) ^ s1 ^ ((s1 << 14) & _U64)
        self.s1 = ((s1 << 36) | (s1 >> 28)) & _U64
        return result

    def next_u32(self, bound: int) -> int:
        return int(self.next() % bound)


# ---------------------------------------------------------------------------
# Bin pattern generation (matches generate_bin_pattern in C++)
# ---------------------------------------------------------------------------

def _generate_bin_pattern(key: int, n_fft: int, n_bins: int):
    """Return list of (bin_index, sign) tuples."""
    rng = _Prng(key)
    lo_bin = n_fft // 16
    hi_bin = n_fft // 2 - 1
    span = hi_bin - lo_bin
    if span <= 0 or n_bins <= 0:
        return []
    bins = []
    for _ in range(n_bins):
        idx = lo_bin + rng.next_u32(span)
        sign = 1 if (rng.next() & 1) else -1
        bins.append((idx, sign))
    return bins


# ---------------------------------------------------------------------------
# Spread-spectrum embed (mirrors crispasr_watermark_embed_impl)
# ---------------------------------------------------------------------------

def spread_spectrum_embed(pcm: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """Embed a spread-spectrum watermark into float32 mono PCM.

    Args:
        pcm: 1-D float32 array of audio samples.
        alpha: Watermark strength (0.08 = ~38 dB SNR, imperceptible on speech).

    Returns:
        Watermarked copy of the PCM array.
    """
    n = len(pcm)
    if n < _FFT_SIZE:
        return pcm.copy()

    bins = _generate_bin_pattern(WATERMARK_KEY, _FFT_SIZE, WATERMARK_NBINS)
    if not bins:
        return pcm.copy()

    window = np.hanning(_FFT_SIZE).astype(np.float32)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    for start in range(0, n - _FFT_SIZE + 1, _HOP):
        frame = pcm[start:start + _FFT_SIZE] * window
        spectrum = np.fft.rfft(frame)

        # RMS magnitude for energy-proportional nudge
        mags = np.abs(spectrum[1:_FFT_SIZE // 2])
        rms_mag = np.sqrt(np.mean(mags ** 2)) if len(mags) > 0 else 0.0
        nudge = alpha * rms_mag

        for b_idx, b_sign in bins:
            if b_idx >= len(spectrum):
                continue
            mag = abs(spectrum[b_idx])
            new_mag = max(mag + nudge * b_sign, 0.0)
            if mag > 1e-15:
                scale = new_mag / mag
                spectrum[b_idx] *= scale
            elif b_sign > 0:
                spectrum[b_idx] = complex(nudge, 0.0)

        reconstructed = np.fft.irfft(spectrum, n=_FFT_SIZE).astype(np.float32)
        out[start:start + _FFT_SIZE] += reconstructed * window
        norm[start:start + _FFT_SIZE] += window ** 2

    result = pcm.copy().astype(np.float64)
    mask = norm > 1e-8
    result[mask] = out[mask] / norm[mask]
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Spread-spectrum detect (mirrors crispasr_watermark_detect_impl)
# ---------------------------------------------------------------------------

def spread_spectrum_detect(pcm: np.ndarray) -> float:
    """Detect spread-spectrum watermark in float32 mono PCM.

    Uses averaged-spectrum detection: computes the mean magnitude spectrum
    across all frames, then correlates the watermark bin pattern against
    the averaged spectrum. This is significantly more robust on tonal/speech
    signals than per-frame detection because frame-level noise averages out.

    Returns:
        Confidence in [0, 1].  >0.65 = watermark present, <0.4 = absent.
    """
    n = len(pcm)
    if n < _FFT_SIZE:
        return 0.0

    bins = _generate_bin_pattern(WATERMARK_KEY, _FFT_SIZE, WATERMARK_NBINS)
    if not bins:
        return 0.0

    window = np.hanning(_FFT_SIZE).astype(np.float32)
    n_fft_half = _FFT_SIZE // 2

    # Phase 1: Accumulate magnitude spectra across all frames
    all_mags = []
    for start in range(0, n - _FFT_SIZE + 1, _HOP):
        frame = pcm[start:start + _FFT_SIZE] * window
        spectrum = np.fft.rfft(frame)
        all_mags.append(np.abs(spectrum[:n_fft_half]).astype(np.float64))

    if not all_mags:
        return 0.0

    # Phase 2: Average spectrum (cancels per-frame noise, preserves watermark)
    avg_mags = np.mean(all_mags, axis=0)

    # Phase 3: Correlate watermark pattern against averaged spectrum
    correlation = 0.0
    valid_bins = 0
    for b_idx, b_sign in bins:
        if b_idx >= len(avg_mags):
            continue
        # Local mean of ±2 neighbours (excluding self)
        neighbours = []
        for d in range(-2, 3):
            nb = b_idx + d
            if 1 <= nb < len(avg_mags) and d != 0:
                neighbours.append(avg_mags[nb])
        if not neighbours:
            continue
        local_mean = sum(neighbours) / len(neighbours)
        if local_mean < 1e-12 and avg_mags[b_idx] < 1e-12:
            continue
        ref = max(local_mean, 1e-12)
        delta = (avg_mags[b_idx] - local_mean) / ref
        correlation += (1.0 if delta > 0 else -1.0) * b_sign
        valid_bins += 1

    if valid_bins == 0:
        return 0.0

    score = (correlation / valid_bins + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# WavMark neural watermark (MIT license — fully free for commercial use)
# ---------------------------------------------------------------------------

_wavmark_model = None


def load_wavmark() -> bool:
    """Load the WavMark neural watermark model (MIT license).

    WavMark embeds a 16-bit payload into 16 kHz mono audio with >38 dB SNR.
    Robust against Gaussian noise, MP3 compression, low-pass filter, and
    speed variation. Fully MIT licensed (code + model weights).

    Requires: pip install wavmark
    Returns True on success.
    """
    global _backend, _wavmark_model
    try:
        import torch
        import wavmark
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _wavmark_model = wavmark.load_model().to(device)
        _backend = "wavmark"
        logger.info("WavMark neural watermark loaded (MIT license).")
        return True
    except ImportError:
        logger.debug("wavmark package not installed.")
        return False
    except Exception as e:
        logger.warning("Failed to load WavMark model: %s", e)
        return False


# CrispTTS AI-generated marker: fixed 16-bit payload for WavMark
# Encodes "CT" (0x43, 0x54) in binary = 0100_0011_0101_0100
_WAVMARK_PAYLOAD = np.array(
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.float64
)


def _embed_wavmark(pcm: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Embed watermark using WavMark (MIT license)."""
    import wavmark
    # WavMark requires 16 kHz mono
    if sample_rate != 16000:
        pcm_16k = _resample_linear(pcm, sample_rate, 16000)
    else:
        pcm_16k = pcm
    watermarked_16k, _ = wavmark.encode_watermark(
        _wavmark_model, pcm_16k.astype(np.float64), _WAVMARK_PAYLOAD,
        show_progress=False,
    )
    if sample_rate != 16000:
        # Compute delta at 16 kHz and resample it back
        delta_16k = (watermarked_16k - pcm_16k).astype(np.float32)
        delta_native = _resample_linear(delta_16k, 16000, sample_rate)
        if len(delta_native) > len(pcm):
            delta_native = delta_native[:len(pcm)]
        elif len(delta_native) < len(pcm):
            delta_native = np.pad(delta_native, (0, len(pcm) - len(delta_native)))
        return pcm + delta_native
    return watermarked_16k.astype(np.float32)


def _detect_wavmark(pcm: np.ndarray, sample_rate: int = 24000) -> float:
    """Detect WavMark watermark. Returns confidence [0, 1]."""
    import wavmark
    if sample_rate != 16000:
        pcm = _resample_linear(pcm, sample_rate, 16000)
    payload_decoded, info = wavmark.decode_watermark(
        _wavmark_model, pcm.astype(np.float64), show_progress=False,
    )
    if payload_decoded is None:
        return 0.0
    # Compare decoded payload against our fixed marker
    match_ratio = float(np.mean(payload_decoded[:16] == _WAVMARK_PAYLOAD))
    return match_ratio


# ---------------------------------------------------------------------------
# AudioSeal dispatcher (multiple backends)
# ---------------------------------------------------------------------------

# Backend priority: wavmark (MIT) > audioseal (Python) > crispasr (C) > spread-spectrum
_backend = "spread_spectrum"  # active backend name
_audioseal_generator = None   # audioseal Python generator model
_audioseal_detector = None    # audioseal Python detector model
_crispasr_wm = None           # crispasr C binding module


def load_audioseal_python() -> bool:
    """Load AudioSeal directly via the audioseal Python package.

    Requires: pip install audioseal
    Returns True on success.
    """
    global _backend, _audioseal_generator, _audioseal_detector
    try:
        from audioseal import AudioSeal
        _audioseal_generator = AudioSeal.load_generator("audioseal_wm_16bits")
        _audioseal_detector = AudioSeal.load_detector("audioseal_detector_16bits")
        _backend = "audioseal_python"
        logger.info("AudioSeal loaded via Python audioseal package.")
        return True
    except ImportError:
        logger.debug("audioseal package not installed.")
        return False
    except Exception as e:
        logger.warning("Failed to load audioseal Python models: %s", e)
        return False


def load_audioseal_model(gguf_path: str) -> bool:
    """Load an AudioSeal GGUF model via the crispasr Python binding.

    Returns True on success, False if crispasr is not available or load fails.
    """
    global _backend, _crispasr_wm
    try:
        import crispasr
        crispasr.watermark_load_model(gguf_path)
        _crispasr_wm = crispasr
        _backend = "audioseal_crispasr"
        logger.info("AudioSeal model loaded via crispasr: %s", gguf_path)
        return True
    except ImportError:
        logger.info("crispasr Python binding not available.")
        return False
    except Exception as e:
        logger.warning("Failed to load AudioSeal model via crispasr: %s", e)
        return False


def _resample_linear(pcm: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Linear interpolation resampling (matches CrispASR's dispatcher)."""
    if from_sr == to_sr:
        return pcm
    ratio = to_sr / from_sr
    new_len = int(len(pcm) * ratio)
    indices = np.arange(new_len, dtype=np.float64) / ratio
    idx_floor = np.clip(np.floor(indices).astype(int), 0, len(pcm) - 1)
    idx_ceil = np.clip(idx_floor + 1, 0, len(pcm) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return pcm[idx_floor] * (1.0 - frac) + pcm[idx_ceil] * frac


def _embed_audioseal_python(pcm: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Embed watermark using the audioseal Python package.

    Resamples to 16 kHz if needed (AudioSeal's native rate), embeds the
    watermark, then resamples the delta back to the original rate.
    """
    import torch
    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        pcm_16k = _resample_linear(pcm, sample_rate, 16000)
    else:
        pcm_16k = pcm
    tensor = torch.from_numpy(pcm_16k).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    watermark = _audioseal_generator.get_watermark(tensor, sample_rate=16000)
    if sample_rate != 16000:
        # Upsample the watermark delta back to original rate and add
        wm_delta = watermark.squeeze().detach().numpy().astype(np.float32)
        wm_delta_native = _resample_linear(wm_delta, 16000, sample_rate)
        # Trim/pad to match original length
        if len(wm_delta_native) > len(pcm):
            wm_delta_native = wm_delta_native[:len(pcm)]
        elif len(wm_delta_native) < len(pcm):
            wm_delta_native = np.pad(wm_delta_native, (0, len(pcm) - len(wm_delta_native)))
        return pcm + wm_delta_native
    result = tensor + watermark
    return result.squeeze().detach().numpy().astype(np.float32)


def _detect_audioseal_python(pcm: np.ndarray, sample_rate: int = 24000) -> float:
    """Detect watermark using the audioseal Python package."""
    import torch
    if sample_rate != 16000:
        pcm = _resample_linear(pcm, sample_rate, 16000)
    tensor = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    result, _ = _audioseal_detector.detect_watermark(tensor, sample_rate=16000)
    return float(result.mean().item())


def watermark_embed(pcm: np.ndarray, alpha: float = 0.08, sample_rate: int = 24000) -> np.ndarray:
    """Embed AI-generated watermark. Dispatches to the best available backend.

    Priority: wavmark (MIT) > audioseal (Python) > crispasr (C/GGUF) > spread-spectrum.

    Args:
        pcm: 1-D float32 mono PCM array.
        alpha: Strength for spread-spectrum (ignored when neural backends active).
        sample_rate: Audio sample rate (needed for neural backend resampling).

    Returns:
        Watermarked PCM (new array, input unchanged).
    """
    if os.environ.get("CRISPTTS_NO_WATERMARK"):
        return pcm.copy()

    if _backend == "wavmark" and _wavmark_model is not None:
        try:
            result = _embed_wavmark(pcm, sample_rate)
            logger.debug("WavMark (MIT) watermark embedded (%d samples).", len(pcm))
            return result
        except Exception as e:
            logger.warning("WavMark embed failed, trying next backend: %s", e)

    if _backend == "audioseal_python" and _audioseal_generator is not None:
        try:
            result = _embed_audioseal_python(pcm, sample_rate)
            logger.debug("AudioSeal (Python) watermark embedded (%d samples).", len(pcm))
            return result
        except Exception as e:
            logger.warning("AudioSeal Python embed failed, trying next backend: %s", e)

    if _backend == "audioseal_crispasr" and _crispasr_wm is not None:
        try:
            wm_pcm = pcm.copy()
            _crispasr_wm.watermark_embed(wm_pcm, alpha)
            logger.debug("AudioSeal (crispasr) watermark embedded (%d samples).", len(pcm))
            return wm_pcm
        except Exception as e:
            logger.warning("AudioSeal crispasr embed failed, falling back to spread-spectrum: %s", e)

    result = spread_spectrum_embed(pcm, alpha)
    logger.debug("Spread-spectrum watermark embedded (%d samples).", len(pcm))
    return result


def watermark_detect(pcm: np.ndarray, sample_rate: int = 24000) -> float:
    """Detect AI-generated watermark. Returns confidence [0, 1].

    Tries all available backends in priority order: wavmark > audioseal > spread-spectrum.
    """
    if _backend == "wavmark" and _wavmark_model is not None:
        try:
            score = _detect_wavmark(pcm, sample_rate)
            if score > 0.4:  # WavMark found something
                return score
            # Fall through to spread-spectrum (may have been watermarked by CrispASR binary)
        except Exception as e:
            logger.warning("WavMark detect failed, trying next backend: %s", e)

    if _backend == "audioseal_python" and _audioseal_detector is not None:
        try:
            return _detect_audioseal_python(pcm, sample_rate)
        except Exception as e:
            logger.warning("AudioSeal Python detect failed, trying next backend: %s", e)

    if _backend == "audioseal_crispasr" and _crispasr_wm is not None:
        try:
            return _crispasr_wm.watermark_detect(pcm.astype(np.float32, copy=True))
        except Exception as e:
            logger.warning("AudioSeal crispasr detect failed, falling back to spread-spectrum: %s", e)

    return spread_spectrum_detect(pcm)


def watermark_verify_file(filepath: str) -> float | None:
    """Read a WAV file and verify its watermark. Returns confidence or None on error."""
    try:
        import soundfile as sf_verify
        data, sr = sf_verify.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return watermark_detect(data, sample_rate=sr)
    except Exception as e:
        logger.warning("Watermark verification failed for %s: %s", filepath, e)
        return None


# ---------------------------------------------------------------------------
# WAV LIST/INFO metadata (AI-provenance)
# ---------------------------------------------------------------------------

def make_wav_info_chunk() -> bytes:
    """Build a RIFF LIST/INFO chunk declaring this audio as AI-generated.

    Returns raw bytes to append after the WAV data chunk (caller must
    patch the RIFF size to account for it).
    """
    def _info_entry(chunk_id: bytes, value: str) -> bytes:
        val_bytes = value.encode("latin-1") + b"\x00"
        entry = chunk_id + struct.pack("<I", len(val_bytes)) + val_bytes
        if len(val_bytes) & 1:
            entry += b"\x00"  # pad to even boundary
        return entry

    body = b"INFO"
    body += _info_entry(b"ISFT", "CrispTTS (AI-generated audio)")
    body += _info_entry(
        b"ICMT",
        "This audio was synthesized by an AI text-to-speech model. "
        "It is not a recording of a human speaker.",
    )
    return b"LIST" + struct.pack("<I", len(body)) + body


def inject_wav_metadata(wav_bytes: bytes) -> bytes:
    """Inject AI-provenance LIST/INFO metadata into a WAV byte string.

    Works on complete in-memory WAV files. If the input is not a valid
    RIFF/WAVE container, returns it unchanged.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes

    info_chunk = make_wav_info_chunk()
    # Append INFO after existing data, patch RIFF size
    new_wav = bytearray(wav_bytes)
    new_wav.extend(info_chunk)
    # RIFF size is at offset 4, little-endian uint32
    new_riff_size = len(new_wav) - 8
    struct.pack_into("<I", new_wav, 4, new_riff_size)
    return bytes(new_wav)


# ---------------------------------------------------------------------------
# MP3 ID3v2 metadata (AI-provenance via TXXX frames)
# ---------------------------------------------------------------------------

def make_id3v2_ai_tag() -> bytes:
    """Build a minimal ID3v2.3 tag with TXXX frames marking AI-generated audio.

    Prepend the returned bytes to raw MP3 data.
    """
    def _make_txxx(description: str, value: str) -> bytes:
        payload = b"\x00" + description.encode("latin-1") + b"\x00" + value.encode("latin-1")
        sz = len(payload)
        frame_header = b"TXXX" + struct.pack(">I", sz) + b"\x00\x00"
        return frame_header + payload

    frames = b""
    frames += _make_txxx("AI_GENERATED", "true")
    frames += _make_txxx("GENERATOR", "CrispTTS")
    frames += _make_txxx(
        "AI_CONTENT_NOTICE",
        "This audio was synthesized by an AI text-to-speech model. "
        "It is not a recording of a human speaker.",
    )

    sz = len(frames)
    header = b"ID3"
    header += b"\x03\x00"  # version 2.3, revision 0
    header += b"\x00"      # flags
    header += bytes([
        (sz >> 21) & 0x7F,
        (sz >> 14) & 0x7F,
        (sz >> 7)  & 0x7F,
        sz         & 0x7F,
    ])
    return header + frames


def inject_mp3_metadata(mp3_bytes: bytes) -> bytes:
    """Prepend AI-provenance ID3v2 tag to MP3 data if not already present."""
    if mp3_bytes[:3] == b"ID3":
        return mp3_bytes  # already has ID3 tag, don't double-tag
    return make_id3v2_ai_tag() + mp3_bytes


# ---------------------------------------------------------------------------
# FLAC Vorbis comment metadata (AI-provenance)
# ---------------------------------------------------------------------------

def inject_flac_metadata(filepath: str) -> bool:
    """Inject AI-provenance Vorbis comments into a FLAC file.

    Uses mutagen if available. Returns True on success, False otherwise.
    """
    try:
        from mutagen.flac import FLAC
        audio = FLAC(filepath)
        audio["AI_GENERATED"] = "true"
        audio["GENERATOR"] = "CrispTTS"
        audio["COMMENT"] = (
            "This audio was synthesized by an AI text-to-speech model. "
            "It is not a recording of a human speaker."
        )
        audio.save()
        logger.debug("FLAC AI-provenance metadata injected: %s", filepath)
        return True
    except ImportError:
        logger.debug("mutagen not installed — FLAC metadata injection skipped.")
        return False
    except Exception as e:
        logger.warning("FLAC metadata injection failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Opus/OGG Vorbis comment metadata (AI-provenance)
# ---------------------------------------------------------------------------

def inject_opus_metadata(filepath: str) -> bool:
    """Inject AI-provenance Vorbis comments into an Opus/OGG file.

    Uses mutagen if available. Returns True on success, False otherwise.
    """
    try:
        from mutagen.oggopus import OggOpus
        audio = OggOpus(filepath)
        audio["AI_GENERATED"] = "true"
        audio["GENERATOR"] = "CrispTTS"
        audio["COMMENT"] = (
            "This audio was synthesized by an AI text-to-speech model. "
            "It is not a recording of a human speaker."
        )
        audio.save()
        logger.debug("Opus AI-provenance metadata injected: %s", filepath)
        return True
    except ImportError:
        logger.debug("mutagen not installed — Opus metadata injection skipped.")
        return False
    except Exception as e:
        logger.warning("Opus metadata injection failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Voice-cloning consent gate
# ---------------------------------------------------------------------------

# Model IDs / handler keys that involve voice cloning
VOICE_CLONING_HANDLER_KEYS = frozenset({
    "synthesize_with_outetts_llamacpp",
    "synthesize_with_outetts_hf",
    "synthesize_with_coqui_xtts_v2",
    "synthesize_with_llasa_hybrid_de_zeroshot",
    "synthesize_with_llasa_german_transformers_zeroshot",
    "synthesize_with_llasa_multilingual_hf_zeroshot",
    "synthesize_with_kartoffelbox_zeroshot",
    "synthesize_with_f5_tts",
    "synthesize_with_zonos",
    "synthesize_with_chatterbox",
})

VOICE_CLONING_MODEL_KEYWORDS = frozenset({
    "zeroshot", "xtts", "clone", "f5_tts", "zonos", "chatterbox",
    "vibevoice", "indextts", "voxcpm2", "qwen3_tts",
})


def requires_consent(model_id: str, handler_key: str, voice_id: str | None = None) -> bool:
    """Check whether a model/handler involves voice cloning.

    Also detects voice cloning when a .wav file is passed as voice_id
    to any backend (including crispasr), since that implies the user
    is cloning a voice from a reference recording.
    """
    if handler_key in VOICE_CLONING_HANDLER_KEYS:
        return True
    model_lower = model_id.lower()
    if any(kw in model_lower for kw in VOICE_CLONING_MODEL_KEYWORDS):
        return True
    # .wav voice path = voice cloning on any backend
    if voice_id and isinstance(voice_id, str) and voice_id.lower().endswith(".wav"):
        return True
    return False


_CONSENT_LOG_PATH = os.path.join(os.path.expanduser("~"), ".cache", "crisptts", "consent_audit.log")


def log_consent_attestation(
    model_id: str,
    voice_id: str | None = None,
    source: str = "CLI --i-have-rights flag",
) -> None:
    """Log a consent attestation to stderr AND a persistent audit log file.

    Format matches CrispASR: [CONSENT] ts=ISO8601 model=X voice=Y attestation="..."

    The persistent log at ~/.cache/crisptts/consent_audit.log ensures the
    audit trail survives even when stderr is not captured.
    """
    import sys
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    voice_str = voice_id or "default"
    msg = f'[CONSENT] ts={ts} model={model_id} voice={voice_str} attestation="{source}"\n'
    sys.stderr.write(msg)
    sys.stderr.flush()

    # Persistent audit log
    try:
        os.makedirs(os.path.dirname(_CONSENT_LOG_PATH), exist_ok=True)
        with open(_CONSENT_LOG_PATH, "a") as f_audit:
            f_audit.write(msg)
    except OSError as e:
        logger.debug("Could not write consent audit log: %s", e)

    logger.info("Consent attestation logged for model=%s voice=%s", model_id, voice_str)


# ---------------------------------------------------------------------------
# Spoken AI disclaimer for voice-cloned audio (EU AI Act Art. 50(4))
# ---------------------------------------------------------------------------

DISCLAIMER_TEXT = "This audio was generated by artificial intelligence."
_DISCLAIMER_SILENCE_SEC = 0.3  # 300ms gap between disclaimer and content


def generate_spoken_disclaimer(sample_rate: int = 24000) -> np.ndarray | None:
    """Generate a spoken AI disclaimer using a non-cloning TTS backend.

    Priority: CrispASR kokoro (local, fast) > Edge TTS (cloud) > beep marker.

    Returns float32 PCM array at the given sample rate, or None on failure.
    """
    # Try CrispASR kokoro (local, no internet, no voice cloning)
    try:
        import shutil
        import subprocess
        import tempfile

        exe = shutil.which("crispasr") or os.environ.get("CRISPASR_EXECUTABLE")
        if exe:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                result = subprocess.run(  # noqa: S603
                    [exe, "-m", "auto", "--backend", "kokoro",
                     "--tts", DISCLAIMER_TEXT, "--tts-output", tmp_wav,
                     "--auto-download", "-t", "4"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and os.path.isfile(tmp_wav) and os.path.getsize(tmp_wav) > 100:
                    import soundfile as sf_disc
                    data, sr = sf_disc.read(tmp_wav, dtype="float32")
                    if data.ndim > 1:
                        data = data[:, 0]
                    if sr != sample_rate:
                        data = _resample_linear(data, sr, sample_rate)
                    logger.info("Spoken disclaimer generated via CrispASR kokoro.")
                    return data
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)
    except Exception as e:
        logger.debug("CrispASR disclaimer generation failed: %s", e)

    # Try edge-tts (cloud, lightweight, no voice cloning concerns)
    try:
        import asyncio
        import tempfile

        import edge_tts

        async def _synth():
            communicate = edge_tts.Communicate(DISCLAIMER_TEXT, "en-US-AriaNeural")
            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                await communicate.save(tmp)
                try:
                    import soundfile as sf_disc
                    data, sr = sf_disc.read(tmp, dtype="float32")
                    if sr != sample_rate:
                        data = _resample_linear(data, sr, sample_rate)
                    return data
                except ImportError:
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(tmp)
                    seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
                    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
                    return raw
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_synth())
        finally:
            loop.close()
    except Exception as e:
        logger.debug("Edge TTS disclaimer generation failed: %s", e)

    # Fallback: generate a simple beep pattern (3 short beeps) as a
    # machine-readable audio marker that something precedes the content
    try:
        duration = 0.15  # each beep
        gap = 0.08
        freq = 880.0
        t_beep = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
        beep = 0.3 * np.sin(2 * np.pi * freq * t_beep)
        # Fade in/out to avoid clicks
        fade_len = int(sample_rate * 0.01)
        beep[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
        beep[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)
        silence_gap = np.zeros(int(sample_rate * gap), dtype=np.float32)
        marker = np.concatenate([beep, silence_gap, beep, silence_gap, beep])
        logger.info("Using beep marker as spoken disclaimer fallback.")
        return marker
    except Exception as e:
        logger.warning("Disclaimer generation failed entirely: %s", e)
        return None


# Cache the disclaimer audio to avoid re-synthesizing
_disclaimer_cache: dict[int, np.ndarray] = {}


def prepend_disclaimer(pcm: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Prepend an AI-generated spoken disclaimer to voice-cloned audio.

    Matches CrispASR's approach: disclaimer + 300ms silence + original audio.
    The disclaimer is cached after first generation.
    """
    if sample_rate not in _disclaimer_cache:
        disclaimer = generate_spoken_disclaimer(sample_rate)
        if disclaimer is not None:
            _disclaimer_cache[sample_rate] = disclaimer
        else:
            return pcm  # can't generate disclaimer, return original

    disclaimer = _disclaimer_cache[sample_rate]
    silence = np.zeros(int(sample_rate * _DISCLAIMER_SILENCE_SEC), dtype=np.float32)
    return np.concatenate([disclaimer, silence, pcm])


# ---------------------------------------------------------------------------
# C2PA content credentials (optional, pip install c2pa-python)
# ---------------------------------------------------------------------------

_C2PA_MANIFEST_JSON = """{
  "claim_generator": "CrispTTS",
  "assertions": [
    {
      "label": "c2pa.actions",
      "data": {
        "actions": [
          {
            "action": "c2pa.created",
            "digitalSourceType":
              "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            "softwareAgent": "CrispTTS"
          }
        ]
      }
    },
    {
      "label": "c2pa.training-mining",
      "data": {
        "entries": [
          {
            "use": "notAllowed",
            "constraint_info": "This AI-generated audio may not be used to train AI models without explicit permission."
          }
        ]
      }
    }
  ]
}"""


def c2pa_sign_file(
    input_path: str,
    output_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
) -> bool:
    """Sign an audio file with C2PA content credentials.

    Args:
        input_path: Path to the audio file (WAV or MP3).
        output_path: Where to write signed file (defaults to overwrite input).
        cert_path: Path to PEM certificate (or env var C2PA_CERT_PATH).
        key_path: Path to PEM private key (or env var C2PA_KEY_PATH).

    Returns True on success, False if c2pa-python is not installed or signing fails.
    """
    cert_path = cert_path or os.environ.get("C2PA_CERT_PATH")
    key_path = key_path or os.environ.get("C2PA_KEY_PATH")

    if not cert_path or not key_path:
        logger.debug("C2PA signing skipped: no certificate/key configured.")
        return False

    try:
        import c2pa
    except ImportError:
        logger.debug("c2pa-python not installed; C2PA signing skipped.")
        return False

    try:
        cert_data = open(cert_path, "rb").read()
        key_data = open(key_path, "rb").read()

        signer = c2pa.create_signer(cert_data, key_data, "es256")
        builder = c2pa.Builder(_C2PA_MANIFEST_JSON)

        effective_output = output_path or input_path
        if effective_output == input_path:
            # c2pa requires different input/output; use a temp file
            import tempfile
            suffix = os.path.splitext(input_path)[1]
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            try:
                builder.sign_file(input_path, tmp_path, signer)
                import shutil
                shutil.move(tmp_path, input_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        else:
            builder.sign_file(input_path, effective_output, signer)

        logger.info("C2PA content credentials signed: %s", effective_output)
        return True
    except Exception as e:
        logger.warning("C2PA signing failed: %s", e)
        return False
