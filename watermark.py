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

#: Watermark comb placement, mirroring ``wm_params`` in CrispASR's
#: ``crispasr_watermark.h``.
#:
#: CrispASR moved the comb down to the speech band (their #260): spreading 32
#: bins across ~1.5–11.7 kHz put ~20 of them where clean TTS speech is nearly
#: silent, so the comb was audible as a "tinny" tone. Capping it below ~4.8 kHz
#: hides it under the formants and *improves* detection on voiced speech.
#:
#: CrispTTS stayed on the old band, which broke both directions of
#: cross-detection: measured on crispasr 0.8.25 kokoro output, CrispASR's own
#: detector reads 0.72 while CrispTTS read 0.41 on the same bytes — below its
#: 0.65 threshold. Same key, same PRNG, same FFT size; only the band differed.
#:
#: ``CRISPASR_WATERMARK_LEGACY=1`` selects the old wideband/louder behaviour,
#: exactly as in CrispASR, so the two can be A/B'd together.
def wm_params(n_fft: int, legacy: bool | None = None) -> tuple[int, int, float]:
    """Return ``(lo_bin, hi_bin, default_alpha)`` for the active band."""
    if legacy is None:
        legacy = bool(os.environ.get("CRISPASR_WATERMARK_LEGACY"))
    lo_bin = n_fft // 16  # skip sub-bass; ~1.5 kHz @ 24 kHz
    if legacy:
        return lo_bin, n_fft // 2 - 1, 0.08  # ~11.7 kHz — audible comb
    return lo_bin, n_fft // 5, 0.05  # ~4.8 kHz — inside the speech band


def _generate_bin_pattern(key: int, n_fft: int, n_bins: int,
                          lo_bin: int | None = None, hi_bin: int | None = None):
    """Return list of (bin_index, sign) tuples.

    ``lo_bin``/``hi_bin`` default to the active band. They are explicit
    parameters — as in the C++ ``generate_bin_pattern`` — so detection can
    sweep both bands without mutating global state.
    """
    if lo_bin is None or hi_bin is None:
        band_lo, band_hi, _ = wm_params(n_fft)
        lo_bin = band_lo if lo_bin is None else lo_bin
        hi_bin = band_hi if hi_bin is None else hi_bin
    rng = _Prng(key)
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

def spread_spectrum_embed(pcm: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Embed a spread-spectrum watermark into float32 mono PCM.

    Args:
        pcm: 1-D float32 array of audio samples.
        alpha: Watermark strength. ``None`` or negative selects the band
            default (0.05 speech-band, 0.08 legacy), matching CrispASR's
            convention. ``0`` is an explicit no-op.

    Returns:
        Watermarked copy of the PCM array.
    """
    n = len(pcm)
    if n < _FFT_SIZE:
        return pcm.copy()

    lo_bin, hi_bin, default_alpha = wm_params(_FFT_SIZE)
    if alpha is None or alpha < 0:
        alpha = default_alpha
    if alpha == 0.0:
        # Explicit zero strength leaves the signal untouched, as in CrispASR.
        # Returning early matters: the STFT analysis/synthesis round-trip is
        # not bit-exact, so running it with a zero nudge would still perturb
        # the audio while embedding nothing.
        return pcm.copy()

    bins = _generate_bin_pattern(WATERMARK_KEY, _FFT_SIZE, WATERMARK_NBINS, lo_bin, hi_bin)
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
    """Detect a spread-spectrum watermark in float32 mono PCM.

    Sweeps **both** comb placements and returns the stronger reading, so a file
    marked by an older CrispTTS (or by CrispASR with
    ``CRISPASR_WATERMARK_LEGACY=1``) still verifies after the band change.
    Detection is the one place where being permissive is right: the cost of
    checking a second band is one extra correlation pass, while missing a real
    mark means discarding correctly-marked audio.
    """
    best = 0.0
    for legacy in (False, True):
        lo_bin, hi_bin, _ = wm_params(_FFT_SIZE, legacy=legacy)
        best = max(best, _spread_spectrum_detect_band(pcm, lo_bin, hi_bin))
    return best


def _spread_spectrum_detect_band(pcm: np.ndarray, lo_bin: int, hi_bin: int) -> float:
    """Correlate one comb placement against the averaged spectrum.

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

    bins = _generate_bin_pattern(WATERMARK_KEY, _FFT_SIZE, WATERMARK_NBINS, lo_bin, hi_bin)
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


def watermark_embed(pcm: np.ndarray, alpha: float = 0.08, sample_rate: int = 24000,
                    force: bool = False) -> np.ndarray:
    """Embed AI-generated watermark. Dispatches to the best available backend.

    Priority: wavmark (MIT) > audioseal (Python) > crispasr (C/GGUF) > spread-spectrum.

    Args:
        pcm: 1-D float32 mono PCM array.
        alpha: Strength for spread-spectrum (ignored when neural backends active).
        sample_rate: Audio sample rate (needed for neural backend resampling).
        force: Embed even when CRISPTTS_NO_WATERMARK is set. Used by the
            watermark floor (see :func:`preflight_marking`): when the output
            container cannot carry a C2PA manifest, the watermark is the only
            robust mark and the opt-out must not be able to strip it.

    Returns:
        Watermarked PCM (new array, input unchanged).
    """
    if os.environ.get("CRISPTTS_NO_WATERMARK") and not force:
        return pcm.copy()

    # Lazy-load: if no neural backend was loaded yet, try loading on first use.
    # This avoids loading 200MB+ models at CLI startup for --list-models etc.
    if _backend == "spread_spectrum" and _wavmark_model is None and _audioseal_generator is None:
        if not load_wavmark():
            load_audioseal_python()

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
# Marker strings used both for injection and for the is_marked() probe
# ---------------------------------------------------------------------------

_AI_MARKER_WAV = b"CrispTTS (AI-generated audio)"
_AI_MARKER_TAG = b"AI_GENERATED"

# How much of a file to scan when probing for an existing marker. Container
# metadata lives at the head (ID3, FLAC/Ogg comments) or the tail (WAV LIST),
# so scanning both ends avoids reading large files in full.
_MARKER_SCAN_BYTES = 128 * 1024


def is_marked(filepath: str) -> bool:
    """Return True if the file already carries CrispTTS AI-provenance metadata.

    Used to keep marking idempotent: a file that has been through
    :func:`mark_audio_file` once must not be watermarked a second time,
    which would degrade audio quality without adding provenance.
    """
    try:
        size = os.path.getsize(filepath)
        with open(filepath, "rb") as f_probe:
            head = f_probe.read(_MARKER_SCAN_BYTES)
            if size > _MARKER_SCAN_BYTES:
                f_probe.seek(max(0, size - _MARKER_SCAN_BYTES))
                tail = f_probe.read(_MARKER_SCAN_BYTES)
            else:
                tail = b""
    except OSError:
        return False
    blob = head + tail
    return _AI_MARKER_WAV in blob or _AI_MARKER_TAG in blob


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
    RIFF/WAVE container, returns it unchanged. Idempotent: a WAV that
    already carries the CrispTTS AI-provenance marker is returned as-is,
    so repeated marking cannot stack duplicate LIST/INFO chunks.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes
    if _AI_MARKER_WAV in wav_bytes:
        return wav_bytes  # already marked — do not double-tag

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
    """Prepend AI-provenance ID3v2 tag to MP3 data if not already present.

    Conservative by design: if the data already carries *any* ID3v2 tag this
    returns it untouched, because prepending a second tag header would produce
    a malformed file. To add the AI marker to an MP3 that already has an ID3
    tag from an encoder, use :func:`inject_mp3_metadata_file`, which merges
    into the existing tag instead.
    """
    if mp3_bytes[:3] == b"ID3":
        return mp3_bytes  # already has ID3 tag, don't double-tag
    return make_id3v2_ai_tag() + mp3_bytes


def inject_mp3_metadata_file(filepath: str) -> bool:
    """Inject AI-provenance TXXX frames into an MP3 file, merging safely.

    Encoders (ffmpeg/LAME, as used by pydub on export) write their own ID3v2
    tag, which made the bytes-level injector skip marking entirely — MP3
    outputs ended up with no AI-provenance metadata at all. mutagen merges our
    frames into whatever tag is already there.

    Returns True if the marker is present in the file afterwards.
    """
    try:
        from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
        try:
            tags = ID3(filepath)
        except ID3NoHeaderError:
            tags = ID3()
        if any(frame.desc == "AI_GENERATED" for frame in tags.getall("TXXX")):
            return True
        tags.add(TXXX(encoding=3, desc="AI_GENERATED", text="true"))
        tags.add(TXXX(encoding=3, desc="GENERATOR", text="CrispTTS"))
        tags.add(TXXX(encoding=3, desc="AI_CONTENT_NOTICE", text=(
            "This audio was synthesized by an AI text-to-speech model. "
            "It is not a recording of a human speaker.")))
        tags.save(filepath)
        return True
    except ImportError:
        logger.debug("mutagen not installed — falling back to bytes-level ID3 injection.")
    except Exception as e:
        logger.warning("MP3 ID3 merge failed for %s: %s", filepath, e)

    # Fallback: only works when the file carries no ID3 tag yet.
    try:
        with open(filepath, "rb") as f_mp3:
            raw = f_mp3.read()
        patched = inject_mp3_metadata(raw)
        if patched is raw or patched == raw:
            return _AI_MARKER_TAG in raw
        with open(filepath, "wb") as f_mp3:
            f_mp3.write(patched)
        return True
    except OSError as e:
        logger.warning("MP3 metadata injection failed for %s: %s", filepath, e)
        return False


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
        logger.warning("mutagen not installed — FLAC metadata injection skipped. "
                       "Install with: pip install mutagen")
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
        logger.warning("mutagen not installed — Opus metadata injection skipped. "
                       "Install with: pip install mutagen")
        return False
    except Exception as e:
        logger.warning("Opus metadata injection failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Voice-cloning consent gate
# ---------------------------------------------------------------------------

# Handler keys whose every model clones a voice from a reference recording.
#
# These are the ``handler_function_key`` values used in ``config.py`` — not the
# ``synthesize_with_*`` function names. An earlier version of this set listed the
# function names, which match no config entry, so the whole tier was dead and the
# gate silently fell through to keyword matching. ``tests/test_watermark.py``
# now asserts that every entry here corresponds to a real handler key.
#
# Handlers serving both cloning and fixed-speaker models (``coqui_tts``,
# ``mlx_audio``, ``crispasr``, ``tts_cpp``) deliberately do NOT belong here:
# they are resolved per-model by the explicit ``voice_cloning`` config key.
VOICE_CLONING_HANDLER_KEYS = frozenset({
    "outetts",
    "zonos",
    "f5_tts",
    "chatterbox",
    "llasa_hybrid",
    "llasa_german_transformers",
    "llasa_multilingual_transformers",
})

VOICE_CLONING_MODEL_KEYWORDS = frozenset({
    "zeroshot", "xtts", "clone", "f5_tts", "zonos", "chatterbox",
    "vibevoice", "indextts", "voxcpm2", "qwen3_tts",
    "dots_tts", "dots-tts", "cosyvoice3", "csm_tts", "csm-tts", "tada",
    "omnivoice",
})


#: Extensions that identify a reference *recording* supplied as the voice.
#: Restricting this to ``.wav`` used to let an identical clone from an MP3 or
#: FLAC reference through the gate untouched.
REFERENCE_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"})


def _is_reference_recording(voice_id: str | None) -> bool:
    """True if ``voice_id`` names an audio file rather than a preset voice."""
    if not voice_id or not isinstance(voice_id, str):
        return False
    return os.path.splitext(voice_id)[1].lower() in REFERENCE_AUDIO_EXTS


def requires_consent(model_id: str, handler_key: str, voice_id: str | None = None,
                     model_config: dict | None = None) -> bool:
    """Check whether a model/handler involves voice cloning.

    Detection order — strongest signal first:

      1. A reference *recording* supplied as ``voice_id``. Handing the system
         somebody's voice to imitate is the cloning act itself, so this is
         checked before anything else and is **not** overridable by config: a
         model declared ``voice_cloning: false`` still needs consent when it is
         driven from a reference clip.
      2. An explicit ``voice_cloning`` key in the model's config entry. Every
         entry in ``config.py`` sets this; it is the authoritative answer for a
         model's default mode, in both directions.
      3. The handler-key and model-ID keyword lists, as a fallback for configs
         that predate the explicit key (e.g. a user's own model dict).

    (3) fails *open* for a backend whose name matches no keyword, which is why
    (2) exists and why ``tests/test_watermark.py`` asserts that every shipped
    model declares it.
    """
    if _is_reference_recording(voice_id):
        return True
    if model_config is not None and model_config.get("voice_cloning") is not None:
        return bool(model_config["voice_cloning"])
    if handler_key in VOICE_CLONING_HANDLER_KEYS:
        return True
    model_lower = model_id.lower()
    return any(kw in model_lower for kw in VOICE_CLONING_MODEL_KEYWORDS)


_CONSENT_LOG_PATH = os.path.join(os.path.expanduser("~"), ".cache", "crisptts", "consent_audit.log")


def _reference_audio_digest(voice_id: str | None) -> str | None:
    """SHA-256 of a reference voice sample, for the consent audit trail.

    The attestation itself is unverifiable self-declaration; recording a
    digest of the exact reference audio at least makes the log evidential —
    it ties an attestation to the specific recording that was cloned.
    """
    if not voice_id or not isinstance(voice_id, str):
        return None
    if not os.path.isfile(voice_id):
        return None
    try:
        import hashlib
        digest = hashlib.sha256()
        with open(voice_id, "rb") as f_ref:
            for chunk in iter(lambda: f_ref.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:32]
    except OSError:
        return None


def log_consent_attestation(
    model_id: str,
    voice_id: str | None = None,
    source: str = "CLI --i-have-rights flag",
) -> None:
    """Log a consent attestation to stderr AND a persistent audit log file.

    Format matches CrispASR: [CONSENT] ts=ISO8601 model=X voice=Y attestation="..."
    plus a ref_sha256 of the reference recording when one was supplied.

    The persistent log at ~/.cache/crisptts/consent_audit.log ensures the
    audit trail survives even when stderr is not captured.
    """
    import sys
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    voice_str = voice_id or "default"
    ref_digest = _reference_audio_digest(voice_id)
    ref_field = f" ref_sha256={ref_digest}" if ref_digest else ""
    msg = (f'[CONSENT] ts={ts} model={model_id} voice={voice_str}{ref_field} '
           f'attestation="{source}"\n')
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

#: Spoken disclosure text per language. Art. 50(4) asks the deployer to
#: disclose that content is artificially generated; a disclosure the audience
#: cannot understand does not do that, so the language follows the synthesized
#: audio rather than being fixed to English.
#:
#: The ``de`` and ``en`` strings are kept identical to Susurrus's
#: ``disclosure.spoken`` (``utils/translations/{de,en}.py``) so the two projects
#: disclose in the same words. "The following audio" rather than "this audio":
#: the disclosure is prepended, so it describes what comes after it.
DISCLAIMER_TEXTS = {
    "de": "Die folgende Aufnahme wurde von künstlicher Intelligenz erzeugt.",
    "en": "The following audio was generated by artificial intelligence.",
    "fr": "L'enregistrement suivant a été généré par une intelligence artificielle.",
    "es": "El siguiente audio fue generado por inteligencia artificial.",
    "it": "Il seguente audio è stato generato dall'intelligenza artificiale.",
    "nl": "De volgende audio is gegenereerd door kunstmatige intelligentie.",
    "pl": "Poniższe nagranie zostało wygenerowane przez sztuczną inteligencję.",
    "pt": "O seguinte áudio foi gerado por inteligência artificial.",
}

#: CrispTTS is a German TTS toolkit, so German is the default disclosure
#: language — not English.
DEFAULT_DISCLAIMER_LANG = "de"

#: Edge TTS voice used per language when CrispASR is unavailable.
_DISCLAIMER_EDGE_VOICES = {
    "de": "de-DE-KatjaNeural",
    "en": "en-US-AriaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "pt": "pt-PT-RaquelNeural",
}

_DISCLAIMER_SILENCE_SEC = 0.3  # 300ms gap between disclaimer and content


class DisclosureError(RuntimeError):
    """Raised when the spoken AI disclosure could not be added to cloned audio.

    Treated like :class:`MarkingError`: voice-cloned output without its
    disclosure is not delivered. The escape hatch is
    ``--no-spoken-disclaimer``, which requires
    ``--accept-marking-responsibility``.
    """


def normalize_disclaimer_lang(language: str | None) -> str:
    """Map a config language code (``de``, ``de-DE``, ``german``) to a key."""
    if not language or not isinstance(language, str):
        return DEFAULT_DISCLAIMER_LANG
    lang = language.strip().lower().replace("_", "-")
    aliases = {"german": "de", "english": "en", "french": "fr", "spanish": "es",
               "italian": "it", "dutch": "nl", "polish": "pl", "portuguese": "pt"}
    if lang in aliases:
        return aliases[lang]
    base = lang.split("-")[0]
    return base if base in DISCLAIMER_TEXTS else DEFAULT_DISCLAIMER_LANG


def disclaimer_text(language: str | None = None) -> str:
    """The spoken disclosure sentence for a language."""
    return DISCLAIMER_TEXTS[normalize_disclaimer_lang(language)]


#: Backwards-compatible alias. Prefer :func:`disclaimer_text`.
DISCLAIMER_TEXT = DISCLAIMER_TEXTS["en"]


def bundled_disclosure_path(language: str | None = None) -> str | None:
    """Path to the pre-rendered disclosure clip for a language, if bundled."""
    lang = normalize_disclaimer_lang(language)
    try:
        from importlib.resources import files
        resource = files("crisptts_assets").joinpath(f"disclosure_{lang}.flac")
        if resource.is_file():
            return str(resource)
    except (ImportError, ModuleNotFoundError, TypeError, FileNotFoundError):
        pass
    # Source checkout without the package installed.
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "crisptts_assets", f"disclosure_{lang}.flac")
    return local if os.path.isfile(local) else None


def _load_bundled_disclosure(sample_rate: int, language: str) -> np.ndarray | None:
    """Decode the bundled disclosure clip, resampled to ``sample_rate``."""
    path = bundled_disclosure_path(language)
    if not path:
        return None
    try:
        import soundfile as sf_bundled
        data, rate = sf_bundled.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if rate != sample_rate:
            data = _resample_linear(data, rate, sample_rate)
        return data if len(data) else None
    except Exception as e:
        logger.debug("Could not load the bundled disclosure clip %s: %s", path, e)
        return None


def generate_spoken_disclaimer(sample_rate: int = 24000,
                               language: str | None = None) -> tuple[np.ndarray | None, str]:
    """Synthesize the spoken AI disclosure with a non-cloning TTS backend.

    Priority: CrispASR kokoro (local, fast) > Edge TTS (cloud) > tone marker.

    Returns:
        ``(pcm, kind)`` where ``kind`` is ``"spoken"`` for real speech or
        ``"tone-marker"`` for the beep fallback. The distinction matters:
        three beeps are an audible marker, not a disclosure a listener can
        understand, so callers must not present them as one. ``(None, "none")``
        if nothing could be produced.
    """
    lang = normalize_disclaimer_lang(language)
    text = DISCLAIMER_TEXTS[lang]

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
                     "--tts", text, "--tts-output", tmp_wav,
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
                    logger.info("Spoken disclaimer (%s) generated via CrispASR kokoro.", lang)
                    return data, "spoken"
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)
    except Exception as e:
        logger.info("CrispASR disclaimer generation failed: %s", e)

    # Try edge-tts (cloud, lightweight, no voice cloning concerns)
    try:
        import asyncio
        import tempfile

        import edge_tts

        async def _synth():
            voice = _DISCLAIMER_EDGE_VOICES.get(lang, _DISCLAIMER_EDGE_VOICES["en"])
            communicate = edge_tts.Communicate(text, voice)
            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                await communicate.save(tmp)
                try:
                    import soundfile as sf_disc
                    data, sr = sf_disc.read(tmp, dtype="float32")
                    if data.ndim > 1:
                        data = data[:, 0]
                    if sr != sample_rate:
                        data = _resample_linear(data, sr, sample_rate)
                    return data
                except ImportError:
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(tmp)
                    seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
                    return np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

        loop = asyncio.new_event_loop()
        try:
            data = loop.run_until_complete(_synth())
        finally:
            loop.close()
        if data is not None and len(data):
            logger.info("Spoken disclaimer (%s) generated via Edge TTS.", lang)
            return data, "spoken"
    except Exception as e:
        logger.info("Edge TTS disclaimer generation failed: %s", e)

    # Bundled pre-rendered clip. This is what makes the disclosure work with no
    # TTS backend, no model download and no network — the one configuration in
    # which Art. 50(4) disclosure would otherwise fail and the cloned output
    # would have to be discarded. It is a real spoken disclosure in the right
    # language, so it counts as one.
    bundled = _load_bundled_disclosure(sample_rate, lang)
    if bundled is not None:
        logger.info("Spoken disclaimer (%s) taken from the bundled clip.", lang)
        return bundled, "spoken"

    # Last resort: an audible tone marker. NOT a disclosure — it signals that
    # something precedes the content, but conveys nothing to a listener who
    # does not already know the convention. Reported as such so callers can
    # refuse to treat it as an Art. 50(4) disclosure.
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
        logger.warning("No TTS backend available for the spoken AI disclosure; "
                       "falling back to a tone marker, which is NOT a disclosure "
                       "a listener can understand. Install edge-tts, or run with "
                       "CrispASR available, for a real spoken disclosure.")
        return marker, "tone-marker"
    except Exception as e:
        logger.warning("Disclaimer generation failed entirely: %s", e)
        return None, "none"


# Cache the disclaimer audio to avoid re-synthesizing. Keyed by (rate, lang).
_disclaimer_cache: dict[tuple[int, str], tuple[np.ndarray, str]] = {}


def prepend_disclaimer(pcm: np.ndarray, sample_rate: int = 24000,
                       language: str | None = None) -> tuple[np.ndarray, str]:
    """Prepend the spoken AI disclosure to voice-cloned audio.

    Layout: disclosure + 300 ms silence + original audio. The generated
    disclosure is cached per (sample rate, language).

    Returns:
        ``(pcm, kind)`` — see :func:`generate_spoken_disclaimer` for ``kind``.
        On failure the original PCM is returned unchanged with ``"none"``.
    """
    lang = normalize_disclaimer_lang(language)
    key = (sample_rate, lang)
    if key not in _disclaimer_cache:
        disclaimer, kind = generate_spoken_disclaimer(sample_rate, lang)
        if disclaimer is None:
            return pcm, "none"
        _disclaimer_cache[key] = (disclaimer, kind)

    disclaimer, kind = _disclaimer_cache[key]
    silence = np.zeros(int(sample_rate * _DISCLAIMER_SILENCE_SEC), dtype=np.float32)
    return np.concatenate([disclaimer, silence, pcm]), kind


def prepend_disclaimer_file(filepath: str, language: str | None = None,
                            require_spoken: bool = True) -> str:
    """Prepend the spoken AI disclosure to an audio file, in place.

    Format-agnostic: works for every container the marking pipeline supports.

    Supports the deployer's Art. 50(4) duty to disclose deepfake content. That
    obligation is theirs, but voice-cloned output carries the disclosure by
    default so it is present unless deliberately removed — which is why this
    raises rather than returning a value callers can ignore.

    Args:
        filepath: Audio file to modify in place.
        language: Language of the synthesized speech; the disclosure follows it.
        require_spoken: When True, a tone-marker fallback is not accepted as a
            disclosure and raises instead.

    Returns:
        The disclosure kind that was applied: ``"spoken"`` or ``"tone-marker"``.

    Raises:
        DisclosureError: If no disclosure could be added, or only a tone marker
            could be and ``require_spoken`` is set.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in _SOUNDFILE_EXTS and ext not in _PYDUB_EXTS:
        raise DisclosureError(
            f"Cannot prepend the spoken AI disclosure to '{ext}' output. "
            f"Use one of: {', '.join(sorted(_SOUNDFILE_EXTS | _PYDUB_EXTS))}.")
    try:
        pcm, sr = _read_pcm_any(filepath, ext)
        combined, kind = prepend_disclaimer(pcm, sample_rate=sr, language=language)
    except DisclosureError:
        raise
    except Exception as e:
        raise DisclosureError(
            f"Could not add the spoken AI disclosure to {filepath}: {e}") from e

    if kind == "none" or len(combined) == len(pcm):
        raise DisclosureError(
            f"The spoken AI disclosure could not be generated for {filepath}. "
            "Voice-cloned audio is not delivered without it — install edge-tts "
            "for a local-network-free fallback, or pass --no-spoken-disclaimer "
            "--accept-marking-responsibility to take on the disclosure duty.")
    if kind == "tone-marker" and require_spoken:
        raise DisclosureError(
            f"Only a tone marker could be produced for {filepath}, which is not "
            "a disclosure a listener can understand. Install edge-tts (or make "
            "CrispASR available) for a real spoken disclosure, or pass "
            "--no-spoken-disclaimer --accept-marking-responsibility.")

    try:
        _write_pcm_any(filepath, combined, sr, ext)
    except Exception as e:
        raise DisclosureError(
            f"Could not write the disclosed audio for {filepath}: {e}") from e
    logger.info("AI disclosure (%s, %s) prepended to voice-cloned output: %s",
                kind, normalize_disclaimer_lang(language), filepath)
    return kind


# ---------------------------------------------------------------------------
# C2PA content credentials (optional, pip install c2pa-python)
# ---------------------------------------------------------------------------

def _c2pa_manifest(model_id: str | None = None) -> dict:
    """Build the C2PA manifest asserting this audio is AI-generated.

    ``digitalSourceType: trainedAlgorithmicMedia`` is the assertion that makes
    the manifest an AI-provenance claim rather than a bare integrity seal, so
    it is what Art. 50(2) marking leans on. It must be attached on *every*
    signing path — an earlier version built this only for one of two code
    paths, and the other signed files with no AI assertion at all.
    """
    software_agent: dict = {"name": "CrispTTS", "version": _crisptts_version()}
    if model_id:
        software_agent["softwareAgentModel"] = model_id
    return {
        "claim_generator_info": [{"name": "CrispTTS", "version": _crisptts_version()}],
        "title": "AI-generated speech",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType":
                                "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                "trainedAlgorithmicMedia",
                            "softwareAgent": software_agent,
                        }
                    ]
                },
            },
            {
                "label": "c2pa.training-mining",
                "data": {
                    "entries": {
                        "c2pa.ai_generative_training": {"use": "notAllowed"},
                        "c2pa.ai_inference": {"use": "notAllowed"},
                        "c2pa.ai_training": {"use": "notAllowed"},
                        "c2pa.data_mining": {"use": "notAllowed"},
                    }
                },
            },
        ],
    }


def _crisptts_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("crisptts")
        except PackageNotFoundError:
            return "0.0.0+source"
    except ImportError:
        return "0.0.0+source"


def _c2pa_signer(cert_pem: bytes, key_pem: bytes):
    """Build a c2pa ``Signer`` from PEM bytes.

    Works around two undocumented requirements of c2pa-python that silently
    produce useless errors otherwise:

      * ``ta_url`` must be a real NULL pointer. The wrapper's ``__init__``
        rejects ``None`` and an empty ``b""`` is parsed as a timestamp-authority
        URL, failing with "Signature: empty string" — so the ctypes struct is
        populated directly.
      * the private key must be PKCS#8 ("BEGIN PRIVATE KEY"), not SEC1
        ("BEGIN EC PRIVATE KEY").
    """
    import ctypes

    import c2pa as c2pa_rs

    info = ctypes.Structure.__new__(c2pa_rs.C2paSignerInfo)
    ctypes.Structure.__init__(info, b"es256", cert_pem, key_pem, None)
    return c2pa_rs.Signer.from_info(info)


def c2pa_sign_file(
    input_path: str,
    output_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
) -> bool:
    """Sign an audio file with C2PA content credentials.

    Thin boolean wrapper around :func:`c2pa_sign_file_ex`.
    """
    ok, _signer = c2pa_sign_file_ex(input_path, output_path, cert_path, key_path)
    return ok


#: Backend selection for C2PA signing, via ``CRISPTTS_C2PA_BACKEND``:
#: ``auto`` (default), ``python``, ``audio``, ``crispasr``, or ``off``.
_C2PA_BACKENDS = ("auto", "python", "audio", "off")

#: IPTC digital-source-type that marks content as AI-generated. Its presence
#: is what makes a manifest an Art. 50(2) provenance claim rather than a plain
#: integrity seal, so every signing path is checked for it.
AI_DIGITAL_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")


def _c2pa_backend_preference() -> str:
    value = (os.environ.get("CRISPTTS_C2PA_BACKEND") or "auto").strip().lower()
    if value not in _C2PA_BACKENDS:
        logger.warning("Unknown CRISPTTS_C2PA_BACKEND=%r; using 'auto'. Valid: %s",
                       value, ", ".join(_C2PA_BACKENDS))
        return "auto"
    return value


def manifest_asserts_ai(filepath: str) -> bool | None:
    """Does the manifest embedded in ``filepath`` claim AI generation?

    Returns True/False when the manifest could be read, or None when it could
    not be (c2pa-python missing, no manifest, unparseable).

    This is the guard that lets native signers be used safely. A signer we do
    not hand a manifest to — c2pa-audio's ``sign_wav`` takes only a cert and a
    key — decides its own assertions, and a manifest without this claim marks
    the file as *unmodified* rather than as *AI-generated*. That is how the
    earlier c2pa-audio path came to sign files with no AI assertion at all. So
    rather than trusting any backend, we read the result back.
    """
    try:
        import json

        import c2pa as c2pa_rs
    except ImportError:
        return None
    try:
        report = json.loads(c2pa_rs.Reader(filepath).json())
        active = report["manifests"][report["active_manifest"]]
    except Exception:
        return None
    for assertion in active.get("assertions", []):
        if not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        for action in assertion.get("data", {}).get("actions", []):
            if action.get("digitalSourceType") == AI_DIGITAL_SOURCE_TYPE:
                return True
    return False


def _sign_with_c2pa_audio(input_path: str, output_path: str,
                          cert_pem: str | None, key_pem: str | None) -> bool:
    """Sign via the native c2pa-audio library, if it is importable.

    Not on PyPI — built from https://github.com/CrispStrobe/c2pa-audio — so
    this is a fast path when present, never a requirement. Two API shapes are
    in use across the Crisp projects, so both are attempted:
    ``sign_wav(data, cert_pem=, key_pem=)`` (Susurrus) and
    ``sign(data, mime, cert, key)`` (older CrispTTS).
    """
    try:
        from c2pa_audio import C2paAudio
    except (ImportError, OSError):
        return False

    try:
        signer = C2paAudio()
        with open(input_path, "rb") as f_in:
            data = f_in.read()
        ext = os.path.splitext(input_path)[1].lower()

        if hasattr(signer, "sign_wav") and ext == ".wav":
            signed = signer.sign_wav(data, cert_pem=cert_pem, key_pem=key_pem)
        elif hasattr(signer, "sign"):
            mime = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4"}.get(ext, "audio/wav")
            signed = signer.sign(data, mime, cert_pem, key_pem)
        else:
            return False

        if not signed:
            return False
        with open(output_path, "wb") as f_out:
            f_out.write(signed)
        return True
    except Exception as e:
        logger.debug("c2pa-audio signing failed for %s: %s", input_path, e)
        return False


# CrispASR is deliberately NOT a signing backend here.
#
# Probed against crispasr 0.8.25: its --c2pa-cert/--c2pa-key configure signing
# of its *own* synthesis output, and there is no flag that signs an existing
# file. An earlier version of this module probed --help for
# "--c2pa-sign-file|--c2pa-sign|--sign-c2pa|--c2pa" and would have matched the
# last of those as a substring of "--c2pa-cert", then built a command the
# binary does not accept.
#
# What CrispASR *does* provide is better than a signing backend: it signs
# during synthesis, by default, with a manifest that already asserts
# trainedAlgorithmicMedia. mark_audio_file() detects that manifest and
# preserves it rather than overwriting it — see _preserve_existing_manifest.


def _sign_with_c2pa_python(input_path: str, output_path: str,
                           cert_pem: bytes, key_pem: bytes,
                           model_id: str | None) -> bool:
    """Sign via c2pa-python, the one path where we control the manifest."""
    try:
        import c2pa as c2pa_rs

        signer = _c2pa_signer(cert_pem, key_pem)
        builder = c2pa_rs.Builder(_c2pa_manifest(model_id))

        if output_path == input_path:
            # sign_file refuses to write over its own source.
            import tempfile
            suffix = os.path.splitext(input_path)[1]
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            try:
                os.unlink(tmp_path)  # must not exist
                builder.sign_file(input_path, tmp_path, signer)
                import shutil
                shutil.move(tmp_path, input_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        else:
            builder.sign_file(input_path, output_path, signer)
        return True
    except ImportError:
        logger.debug("c2pa-python not installed; C2PA signing skipped.")
        return False
    except Exception as e:
        logger.warning("C2PA signing failed for %s: %s", input_path, e)
        return False


def c2pa_sign_file_ex(
    input_path: str,
    output_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
    model_id: str | None = None,
) -> tuple[bool, str | None]:
    """Sign an audio file with C2PA content credentials, reporting the signer.

    Signing is attempted for every C2PA-capable container. When no certificate
    is supplied the bundled development credential is used, so a default
    install produces an interoperable provenance manifest rather than nothing.

    Backends are tried in order, and **every native result is verified**: if
    the manifest a native signer produced does not actually assert AI
    generation, it is discarded and c2pa-python re-signs with a manifest that
    does. A signer that does not take a manifest cannot be taken on trust.

      1. c2pa-audio, if importable (native, not on PyPI)
      2. c2pa-python — always available, and the only path where CrispTTS
         controls the manifest contents

    Selectable with ``CRISPTTS_C2PA_BACKEND=auto|python|audio|off``.

    Args:
        input_path: Path to the audio file (WAV or MP3).
        output_path: Where to write the signed file (defaults to overwrite).
        cert_path: PEM certificate — leaf first, then its CA chain.
        key_path: PKCS#8 PEM private key for the leaf certificate.
        model_id: TTS model that produced the audio, recorded in the manifest.

    Returns:
        ``(success, signer_kind)`` where ``signer_kind`` is ``"ca-issued"``
        when an explicit certificate was supplied, ``"self-signed"`` when the
        bundled development certificate was used, or ``None`` on failure.

        A ``"self-signed"`` manifest proves the file has not been altered
        since signing, but will NOT validate against C2PA trust lists — it
        is not equivalent to a credential from a recognised authority.
    """
    backend = _c2pa_backend_preference()
    if backend == "off":
        logger.debug("C2PA signing disabled via CRISPTTS_C2PA_BACKEND=off.")
        return False, None

    cert_path = cert_path or os.environ.get("C2PA_CERT_PATH")
    key_path = key_path or os.environ.get("C2PA_KEY_PATH")
    effective_output = output_path or input_path

    if cert_path and key_path:
        try:
            with open(cert_path, "rb") as f_cert:
                cert_pem = f_cert.read()
            with open(key_path, "rb") as f_key:
                key_pem = f_key.read()
            signer_kind = "ca-issued"
        except OSError as e:
            logger.warning("Could not read C2PA credential (%s); C2PA signing skipped.", e)
            return False, None
    else:
        try:
            from c2pa_dev_cert import DEV_CERT_CHAIN_PEM, DEV_PRIVATE_KEY_PEM
        except ImportError:
            logger.debug("Bundled C2PA development credential unavailable.")
            return False, None
        cert_pem = DEV_CERT_CHAIN_PEM.encode()
        key_pem = DEV_PRIVATE_KEY_PEM.encode()
        signer_kind = "self-signed"

    # --- Native fast paths, each verified before being accepted ---
    native = []
    if backend in ("auto", "audio"):
        native.append(("c2pa-audio", lambda: _sign_with_c2pa_audio(
            input_path, effective_output,
            cert_pem.decode(errors="replace"), key_pem.decode(errors="replace"))))
    for name, attempt in native:
        if not attempt():
            continue
        asserts_ai = manifest_asserts_ai(effective_output)
        if asserts_ai:
            logger.info("C2PA signed via %s (%s): %s", name, signer_kind, effective_output)
            if signer_kind == "self-signed":
                _warn_self_signed_once()
            return True, signer_kind
        logger.warning(
            "%s produced a C2PA manifest that does not assert AI generation "
            "(%s); discarding it and re-signing with c2pa-python so the "
            "output carries a real provenance claim.",
            name, "no AI assertion" if asserts_ai is False else "manifest unreadable")
        break  # fall through to the path where we control the manifest

    if backend == "audio":
        logger.warning("CRISPTTS_C2PA_BACKEND=audio requested but c2pa-audio could not "
                       "sign %s; falling back to c2pa-python.", input_path)

    # --- c2pa-python: the manifest is ours, so the AI assertion is certain ---
    if not _sign_with_c2pa_python(input_path, effective_output, cert_pem, key_pem, model_id):
        return False, None

    if signer_kind == "self-signed":
        _warn_self_signed_once()
        logger.info("C2PA signed with the bundled development certificate: %s — the "
                    "manifest proves the file is unaltered but will not validate "
                    "against C2PA trust lists.", effective_output)
    else:
        logger.info("C2PA signed with the supplied certificate: %s", effective_output)
    return True, signer_kind


_self_signed_warned = False


def _warn_self_signed_once() -> None:
    """Say once per run that the bundled credential is not a trusted one."""
    global _self_signed_warned
    if _self_signed_warned:
        return
    _self_signed_warned = True
    logger.warning(
        "C2PA manifests are being signed with the bundled development "
        "certificate, whose private key is public. They prove integrity, not "
        "authorship. Pass --c2pa-cert/--c2pa-key for a credential others can "
        "attribute to you."
    )


# ---------------------------------------------------------------------------
# Central marking entry point (EU AI Act Art. 50(2))
# ---------------------------------------------------------------------------
#
# Every code path that produces an audio file — CLI single synthesis,
# --test-all, batch mode, and the HTTP server — marks its output through
# mark_audio_file() and nothing else. Having exactly one implementation is
# what keeps coverage uniform across formats and prevents the same file from
# being watermarked twice.


class MarkingError(RuntimeError):
    """Raised when AI-provenance marking could not be applied to an output.

    Callers must treat this as fatal for the output in question: under
    Art. 50(2) an unmarked synthetic-audio file should not be delivered.
    The only intended escape hatch is ``allow_unmarked=True``, which the CLI
    exposes as ``--allow-unmarked`` and the environment as
    ``CRISPTTS_ALLOW_UNMARKED=1``.
    """


# ---------------------------------------------------------------------------
# Marking-sufficiency policy (ported from CrispASR's watertight-CLI guarantee)
# ---------------------------------------------------------------------------
#
# Two ideas taken from the C++ CLI (examples/cli/crispasr_run.cpp):
#
#   1. Watermark floor. If the output container cannot carry a C2PA manifest,
#      the audio watermark is the ONLY robust machine-readable mark, so
#      --no-watermark is overridden rather than honoured. No path can emit a
#      fully unmarked AI file.
#
#   2. Marking attestation. Any provenance opt-out requires an explicit
#      --accept-marking-responsibility, mirroring the voice-clone
#      --i-have-rights gate, and is recorded as a [MARKING] audit line.
#
# CrispTTS adds a third, which neither sibling has: marking is *verified* and
# generation is gated on the verification, not merely warned about. Metadata
# alone never counts as sufficient — it is strippable by any transcode.

#: Containers into which c2pa-python can actually *embed* a signed manifest.
#:
#: Deliberately narrower than ``Builder.get_supported_mime_types()``, which
#: also lists FLAC and M4A — both are accepted for reading but fail signing
#: with "NotSupported: type is unsupported". This set feeds the watermark
#: floor, so an over-broad entry is not cosmetic: it would let
#: ``--no-watermark`` be honoured for a container that then carries no
#: manifest, leaving only strippable metadata. ``.m4a`` was listed here for
#: exactly that reason and is now removed.
#: ``tests/test_watermark.py`` verifies each entry by really signing a file.
C2PA_CAPABLE_EXTS = frozenset({".wav", ".mp3"})


def c2pa_available(cert_path: str | None = None, key_path: str | None = None) -> bool:
    """True if C2PA signing can run in this environment.

    Signing no longer depends on the caller supplying a credential: without
    one, the bundled development certificate is used. So this reduces to
    "is c2pa-python importable", which for a normal install is always true —
    c2pa-python is a core dependency precisely so that the interoperable
    provenance layer is on by default rather than opt-in.
    """
    del cert_path, key_path  # signing works with the bundled credential too
    try:
        import c2pa  # noqa: F401
    except ImportError:
        return False
    try:
        from c2pa_dev_cert import DEV_CERT_CHAIN_PEM  # noqa: F401
    except ImportError:
        # No bundled credential: signing only possible with an explicit one.
        return bool((os.environ.get("C2PA_CERT_PATH") and os.environ.get("C2PA_KEY_PATH")))
    return True


def output_carries_c2pa(filepath: str | None, cert_path: str | None = None,
                        key_path: str | None = None) -> bool:
    """True if this output will carry a C2PA manifest.

    Mirrors ``crispasr_output_carries_c2pa``. Used to decide whether the audio
    watermark may be opted out of: when this is False the watermark is the only
    robust mark left, so it becomes mandatory.
    """
    if not filepath:
        return False
    if os.path.splitext(filepath)[1].lower() not in C2PA_CAPABLE_EXTS:
        return False
    return c2pa_available(cert_path, key_path)


def log_marking_attestation(
    no_watermark: bool = False,
    allow_unmarked: bool = False,
    no_spoken_disclaimer: bool = False,
    source: str = "CLI --accept-marking-responsibility",
) -> None:
    """Record an honoured provenance opt-out as a [MARKING] audit line.

    Parallel to :func:`log_consent_attestation`'s [CONSENT] line, and to
    CrispASR's ``[MARKING]`` line, so the two ecosystems produce a compatible
    audit trail.
    """
    import sys
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    msg = (f'[MARKING] ts={ts} no_watermark={"yes" if no_watermark else "no"} '
           f'allow_unmarked={"yes" if allow_unmarked else "no"} '
           f'no_spoken_disclaimer={"yes" if no_spoken_disclaimer else "no"} '
           f'attestation="{source}"\n')
    sys.stderr.write(msg)
    sys.stderr.flush()
    try:
        os.makedirs(os.path.dirname(_CONSENT_LOG_PATH), exist_ok=True)
        with open(_CONSENT_LOG_PATH, "a") as f_audit:
            f_audit.write(msg)
    except OSError as e:
        logger.debug("Could not write marking audit log: %s", e)


def preflight_marking(
    output_path: str | None,
    *,
    handler_key: str | None = None,
    no_watermark: bool = False,
    allow_unmarked: bool = False,
    responsibility_accepted: bool = False,
    no_spoken_disclaimer: bool = False,
    c2pa_cert: str | None = None,
    c2pa_key: str | None = None,
) -> dict:
    """Decide, BEFORE synthesis, whether this output can be marked sufficiently.

    Refusing here rather than after generation means no model is loaded and no
    compute is spent producing audio we would have to throw away — and, more
    importantly, that unmarkable audio never exists on disk at all.

    Returns:
        A policy dict consumed by :func:`mark_audio_file`:
        ``embed_watermark``, ``forced``, ``expect_c2pa``, ``allow_unmarked``,
        ``note``.

    Raises:
        MarkingError: If generation must not proceed.
    """
    env_no_watermark = bool(os.environ.get("CRISPTTS_NO_WATERMARK"))
    env_allow_unmarked = bool(os.environ.get("CRISPTTS_ALLOW_UNMARKED"))
    env_accepted = bool(os.environ.get("CRISPTTS_ACCEPT_MARKING_RESPONSIBILITY"))

    no_watermark = no_watermark or env_no_watermark
    allow_unmarked = allow_unmarked or env_allow_unmarked
    responsibility_accepted = responsibility_accepted or env_accepted

    opt_out = no_watermark or allow_unmarked or no_spoken_disclaimer

    # --- Attestation gate: opting out of provenance is an explicit act ---
    if opt_out and not responsibility_accepted:
        which = ("--no-watermark" if no_watermark else
                 "--allow-unmarked" if allow_unmarked else
                 "--no-spoken-disclaimer")
        raise MarkingError(
            f"{which} requires --accept-marking-responsibility.\n"
            "  Disabling AI-content provenance marking shifts the marking and\n"
            "  disclosure duty to you, the operator. By passing\n"
            "  --accept-marking-responsibility you affirm you accept that\n"
            "  responsibility for this output."
        )

    expect_c2pa = output_carries_c2pa(output_path, c2pa_cert, c2pa_key)
    policy = {
        "embed_watermark": True,
        "forced": False,
        "expect_c2pa": expect_c2pa,
        "allow_unmarked": allow_unmarked,
        "note": None,
    }

    # --- Watermark floor: honour --no-watermark only if C2PA still marks it ---
    if no_watermark:
        if expect_c2pa:
            policy["embed_watermark"] = False
        else:
            policy["forced"] = True
            policy["note"] = (
                f"'{output_path or '<stream>'}' will not carry a C2PA manifest, so "
                "--no-watermark is overridden — the audio watermark is kept so the "
                "output stays marked as AI-generated. Use a C2PA-capable container "
                f"({'/'.join(sorted(e.lstrip('.').upper() for e in C2PA_CAPABLE_EXTS))}) "
                "to allow --no-watermark."
            )
            logger.warning("%s", policy["note"])

    if opt_out and responsibility_accepted:
        log_marking_attestation(no_watermark=no_watermark, allow_unmarked=allow_unmarked,
                                no_spoken_disclaimer=no_spoken_disclaimer)

    if allow_unmarked:
        return policy  # blunt override: caller has accepted responsibility

    # --- Refuse now if marking could not possibly succeed on this output ---
    if policy["embed_watermark"] and output_path and handler_key != "crispasr":
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in _SOUNDFILE_EXTS and ext not in _PYDUB_EXTS:
            raise MarkingError(
                f"Refusing to synthesize: output format '{ext or '(none)'}' cannot carry "
                "an audio watermark, so the result could not be marked as AI-generated. "
                f"Use one of: {', '.join(sorted(_SOUNDFILE_EXTS | _PYDUB_EXTS))}."
            )
        missing = _missing_codec_dependency(ext)
        if missing:
            raise MarkingError(
                f"Refusing to synthesize: {missing} is required to watermark '{ext}' output "
                "and is not installed, so the result could not be marked as AI-generated."
            )
    return policy


def _missing_codec_dependency(ext: str) -> str | None:
    """Name of the missing package needed to watermark `ext`, or None."""
    if ext in _SOUNDFILE_EXTS:
        try:
            import soundfile  # noqa: F401
        except ImportError:
            return "soundfile"
        return None
    try:
        import pydub  # noqa: F401
    except ImportError:
        return "pydub"
    return None


class MarkResult:
    """Outcome of marking one audio file.

    Attributes:
        marked: True only if an audio watermark layer was actually applied
            (or was already present). This is what provenance claims —
            such as the server's ``X-CrispTTS-Watermarked`` header — must
            be derived from.
        backend: Active watermark backend name.
        layers: Which provenance layers were applied.
        confidence: Post-embed detection confidence, when verification ran.
        c2pa_signer: ``"ca-issued"``, ``"self-signed"`` or None.
        reason: Why marking was skipped or degraded, if it was.
    """

    __slots__ = ("marked", "backend", "layers", "confidence", "c2pa_signer", "reason")

    def __init__(self, marked: bool, backend: str | None = None,
                 layers: tuple[str, ...] = (), confidence: float | None = None,
                 c2pa_signer: str | None = None, reason: str | None = None):
        self.marked = marked
        self.backend = backend
        self.layers = layers
        self.confidence = confidence
        self.c2pa_signer = c2pa_signer
        self.reason = reason

    def __repr__(self) -> str:
        return (f"MarkResult(marked={self.marked}, backend={self.backend!r}, "
                f"layers={self.layers!r}, confidence={self.confidence!r}, "
                f"c2pa_signer={self.c2pa_signer!r}, reason={self.reason!r})")


# Formats we can decode/re-encode for a real PCM watermark embed.
_SOUNDFILE_EXTS = frozenset({".wav", ".flac"})
_PYDUB_EXTS = frozenset({".mp3", ".opus", ".ogg", ".m4a"})

# Detection confidence below which we warn that the mark may not survive
# downstream processing. Matches the documented spread-spectrum threshold.
_VERIFY_THRESHOLD = 0.65

_weak_backend_warned = False
_unmarked_warned = False


def marking_enabled() -> bool:
    """False when marking has been disabled via CRISPTTS_NO_WATERMARK."""
    return not os.environ.get("CRISPTTS_NO_WATERMARK")


def allow_unmarked_default() -> bool:
    """True when the environment permits delivering unmarked output."""
    return bool(os.environ.get("CRISPTTS_ALLOW_UNMARKED"))


def _warn_if_weak_backend(has_c2pa: bool = False) -> None:
    """Warn once when only the built-in spread-spectrum watermark is active.

    On the speech-band comb the built-in mark now survives the transforms that
    used to defeat it — measured on 20 s of speech: 0.84 after embedding, 0.78
    after a 44.1k->16k->44.1k resample, 0.81 after an MP3 round-trip. It is
    still a fixed-key comb rather than a learned watermark, so a neural backend
    remains the stronger option under adversarial removal, and Art. 50(2) asks
    for marking robust "as far as technically feasible" — worth saying once.

    Args:
        has_c2pa: True when this output also carries a signed C2PA manifest.
            The manifest survives what the spread-spectrum mark does not, so
            the warning is softened — but not dropped, since a transcode that
            strips the manifest also leaves the weak watermark behind.
    """
    global _weak_backend_warned
    if _weak_backend_warned or _backend != "spread_spectrum":
        return
    _weak_backend_warned = True
    if has_c2pa:
        logger.debug(
            "Audio watermarking uses the built-in spread-spectrum backend "
            "alongside a signed C2PA manifest. For a learned watermark that is "
            "harder to remove deliberately: pip install 'crisptts[robust]'"
        )
        return
    logger.info(
        "Watermarking with the built-in spread-spectrum backend only; this "
        "container carries no C2PA manifest. The mark survives resampling and "
        "transcoding but is a fixed-key comb, so it is removable by someone who "
        "knows the scheme. For a learned neural watermark: "
        "pip install 'crisptts[robust]'"
    )


def _warn_unmarked_once() -> None:
    """Warn prominently, once, that output is being delivered unmarked."""
    global _unmarked_warned
    if _unmarked_warned:
        return
    _unmarked_warned = True
    import sys
    sys.stderr.write(
        "\n*** WARNING: AI watermarking is DISABLED. Synthetic audio will be "
        "produced WITHOUT machine-readable AI-provenance marking.\n"
        "*** Under EU AI Act Art. 50(2) marking is required for synthetic "
        "audio; responsibility for unmarked output rests with you.\n\n"
    )
    sys.stderr.flush()


def _read_pcm_any(filepath: str, ext: str) -> tuple[np.ndarray, int]:
    """Decode an audio file to (float32 mono PCM, true sample rate).

    Reading the real sample rate matters: the neural backends resample
    internally, so passing a wrong rate embeds the watermark at the wrong
    frequency and makes it undetectable.
    """
    if ext in _SOUNDFILE_EXTS:
        import soundfile as sf_read
        data, sr = sf_read.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return data, int(sr)

    from pydub import AudioSegment as _Seg
    seg = _Seg.from_file(filepath)
    seg = seg.set_channels(1).set_sample_width(2)
    pcm = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    return pcm, int(seg.frame_rate)


def _write_pcm_any(filepath: str, pcm: np.ndarray, sample_rate: int, ext: str) -> None:
    """Re-encode float32 mono PCM back into the file's original container."""
    if ext in _SOUNDFILE_EXTS:
        import soundfile as sf_write
        subtype = "PCM_16" if ext == ".wav" else None
        sf_write.write(filepath, pcm, sample_rate, subtype=subtype)
        return

    from pydub import AudioSegment as _Seg
    raw = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
    seg = _Seg(data=raw, sample_width=2, frame_rate=sample_rate, channels=1)
    export_format = {".m4a": "ipod", ".opus": "opus", ".ogg": "ogg", ".mp3": "mp3"}[ext]
    seg.export(filepath, format=export_format)


def _inject_container_metadata(filepath: str, ext: str) -> bool:
    """Inject AI-provenance metadata appropriate to the container. Best-effort."""
    if ext == ".wav":
        with open(filepath, "rb") as f_meta:
            patched = inject_wav_metadata(f_meta.read())
        with open(filepath, "wb") as f_meta:
            f_meta.write(patched)
        return True
    if ext == ".mp3":
        return inject_mp3_metadata_file(filepath)
    if ext == ".flac":
        return inject_flac_metadata(filepath)
    if ext in (".opus", ".ogg"):
        return inject_opus_metadata(filepath)
    return False


def mark_audio_file(
    filepath: str,
    *,
    handler_key: str | None = None,
    allow_unmarked: bool | None = None,
    c2pa_cert: str | None = None,
    c2pa_key: str | None = None,
    verify: bool = True,
    policy: dict | None = None,
    model_id: str | None = None,
) -> MarkResult:
    """Apply AI-provenance marking to a synthesized audio file, in place.

    This is the single marking path for the whole project. It embeds the
    audio watermark at the file's true sample rate, injects container
    metadata, optionally attaches C2PA content credentials, and verifies
    that the result is detectable.

    Idempotent: a file that already carries the CrispTTS marker is left
    untouched rather than watermarked a second time (double embedding costs
    roughly 6 dB of SNR and adds no provenance).

    Args:
        filepath: Audio file to mark in place.
        handler_key: Originating handler. ``"crispasr"`` outputs are already
            watermarked by that binary, so the PCM embed is skipped while
            metadata and verification still run.
        allow_unmarked: When True, marking failures are downgraded to a
            warning and reported via the result instead of raising. Defaults
            to the ``CRISPTTS_ALLOW_UNMARKED`` environment variable.
        c2pa_cert: PEM certificate for C2PA signing.
        c2pa_key: PEM private key for C2PA signing.
        verify: Read the file back and confirm the watermark is detectable.
            When True the verification is a **gate**, not a warning: an output
            whose mark cannot be detected above the threshold is refused unless
            some other robust layer (C2PA) marked it.
        policy: Resolved policy from :func:`preflight_marking`. Supplying it
            enables the watermark floor — an opt-out that preflight overrode
            stays overridden here, so the env var cannot re-disable it.

    Returns:
        A :class:`MarkResult` describing what was applied.

    Raises:
        MarkingError: If marking failed, or succeeded but is not sufficient,
            and unmarked output is not allowed.
    """
    forced = bool(policy and policy.get("forced"))
    skip_watermark_layer = bool(policy and not policy.get("embed_watermark", True))
    if allow_unmarked is None:
        allow_unmarked = bool(policy["allow_unmarked"]) if policy else allow_unmarked_default()

    def _fail(reason: str) -> MarkResult:
        if allow_unmarked:
            logger.warning("Delivering UNMARKED audio for %s: %s", filepath, reason)
            return MarkResult(marked=False, backend=_backend, reason=reason)
        raise MarkingError(
            f"Could not apply AI-provenance marking to {filepath}: {reason}. "
            "Refusing to deliver unmarked synthetic audio. Pass --allow-unmarked "
            "(or set CRISPTTS_ALLOW_UNMARKED=1) to override."
        )

    if not filepath or not os.path.isfile(filepath):
        return _fail("output file does not exist")

    # The floor wins over the env opt-out: if preflight determined this output
    # has no other robust mark, the watermark is not optional here either.
    if not marking_enabled() and not forced:
        _warn_unmarked_once()
        return MarkResult(marked=False, backend=_backend, reason="disabled via CRISPTTS_NO_WATERMARK")

    if is_marked(filepath):
        logger.debug("Already marked, skipping re-embed: %s", filepath)
        return MarkResult(marked=True, backend=_backend, layers=("already-marked",),
                          reason="already-marked")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in _SOUNDFILE_EXTS and ext not in _PYDUB_EXTS:
        return _fail(f"unsupported output format '{ext}' — cannot embed a watermark")

    layers: list[str] = []
    confidence: float | None = None
    sample_rate: int | None = None

    # --- Layer 0: an upstream manifest we must not destroy ---
    #
    # CrispASR signs its own TTS output during synthesis, by default, with a
    # manifest that already asserts trainedAlgorithmicMedia. Everything below
    # rewrites the file, and any rewrite breaks that manifest's hash — measured:
    # injecting the LIST/INFO chunk alone takes a CrispASR WAV from
    # validation_state "Valid" to "Invalid". Re-signing afterwards papered over
    # it, at the cost of discarding the upstream signer's identity and leaving
    # a *tamper-looking* file behind whenever the re-sign failed.
    #
    # So an existing AI-asserting manifest is preserved as-is: it is already the
    # strongest layer available, and container metadata is strippable anyway.
    preserved_manifest = manifest_asserts_ai(filepath) is True
    if preserved_manifest:
        logger.info("Preserving the existing C2PA manifest on %s — it already asserts "
                    "AI generation, and re-marking would invalidate it.", filepath)
        confidence = None
        if verify:
            try:
                check_pcm, check_sr = _read_pcm_any(filepath, ext)
                confidence = watermark_detect(check_pcm, sample_rate=check_sr)
            except Exception as e:
                logger.debug("Could not measure the watermark on %s: %s", filepath, e)
        watermark_layers: tuple[str, ...] = ()
        if confidence is not None and confidence >= _VERIFY_THRESHOLD:
            watermark_layers = ("audio-watermark:upstream",)
        return MarkResult(marked=True, backend=_backend,
                          layers=("c2pa:preserved", *watermark_layers),
                          confidence=confidence, c2pa_signer="preserved",
                          reason="upstream manifest preserved")

    # --- Layer 1: audio watermark (the layer that survives transcoding) ---
    if skip_watermark_layer:
        # --no-watermark honoured because C2PA will carry the provenance.
        logger.info("Audio watermark skipped for %s — C2PA manifest carries provenance.",
                    filepath)
    elif handler_key == "crispasr":
        # The CrispASR binary watermarks during synthesis — but that is a claim
        # about another program's behaviour, so it is not recorded as a layer
        # here. The verification below decides whether a watermark is really
        # present; measured on crispasr 0.8.25 kokoro output, CrispTTS's
        # detector reads 0.44 (its noise floor), so asserting the layer
        # unconditionally would have put a mark in MarkResult.layers that no
        # detector can find.
        logger.debug("Skipping the PCM embed for CrispASR output; its own watermark "
                     "is measured during verification rather than assumed.")
    else:
        try:
            pcm, sample_rate = _read_pcm_any(filepath, ext)
        except ImportError as e:
            return _fail(f"cannot decode {ext} for watermarking ({e}); "
                         "install soundfile (wav/flac) or pydub+ffmpeg (mp3/opus/ogg/m4a)")
        except Exception as e:
            return _fail(f"cannot decode {ext} for watermarking: {e}")

        _warn_if_weak_backend(
            has_c2pa=bool(policy["expect_c2pa"]) if policy and "expect_c2pa" in policy
            else output_carries_c2pa(filepath, c2pa_cert, c2pa_key))
        try:
            pcm = watermark_embed(pcm, sample_rate=sample_rate, force=forced)
            _write_pcm_any(filepath, pcm, sample_rate, ext)
        except Exception as e:
            return _fail(f"watermark embedding failed: {e}")
        layers.append("audio-watermark")

    # --- Layer 2: container metadata (best-effort; strippable by design) ---
    try:
        if _inject_container_metadata(filepath, ext):
            layers.append("metadata")
        else:
            logger.warning("Container metadata not injected for %s "
                           "(install mutagen for FLAC/Opus). Audio watermark is unaffected.",
                           filepath)
    except Exception as e:
        logger.warning("Container metadata injection failed for %s: %s", filepath, e)

    # --- Layer 3: C2PA content credentials (optional) ---
    c2pa_signer = None
    try:
        signed, c2pa_signer = c2pa_sign_file_ex(filepath, cert_path=c2pa_cert,
                                                key_path=c2pa_key, model_id=model_id)
        if signed:
            layers.append(f"c2pa:{c2pa_signer}")
    except Exception as e:
        logger.debug("C2PA signing skipped for %s: %s", filepath, e)

    # --- Verify, and GATE on the verification ---
    #
    # A watermark that cannot be read back is not a mark. The embed silently
    # no-ops on audio shorter than one FFT frame and on digital silence, and
    # can be wiped by aggressive post-processing — in all of those cases the
    # only thing left is strippable container metadata, which is not the
    # "machine-readable, detectable" marking Art. 50(2) asks for. So the
    # verification result decides whether this output may be delivered.
    verified = False
    if verify:
        try:
            check_pcm, check_sr = _read_pcm_any(filepath, ext)
            confidence = watermark_detect(check_pcm, sample_rate=check_sr)
            verified = confidence is not None and confidence >= _VERIFY_THRESHOLD
            if verified:
                logger.debug("Watermark verified for %s (confidence=%.3f).", filepath, confidence)
                if handler_key == "crispasr":
                    # Only now is the upstream watermark a fact rather than an
                    # assumption, so only now does it become a reported layer.
                    layers.append("audio-watermark:upstream")
            else:
                logger.warning(
                    "Watermark NOT detectable in %s (confidence=%.3f, threshold=%.2f).",
                    filepath, confidence if confidence is not None else -1.0, _VERIFY_THRESHOLD)
        except Exception as e:
            logger.warning("Watermark verification failed for %s: %s", filepath, e)

    # Sufficiency: at least one ROBUST layer must be present. Container
    # metadata alone never qualifies — any transcode removes it.
    if verify:
        robust = verified or (c2pa_signer is not None)
        if not robust:
            detail = (f"watermark not detectable after embedding "
                      f"(confidence={confidence:.3f}, threshold={_VERIFY_THRESHOLD:.2f})"
                      if confidence is not None else "watermark could not be verified")
            hint = ""
            if confidence is not None and confidence <= 0.55:
                hint = (" The audio may be too short (under ~0.25 s), silent, or otherwise "
                        "unable to carry a watermark.")
            return _fail(
                f"marking is not sufficient: {detail}. Container metadata alone is "
                f"strippable and does not satisfy machine-readable marking.{hint}"
            )

    logger.info("AI-provenance marking applied to %s (backend=%s, layers=%s, confidence=%s).",
                filepath, _backend, "+".join(layers),
                f"{confidence:.3f}" if confidence is not None else "n/a")
    return MarkResult(marked=True, backend=_backend, layers=tuple(layers),
                      confidence=confidence, c2pa_signer=c2pa_signer)
