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
        seed, self.s0 = _splitmix64(seed)
        _, self.s1 = _splitmix64(self.s0)

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

def spread_spectrum_embed(pcm: np.ndarray, alpha: float = 0.005) -> np.ndarray:
    """Embed a spread-spectrum watermark into float32 mono PCM.

    Args:
        pcm: 1-D float32 array of audio samples.
        alpha: Watermark strength (0.005 = ~-46 dB, imperceptible).

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
    n_frames = 0
    correlation = 0.0

    for start in range(0, n - _FFT_SIZE + 1, _HOP):
        frame = pcm[start:start + _FFT_SIZE] * window
        spectrum = np.fft.rfft(frame)
        mags = np.abs(spectrum[:_FFT_SIZE // 2]).astype(np.float64)

        for b_idx, b_sign in bins:
            if b_idx >= len(mags):
                continue
            # Local mean of ±2 neighbours (excluding self)
            neighbours = []
            for d in range(-2, 3):
                nb = b_idx + d
                if 1 <= nb < len(mags) and d != 0:
                    neighbours.append(mags[nb])
            if not neighbours:
                continue
            local_mean = sum(neighbours) / len(neighbours)
            if local_mean < 1e-12 and mags[b_idx] < 1e-12:
                continue
            ref = max(local_mean, 1e-12)
            delta = (mags[b_idx] - local_mean) / ref
            correlation += (1.0 if delta > 0 else -1.0) * b_sign
        n_frames += 1

    if n_frames == 0:
        return 0.0

    max_corr = n_frames * len(bins)
    score = (correlation / max_corr + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# AudioSeal dispatcher (multiple backends)
# ---------------------------------------------------------------------------

# Backend priority: audioseal (Python) > crispasr (C binding) > spread-spectrum
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


def _embed_audioseal_python(pcm: np.ndarray) -> np.ndarray:
    """Embed watermark using the audioseal Python package."""
    import torch
    tensor = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    watermark = _audioseal_generator.get_watermark(tensor, sample_rate=16000)
    result = tensor + watermark
    return result.squeeze().detach().numpy().astype(np.float32)


def _detect_audioseal_python(pcm: np.ndarray) -> float:
    """Detect watermark using the audioseal Python package."""
    import torch
    tensor = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    result, _ = _audioseal_detector.detect_watermark(tensor, sample_rate=16000)
    return float(result.mean().item())


def watermark_embed(pcm: np.ndarray, alpha: float = 0.005) -> np.ndarray:
    """Embed AI-generated watermark. Dispatches to the best available backend.

    Priority: audioseal (Python) > crispasr (C/GGUF) > spread-spectrum.

    Args:
        pcm: 1-D float32 mono PCM array.
        alpha: Strength for spread-spectrum (ignored when AudioSeal active).

    Returns:
        Watermarked PCM (new array, input unchanged).
    """
    if os.environ.get("CRISPTTS_NO_WATERMARK"):
        return pcm.copy()

    if _backend == "audioseal_python" and _audioseal_generator is not None:
        try:
            result = _embed_audioseal_python(pcm)
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


def watermark_detect(pcm: np.ndarray) -> float:
    """Detect AI-generated watermark. Returns confidence [0, 1]."""
    if _backend == "audioseal_python" and _audioseal_detector is not None:
        try:
            return _detect_audioseal_python(pcm)
        except Exception as e:
            logger.warning("AudioSeal Python detect failed, trying next backend: %s", e)

    if _backend == "audioseal_crispasr" and _crispasr_wm is not None:
        try:
            return _crispasr_wm.watermark_detect(pcm.astype(np.float32, copy=True))
        except Exception as e:
            logger.warning("AudioSeal crispasr detect failed, falling back to spread-spectrum: %s", e)

    return spread_spectrum_detect(pcm)


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
})


def requires_consent(model_id: str, handler_key: str) -> bool:
    """Check whether a model/handler involves voice cloning."""
    if handler_key in VOICE_CLONING_HANDLER_KEYS:
        return True
    model_lower = model_id.lower()
    return any(kw in model_lower for kw in VOICE_CLONING_MODEL_KEYWORDS)


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
