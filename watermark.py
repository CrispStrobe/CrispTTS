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

import contextlib
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
#: Frames per batched FFT. Bounds peak memory (a complex spectra block is
#: _FRAME_BLOCK x 513 x 16 bytes, ~4 MB here) while keeping the per-frame
#: Python loop out of the embed and detect hot paths.
_FRAME_BLOCK = 512

#: Detection needs enough frames for the statistic to mean anything. Measured,
#: 0.5 s of audio (14 frames at 16 kHz) is where a real mark stops being
#: distinguishable from unmarked audio. Set below 30 deliberately: one second
#: at 16 kHz yields exactly 30 frames, and refusing that would report 0.0 for
#: a validly marked one-second clip — which the marking gate reads as "not
#: marked" and answers by deleting the user's audio.
_DETECT_MIN_FRAMES = 20

#: Decoy sign patterns used to calibrate the null against the signal itself.
#: Cheap — they share the FFT with the real pattern — and 15 is enough for a
#: stable median/MAD.
_DETECT_DECOYS = 15

#: Floor on the decoy spread, so that audio where every decoy agrees (highly
#: structured or near-silent input) cannot divide a small difference by a
#: vanishing scale and manufacture certainty.
_DETECT_MIN_SCALE = 0.75

#: Calibration mapping the standardised score onto the [0, 1] confidence scale
#: the rest of the codebase and the docs already speak in, so _VERIFY_THRESHOLD
#: stays 0.65 and the familiar landmarks survive.
#:
#: Operating point, chosen from a grid over a corpus of 53 unmarked and 159
#: marked clips spanning 16/22.05/24/44.1 kHz, 1-5 s, clean plus 64 kbps MP3
#: and resample attacks, and including deliberately pathological synthetic
#: signals (stationary tones, a single tone, white noise):
#:
#:     rule                       FP      TP
#:     sign test (what shipped)   8.6%    97.0%
#:     t>=3.0 and z>=1.0          1.9%    99.4%
#:     t>=3.0 and z>=1.5          0.0%    94.3%
#:
#: The zero-false-positive row is not the right trade here: marking fails
#: closed, so rejecting 5.7% of *valid* marks means deleting that share of
#: users' audio. The chosen row improves on the shipped detector in both
#: directions at once.
_DETECT_T_MIN = 3.0
_DETECT_Z_MIN = 1.0

#: Third condition, added in v0.9.12: the real pattern must also out-score the
#: single strongest decoy, not merely the decoy median. This is what finally
#: removed the stationary-tone false positive that survived Phase 28.
#:
#:     rule                                   FP      TP
#:     t>=3.0 and z>=1.0                      1.89%   99.37%
#:     + t_true > 0.70 * max(|t_decoy|)       0.00%   99.37%
#:
#: Free, in the sense that no true positive pays for it. The tone scores 0.59
#: on this ratio (t_true 11.44 against a decoy maximum of 19.44 — every absent
#: pattern beats the real one, which is the tell), while the weakest genuine
#: mark in the corpus scores 0.84. Any threshold in (0.59, 0.84) separates them;
#: 0.70 is the midpoint, so neither margin is thin.
_DETECT_MAX_DECOY_RATIO = 0.70

#: Confidence is a logistic in the binding constraint's margin, arranged so
#: that the decision point (a ratio of 1.0) is exactly _VERIFY_THRESHOLD and
#: the familiar landmarks survive: unmarked audio reads ~0.19 (it read ~0.44
#: before) and a healthy mark ~0.99 (it read ~0.84).
_T_SCALE = 0.35
_T_CENTRE = 1.0 - _T_SCALE * 0.6190392  # ln(0.65/0.35)


def _t_to_confidence(t_stat: float) -> float:
    """Squash the standardised score onto [0, 1] with the calibration above."""
    z = (t_stat - _T_CENTRE) / _T_SCALE
    # Guard the exponential: |z| beyond ~700 overflows, and the result is
    # saturated long before that anyway.
    if z > 60.0:
        return 1.0
    if z < -60.0:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-z)))
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
    starts = np.arange(0, n - _FFT_SIZE + 1, _HOP)
    if len(starts) == 0:
        return pcm.copy()

    # Batched FFT over blocks of frames, rather than one call per frame. The
    # loop below still runs once per *bin* (32 iterations) but no longer once
    # per frame, which is where the cost was: a 20 s file at 44.1 kHz is ~1700
    # frames, so this trades ~3400 FFT calls for a couple of dozen.
    #
    # Blocked rather than all-at-once on purpose: the spectra array is complex
    # and (frames x 513), so a whole audiobook in one batch would be hundreds
    # of megabytes. _FRAME_BLOCK caps that at a few MB while keeping the win.
    all_frames = np.lib.stride_tricks.sliding_window_view(pcm, _FFT_SIZE)[::_HOP]
    w_sq = (window ** 2).astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    offsets = np.arange(_FFT_SIZE)

    for block_start in range(0, len(starts), _FRAME_BLOCK):
        sl = slice(block_start, block_start + _FRAME_BLOCK)
        block_starts = starts[sl]
        spectra = np.fft.rfft(all_frames[sl] * window, axis=1)

        # RMS magnitude per frame, for the energy-proportional nudge
        mags = np.abs(spectra[:, 1:_FFT_SIZE // 2])
        rms_mag = (np.sqrt(np.mean(mags ** 2, axis=1)) if mags.shape[1] > 0
                   else np.zeros(len(block_starts)))
        nudge = alpha * rms_mag  # (F,)

        # Bins are applied in order and may repeat — `_generate_bin_pattern`
        # draws with replacement, and a repeated index must be nudged twice,
        # cumulatively. So this stays a loop over bins; only frames vectorise.
        for b_idx, b_sign in bins:
            if b_idx >= spectra.shape[1]:
                continue
            col = spectra[:, b_idx]
            mag = np.abs(col)
            new_mag = np.maximum(mag + nudge * b_sign, 0.0)
            big = mag > 1e-15
            # Where the bin has energy, rescale it; where it is empty and the
            # sign is positive, plant the nudge as a real value. Mirrors the
            # scalar form exactly.
            col *= np.where(big, new_mag / np.where(big, mag, 1.0), 1.0)
            if b_sign > 0:
                col[~big] = nudge[~big]
            spectra[:, b_idx] = col

        reconstructed = np.fft.irfft(spectra, n=_FFT_SIZE, axis=1).astype(np.float32) * window

        # Overlap-add. _HOP is _FFT_SIZE // 2, so frames of the same parity
        # never overlap each other — within one parity the target indices are
        # unique, so plain fancy-index accumulation is correct here (and,
        # unlike np.add.at, fast).
        for parity in (0, 1):
            sel = np.arange(parity, len(block_starts), 2)
            if len(sel) == 0:
                continue
            idx = (block_starts[sel][:, None] + offsets).reshape(-1)
            out[idx] += reconstructed[sel].reshape(-1)
            norm[idx] += np.tile(w_sq, len(sel))

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
    """Correlate one comb placement against the signal, frame by frame.

    Measures the comb's excess over its local spectral neighbourhood in **each
    frame**, then asks whether the mean of those per-frame readings is
    significantly positive — a one-sample t-statistic across frames — and maps
    that to a confidence in [0, 1].

    Why this, and not the test it replaced
    --------------------------------------
    The previous detector averaged the spectrum over all frames, compared 32
    bins against their neighbours, and scored each ``+1`` or ``-1`` by the
    *sign* of the difference, discarding its size. Under the null that is a
    coin flip per bin, so the score had mean 0.5 and standard deviation
    ``sqrt(32) / (2 * 32)`` = 0.088 — leaving the 0.65 threshold just 1.7 sigma
    above chance. Sweeping two bands and keeping the larger reading doubled the
    exposure again. Measured over 197 clips of genuinely unmarked audio (a real
    human recording plus the bundled disclosure assets, which
    ``scripts/make_disclosure_assets.py`` renders straight from edge_tts and
    never marks):

        old sign test   FP 8.6%   TP  97.0%   separation -0.125 (overlapping)
        per-frame t     FP 0.0%   TP 100.0%   separation +0.63

    So the old test was both flagging unmarked audio and missing real marks.
    Two things make this form better: it keeps the magnitude of each difference
    instead of only its sign, and its sample count is the number of frames —
    hundreds to thousands — rather than 32. Measured, the null barely moves as
    clips lengthen (max t 3.03 at 1 s, 2.70 at 5 s) while a real mark grows with
    the evidence available (min t 3.67 at 1 s, 14.06 at 5 s).

    The **embed is unchanged**, so audio marked by any earlier CrispTTS, or by
    CrispASR, reads through this detector too — better than before, not worse.

    Returns:
        Confidence in [0, 1]. >0.65 means present. On unmarked audio the median
        reading is ~0.34 and the largest observed was 0.61.
    """
    n = len(pcm)
    if n < _FFT_SIZE:
        return 0.0

    bins = _generate_bin_pattern(WATERMARK_KEY, _FFT_SIZE, WATERMARK_NBINS, lo_bin, hi_bin)
    if not bins:
        return 0.0

    window = np.hanning(_FFT_SIZE).astype(np.float32)
    n_fft_half = _FFT_SIZE // 2
    all_frames = np.lib.stride_tricks.sliding_window_view(pcm, _FFT_SIZE)[::_HOP]
    n_frames = all_frames.shape[0]
    if n_frames < _DETECT_MIN_FRAMES:
        return 0.0

    idx = np.array([b for b, _ in bins])
    sgn = np.array([s for _, s in bins], dtype=np.float64)
    keep = idx < n_fft_half
    idx, sgn = idx[keep], sgn[keep]
    if idx.size == 0:
        return 0.0

    # Decoy sign patterns over the *same* bins, from keys we never embed with.
    # These calibrate the null against this particular signal, which a fixed
    # threshold cannot do. Structured audio correlates with an arbitrary sign
    # pattern about as well as with ours — measured on a stationary three-tone
    # signal, unmarked, the true pattern scored a mean excess of 0.116, as
    # large as a real watermark on real speech (0.108), and a plain t-test
    # called it present with t = 11.4. The decoys score just as high on that
    # signal and near zero on marked audio, so comparing against them is what
    # separates "this audio has spectral structure" from "this audio carries
    # our comb".
    decoy_sgns = []
    for k in range(_DETECT_DECOYS):
        rng = _Prng(WATERMARK_KEY ^ (0x9E3779B97F4A7C15 * (k + 1) & 0xFFFFFFFFFFFFFFFF))
        decoy_sgns.append(np.array([1.0 if (rng.next() & 1) else -1.0
                                    for _ in range(len(idx))], dtype=np.float64))
    all_sgns = np.vstack([sgn] + decoy_sgns)  # (1 + D, B)

    # Local baseline: the same +-2 neighbours the previous detector used.
    # Deliberately *not* excluding other comb bins from it — that was tried and
    # measured worse (separation +0.031 -> +0.017), because the comb's signs are
    # random, so an opposite-signed neighbour increases the contrast rather than
    # muddying it. Widening the window to +-4 or +-6 was worse still.
    nb_off = np.array([-2, -1, 1, 2])
    nb_idx = idx[None, :] + nb_off[:, None]
    valid = ((nb_idx >= 1) & (nb_idx < n_fft_half)).astype(np.float64)
    nb_idx = np.clip(nb_idx, 1, n_fft_half - 1)

    # Per-frame correlation for the true pattern and every decoy at once,
    # accumulated in blocks to bound memory exactly as the embed side does.
    # The FFT is shared across all patterns, so the decoys cost almost nothing.
    per_frame = np.empty((all_sgns.shape[0], n_frames), dtype=np.float64)
    for block_start in range(0, n_frames, _FRAME_BLOCK):
        block = all_frames[block_start:block_start + _FRAME_BLOCK] * window
        mags = np.abs(np.fft.rfft(block, axis=1)[:, :n_fft_half]).astype(np.float64)
        local_mean = ((mags[:, nb_idx] * valid[None, :, :]).sum(axis=1)
                      / np.maximum(valid.sum(axis=0)[None, :], 1e-12))
        excess = (mags[:, idx] - local_mean) / np.maximum(local_mean, 1e-12)
        # (patterns, frames): correlate the block's excess with each pattern
        per_frame[:, block_start:block_start + len(block)] = (excess @ all_sgns.T).T / len(idx)

    spread = np.std(per_frame, axis=1, ddof=1)
    spread = np.maximum(spread, 1e-12)
    t_all = per_frame.mean(axis=1) / (spread / np.sqrt(n_frames))
    t_true, t_decoy = float(t_all[0]), t_all[1:]

    # Two questions, and a mark has to answer both.
    #
    #   t_true  - is the comb's excess consistent across frames at all?
    #   z       - is that specific to *our* pattern, or would any pattern
    #             score as well on this audio?
    #
    # Neither alone is enough, which measurement showed the hard way:
    #
    #                     t_true  med_dec  scale_dec       z
    #   tone,  unmarked    11.44    -5.07      11.09    1.49
    #   tone,  marked     133.64     0.68      40.32    3.30
    #   speech, unmarked   -0.14     0.06       1.45   -0.14
    #   speech, marked     11.12     0.40       3.79    2.83
    #
    # On a stationary tone a raw t of 11.4 means nothing, because every decoy
    # scores just as extremely — only z tells them apart. But z alone rejects
    # real marks at 44.1 kHz, where the comb sits in a low-energy region and
    # the decoy spread grows. Requiring both, at the operating point measured
    # below, beats the previous detector on *both* error rates.
    centre = float(np.median(t_decoy))
    mad = float(np.median(np.abs(t_decoy - centre)))
    scale = max(1.4826 * mad, _DETECT_MIN_SCALE)
    z = (t_true - centre) / scale
    # Third condition: beat the strongest decoy, not just the typical one. On a
    # stationary tone the real pattern scores high *and* so does every decoy,
    # and comparing against the median does not catch that; comparing against
    # the maximum does.
    strongest_decoy = float(np.max(np.abs(t_decoy)))
    ratio = t_true / max(strongest_decoy, 1e-9)
    # The binding constraint decides. 1.0 on this scale is the decision point;
    # confidence crosses _VERIFY_THRESHOLD there.
    return _t_to_confidence(min(t_true / _DETECT_T_MIN,
                                z / _DETECT_Z_MIN,
                                ratio / _DETECT_MAX_DECOY_RATIO))


# ---------------------------------------------------------------------------
# WavMark neural watermark (MIT license — fully free for commercial use)
# ---------------------------------------------------------------------------

_wavmark_model = None
_wavmark_device = None  # torch.device the model sits on; set by load_wavmark()


def load_wavmark() -> bool:
    """Load the WavMark neural watermark model (MIT license).

    WavMark embeds a 16-bit payload into 16 kHz mono audio, robust against
    Gaussian noise, MP3 compression, low-pass filtering and speed variation.
    Fully MIT licensed (code + model weights).

    On SNR: this docstring used to claim ">38 dB". That reads upstream's
    number backwards — ``wavmark.encode_watermark`` takes ``min_snr=20,
    max_snr=38`` and runs an iterative per-chunk search inside that band, so
    38 dB is the *ceiling* it targets, not a floor it clears.

    On cost: WavMark used to be unusable here, and the cause was never the
    model — it was the device. One forward pass on a 1 s chunk, measured:

        CPU, 4 threads (torch's default on this box)   16-30 s
        CPU, 8 threads                                  5.4 s
        MPS                                             0.54 s

    So this loader now prefers CUDA, then MPS, then CPU, and lifts torch's
    thread count on the CPU path. Earlier revisions selected CUDA-or-CPU only,
    which meant every Apple Silicon machine took the slowest path available to
    it. Audio shorter than one 16 kHz chunk still raises upstream and falls
    back to spread-spectrum via the caller's exception handler.

    Requires: pip install wavmark
    Returns True on success.
    """
    global _backend, _wavmark_model, _wavmark_device
    try:
        import torch
        import wavmark
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            # torch defaults to physical *performance* cores on Apple Silicon,
            # which leaves most of the machine idle for this model. Measured
            # 3-5x from raising it; capped at the CPU count so it cannot
            # oversubscribe a small container.
            try:
                torch.set_num_threads(max(torch.get_num_threads(), os.cpu_count() or 1))
            except Exception:  # noqa: S110 - thread count is an optimisation, not a requirement
                pass
        _wavmark_model = wavmark.load_model().to(device)
        _wavmark_device = device
        _backend = "wavmark"
        logger.info("WavMark neural watermark loaded (MIT license) on %s.", device)
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


#: Window stride for the WavMark scan, in samples. Mirrors upstream's
#: ``shift_range * num_point * shift_range_p`` = 0.1 * 16000 * 0.5.
_WAVMARK_SHIFT = 800
_WAVMARK_WINDOW = 16000
_WAVMARK_BATCH = 32


def _detect_wavmark(pcm: np.ndarray, sample_rate: int = 24000) -> float:
    """Detect WavMark watermark. Returns confidence [0, 1].

    Scans the same window positions as ``wavmark.decode_watermark`` but stops
    at the first window whose 16 start bits match exactly, instead of scanning
    every position and averaging all the hits.

    That difference is the whole cost of the backend on marked audio. Upstream
    has to finish the scan to compute its mean; CrispTTS only ever asks "is
    this marked, and with our payload" — one confident window answers it.
    Measured on MPS:

                              upstream    early exit
        10 s of marked audio    34.7 s        9.3 s
        20 s of marked audio    79.3 s        6.8 s

    Upstream scales with duration; this does not, because the mark is found in
    the first batch either way. Unmarked audio is the worst case for both and
    costs the same — there is no hit to stop on, so the full scan runs.

    Uses only ``model.decode``, wavmark's public model API, so no fork or patch
    of the package is involved. Falls back to ``decode_watermark`` if anything
    here raises.
    """
    import torch

    if sample_rate != 16000:
        pcm = _resample_linear(pcm, sample_rate, 16000)
    data = pcm.astype(np.float64)

    try:
        from wavmark.utils.wm_add_util import fix_pattern
        start_bit = np.array(fix_pattern[0:16])
        n_windows = (len(data) - _WAVMARK_WINDOW) // _WAVMARK_SHIFT
        if n_windows <= 0:
            return 0.0
        points = [i * _WAVMARK_SHIFT for i in range(n_windows)]
        device = _wavmark_device or torch.device("cpu")
        for i in range(0, len(points), _WAVMARK_BATCH):
            group = points[i:i + _WAVMARK_BATCH]
            batch = np.array([data[p:p + _WAVMARK_WINDOW] for p in group])
            with torch.no_grad():
                decoded = (_wavmark_model.decode(
                    torch.FloatTensor(batch).to(device)) >= 0.5).int().cpu().numpy()
            # Average every exact match in this batch rather than taking the
            # first. Individual windows carry the odd bit error, and upstream
            # corrects that by averaging across all hits in the file; averaging
            # within the batch recovers most of that accuracy at no extra cost,
            # because the batch has already been decoded. Measured on a noisy
            # 3 s signal: first-hit 0.8125, batch-mean 0.875, upstream 0.75.
            hits = [bits for bits in decoded if np.array_equal(bits[:16], start_bit)]
            if hits:
                mean_bits = (np.mean(np.array(hits), axis=0) >= 0.5).astype(int)
                return float(np.mean(mean_bits[16:] == _WAVMARK_PAYLOAD))
        return 0.0
    except Exception as e:
        logger.debug("Fast WavMark scan failed (%s); using upstream decode.", e)

    import wavmark
    payload_decoded, _info = wavmark.decode_watermark(
        _wavmark_model, data, show_progress=False,
    )
    if payload_decoded is None:
        return 0.0
    # Compare decoded payload against our fixed marker
    return float(np.mean(payload_decoded[:16] == _WAVMARK_PAYLOAD))


# ---------------------------------------------------------------------------
# AudioSeal dispatcher (multiple backends)
# ---------------------------------------------------------------------------

# Backend priority: audioseal (Python) > wavmark (MIT) > crispasr (C) > spread-spectrum.
#
# AudioSeal leads on measurement, not on principle. Both are MIT for code *and*
# weights (AudioSeal's went MIT in April 2024, replacing CC-BY-NC), so the
# choice comes down to what they cost and what they survive. Measured on this
# integration, 10 s of speech at 16 kHz:
#
#                     AudioSeal    WavMark
#   model load          1.9 s       21 s
#   embed               2.0 s      ~180 s (extrapolated)
#   detect              0.45 s     did not return in 10 minutes
#   SNR                28.9 dB     36.3 dB
#   MP3 64k conf        1.000      -
#   Opus round-trip     1.000      -
#   false positive      0.000      -
#
# WavMark is ~7 dB quieter and that is its one real advantage. It is not worth
# a detect step that never finishes, because mark_audio_file() detects after
# every embed as its verification gate — so WavMark's cost is paid on every
# marked file, not only when someone asks to verify one.
_backend = "spread_spectrum"  # active backend name
_audioseal_generator = None   # audioseal Python generator model
_audioseal_detector = None    # audioseal Python detector model
_crispasr_wm = None           # crispasr C binding module


def load_audioseal_python() -> bool:
    """Load AudioSeal directly via the audioseal Python package.

    Requires: pip install audioseal
    Returns True on success.

    TorchDynamo is disabled for the load and for every call that follows.
    AudioSeal's SEANet layers go through ``torch.compile``, and CrispTTS feeds
    it a different tensor shape on nearly every run — TTS outputs vary in
    length — so Dynamo recompiles instead of reusing a graph and then trips its
    own ``recompile_limit``. Measured on a single cold 10 s embed: 56.5 s
    compiled versus **2.0 s** with Dynamo off, identical detection confidence
    (1.000). The compile only ever pays off for repeated identical shapes,
    which is not this workload.
    """
    global _backend, _audioseal_generator, _audioseal_detector
    # Set before the import: audioseal applies the decorators at import time,
    # so flipping this afterwards would come too late for an already-imported
    # module in a long-lived process (the API server).
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        from audioseal import AudioSeal
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
        except Exception:  # noqa: S110 - older torch, or no dynamo; env var already set
            pass
        _audioseal_generator = AudioSeal.load_generator("audioseal_wm_16bits")
        _audioseal_detector = AudioSeal.load_detector("audioseal_detector_16bits")
        _backend = "audioseal_python"
        logger.info("AudioSeal loaded via Python audioseal package (TorchDynamo disabled).")
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


def watermark_embed(pcm: np.ndarray, alpha: float | None = None, sample_rate: int = 24000,
                    force: bool = False) -> np.ndarray:
    """Embed AI-generated watermark. Dispatches to the best available backend.

    Priority: audioseal (Python) > wavmark (MIT) > crispasr (C/GGUF) >
    spread-spectrum. See the module-level table next to ``_backend`` for the
    measurements behind that order.

    Args:
        pcm: 1-D float32 mono PCM array.
        alpha: Strength for spread-spectrum (ignored when neural backends
            active). ``None`` selects the active band's default — 0.05 for the
            speech band, 0.08 for legacy. This default used to be a hardcoded
            0.08: the legacy band's value, left behind when the comb moved into
            the speech band. Every embed therefore ran 1.6x hotter than the
            band was designed for, costing 3-4 dB of SNR for a confidence gain
            that was never needed (both alphas clear the 0.65 threshold by a
            wide margin after resampling and 64 kbps MP3).
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
        if not load_audioseal_python():
            load_wavmark()

    if _backend == "audioseal_python" and _audioseal_generator is not None:
        try:
            result = _embed_audioseal_python(pcm, sample_rate)
            logger.debug("AudioSeal (Python) watermark embedded (%d samples).", len(pcm))
            return result
        except Exception as e:
            logger.warning("AudioSeal Python embed failed, trying next backend: %s", e)

    if _backend == "wavmark" and _wavmark_model is not None:
        try:
            result = _embed_wavmark(pcm, sample_rate)
            logger.debug("WavMark (MIT) watermark embedded (%d samples).", len(pcm))
            return result
        except Exception as e:
            logger.warning("WavMark embed failed, trying next backend: %s", e)

    if _backend == "audioseal_crispasr" and _crispasr_wm is not None:
        try:
            wm_pcm = pcm.copy()
            # The C binding takes a float, not None — resolve the band default
            # here rather than pushing Python's sentinel across the boundary.
            _crispasr_wm.watermark_embed(
                wm_pcm, wm_params(_FFT_SIZE)[2] if alpha is None else alpha)
            logger.debug("AudioSeal (crispasr) watermark embedded (%d samples).", len(pcm))
            return wm_pcm
        except Exception as e:
            logger.warning("AudioSeal crispasr embed failed, falling back to spread-spectrum: %s", e)

    result = spread_spectrum_embed(pcm, alpha)
    logger.debug("Spread-spectrum watermark embedded (%d samples).", len(pcm))
    return result


def watermark_detect(pcm: np.ndarray, sample_rate: int = 24000) -> float:
    """Detect AI-generated watermark. Returns confidence [0, 1].

    Tries available backends in priority order: audioseal > wavmark >
    spread-spectrum.

    Note that the backends do not share a scale. AudioSeal's detector saturates
    — measured 1.000 on watermarked speech and 0.000 on clean speech — while
    the spread-spectrum detector spans roughly 0.44 (its noise floor) to 0.91.
    Both are compared against the same 0.65 gate, which each clears
    unambiguously, but a confidence value is only comparable to another from
    the same backend.
    """
    if _backend == "audioseal_python" and _audioseal_detector is not None:
        try:
            score = _detect_audioseal_python(pcm, sample_rate)
            if score > 0.4:
                return score
            # Fall through: the file may carry a spread-spectrum mark instead,
            # e.g. written by the CrispASR binary or an earlier CrispTTS.
        except Exception as e:
            logger.warning("AudioSeal Python detect failed, trying next backend: %s", e)

    if _backend == "wavmark" and _wavmark_model is not None:
        try:
            score = _detect_wavmark(pcm, sample_rate)
            if score > 0.4:  # WavMark found something
                return score
            # Fall through to spread-spectrum (may have been watermarked by CrispASR binary)
        except Exception as e:
            logger.warning("WavMark detect failed, trying next backend: %s", e)

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


#: Verdict bands per backend, because the scores are not on one scale. The
#: spread-spectrum reading is a calibrated statistic (unmarked ~0.17 median,
#: a healthy mark ~0.99); AudioSeal's detector saturates at 0.000/1.000, so it
#: has no meaningful middle ground to report as uncertain; WavMark returns a
#: payload match ratio. Applying one set of bands to all three — which this
#: CLI did until v0.9.9 — reads a number from one instrument off another's
#: dial. Susurrus already returned the backend alongside the score; this is
#: that idea brought over.
#: Values are the floor of the "inconclusive" band; the detected bar is always
#: _VERIFY_THRESHOLD, which is defined further down this module.
_DETECT_UNCERTAIN_FLOOR = {
    "spread_spectrum": 0.50,
    "audioseal_python": None,    # saturates at 0.000/1.000 — no middle ground
    "audioseal_crispasr": None,
    "wavmark": 0.55,
}


def describe_detection(filepath: str) -> dict | None:
    """Read a file and report what its watermark reading actually supports.

    Returns a dict with ``confidence``, ``backend``, ``verdict`` and ``caveat``,
    or None if the file could not be read.

    The verdict is deliberately three-way. CrispASR reached the same conclusion
    from the other direction — see ``docs/eu-ai-act.md`` §6.7 there — after
    measuring that its binary "> 0.65 means detected" rule called 4.8% of clean
    speech watermarked, in the confident past tense. The honest answer for a
    reading that clears chance but not the bar is "inconclusive", not a claim.
    """
    confidence = watermark_verify_file(filepath)
    if confidence is None:
        return None
    detected_at = _VERIFY_THRESHOLD
    uncertain_at = _DETECT_UNCERTAIN_FLOOR.get(_backend, 0.50)
    if confidence >= detected_at:
        verdict = "AI-GENERATED WATERMARK DETECTED"
    elif uncertain_at is not None and confidence >= uncertain_at:
        verdict = "INCONCLUSIVE — above chance, but not evidence"
    else:
        verdict = "No watermark detected"
    return {
        "confidence": float(confidence),
        "backend": _backend,
        "threshold": detected_at,
        "verdict": verdict,
        # The asymmetry matters and is easy to misread: finding the mark is
        # strong evidence, not finding it is weak. Every lossy step erodes the
        # watermark, and most audio in the world was simply never marked.
        "caveat": ("A negative result is NOT evidence the audio is human-made — "
                   "it is often just a short, transcoded or unmarked clip."),
    }


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

#: Days an audit line is kept before :func:`prune_audit_log` drops it.
#: The log records reference-audio *paths*, which routinely contain personal
#: names, so it is personal data under GDPR and Art. 5(1)(e) requires a
#: retention limit rather than an append-forever file. Two years is a
#: defensible default for an evidential record of a consent attestation;
#: override with ``CRISPTTS_CONSENT_LOG_RETENTION_DAYS`` (0 disables pruning).
_CONSENT_LOG_RETENTION_DAYS = 730

#: The audit log is personal data; keep it owner-only rather than umask-default.
_CONSENT_LOG_MODE = 0o600

#: Hash-chain state mirrored outside the log itself. Ported from Susurrus's
#: ``utils/audit_log.py``, which chains its biometric records the same way.
#:
#: The point is evidential. This log's whole job is to record that somebody
#: attested they had the right to clone a voice, tied to a digest of the exact
#: recording. A plain text file that anyone can silently edit is weak evidence
#: of that; a chain where each line carries the hash of its predecessor makes
#: deletion and rewriting *detectable*.
#:
#: It is tamper-evidence, not tamper-proofing: whoever can write the file can
#: rebuild the chain. And a chain cannot detect truncation of its own tail —
#: drop the last n lines and the remainder still verifies — so the entry count
#: and head hash are mirrored into this sibling file after every write.
_CONSENT_ANCHOR_SUFFIX = ".anchor"
_CHAIN_GENESIS = "0" * 64


def consent_log_path() -> str:
    """Filesystem path of the persistent consent audit log."""
    return _CONSENT_LOG_PATH


def _consent_log_retention_days() -> int:
    raw = os.environ.get("CRISPTTS_CONSENT_LOG_RETENTION_DAYS")
    if raw is None:
        return _CONSENT_LOG_RETENTION_DAYS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid CRISPTTS_CONSENT_LOG_RETENTION_DAYS=%r.", raw)
        return _CONSENT_LOG_RETENTION_DAYS


def _parse_audit_timestamp(line: str):
    """Extract the ``ts=`` field from an audit line, or None."""
    from datetime import datetime
    marker = " ts="
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find(" ", start)
    stamp = line[start:] if end < 0 else line[start:end]
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def prune_audit_log(retention_days: int | None = None) -> int:
    """Drop audit lines older than the retention window (GDPR Art. 5(1)(e)).

    Runs on every append, so the log bounds itself without the operator having
    to remember. Lines with no parseable timestamp are kept — an unreadable
    record is not evidence that it has expired.

    Args:
        retention_days: Override the configured window. 0 disables pruning.

    Returns:
        Number of lines removed.
    """
    days = _consent_log_retention_days() if retention_days is None else max(0, retention_days)
    if not days or not os.path.isfile(_CONSENT_LOG_PATH):
        return 0
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    # One lock for read, rewrite and re-chain: a concurrent append landing
    # between them would be pruned away or would chain onto a head that no
    # longer exists.
    with _audit_lock():
        try:
            with open(_CONSENT_LOG_PATH) as f_audit:
                lines = f_audit.readlines()
        except OSError as e:
            logger.debug("Could not read the consent audit log for pruning: %s", e)
            return 0

        # The log is append-only and therefore chronological, so the expired
        # entries are a prefix: scan from the front and stop at the first line
        # inside the window, because everything after it is newer.
        #
        # This matters because pruning runs on every append. Parsing a
        # timestamp on every line made that O(n) per write and the log O(n^2)
        # overall — measured 19 ms/append at 25 entries, 55 ms at 200, on a log
        # that reaches 848 in ordinary use. Stopping early makes the normal
        # case (nothing to prune) a single strptime.
        #
        # Doing it by append count instead was tried and rejected: on an
        # install used a few times a month, "prune every 100 appends" leaves
        # expired personal data for years, which is the opposite of what
        # Art. 5(1)(e) asks for.
        kept: list[str] = []
        index, total = 0, len(lines)
        while index < total:
            ts = _parse_audit_timestamp(lines[index])
            if ts is None:
                # Unreadable timestamp: kept, per the rule above, and it does
                # not tell us the live entries have been reached.
                kept.append(lines[index])
                index += 1
                continue
            if ts >= cutoff:
                break  # this line and everything after it is inside the window
            index += 1  # expired: dropped
        kept.extend(lines[index:])
        removed = total - len(kept)
        if not removed:
            return 0
        try:
            tmp = f"{_CONSENT_LOG_PATH}.tmp"
            with open(tmp, "w") as f_new:
                f_new.writelines(kept)
            os.chmod(tmp, _CONSENT_LOG_MODE)
            os.replace(tmp, _CONSENT_LOG_PATH)
            logger.info("Pruned %d consent audit line(s) older than %d days.", removed, days)
        except OSError as e:
            logger.debug("Could not prune the consent audit log: %s", e)
            return 0
        _record_chain_rebuild(f"retention prune >{days}d", removed)
    return removed


def erase_audit_log(subject: str | None = None) -> int:
    """Erase audit lines, for a GDPR Art. 17 erasure request.

    Args:
        subject: Only erase lines containing this substring — a reference-audio
            path or its ``ref_sha256`` digest, which is how a specific
            speaker's attestations are identified. When None, the whole log is
            removed.

    Returns:
        Number of lines removed.
    """
    if not os.path.isfile(_CONSENT_LOG_PATH):
        return 0
    with _audit_lock():
        return _erase_audit_log_locked(subject)


def _erase_audit_log_locked(subject: str | None) -> int:
    """Body of :func:`erase_audit_log`; the caller holds the audit lock."""
    if subject is None:
        try:
            with open(_CONSENT_LOG_PATH) as f_audit:
                count = sum(1 for _ in f_audit)
            os.unlink(_CONSENT_LOG_PATH)
            # The anchor describes a log that no longer exists; leaving it
            # behind would make the next append look like a truncated chain.
            try:
                os.unlink(anchor_path())
            except OSError:
                pass
            logger.info("Erased the consent audit log (%d line(s)).", count)
            return count
        except OSError as e:
            logger.warning("Could not erase the consent audit log: %s", e)
            return 0
    try:
        with open(_CONSENT_LOG_PATH) as f_audit:
            lines = f_audit.readlines()
        kept = [ln for ln in lines if subject not in ln]
        removed = len(lines) - len(kept)
        if removed:
            tmp = f"{_CONSENT_LOG_PATH}.tmp"
            with open(tmp, "w") as f_new:
                f_new.writelines(kept)
            os.chmod(tmp, _CONSENT_LOG_MODE)
            os.replace(tmp, _CONSENT_LOG_PATH)
            # Deliberately does not name the subject: this record has to survive
            # the erasure it documents, so it must not re-introduce the personal
            # data that was just removed.
            _record_chain_rebuild("GDPR Art. 17 erasure request", removed)
        logger.info("Erased %d consent audit line(s) matching %r.", removed, subject)
        return removed
    except OSError as e:
        logger.warning("Could not erase from the consent audit log: %s", e)
        return 0


def anchor_path() -> str:
    """Path of the sidecar that mirrors the chain head outside the chain."""
    return _CONSENT_LOG_PATH + _CONSENT_ANCHOR_SUFFIX


@contextlib.contextmanager
def _audit_lock():
    """Serialise the log's read-modify-write across threads and processes.

    Chaining turned appending from a bare ``open(..., "a")`` — which the OS
    already makes atomic — into read-the-tail, hash it, write. Two of those
    interleaved produce two entries claiming the same predecessor, and the
    chain reads as tampered afterwards. CrispTTS has two concurrent paths that
    reach here: the threading API server (``daemon_threads = True``) and batch
    mode with ``--jobs > 1``. Measured before this lock existed: 24 attestations
    over 8 threads left all 24 entries present and the chain broken in four
    places — a false tamper alarm produced by ordinary use, which is worse than
    having no chain at all.

    A separate lock file, so the lock can be taken before the log exists.
    Failure to lock is not failure to record: if the platform has no ``fcntl``
    or the lock file cannot be opened, the attestation is still written. A
    missing audit line is a worse outcome than a racy one.
    """
    lock_file = _CONSENT_LOG_PATH + ".lock"
    fd = None
    try:
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, _CONSENT_LOG_MODE)
    except OSError as e:
        logger.debug("Could not open the consent audit lock (%s); proceeding unserialised.", e)
        yield
        return
    try:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError) as e:  # non-POSIX, or a filesystem without locks
            logger.debug("Consent audit lock unavailable (%s); proceeding unserialised.", e)
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _line_hash(prev: str, line: str) -> str:
    """Hash of one audit line, bound to its predecessor."""
    import hashlib
    return hashlib.sha256((prev + line.rstrip("\n")).encode("utf-8")).hexdigest()


def _chain_head(lines: list[str]) -> str:
    """Recompute the chain head over ``lines`` from genesis."""
    head = _CHAIN_GENESIS
    for line in lines:
        if line.strip():
            head = _line_hash(head, line)
    return head


def _write_anchor_direct(entries: int, head: str) -> None:
    """Mirror an already-computed count and head beside the log, atomically."""
    try:
        tmp = anchor_path() + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(f"entries={entries} head={head}\n")
        os.chmod(tmp, _CONSENT_LOG_MODE)
        os.replace(tmp, anchor_path())
    except OSError as e:
        logger.debug("Could not update the consent audit anchor: %s", e)


def _write_anchor(lines: list[str]) -> None:
    """Mirror entry count and head hash beside the log, recomputing both.

    Used where the log has just been rewritten wholesale (prune, erase) and
    there is no incremental head to carry forward. The append path uses
    :func:`_write_anchor_direct` instead, because recomputing here is what made
    appending quadratic.
    """
    _write_anchor_direct(sum(1 for ln in lines if ln.strip()), _chain_head(lines))


def _read_log_lines() -> list[str]:
    try:
        with open(_CONSENT_LOG_PATH) as fh:
            return fh.readlines()
    except OSError:
        return []


def _read_anchor() -> tuple[int, str] | None:
    """Return ``(entries, head)`` from the anchor, or None if unusable."""
    try:
        with open(anchor_path()) as fh:
            fields = dict(part.split("=", 1) for part in fh.read().split() if "=" in part)
        return int(fields["entries"]), fields["head"]
    except (OSError, KeyError, ValueError):
        return None




def _record_chain_rebuild(reason: str, removed: int) -> None:
    """Note that the chain was rebuilt lawfully, so it does not read as tampering.

    Retention pruning and Art. 17 erasure both *have* to remove entries, which
    is exactly what a hash chain exists to detect. The resolution is not to
    exempt them but to record them: an unexplained gap is tampering, a gap with
    a rebuild record next to it is a documented erasure. Verification reports
    the rebuilds it finds rather than hiding them.
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")

    # Re-chain the survivors. Their recorded predecessors refer to entries that
    # are lawfully gone, so without this every prune would leave the log
    # permanently unverifiable and real tampering would hide in the noise.
    survivors = [ln for ln in _read_log_lines() if ln.strip()]
    lines, head = [], _CHAIN_GENESIS
    for survivor in survivors:
        body = survivor.rstrip("\n").rpartition(" prev=")[0] or survivor.rstrip("\n")
        chained = f"{body} prev={head}\n"
        head = _line_hash(head, chained)
        lines.append(chained)
    rebuilt = f'[CHAIN-REBUILT] ts={ts} reason="{reason}" removed={removed} prev={head}\n'
    lines.append(rebuilt)
    try:
        tmp = f"{_CONSENT_LOG_PATH}.tmp"
        with open(tmp, "w") as fh:
            fh.writelines(lines)
        os.chmod(tmp, _CONSENT_LOG_MODE)
        os.replace(tmp, _CONSENT_LOG_PATH)
    except OSError as e:
        logger.debug("Could not record the chain rebuild: %s", e)
        return
    _write_anchor(lines)


def verify_audit_chain() -> dict:
    """Check the consent log's hash chain against its anchor.

    Returns a dict with ``ok``, ``entries``, ``rebuilds`` and ``issues``. An
    absent log is fine and reports ok; an absent *anchor* is reported as an
    issue, because the anchor is what makes tail truncation visible.
    """
    result = {"ok": True, "entries": 0, "rebuilds": 0, "legacy": 0, "issues": []}
    if not os.path.isfile(_CONSENT_LOG_PATH):
        return result

    lines = [ln for ln in _read_log_lines() if ln.strip()]
    result["entries"] = len(lines)
    result["rebuilds"] = sum(1 for ln in lines if ln.startswith("[CHAIN-REBUILT]"))

    head = _CHAIN_GENESIS
    for n, line in enumerate(lines, 1):
        body, _, recorded = line.rstrip("\n").rpartition(" prev=")
        if not recorded or len(recorded) != 64:
            # Written before v0.9.10, when chaining was added. Counted, not
            # flagged: an upgrade must not tell every existing user their audit
            # log has been tampered with. These lines are folded into the head,
            # so everything appended from now on is covered.
            result["legacy"] += 1
            head = _line_hash(head, line)
            continue
        if recorded != head:
            result["issues"].append(f"line {n}: chain broken — a preceding entry was changed or removed")
            head = recorded  # resynchronise so one break does not cascade
        head = _line_hash(head, body + " prev=" + recorded)

    chained = len(lines) - result["legacy"]
    try:
        with open(anchor_path()) as fh:
            anchor = fh.read().strip()
    except OSError:
        anchor = None
        if chained:
            result["issues"].append("no anchor file — truncation of the log's tail cannot be detected")
    if anchor:
        fields = dict(part.split("=", 1) for part in anchor.split() if "=" in part)
        if int(fields.get("entries", -1)) != len(lines):
            result["issues"].append(
                f"anchor expects {fields.get('entries')} entries, found {len(lines)} — "
                "entries were removed without a rebuild record")
        elif fields.get("head") != head:
            result["issues"].append("anchor head does not match the log — the log was rewritten")

    result["ok"] = not result["issues"]
    return result


def _append_audit_line(msg: str) -> None:
    """Append a chained line to the audit log, owner-only, pruning expired entries."""
    entries = 0
    try:
        with _audit_lock():
            os.makedirs(os.path.dirname(_CONSENT_LOG_PATH), exist_ok=True)
            existed = os.path.exists(_CONSENT_LOG_PATH)
            lines = _read_log_lines() if existed else []
            # Take the previous head from the anchor rather than rehashing the
            # whole log. Recomputing it per append was O(n) in hashes, twice
            # over (once here, once for the anchor), which made writing the log
            # quadratic. The anchor is only trusted when its entry count still
            # matches the file; otherwise fall back to the full recompute,
            # which is also what repairs a log appended to by an older version.
            entries = sum(1 for ln in lines if ln.strip())
            anchor = _read_anchor()
            head = anchor[1] if anchor and anchor[0] == entries else _chain_head(lines)
            chained = msg.rstrip("\n") + f" prev={head}\n"
            with open(_CONSENT_LOG_PATH, "a") as f_audit:
                f_audit.write(chained)
            if not existed:
                os.chmod(_CONSENT_LOG_PATH, _CONSENT_LOG_MODE)
            entries += 1
            _write_anchor_direct(entries, _line_hash(head, chained))
    except OSError as e:
        logger.debug("Could not write consent audit log: %s", e)
        return
    # Outside the lock: prune_audit_log() takes it itself, and flock is per-fd
    # rather than reentrant, so nesting them would deadlock. Still on every
    # append — it stops at the first live entry, so the usual case costs one
    # timestamp parse.
    prune_audit_log()


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
    _append_audit_line(msg)
    logger.info("Consent attestation logged for model=%s voice=%s", model_id, voice_str)


# ---------------------------------------------------------------------------
# Whose voice is it? (EU AI Act Art. 3(60) / Art. 50(4))
# ---------------------------------------------------------------------------
#
# Art. 3(60) defines a deep fake by what the output *resembles* — "an existing
# person" — not by how the resemblance was produced. Cloning from a reference
# recording at inference time is only the most obvious route there; a
# single-speaker model finetuned on one identifiable person's recordings
# produces audio of that person just as much, and the audience has no way to
# tell the two apart.
#
# So the spoken Art. 50(4) disclosure is keyed on this, not on `voice_cloning`
# alone. The voice donor's consent to their recordings being used for training
# is a licensing question; it is not the audience knowing the audio is
# synthetic, which is what Art. 50(4) is about.

#: Permitted values for a model's ``speaker_identity`` config key.
#:
#: ``real_person``  the preset voice is that of an identifiable individual
#:                  (a named donor, or a corpus speaker such as VCTK's p225).
#: ``synthetic``    a designed or blended voice that is not any one person.
#: ``unknown``      the training voices' provenance is not established. Treated
#:                  as a question the deployer must answer, not as "synthetic":
#:                  the same choice Phase 19 made for ``"multilingual"``
#:                  disclosure languages, and for the same reason.
SPEAKER_IDENTITY_VALUES = frozenset({"real_person", "synthetic", "unknown"})

_warned_speaker_identity: set[str] = set()


def resolve_speaker_identity(model_config: dict | None = None,
                             override: str | None = None) -> str:
    """Resolve whose voice a fixed-speaker model produces.

    Precedence: an explicit ``--speaker-identity`` override, then the model's
    declared ``speaker_identity``, then ``"unknown"``. An unrecognised value
    resolves to ``"unknown"`` rather than being trusted.
    """
    if override:
        value = str(override).strip().lower()
        if value in SPEAKER_IDENTITY_VALUES:
            return value
        logger.warning("Unrecognised --speaker-identity %r; treating as 'unknown'. "
                       "Expected one of: %s", override,
                       ", ".join(sorted(SPEAKER_IDENTITY_VALUES)))
        return "unknown"
    if model_config:
        declared = model_config.get("speaker_identity")
        if declared in SPEAKER_IDENTITY_VALUES:
            return declared
        if declared is not None:
            logger.warning("Model declares an unrecognised speaker_identity %r; "
                           "treating as 'unknown'.", declared)
    return "unknown"


def requires_spoken_disclosure(is_voice_cloning: bool, speaker_identity: str,
                               model_id: str | None = None) -> bool:
    """Whether this output needs the spoken Art. 50(4) disclosure prepended.

    True when the voice is cloned from a reference recording, or when the
    model's preset voice belongs to an identifiable person. ``"unknown"``
    warns once per model and does **not** force a disclosure: guessing wrong in
    that direction would prepend a sentence to every stock TTS voice, and a
    warning nobody can act on is worse than one they can.
    """
    if is_voice_cloning:
        return True
    if speaker_identity == "real_person":
        return True
    if speaker_identity == "unknown":
        _warn_unknown_speaker_identity_once(model_id)
    return False


def _warn_unknown_speaker_identity_once(model_id: str | None) -> None:
    """Warn once per model that the Art. 50(4) question is unanswered."""
    key = model_id or "<unknown-model>"
    if key in _warned_speaker_identity:
        return
    _warned_speaker_identity.add(key)
    logger.warning(
        "Model '%s' does not record whether its preset voice belongs to a real "
        "person, so no spoken AI disclosure was added. If the voice is that of an "
        "identifiable individual, the output is a deep fake under EU AI Act "
        "Art. 3(60) and you carry the Art. 50(4) duty to disclose it. Pass "
        "--speaker-identity real_person to have CrispTTS prepend the disclosure, "
        "or --speaker-identity synthetic to silence this.", key)


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
#:
#: All 24 EU official languages are covered. The Regulation governs content
#: placed on the EU market, so "a disclosure the audience can understand" means
#: any of them — a German sentence in front of Greek audio discloses nothing to
#: a Greek listener. ``zh``/``ja``/``ko`` are here because shipped models
#: (CosyVoice3, IndexTTS, VibeVoice, MOSS, OuteTTS) target them directly.
DISCLAIMER_TEXTS = {
    # EU official languages
    "bg": "Следващият аудиозапис е генериран от изкуствен интелект.",
    "cs": "Následující zvuková nahrávka byla vytvořena umělou inteligencí.",
    "da": "Følgende lydoptagelse er genereret af kunstig intelligens.",
    "de": "Die folgende Aufnahme wurde von künstlicher Intelligenz erzeugt.",
    "el": "Το ακόλουθο ηχητικό απόσπασμα δημιουργήθηκε από τεχνητή νοημοσύνη.",
    "en": "The following audio was generated by artificial intelligence.",
    "es": "El siguiente audio fue generado por inteligencia artificial.",
    "et": "Järgnev helisalvestis on loodud tehisintellekti abil.",
    "fi": "Seuraava äänite on tekoälyn tuottama.",
    "fr": "L'enregistrement suivant a été généré par une intelligence artificielle.",
    "ga": "Gineadh an taifead fuaime seo a leanas le hintleacht shaorga.",
    "hr": "Sljedeći zvučni zapis generirala je umjetna inteligencija.",
    "hu": "A következő hangfelvételt mesterséges intelligencia készítette.",
    "it": "Il seguente audio è stato generato dall'intelligenza artificiale.",
    "lt": "Šis garso įrašas sugeneruotas dirbtinio intelekto.",
    "lv": "Šo audioierakstu ir ģenerējis mākslīgais intelekts.",
    "mt": "Ir-reġistrazzjoni awdjo li ġejja ġiet iġġenerata minn intelliġenza artifiċjali.",
    "nl": "De volgende audio is gegenereerd door kunstmatige intelligentie.",
    "pl": "Poniższe nagranie zostało wygenerowane przez sztuczną inteligencję.",
    "pt": "O seguinte áudio foi gerado por inteligência artificial.",
    "ro": "Următoarea înregistrare audio a fost generată de inteligență artificială.",
    "sk": "Nasledujúca zvuková nahrávka bola vytvorená umelou inteligenciou.",
    "sl": "Naslednji zvočni posnetek je ustvarila umetna inteligenca.",
    "sv": "Följande ljudinspelning har genererats av artificiell intelligens.",
    # Non-EU languages targeted by shipped models
    "ja": "以下の音声は人工知能によって生成されました。",
    "ko": "다음 오디오는 인공지능으로 생성되었습니다.",
    "zh": "以下音频由人工智能生成。",
}

#: CrispTTS is a German TTS toolkit, so German is the default disclosure
#: language — not English.
DEFAULT_DISCLAIMER_LANG = "de"

#: Config ``language`` values that name no single spoken language. For a
#: multilingual model the output language is a property of the *input text*,
#: not of the model, so it cannot be derived from the config at all — it has to
#: be supplied with ``--disclosure-lang`` / ``"disclosure_lang"``. Treating
#: these as "unknown" rather than silently substituting German is the whole
#: point: a wrong-language disclosure is not a disclosure.
MULTILINGUAL_LANG_MARKERS = frozenset({
    "multilingual", "multi", "mul", "any", "auto", "none", "unknown",
})

#: Edge TTS voice used per language when CrispASR is unavailable.
_DISCLAIMER_EDGE_VOICES = {
    "bg": "bg-BG-KalinaNeural",
    "cs": "cs-CZ-VlastaNeural",
    "da": "da-DK-ChristelNeural",
    "de": "de-DE-KatjaNeural",
    "el": "el-GR-AthinaNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "et": "et-EE-AnuNeural",
    "fi": "fi-FI-NooraNeural",
    "fr": "fr-FR-DeniseNeural",
    "ga": "ga-IE-OrlaNeural",
    "hr": "hr-HR-GabrijelaNeural",
    "hu": "hu-HU-NoemiNeural",
    "it": "it-IT-ElsaNeural",
    "lt": "lt-LT-OnaNeural",
    "lv": "lv-LV-EveritaNeural",
    "mt": "mt-MT-GraceNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "pt": "pt-PT-RaquelNeural",
    "ro": "ro-RO-AlinaNeural",
    "sk": "sk-SK-ViktoriaNeural",
    "sl": "sl-SI-PetraNeural",
    "sv": "sv-SE-SofieNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

_DISCLAIMER_SILENCE_SEC = 0.3  # 300ms gap between disclaimer and content


class DisclosureError(RuntimeError):
    """Raised when the spoken AI disclosure could not be added to cloned audio.

    Treated like :class:`MarkingError`: voice-cloned output without its
    disclosure is not delivered. The escape hatch is
    ``--no-spoken-disclaimer``, which requires
    ``--accept-marking-responsibility``.
    """


#: Long-form language names sometimes used in model configs.
_DISCLAIMER_LANG_ALIASES = {
    "german": "de", "english": "en", "french": "fr", "spanish": "es",
    "italian": "it", "dutch": "nl", "polish": "pl", "portuguese": "pt",
    "bulgarian": "bg", "czech": "cs", "danish": "da", "greek": "el",
    "estonian": "et", "finnish": "fi", "irish": "ga", "croatian": "hr",
    "hungarian": "hu", "lithuanian": "lt", "latvian": "lv", "maltese": "mt",
    "romanian": "ro", "slovak": "sk", "slovene": "sl", "slovenian": "sl",
    "swedish": "sv", "japanese": "ja", "korean": "ko",
    "chinese": "zh", "mandarin": "zh",
}


def resolve_disclaimer_lang(language: str | None,
                            override: str | None = None) -> tuple[str, bool]:
    """Resolve which language the spoken disclosure should be in.

    Precedence: an explicit ``override`` (``--disclosure-lang`` /
    ``"disclosure_lang"``) beats the model's declared language, which beats
    :data:`DEFAULT_DISCLAIMER_LANG`.

    Args:
        language: The model config's ``language`` value, if any.
        override: An explicit caller-supplied language code.

    Returns:
        ``(lang, known)``. ``known`` is False when neither the override nor the
        config identified a language we have a disclosure for, and the default
        was substituted. Callers must surface that: substituting German for a
        multilingual model's Mandarin output produces audio whose "disclosure"
        the audience cannot read, which is the failure Art. 50(4) is about.
    """
    for candidate in (override, language):
        if not candidate or not isinstance(candidate, str):
            continue
        lang = candidate.strip().lower().replace("_", "-")
        if not lang or lang in MULTILINGUAL_LANG_MARKERS:
            continue
        if lang in _DISCLAIMER_LANG_ALIASES:
            return _DISCLAIMER_LANG_ALIASES[lang], True
        base = lang.split("-")[0]
        if base in DISCLAIMER_TEXTS:
            return base, True
        logger.warning(
            "No spoken AI disclosure is available in language '%s'. Supported: %s.",
            candidate, ", ".join(sorted(DISCLAIMER_TEXTS)))
    return DEFAULT_DISCLAIMER_LANG, False


def normalize_disclaimer_lang(language: str | None) -> str:
    """Map a config language code (``de``, ``de-DE``, ``german``) to a key.

    Kept for callers that only need the code. Prefer
    :func:`resolve_disclaimer_lang`, which also reports whether the language
    was actually known or merely defaulted to.
    """
    return resolve_disclaimer_lang(language)[0]


def disclaimer_text(language: str | None = None) -> str:
    """The spoken disclosure sentence for a language."""
    return DISCLAIMER_TEXTS[normalize_disclaimer_lang(language)]


#: Languages already warned about, so ``--test-all`` does not emit the same
#: caution once per model.
_warned_default_disclosure_langs: set[str | None] = set()


def _warn_defaulted_disclosure_lang(language: str | None) -> None:
    """Warn that the disclosure language had to be guessed."""
    if language in _warned_default_disclosure_langs:
        return
    _warned_default_disclosure_langs.add(language)
    logger.warning(
        "This model does not declare a single output language (language=%r), so the "
        "spoken AI disclosure defaults to '%s'. If you are synthesizing in another "
        "language, pass --disclosure-lang (CLI) or \"disclosure_lang\" (API) — a "
        "disclosure the audience cannot understand does not satisfy the Art. 50(4) "
        "duty to disclose. Available: %s.",
        language, DEFAULT_DISCLAIMER_LANG, ", ".join(sorted(DISCLAIMER_TEXTS)))


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
                       language: str | None = None,
                       disclosure_lang: str | None = None) -> tuple[np.ndarray, str]:
    """Prepend the spoken AI disclosure to voice-cloned audio.

    Layout: disclosure + 300 ms silence + original audio. The generated
    disclosure is cached per (sample rate, language).

    Args:
        pcm: The synthesized audio.
        sample_rate: Sample rate of ``pcm``.
        language: The model's declared language, used when no override is given.
        disclosure_lang: Explicit disclosure language, overriding ``language``.

    Returns:
        ``(pcm, kind)`` — see :func:`generate_spoken_disclaimer` for ``kind``.
        On failure the original PCM is returned unchanged with ``"none"``.
    """
    lang, known = resolve_disclaimer_lang(language, disclosure_lang)
    if not known:
        _warn_defaulted_disclosure_lang(language)
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
                            require_spoken: bool = True,
                            disclosure_lang: str | None = None) -> str:
    """Prepend the spoken AI disclosure to an audio file, in place.

    Format-agnostic: works for every container the marking pipeline supports.

    Supports the deployer's Art. 50(4) duty to disclose deepfake content. That
    obligation is theirs, but voice-cloned output carries the disclosure by
    default so it is present unless deliberately removed — which is why this
    raises rather than returning a value callers can ignore.

    Args:
        filepath: Audio file to modify in place.
        language: The model's declared language; used only when
            ``disclosure_lang`` is not supplied.
        require_spoken: When True, a tone-marker fallback is not accepted as a
            disclosure and raises instead.
        disclosure_lang: Explicit language for the disclosure, overriding
            ``language``. Needed for multilingual models, whose output language
            is determined by the input text rather than by the config.

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
        combined, kind = prepend_disclaimer(pcm, sample_rate=sr, language=language,
                                            disclosure_lang=disclosure_lang)
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
                kind, resolve_disclaimer_lang(language, disclosure_lang)[0], filepath)
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


#: Extension -> the format string c2pa-rs wants for that container.
#: Only containers that genuinely embed belong here; see C2PA_CAPABLE_EXTS.
_C2PA_FORMATS = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".m4a": "m4a"}


def _sign_with_c2pa_python(input_path: str, output_path: str,
                           cert_pem: bytes, key_pem: bytes,
                           model_id: str | None) -> bool:
    """Sign via c2pa-python, the one path where we control the manifest.

    Uses the streaming ``Builder.sign()`` rather than ``sign_file()``.
    ``sign_file()`` refuses FLAC and M4A with "NotSupported: type is
    unsupported" even though c2pa-rs advertises both in
    ``get_supported_mime_types()``; ``sign()`` with an explicit format signs
    them, and the result reads back ``validation_state: Valid`` carrying
    ``trainedAlgorithmicMedia``. The stream path is a strict superset — WAV
    and MP3 sign identically through it — so it is now the only path, and
    FLAC and M4A gained a manifest they were previously denied.
    """
    try:
        import c2pa as c2pa_rs

        ext = os.path.splitext(input_path)[1].lower()
        fmt = _C2PA_FORMATS.get(ext)
        if fmt is None:
            logger.debug("No C2PA format binding for '%s'; signing skipped.", ext)
            return False

        signer = _c2pa_signer(cert_pem, key_pem)
        builder = c2pa_rs.Builder(_c2pa_manifest(model_id))

        # sign() streams source -> destination and cannot write over its own
        # source, so it always goes via a temp file.
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            with open(input_path, "rb") as src, open(tmp_path, "wb") as dst:
                builder.sign(signer, fmt, src, dst)
            # A signer that writes nothing must not be reported as success —
            # the caller would take an unsigned file for a signed one.
            if os.path.getsize(tmp_path) <= 0:
                raise RuntimeError("C2PA signer produced an empty file")
            import shutil
            shutil.move(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
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
#: FLAC and M4A were excluded here on the evidence that they fail signing with
#: "NotSupported: type is unsupported" — true of ``sign_file()``, but not of
#: the streaming ``Builder.sign()``, which signs both into a manifest that
#: reads back Valid. ``_sign_with_c2pa_python()`` now uses the stream path, so
#: both belong here. Opus/OGG stays out: c2pa-rs does not list it among its
#: supported types at all, and every format string tried returns NotSupported.
#:
#: This set feeds the watermark floor, so an over-broad entry is not cosmetic:
#: it would let ``--no-watermark`` be honoured for a container that then
#: carries no manifest, leaving only strippable metadata.
#: ``tests/test_watermark.py`` verifies each entry by really signing a file.
C2PA_CAPABLE_EXTS = frozenset({".wav", ".mp3", ".flac", ".m4a"})


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
    _append_audit_line(msg)


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

        # --- A container with no manifest needs a strong watermark ---
        # For WAV/MP3/FLAC/M4A the C2PA manifest is the durable, interoperable
        # layer and the watermark is what survives the manifest being stripped.
        # Opus and OGG can carry no manifest at all, so the watermark is the
        # *only* robust layer — and on a default install that is the built-in
        # fixed-key comb. Art. 50(2) asks for marking that is robust as far as
        # technically feasible; when a neural backend is one pip install away,
        # shipping the weakest layer as the sole layer is not that. Refuse
        # rather than emit it.
        if ext not in C2PA_CAPABLE_EXTS and not neural_watermark_available():
            raise MarkingError(
                f"Refusing to synthesize: '{ext}' cannot carry a C2PA manifest, so the "
                "audio watermark would be its only robust mark — and only the built-in "
                "spread-spectrum backend is installed.\n"
                "  Either install a neural watermark:  pip install 'crisptts[robust]'\n"
                f"  or choose a container that carries a manifest: "
                f"{', '.join(sorted(C2PA_CAPABLE_EXTS))}\n"
                "  or take the marking duty on yourself with "
                "--allow-unmarked --accept-marking-responsibility."
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


def neural_watermark_available() -> bool:
    """True if a neural watermark backend could be loaded — without loading it.

    Deliberately a package-presence check rather than a load: this runs in
    preflight, before any model is pulled in, and ``--list-models`` has to
    stay instant. It answers "would :func:`watermark_embed` have something
    stronger than the built-in comb to reach for", not "is it loaded".
    """
    if _backend in ("wavmark", "audioseal_python", "audioseal_crispasr"):
        return True
    if _crispasr_wm is not None:
        return True
    import importlib.util
    for package in ("wavmark", "audioseal"):
        try:
            if importlib.util.find_spec(package) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


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


def _existing_watermark_detectable(filepath: str, ext: str) -> bool:
    """True if `filepath` already carries a watermark this detector can read.

    Used to decide whether a re-embed would be redundant. A failure to decode
    answers "no": that sends the caller down the embedding path, which will
    surface the real decode error rather than swallowing it here.
    """
    try:
        pcm, sr = _read_pcm_any(filepath, ext)
    except Exception as e:
        logger.debug("Could not measure an existing watermark on %s: %s", filepath, e)
        return False
    conf = watermark_detect(pcm, sample_rate=sr)
    return conf is not None and conf >= _VERIFY_THRESHOLD


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

    Idempotent: a file whose watermark is already *detectable* is not
    watermarked a second time (double embedding costs roughly 6 dB of SNR and
    adds no provenance). That is decided by measuring the audio, not by the
    presence of the container marker — a file carrying only the marker string
    goes down the full path and is gated like any other output.

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

    # Container metadata says *someone* marked this file. That is a hint about
    # what to do next, not a result: the marker is a strippable string, and a
    # file carrying nothing else is exactly the output this function exists to
    # refuse. Measured on the CrispASR streaming path, which injects the WAV
    # LIST/INFO chunk itself before returning — returning early here delivered
    # audio at watermark confidence 0.625 (below the 0.65 threshold) with no
    # manifest, reported as marked=True. So it now only suppresses *re-doing*
    # work that is already done; the verification gate below still decides.
    already_marked = is_marked(filepath)

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
    elif already_marked and _existing_watermark_detectable(filepath, ext):
        # A file this function has already been through. Re-embedding costs
        # ~6 dB of SNR and adds no provenance, so skip it — but only because
        # the watermark was *measured*, not because a metadata string said so.
        logger.debug("Watermark already present and detectable in %s; skipping re-embed.",
                     filepath)
        layers.append("audio-watermark:existing")
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
    if already_marked:
        # The marker is already in the container; injecting a second LIST/INFO
        # chunk or TXXX frame would only duplicate it.
        layers.append("metadata:existing")
    else:
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
