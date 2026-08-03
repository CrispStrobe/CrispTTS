"""Tests for watermark.py — spread-spectrum watermark, metadata, consent gate."""

import os
import struct
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# soundfile and c2pa are declared core dependencies, but a checkout can be run
# without them. These tests must then SKIP, not error: they cover the marking
# and disclosure gates, and a bare ModuleNotFoundError reads as a broken test
# suite rather than as "the guardrail tests did not run here".
try:
    import soundfile  # noqa: F401
    _HAVE_SOUNDFILE = True
except ImportError:
    _HAVE_SOUNDFILE = False

try:
    import c2pa  # noqa: F401
    _HAVE_C2PA = True
except ImportError:
    _HAVE_C2PA = False

requires_soundfile = unittest.skipUnless(
    _HAVE_SOUNDFILE, "soundfile not installed — marking gate tests cannot run")
requires_c2pa = unittest.skipUnless(
    _HAVE_C2PA, "c2pa-python not installed — C2PA tests cannot run")


def read_optional_dependencies():
    """Return pyproject's ``[project.optional-dependencies]`` as a dict.

    ``tomllib`` is stdlib only from 3.11, and this project supports 3.10 (the
    CI matrix is 3.10-3.12), so the extras tests cannot simply import it — they
    passed locally on 3.11 and broke the 3.10 job. ``tomli`` is not a
    dependency either, so on 3.10 fall back to a small parser that reads just
    the one table these tests need: names and quoted requirement strings.
    """
    path = PROJECT_ROOT / "pyproject.toml"
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)["project"]["optional-dependencies"]
    except ModuleNotFoundError:
        pass
    try:
        import tomli
        with open(path, "rb") as fh:
            return tomli.load(fh)["project"]["optional-dependencies"]
    except ModuleNotFoundError:
        pass

    import re
    extras, current = {}, None
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = stripped == "[project.optional-dependencies]"
            current = None
            continue
        if not in_table or stripped.startswith("#") or not stripped:
            continue
        m = re.match(r'^([A-Za-z0-9._-]+)\s*=\s*\[(.*)$', stripped)
        if m:
            current = m.group(1)
            extras[current] = []
            rest = m.group(2)
            extras[current].extend(re.findall(r'"([^"]+)"', rest))
            if "]" in rest:
                current = None
            continue
        if current is not None:
            extras[current].extend(re.findall(r'"([^"]+)"', stripped))
            if stripped.startswith("]"):
                current = None
    return extras


def force_spread_spectrum(testcase):
    """Pin the dispatcher to the built-in backend for one test.

    Several gate tests assert what happens when the watermark *cannot* be
    embedded — audio shorter than an FFT frame, digital silence. Those are
    limits of the spread-spectrum comb, not of marking in general: AudioSeal
    marks both successfully and the gate then correctly passes.

    Without pinning, those tests only hold on a machine where no neural extra
    is installed, and turn red the moment someone runs the install the README
    recommends. The loaders are stubbed too, because the lazy-load path in
    ``watermark_embed`` would otherwise re-load a backend the moment it sees
    the cleared globals.
    """
    import watermark
    saved = {
        "_backend": watermark._backend,
        "_audioseal_generator": watermark._audioseal_generator,
        "_audioseal_detector": watermark._audioseal_detector,
        "_wavmark_model": watermark._wavmark_model,
        "load_audioseal_python": watermark.load_audioseal_python,
        "load_wavmark": watermark.load_wavmark,
    }

    def restore():
        for name, value in saved.items():
            setattr(watermark, name, value)

    testcase.addCleanup(restore)
    watermark._backend = "spread_spectrum"
    watermark._audioseal_generator = None
    watermark._audioseal_detector = None
    watermark._wavmark_model = None
    watermark.load_audioseal_python = lambda: False
    watermark.load_wavmark = lambda: False


class TestPrng(unittest.TestCase):
    """Verify PRNG produces deterministic output matching the C++ implementation."""

    def test_deterministic(self):
        from watermark import _Prng
        rng1 = _Prng(42)
        rng2 = _Prng(42)
        for _ in range(100):
            self.assertEqual(rng1.next(), rng2.next())

    def test_different_seeds_differ(self):
        from watermark import _Prng
        rng1 = _Prng(1)
        rng2 = _Prng(2)
        values1 = [rng1.next() for _ in range(10)]
        values2 = [rng2.next() for _ in range(10)]
        self.assertNotEqual(values1, values2)


class TestBinPattern(unittest.TestCase):
    """Test bin pattern generation."""

    def test_correct_count(self):
        from watermark import _generate_bin_pattern
        bins = _generate_bin_pattern(0x437269737041535F, 1024, 32)
        self.assertEqual(len(bins), 32)

    def test_bins_in_range(self):
        from watermark import _generate_bin_pattern
        bins = _generate_bin_pattern(0x437269737041535F, 1024, 32)
        lo = 1024 // 16
        hi = 1024 // 2 - 1
        for idx, sign in bins:
            self.assertGreaterEqual(idx, lo)
            self.assertLess(idx, lo + (hi - lo))
            self.assertIn(sign, (-1, 1))

    def test_deterministic(self):
        from watermark import _generate_bin_pattern
        b1 = _generate_bin_pattern(123, 1024, 32)
        b2 = _generate_bin_pattern(123, 1024, 32)
        self.assertEqual(b1, b2)

    def test_empty_on_bad_input(self):
        from watermark import _generate_bin_pattern
        self.assertEqual(_generate_bin_pattern(42, 1024, 0), [])
        self.assertEqual(_generate_bin_pattern(42, 0, 32), [])


class TestSpreadSpectrumRoundTrip(unittest.TestCase):
    """Test embed → detect round-trip."""

    def _make_sine(self, freq=440.0, sr=24000, duration=1.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * freq * t)

    def test_embed_detect_roundtrip(self):
        from watermark import spread_spectrum_detect, spread_spectrum_embed
        pcm = self._make_sine(duration=1.0)
        wm_pcm = spread_spectrum_embed(pcm, alpha=0.005)
        confidence = spread_spectrum_detect(wm_pcm)
        self.assertGreater(confidence, 0.65,
                           f"Watermark should be detected (confidence={confidence:.3f})")

    def test_unwatermarked_low_confidence(self):
        from watermark import spread_spectrum_detect
        pcm = self._make_sine(duration=1.0)
        confidence = spread_spectrum_detect(pcm)
        self.assertLess(confidence, 0.65,
                        f"Unwatermarked audio should have low confidence ({confidence:.3f})")

    def test_imperceptibility_snr(self):
        """SNR between original and watermarked should be > 20 dB.

        Pure sine waves yield lower SNR (~22 dB) because all energy
        concentrates in one bin; broadband speech easily exceeds 28 dB.
        20 dB is well below human perception threshold for speech.
        """
        from watermark import spread_spectrum_embed
        pcm = self._make_sine(duration=1.0)
        wm_pcm = spread_spectrum_embed(pcm, alpha=0.005)
        noise = wm_pcm - pcm
        signal_power = np.mean(pcm ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            self.assertGreater(snr_db, 20.0,
                               f"SNR should be > 20 dB (got {snr_db:.1f} dB)")

    def test_survives_volume_scaling(self):
        """Watermark should survive 2x volume scaling."""
        from watermark import spread_spectrum_detect, spread_spectrum_embed
        pcm = self._make_sine(duration=1.0)
        wm_pcm = spread_spectrum_embed(pcm, alpha=0.005)
        scaled = wm_pcm * 2.0
        confidence = spread_spectrum_detect(scaled)
        self.assertGreater(confidence, 0.6,
                           f"Watermark should survive volume scaling (confidence={confidence:.3f})")

    def test_short_audio_noop(self):
        """Audio shorter than 1 FFT frame should be returned unchanged."""
        from watermark import spread_spectrum_embed
        pcm = np.zeros(500, dtype=np.float32)
        result = spread_spectrum_embed(pcm)
        np.testing.assert_array_equal(result, pcm)

    def test_silence_detection(self):
        """Silent audio should return low confidence."""
        from watermark import spread_spectrum_detect
        pcm = np.zeros(24000, dtype=np.float32)
        confidence = spread_spectrum_detect(pcm)
        self.assertLessEqual(confidence, 0.5)


@requires_soundfile
@requires_c2pa
class TestSelfSignedManifestStillMarks(unittest.TestCase):
    """A self-signed manifest is an *attribution* limit, not a marking one.

    Earlier docs called the bundled certificate the largest remaining
    Art. 50(2) gap. Art. 50(2) asks for outputs "marked in a machine-readable
    format and detectable as artificially generated" — it does not ask the mark
    to prove who generated it. This pins the distinction to measured behaviour
    so the claim cannot drift back into prose.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _signed_wav(self):
        import soundfile as sf

        from watermark import c2pa_sign_file_ex
        path = os.path.join(self.tmp, "a.wav")
        sr = 24000
        t = np.linspace(0, 2, sr * 2, endpoint=False, dtype=np.float32)
        sf.write(path, (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)
        ok, signer = c2pa_sign_file_ex(path, model_id="test")
        self.assertTrue(ok)
        self.assertEqual(signer, "self-signed")
        return path

    def _manifest(self, path):
        import json

        import c2pa
        with open(path, "rb") as fh:
            return json.loads(c2pa.Reader("audio/wav", fh).json())

    def test_manifest_validates_and_carries_the_ai_assertion(self):
        manifest = self._manifest(self._signed_wav())
        self.assertEqual(manifest.get("validation_state"), "Valid")
        active = manifest["manifests"][manifest["active_manifest"]]
        source_types = [
            action.get("digitalSourceType", "")
            for assertion in active.get("assertions", [])
            for action in assertion.get("data", {}).get("actions", [])
        ]
        self.assertTrue(any("trainedAlgorithmicMedia" in s for s in source_types),
                        f"AI assertion missing from a self-signed manifest: {source_types}")

    def test_the_only_failure_is_signer_trust(self):
        """If a self-signed manifest ever fails for another reason, marking is affected."""
        manifest = self._manifest(self._signed_wav())
        failures = {
            entry["code"]
            for entry in manifest["validation_results"]["activeManifest"]["failure"]
        }
        self.assertEqual(
            failures, {"signingCredential.untrusted"},
            f"a self-signed manifest should fail only on signer trust, got {failures}")


class TestConsentLogChain(unittest.TestCase):
    """The consent log is evidence, so silent edits to it must be detectable.

    Ported from Susurrus's utils/audit_log.py, which hash-chains and anchors
    its biometric records. This log's job is to record that somebody attested
    they had the right to clone a voice, tied to a digest of the exact
    recording — a plain text file anyone can edit is weak evidence of that.
    """

    def setUp(self):
        import tempfile

        import watermark
        self.tmp = tempfile.mkdtemp()
        self._saved = watermark._CONSENT_LOG_PATH
        watermark._CONSENT_LOG_PATH = os.path.join(self.tmp, "consent_audit.log")

    def tearDown(self):
        import shutil

        import watermark
        watermark._CONSENT_LOG_PATH = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _log(self, model, voice):
        import contextlib
        import io

        from watermark import log_consent_attestation
        with contextlib.redirect_stderr(io.StringIO()):
            log_consent_attestation(model, voice, source="unit test")

    def _lines(self):
        import watermark
        with open(watermark._CONSENT_LOG_PATH) as fh:
            return fh.readlines()

    def _write(self, lines):
        import watermark
        with open(watermark._CONSENT_LOG_PATH, "w") as fh:
            fh.writelines(lines)

    def test_clean_chain_verifies(self):
        from watermark import verify_audit_chain
        for i in range(4):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        report = verify_audit_chain()
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["entries"], 4)

    def test_editing_a_line_is_detected(self):
        from watermark import verify_audit_chain
        for i in range(4):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        lines = self._lines()
        lines[1] = lines[1].replace("model1", "modelX")
        self._write(lines)
        self.assertFalse(verify_audit_chain()["ok"])

    def test_truncating_the_tail_is_detected_by_the_anchor(self):
        """A hash chain cannot see its own tail being cut — the anchor can."""
        from watermark import verify_audit_chain
        for i in range(4):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        self._write(self._lines()[:2])
        report = verify_audit_chain()
        self.assertFalse(report["ok"])
        self.assertTrue(any("anchor" in issue for issue in report["issues"]), report["issues"])

    def test_lawful_erasure_leaves_a_verifiable_chain(self):
        """GDPR Art. 17 must not permanently break the evidence trail."""
        from watermark import erase_audit_log, verify_audit_chain
        for i in range(5):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        self.assertEqual(erase_audit_log(subject="speaker2"), 1)
        report = verify_audit_chain()
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["rebuilds"], 1)
        self.assertNotIn("speaker2", "".join(self._lines()),
                         "the erased subject must be gone")

    def test_rebuild_record_does_not_reintroduce_the_subject(self):
        """The record of an erasure must not re-add the data it erased."""
        from watermark import erase_audit_log
        for i in range(3):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        erase_audit_log(subject="speaker1")
        rebuilds = [ln for ln in self._lines() if ln.startswith("[CHAIN-REBUILT]")]
        self.assertEqual(len(rebuilds), 1)
        self.assertNotIn("speaker1", rebuilds[0])

    def test_tampering_after_a_lawful_rebuild_is_still_detected(self):
        """Re-chaining must not become a way to launder later edits."""
        from watermark import erase_audit_log, verify_audit_chain
        for i in range(4):
            self._log(f"model{i}", f"/refs/speaker{i}.wav")
        erase_audit_log(subject="speaker1")
        lines = self._lines()
        lines[0] = lines[0].replace("model0", "modelX")
        self._write(lines)
        self.assertFalse(verify_audit_chain()["ok"])

    def test_absent_log_is_not_an_error(self):
        from watermark import verify_audit_chain
        report = verify_audit_chain()
        self.assertTrue(report["ok"])
        self.assertEqual(report["entries"], 0)

    def test_a_log_written_before_chaining_is_not_reported_as_tampered(self):
        """Upgrading must not accuse every existing user of tampering.

        Entries written before v0.9.10 carry no chain hash. They are counted as
        legacy and folded into the head so later appends are covered, but they
        are not evidence of anything having been edited.
        """
        from watermark import verify_audit_chain
        self._write([
            '[CONSENT] ts=2026-01-01T00:00:00+0000 model=m voice=v attestation="old"\n',
            '[CONSENT] ts=2026-01-02T00:00:00+0000 model=m voice=v attestation="old"\n',
        ])
        report = verify_audit_chain()
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["legacy"], 2)

    def test_appending_to_a_legacy_log_covers_everything_after(self):
        from watermark import verify_audit_chain
        self._write(['[CONSENT] ts=2026-01-01T00:00:00+0000 model=m voice=v attestation="old"\n'])
        self._log("new_model", "/refs/new.wav")
        self.assertTrue(verify_audit_chain()["ok"])
        # Editing the legacy line now breaks the chain, because the new entry
        # committed to it.
        lines = self._lines()
        lines[0] = lines[0].replace("model=m", "model=TAMPERED")
        self._write(lines)
        self.assertFalse(verify_audit_chain()["ok"])


@requires_soundfile
class TestDetectionReport(unittest.TestCase):
    """`--detect-watermark` must not read one backend's number off another's dial.

    The CLI applied a fixed 0.65/0.4 pair of bands to whatever score came back,
    but three backends can produce it and they are not on one scale: the
    spread-spectrum reading is a calibrated statistic, AudioSeal's saturates at
    0.000/1.000, WavMark returns a payload match ratio. The 0.4 boundary was
    also tied to the *old* detector's ~0.44 noise floor and outlived it.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, pcm, sr=24000):
        import soundfile as sf
        path = os.path.join(self.tmp, name)
        sf.write(path, pcm, sr)
        return path

    def _speech(self, seconds=4.0, sr=24000, seed=5):
        rng = np.random.default_rng(seed)
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        tone = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.12 * np.sin(2 * np.pi * 650 * t)
        env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)).astype(np.float32)
        return (tone * env + 0.01 * rng.standard_normal(len(t))).astype(np.float32)

    def test_reports_backend_and_threshold(self):
        from watermark import describe_detection
        report = describe_detection(self._write("a.wav", self._speech()))
        self.assertIsNotNone(report)
        for key in ("confidence", "backend", "threshold", "verdict", "caveat"):
            self.assertIn(key, report)

    def test_marked_and_unmarked_get_opposite_verdicts(self):
        from watermark import describe_detection, spread_spectrum_embed
        pcm = self._speech()
        clean = describe_detection(self._write("clean.wav", pcm))
        marked = describe_detection(self._write("marked.wav", spread_spectrum_embed(pcm)))
        self.assertTrue(marked["verdict"].startswith("AI-GENERATED"))
        self.assertFalse(clean["verdict"].startswith("AI-GENERATED"))

    def test_negative_result_carries_the_caveat(self):
        """Not finding a mark is weak evidence, and must say so."""
        from watermark import describe_detection
        report = describe_detection(self._write("q.wav", self._speech()))
        self.assertIn("NOT evidence", report["caveat"])

    def test_missing_file_reports_none(self):
        from watermark import describe_detection
        self.assertIsNone(describe_detection(os.path.join(self.tmp, "nope.wav")))

    def test_saturating_backends_have_no_uncertain_band(self):
        """AudioSeal reads 0.000 or 1.000, so an "inconclusive" band is a fiction."""
        from watermark import _DETECT_UNCERTAIN_FLOOR
        self.assertIsNone(_DETECT_UNCERTAIN_FLOOR["audioseal_python"])
        self.assertIsNone(_DETECT_UNCERTAIN_FLOOR["audioseal_crispasr"])
        self.assertIsNotNone(_DETECT_UNCERTAIN_FLOOR["spread_spectrum"])


class TestFrameBlocking(unittest.TestCase):
    """The batched FFT path must not depend on where the block boundary falls.

    embed and detect process frames in blocks of _FRAME_BLOCK to bound memory.
    Overlap-add spans block boundaries, so a bug there would show up only on
    signals long enough to need more than one block — which the rest of the
    suite, built on 1-2 s clips, would never reach.
    """

    def _speech_like(self, n, seed=3):
        rng = np.random.default_rng(seed)
        t = np.arange(n, dtype=np.float32) / 24000.0
        tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 700 * t)
        return (tone * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
                + 0.01 * rng.standard_normal(n)).astype(np.float32)

    def test_detects_across_block_boundaries(self):
        from watermark import _FFT_SIZE, _FRAME_BLOCK, _HOP, spread_spectrum_detect, spread_spectrum_embed
        # Comfortably more than one block of frames
        n = _FFT_SIZE + _HOP * (_FRAME_BLOCK * 2 + 7)
        pcm = self._speech_like(n)
        conf = spread_spectrum_detect(spread_spectrum_embed(pcm))
        self.assertGreater(conf, 0.65, f"multi-block embed not detected ({conf:.3f})")

    def test_block_size_does_not_change_the_result(self):
        """Same audio, different block size — same samples out."""
        import watermark as wmod
        n = wmod._FFT_SIZE + wmod._HOP * 300
        pcm = self._speech_like(n)
        original = wmod._FRAME_BLOCK
        try:
            wmod._FRAME_BLOCK = 4096  # one block
            whole = wmod.spread_spectrum_embed(pcm)
            wmod._FRAME_BLOCK = 7  # many small blocks, boundaries everywhere
            chopped = wmod.spread_spectrum_embed(pcm)
        finally:
            wmod._FRAME_BLOCK = original
        np.testing.assert_allclose(whole, chopped, rtol=1e-5, atol=1e-6)


class TestDispatcher(unittest.TestCase):
    """Test the watermark_embed/detect dispatcher."""

    def test_dispatcher_uses_spread_spectrum_by_default(self):
        from watermark import watermark_detect, watermark_embed
        pcm = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
        )
        wm = watermark_embed(pcm)
        self.assertEqual(len(wm), len(pcm))
        conf = watermark_detect(wm)
        self.assertGreater(conf, 0.6)

    def test_no_watermark_env_var(self):
        """CRISPTTS_NO_WATERMARK should disable watermarking."""
        from watermark import watermark_embed
        pcm = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
        )
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
        try:
            wm = watermark_embed(pcm)
            np.testing.assert_array_equal(wm, pcm)
        finally:
            del os.environ["CRISPTTS_NO_WATERMARK"]


class TestWavMetadata(unittest.TestCase):
    """Test WAV LIST/INFO metadata injection."""

    def _make_minimal_wav(self) -> bytes:
        """Create a minimal valid WAV file (1 second of silence at 16 kHz)."""
        sr = 16000
        n_samples = sr
        data_size = n_samples * 2
        riff_size = 36 + data_size
        wav = bytearray()
        wav.extend(b"RIFF")
        wav.extend(struct.pack("<I", riff_size))
        wav.extend(b"WAVE")
        wav.extend(b"fmt ")
        wav.extend(struct.pack("<I", 16))      # fmt chunk size
        wav.extend(struct.pack("<H", 1))       # PCM
        wav.extend(struct.pack("<H", 1))       # mono
        wav.extend(struct.pack("<I", sr))      # sample rate
        wav.extend(struct.pack("<I", sr * 2))  # byte rate
        wav.extend(struct.pack("<H", 2))       # block align
        wav.extend(struct.pack("<H", 16))      # bits per sample
        wav.extend(b"data")
        wav.extend(struct.pack("<I", data_size))
        wav.extend(b"\x00" * data_size)
        return bytes(wav)

    def test_inject_wav_metadata(self):
        from watermark import inject_wav_metadata
        wav = self._make_minimal_wav()
        result = inject_wav_metadata(wav)
        self.assertGreater(len(result), len(wav))
        self.assertIn(b"LIST", result)
        self.assertIn(b"INFO", result)
        self.assertIn(b"CrispTTS", result)
        self.assertIn(b"AI-generated", result)

    def test_riff_size_patched(self):
        from watermark import inject_wav_metadata
        wav = self._make_minimal_wav()
        result = inject_wav_metadata(wav)
        riff_size = struct.unpack_from("<I", result, 4)[0]
        self.assertEqual(riff_size, len(result) - 8)

    def test_non_wav_unchanged(self):
        from watermark import inject_wav_metadata
        data = b"not a wav file"
        self.assertEqual(inject_wav_metadata(data), data)


class TestMp3Metadata(unittest.TestCase):
    """Test MP3 ID3v2 metadata generation."""

    def test_make_id3v2_tag_structure(self):
        from watermark import make_id3v2_ai_tag
        tag = make_id3v2_ai_tag()
        self.assertTrue(tag.startswith(b"ID3"))
        self.assertEqual(tag[3], 0x03)  # version 2.3
        self.assertIn(b"AI_GENERATED", tag)
        self.assertIn(b"CrispTTS", tag)

    def test_inject_mp3_metadata(self):
        from watermark import inject_mp3_metadata
        fake_mp3 = b"\xff\xfb" + b"\x00" * 100  # fake MP3 sync
        result = inject_mp3_metadata(fake_mp3)
        self.assertTrue(result.startswith(b"ID3"))
        self.assertTrue(result.endswith(fake_mp3))

    def test_no_double_tag(self):
        from watermark import inject_mp3_metadata
        fake_mp3 = b"ID3" + b"\x00" * 100
        result = inject_mp3_metadata(fake_mp3)
        self.assertEqual(result, fake_mp3)


class TestAudioSealPythonBackend(unittest.TestCase):
    """Test audioseal Python package integration (skipped if not installed)."""

    def test_load_audioseal_python_missing(self):
        """load_audioseal_python returns False if package not installed."""
        from watermark import load_audioseal_python
        # This will return False if audioseal is not installed, True if it is
        result = load_audioseal_python()
        self.assertIsInstance(result, bool)

    def test_backend_name(self):
        """Backend should be a known string."""
        from watermark import _backend
        self.assertIn(_backend, ("spread_spectrum", "audioseal_python", "audioseal_crispasr", "wavmark"))

    def test_audioseal_is_tried_before_wavmark(self):
        """Lazy-load must prefer AudioSeal, and only fall back to WavMark.

        The order is what makes marking affordable: WavMark's detect did not
        return within 10 minutes on 3 s of audio, and mark_audio_file() detects
        after every embed. Asserted by observing which loader the dispatcher
        calls first, so it fails if the two `if` branches are ever swapped
        back.
        """
        import numpy as np

        import watermark as wmod

        calls = []
        orig_as, orig_wm = wmod.load_audioseal_python, wmod.load_wavmark
        orig_backend, orig_gen, orig_model = (
            wmod._backend, wmod._audioseal_generator, wmod._wavmark_model)
        try:
            wmod._backend = "spread_spectrum"
            wmod._audioseal_generator = None
            wmod._wavmark_model = None
            wmod.load_audioseal_python = lambda: (calls.append("audioseal"), False)[1]
            wmod.load_wavmark = lambda: (calls.append("wavmark"), False)[1]
            wmod.watermark_embed(np.zeros(4096, dtype=np.float32), sample_rate=16000)
        finally:
            wmod.load_audioseal_python, wmod.load_wavmark = orig_as, orig_wm
            wmod._backend = orig_backend
            wmod._audioseal_generator = orig_gen
            wmod._wavmark_model = orig_model

        self.assertEqual(calls, ["audioseal", "wavmark"],
                         "AudioSeal must be attempted before WavMark")

    def test_audioseal_loader_disables_torchdynamo(self):
        """Dynamo must be off before audioseal is imported.

        AudioSeal's SEANet layers go through torch.compile and CrispTTS feeds a
        new tensor shape almost every run, so Dynamo recompiles rather than
        reusing a graph: measured 56.5 s vs 2.0 s for one cold 10 s embed.
        """
        import os

        from watermark import load_audioseal_python

        prior = os.environ.pop("TORCHDYNAMO_DISABLE", None)
        try:
            load_audioseal_python()
            self.assertEqual(os.environ.get("TORCHDYNAMO_DISABLE"), "1")
        finally:
            if prior is None:
                os.environ.pop("TORCHDYNAMO_DISABLE", None)
            else:
                os.environ["TORCHDYNAMO_DISABLE"] = prior


class TestC2PA(unittest.TestCase):
    """Test C2PA signing integration."""

    def test_c2pa_sign_no_cert(self):
        """c2pa_sign_file returns False when no cert is configured."""
        from watermark import c2pa_sign_file
        # Clear env vars to ensure no cert is set
        env_backup = {}
        for k in ("C2PA_CERT_PATH", "C2PA_KEY_PATH"):
            env_backup[k] = os.environ.pop(k, None)
        try:
            result = c2pa_sign_file("/nonexistent/file.wav")
            self.assertFalse(result)
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_manifest_asserts_ai_generation(self):
        """The manifest must carry the assertion that makes it AI provenance."""
        from watermark import _c2pa_manifest
        manifest = _c2pa_manifest("f5_tts_german")
        actions = next(a for a in manifest["assertions"] if a["label"] == "c2pa.actions")
        created = actions["data"]["actions"][0]
        self.assertEqual(created["action"], "c2pa.created")
        self.assertTrue(created["digitalSourceType"].endswith("trainedAlgorithmicMedia"))
        self.assertEqual(created["softwareAgent"]["softwareAgentModel"], "f5_tts_german")

    def test_manifest_is_json_serializable(self):
        import json

        from watermark import _c2pa_manifest
        json.loads(json.dumps(_c2pa_manifest()))


class TestConsentGate(unittest.TestCase):
    """Test voice-cloning consent gate."""

    def test_cloning_model_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("llasa_hybrid_de_zeroshot",
                                         "llasa_hybrid"))

    def test_regular_model_no_consent(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent("edge", "edge"))

    def test_keyword_detection(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("my_custom_zeroshot_model", "custom_handler"))
        self.assertTrue(requires_consent("coqui_xtts_v2", "custom_handler"))

    def test_handler_key_detection(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("some_model", "f5_tts"))


class TestResampling(unittest.TestCase):
    """Test linear interpolation resampling for AudioSeal."""

    def test_identity(self):
        from watermark import _resample_linear
        pcm = np.random.randn(1000).astype(np.float32)
        result = _resample_linear(pcm, 16000, 16000)
        np.testing.assert_array_equal(result, pcm)

    def test_downsample_length(self):
        from watermark import _resample_linear
        pcm = np.random.randn(24000).astype(np.float32)
        result = _resample_linear(pcm, 24000, 16000)
        self.assertEqual(len(result), 16000)

    def test_upsample_length(self):
        from watermark import _resample_linear
        pcm = np.random.randn(16000).astype(np.float32)
        result = _resample_linear(pcm, 16000, 24000)
        self.assertEqual(len(result), 24000)


class TestConsentLogging(unittest.TestCase):
    """Test consent attestation logging."""

    def test_log_consent_attestation(self):
        import io
        import sys

        from watermark import log_consent_attestation
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            log_consent_attestation("test_model", "test_voice")
        finally:
            sys.stderr = old_stderr
        output = captured.getvalue()
        self.assertIn("[CONSENT]", output)
        self.assertIn("test_model", output)
        self.assertIn("test_voice", output)
        self.assertIn("--i-have-rights", output)


@requires_soundfile
class TestSpokenDisclaimer(unittest.TestCase):
    """Test spoken disclaimer generation."""

    def test_generate_spoken_disclaimer_returns_audio_and_kind(self):
        from watermark import generate_spoken_disclaimer
        # At minimum the tone-marker fallback, even with no TTS backend.
        pcm, kind = generate_spoken_disclaimer(sample_rate=24000)
        self.assertIsNotNone(pcm)
        self.assertIsInstance(pcm, np.ndarray)
        self.assertGreater(len(pcm), 0)
        # The kind is what tells callers whether this is a real disclosure.
        self.assertIn(kind, ("spoken", "tone-marker"))

    def test_prepend_disclaimer(self):
        from watermark import prepend_disclaimer
        pcm = np.random.randn(24000).astype(np.float32)
        result, kind = prepend_disclaimer(pcm, sample_rate=24000)
        # Result should be longer than original (disclaimer + silence + original)
        self.assertGreater(len(result), len(pcm))
        # Original audio should be at the end
        np.testing.assert_array_equal(result[-len(pcm):], pcm)
        self.assertIn(kind, ("spoken", "tone-marker"))


@requires_soundfile
class TestWatermarkVerification(unittest.TestCase):
    """Test post-embed watermark verification."""

    def test_verify_file(self):
        import tempfile

        from watermark import spread_spectrum_detect, spread_spectrum_embed
        try:
            import soundfile as sf_test
        except ImportError:
            self.skipTest("soundfile not installed")
        pcm = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
        )
        wm_pcm = spread_spectrum_embed(pcm)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf_test.write(f.name, wm_pcm, 24000)
            # Read back and verify with spread-spectrum detector directly
            # (watermark_verify_file may dispatch to AudioSeal which
            # can't detect spread-spectrum watermarks)
            data, sr = sf_test.read(f.name, dtype="float32")
        os.unlink(f.name)
        confidence = spread_spectrum_detect(data)
        self.assertGreater(confidence, 0.6)

    def test_verify_nonexistent_file(self):
        from watermark import watermark_verify_file
        result = watermark_verify_file("/nonexistent/file.wav")
        self.assertIsNone(result)


class TestWavMarkBackend(unittest.TestCase):
    """Test WavMark (MIT) neural watermark integration."""

    def test_load_wavmark(self):
        """load_wavmark returns a bool."""
        from watermark import load_wavmark
        result = load_wavmark()
        self.assertIsInstance(result, bool)

    def test_wavmark_payload_is_16bit(self):
        """The fixed WavMark payload should be exactly 16 bits."""
        from watermark import _WAVMARK_PAYLOAD
        self.assertEqual(len(_WAVMARK_PAYLOAD), 16)
        for bit in _WAVMARK_PAYLOAD:
            self.assertIn(int(bit), (0, 1))

    def test_loader_prefers_accelerator_over_cpu(self):
        """MPS must be chosen when CUDA is absent — not skipped for CPU.

        The loader used to read `cuda:0 if cuda.is_available() else cpu`, so
        every Apple Silicon machine took the slowest device it had. Measured on
        one 1 s chunk: 16-30 s on CPU at torch's default thread count, 0.54 s
        on MPS.
        """
        import importlib.util
        if importlib.util.find_spec("wavmark") is None:
            self.skipTest("wavmark not installed")
        import torch

        import watermark as wmod

        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            self.skipTest("no MPS on this machine")
        if torch.cuda.is_available():
            self.skipTest("CUDA present; it legitimately outranks MPS")

        saved = (wmod._wavmark_model, wmod._wavmark_device, wmod._backend)
        try:
            self.assertTrue(wmod.load_wavmark())
            self.assertEqual(wmod._wavmark_device.type, "mps")
        finally:
            wmod._wavmark_model, wmod._wavmark_device, wmod._backend = saved

    def test_fast_scan_agrees_with_upstream_decode(self):
        """The early-exit scan must reach the same verdict as wavmark's own.

        It stops at the first exact start-bit match instead of scanning every
        window and averaging, so this asserts the shortcut does not change the
        answer — on marked audio *and* on unmarked audio, where there is no hit
        to stop at and the full scan runs either way.
        """
        import importlib.util
        if importlib.util.find_spec("wavmark") is None:
            self.skipTest("wavmark not installed")

        import wavmark

        import watermark as wmod

        saved = (wmod._wavmark_model, wmod._wavmark_device, wmod._backend)
        try:
            self.assertTrue(wmod.load_wavmark())
            rng = np.random.default_rng(7)
            # 3 s of noise-like signal: long enough for several scan windows
            clean = (0.1 * rng.standard_normal(16000 * 3)).astype(np.float32)
            marked = wmod._embed_wavmark(clean, 16000)

            fast_marked = wmod._detect_wavmark(marked, 16000)
            fast_clean = wmod._detect_wavmark(clean, 16000)

            up_payload, _ = wavmark.decode_watermark(
                wmod._wavmark_model, marked.astype(np.float64), show_progress=False)
            self.assertIsNotNone(up_payload, "upstream did not find its own watermark")
            up_marked = float(np.mean(up_payload[:16] == wmod._WAVMARK_PAYLOAD))

            # Not asserted equal: upstream averages every exact match in the
            # file, the fast scan averages those in the first batch containing
            # one, so they can differ by a bit or two on noisy input. What must
            # agree is the verdict, and the shortcut must not be the *worse*
            # reading of the two by any meaningful margin.
            self.assertGreater(fast_marked, 0.65,
                               f"fast scan missed a real mark ({fast_marked:.3f})")
            self.assertGreater(up_marked, 0.65)
            self.assertGreaterEqual(fast_marked, up_marked - 0.07)
            self.assertLess(fast_clean, 0.4, "unmarked audio must not read as marked")
        finally:
            wmod._wavmark_model, wmod._wavmark_device, wmod._backend = saved

    def test_wavmark_payload_encodes_ct(self):
        """Payload should encode 'CT' = 0x43 0x54."""
        from watermark import _WAVMARK_PAYLOAD
        # C = 0x43 = 0100_0011, T = 0x54 = 0101_0100
        expected = [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
        for i, (got, exp) in enumerate(zip(_WAVMARK_PAYLOAD, expected, strict=True)):
            self.assertEqual(int(got), exp, f"Bit {i}: expected {exp}, got {int(got)}")

    def test_declared_watermark_floors_are_installable(self):
        """Every watermark extra must name a version that actually exists.

        `robust` pinned wavmark>=0.3.0 until v0.9.4 — a transposition of 0.0.3,
        the highest version ever published — so `pip install crisptts[robust]`
        failed to resolve. That is the command the README gives for Opus/OGG
        output, where a neural watermark is required rather than optional, so
        the documented route to a legal Opus file did not work.

        Checked against installed distributions rather than the network, so
        each package is skipped where it is absent (the default install has
        neither). Whichever extras a machine does have get verified.
        """
        import re
        from importlib.metadata import PackageNotFoundError, version

        extras = read_optional_dependencies()

        def parts(v):
            return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

        checked = 0
        for extra, specs in extras.items():
            for spec in specs:
                pkg = re.match(r"^[A-Za-z0-9._-]+", spec).group(0)
                if pkg not in ("wavmark", "audioseal"):
                    continue
                try:
                    installed = version(pkg)
                except PackageNotFoundError:
                    continue
                floor = re.search(r">=\s*([\d.]+)", spec)
                self.assertIsNotNone(floor, f"{extra}: no floor in {spec!r}")
                self.assertLessEqual(
                    parts(floor.group(1)), parts(installed),
                    f"extra {extra!r} pins {spec!r} but the installed {pkg} is "
                    f"{installed}; no release satisfies that floor")
                checked += 1
        if not checked:
            self.skipTest("neither wavmark nor audioseal installed")

    def test_robust_extra_names_the_preferred_backend(self):
        """`robust` must install whichever backend the dispatcher prefers.

        The two are set independently — one in pyproject.toml, one in
        watermark.py — so they can drift. If `robust` installed WavMark while
        the dispatcher preferred AudioSeal, the documented "recommended"
        install would silently leave the preferred backend absent.
        """
        robust = read_optional_dependencies()["robust"]
        self.assertTrue(
            any(s.startswith("audioseal") for s in robust),
            f"the dispatcher prefers AudioSeal, but `robust` installs {robust}")


class TestVoiceCloningKeywords(unittest.TestCase):
    """Test expanded voice-cloning detection for CrispASR backends."""

    def test_crispasr_cloning_backends_detected(self):
        from watermark import requires_consent
        for model_id in ("crispasr_vibevoice_tts", "crispasr_indextts",
                         "crispasr_voxcpm2", "crispasr_qwen3_tts"):
            self.assertTrue(requires_consent(model_id, "crispasr"),
                            f"{model_id} should require consent")

    def test_wav_voice_path_triggers_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_kokoro", "crispasr", "/path/to/ref.wav"))

    def test_non_cloning_crispasr_no_consent(self):
        from watermark import requires_consent
        # kokoro with a named voice (not a .wav path) is not cloning
        self.assertFalse(requires_consent("crispasr_kokoro", "crispasr", "af_heart"))


class TestPersistentAuditLog(unittest.TestCase):
    """Test that consent attestations are written to a persistent log file."""

    def test_audit_log_written(self):
        import io
        import sys

        from watermark import _CONSENT_LOG_PATH, log_consent_attestation

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            log_consent_attestation("test_audit_model", "test_voice", source="unit test")
        finally:
            sys.stderr = old_stderr

        self.assertTrue(os.path.isfile(_CONSENT_LOG_PATH),
                        f"Audit log should exist at {_CONSENT_LOG_PATH}")
        with open(_CONSENT_LOG_PATH) as f:
            content = f.read()
        self.assertIn("test_audit_model", content)
        self.assertIn("unit test", content)


class TestFlacMetadata(unittest.TestCase):
    """Test FLAC Vorbis comment metadata injection."""

    def test_inject_flac_returns_bool(self):
        from watermark import inject_flac_metadata
        # Should return False for nonexistent file (graceful failure)
        result = inject_flac_metadata("/nonexistent/file.flac")
        self.assertIsInstance(result, bool)
        self.assertFalse(result)


class TestOpusMetadata(unittest.TestCase):
    """Test Opus/OGG Vorbis comment metadata injection."""

    def test_inject_opus_returns_bool(self):
        from watermark import inject_opus_metadata
        # Should return False for nonexistent file (graceful failure)
        result = inject_opus_metadata("/nonexistent/file.opus")
        self.assertIsInstance(result, bool)
        self.assertFalse(result)


class TestWatermarkEmbedDispatcher(unittest.TestCase):
    """Test the full watermark_embed dispatcher with different sample rates."""

    def _make_sine(self, sr=24000, duration=1.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * 440 * t)

    def test_embed_preserves_length(self):
        from watermark import watermark_embed
        pcm = self._make_sine(sr=24000)
        result = watermark_embed(pcm, sample_rate=24000)
        self.assertEqual(len(result), len(pcm))

    def test_embed_at_16khz(self):
        from watermark import watermark_embed
        pcm = self._make_sine(sr=16000)
        result = watermark_embed(pcm, sample_rate=16000)
        self.assertEqual(len(result), len(pcm))

    def test_embed_at_44100(self):
        from watermark import watermark_embed
        pcm = self._make_sine(sr=44100)
        result = watermark_embed(pcm, sample_rate=44100)
        self.assertEqual(len(result), len(pcm))

    def test_embed_returns_new_array(self):
        """watermark_embed should return a new array, not modify in place."""
        from watermark import watermark_embed
        pcm = self._make_sine()
        original = pcm.copy()
        _ = watermark_embed(pcm)
        np.testing.assert_array_equal(pcm, original)

    def test_default_alpha_is_the_band_default_not_legacy(self):
        """An unspecified alpha must follow the active band, not a hardcoded value.

        The comb moved into the speech band (alpha 0.05) but watermark_embed
        kept the legacy band's 0.08 as its signature default, so every real
        embed ran 1.6x hotter than designed and lost 3-4 dB of SNR. Asserting
        equivalence to the resolved band default keeps the two in step if the
        band is ever retuned again.
        """
        from watermark import _FFT_SIZE, spread_spectrum_embed, watermark_embed, wm_params
        force_spread_spectrum(self)  # alpha only reaches the built-in backend
        band_alpha = wm_params(_FFT_SIZE)[2]
        pcm = self._make_sine(sr=24000, duration=2.0)
        np.testing.assert_allclose(
            watermark_embed(pcm, sample_rate=24000),
            spread_spectrum_embed(pcm, alpha=band_alpha),
            rtol=1e-6, atol=1e-6,
            err_msg="watermark_embed() is not using the active band's default alpha")

    def test_default_alpha_is_quieter_than_legacy(self):
        """The band default must be measurably gentler than the legacy 0.08."""
        from watermark import spread_spectrum_embed, watermark_embed
        force_spread_spectrum(self)  # alpha only reaches the built-in backend

        def snr(clean, marked):
            return 10 * np.log10(np.sum(clean ** 2) / np.sum((marked - clean) ** 2))

        pcm = self._make_sine(sr=24000, duration=2.0)
        self.assertGreater(
            snr(pcm, watermark_embed(pcm, sample_rate=24000)),
            snr(pcm, spread_spectrum_embed(pcm, alpha=0.08)),
            "default embed should be quieter than the legacy alpha")


class TestC2paSigning(unittest.TestCase):
    """C2PA signing works out of the box, with no user-supplied credential."""

    def test_c2pa_sign_returns_bool(self):
        from watermark import c2pa_sign_file
        self.assertIsInstance(c2pa_sign_file("/nonexistent/file.wav"), bool)

    def test_missing_file_fails_cleanly(self):
        from watermark import c2pa_sign_file
        self.assertFalse(c2pa_sign_file("/nonexistent/file.wav"))

    @requires_c2pa
    @requires_soundfile
    def test_bundled_credential_signs_without_cert(self):
        """The default install must produce a manifest, not skip signing."""
        import tempfile

        from watermark import c2pa_sign_file_ex
        env_backup = {k: os.environ.pop(k, None) for k in ("C2PA_CERT_PATH", "C2PA_KEY_PATH")}
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "s.wav")
                _write_tone_wav(path, seconds=1.0)
                ok, signer = c2pa_sign_file_ex(path, model_id="test_model")
                self.assertTrue(ok, "bundled credential must sign without configuration")
                self.assertEqual(signer, "self-signed")
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    @requires_c2pa
    @requires_soundfile
    def test_signed_manifest_reads_back_as_ai_generated(self):
        """A verifier must be able to read the AI assertion back out."""
        import json
        import tempfile

        import c2pa

        from watermark import c2pa_sign_file_ex
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.wav")
            _write_tone_wav(path, seconds=1.0)
            ok, _ = c2pa_sign_file_ex(path, model_id="f5_tts_german")
            self.assertTrue(ok)
            report = json.loads(c2pa.Reader(path).json())
            active = report["manifests"][report["active_manifest"]]
            sources = [
                action["digitalSourceType"]
                for assertion in active["assertions"]
                if assertion["label"].startswith("c2pa.actions")
                for action in assertion["data"]["actions"]
                if "digitalSourceType" in action
            ]
            self.assertTrue(any(s.endswith("trainedAlgorithmicMedia") for s in sources),
                            f"no AI-generation assertion in manifest: {sources}")

    @requires_c2pa
    @requires_soundfile
    def test_every_capable_ext_really_signs(self):
        """C2PA_CAPABLE_EXTS feeds the watermark floor, so it must not overclaim.

        An extension listed here but unsignable would let --no-watermark be
        honoured for output that then carries no manifest at all.
        """
        import tempfile

        import soundfile as sf

        from watermark import C2PA_CAPABLE_EXTS, c2pa_sign_file_ex
        pcm = 0.3 * np.sin(
            2 * np.pi * 180 * np.linspace(0, 2, 48000, endpoint=False, dtype=np.float32))
        with tempfile.TemporaryDirectory() as d:
            for ext in sorted(C2PA_CAPABLE_EXTS):
                path = os.path.join(d, f"s{ext}")
                if ext == ".wav":
                    sf.write(path, pcm, 24000, subtype="PCM_16")
                else:
                    try:
                        from pydub import AudioSegment
                    except ImportError:
                        self.skipTest("pydub not installed")
                    raw = (pcm * 32767).astype(np.int16).tobytes()
                    fmt = {".mp3": "mp3", ".m4a": "ipod", ".flac": "flac"}[ext]
                    try:
                        AudioSegment(data=raw, sample_width=2, frame_rate=24000,
                                     channels=1).export(path, format=fmt)
                    except (FileNotFoundError, OSError):
                        # Encoding these needs ffmpeg. CI installs it so this
                        # assertion really runs there; skip on checkouts without.
                        self.skipTest(f"ffmpeg not available to encode {ext}")
                ok, _ = c2pa_sign_file_ex(path, model_id="t")
                self.assertTrue(ok, f"{ext} is in C2PA_CAPABLE_EXTS but cannot be signed")
                from watermark import manifest_asserts_ai
                self.assertTrue(manifest_asserts_ai(path),
                                f"{ext} was signed but the manifest does not assert "
                                "trainedAlgorithmicMedia")

    @requires_c2pa
    @requires_soundfile
    def test_streaming_sign_covers_what_sign_file_refuses(self):
        """FLAC and M4A are why the signer uses Builder.sign(), not sign_file().

        c2pa-rs advertises both in get_supported_mime_types() but sign_file()
        returns "NotSupported: type is unsupported" for them, which is what
        kept them out of C2PA_CAPABLE_EXTS. The streaming API signs them. If a
        future c2pa-python makes sign_file() work too this test still passes —
        it pins the capability, not the workaround.
        """
        import tempfile

        import soundfile as sf

        from c2pa_dev_cert import DEV_CERT_CHAIN_PEM, DEV_PRIVATE_KEY_PEM
        from watermark import _sign_with_c2pa_python as sign
        from watermark import manifest_asserts_ai

        pcm = 0.3 * np.sin(
            2 * np.pi * 180 * np.linspace(0, 2, 48000, endpoint=False, dtype=np.float32))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.flac")
            sf.write(path, pcm.astype(np.float32), 24000)
            self.assertTrue(
                sign(path, path, DEV_CERT_CHAIN_PEM.encode(),
                     DEV_PRIVATE_KEY_PEM.encode(), "t"),
                "FLAC signing failed; the streaming signer has regressed")
            self.assertTrue(manifest_asserts_ai(path))
            self.assertGreater(os.path.getsize(path), 0)
            sf.info(path)  # still decodable audio, not just a signed blob


class TestComplianceCoverage(unittest.TestCase):
    """Verify all output paths have watermark coverage."""

    def test_wav_watermark_roundtrip(self):
        """WAV files should have detectable watermark after embed."""
        from watermark import spread_spectrum_detect, watermark_embed
        # Asserts with the *spread-spectrum* detector specifically, so the
        # embed has to be the spread-spectrum one for the pair to match.
        force_spread_spectrum(self)
        pcm = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
        )
        wm = watermark_embed(pcm, sample_rate=24000)
        conf = spread_spectrum_detect(wm)
        self.assertGreater(conf, 0.6,
                           f"WAV watermark should be detectable (confidence={conf:.3f})")

    def test_wav_metadata_contains_ai_tag(self):
        """WAV metadata should contain AI-generated declaration."""
        # Minimal WAV
        import struct

        from watermark import inject_wav_metadata
        sr = 16000
        data_size = sr * 2
        riff_size = 36 + data_size
        wav = bytearray()
        wav.extend(b"RIFF")
        wav.extend(struct.pack("<I", riff_size))
        wav.extend(b"WAVE")
        wav.extend(b"fmt ")
        wav.extend(struct.pack("<I", 16))
        wav.extend(struct.pack("<HHI I HH", 1, 1, sr, sr * 2, 2, 16))
        wav.extend(b"data")
        wav.extend(struct.pack("<I", data_size))
        wav.extend(b"\x00" * data_size)
        result = inject_wav_metadata(bytes(wav))
        self.assertIn(b"AI-generated", result)
        self.assertIn(b"CrispTTS", result)

    def test_mp3_metadata_contains_ai_tag(self):
        """MP3 metadata should contain AI_GENERATED tag."""
        from watermark import inject_mp3_metadata
        fake_mp3 = b"\xff\xfb" + b"\x00" * 100
        result = inject_mp3_metadata(fake_mp3)
        self.assertIn(b"AI_GENERATED", result)

    def test_voice_cloning_keywords_comprehensive(self):
        """All known cloning-capable backends should trigger consent."""
        from watermark import requires_consent
        cloning_models = [
            "crispasr_vibevoice_tts", "crispasr_indextts",
            "crispasr_voxcpm2", "crispasr_qwen3_tts",
            "crispasr_dots_tts", "crispasr_cosyvoice3_tts",
            "crispasr_csm_tts", "crispasr_omnivoice_tts",
        ]
        for mid in cloning_models:
            self.assertTrue(requires_consent(mid, "crispasr"),
                            f"{mid} should require consent")
        # Non-cloning should not trigger
        self.assertFalse(requires_consent("crispasr_kokoro", "crispasr", "af_heart"))
        self.assertFalse(requires_consent("edge", "edge"))

    def test_disclaimer_generates_audio(self):
        """Spoken disclaimer should produce non-empty audio."""
        from watermark import generate_spoken_disclaimer
        pcm, kind = generate_spoken_disclaimer(sample_rate=24000)
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm), 100)
        self.assertNotEqual(kind, "none")

    def test_consent_audit_log_path_exists(self):
        """Consent audit log path should be defined."""
        from watermark import _CONSENT_LOG_PATH
        self.assertTrue(_CONSENT_LOG_PATH.endswith("consent_audit.log"))


# ---------------------------------------------------------------------------
# Phase 16: central marking path (EU AI Act Art. 50(2))
# ---------------------------------------------------------------------------

def _write_tone_wav(path, sample_rate=22050, seconds=2.0):
    """Write a short speech-like tone to `path`. Returns the PCM array."""
    import soundfile as sf
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    pcm = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    sf.write(path, pcm, sample_rate)
    return pcm


@requires_soundfile
class TestMarkAudioFile(unittest.TestCase):
    """The single marking entry point used by CLI, --test-all and server."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    def test_marks_wav_and_is_detectable(self):
        import soundfile as sf

        from watermark import mark_audio_file, watermark_detect
        path = self._path("a.wav")
        _write_tone_wav(path)
        result = mark_audio_file(path)
        self.assertTrue(result.marked)
        self.assertIn("audio-watermark", result.layers)
        self.assertIn("metadata", result.layers)
        data, sr = sf.read(path, dtype="float32")
        self.assertGreater(watermark_detect(data, sample_rate=sr), 0.65)

    def test_marking_is_idempotent(self):
        """Re-marking must not embed a second watermark (~6 dB SNR cost).

        The reason reported depends on which robust layer the first pass left
        behind: with C2PA available the manifest is preserved, without it the
        measured watermark is. Either way the bytes must not change — that is
        the property this test is for. It deliberately does not assert
        ``reason == "already-marked"`` any more: that string used to come from
        an early return that skipped verification entirely, which is the
        defect ``test_metadata_marker_alone_does_not_satisfy_the_gate`` covers.
        """
        from watermark import mark_audio_file
        path = self._path("b.wav")
        _write_tone_wav(path)
        mark_audio_file(path)
        with open(path, "rb") as f:
            after_first = f.read()
        second = mark_audio_file(path)
        with open(path, "rb") as f:
            after_second = f.read()
        self.assertTrue(second.marked)
        self.assertEqual(after_first, after_second, "re-marking rewrote the file")
        self.assertNotIn("audio-watermark", second.layers,
                         "a second watermark was embedded over the first")

    def test_metadata_marker_alone_does_not_satisfy_the_gate(self):
        """Regression: is_marked() used to short-circuit the whole function.

        A file carrying only the container marker — which is what the CrispASR
        streaming path produces, injecting the LIST/INFO chunk itself before
        returning — was reported ``marked=True`` with no watermark embedded, no
        manifest, and the verification gate never run. Measured confidence on
        the delivered file was 0.625, below the 0.65 threshold.

        The marker may now suppress redundant work, but it must never stand in
        for evidence: either a robust layer is really there, or this raises.
        """
        import soundfile as sf

        import watermark
        from watermark import (
            MarkingError,
            _inject_container_metadata,
            is_marked,
            mark_audio_file,
            watermark_detect,
        )
        # No C2PA, so the watermark is the only robust layer available and the
        # gate has to rest on it alone.
        original = watermark.c2pa_sign_file_ex
        watermark.c2pa_sign_file_ex = lambda *a, **k: (False, None)
        self.addCleanup(setattr, watermark, "c2pa_sign_file_ex", original)

        path = self._path("metadata_only.wav")
        _write_tone_wav(path)
        _inject_container_metadata(path, ".wav")
        self.assertTrue(is_marked(path), "precondition: the marker is present")

        # CrispASR output: we do not embed for that handler, so nothing else
        # can rescue this file and it must be refused outright.
        with self.assertRaises(MarkingError):
            mark_audio_file(path, handler_key="crispasr")

        # Any other handler: the missing watermark is embedded rather than
        # assumed, and the result verifies.
        path2 = self._path("metadata_only2.wav")
        _write_tone_wav(path2)
        _inject_container_metadata(path2, ".wav")
        result = mark_audio_file(path2)
        self.assertTrue(result.marked)
        self.assertIn("audio-watermark", result.layers)
        data, sr = sf.read(path2, dtype="float32")
        self.assertGreater(watermark_detect(data, sample_rate=sr), 0.65)

    def test_no_duplicate_metadata_chunks(self):
        from watermark import mark_audio_file
        path = self._path("c.wav")
        _write_tone_wav(path)
        mark_audio_file(path)
        mark_audio_file(path)
        with open(path, "rb") as f:
            blob = f.read()
        self.assertEqual(blob.count(b"CrispTTS (AI-generated audio)"), 1)

    def test_uses_true_sample_rate_not_default(self):
        """A non-24 kHz file must be watermarked at its own rate."""
        import soundfile as sf

        from watermark import mark_audio_file, watermark_detect
        for rate in (16000, 44100):
            path = self._path(f"sr{rate}.wav")
            _write_tone_wav(path, sample_rate=rate)
            mark_audio_file(path)
            data, sr = sf.read(path, dtype="float32")
            self.assertEqual(sr, rate)
            self.assertGreater(watermark_detect(data, sample_rate=sr), 0.65,
                               f"watermark not detectable at {rate} Hz")

    def test_fails_closed_on_unsupported_format(self):
        from watermark import MarkingError, mark_audio_file
        path = self._path("x.xyz")
        with open(path, "wb") as f:
            f.write(b"not audio")
        with self.assertRaises(MarkingError):
            mark_audio_file(path)

    def test_allow_unmarked_overrides_fail_closed(self):
        from watermark import mark_audio_file
        path = self._path("y.xyz")
        with open(path, "wb") as f:
            f.write(b"not audio")
        result = mark_audio_file(path, allow_unmarked=True)
        self.assertFalse(result.marked)
        self.assertIsNotNone(result.reason)

    def test_missing_file_fails_closed(self):
        from watermark import MarkingError, mark_audio_file
        with self.assertRaises(MarkingError):
            mark_audio_file(self._path("nope.wav"))

    def test_no_watermark_env_disables_every_layer(self):
        """--no-watermark must be coherent: no PCM mark, no metadata, no C2PA."""
        from watermark import is_marked, mark_audio_file
        path = self._path("d.wav")
        _write_tone_wav(path)
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
        try:
            result = mark_audio_file(path)
        finally:
            del os.environ["CRISPTTS_NO_WATERMARK"]
        self.assertFalse(result.marked)
        self.assertFalse(is_marked(path))

    def test_crispasr_embed_is_verified_not_trusted(self):
        """CrispASR marks in its binary — we still verify it actually did."""
        import soundfile as sf

        from watermark import mark_audio_file, watermark_embed
        path = self._path("e.wav")
        pcm = _write_tone_wav(path)
        # Simulate what the CrispASR binary does before handing us the file.
        sf.write(path, watermark_embed(pcm, sample_rate=22050), 22050)
        result = mark_audio_file(path, handler_key="crispasr")
        self.assertTrue(result.marked)
        # Reported as "upstream" and only because verification found it — the
        # layer is evidence, not a claim about what the binary was asked to do.
        self.assertIn("audio-watermark:upstream", result.layers)
        self.assertIn("metadata", result.layers)
        self.assertGreater(result.confidence, 0.65)

    def test_crispasr_output_without_watermark_is_refused(self):
        """Regression: the binary's marking used to be taken on trust.

        If CrispASR ran with its own --no-watermark, or is an old build, the
        file reaches us unmarked. Trusting handler_key alone would ship it.
        """
        import watermark
        from watermark import MarkingError, mark_audio_file
        original = watermark.c2pa_sign_file_ex
        watermark.c2pa_sign_file_ex = lambda *a, **k: (False, None)
        self.addCleanup(setattr, watermark, "c2pa_sign_file_ex", original)
        path = self._path("e2.wav")
        _write_tone_wav(path)  # never watermarked
        with self.assertRaises(MarkingError):
            mark_audio_file(path, handler_key="crispasr")

    def test_is_marked_false_on_plain_audio(self):
        from watermark import is_marked
        path = self._path("f.wav")
        _write_tone_wav(path)
        self.assertFalse(is_marked(path))


@requires_soundfile
class TestMarkingSufficiencyGate(unittest.TestCase):
    """Generation is gated on marking that is verifiably sufficient.

    Metadata alone never counts: it is stripped by any transcode. The audio
    watermark must be detectable above threshold, or C2PA must have signed.
    """

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, pcm, sr=22050):
        import soundfile as sf
        path = os.path.join(self.tmpdir, name)
        sf.write(path, pcm, sr)
        return path

    def _without_c2pa(self):
        """Disable C2PA so the watermark-sufficiency logic is tested alone.

        C2PA now signs by default, and a signed manifest is itself a robust
        layer — so without this, these cases pass on the manifest and the
        watermark rule they exist to cover is never exercised.
        """
        import watermark
        original = watermark.c2pa_sign_file_ex
        watermark.c2pa_sign_file_ex = lambda *a, **k: (False, None)
        self.addCleanup(setattr, watermark, "c2pa_sign_file_ex", original)

    def test_sub_frame_audio_refused(self):
        """Shorter than one FFT frame: the embed silently no-ops."""
        force_spread_spectrum(self)
        from watermark import MarkingError, mark_audio_file
        self._without_c2pa()
        path = self._write("tiny.wav", np.zeros(400, dtype=np.float32) + 0.1)
        with self.assertRaises(MarkingError) as ctx:
            mark_audio_file(path)
        self.assertIn("not sufficient", str(ctx.exception))

    def test_digital_silence_refused(self):
        force_spread_spectrum(self)
        from watermark import MarkingError, mark_audio_file
        self._without_c2pa()
        path = self._write("silence.wav", np.zeros(22050 * 2, dtype=np.float32))
        with self.assertRaises(MarkingError):
            mark_audio_file(path)

    def test_normal_audio_accepted(self):
        from watermark import mark_audio_file
        t = np.linspace(0, 2, 44100, endpoint=False, dtype=np.float32)
        path = self._write("ok.wav", (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        result = mark_audio_file(path)
        self.assertTrue(result.marked)
        self.assertGreater(result.confidence, 0.65)

    def test_insufficient_but_allowed_when_responsibility_taken(self):
        force_spread_spectrum(self)
        from watermark import mark_audio_file
        self._without_c2pa()
        path = self._write("tiny2.wav", np.zeros(400, dtype=np.float32) + 0.1)
        result = mark_audio_file(path, allow_unmarked=True)
        self.assertFalse(result.marked)

    @requires_c2pa
    def test_c2pa_manifest_alone_is_sufficient(self):
        """Audio too short to watermark is still marked, if C2PA signed it.

        This is what making C2PA a default dependency buys: a signed,
        interoperable manifest is a robust layer in its own right, so output
        that the watermark cannot carry is no longer refused outright.
        """
        from watermark import mark_audio_file
        path = self._write("tiny3.wav", np.zeros(400, dtype=np.float32) + 0.1)
        result = mark_audio_file(path)
        self.assertTrue(result.marked)
        self.assertTrue(any(layer.startswith("c2pa:") for layer in result.layers),
                        f"expected a c2pa layer, got {result.layers}")

    def test_metadata_alone_is_never_sufficient(self):
        """The rule C2PA must not quietly relax: tags are not marking."""
        force_spread_spectrum(self)
        from watermark import MarkingError, mark_audio_file
        self._without_c2pa()
        path = self._write("silent2.wav", np.zeros(22050 * 2, dtype=np.float32))
        with self.assertRaises(MarkingError) as ctx:
            mark_audio_file(path)
        self.assertIn("metadata", str(ctx.exception).lower())


@requires_soundfile
class TestMarkingPolicyPreflight(unittest.TestCase):
    """Refuse before generating, and never let an opt-out leave output unmarked."""

    def test_opt_out_requires_attestation(self):
        from watermark import MarkingError, preflight_marking
        for kwargs in ({"no_watermark": True}, {"allow_unmarked": True},
                       {"no_spoken_disclaimer": True}):
            with self.assertRaises(MarkingError) as ctx:
                preflight_marking("out.wav", **kwargs)
            self.assertIn("--accept-marking-responsibility", str(ctx.exception))

    def test_watermark_floor_overrides_opt_out_without_c2pa(self):
        """The CrispASR watertight rule: no path emits a fully unmarked file."""
        from unittest.mock import patch

        from watermark import preflight_marking
        with patch("watermark.c2pa_available", return_value=False):
            policy = preflight_marking("out.wav", no_watermark=True,
                                       responsibility_accepted=True)
        self.assertTrue(policy["embed_watermark"], "watermark must be forced on")
        self.assertTrue(policy["forced"])
        self.assertIn("overridden", policy["note"])

    def test_opt_out_honoured_when_c2pa_carries_provenance(self):
        from unittest.mock import patch

        from watermark import preflight_marking
        with patch("watermark.c2pa_available", return_value=True):
            policy = preflight_marking("out.wav", no_watermark=True,
                                       responsibility_accepted=True)
        self.assertFalse(policy["embed_watermark"])
        self.assertTrue(policy["expect_c2pa"])

    def test_unwatermarkable_format_refused_before_synthesis(self):
        from watermark import MarkingError, preflight_marking
        for path in ("out.aiff", "out.wma", "out"):
            with self.assertRaises(MarkingError) as ctx:
                preflight_marking(path)
            self.assertIn("Refusing to synthesize", str(ctx.exception))

    def test_manifest_capable_formats_pass(self):
        from watermark import preflight_marking
        for path in ("out.wav", "out.mp3", "out.flac", "out.m4a"):
            policy = preflight_marking(path)
            self.assertTrue(policy["embed_watermark"])

    def test_manifestless_formats_need_a_neural_backend(self):
        """Opus/OGG carry no manifest, so the watermark is their only robust layer.

        The built-in comb is a fixed-key mark; being the *sole* layer is what
        makes it insufficient, not being present. So these formats pass when a
        neural backend is installed and are refused when it is not.
        """
        from unittest.mock import patch

        from watermark import MarkingError, preflight_marking
        for path in ("out.opus", "out.ogg"):
            with patch("watermark.neural_watermark_available", return_value=False):
                with self.assertRaises(MarkingError) as ctx:
                    preflight_marking(path)
                self.assertIn("crisptts[robust]", str(ctx.exception),
                              "the refusal must name the way out of it")
            with patch("watermark.neural_watermark_available", return_value=True):
                policy = preflight_marking(path)
                self.assertTrue(policy["embed_watermark"])

    def test_manifestless_format_still_allows_the_documented_opt_out(self):
        """The gate must not close the escape hatch the rest of the design offers."""
        from unittest.mock import patch

        from watermark import preflight_marking
        with patch("watermark.neural_watermark_available", return_value=False):
            policy = preflight_marking("out.opus", allow_unmarked=True,
                                       responsibility_accepted=True)
            self.assertTrue(policy["allow_unmarked"])

    def test_forced_policy_beats_env_opt_out(self):
        """A forced policy must survive CRISPTTS_NO_WATERMARK in the env."""
        import tempfile

        import soundfile as sf

        from watermark import mark_audio_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.wav")
            t = np.linspace(0, 2, 44100, endpoint=False, dtype=np.float32)
            sf.write(path, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 22050)
            os.environ["CRISPTTS_NO_WATERMARK"] = "1"
            try:
                result = mark_audio_file(path, policy={
                    "embed_watermark": True, "forced": True,
                    "expect_c2pa": False, "allow_unmarked": False, "note": None})
            finally:
                del os.environ["CRISPTTS_NO_WATERMARK"]
            self.assertTrue(result.marked, "floor must override the env opt-out")

    def test_c2pa_capability_reported(self):
        from unittest.mock import patch

        from watermark import output_carries_c2pa
        with patch("watermark.c2pa_available", return_value=True):
            self.assertTrue(output_carries_c2pa("a.wav"))
            self.assertTrue(output_carries_c2pa("a.mp3"))
            # FLAC and M4A sign through the streaming Builder.sign(); only
            # sign_file() refused them, which is what excluded them before.
            self.assertTrue(output_carries_c2pa("a.flac"))
            self.assertTrue(output_carries_c2pa("a.m4a"))
            # Opus/OGG is not in c2pa-rs's supported types at all → the
            # watermark stays mandatory and is the only robust layer.
            self.assertFalse(output_carries_c2pa("a.opus"))
            self.assertFalse(output_carries_c2pa("a.ogg"))


class TestMarkingAuditLog(unittest.TestCase):
    """Honoured opt-outs leave an audit trail, like [CONSENT] does."""

    def test_marking_line_format(self):
        import io
        from contextlib import redirect_stderr

        from watermark import log_marking_attestation
        buf = io.StringIO()
        with redirect_stderr(buf):
            log_marking_attestation(no_watermark=True, source="unit test")
        line = buf.getvalue()
        self.assertIn("[MARKING]", line)
        self.assertIn("no_watermark=yes", line)
        self.assertIn('attestation="unit test"', line)


@requires_soundfile
class TestMp3Marking(unittest.TestCase):
    """MP3 outputs must get a real audio watermark, not only ID3 tags."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        try:
            from pydub import AudioSegment  # noqa: F401
        except ImportError:
            self.skipTest("pydub not installed")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mp3(self):
        from pydub import AudioSegment
        wav = os.path.join(self.tmpdir, "src.wav")
        mp3 = os.path.join(self.tmpdir, "out.mp3")
        _write_tone_wav(wav)
        try:
            AudioSegment.from_file(wav).export(mp3, format="mp3")
        except Exception:
            self.skipTest("ffmpeg/libmp3lame not available")
        return mp3

    def test_mp3_gets_audio_watermark_and_tag(self):
        from watermark import mark_audio_file
        mp3 = self._make_mp3()
        result = mark_audio_file(mp3)
        self.assertTrue(result.marked)
        self.assertIn("audio-watermark", result.layers)
        self.assertIn("metadata", result.layers)
        self.assertIsNotNone(result.confidence)
        self.assertGreater(result.confidence, 0.65)

    def test_ai_tag_merged_into_encoder_written_id3(self):
        """Regression: encoders write their own ID3, which used to make the
        bytes-level injector skip the AI marker entirely."""
        from watermark import mark_audio_file
        mp3 = self._make_mp3()
        mark_audio_file(mp3)
        with open(mp3, "rb") as f:
            blob = f.read()
        self.assertIn(b"AI_GENERATED", blob)


class TestConsentGateConfig(unittest.TestCase):
    """Explicit config key beats the fail-open keyword heuristic."""

    def test_explicit_voice_cloning_true(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent(
            "some_new_backend", "some_new_handler", None,
            model_config={"voice_cloning": True}))

    def test_explicit_voice_cloning_false_overrides_keywords(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent(
            "xtts_readonly_demo", "demo", None,
            model_config={"voice_cloning": False}))

    def test_falls_back_to_keywords_without_explicit_key(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("kartoffelbox_zeroshot", "unknown", None,
                                         model_config={}))

    def test_every_cloning_model_in_config_is_gated(self):
        """Guards against adding a cloning backend that slips past the gate."""
        from config import GERMAN_TTS_MODELS
        from watermark import requires_consent
        hints = ("clon", "zero-shot", "zeroshot", "reference audio",
                 "voice prompt", "speaker prompt")
        missed = []
        for mid, cfg in GERMAN_TTS_MODELS.items():
            notes = (str(cfg.get("notes", "")) + " " + str(cfg.get("description", ""))).lower()
            if any(h in notes for h in hints):
                if not requires_consent(mid, cfg.get("handler_function_key", mid),
                                        cfg.get("default_voice_id"), model_config=cfg):
                    missed.append(mid)
        self.assertEqual(missed, [], f"cloning models not gated by consent: {missed}")


@requires_soundfile
class TestConsentAuditEvidence(unittest.TestCase):
    """The audit log should tie an attestation to the actual reference audio."""

    def test_logs_reference_digest(self):
        import tempfile

        from watermark import _reference_audio_digest
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, "ref.wav")
            _write_tone_wav(ref)
            digest = _reference_audio_digest(ref)
            self.assertIsNotNone(digest)
            self.assertEqual(len(digest), 32)
            self.assertEqual(digest, _reference_audio_digest(ref))

    def test_no_digest_for_non_file_voice(self):
        from watermark import _reference_audio_digest
        self.assertIsNone(_reference_audio_digest("af_heart"))
        self.assertIsNone(_reference_audio_digest(None))


class TestC2paSignerDisclosure(unittest.TestCase):
    """Self-signed C2PA manifests must be reported as such, not as trusted."""

    def test_returns_signer_kind(self):
        from watermark import c2pa_sign_file_ex
        ok, signer = c2pa_sign_file_ex("/nonexistent/file.wav")
        self.assertIsInstance(ok, bool)
        if ok:
            self.assertIn(signer, ("self-signed", "ca-issued"))
        else:
            self.assertIsNone(signer)

    def test_bool_wrapper_preserved(self):
        from watermark import c2pa_sign_file
        self.assertIsInstance(c2pa_sign_file("/nonexistent/file.wav"), bool)


class TestC2paBackendTiering(unittest.TestCase):
    """Native signers are fast paths, never trusted blindly."""

    def setUp(self):
        self._env = os.environ.pop("CRISPTTS_C2PA_BACKEND", None)
        import watermark
        watermark._crispasr_c2pa_flag = False  # reset the probe cache

    def tearDown(self):
        os.environ.pop("CRISPTTS_C2PA_BACKEND", None)
        if self._env is not None:
            os.environ["CRISPTTS_C2PA_BACKEND"] = self._env
        import watermark
        watermark._crispasr_c2pa_flag = False

    def test_backend_off_skips_signing(self):
        from watermark import c2pa_sign_file_ex
        os.environ["CRISPTTS_C2PA_BACKEND"] = "off"
        ok, signer = c2pa_sign_file_ex("/nonexistent/file.wav")
        self.assertFalse(ok)
        self.assertIsNone(signer)

    def test_unknown_backend_falls_back_to_auto(self):
        from watermark import _c2pa_backend_preference
        os.environ["CRISPTTS_C2PA_BACKEND"] = "not-a-backend"
        self.assertEqual(_c2pa_backend_preference(), "auto")

    def test_crispasr_is_not_a_signing_backend(self):
        """crispasr 0.8.25 has no flag that signs an existing file.

        It was briefly probed for one, and "--c2pa" would have matched as a
        substring of "--c2pa-cert". Its output is handled by preserving the
        manifest it already wrote — see TestUpstreamManifestPreserved.
        """
        from watermark import _C2PA_BACKENDS
        self.assertNotIn("crispasr", _C2PA_BACKENDS)

    @requires_c2pa
    @requires_soundfile
    def test_native_signer_without_ai_assertion_is_discarded(self):
        """The regression that started all this.

        A native signer that produces a manifest with no AI-generation claim
        must not be accepted — the file would carry an integrity seal that
        looks like provenance but asserts nothing about being AI-generated.
        """
        import tempfile
        from unittest.mock import patch

        import watermark
        from watermark import c2pa_sign_file_ex, manifest_asserts_ai
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.wav")
            _write_tone_wav(path, seconds=1.0)
            # A native signer that "succeeds" but writes no AI assertion.
            with patch.object(watermark, "_sign_with_c2pa_audio", return_value=True):
                ok, signer = c2pa_sign_file_ex(path, model_id="t")
            self.assertTrue(ok)
            self.assertEqual(signer, "self-signed")
            self.assertIs(manifest_asserts_ai(path), True,
                          "fallback must leave a manifest that asserts AI generation")

    @requires_c2pa
    @requires_soundfile
    def test_manifest_asserts_ai_detects_real_manifest(self):
        import tempfile

        from watermark import c2pa_sign_file_ex, manifest_asserts_ai
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.wav")
            _write_tone_wav(path, seconds=1.0)
            self.assertIsNone(manifest_asserts_ai(path), "unsigned file has no manifest")
            c2pa_sign_file_ex(path, model_id="t")
            self.assertIs(manifest_asserts_ai(path), True)

    def test_manifest_asserts_ai_on_missing_file(self):
        from watermark import manifest_asserts_ai
        self.assertIsNone(manifest_asserts_ai("/nonexistent/file.wav"))


@requires_soundfile
class TestBundledDisclosureAssets(unittest.TestCase):
    """The disclosure must work with no TTS backend and no network."""

    def test_default_language_clip_is_bundled(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, bundled_disclosure_path
        path = bundled_disclosure_path(DEFAULT_DISCLAIMER_LANG)
        self.assertIsNotNone(path, "the default-language disclosure must ship in the wheel")
        self.assertTrue(os.path.isfile(path))

    def test_bundled_clip_decodes_to_audio(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, _load_bundled_disclosure
        pcm = _load_bundled_disclosure(24000, DEFAULT_DISCLAIMER_LANG)
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm) / 24000, 1.0, "a disclosure sentence is over a second")
        self.assertGreater(float(np.max(np.abs(pcm))), 0.05, "clip must not be silence")

    def test_bundled_clip_resamples_to_requested_rate(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, _load_bundled_disclosure
        at_16k = _load_bundled_disclosure(16000, DEFAULT_DISCLAIMER_LANG)
        at_48k = _load_bundled_disclosure(48000, DEFAULT_DISCLAIMER_LANG)
        self.assertAlmostEqual(len(at_48k) / len(at_16k), 3.0, delta=0.05)

    def test_generator_falls_back_to_bundled_not_tone_marker(self):
        """With every TTS route dead, the result must still be a disclosure."""
        from unittest.mock import patch

        import watermark
        from watermark import generate_spoken_disclaimer
        with patch("shutil.which", return_value=None), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CRISPASR_EXECUTABLE", None)
            # Make the edge-tts route unavailable too.
            with patch.dict("sys.modules", {"edge_tts": None}):
                pcm, kind = generate_spoken_disclaimer(
                    24000, watermark.DEFAULT_DISCLAIMER_LANG)
        self.assertEqual(kind, "spoken",
                         "bundled clip must count as a real spoken disclosure")
        self.assertIsNotNone(pcm)

    def test_unbundled_language_returns_none(self):
        from watermark import bundled_disclosure_path
        # normalize_disclaimer_lang maps unknown codes to the default, which is
        # bundled — so this asserts the lookup is by resolved language.
        self.assertEqual(bundled_disclosure_path("klingon"),
                         bundled_disclosure_path(None))

    def test_every_bundled_clip_matches_a_known_language(self):
        import glob

        from watermark import DISCLAIMER_TEXTS
        asset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crisptts_assets")
        for path in glob.glob(os.path.join(asset_dir, "disclosure_*.flac")):
            lang = os.path.basename(path)[len("disclosure_"):-len(".flac")]
            self.assertIn(lang, DISCLAIMER_TEXTS,
                          f"bundled clip {lang} has no matching disclosure text")


class TestUpstreamManifestPreserved(unittest.TestCase):
    """An upstream AI manifest is stronger than anything we would add."""

    @requires_c2pa
    @requires_soundfile
    def test_existing_ai_manifest_is_not_overwritten(self):
        import tempfile

        from watermark import c2pa_sign_file_ex, manifest_asserts_ai, mark_audio_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "upstream.wav")
            _write_tone_wav(path)
            # Stand in for CrispASR: a file that arrives already signed.
            ok, _ = c2pa_sign_file_ex(path, model_id="upstream_engine")
            self.assertTrue(ok)
            before = os.path.getsize(path)

            result = mark_audio_file(path, handler_key="crispasr")
            self.assertTrue(result.marked)
            self.assertIn("c2pa:preserved", result.layers)
            self.assertEqual(result.c2pa_signer, "preserved")
            self.assertIs(manifest_asserts_ai(path), True,
                          "the preserved manifest must still assert AI generation")
            self.assertEqual(os.path.getsize(path), before,
                             "preserving means leaving the bytes alone")

    @requires_c2pa
    @requires_soundfile
    def test_metadata_injection_would_have_invalidated_it(self):
        """Documents why preservation is necessary, not merely tidier."""
        import json
        import tempfile

        import c2pa

        from watermark import _inject_container_metadata, c2pa_sign_file_ex
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "upstream.wav")
            _write_tone_wav(path)
            c2pa_sign_file_ex(path, model_id="upstream_engine")
            state_before = json.loads(c2pa.Reader(path).json()).get("validation_state")
            self.assertEqual(state_before, "Valid")

            _inject_container_metadata(path, ".wav")
            try:
                state_after = json.loads(c2pa.Reader(path).json()).get("validation_state")
            except Exception:
                state_after = "unreadable"
            self.assertNotEqual(state_after, "Valid",
                                "if this ever passes, preservation can be relaxed")

    @requires_soundfile
    def test_crispasr_watermark_layer_only_claimed_when_detected(self):
        """We must not report a mark that no detector can find."""
        import tempfile

        import watermark
        from watermark import mark_audio_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plain.wav")
            _write_tone_wav(path)  # never watermarked by anyone
            original = watermark.c2pa_sign_file_ex
            watermark.c2pa_sign_file_ex = lambda *a, **k: (True, "self-signed")
            try:
                result = mark_audio_file(path, handler_key="crispasr")
            finally:
                watermark.c2pa_sign_file_ex = original
            self.assertNotIn("audio-watermark:crispasr", result.layers)
            self.assertNotIn("audio-watermark:upstream", result.layers)


class TestWatermarkBandMatchesCrispASR(unittest.TestCase):
    """The comb placement is an interop contract with CrispASR's wm_params.

    Regression: CrispTTS stayed on the pre-#260 wideband comb after CrispASR
    moved to the speech band, so neither could detect the other's watermark.
    Measured on crispasr 0.8.25 kokoro output, CrispASR read 0.72 and CrispTTS
    read 0.41 on the same bytes.
    """

    def setUp(self):
        self._env = os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)

    def tearDown(self):
        os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)
        if self._env is not None:
            os.environ["CRISPASR_WATERMARK_LEGACY"] = self._env

    def test_default_band_matches_crispasr_speech_band(self):
        from watermark import wm_params
        lo, hi, alpha = wm_params(1024)
        self.assertEqual((lo, hi), (1024 // 16, 1024 // 5))
        self.assertAlmostEqual(alpha, 0.05)

    def test_legacy_env_restores_the_old_band(self):
        from watermark import wm_params
        os.environ["CRISPASR_WATERMARK_LEGACY"] = "1"
        lo, hi, alpha = wm_params(1024)
        self.assertEqual((lo, hi), (1024 // 16, 1024 // 2 - 1))
        self.assertAlmostEqual(alpha, 0.08)

    def test_legacy_flag_overrides_env_explicitly(self):
        from watermark import wm_params
        self.assertEqual(wm_params(1024, legacy=True)[1], 1024 // 2 - 1)
        self.assertEqual(wm_params(1024, legacy=False)[1], 1024 // 5)

    def _speech_like(self, seconds=3.0, sr=24000):
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
        return (0.30 * np.sin(2 * np.pi * 180 * t)
                + 0.12 * np.sin(2 * np.pi * 420 * t)
                + 0.05 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)

    def test_detection_sweeps_both_bands(self):
        """Audio marked on the old band must still verify after the change."""
        from watermark import spread_spectrum_detect, spread_spectrum_embed
        pcm = self._speech_like()
        os.environ["CRISPASR_WATERMARK_LEGACY"] = "1"
        legacy_marked = spread_spectrum_embed(pcm)
        del os.environ["CRISPASR_WATERMARK_LEGACY"]
        self.assertGreater(spread_spectrum_detect(legacy_marked), 0.65,
                           "a legacy-band watermark must still be detectable")

    def test_current_band_roundtrip(self):
        from watermark import spread_spectrum_detect, spread_spectrum_embed
        pcm = self._speech_like()
        self.assertGreater(spread_spectrum_detect(spread_spectrum_embed(pcm)), 0.65)

    def test_sweeping_does_not_raise_false_positives(self):
        """Checking two bands must not make unmarked *recorded* audio look marked.

        Uses a noisy, amplitude-varying signal rather than the bare sum of
        sines this test used to build. That is deliberate and is a real
        limitation being recorded, not a test being loosened to fit:

        A perfectly stationary synthetic tone is the one input the detector
        still reads as marked (measured 0.88). Every frame of it is identical,
        so any chance correlation with the comb's sign pattern repeats
        endlessly and looks like consistency. The decoy-calibrated term catches
        most of it — the tone scores z=1.49 where real marks reach 2.8 — but
        clearing it entirely needs z>=1.5, which also rejects 5.7% of genuinely
        marked audio. Marking fails closed, so that setting deletes 5.7% of
        users' output to avoid one signal that does not occur in a recording.

        The variation added here is what every real source has: german.wav and
        all 27 bundled disclosure clips read well below threshold unmarked.
        """
        from watermark import _VERIFY_THRESHOLD, spread_spectrum_detect
        rng = np.random.default_rng(11)
        base = self._speech_like()
        t = np.arange(len(base), dtype=np.float32) / 24000.0
        envelope = (0.6 + 0.4 * np.sin(2 * np.pi * 2.7 * t)).astype(np.float32)
        realistic = (base * envelope + 0.01 * rng.standard_normal(len(base))).astype(np.float32)
        self.assertLess(spread_spectrum_detect(realistic), _VERIFY_THRESHOLD)

    def test_stationary_tone_is_a_known_false_positive(self):
        """Pin the limitation above so it cannot regress silently.

        If a future change makes the bare tone read as unmarked, that is an
        improvement — and this test should then be deleted, not adjusted.
        """
        from watermark import _VERIFY_THRESHOLD, spread_spectrum_detect
        conf = spread_spectrum_detect(self._speech_like())
        self.assertGreater(
            conf, _VERIFY_THRESHOLD,
            "a perfectly stationary tone no longer false-positives — good; "
            "delete this test and the caveat in _spread_spectrum_detect_band")

    def test_alpha_none_selects_band_default(self):
        from watermark import spread_spectrum_embed
        pcm = self._speech_like()
        np.testing.assert_allclose(spread_spectrum_embed(pcm),
                                   spread_spectrum_embed(pcm, alpha=-1.0), atol=1e-6)

    def test_alpha_zero_is_an_explicit_noop(self):
        from watermark import spread_spectrum_embed
        pcm = self._speech_like()
        np.testing.assert_allclose(spread_spectrum_embed(pcm, alpha=0.0), pcm, atol=1e-6)

    def test_survives_resample_on_the_speech_band(self):
        """The transform that defeated the old band: 0.63 then, above 0.65 now."""
        from watermark import _resample_linear, spread_spectrum_detect, spread_spectrum_embed
        marked = spread_spectrum_embed(self._speech_like(seconds=6.0))
        there_and_back = _resample_linear(_resample_linear(marked, 24000, 16000), 16000, 24000)
        self.assertGreater(spread_spectrum_detect(there_and_back), 0.65)


class TestDisclosureWordingMatchesSusurrus(unittest.TestCase):
    """The Crisp projects should disclose in the same words."""

    def test_de_and_en_match_susurrus_strings(self):
        from watermark import DISCLAIMER_TEXTS
        self.assertEqual(
            DISCLAIMER_TEXTS["de"],
            "Die folgende Aufnahme wurde von künstlicher Intelligenz erzeugt.")
        self.assertEqual(
            DISCLAIMER_TEXTS["en"],
            "The following audio was generated by artificial intelligence.")

    def test_all_disclosures_describe_what_follows(self):
        """The disclosure is prepended, so it must not say "this audio"."""
        from watermark import DISCLAIMER_TEXTS
        # "。" is the sentence terminator in Japanese and Chinese; "." there
        # would be the wrong character, not a stricter one.
        terminators = (".", "。")
        for lang, text in DISCLAIMER_TEXTS.items():
            self.assertTrue(text.strip().endswith(terminators), lang)
            # CJK says the same thing in far fewer characters, so the
            # length floor has to be script-aware to mean anything.
            floor = 10 if lang in {"ja", "ko", "zh"} else 20
            self.assertGreater(len(text), floor, lang)

    def test_every_eu_official_language_is_covered(self):
        """Art. 50 governs the EU market, so EU audiences must be disclosable to."""
        from watermark import DISCLAIMER_TEXTS
        eu_official = {
            "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga",
            "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk",
            "sl", "sv",
        }
        missing = sorted(eu_official - set(DISCLAIMER_TEXTS))
        self.assertEqual(missing, [], f"no disclosure text for: {missing}")

    def test_every_language_has_an_edge_voice(self):
        from watermark import _DISCLAIMER_EDGE_VOICES, DISCLAIMER_TEXTS
        missing = sorted(set(DISCLAIMER_TEXTS) - set(_DISCLAIMER_EDGE_VOICES))
        self.assertEqual(missing, [], f"no Edge TTS voice for: {missing}")

    def test_every_language_ships_an_offline_clip(self):
        """Tier 3 is what makes disclosure work offline; a gap silently loses it."""
        from watermark import DISCLAIMER_TEXTS, bundled_disclosure_path
        missing = sorted(lang for lang in DISCLAIMER_TEXTS
                         if not bundled_disclosure_path(lang))
        self.assertEqual(missing, [], f"no bundled disclosure clip for: {missing}")


class TestConsentDetectionTiers(unittest.TestCase):
    """Every tier of the consent gate must actually be reachable.

    The handler-key tier once listed ``synthesize_with_*`` function names,
    which match no config entry, so it silently matched nothing and the gate
    fell through to substring matching on model IDs.
    """

    def test_handler_keys_correspond_to_real_handlers(self):
        from config import GERMAN_TTS_MODELS
        from watermark import VOICE_CLONING_HANDLER_KEYS
        real = {cfg.get("handler_function_key", mid)
                for mid, cfg in GERMAN_TTS_MODELS.items()}
        dead = VOICE_CLONING_HANDLER_KEYS - real
        self.assertEqual(dead, set(),
                         f"VOICE_CLONING_HANDLER_KEYS entries match no handler: {sorted(dead)}")

    def test_every_shipped_model_declares_voice_cloning(self):
        """Tier 1 is the only non-heuristic tier; it must cover every model."""
        from config import GERMAN_TTS_MODELS
        missing = [mid for mid, cfg in GERMAN_TTS_MODELS.items()
                   if "voice_cloning" not in cfg]
        self.assertEqual(missing, [],
                         f"models with no explicit voice_cloning key: {missing}")

    def test_every_fixed_speaker_model_declares_speaker_identity(self):
        """A model that does not clone still has to answer whose voice it is.

        Art. 3(60) defines a deep fake by resemblance to an existing person,
        not by how the resemblance was made — so ``voice_cloning: False`` is
        not on its own an answer. A new backend must not be able to skip the
        question by omission, which is how all 27 of these once did.
        """
        from config import GERMAN_TTS_MODELS
        from watermark import SPEAKER_IDENTITY_VALUES
        missing, bad = [], []
        for mid, cfg in GERMAN_TTS_MODELS.items():
            if cfg.get("voice_cloning") is not False:
                continue
            if "speaker_identity" not in cfg:
                missing.append(mid)
            elif cfg["speaker_identity"] not in SPEAKER_IDENTITY_VALUES:
                bad.append((mid, cfg["speaker_identity"]))
        self.assertEqual(missing, [],
                         f"non-cloning models with no speaker_identity key: {missing}")
        self.assertEqual(bad, [], f"invalid speaker_identity values: {bad}")

    def test_real_person_preset_voice_gets_the_spoken_disclosure(self):
        """The Art. 50(4) trigger is resemblance, not the cloning mechanism."""
        from watermark import requires_spoken_disclosure
        self.assertTrue(requires_spoken_disclosure(False, "real_person",
                                                   model_id="coqui_tts_thorsten_vits"))
        self.assertFalse(requires_spoken_disclosure(False, "synthetic",
                                                    model_id="kokoro_onnx"))
        # Cloning still triggers it whatever the preset voice is declared as.
        self.assertTrue(requires_spoken_disclosure(True, "synthetic", model_id="f5_tts_german"))

    def test_unknown_speaker_identity_warns_rather_than_guessing(self):
        """Same choice as 'multilingual' disclosure languages: surface it."""
        import watermark
        from watermark import requires_spoken_disclosure
        watermark._warned_speaker_identity.discard("some_model")
        with self.assertLogs("CrispTTS.watermark", level="WARNING") as captured:
            result = requires_spoken_disclosure(False, "unknown", model_id="some_model")
        self.assertFalse(result, "'unknown' must not silently prepend a disclosure")
        self.assertIn("Art. 3(60)", "\n".join(captured.output))
        self.assertIn("--speaker-identity", "\n".join(captured.output))

    def test_speaker_identity_override_beats_config(self):
        from watermark import resolve_speaker_identity
        cfg = {"speaker_identity": "synthetic"}
        self.assertEqual(resolve_speaker_identity(cfg, "real_person"), "real_person")
        self.assertEqual(resolve_speaker_identity(cfg), "synthetic")
        # Anything unrecognised, from either source, resolves to unknown.
        self.assertEqual(resolve_speaker_identity(cfg, "nonsense"), "unknown")
        self.assertEqual(resolve_speaker_identity({"speaker_identity": "yes"}), "unknown")
        self.assertEqual(resolve_speaker_identity(None), "unknown")

    def test_reference_recording_beats_explicit_false(self):
        """Handing the system a voice to imitate always needs consent."""
        from watermark import requires_consent
        for ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"):
            self.assertTrue(
                requires_consent("kokoro_onnx", "kokoro_onnx", f"/refs/someone{ext}",
                                 model_config={"voice_cloning": False}),
                f"{ext} reference recording bypassed the consent gate")

    def test_preset_voice_with_explicit_false_is_not_gated(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent("kokoro_onnx", "kokoro_onnx", "af_heart",
                                          model_config={"voice_cloning": False}))

    def test_cloning_models_gated_without_any_voice(self):
        """A zero-shot model is gated even before a reference is chosen."""
        from config import GERMAN_TTS_MODELS
        from watermark import requires_consent
        for mid in ("oute_llamacpp", "oute_hf", "f5_tts_german", "crispasr_moss_tts_local"):
            cfg = GERMAN_TTS_MODELS[mid]
            self.assertTrue(
                requires_consent(mid, cfg.get("handler_function_key", mid), None,
                                 model_config=cfg),
                f"{mid} is a cloning model but was not gated")


class TestSpokenDisclosureLanguage(unittest.TestCase):
    """The disclosure must be in the language of the audio it precedes."""

    def test_defaults_to_german(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, disclaimer_text
        self.assertEqual(DEFAULT_DISCLAIMER_LANG, "de")
        self.assertIn("künstlicher Intelligenz", disclaimer_text(None))

    def test_follows_config_language(self):
        from watermark import disclaimer_text
        self.assertIn("artificial intelligence", disclaimer_text("en"))
        self.assertIn("künstlicher Intelligenz", disclaimer_text("de-DE"))
        self.assertIn("intelligence artificielle", disclaimer_text("fr"))

    def test_normalizes_locale_forms(self):
        from watermark import normalize_disclaimer_lang
        for value, expected in [("de", "de"), ("de-DE", "de"), ("de_DE", "de"),
                                ("german", "de"), ("EN-GB", "en"), (None, "de"),
                                ("klingon", "de"), ("", "de")]:
            self.assertEqual(normalize_disclaimer_lang(value), expected, value)

    def test_every_language_has_an_edge_voice(self):
        from watermark import _DISCLAIMER_EDGE_VOICES, DISCLAIMER_TEXTS
        self.assertEqual(set(DISCLAIMER_TEXTS), set(_DISCLAIMER_EDGE_VOICES))


@requires_soundfile
class TestSpokenDisclosureFailsClosed(unittest.TestCase):
    """A tone marker is not a disclosure, and silence is not an option."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        import watermark
        watermark._disclaimer_cache.clear()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import watermark
        watermark._disclaimer_cache.clear()

    def test_tone_marker_is_refused_by_default(self):
        import watermark
        from watermark import DisclosureError, prepend_disclaimer_file
        path = os.path.join(self.tmpdir, "cloned.wav")
        _write_tone_wav(path)
        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (
            np.zeros(int(sr * 0.5), dtype=np.float32), "tone-marker")
        try:
            with self.assertRaises(DisclosureError):
                prepend_disclaimer_file(path, language="de")
        finally:
            watermark.generate_spoken_disclaimer = original

    def test_tone_marker_accepted_when_explicitly_allowed(self):
        import watermark
        from watermark import prepend_disclaimer_file
        path = os.path.join(self.tmpdir, "cloned.wav")
        _write_tone_wav(path)
        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (
            np.zeros(int(sr * 0.5), dtype=np.float32), "tone-marker")
        try:
            kind = prepend_disclaimer_file(path, language="de", require_spoken=False)
            self.assertEqual(kind, "tone-marker")
        finally:
            watermark.generate_spoken_disclaimer = original

    def test_generation_failure_raises(self):
        import watermark
        from watermark import DisclosureError, prepend_disclaimer_file
        path = os.path.join(self.tmpdir, "cloned.wav")
        _write_tone_wav(path)
        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (None, "none")
        try:
            with self.assertRaises(DisclosureError):
                prepend_disclaimer_file(path, language="de")
        finally:
            watermark.generate_spoken_disclaimer = original

    def test_unsupported_container_raises(self):
        from watermark import DisclosureError, prepend_disclaimer_file
        path = os.path.join(self.tmpdir, "cloned.aiff")
        with open(path, "wb") as f:
            f.write(b"\0" * 1024)
        with self.assertRaises(DisclosureError):
            prepend_disclaimer_file(path)

    def test_spoken_disclosure_lengthens_the_audio(self):
        import soundfile as sf

        import watermark
        from watermark import prepend_disclaimer_file
        path = os.path.join(self.tmpdir, "cloned.wav")
        original_pcm = _write_tone_wav(path)
        gen = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (
            np.full(int(sr * 1.5), 0.05, dtype=np.float32), "spoken")
        try:
            kind = prepend_disclaimer_file(path, language="de")
        finally:
            watermark.generate_spoken_disclaimer = gen
        self.assertEqual(kind, "spoken")
        after, _ = sf.read(path, dtype="float32")
        self.assertGreater(len(after), len(original_pcm))


class TestDisclosureLanguageResolution(unittest.TestCase):
    """The disclosure must not silently claim a language it did not use.

    Before this, every shipped cloning model resolved to German — including
    the multilingual ones and the Mandarin-centric CrispASR backends — because
    an unknown language fell through to the default with no signal. A German
    sentence in front of Mandarin audio is not an Art. 50(4) disclosure.
    """

    def test_explicit_override_wins_over_config(self):
        from watermark import resolve_disclaimer_lang
        self.assertEqual(resolve_disclaimer_lang("de", "zh"), ("zh", True))

    def test_config_language_used_when_no_override(self):
        from watermark import resolve_disclaimer_lang
        self.assertEqual(resolve_disclaimer_lang("fr"), ("fr", True))

    def test_regional_and_long_form_codes(self):
        from watermark import resolve_disclaimer_lang
        self.assertEqual(resolve_disclaimer_lang("pt-BR"), ("pt", True))
        self.assertEqual(resolve_disclaimer_lang("Swedish"), ("sv", True))

    def test_multilingual_marker_is_not_a_language(self):
        from watermark import resolve_disclaimer_lang
        lang, known = resolve_disclaimer_lang("multilingual")
        self.assertFalse(known, "'multilingual' must not pass as a known language")

    def test_missing_language_is_reported_as_defaulted(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, resolve_disclaimer_lang
        self.assertEqual(resolve_disclaimer_lang(None), (DEFAULT_DISCLAIMER_LANG, False))

    def test_override_rescues_a_multilingual_model(self):
        from watermark import resolve_disclaimer_lang
        self.assertEqual(resolve_disclaimer_lang("multilingual", "ja"), ("ja", True))

    def test_unsupported_language_defaults_rather_than_crashing(self):
        from watermark import DEFAULT_DISCLAIMER_LANG, resolve_disclaimer_lang
        lang, known = resolve_disclaimer_lang("xx")
        self.assertEqual(lang, DEFAULT_DISCLAIMER_LANG)
        self.assertFalse(known)

    def test_every_cloning_model_declares_a_language(self):
        """A missing key is indistinguishable from 'German', so require one."""
        from config import GERMAN_TTS_MODELS
        missing = sorted(mid for mid, cfg in GERMAN_TTS_MODELS.items()
                         if cfg.get("voice_cloning") and "language" not in cfg)
        self.assertEqual(missing, [], f"cloning models with no language: {missing}")

    def test_declared_languages_are_resolvable_or_explicitly_multilingual(self):
        from config import GERMAN_TTS_MODELS
        from watermark import MULTILINGUAL_LANG_MARKERS, resolve_disclaimer_lang
        for mid, cfg in GERMAN_TTS_MODELS.items():
            if not cfg.get("voice_cloning"):
                continue
            declared = cfg.get("language")
            _, known = resolve_disclaimer_lang(declared)
            if not known:
                self.assertIn(str(declared).lower(), MULTILINGUAL_LANG_MARKERS,
                              f"{mid}: language {declared!r} is neither resolvable "
                              "nor an explicit multilingual marker")

    def test_prepend_uses_the_override_language(self):
        import tempfile

        import soundfile as sf

        import watermark
        from watermark import prepend_disclaimer_file
        seen = {}

        def fake_gen(sample_rate=24000, language=None):
            seen["lang"] = language
            return np.full(int(sample_rate * 1.0), 0.05, dtype=np.float32), "spoken"

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.wav")
            sf.write(path, np.full(24000, 0.1, dtype=np.float32), 24000)
            real = watermark.generate_spoken_disclaimer
            watermark._disclaimer_cache.clear()
            watermark.generate_spoken_disclaimer = fake_gen
            try:
                prepend_disclaimer_file(path, language="de", disclosure_lang="el")
            finally:
                watermark.generate_spoken_disclaimer = real
                watermark._disclaimer_cache.clear()
        self.assertEqual(seen["lang"], "el")


class TestConsentAuditLogRetention(unittest.TestCase):
    """The audit log records reference-audio paths, so it is personal data.

    GDPR Art. 5(1)(e) wants a retention limit and Art. 17 an erasure path;
    an append-forever file offers neither.
    """

    def setUp(self):
        import tempfile

        import watermark
        self._dir = tempfile.mkdtemp()
        self._saved = watermark._CONSENT_LOG_PATH
        watermark._CONSENT_LOG_PATH = os.path.join(self._dir, "consent_audit.log")

    def tearDown(self):
        import shutil

        import watermark
        watermark._CONSENT_LOG_PATH = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write(self, lines):
        import watermark
        with open(watermark._CONSENT_LOG_PATH, "w") as f:
            f.writelines(lines)

    @staticmethod
    def _line(days_ago, subject="ref.wav"):
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
            "%Y-%m-%dT%H:%M:%S%z")
        return f'[CONSENT] ts={ts} model=m voice={subject} attestation="t"\n'

    def test_expired_entries_are_pruned(self):
        """Asserts the property, not the line count.

        A prune re-chains the survivors and appends a `[CHAIN-REBUILT]` record,
        so the file legitimately holds one more line than the survivors. What
        must hold is that the expired entry is gone and the fresh one stayed.
        """
        import watermark
        from watermark import prune_audit_log
        old, fresh = self._line(1000), self._line(1)
        self._write([old, fresh])
        self.assertEqual(prune_audit_log(), 1)
        with open(watermark._CONSENT_LOG_PATH) as f:
            body = f.read()
        self.assertNotIn(old.split(" model=")[0], body, "the expired line should be gone")
        self.assertIn(fresh.split(" model=")[0], body, "the fresh line should have survived")
        self.assertIn("[CHAIN-REBUILT]", body, "the prune should be recorded, not silent")

    def test_fresh_entries_survive(self):
        from watermark import prune_audit_log
        self._write([self._line(1), self._line(2)])
        self.assertEqual(prune_audit_log(), 0)

    def test_retention_zero_disables_pruning(self):
        from watermark import prune_audit_log
        self._write([self._line(100000)])
        self.assertEqual(prune_audit_log(retention_days=0), 0)

    def test_unparseable_lines_are_kept(self):
        """An unreadable record is not evidence that it has expired."""
        from watermark import prune_audit_log
        self._write(["garbage with no timestamp\n"])
        self.assertEqual(prune_audit_log(retention_days=1), 0)

    def test_erase_by_subject_leaves_others(self):
        import watermark
        from watermark import erase_audit_log
        self._write([self._line(1, "alice.wav"), self._line(1, "bob.wav")])
        self.assertEqual(erase_audit_log("alice.wav"), 1)
        with open(watermark._CONSENT_LOG_PATH) as f:
            rest = f.read()
        self.assertIn("bob.wav", rest)
        self.assertNotIn("alice.wav", rest)

    def test_erase_everything(self):
        import watermark
        from watermark import erase_audit_log
        self._write([self._line(1), self._line(2)])
        self.assertEqual(erase_audit_log(), 2)
        self.assertFalse(os.path.exists(watermark._CONSENT_LOG_PATH))

    def test_erase_on_missing_log_is_harmless(self):
        from watermark import erase_audit_log
        self.assertEqual(erase_audit_log(), 0)

    def test_appending_creates_an_owner_only_file(self):
        import stat

        import watermark
        watermark.log_consent_attestation("some_model", None)
        mode = stat.S_IMODE(os.stat(watermark._CONSENT_LOG_PATH).st_mode)
        self.assertEqual(mode, 0o600, f"audit log is {oct(mode)}, expected 0o600")

    def test_append_prunes_expired_entries(self):
        import contextlib
        import io

        import watermark
        expired = self._line(1000)
        self._write([expired])
        with contextlib.redirect_stderr(io.StringIO()):
            watermark.log_consent_attestation("some_model", None)
        with open(watermark._CONSENT_LOG_PATH) as f:
            body = f.read()
        self.assertNotIn(expired.split(" model=")[0], body,
                         "the expired line should have been pruned")
        self.assertIn("some_model", body, "the new attestation should be there")


if __name__ == "__main__":
    unittest.main()
