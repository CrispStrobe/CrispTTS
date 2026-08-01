"""Tests for watermark.py — spread-spectrum watermark, metadata, consent gate."""

import os
import struct
import unittest

import numpy as np

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

    def test_wavmark_payload_encodes_ct(self):
        """Payload should encode 'CT' = 0x43 0x54."""
        from watermark import _WAVMARK_PAYLOAD
        # C = 0x43 = 0100_0011, T = 0x54 = 0101_0100
        expected = [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
        for i, (got, exp) in enumerate(zip(_WAVMARK_PAYLOAD, expected, strict=True)):
            self.assertEqual(int(got), exp, f"Bit {i}: expected {exp}, got {int(got)}")


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
                    AudioSegment(data=raw, sample_width=2, frame_rate=24000,
                                 channels=1).export(path, format=fmt)
                ok, _ = c2pa_sign_file_ex(path, model_id="t")
                self.assertTrue(ok, f"{ext} is in C2PA_CAPABLE_EXTS but cannot be signed")


class TestComplianceCoverage(unittest.TestCase):
    """Verify all output paths have watermark coverage."""

    def test_wav_watermark_roundtrip(self):
        """WAV files should have detectable watermark after embed."""
        from watermark import spread_spectrum_detect, watermark_embed
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
        """Re-marking must not embed a second watermark (~6 dB SNR cost)."""
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
        self.assertEqual(after_first, after_second)
        self.assertEqual(second.reason, "already-marked")

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
        self.assertIn("audio-watermark:crispasr", result.layers)
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
        from watermark import MarkingError, mark_audio_file
        self._without_c2pa()
        path = self._write("tiny.wav", np.zeros(400, dtype=np.float32) + 0.1)
        with self.assertRaises(MarkingError) as ctx:
            mark_audio_file(path)
        self.assertIn("not sufficient", str(ctx.exception))

    def test_digital_silence_refused(self):
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

    def test_supported_formats_pass(self):
        from watermark import preflight_marking
        for path in ("out.wav", "out.mp3", "out.flac", "out.opus", "out.ogg"):
            policy = preflight_marking(path)
            self.assertTrue(policy["embed_watermark"])

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
            # Opus/FLAC cannot carry a manifest → watermark stays mandatory
            self.assertFalse(output_carries_c2pa("a.opus"))
            self.assertFalse(output_carries_c2pa("a.flac"))


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


if __name__ == "__main__":
    unittest.main()
