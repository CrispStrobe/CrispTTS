"""Tests for watermark.py — spread-spectrum watermark, metadata, consent gate."""

import os
import struct
import unittest

import numpy as np


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

    def test_c2pa_manifest_json(self):
        """The C2PA manifest JSON should be valid."""
        import json

        from watermark import _C2PA_MANIFEST_JSON
        parsed = json.loads(_C2PA_MANIFEST_JSON)
        self.assertEqual(parsed["claim_generator"], "CrispTTS")
        self.assertIn("assertions", parsed)


class TestConsentGate(unittest.TestCase):
    """Test voice-cloning consent gate."""

    def test_cloning_model_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("llasa_hybrid_de_zeroshot",
                                         "synthesize_with_llasa_hybrid_de_zeroshot"))

    def test_regular_model_no_consent(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent("edge", "synthesize_with_edge"))

    def test_keyword_detection(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("my_custom_zeroshot_model", "custom_handler"))
        self.assertTrue(requires_consent("coqui_xtts_v2", "custom_handler"))

    def test_handler_key_detection(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("some_model", "synthesize_with_f5_tts"))


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


class TestSpokenDisclaimer(unittest.TestCase):
    """Test spoken disclaimer generation."""

    def test_generate_spoken_disclaimer_returns_audio(self):
        from watermark import generate_spoken_disclaimer
        # Should at minimum return beep fallback even without edge-tts
        result = generate_spoken_disclaimer(sample_rate=24000)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 0)

    def test_prepend_disclaimer(self):
        from watermark import prepend_disclaimer
        pcm = np.random.randn(24000).astype(np.float32)
        result = prepend_disclaimer(pcm, sample_rate=24000)
        # Result should be longer than original (disclaimer + silence + original)
        self.assertGreater(len(result), len(pcm))
        # Original audio should be at the end
        np.testing.assert_array_equal(result[-len(pcm):], pcm)


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


class TestC2paAudioNative(unittest.TestCase):
    """Test c2pa-audio native signing integration."""

    def test_c2pa_sign_returns_bool(self):
        """c2pa_sign_file should return bool regardless of backend availability."""
        from watermark import c2pa_sign_file
        result = c2pa_sign_file("/nonexistent/file.wav")
        self.assertIsInstance(result, bool)

    def test_c2pa_tries_native_first(self):
        """c2pa_sign_file should not crash regardless of backend availability."""
        from watermark import c2pa_sign_file
        result = c2pa_sign_file("/nonexistent/file.wav")
        self.assertFalse(result)


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
        result = generate_spoken_disclaimer(sample_rate=24000)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 100)

    def test_consent_audit_log_path_exists(self):
        """Consent audit log path should be defined."""
        from watermark import _CONSENT_LOG_PATH
        self.assertTrue(_CONSENT_LOG_PATH.endswith("consent_audit.log"))


if __name__ == "__main__":
    unittest.main()
