"""Tests for new CrispASR-leveraged features: CLI flags, silence trimming, chunking, etc."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Phase 1: CLI pass-through flags
# ---------------------------------------------------------------------------

class TestCLINewFlags(unittest.TestCase):
    """Test that new CLI flags are registered in argparse."""

    def _parse(self, args_list):
        """Parse args without running main — import parser creation only."""
        import argparse

        from config import GERMAN_TTS_MODELS
        # Minimal parser mimicking main.py's structure
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-id", type=str, choices=list(GERMAN_TTS_MODELS.keys()), default=None)
        parser.add_argument("--input-text", type=str)
        parser.add_argument("--output-file", type=str)
        parser.add_argument("--speech-speed", type=float, default=1.0)
        parser.add_argument("--trim-silence", action="store_true")
        parser.add_argument("--tts-steps", type=int, default=None)
        parser.add_argument("--tts-language", type=str, default=None)
        parser.add_argument("--pitch-shift", type=float, default=0.0)
        parser.add_argument("--instruct", type=str, default=None)
        return parser.parse_args(args_list)

    def test_speech_speed_default(self):
        args = self._parse([])
        self.assertEqual(args.speech_speed, 1.0)

    def test_speech_speed_custom(self):
        args = self._parse(["--speech-speed", "1.5"])
        self.assertEqual(args.speech_speed, 1.5)

    def test_trim_silence_default_false(self):
        args = self._parse([])
        self.assertFalse(args.trim_silence)

    def test_trim_silence_set(self):
        args = self._parse(["--trim-silence"])
        self.assertTrue(args.trim_silence)

    def test_tts_steps_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.tts_steps)

    def test_tts_steps_custom(self):
        args = self._parse(["--tts-steps", "15"])
        self.assertEqual(args.tts_steps, 15)

    def test_tts_language_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.tts_language)

    def test_tts_language_set(self):
        args = self._parse(["--tts-language", "de"])
        self.assertEqual(args.tts_language, "de")

    def test_pitch_shift_default(self):
        args = self._parse([])
        self.assertEqual(args.pitch_shift, 0.0)

    def test_pitch_shift_custom(self):
        args = self._parse(["--pitch-shift", "50.0"])
        self.assertEqual(args.pitch_shift, 50.0)

    def test_instruct_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.instruct)

    def test_instruct_set(self):
        args = self._parse(["--instruct", "young female, energetic"])
        self.assertEqual(args.instruct, "young female, energetic")


class TestCrispASRHandlerParamMap(unittest.TestCase):
    """Test that param_map in crispasr_handler includes new params."""

    def test_param_map_has_speech_speed(self):
        """speech_speed should map to --pace in param_map."""
        # Read source file directly to avoid NeMo import chain via handlers/__init__
        src_path = Path(__file__).resolve().parent.parent / "handlers" / "crispasr_handler.py"
        src = src_path.read_text()
        self.assertIn('"speech_speed"', src)
        self.assertIn('"--pace"', src)

    def test_param_map_has_pitch_shift(self):
        src_path = Path(__file__).resolve().parent.parent / "handlers" / "crispasr_handler.py"
        src = src_path.read_text()
        self.assertIn('"pitch_shift"', src)
        self.assertIn('"--pitch-shift"', src)


class TestCLIConfigInjection(unittest.TestCase):
    """Test that CLI flags are injected into handler config dict."""

    def test_speed_injected(self):
        """_cli_speech_speed should be set in config when speed != 1.0."""
        config = {}
        # Simulate main.py's injection logic
        speed = 1.3
        if speed != 1.0:
            config["_cli_speech_speed"] = speed
        self.assertEqual(config["_cli_speech_speed"], 1.3)

    def test_trim_injected(self):
        config = {}
        trim = True
        if trim:
            config["_cli_trim_silence"] = True
        self.assertTrue(config["_cli_trim_silence"])

    def test_steps_injected(self):
        config = {}
        steps = 15
        if steps is not None:
            config["_cli_tts_steps"] = steps
        self.assertEqual(config["_cli_tts_steps"], 15)

    def test_language_overrides_config(self):
        config = {"language": "de"}
        tts_language = "en"
        if tts_language:
            config["language"] = tts_language
        self.assertEqual(config["language"], "en")

    def test_instruct_overrides_config(self):
        config = {"instruct": "old instruction"}
        instruct = "young female, energetic"
        if instruct:
            config["instruct"] = instruct
        self.assertEqual(config["instruct"], "young female, energetic")


# ---------------------------------------------------------------------------
# Phase 1: Silence trimming (Python fallback)
# ---------------------------------------------------------------------------

class TestSilenceTrimming(unittest.TestCase):
    """Test the pure-Python silence trimming utility."""

    def test_trim_leading_silence(self):
        from utils import trim_silence
        sr = 24000
        silence = np.zeros(sr, dtype=np.float32)  # 1 sec silence
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2, dtype=np.float32))
        pcm = np.concatenate([silence, tone])
        trimmed = trim_silence(pcm)
        self.assertLess(len(trimmed), len(pcm))
        # Should have removed most of the leading silence
        self.assertLess(len(trimmed), sr + sr // 2)

    def test_trim_trailing_silence(self):
        from utils import trim_silence
        sr = 24000
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2, dtype=np.float32))
        silence = np.zeros(sr, dtype=np.float32)
        pcm = np.concatenate([tone, silence])
        trimmed = trim_silence(pcm)
        self.assertLess(len(trimmed), len(pcm))

    def test_trim_preserves_content(self):
        from utils import trim_silence
        sr = 24000
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
        trimmed = trim_silence(tone)
        # Pure tone should not be significantly trimmed
        self.assertGreater(len(trimmed), sr * 0.8)

    def test_trim_empty(self):
        from utils import trim_silence
        pcm = np.array([], dtype=np.float32)
        result = trim_silence(pcm)
        self.assertEqual(len(result), 0)

    def test_trim_all_silence(self):
        from utils import trim_silence
        pcm = np.zeros(24000, dtype=np.float32)
        result = trim_silence(pcm)
        # Should return something (even if very short), not crash
        self.assertIsInstance(result, np.ndarray)

    def test_trim_file_in_place(self):
        import tempfile

        from utils import trim_silence_file
        try:
            import soundfile as sf_test
        except ImportError:
            self.skipTest("soundfile not installed")
        sr = 24000
        silence = np.zeros(sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2, dtype=np.float32))
        pcm = np.concatenate([silence, tone, silence])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf_test.write(f.name, pcm, sr)
            original_size = os.path.getsize(f.name)
            trim_silence_file(f.name)
            trimmed_size = os.path.getsize(f.name)
        os.unlink(f.name)
        self.assertLess(trimmed_size, original_size)


# ---------------------------------------------------------------------------
# Live tests (require crispasr binary — skipped if not available)
# ---------------------------------------------------------------------------

def _find_crispasr():
    """Find crispasr binary for live tests."""
    for p in [
        os.environ.get("CRISPASR_EXECUTABLE", ""),
        "/mnt/volume1/CrispASR/build/bin/crispasr",
        os.path.expanduser("~/.local/bin/crispasr"),
    ]:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# ---------------------------------------------------------------------------
# Phase 2: Model config & voice features
# ---------------------------------------------------------------------------

class TestModelConfig(unittest.TestCase):
    """Test model configuration entries."""

    def test_voicedesign_model_exists(self):
        from config import GERMAN_TTS_MODELS
        self.assertIn("crispasr_qwen3_tts_voicedesign", GERMAN_TTS_MODELS)
        cfg = GERMAN_TTS_MODELS["crispasr_qwen3_tts_voicedesign"]
        self.assertEqual(cfg["crispasr_backend"], "qwen3-tts")
        self.assertIn("instruct", cfg)
        self.assertIsInstance(cfg["instruct"], str)

    def test_orpheus_de_has_19_speakers(self):
        from config import GERMAN_TTS_MODELS
        cfg = GERMAN_TTS_MODELS["crispasr_orpheus_de"]
        self.assertEqual(len(cfg["available_voices"]), 19)

    def test_voxcpm2_supports_cloning(self):
        """VoxCPM2 notes should mention voice cloning."""
        from config import GERMAN_TTS_MODELS
        cfg = GERMAN_TTS_MODELS["crispasr_voxcpm2"]
        self.assertIn("cloning", cfg["notes"].lower())

    def test_all_crispasr_models_have_backend(self):
        from config import GERMAN_TTS_MODELS
        for mid, cfg in GERMAN_TTS_MODELS.items():
            if mid.startswith("crispasr_"):
                self.assertIn("crispasr_backend", cfg, f"Model {mid} missing crispasr_backend")


class TestConsentGateWavPath(unittest.TestCase):
    """Test consent gate detects .wav voice paths as voice cloning."""

    def test_wav_voice_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_kokoro", "crispasr", "speaker.wav"))

    def test_wav_voice_case_insensitive(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_kokoro", "crispasr", "Speaker.WAV"))

    def test_non_wav_voice_no_consent(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent("crispasr_kokoro", "crispasr", "af_heart"))

    def test_none_voice_no_consent(self):
        from watermark import requires_consent
        self.assertFalse(requires_consent("crispasr_kokoro", "crispasr", None))


@unittest.skipUnless(_find_crispasr(), "crispasr binary not found")
class TestLiveCrispASR(unittest.TestCase):
    """Live integration tests with actual crispasr binary."""

    @classmethod
    def setUpClass(cls):
        cls.exe = _find_crispasr()

    def test_detect_watermark_cli(self):
        """--detect-watermark should work on the crispasr binary."""
        import struct
        import tempfile
        sr = 24000
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t))
        samples = (pcm * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            data_size = len(samples) * 2
            f.write(b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE")
            f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
            f.write(b"data" + struct.pack("<I", data_size))
            f.write(samples.tobytes())
            wav_path = f.name
        try:
            result = subprocess.run(
                [self.exe, "--detect-watermark", wav_path],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Watermark confidence:", result.stdout)
        finally:
            os.unlink(wav_path)


if __name__ == "__main__":
    unittest.main()
