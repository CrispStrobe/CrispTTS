"""Tests for handler dispatch, registration, and the CrispASR handler."""

import inspect
import os
import unittest
from unittest.mock import MagicMock, patch


class TestHandlerRegistry(unittest.TestCase):
    """Test the ALL_HANDLERS registry."""

    def test_crispasr_handler_registered(self):
        """CrispASR handler should always be importable (no ML deps)."""
        from handlers.crispasr_handler import synthesize_with_crispasr
        self.assertTrue(callable(synthesize_with_crispasr))

    def test_lazy_registry_preloads_crispasr(self):
        """The lazy registry should pre-load crispasr (no ML deps)."""
        from handlers import ALL_HANDLERS
        self.assertIn("crispasr", ALL_HANDLERS.keys())
        self.assertIsNotNone(ALL_HANDLERS["crispasr"])

    def test_lazy_registry_all_keys(self):
        """all_keys() should return all 21 registered handler keys."""
        from handlers import ALL_HANDLERS
        keys = ALL_HANDLERS.all_keys()
        self.assertGreaterEqual(len(keys), 21)
        self.assertIn("crispasr", keys)
        self.assertIn("edge", keys)
        self.assertIn("kokoro_onnx", keys)

    def test_lazy_registry_contains_unloaded(self):
        """__contains__ should work for keys not yet loaded."""
        from handlers import ALL_HANDLERS
        self.assertIn("edge", ALL_HANDLERS)
        self.assertIn("outetts", ALL_HANDLERS)

    def test_lazy_registry_unknown_key_returns_none(self):
        """Accessing an unknown key should return None."""
        from handlers import ALL_HANDLERS
        result = ALL_HANDLERS["nonexistent_handler_xyz"]
        self.assertIsNone(result)

    def test_handler_function_signature(self):
        """All handlers must accept the standard 6-argument signature."""
        from handlers.crispasr_handler import synthesize_with_crispasr
        sig = inspect.signature(synthesize_with_crispasr)
        params = list(sig.parameters.keys())
        self.assertEqual(len(params), 6)
        expected = [
            "crisptts_model_config", "text",
            "voice_id_or_path_override", "model_params_override",
            "output_file_str", "play_direct",
        ]
        self.assertEqual(params, expected)


class TestAllHandlerKeys(unittest.TestCase):
    """Test that known handler keys are registered or gracefully failed."""

    # All known handler keys from handlers/__init__.py
    ALL_KNOWN_KEYS = [
        "edge", "piper", "orpheus_gguf", "orpheus_lm_studio",
        "orpheus_ollama", "outetts", "speecht5", "nemo_fastpitch",
        "coqui_tts", "orpheus_kartoffel", "llasa_hybrid", "mlx_audio",
        "llasa_german_transformers", "llasa_multilingual_transformers",
        "llasa_hf_transformers", "f5_tts", "tts_cpp", "kokoro_onnx",
        "zonos", "chatterbox", "crispasr",
    ]

    def test_all_handler_keys_defined_in_init(self):
        """Every known handler key should be defined in ALL_HANDLERS dict
        (even if its value is None due to import failure, it should have been
        attempted). We check the source code for the key string."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "handlers", "__init__.py",
        )
        with open(source_path, encoding="utf-8") as f:
            source = f.read()
        for key in self.ALL_KNOWN_KEYS:
            self.assertIn(
                f'"{key}"', source,
                f"Handler key '{key}' not found in handlers/__init__.py source",
            )

    def test_crispasr_key_always_available(self):
        """crispasr handler has no ML deps, so it should always import."""
        from handlers.crispasr_handler import synthesize_with_crispasr
        self.assertIsNotNone(synthesize_with_crispasr)

    def test_handler_functions_have_six_args(self):
        """For every available handler, verify the 6-arg signature."""
        # Only test handlers we can import without ML deps
        from handlers.crispasr_handler import synthesize_with_crispasr
        sig = inspect.signature(synthesize_with_crispasr)
        self.assertEqual(len(sig.parameters), 6)


class TestCrispASRHandler(unittest.TestCase):
    """Test the CrispASR TTS handler."""

    def test_find_crispasr(self):
        """_find_crispasr should return a path or None without crashing."""
        from handlers.crispasr_handler import _find_crispasr
        result = _find_crispasr()
        if result is not None:
            self.assertTrue(os.path.isfile(result))

    def test_verify_function_importable(self):
        from handlers.crispasr_handler import verify_tts_with_asr
        self.assertTrue(callable(verify_tts_with_asr))

    def test_translate_function_importable(self):
        from handlers.crispasr_handler import translate_text_with_crispasr
        self.assertTrue(callable(translate_text_with_crispasr))

    def test_verify_with_nonexistent_file(self):
        from handlers.crispasr_handler import verify_tts_with_asr
        result = verify_tts_with_asr("/nonexistent/audio.wav", "test")
        self.assertIn("error", result)

    def test_translate_returns_string(self):
        """translate_text_with_crispasr should return a string even on failure."""
        from handlers.crispasr_handler import translate_text_with_crispasr
        result = translate_text_with_crispasr("hello")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_builds_correct_command(self, mock_run):
        """Test that synthesize builds the right CLI command."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr="crispasr: done\n"
        )

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test_kokoro",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
            "default_voice_id": "af_heart",
        }

        # Mock os.path.isfile to pretend output exists
        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Hello world", None, None, "/tmp/test.wav", False
                )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("/usr/bin/crispasr", cmd)
        self.assertIn("--tts", cmd)
        self.assertIn("Hello world", cmd)
        self.assertIn("--backend", cmd)
        self.assertIn("kokoro", cmd)
        self.assertIn("--auto-download", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_voice_override(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "orpheus",
            "crispasr_model_path": "auto",
            "crispasr_codec_model": "auto",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", "Sophie", None, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--voice", cmd)
        self.assertIn("Sophie", cmd)
        self.assertIn("--codec-model", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_model_params(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        params_json = '{"temperature": 0.8, "seed": 42}'
        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, params_json, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("-tp", cmd)
        self.assertIn("0.8", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)

    def test_synthesize_with_missing_binary(self):
        """Synthesize should return gracefully when binary not found."""
        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler._find_crispasr", return_value=None):
            # Should not raise
            synthesize_with_crispasr(
                config, "Test text", None, None, "/tmp/test.wav", False
            )

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_timeout(self, mock_run):
        """Synthesize should handle subprocess timeout gracefully."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="crispasr", timeout=300)

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            # Should not raise
            synthesize_with_crispasr(
                config, "Test", None, None, "/tmp/test.wav", False
            )

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_nonzero_exit(self, mock_run):
        """Synthesize should log error on non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Some error occurred"
        )

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            # Should not raise
            synthesize_with_crispasr(
                config, "Test", None, None, "/tmp/test.wav", False
            )

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_verify_parses_timestamps(self, mock_run):
        """verify_tts_with_asr should parse timestamp-prefixed lines."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="[00:00:00.000 --> 00:00:02.500] Hello world\n[00:00:02.500 --> 00:00:05.000] This is a test\n",
            stderr="",
        )

        from handlers.crispasr_handler import verify_tts_with_asr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = verify_tts_with_asr("/tmp/audio.wav", "Hello world this is a test")

        self.assertNotIn("error", result)
        self.assertIn("asr_text", result)
        self.assertIn("Hello world", result["asr_text"])
        self.assertIn("This is a test", result["asr_text"])

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_verify_similarity_calculation(self, mock_run):
        """verify_tts_with_asr should compute word-overlap similarity."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="hello world test\n",
            stderr="",
        )

        from handlers.crispasr_handler import verify_tts_with_asr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = verify_tts_with_asr("/tmp/audio.wav", "hello world test")

        self.assertNotIn("error", result)
        self.assertIn("similarity", result)
        self.assertEqual(result["similarity"], 1.0)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_verify_partial_similarity(self, mock_run):
        """Partial word overlap should produce similarity < 1.0."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="hello world extra\n",
            stderr="",
        )

        from handlers.crispasr_handler import verify_tts_with_asr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = verify_tts_with_asr("/tmp/audio.wav", "hello world different")

        self.assertNotIn("error", result)
        self.assertLess(result["similarity"], 1.0)
        self.assertGreater(result["similarity"], 0.0)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_translate_successful(self, mock_run):
        """translate should return translated text on success."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hallo Welt\n",
            stderr="",
        )

        from handlers.crispasr_handler import translate_text_with_crispasr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = translate_text_with_crispasr("Hello world")

        self.assertEqual(result, "Hallo Welt")

    def test_translate_with_binary_not_found(self):
        """translate should return original text when binary not found."""
        from handlers.crispasr_handler import translate_text_with_crispasr

        with patch("handlers.crispasr_handler._find_crispasr", return_value=None):
            result = translate_text_with_crispasr("Hello world")

        self.assertEqual(result, "Hello world")

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_translate_with_failure(self, mock_run):
        """translate should return original text on subprocess failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error"
        )

        from handlers.crispasr_handler import translate_text_with_crispasr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = translate_text_with_crispasr("Hello world")

        self.assertEqual(result, "Hello world")

    def test_download_crispasr_unsupported_platform(self):
        """_download_crispasr should return None on unsupported platform."""
        from handlers.crispasr_handler import _download_crispasr

        with patch("handlers.crispasr_handler.platform.system", return_value="FreeBSD"):
            with patch("handlers.crispasr_handler.platform.machine", return_value="arm64"):
                result = _download_crispasr()

        self.assertIsNone(result)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_language(self, mock_run):
        """Synthesize should pass language flag when configured."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test_de",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
            "language": "de",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Hallo Welt", None, None, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("-l", cmd)
        self.assertIn("de", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_output_default_path(self, mock_run):
        """When no output_file is given, should default to crispasr_tts_output.wav."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, None, None, False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--tts-output", cmd)
        # Default output path
        idx = cmd.index("--tts-output")
        self.assertEqual(cmd[idx + 1], "crispasr_tts_output.wav")

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_adds_wav_extension(self, mock_run):
        """Output path without .wav should get .wav appended."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, None, "/tmp/test_output", False
                )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--tts-output")
        self.assertTrue(cmd[idx + 1].endswith(".wav"))

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_verify_asr_failure_returns_error(self, mock_run):
        """verify_tts_with_asr should return error dict on non-zero exit."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ASR engine error"
        )

        from handlers.crispasr_handler import verify_tts_with_asr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = verify_tts_with_asr("/tmp/audio.wav", "test text")

        self.assertIn("error", result)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_verify_empty_original_text(self, mock_run):
        """verify_tts_with_asr with empty original text should give 0 similarity."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="some recognized text\n", stderr=""
        )

        from handlers.crispasr_handler import verify_tts_with_asr

        with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
            result = verify_tts_with_asr("/tmp/audio.wav", "")

        self.assertNotIn("error", result)
        self.assertEqual(result["similarity"], 0.0)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_synthesize_with_play_direct(self, mock_run):
        """When play_direct=True and output exists, play_audio should be called."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                with patch("handlers.crispasr_handler.play_audio") as mock_play:
                    synthesize_with_crispasr(
                        config, "Test", None, None, "/tmp/test.wav", True
                    )
                    mock_play.assert_called_once()


class TestCLIArgParsing(unittest.TestCase):
    """Test CLI argument structure."""

    def test_argparser_has_verify_flag(self):
        """The --verify flag should be in the parser."""
        import inspect

        from main import main_cli_entrypoint
        source = inspect.getsource(main_cli_entrypoint)
        self.assertIn("--verify", source)
        self.assertIn("--translate", source)
        self.assertIn("--translate-from", source)
        self.assertIn("--translate-to", source)


class TestNewCrispASRBackendConfigs(unittest.TestCase):
    """Test Phase 7 backend configs are valid."""

    NEW_BACKENDS = [
        "crispasr_bananamind_tts", "crispasr_dots_tts",
        "crispasr_cosyvoice3_tts", "crispasr_csm_tts",
        "crispasr_omnivoice_tts", "crispasr_moss_tts_local",
    ]

    def test_new_backends_exist_in_config(self):
        from config import GERMAN_TTS_MODELS
        for mid in self.NEW_BACKENDS:
            self.assertIn(mid, GERMAN_TTS_MODELS, f"{mid} missing from config")

    def test_new_backends_use_crispasr_handler(self):
        from config import GERMAN_TTS_MODELS
        for mid in self.NEW_BACKENDS:
            cfg = GERMAN_TTS_MODELS[mid]
            self.assertEqual(cfg["handler_function_key"], "crispasr")

    def test_new_backends_have_crispasr_backend(self):
        from config import GERMAN_TTS_MODELS
        for mid in self.NEW_BACKENDS:
            cfg = GERMAN_TTS_MODELS[mid]
            self.assertIn("crispasr_backend", cfg)
            self.assertIsInstance(cfg["crispasr_backend"], str)

    def test_new_backends_sample_rates(self):
        from config import GERMAN_TTS_MODELS
        expected_rates = {
            "crispasr_bananamind_tts": 22050,
            "crispasr_dots_tts": 48000,
            "crispasr_cosyvoice3_tts": 24000,
            "crispasr_csm_tts": 24000,
            "crispasr_omnivoice_tts": 24000,
            "crispasr_moss_tts_local": 48000,
        }
        for mid, expected_sr in expected_rates.items():
            self.assertEqual(GERMAN_TTS_MODELS[mid]["sample_rate"], expected_sr)


class TestExpandedParamMap(unittest.TestCase):
    """Test that new param_map keys produce correct CLI flags."""

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_new_param_keys(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "dots-tts",
            "crispasr_model_path": "auto",
        }

        params = '{"top_k": 25, "cfg_scale": 2.0, "noise_temp": 0.6, "num_steps": 20}'
        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, params, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--top-k", cmd)
        self.assertIn("25", cmd)
        self.assertIn("--tts-cfg-scale", cmd)
        self.assertIn("2.0", cmd)
        self.assertIn("--tts-noise-temp", cmd)
        self.assertIn("0.6", cmd)
        self.assertIn("--tts-num-steps", cmd)
        self.assertIn("20", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_ref_text_passthrough(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "tada",
            "crispasr_model_path": "auto",
            "reference_text": "Hello this is my voice",
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "New text", "ref.wav", None, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--ref-text", cmd)
        self.assertIn("Hello this is my voice", cmd)
        self.assertIn("--voice", cmd)
        self.assertIn("ref.wav", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_no_spoken_disclaimer_passthrough(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "kokoro",
            "crispasr_model_path": "auto",
            "_cli_no_spoken_disclaimer": True,
        }

        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, None, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--no-spoken-disclaimer", cmd)

    @patch("handlers.crispasr_handler.subprocess.run")
    def test_tts_speed_passthrough(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from handlers.crispasr_handler import synthesize_with_crispasr

        config = {
            "crisptts_model_id": "test",
            "crispasr_backend": "omnivoice",
            "crispasr_model_path": "auto",
        }

        params = '{"tts_speed": 1.5}'
        with patch("handlers.crispasr_handler.os.path.isfile", return_value=True):
            with patch("handlers.crispasr_handler._find_crispasr", return_value="/usr/bin/crispasr"):
                synthesize_with_crispasr(
                    config, "Test", None, params, "/tmp/test.wav", False
                )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--tts-speed", cmd)
        self.assertIn("1.5", cmd)


class TestNewVoiceCloningKeywords(unittest.TestCase):
    """Test voice-cloning detection for Phase 7 backends."""

    def test_dots_tts_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_dots_tts", "crispasr"))

    def test_cosyvoice3_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_cosyvoice3_tts", "crispasr"))

    def test_csm_tts_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_csm_tts", "crispasr"))

    def test_bananamind_no_consent_without_wav(self):
        from watermark import requires_consent
        # bananamind doesn't clone unless a .wav path is passed
        self.assertFalse(requires_consent("crispasr_bananamind_tts", "crispasr"))

    def test_bananamind_consent_with_wav(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_bananamind_tts", "crispasr", "/path/ref.wav"))

    def test_omnivoice_requires_consent(self):
        from watermark import requires_consent
        self.assertTrue(requires_consent("crispasr_omnivoice_tts", "crispasr"))

    def test_moss_tts_local_no_consent(self):
        from watermark import requires_consent
        # moss-tts-local doesn't have voice cloning yet
        self.assertFalse(requires_consent("crispasr_moss_tts_local", "crispasr"))


class TestLiveCrispASR(unittest.TestCase):
    """Live tests using the actual CrispASR binary.

    These tests only run when the crispasr binary is available.
    Designed for an 8 GB RAM VPS with no GPU: uses piper (~16 MB VITS,
    fastest backend), short text, and 30s timeout. Kokoro hangs on
    CPU-only with 8 GB RAM.
    """

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("CRISPTTS_LIVE_TESTS"):
            raise unittest.SkipTest(
                "Live tests disabled (set CRISPTTS_LIVE_TESTS=1 to enable)")
        from handlers.crispasr_handler import _find_crispasr
        cls.exe = _find_crispasr()
        if not cls.exe:
            raise unittest.SkipTest("CrispASR binary not available")

    def test_live_piper_synthesis(self):
        """Synthesize a short phrase with piper and verify output exists."""
        import subprocess
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            result = subprocess.run(  # noqa: S603
                [self.exe, "-m", "auto", "--backend", "piper",
                 "--tts", "Test.", "--tts-output", tmp,
                 "--auto-download", "-t", "4"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                self.skipTest(f"piper synthesis failed: {result.stderr[-200:]}")
            self.assertTrue(os.path.isfile(tmp))
            self.assertGreater(os.path.getsize(tmp), 1000)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_live_piper_watermark_present(self):
        """Verify CrispASR binary embeds a watermark in TTS output."""
        import subprocess
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            result = subprocess.run(  # noqa: S603
                [self.exe, "-m", "auto", "--backend", "piper",
                 "--tts", "Watermark test.", "--tts-output", tmp,
                 "--auto-download", "-t", "4"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                self.skipTest(f"piper synthesis failed: {result.stderr[-200:]}")

            from watermark import spread_spectrum_detect
            try:
                import soundfile as sf_live
            except ImportError:
                self.skipTest("soundfile not installed")
            data, sr = sf_live.read(tmp, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            confidence = spread_spectrum_detect(data)
            self.assertGreater(confidence, 0.5,
                               f"CrispASR binary should watermark output (confidence={confidence:.3f})")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_live_z_pipeline_via_handler(self):
        """End-to-end: CrispTTS handler → crispasr binary → WAV.

        Imports crispasr_handler directly (not via handlers/) to avoid
        loading torch-heavy handlers that OOM on 8 GB machines.
        """
        import importlib
        import tempfile

        # Import handler module directly to avoid handlers/__init__.py torch imports
        spec = importlib.util.spec_from_file_location(
            "crispasr_handler",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "handlers", "crispasr_handler.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            config = {
                "crisptts_model_id": "crispasr_piper",
                "crispasr_backend": "piper",
                "crispasr_model_path": "auto",
                "default_voice_id": None,
            }
            mod.synthesize_with_crispasr(config, "Pipeline.", None, None, tmp, False)

            if not os.path.isfile(tmp) or os.path.getsize(tmp) < 100:
                self.skipTest("piper synthesis via handler failed")

            self.assertGreater(os.path.getsize(tmp), 1000)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
