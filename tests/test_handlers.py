"""Tests for handler dispatch, registration, and the CrispASR handler."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestHandlerRegistry(unittest.TestCase):
    """Test the ALL_HANDLERS registry."""

    def test_crispasr_handler_registered(self):
        """CrispASR handler should always be importable (no ML deps)."""
        # Import directly to avoid triggering all other handlers
        from handlers.crispasr_handler import synthesize_with_crispasr
        self.assertTrue(callable(synthesize_with_crispasr))

    def test_handler_function_signature(self):
        """All handlers must accept the standard 6-argument signature."""
        from handlers.crispasr_handler import synthesize_with_crispasr
        import inspect
        sig = inspect.signature(synthesize_with_crispasr)
        params = list(sig.parameters.keys())
        self.assertEqual(len(params), 6)
        expected = [
            "crisptts_model_config", "text",
            "voice_id_or_path_override", "model_params_override",
            "output_file_str", "play_direct",
        ]
        self.assertEqual(params, expected)


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


class TestCLIArgParsing(unittest.TestCase):
    """Test CLI argument structure."""

    def test_argparser_has_verify_flag(self):
        """The --verify flag should be in the parser."""
        from main import main_cli_entrypoint
        import argparse
        # We can't easily test the parser without running main,
        # but we can verify the flag exists in the source
        import inspect
        source = inspect.getsource(main_cli_entrypoint)
        self.assertIn("--verify", source)
        self.assertIn("--translate", source)
        self.assertIn("--translate-from", source)
        self.assertIn("--translate-to", source)


if __name__ == "__main__":
    unittest.main()
