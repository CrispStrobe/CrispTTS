"""Tests for utils.py — text extraction, audio I/O, helper functions."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestTextExtraction(unittest.TestCase):
    """Test text extraction from various formats."""

    def test_get_text_from_direct_input(self):
        from utils import get_text_from_input
        result = get_text_from_input("Hello world", None)
        self.assertEqual(result, "Hello world")

    def test_get_text_from_none(self):
        from utils import get_text_from_input
        result = get_text_from_input(None, None)
        self.assertIsNone(result)

    def test_get_text_from_txt_file(self):
        from utils import get_text_from_input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content from file")
            f.flush()
            result = get_text_from_input(None, f.name)
        os.unlink(f.name)
        self.assertEqual(result, "Test content from file")

    def test_get_text_from_empty_file(self):
        from utils import get_text_from_input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            result = get_text_from_input(None, f.name)
        os.unlink(f.name)
        # Empty file should return None or empty string
        self.assertFalse(result)

    def test_get_text_from_nonexistent_file(self):
        from utils import get_text_from_input
        result = get_text_from_input(None, "/nonexistent/path/file.txt")
        self.assertIsNone(result)

    def test_get_text_prefers_direct_text_over_file(self):
        """When both direct text and file path are provided, direct text wins."""
        from utils import get_text_from_input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content")
            f.flush()
            result = get_text_from_input("Direct text", f.name)
        os.unlink(f.name)
        self.assertEqual(result, "Direct text")

    def test_extract_text_from_txt(self):
        from utils import extract_text_from_txt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Plain text content\nSecond line")
            f.flush()
            result = extract_text_from_txt(Path(f.name))
        os.unlink(f.name)
        self.assertIsNotNone(result)
        self.assertIn("Plain text content", result)
        self.assertIn("Second line", result)

    def test_extract_md(self):
        try:
            from utils import extract_text_from_md
        except ImportError:
            self.skipTest("markdown/bs4 not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\n\nSome **bold** text.\n")
            f.flush()
            result = extract_text_from_md(Path(f.name))
        os.unlink(f.name)
        if result is None:
            self.skipTest("markdown/bs4 libraries not available at runtime")
        self.assertIn("Heading", result)
        self.assertIn("bold", result)

    def test_extract_text_from_html(self):
        """Test HTML text extraction with a simple file."""
        from utils import extract_text_from_html
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><body><h1>Title</h1><p>Hello world</p><script>var x=1;</script></body></html>")
            f.flush()
            result = extract_text_from_html(Path(f.name))
        os.unlink(f.name)
        if result is None:
            self.skipTest("BeautifulSoup4 not available")
        self.assertIn("Title", result)
        self.assertIn("Hello world", result)
        # Script content should be stripped
        self.assertNotIn("var x", result)

    def test_extract_text_from_html_via_get_text_from_input(self):
        """Test that .html files route through get_text_from_input correctly."""
        from utils import get_text_from_input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<p>HTML paragraph</p>")
            f.flush()
            result = get_text_from_input(None, f.name)
        os.unlink(f.name)
        if result is None:
            self.skipTest("BeautifulSoup4 not available")
        self.assertIn("HTML paragraph", result)

    def test_extract_text_from_epub_without_library(self):
        """epub extraction returns None if ebooklib not installed."""
        from utils import epub, extract_text_from_epub
        if epub is not None:
            self.skipTest("ebooklib IS installed, cannot test missing-library path")
        result = extract_text_from_epub(Path("/dummy/test.epub"))
        self.assertIsNone(result)

    def test_extract_text_from_pdf_without_library(self):
        """PDF extraction returns None if pypdfium2 not installed."""
        from utils import extract_text_from_pdf, pdfium
        if pdfium is not None:
            self.skipTest("pypdfium2 IS installed, cannot test missing-library path")
        result = extract_text_from_pdf(Path("/dummy/test.pdf"))
        self.assertIsNone(result)

    def test_get_text_from_unsupported_extension(self):
        """Unsupported file extensions should return None."""
        from utils import get_text_from_input
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Some content")
            f.flush()
            result = get_text_from_input(None, f.name)
        os.unlink(f.name)
        self.assertIsNone(result)


class TestListModels(unittest.TestCase):
    """Test model listing functions."""

    def test_list_available_models(self):
        from config import GERMAN_TTS_MODELS
        from utils import list_available_models
        # Should not raise
        list_available_models(GERMAN_TTS_MODELS)

    def test_list_available_models_empty(self):
        from utils import list_available_models
        # Should handle empty dict gracefully
        list_available_models({})

    def test_get_voice_info_unknown_model(self):
        from config import GERMAN_TTS_MODELS
        from utils import get_voice_info
        # Should handle gracefully
        get_voice_info("nonexistent_model_xyz", GERMAN_TTS_MODELS)

    def test_get_voice_info_known_model(self):
        from config import GERMAN_TTS_MODELS
        from utils import get_voice_info
        # Should not raise for a known model
        get_voice_info("edge", GERMAN_TTS_MODELS)


class TestSaveAudio(unittest.TestCase):
    """Test audio save utility."""

    def test_save_audio_importable(self):
        from utils import save_audio
        self.assertTrue(callable(save_audio))

    def test_play_audio_importable(self):
        from utils import play_audio
        self.assertTrue(callable(play_audio))

    def test_save_audio_no_data(self):
        """save_audio with no data should return without error."""
        from utils import save_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        try:
            # None data, not source_is_path -> should just warn and return
            save_audio(None, output_path, source_is_path=False)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_audio_source_path_nonexistent(self):
        """save_audio with source_is_path=True and missing file should not crash."""
        from utils import save_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        try:
            save_audio("/nonexistent/audio.wav", output_path, source_is_path=True)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_audio_creates_parent_dirs(self):
        """save_audio should create parent directories."""
        from utils import save_audio
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub", "dir", "output.wav")
            # No data, but parent dirs should be created
            save_audio(None, output_path, source_is_path=False)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "sub", "dir")))


class TestOrpheusUtils(unittest.TestCase):
    """Test Orpheus-specific utility functions."""

    def test_orpheus_format_prompt_known_voice(self):
        from utils import orpheus_format_prompt
        result = orpheus_format_prompt("Hello", "tara", ["tara", "leo", "jess"])
        self.assertEqual(result, "<|audio|>tara: Hello<|eot_id|>")

    def test_orpheus_format_prompt_case_insensitive(self):
        from utils import orpheus_format_prompt
        result = orpheus_format_prompt("Hello", "TARA", ["tara", "leo"])
        self.assertEqual(result, "<|audio|>tara: Hello<|eot_id|>")

    def test_orpheus_format_prompt_unknown_voice_fallback(self):
        from utils import orpheus_format_prompt
        result = orpheus_format_prompt("Hello", "unknown_voice", ["tara", "leo"])
        # Should fall back to first voice in list
        self.assertEqual(result, "<|audio|>tara: Hello<|eot_id|>")

    def test_orpheus_format_prompt_empty_voice_list(self):
        from utils import orpheus_format_prompt
        result = orpheus_format_prompt("Hello", "unknown", [])
        # Should fall back to ORPHEUS_DEFAULT_VOICE
        self.assertIn("<|audio|>", result)
        self.assertIn("Hello", result)

    def test_orpheus_format_prompt_non_list_voices(self):
        """Non-list available_voices should be handled gracefully."""
        from utils import orpheus_format_prompt
        result = orpheus_format_prompt("Hello", "tara", "not_a_list")
        self.assertIn("Hello", result)

    def test_orpheus_turn_token_into_id_valid(self):
        from utils import orpheus_turn_token_into_id
        # Valid token string
        result = orpheus_turn_token_into_id("<custom_token_100>", 0)
        self.assertIsInstance(result, int)

    def test_orpheus_turn_token_into_id_index_variation(self):
        from utils import orpheus_turn_token_into_id
        # Same token, different index -> different result
        r0 = orpheus_turn_token_into_id("<custom_token_5000>", 0)
        r1 = orpheus_turn_token_into_id("<custom_token_5000>", 1)
        self.assertNotEqual(r0, r1)

    def test_orpheus_turn_token_into_id_invalid_format(self):
        from utils import orpheus_turn_token_into_id
        result = orpheus_turn_token_into_id("not_a_token", 0)
        self.assertIsNone(result)

    def test_orpheus_turn_token_into_id_empty_string(self):
        from utils import orpheus_turn_token_into_id
        result = orpheus_turn_token_into_id("", 0)
        self.assertIsNone(result)

    def test_orpheus_turn_token_into_id_malformed_number(self):
        from utils import orpheus_turn_token_into_id
        result = orpheus_turn_token_into_id("<custom_token_abc>", 0)
        self.assertIsNone(result)


class TestSuppressOutput(unittest.TestCase):
    """Test the SuppressOutput context manager."""

    def test_suppress_stdout(self):
        from utils import SuppressOutput
        original_stdout = sys.stdout
        with SuppressOutput(suppress_stdout=True):
            self.assertIsNot(sys.stdout, original_stdout)
        self.assertIs(sys.stdout, original_stdout)

    def test_suppress_stderr(self):
        from utils import SuppressOutput
        original_stderr = sys.stderr
        with SuppressOutput(suppress_stdout=False, suppress_stderr=True):
            self.assertIsNot(sys.stderr, original_stderr)
        self.assertIs(sys.stderr, original_stderr)

    def test_suppress_with_stringio_capture(self):
        from utils import SuppressOutput
        with SuppressOutput(suppress_stdout=True, use_stringio=True) as s:
            print("captured text")
        self.assertIn("captured text", s.captured_output)

    def test_suppress_restores_on_exception(self):
        from utils import SuppressOutput
        original_stdout = sys.stdout
        try:
            with SuppressOutput(suppress_stdout=True):
                raise ValueError("test error")
        except ValueError:
            pass
        self.assertIs(sys.stdout, original_stdout)


class TestHelperFunctions(unittest.TestCase):
    """Test miscellaneous helper functions."""

    def test_get_huggingface_cache_dir_returns_path(self):
        from utils import get_huggingface_cache_dir
        result = get_huggingface_cache_dir()
        self.assertIsInstance(result, Path)

    def test_get_huggingface_cache_dir_respects_hf_home(self):
        from utils import get_huggingface_cache_dir
        with patch.dict(os.environ, {"HF_HOME": "/tmp/test_hf_home"}, clear=False):
            result = get_huggingface_cache_dir()
            self.assertEqual(result, Path("/tmp/test_hf_home"))

    def test_pydub_available_is_boolean(self):
        from utils import PYDUB_AVAILABLE
        self.assertIsInstance(PYDUB_AVAILABLE, bool)

    def test_soundfile_available_is_boolean(self):
        from utils import SOUNDFILE_AVAILABLE
        self.assertIsInstance(SOUNDFILE_AVAILABLE, bool)

    def test_sounddevice_available_is_boolean(self):
        from utils import SOUNDDEVICE_AVAILABLE
        self.assertIsInstance(SOUNDDEVICE_AVAILABLE, bool)


if __name__ == "__main__":
    unittest.main()
