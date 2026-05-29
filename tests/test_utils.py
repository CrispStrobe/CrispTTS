"""Tests for utils.py — text extraction, audio I/O, helper functions."""

import os
import tempfile
import unittest


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

    def test_extract_text_from_txt(self):
        from pathlib import Path
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
        from pathlib import Path
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


class TestListModels(unittest.TestCase):
    """Test model listing functions."""

    def test_list_available_models(self):
        from config import GERMAN_TTS_MODELS
        from utils import list_available_models
        # Should not raise
        list_available_models(GERMAN_TTS_MODELS)

    def test_get_voice_info_unknown_model(self):
        from config import GERMAN_TTS_MODELS
        from utils import get_voice_info
        # Should handle gracefully
        get_voice_info("nonexistent_model_xyz", GERMAN_TTS_MODELS)


class TestSaveAudio(unittest.TestCase):
    """Test audio save utility."""

    def test_save_audio_importable(self):
        from utils import save_audio
        self.assertTrue(callable(save_audio))

    def test_play_audio_importable(self):
        from utils import play_audio
        self.assertTrue(callable(play_audio))


if __name__ == "__main__":
    unittest.main()
