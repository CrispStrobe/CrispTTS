"""Tests for config.py — model definitions, handler mappings, constants."""

import unittest


class TestGermanTTSModels(unittest.TestCase):
    """Validate the GERMAN_TTS_MODELS configuration dict."""

    def test_config_importable(self):
        from config import GERMAN_TTS_MODELS
        self.assertIsInstance(GERMAN_TTS_MODELS, dict)
        self.assertGreater(len(GERMAN_TTS_MODELS), 20)

    def test_all_models_have_handler_key(self):
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            self.assertIn(
                "handler_function_key", cfg,
                f"Model '{model_id}' missing 'handler_function_key'",
            )

    def test_handler_keys_are_strings(self):
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            key = cfg.get("handler_function_key")
            self.assertIsInstance(
                key, str,
                f"Model '{model_id}' handler_function_key is not a string: {key}",
            )

    def test_crispasr_models_have_backend(self):
        """CrispASR models must specify crispasr_backend."""
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            if cfg.get("handler_function_key") == "crispasr":
                self.assertIn(
                    "crispasr_backend", cfg,
                    f"CrispASR model '{model_id}' missing 'crispasr_backend'",
                )

    def test_sample_rates_are_positive(self):
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            sr = cfg.get("sample_rate")
            if sr is not None:
                self.assertGreater(
                    sr, 0,
                    f"Model '{model_id}' has invalid sample_rate: {sr}",
                )

    def test_available_voices_are_lists(self):
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            voices = cfg.get("available_voices")
            if voices is not None:
                self.assertIsInstance(
                    voices, list,
                    f"Model '{model_id}' available_voices is not a list",
                )

    def test_no_duplicate_model_ids(self):
        from config import GERMAN_TTS_MODELS
        ids = list(GERMAN_TTS_MODELS.keys())
        self.assertEqual(len(ids), len(set(ids)), "Duplicate model IDs found")

    def test_crispasr_model_count(self):
        """At least 8 CrispASR models should be configured."""
        from config import GERMAN_TTS_MODELS
        crispasr_count = sum(
            1 for cfg in GERMAN_TTS_MODELS.values()
            if cfg.get("handler_function_key") == "crispasr"
        )
        self.assertGreaterEqual(crispasr_count, 8)


class TestConstants(unittest.TestCase):
    """Validate global constants."""

    def test_api_urls(self):
        from config import LM_STUDIO_API_URL_DEFAULT, OLLAMA_API_URL_DEFAULT
        self.assertIn("http", LM_STUDIO_API_URL_DEFAULT)
        self.assertIn("http", OLLAMA_API_URL_DEFAULT)

    def test_orpheus_voices(self):
        from config import ORPHEUS_AVAILABLE_VOICES_BASE
        self.assertIsInstance(ORPHEUS_AVAILABLE_VOICES_BASE, list)
        self.assertGreater(len(ORPHEUS_AVAILABLE_VOICES_BASE), 0)


if __name__ == "__main__":
    unittest.main()
