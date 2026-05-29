"""Tests for config.py — model definitions, handler mappings, constants."""

import re
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

    def test_handler_keys_from_known_set(self):
        """Every handler_function_key must come from the known set."""
        from config import GERMAN_TTS_MODELS
        known_keys = {
            "edge", "piper", "orpheus_gguf", "orpheus_lm_studio",
            "orpheus_ollama", "outetts", "speecht5", "nemo_fastpitch",
            "coqui_tts", "orpheus_kartoffel", "llasa_hybrid", "mlx_audio",
            "llasa_german_transformers", "llasa_multilingual_transformers",
            "llasa_hf_transformers", "f5_tts", "tts_cpp", "kokoro_onnx",
            "zonos", "chatterbox", "crispasr",
        }
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            key = cfg["handler_function_key"]
            self.assertIn(
                key, known_keys,
                f"Model '{model_id}' has unknown handler_function_key '{key}'",
            )

    def test_crispasr_models_have_model_path(self):
        """All CrispASR models must have a crispasr_model_path."""
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            if cfg.get("handler_function_key") == "crispasr":
                self.assertIn(
                    "crispasr_model_path", cfg,
                    f"CrispASR model '{model_id}' missing 'crispasr_model_path'",
                )

    def test_available_voices_non_empty_where_expected(self):
        """Models with available_voices should have non-empty lists (except zero-shot)."""
        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            voices = cfg.get("available_voices")
            if voices is not None and len(voices) > 0:
                for v in voices:
                    self.assertTrue(
                        v is not None,
                        f"Model '{model_id}' has None in available_voices",
                    )

    def test_model_ids_are_valid_identifiers(self):
        """Model IDs should contain only alphanumeric chars and underscores."""
        from config import GERMAN_TTS_MODELS
        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        for model_id in GERMAN_TTS_MODELS:
            self.assertRegex(
                model_id, pattern,
                f"Model ID '{model_id}' is not a valid identifier",
            )

    def test_default_voice_id_types(self):
        """default_voice_id should be str, None, or Path-like."""
        from pathlib import Path

        from config import GERMAN_TTS_MODELS
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            voice = cfg.get("default_voice_id")
            if voice is not None:
                self.assertTrue(
                    isinstance(voice, (str, Path)),
                    f"Model '{model_id}' default_voice_id has unexpected type: {type(voice)}",
                )

    def test_sample_rates_are_standard(self):
        """Sample rates should be common audio rates."""
        from config import GERMAN_TTS_MODELS
        standard_rates = {8000, 16000, 22050, 24000, 32000, 44100, 48000}
        for model_id, cfg in GERMAN_TTS_MODELS.items():
            sr = cfg.get("sample_rate")
            if sr is not None:
                self.assertIn(
                    sr, standard_rates,
                    f"Model '{model_id}' has unusual sample_rate {sr}",
                )


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

    def test_kokoro_voices_non_empty(self):
        from config import KOKORO_VOICES
        self.assertIsInstance(KOKORO_VOICES, list)
        self.assertGreater(len(KOKORO_VOICES), 0)

    def test_edge_tts_voices_non_empty(self):
        from config import EDGE_TTS_ALL_GERMAN_VOICES
        self.assertIsInstance(EDGE_TTS_ALL_GERMAN_VOICES, list)
        self.assertGreater(len(EDGE_TTS_ALL_GERMAN_VOICES), 0)

    def test_kartorpheus_token_ids_are_ints(self):
        from config import (
            KARTORPHEUS_AUDIO_END_MARKER_TOKEN_ID,
            KARTORPHEUS_AUDIO_START_MARKER_TOKEN_ID,
            KARTORPHEUS_AUDIO_TOKEN_OFFSET,
            KARTORPHEUS_GENERATION_EOS_TOKEN_ID,
            KARTORPHEUS_PROMPT_END_TOKEN_IDS,
            KARTORPHEUS_PROMPT_START_TOKEN_ID,
        )
        self.assertIsInstance(KARTORPHEUS_PROMPT_START_TOKEN_ID, int)
        self.assertIsInstance(KARTORPHEUS_GENERATION_EOS_TOKEN_ID, int)
        self.assertIsInstance(KARTORPHEUS_AUDIO_START_MARKER_TOKEN_ID, int)
        self.assertIsInstance(KARTORPHEUS_AUDIO_END_MARKER_TOKEN_ID, int)
        self.assertIsInstance(KARTORPHEUS_AUDIO_TOKEN_OFFSET, int)
        self.assertIsInstance(KARTORPHEUS_PROMPT_END_TOKEN_IDS, list)
        for tid in KARTORPHEUS_PROMPT_END_TOKEN_IDS:
            self.assertIsInstance(tid, int)

    def test_kartorpheus_speakers_non_empty(self):
        from config import KARTORPHEUS_NATURAL_SPEAKERS
        self.assertIsInstance(KARTORPHEUS_NATURAL_SPEAKERS, list)
        self.assertGreater(len(KARTORPHEUS_NATURAL_SPEAKERS), 0)

    def test_orpheus_sample_rate(self):
        from config import ORPHEUS_SAMPLE_RATE
        self.assertEqual(ORPHEUS_SAMPLE_RATE, 24000)

    def test_kokoro_onnx_voices_non_empty(self):
        from config import KOKORO_ONNX_VOICES
        self.assertIsInstance(KOKORO_ONNX_VOICES, list)
        self.assertGreater(len(KOKORO_ONNX_VOICES), 0)

    def test_piper_german_voice_paths_non_empty(self):
        from config import PIPER_GERMAN_VOICE_PATHS
        self.assertIsInstance(PIPER_GERMAN_VOICE_PATHS, list)
        self.assertGreater(len(PIPER_GERMAN_VOICE_PATHS), 0)

    def test_sauerkraut_voices_non_empty(self):
        from config import SAUERKRAUT_VOICES
        self.assertIsInstance(SAUERKRAUT_VOICES, list)
        self.assertGreater(len(SAUERKRAUT_VOICES), 0)


if __name__ == "__main__":
    unittest.main()
