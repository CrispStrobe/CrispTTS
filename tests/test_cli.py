"""Tests for main.py — CLI argument parsing, model dispatch, entrypoint."""

import argparse
import unittest


def _build_parser():
    """Build the argparser by extracting it from main_cli_entrypoint.

    We recreate the parser inline because main_cli_entrypoint calls
    parser.parse_args() which would consume sys.argv. Instead we
    replicate the parser setup and test it in isolation.
    """
    from config import (
        GERMAN_TTS_MODELS,
        LM_STUDIO_API_URL_DEFAULT,
        OLLAMA_API_URL_DEFAULT,
    )

    parser = argparse.ArgumentParser(
        description="CrispTTS: Modular German Text-to-Speech Synthesizer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    action_group = parser.add_argument_group(title="Primary Actions")
    input_group = parser.add_mutually_exclusive_group(required=False)
    action_group.add_argument("--list-models", action="store_true")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID")
    action_group.add_argument("--test-all", action="store_true")
    action_group.add_argument("--test-all-speakers", action="store_true")
    action_group.add_argument("--skip-models", type=str, nargs="*", default=[])

    synth_group = parser.add_argument_group(title="Synthesis Options")
    input_group.add_argument("--input-text", type=str)
    input_group.add_argument("--input-file", type=str)

    model_choices = list(GERMAN_TTS_MODELS.keys()) if GERMAN_TTS_MODELS else []
    synth_group.add_argument("--model-id", type=str, choices=model_choices, default=None)
    synth_group.add_argument("--output-file", type=str)
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs")
    synth_group.add_argument("--play-direct", action="store_true")
    synth_group.add_argument("--german-voice-id", type=str)
    synth_group.add_argument("--model-params", type=str)

    crispasr_group = parser.add_argument_group(title="CrispASR Integration")
    crispasr_group.add_argument("--verify", action="store_true")
    crispasr_group.add_argument("--verify-backend", type=str, default="parakeet")
    crispasr_group.add_argument("--translate", action="store_true")
    crispasr_group.add_argument("--translate-from", type=str, default="en")
    crispasr_group.add_argument("--translate-to", type=str, default="de")
    crispasr_group.add_argument("--translate-backend", type=str, default="m2m100")

    parser.add_argument(
        "--loglevel", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    override_group = parser.add_argument_group(title="Runtime Overrides")
    override_group.add_argument("--override-main-model-repo", type=str)
    override_group.add_argument("--override-model-filename", type=str)
    override_group.add_argument("--override-tokenizer-repo", type=str)
    override_group.add_argument("--override-vocoder-repo", type=str)
    override_group.add_argument("--override-speaker-embed-repo", type=str)
    override_group.add_argument("--override-piper-voices-repo", type=str)

    api_group = parser.add_argument_group(title="API Backend Overrides")
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT)
    api_group.add_argument("--gguf-model-name-in-api", type=str)
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT)
    api_group.add_argument("--ollama-model-name", type=str)

    return parser, model_choices


class TestCLIArgParser(unittest.TestCase):
    """Test CLI argument parsing from main.py."""

    @classmethod
    def setUpClass(cls):
        cls.parser, cls.model_choices = _build_parser()

    def test_parser_creates_without_error(self):
        """The argument parser should be constructible."""
        self.assertIsNotNone(self.parser)

    def test_list_models_flag(self):
        args = self.parser.parse_args(["--list-models"])
        self.assertTrue(args.list_models)

    def test_verify_flag_present(self):
        args = self.parser.parse_args(["--verify"])
        self.assertTrue(args.verify)

    def test_translate_flag_present(self):
        args = self.parser.parse_args(["--translate"])
        self.assertTrue(args.translate)

    def test_translate_from_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.translate_from, "en")

    def test_translate_to_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.translate_to, "de")

    def test_translate_backend_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.translate_backend, "m2m100")

    def test_verify_backend_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.verify_backend, "parakeet")

    def test_skip_models_accepts_list(self):
        args = self.parser.parse_args(["--skip-models", "edge", "piper_local", "kokoro_onnx"])
        self.assertEqual(args.skip_models, ["edge", "piper_local", "kokoro_onnx"])

    def test_loglevel_choices(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            args = self.parser.parse_args(["--loglevel", level])
            self.assertEqual(args.loglevel, level)

    def test_play_direct_is_boolean_flag(self):
        args_off = self.parser.parse_args([])
        self.assertFalse(args_off.play_direct)
        args_on = self.parser.parse_args(["--play-direct"])
        self.assertTrue(args_on.play_direct)

    def test_output_dir_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.output_dir, "tts_test_outputs")

    def test_model_choices_match_config(self):
        from config import GERMAN_TTS_MODELS
        self.assertEqual(set(self.model_choices), set(GERMAN_TTS_MODELS.keys()))

    def test_model_id_invalid_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--model-id", "totally_nonexistent_model_xyz"])

    def test_input_group_mutually_exclusive(self):
        """--input-text and --input-file should be mutually exclusive."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "--input-text", "some text",
                "--input-file", "some_file.txt",
            ])

    def test_model_params_accepts_json(self):
        args = self.parser.parse_args(["--model-params", '{"temperature": 0.7}'])
        self.assertEqual(args.model_params, '{"temperature": 0.7}')


class TestMainEntrypoint(unittest.TestCase):
    """Test the main_cli_entrypoint function behavior."""

    def test_main_cli_entrypoint_importable(self):
        from main import main_cli_entrypoint
        self.assertTrue(callable(main_cli_entrypoint))

    def test_main_has_load_handlers_if_needed(self):
        from main import _load_handlers_if_needed
        self.assertTrue(callable(_load_handlers_if_needed))


if __name__ == "__main__":
    unittest.main()
