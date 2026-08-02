"""Tests for main.py — CLI argument parsing, model dispatch, entrypoint."""

import argparse
import unittest

# These tests drive the real marking and disclosure pipeline, which needs the
# audio codecs. Skip rather than error where they are absent, so a missing
# dependency reads as "not exercised here" instead of a broken suite.
try:
    import soundfile  # noqa: F401
    _HAVE_SOUNDFILE = True
except ImportError:
    _HAVE_SOUNDFILE = False

requires_soundfile = unittest.skipUnless(
    _HAVE_SOUNDFILE, "soundfile not installed — marking pipeline tests cannot run")


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


@requires_soundfile
class TestRunSynthesisMarking(unittest.TestCase):
    """End-to-end: run_synthesis must mark output, and fail closed if it can't.

    Uses a stub handler so the pipeline is exercised without loading a model.
    """

    def setUp(self):
        import shutil
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def _args(self, output_file, **overrides):
        base = dict(
            model_id="edge", german_voice_id=None, model_params=None,
            output_file=output_file, play_direct=False, input_text="Hallo Welt",
            input_file=None, speech_speed=1.0, trim_silence=False, tts_steps=None,
            tts_language=None, pitch_shift=0.0, instruct=None, ref_text=None,
            no_spoken_disclaimer=False, disclosure_lang=None, lexicon=None, normalize=False,
            output_sample_rate=None, stream=False, verify=False,
            verify_backend="parakeet", i_have_rights=False, allow_unmarked=False,
            c2pa_cert=None, c2pa_key=None, batch=False, translate=False,
            accept_marking_responsibility=False, no_watermark=False,
            override_main_model_repo=None, override_model_filename=None,
            override_tokenizer_repo=None, override_vocoder_repo=None,
            override_speaker_embed_repo=None, override_piper_voices_repo=None,
            lm_studio_api_url=None, gguf_model_name_in_api=None,
            ollama_api_url=None, ollama_model_name=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run_with_stub(self, args, stub):
        from unittest.mock import patch

        import main
        from config import GERMAN_TTS_MODELS

        # Register the stub under whichever handler key the model under test
        # actually uses, so tests can exercise cloning models too.
        handler_key = GERMAN_TTS_MODELS.get(args.model_id, {}).get(
            "handler_function_key", args.model_id)
        with patch.object(main, "_load_handlers_if_needed",
                          return_value={"edge": stub, handler_key: stub}), \
                patch.object(main, "_HANDLERS_LOADED", True):
            main.run_synthesis(args)

    def test_output_is_marked(self):
        import os

        import numpy as np
        import soundfile as sf

        from watermark import is_marked, watermark_detect

        out = os.path.join(self.tmpdir, "out.wav")

        def stub(config, text, voice, params, output_file, play_direct):
            t = np.linspace(0, 2, 44100, endpoint=False, dtype=np.float32)
            sf.write(output_file, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 22050)

        self._run_with_stub(self._args(out), stub)
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(is_marked(out), "CLI output must carry AI-provenance metadata")
        data, sr = sf.read(out, dtype="float32")
        self.assertGreater(watermark_detect(data, sample_rate=sr), 0.65)

    def test_unmarkable_output_is_discarded(self):
        """Fail closed: an output that cannot be marked must not survive."""
        import os

        out = os.path.join(self.tmpdir, "out.xyz")

        def stub(config, text, voice, params, output_file, play_direct):
            with open(output_file, "wb") as f:
                f.write(b"not real audio" * 100)

        self._run_with_stub(self._args(out), stub)
        self.assertFalse(os.path.exists(out),
                         "unmarkable output should have been discarded")

    def test_opt_out_without_attestation_is_refused(self):
        """--allow-unmarked alone must not produce output."""
        import os

        out = os.path.join(self.tmpdir, "noattest.wav")
        called = {"n": 0}

        def stub(config, text, voice, params, output_file, play_direct):
            called["n"] += 1

        self._run_with_stub(self._args(out, allow_unmarked=True), stub)
        self.assertEqual(called["n"], 0, "synthesis must be refused before the handler runs")
        self.assertFalse(os.path.exists(out))

    def test_unwatermarkable_format_refused_before_handler(self):
        """Gate runs before generation: the handler is never invoked."""
        import os

        out = os.path.join(self.tmpdir, "out.aiff")
        called = {"n": 0}

        def stub(config, text, voice, params, output_file, play_direct):
            called["n"] += 1

        self._run_with_stub(self._args(out), stub)
        self.assertEqual(called["n"], 0)
        self.assertFalse(os.path.exists(out))

    def test_allow_unmarked_keeps_output(self):
        import os

        out = os.path.join(self.tmpdir, "keep.xyz")

        def stub(config, text, voice, params, output_file, play_direct):
            with open(output_file, "wb") as f:
                f.write(b"not real audio" * 100)

        self._run_with_stub(
            self._args(out, allow_unmarked=True, accept_marking_responsibility=True), stub)
        self.assertTrue(os.path.exists(out))

    def test_handler_never_plays_unmarked_audio(self):
        """With --play-direct the handler must not do the playback itself."""
        import os

        import numpy as np
        import soundfile as sf

        out = os.path.join(self.tmpdir, "play.wav")
        seen = {}

        def stub(config, text, voice, params, output_file, play_direct):
            seen["play_direct"] = play_direct
            t = np.linspace(0, 2, 44100, endpoint=False, dtype=np.float32)
            sf.write(output_file, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 22050)

        from unittest.mock import patch
        with patch("utils.play_audio"):
            self._run_with_stub(self._args(out, play_direct=True), stub)
        self.assertFalse(seen["play_direct"],
                         "playback must happen after marking, not inside the handler")

    def _cloning_stub(self):
        import numpy as np
        import soundfile as sf

        def stub(config, text, voice, params, output_file, play_direct):
            t = np.linspace(0, 2, 44100, endpoint=False, dtype=np.float32)
            sf.write(output_file, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 22050)
        return stub

    def test_cloned_output_discarded_when_disclosure_fails(self):
        """A deepfake with no AI disclosure must not reach the user.

        Regression: the spoken disclosure used to be best-effort — its failure
        was logged and the cloned audio was delivered anyway.
        """
        import os

        import watermark
        out = os.path.join(self.tmpdir, "cloned.wav")
        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (None, "none")
        try:
            self._run_with_stub(
                self._args(out, model_id="f5_tts_german", i_have_rights=True),
                self._cloning_stub())
        finally:
            watermark.generate_spoken_disclaimer = original
        self.assertFalse(os.path.exists(out),
                         "cloned audio was delivered without its AI disclosure")

    def test_cloned_output_kept_when_disclosure_succeeds(self):
        import os

        import numpy as np

        import watermark
        out = os.path.join(self.tmpdir, "cloned_ok.wav")
        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = lambda sr=24000, language=None: (
            np.full(int(sr * 1.0), 0.05, dtype=np.float32), "spoken")
        watermark._disclaimer_cache.clear()
        try:
            self._run_with_stub(
                self._args(out, model_id="f5_tts_german", i_have_rights=True),
                self._cloning_stub())
        finally:
            watermark.generate_spoken_disclaimer = original
            watermark._disclaimer_cache.clear()
        self.assertTrue(os.path.exists(out))

    def test_disclosure_opt_out_needs_attestation(self):
        """--no-spoken-disclaimer alone must not silently drop the disclosure."""
        import os

        out = os.path.join(self.tmpdir, "cloned_optout.wav")
        self._run_with_stub(
            self._args(out, model_id="f5_tts_german", i_have_rights=True,
                       no_spoken_disclaimer=True),
            self._cloning_stub())
        self.assertFalse(os.path.exists(out),
                         "opting out of the disclosure must require an attestation")

    def test_disclosure_lang_flag_overrides_the_model_language(self):
        """--disclosure-lang is what makes a multilingual model disclosable."""
        import os

        import numpy as np

        import watermark

        out = os.path.join(self.tmpdir, "cloned_el.wav")
        seen = {}

        def fake_gen(sample_rate=24000, language=None):
            seen["lang"] = language
            return (np.full(int(sample_rate * 1.5), 0.05, dtype=np.float32), "spoken")

        original = watermark.generate_spoken_disclaimer
        watermark.generate_spoken_disclaimer = fake_gen
        watermark._disclaimer_cache.clear()
        try:
            self._run_with_stub(
                self._args(out, model_id="f5_tts_german", i_have_rights=True,
                           disclosure_lang="el"),
                self._cloning_stub())
        finally:
            watermark.generate_spoken_disclaimer = original
            watermark._disclaimer_cache.clear()
        self.assertEqual(seen.get("lang"), "el",
                         "the model declares 'de'; --disclosure-lang must win")


class TestPlaybackIsMarkedFirst(unittest.TestCase):
    """Audio must never reach a listener before it has been through marking.

    CrispASR playback used to be exempt, on the assumption its binary marks
    internally — the same assumption mark_audio_file() refuses to make.
    """

    def setUp(self):
        import shutil
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def _run(self, model_id, stream=False, output_file=None):
        from unittest.mock import patch

        import numpy as np
        import soundfile as sf

        import main
        import utils
        import watermark
        from config import GERMAN_TTS_MODELS

        order = []

        def stub(config, text, voice, params, out_path, play_direct):
            if play_direct:
                order.append("handler-played")
            if out_path:
                sf.write(out_path, np.full(24000, 0.1, dtype=np.float32), 24000)

        # run_synthesis imports the streaming entry point directly from the
        # handler module, so the dispatch stub above never sees it. Without
        # patching this too the test would silently depend on a real crispasr
        # binary being installed — green on a dev box, red in CI.
        def streaming_stub(config, text, voice, params, out_path, play_direct):
            order.append("streamed")
            stub(config, text, voice, params, out_path, play_direct)

        def fake_mark(filepath, **kwargs):
            order.append("marked")
            return watermark.MarkResult(marked=True, backend="test",
                                        layers=("audio-watermark",))

        def fake_play(*a, **k):
            order.append("played")

        handler_key = GERMAN_TTS_MODELS.get(model_id, {}).get(
            "handler_function_key", model_id)
        args = argparse.Namespace(
            model_id=model_id, german_voice_id="af_heart", model_params=None,
            output_file=output_file, play_direct=True, input_text="Hallo",
            input_file=None, speech_speed=1.0, trim_silence=False, tts_steps=None,
            tts_language=None, pitch_shift=0.0, instruct=None, ref_text=None,
            no_spoken_disclaimer=False, disclosure_lang=None, lexicon=None,
            normalize=False, output_sample_rate=None, stream=stream, verify=False,
            verify_backend="parakeet", i_have_rights=False, allow_unmarked=False,
            c2pa_cert=None, c2pa_key=None, batch=False, translate=False,
            accept_marking_responsibility=False, no_watermark=False,
            override_main_model_repo=None, override_model_filename=None,
            override_tokenizer_repo=None, override_vocoder_repo=None,
            override_speaker_embed_repo=None, override_piper_voices_repo=None,
            lm_studio_api_url=None, gguf_model_name_in_api=None,
            ollama_api_url=None, ollama_model_name=None,
        )
        import handlers.crispasr_handler as crispasr_handler
        with patch.object(main, "_load_handlers_if_needed",
                          return_value={handler_key: stub}), \
                patch.object(main, "_HANDLERS_LOADED", True), \
                patch.object(crispasr_handler, "synthesize_with_crispasr_streaming",
                             streaming_stub), \
                patch.object(watermark, "mark_audio_file", fake_mark), \
                patch.object(utils, "play_audio", fake_play):
            main.run_synthesis(args)
        return order

    def test_crispasr_playback_is_marked_before_it_is_heard(self):
        order = self._run("crispasr_kokoro")
        self.assertIn("marked", order, "CrispASR playback bypassed marking entirely")
        self.assertIn("played", order)
        self.assertLess(order.index("marked"), order.index("played"),
                        f"played before marking: {order}")
        self.assertNotIn("handler-played", order,
                         "the handler must not play; run_synthesis plays after marking")

    def test_non_crispasr_playback_is_marked_before_it_is_heard(self):
        order = self._run("edge")
        self.assertLess(order.index("marked"), order.index("played"),
                        f"played before marking: {order}")

    def test_streaming_still_marks_the_output_file(self):
        """--stream cannot mark before playback, but the file is still gated."""
        import os
        out = os.path.join(self.tmpdir, "streamed.wav")
        order = self._run("crispasr_kokoro", stream=True, output_file=out)
        self.assertIn("streamed", order, "the streaming path was not taken")
        self.assertIn("marked", order, "a streamed --output-file must still be marked")

    def test_streaming_is_the_only_path_that_plays_before_marking(self):
        """The exemption must be narrow: --stream only, and only for playback."""
        import os
        out = os.path.join(self.tmpdir, "streamed2.wav")
        order = self._run("crispasr_kokoro", stream=True, output_file=out)
        self.assertLess(order.index("streamed"), order.index("marked"))
        # run_synthesis must not additionally play a second time after marking.
        self.assertNotIn("played", order)


class TestConsentGateFailsClosed(unittest.TestCase):
    """An unevaluable consent gate must block, not wave synthesis through."""

    def setUp(self):
        import shutil
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def test_missing_watermark_module_blocks_synthesis(self):
        import os
        import sys
        from unittest.mock import patch

        import numpy as np
        import soundfile as sf

        import main

        out = os.path.join(self.tmpdir, "out.wav")
        called = []

        def stub(config, text, voice, params, out_path, play_direct):
            called.append(True)
            sf.write(out_path, np.full(24000, 0.1, dtype=np.float32), 24000)

        args = argparse.Namespace(
            model_id="edge", german_voice_id=None, model_params=None,
            output_file=out, play_direct=False, input_text="Hallo",
            input_file=None, speech_speed=1.0, trim_silence=False, tts_steps=None,
            tts_language=None, pitch_shift=0.0, instruct=None, ref_text=None,
            no_spoken_disclaimer=False, disclosure_lang=None, lexicon=None,
            normalize=False, output_sample_rate=None, stream=False, verify=False,
            verify_backend="parakeet", i_have_rights=False, allow_unmarked=False,
            c2pa_cert=None, c2pa_key=None, batch=False, translate=False,
            accept_marking_responsibility=False, no_watermark=False,
            override_main_model_repo=None, override_model_filename=None,
            override_tokenizer_repo=None, override_vocoder_repo=None,
            override_speaker_embed_repo=None, override_piper_voices_repo=None,
            lm_studio_api_url=None, gguf_model_name_in_api=None,
            ollama_api_url=None, ollama_model_name=None,
        )
        # Setting the entry to None makes `import watermark` raise ImportError.
        env = {k: v for k, v in os.environ.items() if k != "CRISPTTS_ALLOW_UNMARKED"}
        with patch.dict(sys.modules, {"watermark": None}), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(main, "_load_handlers_if_needed",
                             return_value={"edge": stub}), \
                patch.object(main, "_HANDLERS_LOADED", True):
            main.run_synthesis(args)
        self.assertEqual(called, [],
                         "synthesis ran despite the consent gate being unevaluable")
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
