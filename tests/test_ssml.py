"""Tests for ssml.py — SSML-lite preprocessing."""

import unittest


class TestSSMLParsing(unittest.TestCase):

    def test_plain_text_no_tags(self):
        from ssml import parse_ssml
        result = parse_ssml("Hello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello world")
        self.assertEqual(result[0].speed, 1.0)

    def test_has_ssml_detection(self):
        from ssml import has_ssml
        self.assertFalse(has_ssml("Hello world"))
        self.assertTrue(has_ssml('<break time="500ms"/>'))
        self.assertTrue(has_ssml('<prosody rate="fast">text</prosody>'))
        self.assertTrue(has_ssml("<speak>text</speak>"))

    def test_break_tag(self):
        from ssml import parse_ssml
        result = parse_ssml('Hello <break time="500ms"/> world')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Hello")
        self.assertEqual(result[1].text, "world")
        self.assertAlmostEqual(result[1].silence_ms, 500.0)

    def test_break_seconds(self):
        from ssml import parse_ssml
        result = parse_ssml('A <break time="1.5s"/> B')
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[1].silence_ms, 1500.0)

    def test_prosody_rate_named(self):
        from ssml import parse_ssml
        result = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].speed, 1.25)

    def test_prosody_rate_percent(self):
        from ssml import parse_ssml
        result = parse_ssml('<prosody rate="150%">Faster</prosody>')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].speed, 1.5)

    def test_prosody_rate_resets(self):
        from ssml import parse_ssml
        result = parse_ssml('<prosody rate="fast">Fast</prosody> Normal')
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].speed, 1.25)
        self.assertAlmostEqual(result[1].speed, 1.0)

    def test_speak_wrapper_stripped(self):
        from ssml import parse_ssml
        result = parse_ssml("<speak>Hello</speak>")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello")

    def test_phoneme_tag(self):
        from ssml import parse_ssml
        result = parse_ssml('<phoneme ph="hɛˈloʊ">hello</phoneme>')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "hello")
        self.assertIn("hello", result[0].phoneme_overrides)

    def test_empty_input(self):
        from ssml import parse_ssml
        self.assertEqual(parse_ssml(""), [])
        self.assertEqual(parse_ssml("   "), [])

    def test_unknown_tags_stripped(self):
        from ssml import parse_ssml
        result = parse_ssml("<unknown>Keep this text</unknown>")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Keep this text")


class TestSpellOut(unittest.TestCase):

    def test_spell_out(self):
        from ssml import spell_out
        self.assertEqual(spell_out("ABC"), "A. B. C.")

    def test_spell_out_single(self):
        from ssml import spell_out
        self.assertEqual(spell_out("X"), "X.")


class TestNormalizeAudio(unittest.TestCase):

    def test_normalize_peak(self):
        import numpy as np

        from utils import normalize_audio
        pcm = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        result = normalize_audio(pcm, target_db=-3.0)
        # Peak should be close to 10^(-3/20) ≈ 0.708
        self.assertAlmostEqual(float(np.max(np.abs(result))), 0.708, places=2)

    def test_normalize_silence(self):
        import numpy as np

        from utils import normalize_audio
        pcm = np.zeros(100, dtype=np.float32)
        result = normalize_audio(pcm)
        np.testing.assert_array_equal(result, pcm)

    def test_normalize_preserves_shape(self):
        import numpy as np

        from utils import normalize_audio
        pcm = np.random.randn(1000).astype(np.float32) * 0.5
        result = normalize_audio(pcm)
        self.assertEqual(len(result), len(pcm))


if __name__ == "__main__":
    unittest.main()
