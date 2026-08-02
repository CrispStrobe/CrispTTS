"""Tests for server.py — rate limiting, health endpoint."""

import unittest


class TestRateLimiting(unittest.TestCase):
    """Test the token-bucket rate limiter."""

    def setUp(self):
        from server import _rate_limit_buckets
        _rate_limit_buckets.clear()

    def test_first_request_allowed(self):
        from server import _check_rate_limit
        self.assertTrue(_check_rate_limit("127.0.0.1"))

    def test_within_limit_allowed(self):
        import server
        server._rate_limit_max = 5
        from server import _check_rate_limit
        for _ in range(5):
            self.assertTrue(_check_rate_limit("127.0.0.1"))

    def test_exceeds_limit_blocked(self):
        import server
        server._rate_limit_max = 3
        from server import _check_rate_limit
        for _ in range(3):
            _check_rate_limit("10.0.0.1")
        self.assertFalse(_check_rate_limit("10.0.0.1"))

    def test_different_ips_independent(self):
        import server
        server._rate_limit_max = 2
        from server import _check_rate_limit
        _check_rate_limit("1.1.1.1")
        _check_rate_limit("1.1.1.1")
        # 1.1.1.1 is at limit
        self.assertFalse(_check_rate_limit("1.1.1.1"))
        # 2.2.2.2 is fresh
        self.assertTrue(_check_rate_limit("2.2.2.2"))

    def test_expired_entries_evicted(self):
        import time

        import server
        server._rate_limit_max = 2
        server._rate_limit_window = 0.1  # 100ms window
        from server import _check_rate_limit
        _check_rate_limit("3.3.3.3")
        _check_rate_limit("3.3.3.3")
        self.assertFalse(_check_rate_limit("3.3.3.3"))
        time.sleep(0.15)  # Wait for window to expire
        self.assertTrue(_check_rate_limit("3.3.3.3"))

    def tearDown(self):
        import server
        server._rate_limit_max = 10
        server._rate_limit_window = 60.0


class TestServerSSML(unittest.TestCase):
    """Test SSML handling in server API request parsing."""

    def test_ssml_stripped_from_input(self):
        """SSML tags should be parsed and stripped from input text."""
        from ssml import has_ssml, parse_ssml
        text = '<prosody rate="fast">Hello world</prosody>'
        self.assertTrue(has_ssml(text))
        segments = parse_ssml(text)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Hello world")
        self.assertAlmostEqual(segments[0].speed, 1.25)

    def test_plain_text_unchanged(self):
        """Plain text without SSML should pass through unchanged."""
        from ssml import has_ssml
        self.assertFalse(has_ssml("Just plain text"))


class TestCacheVersioning(unittest.TestCase):
    """Test that cache key includes version to prevent stale results."""

    def test_cache_key_includes_version(self):
        """Different versions should produce different cache keys."""
        import cache
        key1 = cache._cache_key("kokoro", "af_heart", "Hello", None)
        # Temporarily change version
        old_ver = cache._VERSION
        cache._VERSION = "99.99.99"
        try:
            key2 = cache._cache_key("kokoro", "af_heart", "Hello", None)
        finally:
            cache._VERSION = old_ver
        self.assertNotEqual(key1, key2)

    def test_cache_version_is_string(self):
        import cache
        self.assertIsInstance(cache._VERSION, str)
        self.assertGreater(len(cache._VERSION), 0)


class TestMarkingCacheKey(unittest.TestCase):
    """Unmarked audio must not leak to marking-enabled callers via the cache."""

    def test_marking_mode_changes_key(self):
        import os

        import cache
        key_marked = cache._cache_key("kokoro", "af_heart", "Hello", None)
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
        try:
            key_unmarked = cache._cache_key("kokoro", "af_heart", "Hello", None)
        finally:
            del os.environ["CRISPTTS_NO_WATERMARK"]
        self.assertNotEqual(key_marked, key_unmarked)

    def test_marking_mode_reports_disabled(self):
        import os

        import cache
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
        try:
            self.assertEqual(cache._marking_mode(), "unmarked")
        finally:
            del os.environ["CRISPTTS_NO_WATERMARK"]
        self.assertTrue(cache._marking_mode().startswith("marked:"))


class TestProvenanceHeaders(unittest.TestCase):
    """X-CrispTTS-Watermarked must reflect reality, never assert blindly."""

    def _collect(self, mark_result):
        from server import TTSRequestHandler
        sent = {}

        class _Stub:
            send_header = lambda self, k, v: sent.__setitem__(k, v)  # noqa: E731

        TTSRequestHandler._send_marking_headers(_Stub(), mark_result)
        return sent

    def test_none_result_reports_false(self):
        self.assertEqual(self._collect(None)["X-CrispTTS-Watermarked"], "false")

    def test_unmarked_result_reports_false(self):
        from watermark import MarkResult
        headers = self._collect(MarkResult(marked=False, reason="disabled"))
        self.assertEqual(headers["X-CrispTTS-Watermarked"], "false")
        self.assertNotIn("X-CrispTTS-Watermark-Backend", headers)

    def test_marked_result_reports_details(self):
        from watermark import MarkResult
        headers = self._collect(MarkResult(
            marked=True, backend="wavmark", layers=("audio-watermark", "metadata"),
            confidence=0.91))
        self.assertEqual(headers["X-CrispTTS-Watermarked"], "true")
        self.assertEqual(headers["X-CrispTTS-Watermark-Backend"], "wavmark")
        self.assertEqual(headers["X-CrispTTS-Watermark-Confidence"], "0.910")
        self.assertIn("audio-watermark", headers["X-CrispTTS-Provenance-Layers"])

    def test_cached_state_follows_marking_mode(self):
        import os

        from server import _cached_mark_state
        self.assertTrue(_cached_mark_state().marked)
        os.environ["CRISPTTS_NO_WATERMARK"] = "1"
        try:
            self.assertFalse(_cached_mark_state().marked)
        finally:
            del os.environ["CRISPTTS_NO_WATERMARK"]


class TestConsentBeforeCache(unittest.TestCase):
    """The consent gate must precede the cache lookup in do_POST."""

    def test_gate_runs_before_lookup(self):
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        self.assertLess(
            src.index("requires_consent"), src.index("_cache.lookup"),
            "cache lookup must not short-circuit the voice-cloning consent gate")

    def test_gate_fails_closed_when_watermark_is_unavailable(self):
        """An unevaluable gate must 500, not fall through to synthesis."""
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        gate = src.index("requires_consent")
        after = src[gate:src.index("_cache.lookup")]
        handler = after.index("except ImportError")
        tail = after[handler:]
        self.assertIn("_send_error", tail,
                      "a missing watermark module must refuse, not pass silently")
        self.assertNotIn("\n            pass", tail)


class TestDisclosureLanguageIsPartOfTheCacheKey(unittest.TestCase):
    """Cloned audio carries its disclosure inside it.

    A clip disclosed in German is not a valid response to a request that asked
    for Greek, so the language has to participate in the cache key.
    """

    def test_disclosure_lang_reaches_the_cache_params(self):
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        self.assertIn("disclosure_lang", src)
        params = src.index("_cache_params")
        lookup = src.index("_cache.lookup")
        self.assertLess(params, lookup)
        self.assertIn('_cache_params["disclosure_lang"]', src)

    def test_lookup_and_store_use_the_same_params(self):
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        self.assertEqual(src.count("_cache_params_json"), 3,
                         "build once, then use for both lookup and store")

    def test_differing_disclosure_langs_produce_different_keys(self):
        import json

        import cache
        base = ("m", "v", "text")
        de = cache._cache_key(*base, json.dumps({"disclosure_lang": "de"}))
        el = cache._cache_key(*base, json.dumps({"disclosure_lang": "el"}))
        self.assertNotEqual(de, el)


class TestServerFollowsTheWrittenFile(unittest.TestCase):
    """Most handlers force their own container regardless of the path given.

    Reading back only the mkstemp path turned that into "synthesis produced no
    output" for every such backend — the Edge handler included.
    """

    def test_written_output_is_resolved_before_the_empty_check(self):
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        self.assertLess(
            src.index("resolve_written_output"), src.index("Synthesis produced no output"),
            "the handler's real output must be found before declaring failure")

    def test_response_is_converted_rather_than_relabelled(self):
        """The response has a declared Content-Type, so following blindly is wrong."""
        import inspect

        import server
        src = inspect.getsource(server.TTSRequestHandler.do_POST)
        resolved = src.index("resolve_written_output")
        tail = src[resolved:src.index("Synthesis produced no output")]
        self.assertIn("save_audio", tail,
                      "MP3 bytes labelled audio/wav would be worse than the error "
                      "this replaces — convert to the requested format")


if __name__ == "__main__":
    unittest.main()
