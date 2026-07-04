"""Tests for cache.py — synthesis result caching."""

import os
import tempfile
import unittest


class TestCacheKey(unittest.TestCase):
    """Test cache key generation."""

    def test_deterministic(self):
        from cache import _cache_key
        k1 = _cache_key("kokoro", "af_heart", "Hello", None)
        k2 = _cache_key("kokoro", "af_heart", "Hello", None)
        self.assertEqual(k1, k2)

    def test_different_text_different_key(self):
        from cache import _cache_key
        k1 = _cache_key("kokoro", "af_heart", "Hello", None)
        k2 = _cache_key("kokoro", "af_heart", "World", None)
        self.assertNotEqual(k1, k2)

    def test_different_model_different_key(self):
        from cache import _cache_key
        k1 = _cache_key("kokoro", "af_heart", "Hello", None)
        k2 = _cache_key("piper", "af_heart", "Hello", None)
        self.assertNotEqual(k1, k2)

    def test_different_voice_different_key(self):
        from cache import _cache_key
        k1 = _cache_key("kokoro", "af_heart", "Hello", None)
        k2 = _cache_key("kokoro", "bf_emma", "Hello", None)
        self.assertNotEqual(k1, k2)

    def test_key_is_hex_string(self):
        from cache import _cache_key
        k = _cache_key("test", None, "text", None)
        self.assertEqual(len(k), 24)
        self.assertTrue(all(c in "0123456789abcdef" for c in k))


class TestCacheDisabled(unittest.TestCase):
    """Test that disabled cache returns None."""

    def test_lookup_returns_none_when_disabled(self):
        from cache import lookup
        result = lookup("kokoro", "af_heart", "Hello", None)
        self.assertIsNone(result)

    def test_store_returns_none_when_disabled(self):
        from cache import store
        result = store("kokoro", "af_heart", "Hello", None, "/nonexistent.wav")
        self.assertIsNone(result)


class TestCacheRoundtrip(unittest.TestCase):
    """Test store + lookup roundtrip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import cache
        cache.configure(cache_dir=self.tmpdir, max_mb=10, enabled=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import cache
        cache._enabled = False

    def test_miss_then_hit(self):
        from cache import lookup, store
        # Miss
        result = lookup("kokoro", "af_heart", "Test", None)
        self.assertIsNone(result)
        # Create a fake WAV file
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.write(fd, b"RIFF" + b"\x00" * 200)
        os.close(fd)
        try:
            # Store
            cached = store("kokoro", "af_heart", "Test", None, tmp)
            self.assertIsNotNone(cached)
            # Hit
            result = lookup("kokoro", "af_heart", "Test", None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isfile(result))
        finally:
            os.unlink(tmp)

    def test_configure_creates_directory(self):
        import cache
        new_dir = os.path.join(self.tmpdir, "sub", "dir")
        cache.configure(cache_dir=new_dir, max_mb=10, enabled=True)
        self.assertTrue(os.path.isdir(new_dir))


class TestCacheEviction(unittest.TestCase):
    """Test LRU eviction."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import cache
        # Very small cache: 1 KB max
        cache.configure(cache_dir=self.tmpdir, max_mb=0, enabled=True)
        cache._max_bytes = 1024  # 1 KB

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import cache
        cache._enabled = False

    def test_evicts_when_full(self):
        from cache import store
        # Create files that exceed 1 KB total
        for i in range(10):
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.write(fd, b"RIFF" + b"\x00" * 200)
            os.close(fd)
            store("model", None, f"text_{i}", None, tmp)
            os.unlink(tmp)
        # After eviction, total size should be <= 1024 * 0.8
        total = sum(
            os.path.getsize(os.path.join(self.tmpdir, f))
            for f in os.listdir(self.tmpdir)
            if os.path.isfile(os.path.join(self.tmpdir, f))
        )
        self.assertLessEqual(total, 1024)


if __name__ == "__main__":
    unittest.main()
