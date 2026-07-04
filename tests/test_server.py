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


if __name__ == "__main__":
    unittest.main()
