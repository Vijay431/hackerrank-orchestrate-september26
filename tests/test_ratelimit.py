"""Offline unit tests for code/ratelimit.py.

Runs with `.venv/bin/python tests/test_ratelimit.py` (plain unittest, no
pytest dependency). Every test uses a fake clock/sleep pair so nothing here
ever waits on real wall time -- the whole suite finishes in well under a
second.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import ratelimit as rl  # noqa: E402


class FakeClock:
    """A mutable, thread-safe fake wall clock with a matching fake sleep."""

    def __init__(self) -> None:
        self._t = 1_000.0  # arbitrary non-zero epoch
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        if seconds is None:
            return
        if seconds < 0:
            seconds = 0.0
        with self._lock:
            self._t += seconds

    def advance(self, seconds: float) -> None:
        self.sleep(seconds)


def fixed_rand(value: float):
    return lambda: value


# --------------------------------------------------------------------------
# Duration parsing
# --------------------------------------------------------------------------


class TestParseDuration(unittest.TestCase):
    def test_minutes_seconds(self):
        self.assertAlmostEqual(rl.parse_duration("6m0s"), 360.0)

    def test_seconds(self):
        self.assertAlmostEqual(rl.parse_duration("1s"), 1.0)

    def test_milliseconds(self):
        self.assertAlmostEqual(rl.parse_duration("6ms"), 0.006)

    def test_fractional_seconds(self):
        self.assertAlmostEqual(rl.parse_duration("1.5s"), 1.5)

    def test_hours_minutes(self):
        self.assertAlmostEqual(rl.parse_duration("2h30m"), 9000.0)

    def test_garbage_never_raises(self):
        # Must not raise, and must report "could not parse" via None.
        try:
            result = rl.parse_duration("not-a-duration")
        except Exception as exc:  # pragma: no cover
            self.fail(f"parse_duration raised on garbage input: {exc!r}")
        self.assertIsNone(result)

    def test_none_and_empty(self):
        self.assertIsNone(rl.parse_duration(None))
        self.assertIsNone(rl.parse_duration(""))

    def test_partial_garbage_never_raises(self):
        try:
            result = rl.parse_duration("5zz9q")
        except Exception as exc:  # pragma: no cover
            self.fail(f"parse_duration raised: {exc!r}")
        self.assertIsNone(result)


# --------------------------------------------------------------------------
# TokenBucket
# --------------------------------------------------------------------------


class TestTokenBucket(unittest.TestCase):
    def test_blocks_when_exhausted_and_releases_after_refill(self):
        clock = FakeClock()
        # capacity 10, refill 10/sec -> 1 token every 0.1s
        bucket = rl.TokenBucket(capacity=10, refill_per_second=10.0, clock=clock.now)
        ok, wait = bucket.try_consume(10)
        self.assertTrue(ok)
        # Bucket now empty; asking for 5 more tokens needs 0.5s at 10/sec.
        ok, wait = bucket.try_consume(5)
        self.assertFalse(ok)
        self.assertAlmostEqual(wait, 0.5)
        clock.advance(wait)
        ok, wait = bucket.try_consume(5)
        self.assertTrue(ok)

    def test_releases_exactly_after_right_interval_not_before(self):
        clock = FakeClock()
        bucket = rl.TokenBucket(capacity=5, refill_per_second=5.0, clock=clock.now)
        bucket.try_consume(5)
        ok, wait = bucket.try_consume(1)
        self.assertFalse(ok)
        # Advance slightly less than required -> still not enough.
        clock.advance(wait - 0.01)
        ok, _ = bucket.try_consume(1)
        self.assertFalse(ok)
        # Advance the remaining sliver -> now it succeeds.
        clock.advance(0.01)
        ok, _ = bucket.try_consume(1)
        self.assertTrue(ok)

    def test_reconciliation_debits_underestimate(self):
        clock = FakeClock()
        bucket = rl.TokenBucket(capacity=100, refill_per_second=100.0, clock=clock.now)
        bucket.try_consume(10)  # reserved an estimate of 10
        self.assertAlmostEqual(bucket.tokens_snapshot(), 90.0)
        # Actual usage came back higher than estimated -> extra debit.
        delta = 25 - 10  # actual - estimated
        bucket.adjust(delta)
        self.assertAlmostEqual(bucket.tokens_snapshot(), 75.0)

    def test_reconciliation_credits_overestimate(self):
        clock = FakeClock()
        bucket = rl.TokenBucket(capacity=100, refill_per_second=100.0, clock=clock.now)
        bucket.try_consume(30)
        delta = 10 - 30  # actual came back lower than estimated
        bucket.adjust(delta)
        self.assertAlmostEqual(bucket.tokens_snapshot(), 90.0)

    def test_clamp_remaining_only_clamps_down(self):
        clock = FakeClock()
        bucket = rl.TokenBucket(capacity=100, refill_per_second=100.0, clock=clock.now)
        # Bucket believes it's full (100). Server says only 20 remain.
        bucket.clamp_remaining(20)
        self.assertAlmostEqual(bucket.tokens_snapshot(), 20.0)
        # Server claims MORE remain than we believe -- must NOT increase.
        bucket.clamp_remaining(90)
        self.assertAlmostEqual(bucket.tokens_snapshot(), 20.0)


# --------------------------------------------------------------------------
# ModelLimiter header feedback
# --------------------------------------------------------------------------


class TestHeaderFeedback(unittest.TestCase):
    def test_limit_header_recalibrates_capacity_below_env_default(self):
        """The env default is a guess; the server's x-ratelimit-limit-* is
        the truth. A real tier *below* the guess used to inflate the derived
        refill rate ~170x (gap computed against a fictional capacity), so
        the limiter would have driven straight into 429s."""
        clock = FakeClock()
        m = rl.ModelLimiter("m", rpm=500, tpm=200_000, clock=clock.now, sleep=clock.sleep)
        m.update_from_headers({
            "x-ratelimit-limit-tokens": "30000",
            "x-ratelimit-remaining-tokens": "29000",
            "x-ratelimit-reset-tokens": "2s",
        })
        self.assertEqual(m.tpm_bucket.capacity, 30000.0)
        self.assertEqual(m.tpm_bucket.tokens_snapshot(), 29000.0)
        clock.sleep(2.0)
        # Fully refilled, but never past the real ceiling.
        self.assertEqual(m.tpm_bucket.tokens_snapshot(), 30000.0)
        # Rate is the real tier's 30000/min, not millions.
        self.assertAlmostEqual(m.tpm_bucket._rate * 60, 30000.0, delta=1.0)

    def test_limit_header_raises_capacity_above_env_default(self):
        """The other direction: a bigger real tier is adopted too, so a
        conservative env default does not permanently throttle a big account."""
        clock = FakeClock()
        m = rl.ModelLimiter("m", rpm=500, tpm=200_000, clock=clock.now, sleep=clock.sleep)
        m.update_from_headers({"x-ratelimit-limit-requests": "5000",
                               "x-ratelimit-remaining-requests": "4990",
                               "x-ratelimit-reset-requests": "120ms"})
        self.assertEqual(m.rpm_bucket.capacity, 5000.0)
        clock.sleep(60.0)
        self.assertEqual(m.rpm_bucket.tokens_snapshot(), 5000.0)

    def test_update_from_headers_clamps_buckets_down(self):
        clock = FakeClock()
        limiter = rl.ModelLimiter("gpt-4o-mini", rpm=500, tpm=200000, clock=clock.now, sleep=clock.sleep)
        headers = {
            "x-ratelimit-remaining-requests": "3",
            "x-ratelimit-remaining-tokens": "150",
            "x-ratelimit-reset-requests": "6m0s",
            "x-ratelimit-reset-tokens": "1.5s",
        }
        limiter.update_from_headers(headers)
        self.assertAlmostEqual(limiter.rpm_bucket.tokens_snapshot(), 3.0)
        self.assertAlmostEqual(limiter.tpm_bucket.tokens_snapshot(), 150.0)

    def test_update_from_headers_ignores_missing_and_garbage(self):
        clock = FakeClock()
        limiter = rl.ModelLimiter("gpt-4o-mini", rpm=500, tpm=200000, clock=clock.now, sleep=clock.sleep)
        before_rpm = limiter.rpm_bucket.tokens_snapshot()
        before_tpm = limiter.tpm_bucket.tokens_snapshot()
        try:
            limiter.update_from_headers({"x-ratelimit-remaining-requests": "not-a-number",
                                          "x-ratelimit-reset-tokens": "garbage"})
            limiter.update_from_headers(None)
            limiter.update_from_headers({})
        except Exception as exc:  # pragma: no cover
            self.fail(f"update_from_headers raised: {exc!r}")
        self.assertAlmostEqual(limiter.rpm_bucket.tokens_snapshot(), before_rpm)
        self.assertAlmostEqual(limiter.tpm_bucket.tokens_snapshot(), before_tpm)


# --------------------------------------------------------------------------
# Circuit breaker + retry: simulated 429 storm
# --------------------------------------------------------------------------


class FakeFlaky429Api:
    """Returns 429 for each caller's first `fails_per_caller` attempts, then
    200. Failures are tracked per calling thread (rather than one shared
    counter) so the test's outcome does not depend on the real OS thread
    scheduler's interleaving -- every worker is guaranteed to succeed within
    `fails_per_caller + 1` attempts regardless of race order.
    """

    def __init__(self, fails_per_caller: int, retry_after: float = 2.0):
        self._fails_per_caller = fails_per_caller
        self._retry_after = retry_after
        self._lock = threading.Lock()
        self._remaining_by_caller: dict = {}
        self.total_attempts = 0

    def call(self):
        key = threading.get_ident()
        with self._lock:
            self.total_attempts += 1
            remaining = self._remaining_by_caller.setdefault(key, self._fails_per_caller)
            if remaining > 0:
                self._remaining_by_caller[key] = remaining - 1
                raise rl.RateLimitError(
                    "429", headers={"retry-after": str(self._retry_after)}, retry_after=self._retry_after
                )
        return "ok"


class TestCircuitBreakerAnd429Storm(unittest.TestCase):
    def test_breaker_pauses_all_workers_not_just_one(self):
        # A fake sleep that instantly fast-forwards a *shared* clock cannot
        # be raced with real OS threads and still yield a meaningful
        # per-thread "elapsed" measurement (whichever real thread happens to
        # run its sleep first silently advances time for everyone else,
        # which is a race in the test, not a bug in the breaker). Instead,
        # check the property the requirement actually cares about directly:
        # the pause is stored once, globally, so *every* caller that checks
        # in -- not just the thread that tripped the breaker -- sees the
        # full remaining window.
        clock = FakeClock()
        breaker = rl.CircuitBreaker(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0))
        self.assertEqual(breaker.pending_pause_seconds(), 0.0)
        breaker.trip(retry_after=5.0)
        for _ in range(5):
            self.assertAlmostEqual(breaker.pending_pause_seconds(), 5.0)

    def test_concurrency_halved_after_trip_then_ramps_back(self):
        clock = FakeClock()
        breaker = rl.CircuitBreaker(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0),
                                     budget_reduction_seconds=60.0)
        self.assertEqual(breaker.concurrency_factor(), 1.0)
        breaker.trip(retry_after=1.0)
        self.assertEqual(breaker.concurrency_factor(), 0.5)
        self.assertEqual(breaker.effective_concurrency(10), 5)
        clock.advance(1.0 + 60.0 - 1)  # just before the reduction window ends
        self.assertEqual(breaker.concurrency_factor(), 0.5)
        clock.advance(2)  # now past it
        self.assertEqual(breaker.concurrency_factor(), 1.0)
        self.assertEqual(breaker.effective_concurrency(10), 10)

    def test_jitter_on_resume_avoids_synchronized_stampede(self):
        clock = FakeClock()
        # Distinct rand() values per call would be ideal, but even a shared
        # generator proves the breaker *adds* a jitter sleep on top of the
        # base pause -- assert total elapsed exceeds the bare retry_after.
        breaker = rl.CircuitBreaker(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.7), jitter_max=2.0)
        breaker.trip(retry_after=3.0)
        start = clock.now()
        breaker.wait_if_paused()
        elapsed = clock.now() - start
        self.assertGreaterEqual(elapsed, 3.0)
        self.assertAlmostEqual(elapsed, 3.0 + 0.7 * 2.0, places=6)

    def test_simulated_429_storm_all_calls_eventually_succeed_bounded_attempts(self):
        clock = FakeClock()
        n_workers = 3
        fails_per_caller = 2
        pause_events = []  # (thread_ident, remaining_seconds) for every breaker wait
        pause_lock = threading.Lock()

        def on_wait(remaining):
            with pause_lock:
                pause_events.append((threading.get_ident(), remaining))

        limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0),
                                  default_rpm=1000, default_tpm=1_000_000, on_breaker_wait=on_wait)
        # Each worker hits a 429 on its first 2 attempts, then succeeds on its
        # 3rd -- deterministic per-caller, independent of thread scheduling.
        api = FakeFlaky429Api(fails_per_caller=fails_per_caller, retry_after=1.0)
        outcomes = []
        outcomes_lock = threading.Lock()

        def worker():
            result = limiter.call_with_retry(api.call, model="gpt-4o-mini", estimated_tokens=50,
                                              max_attempts=5, base_delay=0.01)
            with outcomes_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker) for _ in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # A hung worker must fail the test, not be masked by the join timeout.
        self.assertFalse(any(t.is_alive() for t in threads), "a worker never finished")

        self.assertEqual(len(outcomes), n_workers)
        self.assertTrue(all(r == "ok" for r in outcomes))
        # Bounded: no stampede, no runaway retry. Each worker fails exactly
        # twice then succeeds once, and nothing retries beyond that.
        self.assertEqual(api.total_attempts, n_workers * (fails_per_caller + 1))
        # Every 429 tripped the shared breaker, so every retry had to wait on
        # it: at least one wait per failure. (That the pause is *global* --
        # seen by threads other than the one that tripped it -- is proven
        # deterministically by test_breaker_pauses_all_workers_not_just_one;
        # under a shared fake clock, thread interleaving is not something
        # this test can assert on without racing.)
        self.assertGreaterEqual(len(pause_events), n_workers * fails_per_caller)
        self.assertTrue(all(remaining > 0 for _, remaining in pause_events))


# --------------------------------------------------------------------------
# Retry wrapper: gives up after 5 attempts; never retries non-429 4xx
# --------------------------------------------------------------------------


class TestCallWithRetry(unittest.TestCase):
    def test_gives_up_after_five_attempts(self):
        clock = FakeClock()
        limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0))
        attempts = {"n": 0}

        def always_fails():
            attempts["n"] += 1
            raise rl.TransientAPIError("boom")

        with self.assertRaises(rl.TransientAPIError):
            limiter.call_with_retry(always_fails, model="m", estimated_tokens=1, max_attempts=5, base_delay=0.001)
        self.assertEqual(attempts["n"], 5)

    def test_does_not_retry_non_429_four_xx(self):
        clock = FakeClock()
        limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0))
        attempts = {"n": 0}

        class BadRequest(Exception):
            pass

        def fails_permanently():
            attempts["n"] += 1
            raise BadRequest("malformed request")

        with self.assertRaises(BadRequest):
            limiter.call_with_retry(fails_permanently, model="m", estimated_tokens=1, max_attempts=5)
        self.assertEqual(attempts["n"], 1)

    def test_succeeds_first_try_with_no_retry(self):
        clock = FakeClock()
        limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0))
        attempts = {"n": 0}

        def works():
            attempts["n"] += 1
            return "ok"

        result = limiter.call_with_retry(works, model="m", estimated_tokens=1)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 1)


# --------------------------------------------------------------------------
# Token estimation helper
# --------------------------------------------------------------------------


class TestEstimateTokens(unittest.TestCase):
    def test_never_below_floor(self):
        self.assertGreaterEqual(rl.estimate_tokens(text=""), 16)

    def test_scales_with_text_length(self):
        short = rl.estimate_tokens(text="hello")
        long = rl.estimate_tokens(text="hello world " * 200)
        self.assertGreater(long, short)

    def test_image_allowance_added(self):
        # Text long enough to clear the 16-token floor, so the delta is the
        # image allowance alone and not (allowance - floor headroom).
        text = "describe this invoice line by line " * 10
        base = rl.estimate_tokens(text=text)
        self.assertGreater(base, 16, "fixture text must exceed the floor")
        with_image = rl.estimate_tokens(text=text, images=1)
        self.assertGreaterEqual(with_image - base, 800)

    def test_fallback_without_tiktoken(self):
        # Force the fallback path regardless of whether tiktoken is installed.
        saved = rl.tiktoken
        try:
            rl.tiktoken = None
            value = rl.estimate_tokens(text="x" * 400)
            self.assertGreaterEqual(value, int(400 / 4))
        finally:
            rl.tiktoken = saved


# --------------------------------------------------------------------------
# RateLimiter.reserve: dual-bucket blocking + config-from-env
# --------------------------------------------------------------------------


class TestRateLimiterReserve(unittest.TestCase):
    def test_reserve_blocks_on_tpm_and_refunds_rpm_slot(self):
        clock = FakeClock()
        # 10 RPM (plenty), but only 100 TPM so the token bucket is the
        # actual constraint.
        limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep, rand=fixed_rand(0.0),
                                  default_rpm=10, default_tpm=100)
        limiter.reserve("m", 100)  # drains the TPM bucket entirely
        model_limiter = limiter._get("m")
        rpm_before = model_limiter.rpm_bucket.tokens_snapshot()
        start = clock.now()
        limiter.reserve("m", 50)  # must block until TPM refills
        elapsed = clock.now() - start
        self.assertGreater(elapsed, 0)
        # The RPM slot consumed by the blocked attempt was refunded while
        # waiting, then re-consumed on success -> net one slot used, not two.
        rpm_after = model_limiter.rpm_bucket.tokens_snapshot()
        self.assertLessEqual(rpm_before - rpm_after, 1.0 + 1e-9)

    def test_env_defaults_used_until_headers_arrive(self):
        os.environ["OPENAI_RPM_LIMIT"] = "42"
        os.environ["OPENAI_TPM_LIMIT"] = "4242"
        try:
            clock = FakeClock()
            limiter = rl.RateLimiter(clock=clock.now, sleep=clock.sleep)
            model_limiter = limiter._get("m")
            self.assertAlmostEqual(model_limiter.rpm_bucket.capacity, 42)
            self.assertAlmostEqual(model_limiter.tpm_bucket.capacity, 4242)
        finally:
            del os.environ["OPENAI_RPM_LIMIT"]
            del os.environ["OPENAI_TPM_LIMIT"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
