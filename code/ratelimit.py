"""Thread-safe, proactive rate limiter for the OpenAI API.

Owned by agent B1. This module is intentionally dependency-free (standard
library plus an *optional* `tiktoken` import) and never imports `openai`, so
it can be unit-tested completely offline with a fake clock -- no real
sleeping, no network.

Design goal: stay *under* OpenAI's per-model RPM/TPM limits rather than
bounce off them with 429s. Two independent token buckets per model (one for
requests/minute, one for tokens/minute) gate every call before it is issued;
after the call returns, the real `usage` count reconciles the TPM bucket so
a biased estimator cannot drift the limiter into over-issuing; and live
`x-ratelimit-*` response headers clamp the buckets down when the server
reports less headroom than we believed.

A single global `CircuitBreaker` handles 429s: tripping it pauses *every*
caller (not just the one that got the 429) for the `Retry-After` window,
with jitter on resume so waiters do not all stampede back in at once, and it
halves the advertised "effective concurrency" for about 60 seconds
afterwards.

Interoperability contract for callers (e.g. the ThreadPoolExecutor owner and
the module that actually issues OpenAI calls): `call_with_retry(fn, ...)`
expects `fn` to raise `RateLimitError` on an HTTP 429 (with the response
headers and/or a parsed `retry_after` seconds attached), `TransientAPIError`
on a 5xx or connection failure, and to let any other exception (e.g. a
non-429 4xx) propagate unchanged so it is never retried. On success `fn` may
return an object with a `.headers` mapping; if present it is fed back into
the limiter automatically.

Every time reference in this module goes through the `clock` /
`sleep` callables injected into each class's constructor (defaulting to
`time.monotonic` / `time.sleep`). Nothing in the module body calls
`time.time()` or `time.sleep()` directly -- that is what makes the whole
thing testable with a fake clock in well under a second of wall time.
"""

from __future__ import annotations

import os
import random
import re
import threading
from typing import Callable, Mapping, Optional

try:  # optional, more accurate token counting when available
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - exercised implicitly by fallback tests
    tiktoken = None  # type: ignore


# --------------------------------------------------------------------------
# Duration parsing (OpenAI's `x-ratelimit-reset-*` headers)
# --------------------------------------------------------------------------

# Order matters: "ms" must be tried before "m" or "6ms" would be misread as
# "6m" + a dangling "s".
_DURATION_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s|us|ns)")
_DURATION_UNIT_SECONDS = {
    "h": 3600.0,
    "m": 60.0,
    "s": 1.0,
    "ms": 0.001,
    "us": 1e-6,
    "ns": 1e-9,
}


def parse_duration(value: object) -> Optional[float]:
    """Parse a Go-style duration string ("6m0s", "1.5s", "2h30m", "6ms").

    Returns seconds as a float, or `None` if `value` is missing, empty, or
    not a well-formed duration. Never raises.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        total = 0.0
        pos = 0
        matched_any = False
        for m in _DURATION_TOKEN_RE.finditer(text):
            if m.start() != pos:
                return None  # gap or leading garbage -> not a clean duration
            total += float(m.group(1)) * _DURATION_UNIT_SECONDS[m.group(2)]
            pos = m.end()
            matched_any = True
        if not matched_any or pos != len(text):
            return None
        return total
    except Exception:
        return None


def _get_header(headers: Optional[Mapping], name: str) -> Optional[str]:
    """Case-tolerant header lookup that never raises."""
    if headers is None:
        return None
    try:
        if name in headers:  # exact key, cheap common case
            return headers[name]
    except Exception:
        pass
    try:
        for key, val in dict(headers).items():
            if str(key).lower() == name.lower():
                return val
    except Exception:
        pass
    return None


def _parse_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Exceptions callers use to signal what happened, per the module contract.
# --------------------------------------------------------------------------


class RateLimitError(Exception):
    """Raise from `fn` passed to `call_with_retry` on an HTTP 429."""

    def __init__(
        self,
        message: str = "rate limited",
        *,
        headers: Optional[Mapping] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.headers = headers
        self.retry_after = retry_after


class TransientAPIError(Exception):
    """Raise from `fn` on a 5xx or connection-level failure (retryable)."""


class PermanentAPIError(Exception):
    """A non-429 4xx. Provided for callers' convenience; never retried.

    `call_with_retry` does not special-case this class -- any exception
    other than `RateLimitError` / `TransientAPIError` already propagates
    immediately without retry, which is exactly the behavior a malformed
    request needs (it will fail identically forever).
    """


# --------------------------------------------------------------------------
# Token bucket
# --------------------------------------------------------------------------


class TokenBucket:
    """A single leaky/token bucket, refilled continuously from wall time.

    `try_consume` never sleeps -- it reports how long the caller would need
    to wait so the caller can release any locks before sleeping.
    """

    def __init__(self, capacity: float, refill_per_second: float, clock: Callable[[], float]):
        self._capacity = float(capacity)
        self._rate = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    def _refill_locked(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
        elif elapsed < 0:
            # Clock moved backwards (shouldn't happen with monotonic clocks,
            # but a fake clock in tests could be misused) -- just resync.
            self._last = now

    def try_consume(self, amount: float) -> "tuple[bool, float]":
        """Attempt to take `amount` tokens. Returns (ok, wait_seconds)."""
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            if self._tokens >= amount:
                self._tokens -= amount
                return True, 0.0
            missing = amount - self._tokens
            wait = missing / self._rate if self._rate > 0 else float("inf")
            return False, wait

    def adjust(self, delta: float) -> None:
        """Debit (`delta` > 0) or credit (`delta` < 0) the bucket.

        Used for reconciliation: `delta = actual - estimated`. Deliberately
        allowed to push `_tokens` negative (representing debt) rather than
        clamping at zero -- that debt is what makes the *next* acquire wait
        the correct extra amount instead of silently forgiving an
        under-estimate.
        """
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            self._tokens = min(self._capacity, self._tokens - delta)

    def clamp_remaining(self, remaining: float) -> None:
        """Clamp down (never up) based on a server-reported remaining count."""
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            if remaining < self._tokens:
                self._tokens = remaining
            self._last = now

    def set_rate(self, rate: float) -> None:
        if rate > 0:
            with self._lock:
                self._rate = rate

    def set_capacity(self, capacity: float) -> None:
        """Adopt the server-reported per-minute limit as the bucket ceiling.

        The env default is only a guess until the first response arrives.
        Capacity feeds the refill-rate derivation in update_from_headers, so
        leaving a too-large guess in place after the server has told us the
        real tier would inflate the refill rate and push straight into 429s.
        """
        if capacity > 0:
            with self._lock:
                now = self._clock()
                self._refill_locked(now)
                self._capacity = float(capacity)
                self._tokens = min(self._tokens, self._capacity)
                self._rate = self._capacity / 60.0

    def tokens_snapshot(self) -> float:
        with self._lock:
            now = self._clock()
            self._refill_locked(now)
            return self._tokens


# --------------------------------------------------------------------------
# Per-model dual bucket
# --------------------------------------------------------------------------


class ModelLimiter:
    """RPM + TPM buckets for one model id."""

    def __init__(
        self,
        model: str,
        rpm: int,
        tpm: int,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.model = model
        self._clock = clock
        self._sleep = sleep
        self.rpm_bucket = TokenBucket(rpm, rpm / 60.0, clock)
        self.tpm_bucket = TokenBucket(tpm, tpm / 60.0, clock)

    def acquire(self, estimated_tokens: float, floor: int = 1) -> None:
        estimated_tokens = max(estimated_tokens, floor)
        while True:
            ok_req, wait_req = self.rpm_bucket.try_consume(1)
            if not ok_req:
                self._sleep(wait_req)
                continue
            ok_tok, wait_tok = self.tpm_bucket.try_consume(estimated_tokens)
            if not ok_tok:
                # Give back the request slot we reserved but didn't use yet.
                self.rpm_bucket.adjust(-1)
                self._sleep(wait_tok)
                continue
            return

    def reconcile(self, estimated_tokens: float, actual_tokens: float) -> None:
        delta = actual_tokens - estimated_tokens
        self.tpm_bucket.adjust(delta)

    def update_from_headers(self, headers: Optional[Mapping]) -> None:
        if not headers:
            return
        try:
            lim_req = _parse_int(_get_header(headers, "x-ratelimit-limit-requests"))
            lim_tok = _parse_int(_get_header(headers, "x-ratelimit-limit-tokens"))
            rem_req = _parse_int(_get_header(headers, "x-ratelimit-remaining-requests"))
            rem_tok = _parse_int(_get_header(headers, "x-ratelimit-remaining-tokens"))
            reset_req = parse_duration(_get_header(headers, "x-ratelimit-reset-requests"))
            reset_tok = parse_duration(_get_header(headers, "x-ratelimit-reset-tokens"))
        except Exception:
            return

        # Capacity first: the gap-based rate derivation below is only
        # meaningful against the server's real ceiling, not our env guess.
        if lim_req is not None:
            self.rpm_bucket.set_capacity(float(lim_req))
        if lim_tok is not None:
            self.tpm_bucket.set_capacity(float(lim_tok))

        if rem_req is not None:
            self.rpm_bucket.clamp_remaining(float(rem_req))
            if reset_req and reset_req > 0:
                gap = self.rpm_bucket.capacity - rem_req
                if gap > 0:
                    self.rpm_bucket.set_rate(gap / reset_req)
        if rem_tok is not None:
            self.tpm_bucket.clamp_remaining(float(rem_tok))
            if reset_tok and reset_tok > 0:
                gap = self.tpm_bucket.capacity - rem_tok
                if gap > 0:
                    self.tpm_bucket.set_rate(gap / reset_tok)


# --------------------------------------------------------------------------
# Global 429 circuit breaker
# --------------------------------------------------------------------------


class CircuitBreaker:
    """Pauses every caller after a 429, with jittered resume + a temporary
    concurrency haircut."""

    def __init__(
        self,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        rand: Callable[[], float] = random.random,
        default_backoff: float = 5.0,
        budget_reduction_seconds: float = 60.0,
        jitter_max: float = 1.0,
        on_wait: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._rand = rand
        self._default_backoff = default_backoff
        self._budget_reduction_seconds = budget_reduction_seconds
        self._jitter_max = jitter_max
        self._on_wait = on_wait
        self._lock = threading.Lock()
        self._paused_until = 0.0
        self._budget_reduced_until = 0.0

    def pending_pause_seconds(self) -> float:
        """How long a caller checking in right now would have to wait.

        Pure introspection (no sleeping); useful to prove the pause is
        global -- every caller sees the same shared value, not just the one
        that tripped the breaker.
        """
        now = self._clock()
        with self._lock:
            return max(0.0, self._paused_until - now)

    def trip(self, retry_after: Optional[float] = None) -> None:
        now = self._clock()
        wait = retry_after if (retry_after is not None and retry_after > 0) else self._default_backoff
        with self._lock:
            self._paused_until = max(self._paused_until, now + wait)
            self._budget_reduced_until = max(self._budget_reduced_until, now + wait + self._budget_reduction_seconds)

    def concurrency_factor(self) -> float:
        now = self._clock()
        with self._lock:
            return 0.5 if now < self._budget_reduced_until else 1.0

    def effective_concurrency(self, base_concurrency: int) -> int:
        return max(1, int(base_concurrency * self.concurrency_factor()))

    def wait_if_paused(self) -> None:
        while True:
            now = self._clock()
            with self._lock:
                remaining = self._paused_until - now
            if remaining <= 0:
                return
            if self._on_wait is not None:
                # Diagnostic hook for tests. Deliberately NOT wrapped in a
                # try/except: swallowing here once hid a broken test barrier
                # behind a silent 5-second stall.
                self._on_wait(remaining)
            self._sleep(remaining)
            # Jitter on resume so every waiter doesn't wake at the exact
            # same instant and stampede the API together.
            jitter = self._rand() * self._jitter_max
            if jitter > 0:
                self._sleep(jitter)
            # Loop: another thread's 429 may have extended the pause while
            # we were asleep.


def _retry_after_from_headers(headers: Optional[Mapping]) -> Optional[float]:
    if not headers:
        return None
    ms = _parse_float(_get_header(headers, "retry-after-ms"))
    if ms is not None:
        return ms / 1000.0
    sec = _get_header(headers, "retry-after")
    if sec is None:
        return None
    as_float = _parse_float(sec)
    if as_float is not None:
        return as_float
    return parse_duration(sec)


# --------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------

_ENCODING_CACHE: dict = {}
_FLOOR_TOKENS = 16
_FIXED_OVERHEAD_TOKENS = 8
_PER_IMAGE_TOKENS = 800


def estimate_tokens(
    text: str = "",
    images: int = 0,
    model: str = "gpt-4o-mini",
    fixed_overhead: int = _FIXED_OVERHEAD_TOKENS,
    per_image_tokens: int = _PER_IMAGE_TOKENS,
    floor: int = _FLOOR_TOKENS,
) -> int:
    """Estimate the prompt token cost of a call, for pre-reserving budget.

    Uses `tiktoken` if importable; otherwise falls back to a conservative
    upper bound of `len(text) / 4 + fixed_overhead`. Image calls add a flat
    per-image allowance on top. Never returns less than `floor`.
    """
    total = 0
    if text:
        if tiktoken is not None:
            try:
                enc = _ENCODING_CACHE.get(model)
                if enc is None:
                    try:
                        enc = tiktoken.encoding_for_model(model)
                    except Exception:
                        enc = tiktoken.get_encoding("cl100k_base")
                    _ENCODING_CACHE[model] = enc
                total += len(enc.encode(text))
            except Exception:
                total += int(len(text) / 4) + fixed_overhead
        else:
            total += int(len(text) / 4) + fixed_overhead
    total += max(images, 0) * per_image_tokens
    return max(total, floor)


# --------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------

_DEFAULT_RPM = 500
_DEFAULT_TPM = 200_000


class RateLimiter:
    """Manages one `ModelLimiter` per model id plus a shared `CircuitBreaker`.

    Config comes from the environment only (`OPENAI_RPM_LIMIT`,
    `OPENAI_TPM_LIMIT`), with conservative defaults used until the first
    real response headers arrive and clamp the buckets to reality.
    """

    def __init__(
        self,
        clock: Callable[[], float] = None,
        sleep: Callable[[float], None] = None,
        rand: Callable[[], float] = random.random,
        default_rpm: Optional[int] = None,
        default_tpm: Optional[int] = None,
        on_breaker_wait: Optional[Callable[[float], None]] = None,
    ) -> None:
        import time as _time  # local import only to source stdlib defaults

        self._clock = clock or _time.monotonic
        self._sleep = sleep or _time.sleep
        self._rand = rand
        self._default_rpm = default_rpm or _parse_int(os.environ.get("OPENAI_RPM_LIMIT")) or _DEFAULT_RPM
        self._default_tpm = default_tpm or _parse_int(os.environ.get("OPENAI_TPM_LIMIT")) or _DEFAULT_TPM
        self._limiters: dict = {}
        self._registry_lock = threading.Lock()
        self.breaker = CircuitBreaker(clock=self._clock, sleep=self._sleep, rand=self._rand, on_wait=on_breaker_wait)

    def _get(self, model: str) -> ModelLimiter:
        with self._registry_lock:
            lim = self._limiters.get(model)
            if lim is None:
                lim = ModelLimiter(model, self._default_rpm, self._default_tpm, self._clock, self._sleep)
                self._limiters[model] = lim
            return lim

    def reserve(self, model: str, estimated_tokens: float) -> None:
        self.breaker.wait_if_paused()
        self._get(model).acquire(estimated_tokens)

    def reconcile(self, model: str, estimated_tokens: float, actual_tokens: float) -> None:
        self._get(model).reconcile(estimated_tokens, actual_tokens)

    def update_from_headers(self, model: str, headers: Optional[Mapping]) -> None:
        self._get(model).update_from_headers(headers)

    def on_rate_limited(self, headers: Optional[Mapping] = None, retry_after: Optional[float] = None) -> None:
        ra = retry_after if retry_after is not None else _retry_after_from_headers(headers)
        self.breaker.trip(ra)

    def call_with_retry(
        self,
        fn: Callable[[], object],
        *,
        model: str,
        estimated_tokens: float = 1,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> object:
        """Run `fn()` with proactive limiting, retry, and breaker interop.

        `fn` must raise `RateLimitError` on 429 and `TransientAPIError` on a
        5xx/connection failure; any other exception propagates immediately
        without retry. On success, if the return value exposes `.headers`,
        it is fed back into the limiter automatically.
        """
        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt < max_attempts:
            attempt += 1
            self.reserve(model, estimated_tokens)
            try:
                result = fn()
            except RateLimitError as exc:
                last_exc = exc
                self.on_rate_limited(headers=exc.headers, retry_after=exc.retry_after)
                if attempt >= max_attempts:
                    raise
                continue
            except TransientAPIError as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                self._sleep(self._rand() * delay)
                continue
            else:
                headers = getattr(result, "headers", None)
                if headers is not None:
                    self.update_from_headers(model, headers)
                return result
        # max_attempts == 0 edge case, or loop exhausted without raising above
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("call_with_retry: exhausted attempts with no recorded exception")
