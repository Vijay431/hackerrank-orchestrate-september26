"""Bounded-concurrency batch runner.

Generic over the work-item type `T` and the result type `R`, so the same
engine drives both the per-request decision pipeline (`RequestRecord` ->
`Decision`) and the extraction passes (`Message`/`ImageRef` -> `Amendment` /
`ImageAmount`) without knowing anything about either domain.

Design invariants (see the task brief this module was built against):

1. Bounded worker pool: `ThreadPoolExecutor(max_workers=concurrency)`. All
   items are submitted up front; the executor's internal queue provides the
   fan-out bound. Callers load shared read-only context once (dataset
   indexes, extraction caches) and hand it to the task callable via closure
   or `functools.partial` -- this module never parses anything itself.
2. Deterministic output ordering: results are collected as they complete
   (for live progress) but the list returned by `run()` is always sorted
   back into input order, so output is byte-identical regardless of
   `concurrency`.
3. Per-item checkpointing to `<checkpoint_dir>/<item_id>.json`, written
   atomically (temp file in the same directory, then `os.replace`) so a
   kill mid-write can never leave a corrupt file visible under its final
   name.
4. `resume=True` skips any item with a valid checkpoint and loads the
   stored result instead of recomputing; a checkpoint that fails to parse
   is treated as absent (recomputed), never a fatal error.
5. A task that raises -- even after the caller's own retries -- still
   yields exactly one result: `on_error(item, exc)` supplies the fallback.
   `run()` always returns exactly one result per input item, in input
   order. This module carries no financial semantics; the fallback is 100%
   the caller's business logic.
6. Throttled progress to stderr (completed/total/failures/in-flight),
   suppressible with `quiet=True`.
7. Optional rate-limit gating: pass `gate=` to the constructor. This
   module does not import or depend on any specific rate limiter -- the
   seam is intentionally narrow, see `_gate_context` below for the exact
   duck-typed contract. Independently, `max_in_flight=` installs the
   runner's own `threading.Semaphore` around task execution, capping
   concurrent task bodies below the thread count regardless of whether a
   `gate` is supplied.
8. Graceful interrupt: on `KeyboardInterrupt`, or when `stop_event` (a
   `threading.Event` the caller can set from a signal handler) becomes set,
   the runner stops scheduling unstarted work, cancels futures that never
   started, waits for in-flight work to finish and checkpoint, and then
   raises `RunInterrupted` carrying whatever results *did* complete. It
   never silently returns a truncated list -- a caller that wants partial
   output must catch `RunInterrupted` and read `.results` explicitly.

Standard library only. No imports from `contracts.py`, `ratelimit.py`,
`extraction.py`, or `scripts/`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Generic, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_CHECKPOINT_DIR = os.path.join("cache", "results")


class RunInterrupted(Exception):
    """Raised when a run is stopped early (KeyboardInterrupt or stop_event).

    `results` holds every result that completed and was checkpointed before
    the stop, sorted back into input order for the items that finished. It
    is intentionally shorter than the input when the run was cut short --
    the caller must opt in to treating that as acceptable by catching this
    exception, rather than a truncated list silently masquerading as a
    complete run.
    """

    def __init__(self, message: str, results: list[Any]):
        super().__init__(message)
        self.results = results


class _NullContext:
    """No-op context manager used when no gate is configured."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


_NULL_CONTEXT = _NullContext()


def _gate_context(gate: Any) -> Any:
    """Resolve the caller-supplied `gate` into a context manager for one call.

    Narrow, duck-typed contract (deliberately not coupled to any concrete
    rate limiter implementation):

    * `gate is None` -> no-op.
    * `gate` already supports the context manager protocol (has `__enter__`
      and `__exit__`, e.g. a `threading.Semaphore` or a purpose-built
      rate-limit gate that is itself reentrant/thread-safe) -> used directly
      with `with gate:`.
    * `gate` is a zero-argument callable that returns a context manager
      (a factory called once per task invocation) -> `with gate():`.

    Anything else is a caller error and raises `TypeError` immediately
    rather than failing silently deep inside a worker thread.
    """
    if gate is None:
        return _NULL_CONTEXT
    if hasattr(gate, "__enter__") and hasattr(gate, "__exit__"):
        return gate
    if callable(gate):
        return gate()
    raise TypeError(
        "gate must be None, a context manager, or a zero-arg callable "
        "returning one"
    )


class _Progress:
    """Thread-safe counters plus throttled stderr reporting."""

    def __init__(self, total: int, quiet: bool, min_interval: float = 0.3) -> None:
        self._lock = threading.Lock()
        self.total = total
        self.completed = 0
        self.failures = 0
        self.in_flight = 0
        self._quiet = quiet
        self._min_interval = min_interval
        self._last_emit = 0.0

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1
        self._maybe_emit()

    def leave(self, failed: bool) -> None:
        with self._lock:
            self.in_flight -= 1
            self.completed += 1
            if failed:
                self.failures += 1
        self._maybe_emit(force=False)

    def _maybe_emit(self, force: bool = False) -> None:
        if self._quiet:
            return
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_emit) < self._min_interval:
                return
            self._last_emit = now
            completed, total, failures, in_flight = (
                self.completed,
                self.total,
                self.failures,
                self.in_flight,
            )
        print(
            f"[runner] completed={completed}/{total} "
            f"failures={failures} in_flight={in_flight}",
            file=sys.stderr,
        )

    def final(self) -> None:
        self._maybe_emit(force=True)


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write `payload` to `path` as JSON, atomically.

    Writes to a temp file in the same directory (so `os.replace` is a same
    filesystem rename, guaranteed atomic on POSIX) then replaces the target.
    A process killed mid-write leaves only the temp file behind; the final
    path either holds the old checkpoint or the new one, never a partial one.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class BatchRunner(Generic[T, R]):
    """Bounded-concurrency batch executor with checkpointing and resume."""

    def __init__(
        self,
        task: Callable[[T], R],
        *,
        item_id: Callable[[T], str],
        on_error: Callable[[T, BaseException], R],
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        concurrency: int = 8,
        resume: bool = False,
        serialize: Callable[[R], Any] = lambda r: r,
        deserialize: Callable[[Any], R] = lambda d: d,
        gate: Any | None = None,
        max_in_flight: int | None = None,
        quiet: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        task:           called once per item on a worker thread -> R.
        item_id:        item -> stable string id, used for the checkpoint
                         filename and for progress/error reporting. Must be
                         filesystem-safe (no path separators).
        on_error:       item, exception -> fallback R. Invoked whenever
                         `task` raises. This is the only place financial (or
                         other domain) fallback semantics may live; this
                         module never invents one.
        checkpoint_dir: directory for `<item_id>.json` checkpoint files.
        concurrency:    ThreadPoolExecutor worker count.
        resume:         when True, an item with a valid existing checkpoint
                         is loaded instead of recomputed; `task` is not
                         invoked for it.
        serialize:      R -> JSON-safe value, stored in the checkpoint.
                         Defaults to identity (fine when R is already
                         JSON-native, e.g. in tests).
        deserialize:    JSON-safe value -> R, the inverse of `serialize`,
                         used when loading a checkpoint on resume.
        gate:           optional external rate-limit hook, see
                         `_gate_context` for the exact contract. Not
                         imported from anywhere -- caller wires it in.
        max_in_flight:  optional cap on concurrently *executing* task
                         bodies, independent of `concurrency` (e.g. run 8
                         worker threads but allow only 3 concurrent API
                         calls). Implemented as a plain
                         `threading.Semaphore`.
        quiet:          suppress stderr progress reporting (tests want this).
        stop_event:     optional `threading.Event`; if set (by caller code,
                         e.g. a signal handler) the runner stops scheduling
                         new items and raises `RunInterrupted` once in-flight
                         work drains.
        """
        self._task = task
        self._item_id = item_id
        self._on_error = on_error
        self._checkpoint_dir = checkpoint_dir
        self._concurrency = concurrency
        self._resume = resume
        self._serialize = serialize
        self._deserialize = deserialize
        self._gate = gate
        self._in_flight_sema = (
            threading.Semaphore(max_in_flight) if max_in_flight else None
        )
        self._quiet = quiet
        self._stop_event = stop_event or threading.Event()

    # -- checkpoint helpers -------------------------------------------------

    def _checkpoint_path(self, iid: str) -> str:
        return os.path.join(self._checkpoint_dir, f"{iid}.json")

    def _load_checkpoint(self, iid: str) -> tuple[bool, R | None]:
        """Return (found_valid, result). Any parse/shape problem -> (False, None)."""
        path = self._checkpoint_path(iid)
        if not os.path.exists(path):
            return False, None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if "result" not in data:
                return False, None
            if not data.get("ok", True):
                # A fallback row from a previous run. Resume exists to finish
                # work that did not succeed, so recompute it -- reusing it
                # would bake a transient API failure into the submission
                # permanently. The file stays on disk for diagnosis.
                return False, None
            result = self._deserialize(data["result"])
            return True, result
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError):
            return False, None

    def _write_checkpoint(
        self,
        iid: str,
        result: R,
        ok: bool,
        error_info: dict | None,
    ) -> None:
        payload = {
            "item_id": iid,
            "ok": ok,
            "result": self._serialize(result),
            "error": error_info,
        }
        try:
            _atomic_write_json(self._checkpoint_path(iid), payload)
        except Exception as exc:  # noqa: BLE001
            # The work already succeeded; only persistence failed. Losing the
            # row here would truncate output.csv, and raising would abort the
            # whole batch -- both are far worse than an un-resumable item.
            # Most likely cause: `serialize` does not cover the result type
            # (the default identity serializer cannot encode Decimal).
            print(
                f"[runner] WARNING: checkpoint write failed for {iid}: "
                f"{type(exc).__name__}: {exc} -- continuing without a "
                f"checkpoint for this item",
                file=sys.stderr,
            )

    # -- single-item execution ----------------------------------------------

    def _run_one(self, item: T, iid: str, progress: _Progress) -> R:
        """Execute one item (task, gating, error fallback, checkpoint)."""
        progress.enter()
        failed = False
        try:
            sema = self._in_flight_sema
            if sema is not None:
                sema.acquire()
            try:
                with _gate_context(self._gate):
                    result = self._task(item)
            finally:
                if sema is not None:
                    sema.release()
            self._write_checkpoint(iid, result, ok=True, error_info=None)
            return result
        except (KeyboardInterrupt, SystemExit):
            # Not a data problem. Converting Ctrl-C into a fallback row would
            # checkpoint a bogus answer for whatever item happened to be in
            # flight; let run()'s interrupt path handle it instead.
            self._stop_event.set()
            raise
        except Exception as exc:  # noqa: BLE001 -- must never lose the row
            failed = True
            tb = traceback.format_exc()
            fallback = self._on_error(item, exc)
            error_info = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            }
            self._write_checkpoint(iid, fallback, ok=False, error_info=error_info)
            return fallback
        finally:
            progress.leave(failed=failed)

    # -- public entry point ---------------------------------------------------

    def run(self, items: Sequence[T]) -> list[R]:
        """Run every item, return results in input order.

        Raises `RunInterrupted` (carrying whatever completed) if a
        KeyboardInterrupt arrives or `stop_event` becomes set while work is
        outstanding. Otherwise always returns `len(items)` results.
        """
        total = len(items)
        progress = _Progress(total=total, quiet=self._quiet)

        # slot[i] will hold the eventual result for items[i].
        slots: list[R | None] = [None] * total
        filled = [False] * total

        ids = [self._item_id(item) for item in items]
        # Ids name checkpoint files: a duplicate would alias two items'
        # checkpoints (one poisons the other on resume) and a path separator
        # would escape checkpoint_dir. Fail before any work starts.
        seen: set[str] = set()
        for iid in ids:
            if not iid or os.sep in iid or (os.altsep and os.altsep in iid) or iid in (".", ".."):
                raise ValueError(f"item_id {iid!r} is not filesystem-safe")
            if iid in seen:
                raise ValueError(f"duplicate item_id {iid!r}")
            seen.add(iid)

        # Resume pass: satisfy anything with a valid checkpoint without
        # touching the thread pool at all, so `task` genuinely is not
        # invoked for these.
        pending_indices: list[int] = []
        if self._resume:
            for idx, iid in enumerate(ids):
                found, result = self._load_checkpoint(iid)
                if found:
                    slots[idx] = result
                    filled[idx] = True
                    progress.completed += 1
                else:
                    pending_indices.append(idx)
        else:
            pending_indices = list(range(total))

        interrupted = False
        executor = ThreadPoolExecutor(max_workers=max(1, self._concurrency))
        future_to_idx: dict[Future, int] = {}
        try:
            for idx in pending_indices:
                if self._stop_event.is_set():
                    interrupted = True
                    break
                fut = executor.submit(
                    self._run_one, items[idx], ids[idx], progress
                )
                future_to_idx[fut] = idx

            try:
                for fut in future_to_idx:
                    if self._stop_event.is_set() and not fut.running() and not fut.done():
                        fut.cancel()
                        continue
                    try:
                        result = fut.result()
                    except BaseException:
                        # _run_one already swallows task exceptions via
                        # on_error; a raise here would mean on_error itself
                        # blew up. Do not silently drop the row: synthesize
                        # a last-resort fallback via on_error again is
                        # unsafe (it just failed), so re-raise -- this is a
                        # caller bug, not a data problem.
                        raise
                    idx = future_to_idx[fut]
                    slots[idx] = result
                    filled[idx] = True
            except KeyboardInterrupt:
                interrupted = True
                self._stop_event.set()
                # Cancel anything that hasn't started; let running ones finish.
                for fut in future_to_idx:
                    fut.cancel()
                # Drain whatever does complete so their checkpoints land and
                # we can report as much as possible in RunInterrupted.
                for fut, idx in future_to_idx.items():
                    if fut.cancelled():
                        continue
                    # fut.exception() blocks like result() but hands back a
                    # worker's stored exception instead of re-raising it, so
                    # a *fresh* Ctrl-C hitting this thread during the wait is
                    # the only thing that propagates -- force-quit still works.
                    exc = fut.exception()
                    if exc is None:
                        slots[idx] = fut.result()
                        filled[idx] = True
                    elif isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        continue  # the worker was interrupted; row simply not done
                    else:
                        # Only reachable if on_error itself raised. Say so
                        # rather than let the row vanish from the partial list.
                        print(f"[runner] WARNING: {ids[idx]} lost during drain: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            progress.final()

        if interrupted or self._stop_event.is_set():
            completed_results = [slots[i] for i in range(total) if filled[i]]
            raise RunInterrupted(
                f"run interrupted after {len(completed_results)}/{total} items",
                completed_results,
            )

        if not all(filled):  # never `assert`: python -O would strip it
            missing = [ids[i] for i in range(total) if not filled[i]]
            raise RuntimeError(f"internal error: no result for {missing}")
        return [slots[i] for i in range(total)]  # type: ignore[misc]
