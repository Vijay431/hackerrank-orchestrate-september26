"""Tests for code/runner.py -- the bounded-concurrency batch runner.

Plain unittest, no pytest dependency. Run with:

    .venv/bin/python tests/test_runner.py

Uses a stub task function (never the real pipeline) against a temp
checkpoint directory per test, so the real cache/results/ is never touched.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

try:  # running with code/ on sys.path
    from runner import BatchRunner, RunInterrupted
except ImportError:  # running with the repo root on sys.path instead
    from code.runner import BatchRunner, RunInterrupted


def make_items(n: int) -> list[dict]:
    return [{"id": i, "value": i * 3} for i in range(n)]


def item_id(item: dict) -> str:
    return f"item_{item['id']}"


def variable_delay_task(item: dict) -> dict:
    """Deterministic-but-variable delay so completion order genuinely
    differs from submission order, exercising the sort-back-to-input-order
    guarantee."""
    delay = random.Random(item["id"] * 7919 + 3).uniform(0, 0.02)
    threading.Event().wait(delay)
    return {"id": item["id"], "doubled": item["value"] * 2}


def default_on_error(item: dict, exc: BaseException) -> dict:
    return {"id": item["id"], "doubled": None, "error": str(exc)}


class DeterministicOrderTest(unittest.TestCase):
    def test_concurrency_1_vs_16_identical_output(self) -> None:
        items = make_items(40)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            runner1 = BatchRunner(
                task=variable_delay_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d1,
                concurrency=1,
                quiet=True,
            )
            runner16 = BatchRunner(
                task=variable_delay_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d2,
                concurrency=16,
                quiet=True,
            )
            result1 = runner1.run(items)
            result16 = runner16.run(items)

        self.assertEqual(len(result1), len(items))
        self.assertEqual(len(result16), len(items))
        self.assertEqual(result1, result16)
        # And it must actually be in input order, not just "equal to itself".
        expected = [{"id": i["id"], "doubled": i["value"] * 2} for i in items]
        self.assertEqual(result1, expected)
        self.assertEqual(result16, expected)


class RaisingTaskTest(unittest.TestCase):
    def test_raising_task_yields_fallback_not_missing_row(self) -> None:
        items = make_items(10)

        def flaky_task(item: dict) -> dict:
            if item["id"] % 3 == 0:
                raise ValueError(f"boom-{item['id']}")
            return {"id": item["id"], "doubled": item["value"] * 2}

        with tempfile.TemporaryDirectory() as d:
            runner = BatchRunner(
                task=flaky_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                quiet=True,
            )
            results = runner.run(items)

        self.assertEqual(len(results), len(items))
        for item, result in zip(items, results):
            if item["id"] % 3 == 0:
                self.assertIsNone(result["doubled"])
                self.assertIn(f"boom-{item['id']}", result["error"])
            else:
                self.assertEqual(result["doubled"], item["value"] * 2)


class ResumeTest(unittest.TestCase):
    def test_resume_skips_completed_items_and_does_not_reinvoke_task(self) -> None:
        items = make_items(12)
        calls: dict[str, int] = {}
        lock = threading.Lock()

        def counting_task(item: dict) -> dict:
            iid = item_id(item)
            with lock:
                calls[iid] = calls.get(iid, 0) + 1
            return {"id": item["id"], "doubled": item["value"] * 2}

        with tempfile.TemporaryDirectory() as d:
            first = BatchRunner(
                task=counting_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                quiet=True,
            )
            first_results = first.run(items)
            self.assertEqual(len(first_results), len(items))
            self.assertTrue(all(v == 1 for v in calls.values()))
            self.assertEqual(len(calls), len(items))

            # Simulate a kill: wipe the checkpoint for a few items so a
            # resumed run must recompute exactly those and nothing else.
            missing_ids = {item_id(items[2]), item_id(items[7])}
            for iid in missing_ids:
                os.remove(os.path.join(d, f"{iid}.json"))
            calls.clear()

            second = BatchRunner(
                task=counting_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                resume=True,
                quiet=True,
            )
            second_results = second.run(items)

        # Only the two wiped items should have been recomputed.
        self.assertEqual(set(calls.keys()), missing_ids)
        self.assertTrue(all(v == 1 for v in calls.values()))
        # Full, correctly ordered result set regardless.
        expected = [{"id": i["id"], "doubled": i["value"] * 2} for i in items]
        self.assertEqual(second_results, expected)

    def test_fully_resumed_run_invokes_task_zero_times(self) -> None:
        items = make_items(6)
        calls: dict[str, int] = {}
        lock = threading.Lock()

        def counting_task(item: dict) -> dict:
            iid = item_id(item)
            with lock:
                calls[iid] = calls.get(iid, 0) + 1
            return {"id": item["id"], "doubled": item["value"] * 2}

        with tempfile.TemporaryDirectory() as d:
            first = BatchRunner(
                task=counting_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                quiet=True,
            )
            first.run(items)
            calls.clear()

            second = BatchRunner(
                task=counting_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                resume=True,
                quiet=True,
            )
            results = second.run(items)

        self.assertEqual(calls, {})
        expected = [{"id": i["id"], "doubled": i["value"] * 2} for i in items]
        self.assertEqual(results, expected)


class CorruptCheckpointTest(unittest.TestCase):
    def test_corrupt_checkpoint_is_recomputed_not_fatal(self) -> None:
        items = make_items(5)
        calls: dict[str, int] = {}
        lock = threading.Lock()

        def counting_task(item: dict) -> dict:
            iid = item_id(item)
            with lock:
                calls[iid] = calls.get(iid, 0) + 1
            return {"id": item["id"], "doubled": item["value"] * 2}

        with tempfile.TemporaryDirectory() as d:
            # Pre-seed a corrupt checkpoint (truncated JSON) for one item,
            # and a valid checkpoint for another, before any run happens.
            corrupt_iid = item_id(items[1])
            valid_iid = item_id(items[3])
            with open(os.path.join(d, f"{corrupt_iid}.json"), "w") as fh:
                fh.write('{"item_id": "item_1", "ok": true, "result": {')  # truncated
            with open(os.path.join(d, f"{valid_iid}.json"), "w") as fh:
                json.dump(
                    {
                        "item_id": valid_iid,
                        "ok": True,
                        "result": {"id": 3, "doubled": 999},
                        "error": None,
                    },
                    fh,
                )

            runner = BatchRunner(
                task=counting_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=4,
                resume=True,
                quiet=True,
            )
            results = runner.run(items)

        # Corrupt checkpoint => recomputed (task invoked for it).
        self.assertIn(corrupt_iid, calls)
        # Valid pre-seeded checkpoint => not recomputed, and its stored
        # (deliberately wrong-looking) value is what resume loads.
        self.assertNotIn(valid_iid, calls)
        by_id = {f"item_{r['id']}": r for r in results}
        self.assertEqual(by_id[valid_iid]["doubled"], 999)
        self.assertEqual(by_id[corrupt_iid]["doubled"], items[1]["value"] * 2)
        self.assertEqual(len(results), len(items))


class InFlightCapTest(unittest.TestCase):
    def test_max_in_flight_caps_concurrent_task_bodies(self) -> None:
        items = make_items(30)
        active = {"count": 0, "peak": 0}
        lock = threading.Lock()

        def tracking_task(item: dict) -> dict:
            with lock:
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
            try:
                delay = random.Random(item["id"] * 101 + 1).uniform(0, 0.015)
                threading.Event().wait(delay)
            finally:
                with lock:
                    active["count"] -= 1
            return {"id": item["id"], "doubled": item["value"] * 2}

        with tempfile.TemporaryDirectory() as d:
            runner = BatchRunner(
                task=tracking_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=16,
                max_in_flight=3,
                quiet=True,
            )
            results = runner.run(items)

        self.assertEqual(len(results), len(items))
        self.assertLessEqual(active["peak"], 3)
        # Sanity: the cap was actually meaningful (pool would otherwise run
        # far more than 3 concurrently with 16 workers and 30 items).
        self.assertGreaterEqual(active["peak"], 1)


class CheckpointFileTest(unittest.TestCase):
    def test_checkpoints_written_to_disk(self) -> None:
        items = make_items(4)
        with tempfile.TemporaryDirectory() as d:
            runner = BatchRunner(
                task=variable_delay_task,
                item_id=item_id,
                on_error=default_on_error,
                checkpoint_dir=d,
                concurrency=2,
                quiet=True,
            )
            runner.run(items)
            for item in items:
                path = os.path.join(d, f"{item_id(item)}.json")
                self.assertTrue(os.path.exists(path))
                with open(path) as fh:
                    data = json.load(fh)
                self.assertTrue(data["ok"])
                self.assertEqual(data["result"]["doubled"], item["value"] * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
