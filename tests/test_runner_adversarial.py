"""Adversarial regression tests for code/runner.py.

These cover three failure modes that the module's own test suite passed
straight through, because each one only shows up when something *outside*
the task function goes wrong. All three were live defects; this file exists
so they cannot come back.
"""

import os
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import runner as R  # noqa: E402


def _runner(task, on_error, d, **kw):
    return R.BatchRunner(
        task=task,
        on_error=on_error,
        item_id=lambda i: f"r{i}",
        checkpoint_dir=d,
        quiet=True,
        **kw,
    )


class CheckpointWriteFailureTest(unittest.TestCase):
    """A checkpoint that cannot be written must not lose the row.

    The default `serialize` is identity, and json.dump cannot encode Decimal
    -- which is exactly how every money value in this project is carried. The
    write happened inside the same try block that produces the fallback, so a
    serialization failure first turned a good result into a fallback and then
    raised out of run(), killing the whole batch.
    """

    def test_result_survives_unserializable_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            run = _runner(
                lambda i: {"amount": Decimal("87170.56")},
                lambda i, e: {"amount": Decimal("0"), "fallback": True},
                d,
                concurrency=4,
            )
            out = run.run(list(range(5)))
            self.assertEqual(len(out), 5, "rows were lost")
            self.assertEqual(
                [o for o in out if o.get("fallback")], [],
                "a successful result was downgraded to a fallback")
            self.assertEqual(out[0]["amount"], Decimal("87170.56"))


class FailedCheckpointNotResumedTest(unittest.TestCase):
    """Resume must retry failures, not replay them.

    A fallback row was checkpointed like any other result, and resume only
    checked that a `result` key existed. One transient API failure therefore
    baked a `not_affordable` answer into every subsequent resumed run.
    """

    def test_fallback_row_is_recomputed_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []

            def task(i):
                calls.append(i)
                if i == 1 and len(calls) < 3:
                    raise ValueError("transient outage")
                return f"good_{i}"

            on_error = lambda i, e: f"FALLBACK_{i}"  # noqa: E731
            first = _runner(task, on_error, d, concurrency=1).run([0, 1, 2])
            self.assertEqual(first[1], "FALLBACK_1")

            second = _runner(task, on_error, d, concurrency=1,
                             resume=True).run([0, 1, 2])
            self.assertEqual(second, ["good_0", "good_1", "good_2"])

    def test_successful_checkpoint_is_still_reused(self):
        """Guard the other direction: the retry rule must not defeat resume."""
        with tempfile.TemporaryDirectory() as d:
            calls = []

            def task(i):
                calls.append(i)
                return f"good_{i}"

            on_error = lambda i, e: f"FALLBACK_{i}"  # noqa: E731
            _runner(task, on_error, d, concurrency=1).run([0, 1, 2])
            calls.clear()
            out = _runner(task, on_error, d, concurrency=1,
                          resume=True).run([0, 1, 2])
            self.assertEqual(out, ["good_0", "good_1", "good_2"])
            self.assertEqual(calls, [], "resume recomputed successful items")


class InterruptNotSwallowedTest(unittest.TestCase):
    """Ctrl-C is not a data error.

    The worker caught BaseException, so KeyboardInterrupt was converted into a
    fallback row and checkpointed -- combined with the bug above, an
    interrupted run permanently poisoned whatever request was in flight.
    """

    def test_keyboard_interrupt_propagates_and_leaves_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            def task(i):
                if i == 1:
                    raise KeyboardInterrupt()
                return f"good_{i}"

            run = _runner(task, lambda i, e: f"FALLBACK_{i}", d, concurrency=1)
            with self.assertRaises(R.RunInterrupted):
                run.run([0, 1, 2])
            self.assertNotIn("r1.json", os.listdir(d),
                             "interrupt was checkpointed as a real answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
