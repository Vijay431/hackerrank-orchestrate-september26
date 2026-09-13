"""Gate 3 integration check: the composed forecast on real dataset fixtures.

Runs fully offline -- every message it needs is already in cache/amendments.json,
and the extractor factory is replaced by one that raises, so an unexpected
cache miss fails the test instead of spending tokens.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from contracts import FORECAST_DAYS  # noqa: E402
from extraction import DiskExtractionCache  # noqa: E402
from forecast import build_context  # noqa: E402
from forecast.series import flow_category, flow_tag  # noqa: E402
from forecast.simulate import earliest_full_payment_date, safe_amount_on  # noqa: E402
from loaders import load_dataset, load_labelled  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def _no_api() -> None:
    raise AssertionError("Gate 3 must not call the API -- cache miss")


class Gate3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(os.path.join(REPO, "dataset")):
            raise unittest.SkipTest("dataset/ missing")
        cls.data = load_dataset(os.path.join(REPO, "dataset"))
        cls.labelled = {
            l.request.request_id: l
            for l in load_labelled(os.path.join(REPO, "dataset", "json", "sample_requests.json"))
        }
        cls.cache = DiskExtractionCache(
            os.path.join(REPO, "cache"), extractor_factory=_no_api, repo_root=REPO
        )

    def test_user_01_request_01_picture(self) -> None:
        lab = self.labelled["request_01"]
        ctx = build_context(lab.request, self.data, self.cache)
        fc = ctx.forecast
        self.assertEqual(len(fc.balances), FORECAST_DAYS)
        self.assertEqual(fc.opening_balance, Decimal("58481.1"))

        by_cat = {(s.category, s.direction): s for s in ctx.series}
        rent = by_cat[("rent", "debit")]
        self.assertTrue(rent.is_fixed)
        self.assertEqual(rent.amount, Decimal("5148"))
        self.assertEqual(rent.cadence_days, 30)
        salary = by_cat[("salary", "credit")]
        self.assertEqual(salary.anchor, date(2024, 3, 15))
        self.assertEqual(salary.source_event_id, "event_103")
        self.assertEqual(salary.amount, Decimal("23320"))

        salary_days = sorted(f.when for f in fc.flows if flow_category(f) == "salary")
        self.assertEqual(salary_days, [date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15)])
        # The scheduled row is emitted once as a one-off; the two projected
        # occurrences carry the same source_event_id but the cadence tag.
        sched = [f for f in fc.flows if f.source_event_id == "event_103" and not f.is_recurring]
        self.assertEqual(len(sched), 1)
        self.assertEqual(flow_tag(sched[0]), "scheduled")
        self.assertEqual(sched[0].when, date(2024, 3, 15))
        self.assertEqual(sched[0].delta, Decimal("23320"))
        projected = [f for f in fc.flows if f.source_event_id == "event_103" and f.is_recurring]
        self.assertEqual([f.when for f in projected], [date(2024, 4, 15), date(2024, 5, 15)])

        pending = [f for f in fc.flows if f.source_event_id == "event_102"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].when, date(2024, 3, 3))
        self.assertEqual(pending[0].delta, Decimal("-567.6"))

        self.assertFalse(any(f.source_event_id == "event_100" for f in fc.flows))
        self.assertFalse(any(e.event_id == "event_100" for e in ctx.events))
        self.assertFalse(any(f.source_event_id in ("event_98", "event_99") for f in fc.flows))

        minimum = ctx.profile.minimum_balance_to_keep
        safe = safe_amount_on(fc, lab.request.request_date, minimum, lab.request.requested_amount)
        print(
            f"\n[gate3] request_01 trough={fc.trough()} safe={safe} "
            f"label={lab.amount_safe_to_pay} flows={len(fc.flows)}"
        )
        self.assertEqual(safe, Decimal("25256"))
        self.assertEqual(
            earliest_full_payment_date(fc, lab.request.requested_amount, minimum),
            date(2024, 3, 3),
        )

    def test_user_02_request_02_salary_raise(self) -> None:
        lab = self.labelled["request_02"]
        ctx = build_context(lab.request, self.data, self.cache)
        fc = ctx.forecast
        self.assertEqual(len(ctx.amendments), 1)
        am = ctx.amendments[0]
        self.assertEqual(am.kind, "salary_change")
        self.assertEqual(am.effective_date, date(2025, 8, 15))
        salary_flows = sorted(
            (f for f in fc.flows if flow_category(f) == "salary"), key=lambda f: f.when
        )
        self.assertTrue(salary_flows, "no projected salary for user_02")
        raised = [f for f in salary_flows if f.when >= date(2025, 8, 15)]
        self.assertTrue(raised)
        for f in raised:
            self.assertEqual(f.delta, Decimal("42750000"))

        minimum = ctx.profile.minimum_balance_to_keep
        safe = safe_amount_on(fc, lab.request.request_date, minimum, lab.request.requested_amount)
        earliest = earliest_full_payment_date(fc, lab.request.requested_amount, minimum)
        print(
            f"\n[gate3] request_02 trough={fc.trough()} safe={safe} "
            f"label={lab.amount_safe_to_pay} earliest={earliest} "
            f"label_earliest={lab.earliest_date_for_full_payment} "
            f"salary_days={[str(f.when) for f in salary_flows]}"
        )


if __name__ == "__main__":
    unittest.main()
