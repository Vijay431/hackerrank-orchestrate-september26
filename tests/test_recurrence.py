import os
import sys
import unittest
from datetime import date
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from contracts import CashFlow, Event, FORECAST_DAYS  # noqa: E402
from forecast.series import (  # noqa: E402
    CADENCE_BIWEEKLY,
    CADENCE_MONTHLY,
    CADENCE_WEEKLY,
    RecurringSeries,
    flow_label,
)
from forecast.recurrence import (  # noqa: E402
    MIN_OCCURRENCES,
    STALE_FACTOR,
    VARIABLE_LOOKBACK,
    detect_recurring,
    project,
    project_all,
    variable_basis,
)


def make_event(
    event_id,
    category,
    direction,
    amount,
    settlement_date,
    status="settled",
    flexibility="fixed",
    minimum_allowed_amount=None,
    user_id="user_01",
    event_date=None,
    event_type="expense",
    description="",
    currency="ZAR",
    linked_event_id=None,
):
    """Build an Event fixture. ``amount`` must already be a Decimal or None."""
    return Event(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        description=description or category,
        category=category,
        direction=direction,
        amount=amount,
        currency=currency,
        event_date=event_date if event_date is not None else settlement_date,
        settlement_date=settlement_date,
        status=status,
        linked_event_id=linked_event_id,
        flexibility=flexibility,
        minimum_allowed_amount=minimum_allowed_amount,
    )


def find_series(series, category, direction):
    for s in series:
        if s.category == category and s.direction == direction:
            return s
    return None


class VariableBasisTests(unittest.TestCase):
    def test_odd_count_is_middle_value(self):
        amounts = [Decimal("1"), Decimal("5"), Decimal("3")]
        self.assertEqual(variable_basis(amounts), Decimal("3"))

    def test_even_count_is_mean_of_middle_two(self):
        amounts = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
        self.assertEqual(variable_basis(amounts), Decimal("2.5"))


class DetectWeeklyVariableTests(unittest.TestCase):
    def test_weekly_variable_series_uses_median_of_last_lookback(self):
        # 10 weekly groceries debits, amounts chosen so the median of the
        # trailing VARIABLE_LOOKBACK (8) differs from the median of all 10.
        amounts = [
            "900", "100", "800", "200", "700", "300", "600", "400", "1000", "50",
        ]
        events = []
        for i, amt in enumerate(amounts):
            when = date(2024, 1, 1)
            when = date.fromordinal(when.toordinal() + 7 * i)
            events.append(make_event(f"g{i}", "groceries", "debit", Decimal(amt), when))

        series = detect_recurring(events, as_of=date(2024, 3, 15))
        s = find_series(series, "groceries", "debit")
        self.assertIsNotNone(s)
        self.assertEqual(s.cadence_days, CADENCE_WEEKLY)
        self.assertFalse(s.is_fixed)
        last_eight = [Decimal(a) for a in amounts[-VARIABLE_LOOKBACK:]]
        self.assertEqual(s.amount, variable_basis(last_eight))
        self.assertEqual(s.anchor, date.fromordinal(date(2024, 1, 1).toordinal() + 7 * 9))
        self.assertEqual(s.source_event_id, "g9")
        self.assertEqual(s.event_ids, tuple(f"g{i}" for i in range(10)))


class DetectFixedMonthlyTests(unittest.TestCase):
    def test_fixed_monthly_rent_exact_amount(self):
        events = [
            make_event("r0", "rent", "debit", Decimal("5148"), date(2023, 10, 2)),
            make_event("r1", "rent", "debit", Decimal("5148"), date(2023, 11, 2)),
            make_event("r2", "rent", "debit", Decimal("5148"), date(2023, 12, 2)),
            make_event("r3", "rent", "debit", Decimal("5148"), date(2024, 1, 2)),
        ]
        series = detect_recurring(events, as_of=date(2024, 1, 3))
        s = find_series(series, "rent", "debit")
        self.assertIsNotNone(s)
        self.assertTrue(s.is_fixed)
        self.assertEqual(s.amount, Decimal("5148"))
        self.assertEqual(s.cadence_days, CADENCE_MONTHLY)
        self.assertEqual(s.anchor, date(2024, 1, 2))
        self.assertEqual(s.source_event_id, "r3")


class DetectSalaryTests(unittest.TestCase):
    def test_salary_from_one_settled_and_one_scheduled(self):
        events = [
            make_event(
                "s0", "salary", "credit", Decimal("12826"), date(2024, 2, 15),
                status="settled",
            ),
            make_event(
                "s1", "salary", "credit", Decimal("23320"), date(2024, 3, 15),
                status="scheduled",
            ),
        ]
        series = detect_recurring(events, as_of=date(2024, 3, 3))
        s = find_series(series, "salary", "credit")
        self.assertIsNotNone(s)
        self.assertEqual(s.anchor, date(2024, 3, 15))
        self.assertEqual(s.amount, Decimal("23320"))
        self.assertEqual(s.source_event_id, "s1")
        self.assertFalse(s.is_fixed)

        flows = project(s, start=date(2024, 3, 3))
        self.assertTrue(flows)
        self.assertEqual(flows[0].when, date(2024, 4, 15))
        self.assertEqual(flows[0].delta, Decimal("23320"))
        self.assertTrue(all(f.when != date(2024, 3, 15) for f in flows))


class MonthlyClampingTests(unittest.TestCase):
    def test_day_of_month_clamps_across_february(self):
        series = RecurringSeries(
            user_id="user_01",
            category="rent",
            direction="debit",
            cadence_days=CADENCE_MONTHLY,
            amount=Decimal("100"),
            anchor=date(2024, 1, 31),
            source_event_id="x",
            flexibility="fixed",
            minimum_allowed_amount=None,
            is_fixed=True,
            event_ids=("x",),
        )
        flows = project(series, start=date(2024, 1, 31), days=120)
        dates = [f.when for f in flows]
        self.assertIn(date(2024, 2, 29), dates)
        self.assertIn(date(2024, 3, 31), dates)
        self.assertIn(date(2024, 4, 30), dates)


class IrregularAndInsufficientHistoryTests(unittest.TestCase):
    def test_irregular_gaps_yield_no_series(self):
        events = [
            make_event("i0", "dining", "debit", Decimal("100"), date(2024, 1, 1)),
            make_event("i1", "dining", "debit", Decimal("100"), date(2024, 1, 5)),
            make_event("i2", "dining", "debit", Decimal("100"), date(2024, 2, 20)),
            make_event("i3", "dining", "debit", Decimal("100"), date(2024, 2, 21)),
            make_event("i4", "dining", "debit", Decimal("100"), date(2024, 4, 30)),
        ]
        series = detect_recurring(events, as_of=date(2024, 5, 1))
        self.assertIsNone(find_series(series, "dining", "debit"))

    def test_single_occurrence_yields_no_series(self):
        events = [
            make_event("o0", "fuel", "debit", Decimal("500"), date(2024, 1, 1)),
        ]
        series = detect_recurring(events, as_of=date(2024, 1, 15))
        self.assertEqual(series, ())

    def test_below_min_occurrences_after_same_day_merge(self):
        # Two rows on the same day merge into a single occurrence, which is
        # below MIN_OCCURRENCES (2), so no series should form.
        self.assertEqual(MIN_OCCURRENCES, 2)
        events = [
            make_event("m0", "fuel", "debit", Decimal("300"), date(2024, 1, 1)),
            make_event("m1", "fuel", "debit", Decimal("200"), date(2024, 1, 1)),
        ]
        series = detect_recurring(events, as_of=date(2024, 1, 2))
        self.assertIsNone(find_series(series, "fuel", "debit"))


class StaleSeriesTests(unittest.TestCase):
    def test_stale_series_is_dropped(self):
        events = [
            make_event("st0", "subscriptions", "debit", Decimal("50"), date(2023, 1, 1)),
            make_event("st1", "subscriptions", "debit", Decimal("50"), date(2023, 2, 1)),
            make_event("st2", "subscriptions", "debit", Decimal("50"), date(2023, 3, 1)),
        ]
        # cadence_days = 30 (monthly); threshold = 2*30+7 = 67 days.
        as_of_fresh = date(2023, 4, 1)  # 31 days after anchor: kept
        as_of_stale = date(2023, 6, 15)  # 106 days after anchor: dropped
        fresh = detect_recurring(events, as_of=as_of_fresh)
        stale = detect_recurring(events, as_of=as_of_stale)
        self.assertIsNotNone(find_series(fresh, "subscriptions", "debit"))
        self.assertIsNone(find_series(stale, "subscriptions", "debit"))


class NonSalaryCreditTests(unittest.TestCase):
    def test_non_salary_credit_never_forms_series(self):
        events = [
            make_event("rf0", "refund", "credit", Decimal("200"), date(2024, 1, 1)),
            make_event("rf1", "refund", "credit", Decimal("200"), date(2024, 2, 1)),
            make_event("rf2", "refund", "credit", Decimal("200"), date(2024, 3, 1)),
        ]
        series = detect_recurring(events, as_of=date(2024, 3, 5))
        self.assertEqual(series, ())


class PendingExclusionTests(unittest.TestCase):
    def test_pending_rows_excluded_from_history(self):
        events = [
            make_event("p0", "fuel", "debit", Decimal("400"), date(2024, 1, 1), status="settled"),
            make_event("p1", "fuel", "debit", Decimal("400"), date(2024, 2, 1), status="settled"),
            make_event("p2", "fuel", "debit", Decimal("400"), date(2024, 3, 1), status="pending"),
        ]
        series = detect_recurring(events, as_of=date(2024, 3, 5))
        s = find_series(series, "fuel", "debit")
        self.assertIsNotNone(s)
        # Anchor must be the last settled/scheduled row, not the pending one.
        self.assertEqual(s.anchor, date(2024, 2, 1))
        self.assertNotIn("p2", s.event_ids)

    def test_defensive_skip_of_non_cash_and_none_amount(self):
        events = [
            make_event("n0", "fuel", "debit", Decimal("400"), date(2024, 1, 1)),
            make_event("n1", "fuel", "debit", Decimal("400"), date(2024, 2, 1)),
            make_event("n2", "fuel", "debit", None, date(2024, 3, 1)),
            make_event(
                "n3", "investment_valuation", "non_cash", Decimal("999"),
                date(2024, 3, 1), status="unrealized",
            ),
        ]
        series = detect_recurring(events, as_of=date(2024, 3, 5))
        s = find_series(series, "fuel", "debit")
        self.assertIsNotNone(s)
        self.assertEqual(s.anchor, date(2024, 2, 1))


class WindowBoundsTests(unittest.TestCase):
    def test_inclusive_start_and_end_exclusive_beyond(self):
        series = RecurringSeries(
            user_id="user_01",
            category="fuel",
            direction="debit",
            cadence_days=CADENCE_WEEKLY,
            amount=Decimal("100"),
            anchor=date(2023, 12, 18),
            source_event_id="x",
            flexibility="reducible",
            minimum_allowed_amount=None,
            is_fixed=True,
            event_ids=("x",),
        )
        flows = project(series, start=date(2024, 1, 1), days=14)
        dates = [f.when for f in flows]
        # anchor + 7 = 2023-12-25 is before start: skipped, but stepping
        # continues past it.
        self.assertNotIn(date(2023, 12, 25), dates)
        # anchor + 14 = 2024-01-01 == start: inclusive.
        self.assertIn(date(2024, 1, 1), dates)
        # anchor + 21 = 2024-01-08: inside window.
        self.assertIn(date(2024, 1, 8), dates)
        # window end = start + 14 - 1 = 2024-01-14; anchor + 28 = 2024-01-15
        # is one day beyond: excluded.
        self.assertNotIn(date(2024, 1, 15), dates)
        self.assertEqual(dates, sorted(dates))

    def test_anchor_itself_never_emitted(self):
        series = RecurringSeries(
            user_id="user_01",
            category="fuel",
            direction="debit",
            cadence_days=CADENCE_WEEKLY,
            amount=Decimal("100"),
            anchor=date(2024, 1, 1),
            source_event_id="x",
            flexibility="reducible",
            minimum_allowed_amount=None,
            is_fixed=True,
            event_ids=("x",),
        )
        flows = project(series, start=date(2024, 1, 1), days=FORECAST_DAYS)
        self.assertNotIn(date(2024, 1, 1), [f.when for f in flows])


class DeterministicOrderingTests(unittest.TestCase):
    def test_detect_recurring_sorted_by_direction_then_category(self):
        events = [
            make_event("z0", "utilities", "debit", Decimal("100"), date(2024, 1, 1)),
            make_event("z1", "utilities", "debit", Decimal("100"), date(2024, 2, 1)),
            make_event("a0", "rent", "debit", Decimal("100"), date(2024, 1, 1)),
            make_event("a1", "rent", "debit", Decimal("100"), date(2024, 2, 1)),
            make_event("c0", "salary", "credit", Decimal("100"), date(2024, 1, 1)),
            make_event("c1", "salary", "credit", Decimal("100"), date(2024, 2, 1)),
        ]
        series = detect_recurring(events, as_of=date(2024, 2, 5))
        keys = [(s.direction, s.category) for s in series]
        self.assertEqual(keys, sorted(keys))

    def test_project_all_sorted_by_when_label_source(self):
        rent = RecurringSeries(
            user_id="user_01", category="rent", direction="debit",
            cadence_days=CADENCE_MONTHLY, amount=Decimal("100"),
            anchor=date(2024, 1, 1), source_event_id="r", flexibility="fixed",
            minimum_allowed_amount=None, is_fixed=True, event_ids=("r",),
        )
        fuel = RecurringSeries(
            user_id="user_01", category="fuel", direction="debit",
            cadence_days=CADENCE_WEEKLY, amount=Decimal("50"),
            anchor=date(2024, 1, 1), source_event_id="f", flexibility="reducible",
            minimum_allowed_amount=None, is_fixed=True, event_ids=("f",),
        )
        flows = project_all([rent, fuel], start=date(2024, 1, 1), days=40)
        whens = [f.when for f in flows]
        self.assertEqual(whens, sorted(whens))
        # Stable tie-break by (when, label, source_event_id) when same date.
        keyed = [(f.when, f.label, f.source_event_id) for f in flows]
        self.assertEqual(keyed, sorted(keyed))


class SameDayMergeTests(unittest.TestCase):
    def test_same_day_events_merge_for_gap_analysis(self):
        # Two debits on the same day count as one occurrence; combined with
        # three more monthly occurrences that gives 4 occurrences total,
        # satisfying MIN_OCCURRENCES with a clean monthly cadence.
        events = [
            make_event("d0", "shopping", "debit", Decimal("100"), date(2024, 1, 1)),
            make_event("d1", "shopping", "debit", Decimal("50"), date(2024, 1, 1)),
            make_event("d2", "shopping", "debit", Decimal("150"), date(2024, 2, 1)),
            make_event("d3", "shopping", "debit", Decimal("150"), date(2024, 3, 1)),
        ]
        series = detect_recurring(events, as_of=date(2024, 3, 5))
        s = find_series(series, "shopping", "debit")
        self.assertIsNotNone(s)
        self.assertEqual(s.cadence_days, CADENCE_MONTHLY)
        self.assertEqual(len(s.event_ids), 4)
        # Same-day members are summed into one occurrence amount, so the
        # series is fixed at 150 rather than variable over [100, 50, 150, 150].
        self.assertTrue(s.is_fixed)
        self.assertEqual(s.amount, Decimal("150"))
        # Representative of the merged day is the last member by event_id.
        self.assertEqual(s.anchor, date(2024, 3, 1))


class LabelAndFlowShapeTests(unittest.TestCase):
    def test_flow_fields_match_series(self):
        series = RecurringSeries(
            user_id="user_01", category="groceries", direction="debit",
            cadence_days=CADENCE_WEEKLY, amount=Decimal("700"),
            anchor=date(2024, 1, 1), source_event_id="g", flexibility="fixed",
            minimum_allowed_amount=None, is_fixed=True, event_ids=("g",),
        )
        flows = project(series, start=date(2024, 1, 1), days=10)
        self.assertTrue(flows)
        flow = flows[0]
        self.assertIsInstance(flow, CashFlow)
        self.assertEqual(flow.delta, Decimal("-700"))
        self.assertEqual(flow.label, flow_label("groceries", "weekly"))
        self.assertEqual(flow.source_event_id, "g")
        self.assertTrue(flow.is_recurring)

    def test_credit_flow_delta_is_positive(self):
        series = RecurringSeries(
            user_id="user_01", category="salary", direction="credit",
            cadence_days=CADENCE_MONTHLY, amount=Decimal("20000"),
            anchor=date(2024, 1, 15), source_event_id="s", flexibility="fixed",
            minimum_allowed_amount=None, is_fixed=True, event_ids=("s",),
        )
        flows = project(series, start=date(2024, 1, 15), days=40)
        self.assertTrue(all(f.delta == Decimal("20000") for f in flows))


class RealDatasetTests(unittest.TestCase):
    def test_user_01_rent_salary_groceries_series(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        dataset_dir = os.path.join(repo_root, "dataset")
        if not os.path.isdir(dataset_dir):
            raise unittest.SkipTest("dataset/ directory not present")

        sys.path.insert(0, os.path.join(repo_root, "code"))
        from loaders import load_dataset  # noqa: E402

        data = load_dataset(dataset_dir)
        user_events = data.events_by_user.get("user_01", ())
        history = tuple(
            e
            for e in user_events
            if e.amount is not None
            and e.is_cash
            and e.status in ("settled", "scheduled")
            and e.currency == "ZAR"
        )

        as_of = date(2024, 3, 3)
        series = detect_recurring(history, as_of=as_of)

        rent = find_series(series, "rent", "debit")
        self.assertIsNotNone(rent)
        self.assertTrue(rent.is_fixed)
        self.assertEqual(rent.amount, Decimal("5148"))
        self.assertEqual(rent.cadence_days, CADENCE_MONTHLY)

        salary = find_series(series, "salary", "credit")
        self.assertIsNotNone(salary)
        self.assertEqual(salary.anchor, date(2024, 3, 15))
        self.assertEqual(salary.amount, Decimal("23320"))
        self.assertEqual(salary.source_event_id, "event_103")

        groceries = find_series(series, "groceries", "debit")
        self.assertIsNotNone(groceries)
        self.assertEqual(groceries.cadence_days, CADENCE_WEEKLY)

        flows = project_all(series, start=date(2024, 3, 3))
        salary_flow_dates = {
            f.when for f in flows if flow_label("salary", "monthly") == f.label
        }
        self.assertNotIn(date(2024, 3, 15), salary_flow_dates)
        self.assertIn(date(2024, 4, 15), salary_flow_dates)
        self.assertIn(date(2024, 5, 15), salary_flow_dates)


if __name__ == "__main__":
    unittest.main()


class InterleavedSalaryStreamsTest(unittest.TestCase):
    """Base pay on the 15th plus commission on the 24th: merged, the gaps
    alternate 9/21 days and look irregular; per-description fallback must
    still recover the base-pay series (and only that one when the second
    stream is itself irregular)."""

    def _events(self):
        events = []
        for i, month in enumerate((1, 2, 3, 4)):
            events.append(
                make_event(
                    f"base_{i}", "salary", "credit", Decimal("23256000"),
                    date(2025, month, 15), event_type="income",
                    description="Base salary",
                )
            )
        events.append(
            make_event(
                "comm_a", "salary", "credit", Decimal("15989420"),
                date(2025, 2, 24), event_type="income",
                description="Performance commission",
            )
        )
        events.append(
            make_event(
                "comm_b", "salary", "credit", Decimal("8502888.2"),
                date(2025, 4, 24), event_type="income",
                description="Monthly sales commission",
            )
        )
        return events

    def test_base_pay_recovered_from_description_fallback(self):
        series = detect_recurring(self._events(), as_of=date(2025, 5, 3))
        salary = [s for s in series if s.category == "salary"]
        self.assertEqual(len(salary), 1)
        self.assertEqual(salary[0].amount, Decimal("23256000"))
        self.assertEqual(salary[0].anchor, date(2025, 4, 15))
        self.assertEqual(salary[0].cadence_days, CADENCE_MONTHLY)
        self.assertEqual(salary[0].event_ids, ("base_0", "base_1", "base_2", "base_3"))

    def test_fallback_is_income_only(self):
        # Same interleaving on a debit category must NOT be split by
        # description: variable spending legitimately spans many descriptions.
        events = []
        for i, month in enumerate((1, 2, 3, 4)):
            events.append(make_event(f"a_{i}", "dining", "debit", Decimal("100"),
                                     date(2025, month, 15), description="Lunch"))
        for i, month in enumerate((2, 4)):
            events.append(make_event(f"b_{i}", "dining", "debit", Decimal("50"),
                                     date(2025, month, 24), description="Dinner"))
        series = detect_recurring(events, as_of=date(2025, 5, 3))
        self.assertEqual([s for s in series if s.category == "dining"], [])


class ConfirmedIncomeTest(unittest.TestCase):
    def _salary(self, amounts, descriptions=None, last_status="settled"):
        events = []
        for i, amt in enumerate(amounts):
            events.append(
                make_event(
                    f"s_{i}", "salary", "credit", Decimal(amt), date(2025, i + 1, 15),
                    event_type="income",
                    description=(descriptions[i] if descriptions else "Payroll credit"),
                    status=last_status if i == len(amounts) - 1 else "settled",
                )
            )
        return events

    def test_variable_income_is_not_projected(self):
        series = detect_recurring(self._salary(["500", "610", "455", "700"]), as_of=date(2025, 4, 20))
        self.assertEqual(series, ())

    def test_one_off_reduced_payslip_keeps_series(self):
        series = detect_recurring(self._salary(["1422.85"] * 4 + ["782.57"]), as_of=date(2025, 5, 20))
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].amount, Decimal("782.57"))

    def test_scheduled_raise_wins_and_does_not_break_confirmation(self):
        series = detect_recurring(
            self._salary(["12826", "23320"], last_status="scheduled"), as_of=date(2025, 2, 3)
        )
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].amount, Decimal("23320"))
        self.assertEqual(series[0].anchor, date(2025, 2, 15))

    def test_final_payslip_ends_income(self):
        events = self._salary(
            ["14740"] * 4,
            descriptions=["Payroll credit"] * 3 + ["Final employer payroll"],
        )
        self.assertEqual(detect_recurring(events, as_of=date(2025, 4, 20)), ())
