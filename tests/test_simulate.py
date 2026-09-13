import os
import random
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from contracts import CashFlow, Forecast, Payment  # noqa: E402
from forecast.simulate import (  # noqa: E402
    ConcreteForecast,
    day_index,
    earliest_full_payment_date,
    is_safe,
    safe_amount_on,
    simulate,
    trough,
    with_payments,
)


class TestSimulateBasics(unittest.TestCase):
    """balances length; day-0 inclusion; last-day inclusion; start+days and
    before-start exclusion; mixed credit/debit cumulative sums; sort order;
    ConcreteForecast typing."""

    def setUp(self):
        self.start = date(2024, 1, 1)
        self.flows = [
            CashFlow(self.start, Decimal("10"), "salary|monthly", "e1", True),
            CashFlow(date(2024, 1, 5), Decimal("-20"), "rent|monthly", "e2", True),
            # before start -- must be dropped
            CashFlow(date(2023, 12, 31), Decimal("-1000"), "oops|before", "e3", False),
            # exactly start + days (index 5, out of [0, 5)) -- must be dropped
            CashFlow(date(2024, 1, 6), Decimal("1000"), "oops|after", "e4", False),
        ]
        self.forecast = simulate("u1", Decimal("100"), self.flows, self.start, days=5)

    def test_balances_length_equals_days(self):
        self.assertEqual(len(self.forecast.balances), 5)

    def test_day_zero_flow_counted(self):
        self.assertEqual(self.forecast.balances[0], Decimal("110"))

    def test_last_day_flow_counted_and_out_of_window_flows_dropped(self):
        # 100 +10 (day0) held flat through day3, -20 lands on day4 (last day)
        self.assertEqual(
            self.forecast.balances,
            (
                Decimal("110"),
                Decimal("110"),
                Decimal("110"),
                Decimal("110"),
                Decimal("90"),
            ),
        )

    def test_out_of_window_flows_excluded_from_stored_flows(self):
        # only the two in-window flows survive, sorted by (when, label, id)
        self.assertEqual(len(self.forecast.flows), 2)
        self.assertEqual(self.forecast.flows[0].source_event_id, "e1")
        self.assertEqual(self.forecast.flows[1].source_event_id, "e2")

    def test_returns_concrete_forecast(self):
        self.assertIsInstance(self.forecast, ConcreteForecast)
        self.assertIsInstance(self.forecast, Forecast)


class TestTroughAndBalanceOn(unittest.TestCase):
    """trough() over the whole window vs from a later day; balance_on();
    day_index() boundaries and out-of-range errors; ConcreteForecast method
    overrides (not just the module-level functions)."""

    def setUp(self):
        self.start = date(2024, 1, 1)
        # dip on day1 (Jan2), big rise on day3 (Jan4)
        flows = [
            CashFlow(date(2024, 1, 2), Decimal("-50"), "fuel|pending", "e1", False),
            CashFlow(date(2024, 1, 4), Decimal("200"), "salary|monthly", "e2", True),
        ]
        self.forecast = simulate("u2", Decimal("100"), flows, self.start, days=5)
        # hand-computed: 100, 100-50=50, 50, 50+200=250, 250
        self.assertEqual(
            self.forecast.balances,
            (
                Decimal("100"),
                Decimal("50"),
                Decimal("50"),
                Decimal("250"),
                Decimal("250"),
            ),
        )

    def test_day_index_boundaries(self):
        self.assertEqual(day_index(self.forecast, date(2024, 1, 1)), 0)
        self.assertEqual(day_index(self.forecast, date(2024, 1, 5)), 4)

    def test_day_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            day_index(self.forecast, date(2023, 12, 31))
        with self.assertRaises(ValueError):
            day_index(self.forecast, date(2024, 1, 6))

    def test_trough_whole_window(self):
        self.assertEqual(trough(self.forecast), Decimal("50"))

    def test_trough_from_later_day_differs_from_whole_window(self):
        self.assertEqual(trough(self.forecast, frm=date(2024, 1, 4)), Decimal("250"))

    def test_balance_on(self):
        self.assertEqual(self.forecast.balance_on(date(2024, 1, 1)), Decimal("100"))
        self.assertEqual(self.forecast.balance_on(date(2024, 1, 2)), Decimal("50"))
        self.assertEqual(self.forecast.balance_on(date(2024, 1, 4)), Decimal("250"))

    def test_concrete_forecast_trough_method_matches_module_function(self):
        self.assertEqual(self.forecast.trough(), trough(self.forecast))
        self.assertEqual(
            self.forecast.trough(frm=date(2024, 1, 4)),
            trough(self.forecast, frm=date(2024, 1, 4)),
        )


class TestSafeAmountOn(unittest.TestCase):
    """floor at 0, cap at `cap`, exact trough-minus-minimum in range;
    cap <= 0 short-circuits to 0."""

    def setUp(self):
        self.start = date(2024, 1, 1)
        flows = [
            CashFlow(date(2024, 1, 2), Decimal("-50"), "fuel|pending", "e1", False),
            CashFlow(date(2024, 1, 4), Decimal("200"), "salary|monthly", "e2", True),
        ]
        self.forecast = simulate("u2", Decimal("100"), flows, self.start, days=5)
        # balances: 100, 50, 50, 250, 250 ; whole-window trough = 50

    def test_equal_to_trough_minus_minimum_when_in_range(self):
        # trough(frm=start)=50, minimum=30 -> headroom 20, cap 1000 (not binding)
        self.assertEqual(
            safe_amount_on(self.forecast, self.start, Decimal("30"), Decimal("1000")),
            Decimal("20"),
        )

    def test_capped_at_cap(self):
        self.assertEqual(
            safe_amount_on(self.forecast, self.start, Decimal("30"), Decimal("10")),
            Decimal("10"),
        )

    def test_floored_at_zero(self):
        # trough 50, minimum 80 -> headroom -30
        self.assertEqual(
            safe_amount_on(self.forecast, self.start, Decimal("80"), Decimal("1000")),
            Decimal("0"),
        )

    def test_cap_le_zero_returns_zero(self):
        self.assertEqual(
            safe_amount_on(self.forecast, self.start, Decimal("0"), Decimal("0")),
            Decimal("0"),
        )
        self.assertEqual(
            safe_amount_on(self.forecast, self.start, Decimal("0"), Decimal("-5")),
            Decimal("0"),
        )

    def test_uses_trough_from_the_given_day_not_whole_window(self):
        # trough(frm=Jan4) = 250, minimum 30 -> headroom 220
        self.assertEqual(
            safe_amount_on(
                self.forecast, date(2024, 1, 4), Decimal("30"), Decimal("1000")
            ),
            Decimal("220"),
        )


class TestEarliestFullPaymentDate(unittest.TestCase):
    def test_affordable_now_returns_start(self):
        start = date(2024, 1, 1)
        forecast = simulate("a", Decimal("200"), [], start, days=5)
        self.assertEqual(
            earliest_full_payment_date(forecast, Decimal("100"), Decimal("50")), start
        )

    def test_never_affordable_returns_none(self):
        start = date(2024, 1, 1)
        forecast = simulate("b", Decimal("100"), [], start, days=5)
        # threshold = 90 + 50 = 140 ; balances are flat at 100
        self.assertIsNone(
            earliest_full_payment_date(forecast, Decimal("50"), Decimal("90"))
        )

    def test_lands_on_salary_day_and_safe_amount_grows_after_it(self):
        start = date(2024, 1, 1)
        salary_day = date(2024, 1, 5)
        flows = [CashFlow(salary_day, Decimal("200"), "salary|monthly", "e1", True)]
        forecast = simulate("c", Decimal("100"), flows, start, days=10)
        # balances: 100,100,100,100,300,300,300,300,300,300
        self.assertEqual(
            forecast.balances,
            tuple(
                Decimal(v)
                for v in [100, 100, 100, 100, 300, 300, 300, 300, 300, 300]
            ),
        )
        minimum = Decimal("50")
        amount = Decimal("220")
        # before salary: trough(frm=Jan4)=100, headroom capped at 50 < amount
        self.assertEqual(
            safe_amount_on(forecast, date(2024, 1, 4), minimum, amount),
            Decimal("50"),
        )
        # on salary day: trough(frm=Jan5)=300, headroom 250 capped at amount 220
        self.assertEqual(
            safe_amount_on(forecast, salary_day, minimum, amount), Decimal("220")
        )
        self.assertEqual(
            earliest_full_payment_date(forecast, amount, minimum), salary_day
        )

    def test_amount_le_zero_returns_start(self):
        start = date(2024, 1, 1)
        flows = [CashFlow(date(2024, 1, 5), Decimal("200"), "salary|monthly", "e1", True)]
        forecast = simulate("c", Decimal("100"), flows, start, days=10)
        self.assertEqual(
            earliest_full_payment_date(forecast, Decimal("0"), Decimal("50")), start
        )
        self.assertEqual(
            earliest_full_payment_date(forecast, Decimal("-10"), Decimal("50")), start
        )


class TestWithPayments(unittest.TestCase):
    def test_reduces_from_payment_day_leaves_earlier_days_unchanged(self):
        start = date(2024, 1, 1)
        forecast = simulate("a", Decimal("200"), [], start, days=5)
        new_forecast = with_payments(
            forecast, [Payment(date(2024, 1, 3), Decimal("50"))]
        )
        self.assertEqual(
            new_forecast.balances,
            tuple(Decimal(v) for v in [200, 200, 150, 150, 150]),
        )
        # original untouched
        self.assertEqual(
            forecast.balances, tuple(Decimal(v) for v in [200, 200, 200, 200, 200])
        )
        self.assertIsInstance(new_forecast, ConcreteForecast)

    def test_combines_with_existing_flows(self):
        start = date(2024, 1, 1)
        flows = [
            CashFlow(date(2024, 1, 2), Decimal("-50"), "fuel|pending", "e1", False),
            CashFlow(date(2024, 1, 4), Decimal("200"), "salary|monthly", "e2", True),
        ]
        forecast = simulate("u2", Decimal("100"), flows, start, days=5)
        new_forecast = with_payments(forecast, [Payment(start, Decimal("30"))])
        # day_deltas: idx0 -30(payment), idx1 -50(orig), idx3 +200(orig)
        # running: 70, 20, 20, 220, 220
        self.assertEqual(
            new_forecast.balances,
            tuple(Decimal(v) for v in [70, 20, 20, 220, 220]),
        )
        # original unaffected
        self.assertEqual(
            forecast.balances, tuple(Decimal(v) for v in [100, 50, 50, 250, 250])
        )

    def test_payment_outside_window_is_ignored(self):
        start = date(2024, 1, 1)
        forecast = simulate("a", Decimal("200"), [], start, days=5)
        new_forecast = with_payments(
            forecast, [Payment(date(2024, 1, 10), Decimal("999"))]
        )
        self.assertEqual(
            new_forecast.balances, tuple(Decimal(v) for v in [200, 200, 200, 200, 200])
        )


class TestIsSafe(unittest.TestCase):
    def setUp(self):
        start = date(2024, 1, 1)
        flows = [
            CashFlow(date(2024, 1, 2), Decimal("-50"), "fuel|pending", "e1", False),
            CashFlow(date(2024, 1, 4), Decimal("200"), "salary|monthly", "e2", True),
        ]
        self.forecast = simulate("u2", Decimal("100"), flows, start, days=5)
        # balances: 100, 50, 50, 250, 250

    def test_safe_whole_window(self):
        self.assertTrue(is_safe(self.forecast, Decimal("30")))

    def test_unsafe_whole_window(self):
        self.assertFalse(is_safe(self.forecast, Decimal("60")))

    def test_safe_from_a_later_day(self):
        self.assertTrue(is_safe(self.forecast, Decimal("60"), frm=date(2024, 1, 4)))


class TestSuffixMinMatchesBruteForce(unittest.TestCase):
    """The O(days) suffix-minimum implementation of earliest_full_payment_date
    must agree with a per-day brute-force scan (which calls safe_amount_on,
    and therefore trough, once per candidate day) on a large randomised
    fixture."""

    def test_matches_brute_force_on_randomised_fixture(self):
        rng = random.Random(42)
        start = date(2024, 1, 1)
        days = 90
        labels = ["groceries|weekly", "salary|monthly", "rent|monthly", "fuel|pending"]
        flows = []
        for i in range(200):
            offset = rng.randint(-10, 100)  # some land outside the window
            when = start + timedelta(days=offset)
            amount = Decimal(str(rng.randint(-500, 500)))
            label = rng.choice(labels)
            source_event_id = f"e{i}" if rng.random() > 0.3 else None
            flows.append(
                CashFlow(
                    when=when,
                    delta=amount,
                    label=label,
                    source_event_id=source_event_id,
                    is_recurring=rng.random() > 0.5,
                )
            )
        forecast = simulate("rand_user", Decimal("5000"), flows, start, days=days)

        def brute_force_earliest(amount: Decimal, minimum: Decimal):
            if amount <= 0:
                return forecast.start
            for i in range(len(forecast.balances)):
                when = forecast.start + timedelta(days=i)
                if safe_amount_on(forecast, when, minimum, amount) >= amount:
                    return when
            return None

        cases = [
            (Decimal("100"), Decimal("0")),
            (Decimal("1000"), Decimal("500")),
            (Decimal("6000"), Decimal("1000")),
            (Decimal("0"), Decimal("100")),
            (Decimal("-50"), Decimal("100")),
            (Decimal("20000"), Decimal("0")),
        ]
        for amount, minimum in cases:
            with self.subTest(amount=amount, minimum=minimum):
                self.assertEqual(
                    earliest_full_payment_date(forecast, amount, minimum),
                    brute_force_earliest(amount, minimum),
                )


class TestUser01ShapedScenario(unittest.TestCase):
    """Brief-mandated fixture. Hand-computed trough derivation (chronological
    running balance from opening=58481.1):

        Mar03  58481.1  - 567.6 (fuel)      = 57913.5   <- day 0
        Mar06  57913.5  - 761   (groceries) = 57152.5
        Mar13  57152.5  - 761   (groceries) = 56391.5   <- global trough
        Mar15  56391.5  + 23320 (salary)    = 79711.5
        Mar20  79711.5  - 761                = 78950.5
        Mar27  78950.5  - 761                = 78189.5
        Apr01  78189.5  - 5148  (rent)       = 73041.5
        Apr03  73041.5  - 761                = 72280.5
        Apr10  72280.5  - 761                = 71519.5
        Apr15  71519.5  + 23320              = 94839.5
        Apr17  94839.5  - 761                = 94078.5
        Apr24  94078.5  - 761                = 93317.5
        May01  93317.5  - 5148 - 761         = 87408.5
        May08  87408.5  - 761                = 86647.5
        May15  86647.5  + 23320 - 761        = 109206.5
        May22  109206.5 - 761                = 108445.5
        May29  108445.5 - 761                = 107684.5

    Every subsequent monthly leg (salary +23320 against rent 5148 + four
    weekly groceries debits of 761 = 8192 total outflow) nets positive, so
    the balance only grows once past 2024-03-15; 56391.5 (2024-03-13, the
    day before the first salary credit) is therefore the global minimum
    over the whole 90-day window [2024-03-03, 2024-05-31].
    """

    def test_trough_safe_amount_and_earliest_date(self):
        opening = Decimal("58481.1")
        minimum = Decimal("18000")
        start = date(2024, 3, 3)

        flows = [
            CashFlow(date(2024, 3, 3), Decimal("-567.6"), "fuel|pending", "e_fuel", False),
            CashFlow(date(2024, 4, 1), Decimal("-5148"), "rent|monthly", "e_rent", True),
            CashFlow(date(2024, 5, 1), Decimal("-5148"), "rent|monthly", "e_rent", True),
            CashFlow(date(2024, 3, 15), Decimal("23320"), "salary|monthly", "e_sal", True),
            CashFlow(date(2024, 4, 15), Decimal("23320"), "salary|monthly", "e_sal", True),
            CashFlow(date(2024, 5, 15), Decimal("23320"), "salary|monthly", "e_sal", True),
        ]
        groceries_dates = [date(2024, 3, 6) + timedelta(days=7 * k) for k in range(13)]
        # window is [2024-03-03, 2024-05-31] (89 days after start); the 13th
        # occurrence (k=12) is 2024-05-29 (in window); the 14th would be
        # 2024-06-05 (out of window) and is deliberately not included here.
        self.assertEqual(groceries_dates[-1], date(2024, 5, 29))
        self.assertEqual(start + timedelta(days=89), date(2024, 5, 31))
        flows += [
            CashFlow(d, Decimal("-761"), "groceries|weekly", "e_groc", True)
            for d in groceries_dates
        ]

        forecast = simulate("user_01", opening, flows, start)  # default days=90
        self.assertEqual(len(forecast.balances), 90)

        self.assertEqual(trough(forecast), Decimal("56391.5"))

        self.assertEqual(
            safe_amount_on(forecast, start, minimum, cap=Decimal("25256")),
            Decimal("25256"),
        )
        self.assertEqual(
            earliest_full_payment_date(forecast, Decimal("25256"), minimum), start
        )


if __name__ == "__main__":
    unittest.main()
