import os
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from contracts import (  # noqa: E402
    Amendment,
    CashFlow,
    Dataset,
    Event,
    ImageAmount,
    ImageRef,
    Message,
    Profile,
    RequestRecord,
)
from forecast.amendments import (  # noqa: E402
    MIN_CONFIDENCE,
    RATE_SEARCH_DAYS,
    apply_amendments,
    convert_amount,
    normalise_events,
    one_off_flows,
    relevant_amendments,
)


# --------------------------------------------------------------------------
# Test fixtures: a concrete Dataset (mirrors loaders._ConcreteDataset) and a
# stub ExtractionCache -- code/extraction.py is never imported here.
# --------------------------------------------------------------------------


class _TestDataset(Dataset):
    def convert(self, amount, frm, to, on):
        if frm == to:
            return amount
        direct = self.rates.get((on, frm, to))
        if direct is not None:
            return amount * direct
        inverse = self.rates.get((on, to, frm))
        if inverse is not None:
            return amount / inverse
        raise ValueError(f"no rate {frm}->{to} on {on}")


class StubCache:
    def __init__(self, amendments=None, images=None, amendment_errors=(), image_errors=()):
        self.amendments = amendments or {}
        self.images = images or {}
        self.amendment_errors = set(amendment_errors)
        self.image_errors = set(image_errors)

    def amendment_for(self, message):
        if message.message_id in self.amendment_errors:
            raise RuntimeError("extraction failed")
        return self.amendments[message.message_id]

    def amount_for(self, image):
        if image.image_id in self.image_errors:
            raise FileNotFoundError("missing image")
        return self.images[image.image_id]


def make_profile(**overrides):
    defaults = dict(
        user_id="u1",
        home_currency="USD",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("100"),
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=(),
        max_installment_months=None,
    )
    defaults.update(overrides)
    return Profile(**defaults)


def make_event(**overrides):
    defaults = dict(
        event_id="e1",
        user_id="u1",
        event_type="expense",
        description="test",
        category="groceries",
        direction="debit",
        amount=Decimal("50"),
        currency="USD",
        event_date=date(2026, 1, 1),
        settlement_date=date(2026, 1, 1),
        status="settled",
        linked_event_id=None,
        flexibility="reducible",
        minimum_allowed_amount=None,
    )
    defaults.update(overrides)
    return Event(**defaults)


def make_dataset(**overrides):
    defaults = dict(
        profiles={},
        events_by_user={},
        events_by_id={},
        options_by_request={},
        messages_by_user={},
        images_by_event={},
        rates={},
    )
    defaults.update(overrides)
    return _TestDataset(**defaults)


def make_message(**overrides):
    defaults = dict(
        message_id="m1",
        user_id="u1",
        request_id=None,
        related_event_id=None,
        sent_at="2026-01-01T00:00:00",
        source_type="chat",
        message_text="ignored -- untrusted",
    )
    defaults.update(overrides)
    return Message(**defaults)


def make_amendment(**overrides):
    defaults = dict(
        message_id="m1",
        user_id="u1",
        kind="no_op",
        effective_date=None,
        new_amount=None,
        pct_change=None,
        currency=None,
        related_event_id=None,
        confidence=0.9,
    )
    defaults.update(overrides)
    return Amendment(**defaults)


def make_request(**overrides):
    defaults = dict(
        request_id="r1",
        user_id="u1",
        request_date=date(2026, 1, 1),
        request_type="purchase",
        requested_amount=Decimal("100"),
        desired_completion_date=date(2026, 2, 1),
        allows_partial_payment=True,
        request_text="buy stuff",
    )
    defaults.update(overrides)
    return RequestRecord(**defaults)


def make_flow(**overrides):
    defaults = dict(
        when=date(2026, 1, 1),
        delta=Decimal("100"),
        label="salary|monthly",
        source_event_id="e1",
        is_recurring=True,
    )
    defaults.update(overrides)
    return CashFlow(**defaults)


START = date(2026, 1, 1)


# --------------------------------------------------------------------------
# normalise_events
# --------------------------------------------------------------------------


class NormaliseEventsTests(unittest.TestCase):
    def test_drops_non_cash_and_unrealized(self):
        profile = make_profile()
        ev = make_event(direction="non_cash", status="unrealized", amount=Decimal("10"))
        out = normalise_events([ev], profile, make_dataset(), StubCache())
        self.assertEqual(out, ())

    def test_drops_failed_and_cancelled(self):
        profile = make_profile()
        evs = [
            make_event(event_id="f1", status="failed"),
            make_event(event_id="c1", status="cancelled"),
        ]
        out = normalise_events(evs, profile, make_dataset(), StubCache())
        self.assertEqual(out, ())

    def test_keeps_settled_pending_scheduled(self):
        profile = make_profile()
        evs = [
            make_event(event_id="s1", status="settled"),
            make_event(event_id="p1", status="pending"),
            make_event(event_id="sc1", status="scheduled"),
        ]
        out = normalise_events(evs, profile, make_dataset(), StubCache())
        self.assertEqual({e.event_id for e in out}, {"s1", "p1", "sc1"})

    def test_none_amount_resolved_from_image(self):
        profile = make_profile()
        ev = make_event(event_id="e1", amount=None, currency="USD")
        image = ImageRef(image_id="img1", user_id="u1", request_id=None, related_event_id="e1")
        data = make_dataset(images_by_event={"e1": image})
        cache = StubCache(images={"img1": ImageAmount(image_id="img1", amount=Decimal("42"), currency="USD", confidence=0.9)})
        out = normalise_events([ev], profile, data, cache)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].amount, Decimal("42"))

    def test_none_amount_no_image_dropped(self):
        profile = make_profile()
        ev = make_event(event_id="e1", amount=None)
        out = normalise_events([ev], profile, make_dataset(), StubCache())
        self.assertEqual(out, ())

    def test_image_cache_exception_drops_row(self):
        profile = make_profile()
        ev = make_event(event_id="e1", amount=None)
        image = ImageRef(image_id="img1", user_id="u1", request_id=None, related_event_id="e1")
        data = make_dataset(images_by_event={"e1": image})
        cache = StubCache(image_errors={"img1"})
        # Must not raise.
        out = normalise_events([ev], profile, data, cache)
        self.assertEqual(out, ())

    def test_fx_on_settlement_date(self):
        profile = make_profile(home_currency="USD")
        ev = make_event(
            amount=Decimal("50"),
            currency="EUR",
            event_date=date(2026, 1, 5),
            settlement_date=date(2026, 1, 10),
        )
        data = make_dataset(rates={(date(2026, 1, 10), "EUR", "USD"): Decimal("1.1")})
        out = normalise_events([ev], profile, data, StubCache())
        self.assertEqual(out[0].amount, Decimal("55.0"))
        self.assertEqual(out[0].currency, "USD")

    def test_fx_falls_back_to_event_date(self):
        profile = make_profile(home_currency="USD")
        ev = make_event(
            amount=Decimal("50"),
            currency="EUR",
            event_date=date(2026, 1, 5),
            settlement_date=date(2026, 1, 10),
        )
        # No rate on settlement_date; only on event_date.
        data = make_dataset(rates={(date(2026, 1, 5), "EUR", "USD"): Decimal("1.2")})
        out = normalise_events([ev], profile, data, StubCache())
        self.assertEqual(out[0].amount, Decimal("60.0"))

    def test_fx_nearest_date_search(self):
        profile = make_profile(home_currency="USD")
        ev = make_event(
            amount=Decimal("50"),
            currency="EUR",
            event_date=date(2026, 1, 5),
            settlement_date=date(2026, 3, 1),
        )
        # No exact rate on settlement_date (Mar 1) or event_date (Jan 5), but
        # a rate 4 days after event_date -- far outside settlement_date's
        # search window, well inside event_date's.
        data = make_dataset(rates={(date(2026, 1, 9), "EUR", "USD"): Decimal("1.5")})
        out = normalise_events([ev], profile, data, StubCache())
        self.assertEqual(out[0].amount, Decimal("75.0"))

    def test_fx_no_rate_drops_row(self):
        profile = make_profile(home_currency="USD")
        ev = make_event(currency="EUR")
        out = normalise_events([ev], profile, make_dataset(), StubCache())
        self.assertEqual(out, ())

    def test_does_not_mutate_input(self):
        profile = make_profile()
        ev = make_event(event_id="e1", status="settled")
        events = [ev]
        original = list(events)
        normalise_events(events, profile, make_dataset(), StubCache())
        self.assertEqual(events, original)
        self.assertIs(events[0], ev)


# --------------------------------------------------------------------------
# convert_amount (direct unit coverage of tie-breaking)
# --------------------------------------------------------------------------


class ConvertAmountTests(unittest.TestCase):
    def test_prefers_earlier_date_on_tie(self):
        on = date(2026, 1, 15)
        data = make_dataset(
            rates={
                (date(2026, 1, 12), "EUR", "USD"): Decimal("2"),
                (date(2026, 1, 18), "EUR", "USD"): Decimal("3"),
            }
        )
        result = convert_amount(data, Decimal("10"), "EUR", "USD", on)
        self.assertEqual(result, Decimal("20"))

    def test_beyond_search_window_returns_none(self):
        on = date(2026, 1, 1)
        far_date = on + timedelta(days=RATE_SEARCH_DAYS + 5)
        data = make_dataset(rates={(far_date, "EUR", "USD"): Decimal("2")})
        self.assertIsNone(convert_amount(data, Decimal("10"), "EUR", "USD", on))


# --------------------------------------------------------------------------
# one_off_flows
# --------------------------------------------------------------------------


class OneOffFlowsTests(unittest.TestCase):
    def test_pending_debit_reserved_day_zero_even_if_settles_later(self):
        ev = make_event(
            event_id="p1",
            direction="debit",
            status="pending",
            category="fuel",
            amount=Decimal("50"),
            settlement_date=START + timedelta(days=30),
        )
        out = one_off_flows([ev], START)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].when, START)
        self.assertEqual(out[0].delta, Decimal("-50"))
        self.assertEqual(out[0].label, "fuel|pending")
        self.assertEqual(out[0].source_event_id, "p1")
        self.assertFalse(out[0].is_recurring)

    def test_pending_credit_ignored(self):
        ev = make_event(direction="credit", status="pending", amount=Decimal("50"))
        out = one_off_flows([ev], START)
        self.assertEqual(out, ())

    def test_scheduled_credit_inside_window(self):
        ev = make_event(
            event_id="sc1",
            direction="credit",
            status="scheduled",
            category="salary",
            amount=Decimal("2000"),
            settlement_date=START + timedelta(days=10),
        )
        out = one_off_flows([ev], START)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].delta, Decimal("2000"))
        self.assertEqual(out[0].label, "salary|scheduled")

    def test_scheduled_credit_outside_window_dropped(self):
        ev = make_event(
            direction="credit",
            status="scheduled",
            settlement_date=START + timedelta(days=200),
        )
        out = one_off_flows([ev], START)
        self.assertEqual(out, ())

    def test_scheduled_debit_before_start_clamped_to_start(self):
        ev = make_event(
            event_id="sd1",
            direction="debit",
            status="scheduled",
            category="rent",
            amount=Decimal("500"),
            settlement_date=START - timedelta(days=5),
        )
        out = one_off_flows([ev], START)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].when, START)
        self.assertEqual(out[0].delta, Decimal("-500"))

    def test_settled_excluded(self):
        ev = make_event(direction="debit", status="settled", amount=Decimal("50"))
        out = one_off_flows([ev], START)
        self.assertEqual(out, ())

    def test_sort_order_deterministic(self):
        e1 = make_event(event_id="z", direction="debit", status="pending", category="b", amount=Decimal("1"))
        e2 = make_event(event_id="a", direction="debit", status="pending", category="a", amount=Decimal("1"))
        out = one_off_flows([e1, e2], START)
        self.assertEqual([f.source_event_id for f in out], ["a", "z"])

    def test_does_not_mutate_input(self):
        ev = make_event(status="pending", direction="debit")
        events = [ev]
        original = list(events)
        one_off_flows(events, START)
        self.assertEqual(events, original)


# --------------------------------------------------------------------------
# relevant_amendments
# --------------------------------------------------------------------------


class RelevantAmendmentsTests(unittest.TestCase):
    def test_filters_by_request_id_confidence_and_user(self):
        request = make_request(request_id="r1", user_id="u1")
        m_untargeted = make_message(message_id="m1", user_id="u1", request_id=None)
        m_match = make_message(message_id="m2", user_id="u1", request_id="r1", sent_at="2026-01-01T01:00:00")
        m_other_request = make_message(message_id="m3", user_id="u1", request_id="r2")
        m_low_conf = make_message(message_id="m4", user_id="u1", request_id="r1")
        m_error = make_message(message_id="m5", user_id="u1", request_id="r1")
        m_wrong_user = make_message(message_id="m6", user_id="u2", request_id="r1")

        amendments = {
            "m1": make_amendment(message_id="m1", kind="salary_change", confidence=0.9),
            "m2": make_amendment(message_id="m2", kind="salary_change", confidence=0.9),
            "m3": make_amendment(message_id="m3", kind="salary_change", confidence=0.9),
            "m4": make_amendment(message_id="m4", kind="salary_change", confidence=0.1),
            "m6": make_amendment(message_id="m6", kind="salary_change", confidence=0.9),
        }
        cache = StubCache(amendments=amendments, amendment_errors={"m5"})

        out = relevant_amendments(
            [m_untargeted, m_match, m_other_request, m_low_conf, m_error, m_wrong_user],
            request,
            cache,
        )
        self.assertEqual([a.message_id for a in out], ["m1", "m2"])

    def test_orders_by_sent_at_then_message_id(self):
        request = make_request(request_id="r1", user_id="u1")
        m_late = make_message(message_id="m_late", user_id="u1", sent_at="2026-01-02T00:00:00")
        m_early = make_message(message_id="m_early", user_id="u1", sent_at="2026-01-01T00:00:00")
        amendments = {
            "m_late": make_amendment(message_id="m_late", kind="salary_end"),
            "m_early": make_amendment(message_id="m_early", kind="salary_end"),
        }
        cache = StubCache(amendments=amendments)
        out = relevant_amendments([m_late, m_early], request, cache)
        self.assertEqual([a.message_id for a in out], ["m_early", "m_late"])


# --------------------------------------------------------------------------
# apply_amendments
# --------------------------------------------------------------------------


class ApplyAmendmentsTests(unittest.TestCase):
    def test_salary_change_amount_effective_mid_window_earlier_untouched(self):
        early = make_flow(when=date(2026, 1, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        later = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s2")
        amendment = make_amendment(kind="salary_change", effective_date=date(2026, 1, 15), new_amount=Decimal("1200"))
        out = apply_amendments([early, later], [amendment], make_profile(), make_dataset(), START)
        by_id = {f.source_event_id: f for f in out}
        self.assertEqual(by_id["s1"].delta, Decimal("1000"))
        self.assertEqual(by_id["s2"].delta, Decimal("1200"))

    def test_salary_change_pct(self):
        flow = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(kind="salary_change", effective_date=date(2026, 1, 1), pct_change=Decimal("10"))
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("1100.0"))

    def test_salary_change_new_amount_wins_over_pct(self):
        flow = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(
            kind="salary_change",
            effective_date=date(2026, 1, 1),
            new_amount=Decimal("1500"),
            pct_change=Decimal("10"),
        )
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("1500"))

    def test_salary_change_foreign_currency_converted(self):
        flow = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(
            kind="salary_change",
            effective_date=date(2026, 1, 15),
            new_amount=Decimal("1000"),
            currency="EUR",
        )
        data = make_dataset(rates={(date(2026, 1, 15), "EUR", "USD"): Decimal("1.1")})
        out = apply_amendments([flow], [amendment], make_profile(home_currency="USD"), data, START)
        self.assertEqual(out[0].delta, Decimal("1100.0"))

    def test_salary_change_foreign_currency_no_rate_skips_amendment(self):
        flow = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(
            kind="salary_change", effective_date=date(2026, 1, 15), new_amount=Decimal("1000"), currency="EUR"
        )
        out = apply_amendments([flow], [amendment], make_profile(home_currency="USD"), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("1000"))

    def test_salary_date_change_with_related_event_id_moves_only_that_flow(self):
        # Both flows must already fall on/after eff to be candidates at all.
        target = make_flow(when=date(2026, 1, 25), delta=Decimal("1000"), label="salary|scheduled", source_event_id="sal_evt", is_recurring=False)
        other = make_flow(when=date(2026, 1, 25), delta=Decimal("1000"), label="salary|monthly", source_event_id="other", is_recurring=True)
        amendment = make_amendment(
            kind="salary_date_change",
            effective_date=date(2026, 1, 20),
            related_event_id="sal_evt",
        )
        out = apply_amendments([target, other], [amendment], make_profile(), make_dataset(), START)
        by_id = {f.source_event_id: f for f in out}
        self.assertEqual(by_id["sal_evt"].when, date(2026, 1, 20))
        self.assertEqual(by_id["other"].when, date(2026, 1, 25))

    def test_salary_date_change_without_related_event_id_shifts_day_of_month(self):
        # Both flows must already fall on/after eff (2026-01-31) to be candidates.
        flow_jan = make_flow(when=date(2026, 1, 31), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        flow_feb = make_flow(when=date(2026, 2, 15), delta=Decimal("1000"), label="salary|monthly", source_event_id="s2")
        # eff.day == 31; February 2026 has 28 days -> clamp.
        amendment = make_amendment(kind="salary_date_change", effective_date=date(2026, 1, 31))
        out = apply_amendments([flow_jan, flow_feb], [amendment], make_profile(), make_dataset(), START)
        by_id = {f.source_event_id: f for f in out}
        self.assertEqual(by_id["s1"].when, date(2026, 1, 31))
        self.assertEqual(by_id["s2"].when, date(2026, 2, 28))

    def test_salary_date_change_drops_flow_that_leaves_window(self):
        # eff.day == 1, so the March flow's day-of-month becomes 2026-03-01,
        # which is well outside a 1-day window starting at START.
        flow = make_flow(when=date(2026, 3, 5), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(kind="salary_date_change", effective_date=date(2026, 1, 1))
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START, days=1)
        self.assertEqual(out, ())

    def test_salary_end_removes_future_salary_flows_only(self):
        early = make_flow(when=date(2026, 1, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        later = make_flow(when=date(2026, 2, 1), delta=Decimal("1000"), label="salary|monthly", source_event_id="s2")
        other = make_flow(when=date(2026, 2, 1), delta=Decimal("-50"), label="groceries|weekly", source_event_id="g1")
        amendment = make_amendment(kind="salary_end", effective_date=date(2026, 1, 15))
        out = apply_amendments([early, later, other], [amendment], make_profile(), make_dataset(), START)
        ids = {f.source_event_id for f in out}
        self.assertEqual(ids, {"s1", "g1"})

    def test_expense_change_by_related_event_id(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("-100"), label="subscriptions|monthly", source_event_id="exp1")
        other = make_flow(when=date(2026, 1, 10), delta=Decimal("-30"), label="dining|weekly", source_event_id="exp2")
        amendment = make_amendment(
            kind="expense_change", effective_date=date(2026, 1, 1), related_event_id="exp1", new_amount=Decimal("80")
        )
        out = apply_amendments([flow, other], [amendment], make_profile(), make_dataset(), START)
        by_id = {f.source_event_id: f for f in out}
        self.assertEqual(by_id["exp1"].delta, Decimal("-80"))
        self.assertEqual(by_id["exp2"].delta, Decimal("-30"))

    def test_expense_change_pct(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("-100"), label="subscriptions|monthly", source_event_id="exp1")
        amendment = make_amendment(
            kind="expense_change", effective_date=date(2026, 1, 1), related_event_id="exp1", pct_change=Decimal("-10")
        )
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("-90.0"))

    def test_expense_change_without_related_event_id_skipped(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("-100"), label="subscriptions|monthly", source_event_id="exp1")
        amendment = make_amendment(kind="expense_change", effective_date=date(2026, 1, 1), new_amount=Decimal("80"))
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("-100"))

    def test_refund_dispute_no_op_are_no_ops(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("-100"), label="subscriptions|monthly", source_event_id="exp1")
        for kind in ("refund_status", "dispute", "no_op"):
            amendment = make_amendment(kind=kind, related_event_id="exp1", new_amount=Decimal("1"))
            out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
            self.assertEqual(out[0].delta, Decimal("-100"), msg=kind)

    def test_low_confidence_amendment_is_no_op(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        amendment = make_amendment(kind="salary_change", new_amount=Decimal("2000"), confidence=MIN_CONFIDENCE - 0.01)
        out = apply_amendments([flow], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("1000"))

    def test_later_amendment_overrides_earlier_for_same_target(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        a1 = make_amendment(kind="salary_change", new_amount=Decimal("1100"))
        a2 = make_amendment(kind="salary_change", new_amount=Decimal("1300"))
        out = apply_amendments([flow], [a1, a2], make_profile(), make_dataset(), START)
        self.assertEqual(out[0].delta, Decimal("1300"))

    def test_output_sorted_deterministically(self):
        f1 = make_flow(when=date(2026, 1, 5), label="b|x", source_event_id="z", delta=Decimal("-10"))
        f2 = make_flow(when=date(2026, 1, 1), label="a|x", source_event_id="y", delta=Decimal("-10"))
        f3 = make_flow(when=date(2026, 1, 1), label="a|x", source_event_id="a", delta=Decimal("-10"))
        amendment = make_amendment(kind="no_op")
        out = apply_amendments([f1, f2, f3], [amendment], make_profile(), make_dataset(), START)
        self.assertEqual([f.source_event_id for f in out], ["a", "y", "z"])

    def test_does_not_mutate_inputs(self):
        flow = make_flow(when=date(2026, 1, 10), delta=Decimal("1000"), label="salary|monthly", source_event_id="s1")
        flows = [flow]
        amendments = [make_amendment(kind="salary_change", new_amount=Decimal("2000"))]
        original_flows = list(flows)
        original_amendments = list(amendments)
        apply_amendments(flows, amendments, make_profile(), make_dataset(), START)
        self.assertEqual(flows, original_flows)
        self.assertEqual(amendments, original_amendments)
        # The original flow object itself must be untouched (frozen + no replace in place).
        self.assertEqual(flow.delta, Decimal("1000"))


if __name__ == "__main__":
    unittest.main()
