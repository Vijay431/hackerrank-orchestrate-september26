"""Decision layer: turn a 90-day forecast into the eight output columns.

Enumerates every candidate payment plan the problem statement allows (full
payment today, wait, installments, partial payment, full payment plus
permitted spending changes), filters to the ones that are both eligible
(the user's preferences allow them) and safe (the balance never dips below
`minimum_balance_to_keep`), and ranks the survivors by the exact ordering in
problem_statement.md's "Choosing Between Safe Plans": deadline compliance,
then no spending changes, then lowest total paid, then earliest start, then
fewest payments, then lowest payment_option_id.

Pure, deterministic, stdlib only. Money is `Decimal` throughout; rounding to
the CSV-facing string representation happens only in `plain_amount` and the
`amount_safe_to_pay` quantization helper, both isolated at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import ROUND_DOWN, Decimal
from itertools import combinations, product
from typing import Sequence

from contracts import (
    AffordabilityStatus,
    CandidatePlan,
    CashFlow,
    Dataset,
    Decision,
    Forecast,
    Payment,
    Profile,
    RequestRecord,
    SpendingChange,
)
from forecast import ForecastContext
from forecast.series import RecurringSeries
from forecast.simulate import (
    earliest_full_payment_date,
    is_safe,
    safe_amount_on,
    with_payments,
)

# --------------------------------------------------------------------------
# Public result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """The chosen plan plus the two preference-independent facts about it."""

    plan: CandidatePlan
    status: AffordabilityStatus
    amount_safe_to_pay: Decimal  # already quantized (see Rounding, bottom of file)
    earliest: date | None  # before spending changes, independent of preferences


# --------------------------------------------------------------------------
# Rounding / formatting helpers
# --------------------------------------------------------------------------


def _format_decimal(d: Decimal) -> Decimal:
    """integral -> integer Decimal, else exactly 2dp (no extra rounding)."""
    if d == d.to_integral():
        return d.quantize(Decimal(1))
    return d.quantize(Decimal("0.01"))


def plain_amount(d: Decimal) -> str:
    """25256 -> "25256", 620.4 -> "620.40", 22590.19 -> "22590.19"."""
    return str(_format_decimal(d))


def _round_amount_safe_to_pay(raw: Decimal) -> Decimal:
    """Quantize to 2dp with ROUND_DOWN (conservative), then normalize so the
    CSV shows 87170.56 / 603.3 / 25256 (writer prints with format(d, "f"))."""
    d = raw.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def render_plan(plan: CandidatePlan) -> str:
    """"YYYY-MM-DD:amt|..." chronological, or "none" when no payments."""
    if not plan.payments:
        return "none"
    ordered = sorted(plan.payments, key=lambda p: p.when)
    return "|".join(f"{p.when.isoformat()}:{plain_amount(p.amount)}" for p in ordered)


def render_changes(changes: Sequence[SpendingChange]) -> str:
    """"a|b|c" in the order given, or "none" when empty."""
    if not changes:
        return "none"
    return "|".join(c.render() for c in changes)


def _numeric_suffix(event_id: str) -> int:
    """"event_1815" -> 1815; no trailing digits -> 0."""
    m = re.search(r"(\d+)$", event_id)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------
# Spending changes (candidate 5)
# --------------------------------------------------------------------------


def _reduce_allowed(series: RecurringSeries, profile: Profile) -> bool:
    return (
        series.flexibility in ("reducible", "reducible_or_stoppable")
        and series.category in profile.expense_categories_user_is_willing_to_reduce
        and series.minimum_allowed_amount is not None
        and Decimal(0) <= series.minimum_allowed_amount < series.amount
    )


def _stop_allowed(series: RecurringSeries, profile: Profile) -> bool:
    return (
        series.flexibility in ("stoppable", "reducible_or_stoppable")
        and series.category in profile.expense_categories_user_is_willing_to_stop
    )


def _allowed_actions(series: RecurringSeries, profile: Profile) -> list[SpendingChange]:
    """Reduce first (preferred), then stop -- mirrors the greedy preference."""
    acts: list[SpendingChange] = []
    if _reduce_allowed(series, profile):
        acts.append(
            SpendingChange(
                "reduce_to", series.source_event_id, _format_decimal(series.minimum_allowed_amount)
            )
        )
    if _stop_allowed(series, profile):
        acts.append(SpendingChange("stop", series.source_event_id, None))
    return acts


def _apply_changes(
    flows: tuple[CashFlow, ...], changes: Sequence[SpendingChange]
) -> tuple[CashFlow, ...]:
    """stop removes every flow sharing the event id; reduce_to replaces the
    delta of every matching debit flow, keeping other fields (may touch both
    a series' projected occurrences and a shared scheduled anchor row)."""
    stop_ids = {c.event_id for c in changes if c.action == "stop"}
    reduce_map = {c.event_id: c.new_amount for c in changes if c.action == "reduce_to"}
    out: list[CashFlow] = []
    for f in flows:
        if f.source_event_id in stop_ids:
            continue
        if f.source_event_id in reduce_map and f.delta < 0:
            out.append(replace(f, delta=-reduce_map[f.source_event_id]))
        else:
            out.append(f)
    return tuple(out)


def _candidate_series(ctx: ForecastContext) -> list[RecurringSeries]:
    protect = ctx.profile.expense_categories_to_protect
    series = [
        s for s in ctx.series if s.direction == "debit" and s.category not in protect
    ]
    series.sort(key=lambda s: _numeric_suffix(s.source_event_id))
    return series


def _full_today_safe_with(
    ctx: ForecastContext, changes: Sequence[SpendingChange], minimum: Decimal
) -> bool:
    request = ctx.request
    new_flows = _apply_changes(ctx.flows, changes)
    resim = ctx.resimulate(new_flows)
    payment = (Payment(request.request_date, request.requested_amount),)
    return is_safe(with_payments(resim, payment), minimum)


def _find_spending_changes(
    ctx: ForecastContext, minimum: Decimal
) -> tuple[SpendingChange, ...] | None:
    profile = ctx.profile
    candidates = _candidate_series(ctx)

    # Greedy accumulation: reduce preferred over stop, cap 3, stop at first safe set.
    greedy: list[SpendingChange] = []
    for series in candidates:
        if len(greedy) >= 3:
            break
        acts = _allowed_actions(series, profile)
        if not acts:
            continue
        greedy.append(acts[0])
        if _full_today_safe_with(ctx, greedy, minimum):
            return tuple(greedy)

    # Fallback: brute force over every set of <=3 actions, at most one action
    # per series (a reducible_or_stoppable series offers both variants).
    per_series = [(s, _allowed_actions(s, profile)) for s in candidates]
    per_series = [(s, acts) for s, acts in per_series if acts]

    for k in range(1, 4):
        safe_combos: list[tuple[SpendingChange, ...]] = []
        for combo in combinations(per_series, k):
            option_lists = [acts for _, acts in combo]
            for chosen in product(*option_lists):
                if _full_today_safe_with(ctx, chosen, minimum):
                    safe_combos.append(chosen)
        if safe_combos:
            def _key(chosen: tuple[SpendingChange, ...]) -> tuple:
                stops = sum(1 for c in chosen if c.action == "stop")
                ids = tuple(sorted(_numeric_suffix(c.event_id) for c in chosen))
                return (stops, ids)

            return min(safe_combos, key=_key)

    return None


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _rank_key(plan: CandidatePlan, deadline: date) -> tuple:
    last_date = max(p.when for p in plan.payments)
    first_date = min(p.when for p in plan.payments)
    completes_by_deadline = last_date <= deadline
    opt_key = _numeric_suffix(plan.payment_option_id) if plan.payment_option_id else 10**18
    return (
        not completes_by_deadline,
        bool(plan.spending_changes),
        plan.total_paid,
        first_date,
        len(plan.payments),
        opt_key,
    )


def _status_for(plan: CandidatePlan, earliest: date | None) -> AffordabilityStatus:
    if plan.method == "not_recommended":
        # A full payment that becomes safe later but that no eligible method
        # can deliver (the user rejects full_payment, no option fits) is still
        # "expected to become safe later"; only a missing date is not_affordable,
        # which is also what keeps earliest_date_for_full_payment populated.
        return "not_affordable" if earliest is None else "affordable_later"
    if plan.method == "wait":
        return "affordable_later"
    if plan.method == "full_payment" and not plan.spending_changes:
        return "affordable_now"
    return "affordable_with_plan"


_NOT_RECOMMENDED = CandidatePlan("not_recommended", (), (), None, Decimal(0), False)


# --------------------------------------------------------------------------
# Candidate enumeration
# --------------------------------------------------------------------------


def choose(request: RequestRecord, ctx: ForecastContext, data: Dataset) -> Choice:
    profile = ctx.profile
    minimum = profile.minimum_balance_to_keep
    requested = request.requested_amount
    today = request.request_date
    deadline = request.desired_completion_date
    methods = set(profile.payment_methods_user_will_consider)
    base = ctx.forecast

    safe_raw = safe_amount_on(base, today, minimum, requested)
    safe_q = _round_amount_safe_to_pay(safe_raw)
    earliest = earliest_full_payment_date(base, requested, minimum)

    options = data.options_by_request.get(request.request_id, ())
    full_option_id = next(
        (o.payment_option_id for o in options if o.payment_method == "full_payment"), None
    )

    safe_plans: list[CandidatePlan] = []

    # Candidate 1: full today.
    full_plan = CandidatePlan(
        method="full_payment",
        payments=(Payment(today, requested),),
        spending_changes=(),
        payment_option_id=full_option_id,
        total_paid=requested,
        completes_request=True,
    )
    full_eligible = "full_payment" in methods
    full_is_safe = is_safe(with_payments(base, full_plan.payments), minimum)
    if full_eligible and full_is_safe:
        safe_plans.append(full_plan)

    # Candidate 2: wait.
    if earliest is not None and earliest > today and "full_payment" in methods:
        wait_plan = CandidatePlan(
            method="wait",
            payments=(Payment(earliest, requested),),
            spending_changes=(),
            payment_option_id=None,
            total_paid=requested,
            completes_request=True,
        )
        if is_safe(with_payments(base, wait_plan.payments), minimum):
            safe_plans.append(wait_plan)

    # Candidate 3: installments (one per eligible payment option).
    if "installments" in methods and profile.max_installment_months is not None:
        for option in options:
            if option.payment_method != "installments":
                continue
            if option.number_of_payments > profile.max_installment_months:
                continue
            freq = option.payment_frequency_days or 30
            pay_dates = [
                option.first_payment_date + timedelta(days=freq * i)
                for i in range(option.number_of_payments)
            ]
            if any(d < today for d in pay_dates):
                continue  # option dates are >= request_date; drop it otherwise
            plan = CandidatePlan(
                method="installments",
                payments=tuple(Payment(d, option.payment_amount) for d in pay_dates),
                spending_changes=(),
                payment_option_id=option.payment_option_id,
                total_paid=option.total_payable_amount,
                completes_request=True,
            )
            if is_safe(with_payments(base, plan.payments), minimum):
                safe_plans.append(plan)

    # Candidate 4: partial payment.
    if (
        request.allows_partial_payment
        and "partial_payment" in methods
        and Decimal(0) < safe_q < requested
        and earliest is not None
        and earliest <= deadline
    ):
        remainder = requested - safe_q
        partial_plan = CandidatePlan(
            method="partial_payment",
            payments=(Payment(today, safe_q), Payment(earliest, remainder)),
            spending_changes=(),
            payment_option_id=None,
            total_paid=requested,
            completes_request=True,
        )
        if is_safe(with_payments(base, partial_plan.payments), minimum):
            safe_plans.append(partial_plan)

    # Candidate 5: full today + spending changes -- only when full payment is
    # an eligible method but plain full-today is unsafe.
    if full_eligible and not full_is_safe:
        changes = _find_spending_changes(ctx, minimum)
        if changes:
            new_flows = _apply_changes(ctx.flows, changes)
            resim = ctx.resimulate(new_flows)
            changed_plan = CandidatePlan(
                method="full_payment",
                payments=(Payment(today, requested),),
                spending_changes=changes,
                payment_option_id=full_option_id,
                total_paid=requested,
                completes_request=True,
            )
            if is_safe(with_payments(resim, changed_plan.payments), minimum):
                safe_plans.append(changed_plan)

    if safe_plans:
        chosen = min(safe_plans, key=lambda p: _rank_key(p, deadline))
    else:
        chosen = _NOT_RECOMMENDED

    status = _status_for(chosen, earliest)
    return Choice(plan=chosen, status=status, amount_safe_to_pay=safe_q, earliest=earliest)


# --------------------------------------------------------------------------
# contracts.py entry point
# --------------------------------------------------------------------------


def _wrap_bare_forecast(request: RequestRecord, forecast: Forecast, data: Dataset) -> ForecastContext:
    """No `series` on a bare Forecast, so no spending change is possible."""
    profile = data.profiles[request.user_id]
    return ForecastContext(
        request=request,
        profile=profile,
        events=(),
        series=(),
        recurring_flows=(),
        one_off_flows=(),
        amendments=(),
        flows=tuple(forecast.flows),
        forecast=forecast,
    )


def decide(
    request: RequestRecord,
    forecast: Forecast | ForecastContext,
    data: Dataset,
) -> Decision:
    ctx = forecast if isinstance(forecast, ForecastContext) else _wrap_bare_forecast(
        request, forecast, data
    )
    choice = choose(request, ctx, data)
    plan = choice.plan

    descriptions = {
        c.event_id: data.events_by_id[c.event_id].description for c in plan.spending_changes
    }
    try:
        from explain import explain  # written by another agent, not present here

        explanation = explain(
            request,
            plan,
            choice.status,
            ctx.profile,
            descriptions=descriptions,
            amount_safe_to_pay=choice.amount_safe_to_pay,
        )
    except ImportError:
        explanation = f"{choice.status}: {plan.method}"

    return Decision(
        request_id=request.request_id,
        amount_safe_to_pay=choice.amount_safe_to_pay,
        affordability_status=choice.status,
        recommended_payment_method=plan.method,
        payment_plan=render_plan(plan),
        earliest_date_for_full_payment=choice.earliest.isoformat() if choice.earliest else "",
        spending_changes_needed=render_changes(plan.spending_changes),
        decision_explanation=explanation,
    )
