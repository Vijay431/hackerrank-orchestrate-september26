"""Daily balance simulation, trough queries, and safe-amount/earliest-date solvers.

`simulate` turns a user's opening balance and a set of projected `CashFlow`s
into a `ConcreteForecast` -- one balance entry per day of the FORECAST_DAYS
window, day 0 == `start` (the request date). `trough`, `safe_amount_on`, and
`earliest_full_payment_date` answer the two most heavily scored questions in
the output contract: how much can the user safely pay today, and how soon can
they safely pay in full. Boundary exactness (day 0 and the last day both
inclusive, the day after the window excluded) matters more here than
anywhere else in the pipeline.

`contracts.Forecast` is frozen and its `trough`/`balance_on` methods are
placeholders (`raise NotImplementedError`); `ConcreteForecast` below is a
subclass declaring no new fields (mirroring `loaders._ConcreteDataset`) so
the dataclass-generated `__init__`, `__eq__`, and `__hash__` are inherited
unchanged, and only the two methods are overridden -- delegating to the
module-level `trough`/`day_index` functions below, which are the versions
every other module calls directly.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence

from contracts import CashFlow, FORECAST_DAYS, Forecast, Payment


class ConcreteForecast(Forecast):
    """`contracts.Forecast` with working `trough()`/`balance_on()`; no new fields."""

    def trough(self, frm: date | None = None) -> Decimal:
        return trough(self, frm)

    def balance_on(self, when: date) -> Decimal:
        return self.balances[day_index(self, when)]


def simulate(
    user_id: str,
    opening_balance: Decimal,
    flows: Sequence[CashFlow],
    start: date,
    days: int = FORECAST_DAYS,
) -> ConcreteForecast:
    """Project `opening_balance` forward `days` days from `start`.

    `balances[i] = opening_balance + sum(f.delta for f in flows if
    start <= f.when <= start + i)`, for `i in range(days)`. Flows with
    `when < start` or `when > start + days - 1` are dropped entirely -- they
    never appear on the returned forecast. The flows stored on the result are
    exactly the in-window ones, sorted by `(when, label, source_event_id)`;
    a `None` `source_event_id` sorts as `""` for this purpose (two flows
    sharing `when` and `label` differing only in a missing id is an edge case
    with no specified order otherwise). Decimal arithmetic only.
    O(days + len(flows)): one pass to bucket in-window deltas by day index,
    one pass to prefix-sum them into balances.
    """
    last_day = start + timedelta(days=days - 1)
    in_window = [f for f in flows if start <= f.when <= last_day]
    in_window.sort(key=lambda f: (f.when, f.label, f.source_event_id or ""))

    day_deltas = [Decimal(0)] * days
    for f in in_window:
        day_deltas[(f.when - start).days] += f.delta

    balances: list[Decimal] = []
    running = opening_balance
    for delta in day_deltas:
        running = running + delta
        balances.append(running)

    return ConcreteForecast(
        user_id=user_id,
        start=start,
        opening_balance=opening_balance,
        flows=tuple(in_window),
        balances=tuple(balances),
    )


def day_index(forecast: Forecast, when: date) -> int:
    """`(when - forecast.start).days`; raise `ValueError` if outside `[0, days)`."""
    idx = (when - forecast.start).days
    if not (0 <= idx < len(forecast.balances)):
        last = forecast.start + timedelta(days=len(forecast.balances) - 1)
        raise ValueError(
            f"{when.isoformat()} is outside the forecast window "
            f"[{forecast.start.isoformat()}, {last.isoformat()}]"
        )
    return idx


def trough(forecast: Forecast, frm: date | None = None) -> Decimal:
    """`min(balances[day_index(frm):])`; `frm=None` means the whole window."""
    start_idx = 0 if frm is None else day_index(forecast, frm)
    return min(forecast.balances[start_idx:])


def safe_amount_on(
    forecast: Forecast, when: date, minimum: Decimal, cap: Decimal
) -> Decimal:
    """Largest `x` in `[0, cap]` such that paying `x` on `when` keeps every
    balance from `when` onward `>= minimum`.

    Closed form: `max(Decimal(0), min(cap, trough(frm=when) - minimum))`.
    Returned exactly, with no rounding (rounding is an output-formatting
    concern, not this module's). `cap <= 0` returns `Decimal(0)` immediately,
    without evaluating the trough (and without requiring `when` to be a valid
    window day in that case).
    """
    if cap <= 0:
        return Decimal(0)
    headroom = trough(forecast, when) - minimum
    return max(Decimal(0), min(cap, headroom))


def earliest_full_payment_date(
    forecast: Forecast, amount: Decimal, minimum: Decimal
) -> date | None:
    """First day in the window (scanning from `start`) with
    `safe_amount_on(day, minimum, amount) >= amount`; `None` if no such day.
    `amount <= 0` returns `start` unconditionally.

    For `amount > 0`, `safe_amount_on(day, minimum, amount) >= amount`
    reduces to `trough(frm=day) >= minimum + amount`: the closed form is
    `max(0, min(amount, trough(day) - minimum))`, and that expression can
    only reach `amount` (its own cap) when `trough(day) - minimum >= amount`.
    So this computes one right-to-left suffix-minimum pass over `balances`
    (O(days)) and scans it once for the first day past the threshold --
    it never calls `trough` (or `safe_amount_on`) per day.
    """
    if amount <= 0:
        return forecast.start
    balances = forecast.balances
    n = len(balances)
    threshold = minimum + amount

    suffix_mins: list[Decimal] = [Decimal(0)] * n
    suffix_mins[n - 1] = balances[n - 1]
    for i in range(n - 2, -1, -1):
        suffix_mins[i] = min(balances[i], suffix_mins[i + 1])

    for i, m in enumerate(suffix_mins):
        if m >= threshold:
            return forecast.start + timedelta(days=i)
    return None


def with_payments(forecast: Forecast, payments: Sequence[Payment]) -> ConcreteForecast:
    """New forecast with one extra debit `CashFlow` per payment --
    `CashFlow(when=p.when, delta=-p.amount, label="payment|plan",
    source_event_id=None, is_recurring=False)` -- added to `forecast`'s
    existing flows. Same opening balance/start/day count. Does not mutate
    `forecast` (it is a frozen dataclass; this always returns a fresh
    `ConcreteForecast` built by `simulate`). Payments outside the window are
    dropped by `simulate`'s own in-window filtering, same as any other flow.
    """
    extra = tuple(
        CashFlow(
            when=p.when,
            delta=-p.amount,
            label="payment|plan",
            source_event_id=None,
            is_recurring=False,
        )
        for p in payments
    )
    return simulate(
        forecast.user_id,
        forecast.opening_balance,
        tuple(forecast.flows) + extra,
        forecast.start,
        days=len(forecast.balances),
    )


def is_safe(forecast: Forecast, minimum: Decimal, frm: date | None = None) -> bool:
    """`trough(frm) >= minimum`."""
    return trough(forecast, frm) >= minimum
