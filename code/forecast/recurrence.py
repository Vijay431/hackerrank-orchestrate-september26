"""Recurring-series detection and projection.

``detect_recurring`` scans a user's settled/scheduled cash events for
recurring commitments and income (rent, groceries, salary, ...) and
``project``/``project_all`` turn each detected :class:`RecurringSeries` into
concrete :class:`CashFlow` occurrences across the forecast window.

Input events are assumed already normalised (home currency, no ``None``
amounts, only ``settled``/``pending``/``scheduled`` statuses, cash rows only)
by an upstream module, but every entry point here re-checks the cheap
invariants (``amount is not None``, ``is_cash``) defensively before trusting
them.

Everything below is a pure function: no globals, no I/O, no mutation of the
inputs, so callers may run detection and projection for many users
concurrently without coordination.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence

from contracts import CashFlow, Event, FORECAST_DAYS
from forecast.series import (
    CADENCE_BIWEEKLY,
    CADENCE_MONTHLY,
    CADENCE_WEEKLY,
    RecurringSeries,
)

MIN_OCCURRENCES = 2
VARIABLE_LOOKBACK = 8
STALE_FACTOR = 2

_HISTORY_STATUSES = ("settled", "scheduled")
_CASH_DIRECTIONS = ("debit", "credit")


def variable_basis(amounts: Sequence[Decimal]) -> Decimal:
    """Basis amount for a variable series: the median of ``amounts``.

    For an even count, the median is the mean of the two middle values
    (after sorting). Kept as a single helper so a later calibration pass can
    swap the statistic without touching the detection logic.
    """
    ordered = sorted(amounts)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / Decimal(2)


def _classify_gap(gap: int) -> int | None:
    """Cadence class a single gap (in days) falls into, or ``None``."""
    if gap <= 10:
        return CADENCE_WEEKLY
    if gap <= 20:
        return CADENCE_BIWEEKLY
    if gap <= 45:
        return CADENCE_MONTHLY
    return None


def _median_gap(gaps: Sequence[int]) -> float:
    ordered = sorted(gaps)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _classify_cadence(gaps: Sequence[int]) -> int | None:
    """Cadence class for a whole gap sequence, or ``None`` if irregular.

    The median gap picks the class; at least 60% of the individual gaps
    must fall in that same class, otherwise the history is too irregular to
    call it a recurring series.
    """
    if not gaps:
        return None
    median = _median_gap(gaps)
    if median <= 10:
        band = CADENCE_WEEKLY
    elif median <= 20:
        band = CADENCE_BIWEEKLY
    elif median <= 45:
        band = CADENCE_MONTHLY
    else:
        return None
    matching = sum(1 for gap in gaps if _classify_gap(gap) == band)
    if matching / len(gaps) < 0.6:
        return None
    return band


@dataclass(frozen=True)
class _Occurrence:
    """One or more same-day events merged for gap analysis."""

    when: date
    representative: Event


def _merge_same_day(events: Sequence[Event]) -> tuple[_Occurrence, ...]:
    """Merge events sharing a ``settlement_date`` into one occurrence.

    ``events`` must already be sorted by ``(settlement_date, event_id)``. A
    zero-day gap (two rows on the same date) then counts as a single
    occurrence for cadence/gap analysis. The representative of a merged
    group is its last member by ``event_id`` -- the same tie-break the group
    was sorted with.
    """
    occurrences: list[_Occurrence] = []
    i = 0
    n = len(events)
    while i < n:
        when = events[i].settlement_date
        j = i + 1
        while j < n and events[j].settlement_date == when:
            j += 1
        representative = events[j - 1]
        occurrences.append(_Occurrence(when=when, representative=representative))
        i = j
    return tuple(occurrences)


# An income series is "confirmed" when at least this share of its settled
# amounts are identical (a one-off reduced payslip or a raise mid-history
# still passes; commissions and gig payouts do not).
INCOME_CONFIRM_SHARE = Decimal("0.5")
# A payslip described with one of these words ends the income stream.
FINAL_INCOME_MARKERS = ("final",)


def _income_confirmed(settled_amounts: Sequence[Decimal]) -> bool:
    counts: dict[Decimal, int] = {}
    for amount in settled_amounts:
        counts[amount] = counts.get(amount, 0) + 1
    return Decimal(max(counts.values())) / Decimal(len(settled_amounts)) >= INCOME_CONFIRM_SHARE


def _is_final_income(event: Event) -> bool:
    text = event.description.lower()
    return any(marker in text for marker in FINAL_INCOME_MARKERS)


def _series_from_members(
    category: str,
    direction: str,
    members: Sequence[Event],
    as_of: date,
) -> RecurringSeries | None:
    """Build one series from a candidate group, or ``None`` if the history
    does not support recurrence (too few rows, irregular gaps, stale)."""
    ordered = tuple(sorted(members, key=lambda e: (e.settlement_date, e.event_id)))
    occurrences = _merge_same_day(ordered)
    if len(occurrences) < MIN_OCCURRENCES:
        return None

    gaps = [
        (occurrences[idx].when - occurrences[idx - 1].when).days
        for idx in range(1, len(occurrences))
    ]
    cadence_days = _classify_cadence(gaps)
    if cadence_days is None:
        return None

    last_occurrence = occurrences[-1]
    anchor = last_occurrence.when
    if (as_of - anchor).days > STALE_FACTOR * cadence_days + 7:
        return None

    representative = last_occurrence.representative
    # `amount is None` events were filtered out by the caller, so every
    # member here has a concrete Decimal amount.
    amounts = [e.amount for e in ordered if e.amount is not None]
    is_fixed = all(a == amounts[0] for a in amounts)

    if direction == "credit":
        # Only confirmed salary may be projected: a stream whose settled
        # amount keeps moving (commissions, gig payouts, a second household
        # income) is not confirmed income. A scheduled row may carry a new
        # amount -- that is a confirmed raise, so it is excluded from the
        # stability check but wins as the projected amount.
        settled_amounts = [e.amount for e in ordered if e.status == "settled" and e.amount is not None]
        if settled_amounts and not _income_confirmed(settled_amounts):
            return None
        amount = ordered[-1].amount
    elif is_fixed:
        amount = amounts[0]
    else:
        amount = variable_basis(amounts[-VARIABLE_LOOKBACK:])
    if amount is None:
        return None

    return RecurringSeries(
        user_id=representative.user_id,
        category=category,
        direction=direction,  # type: ignore[arg-type]
        cadence_days=cadence_days,
        amount=amount,
        anchor=anchor,
        source_event_id=representative.event_id,
        flexibility=representative.flexibility,
        minimum_allowed_amount=representative.minimum_allowed_amount,
        is_fixed=is_fixed,
        event_ids=tuple(e.event_id for e in ordered),
    )


def detect_recurring(events: Sequence[Event], as_of: date) -> tuple[RecurringSeries, ...]:
    """Infer recurring series from ``events`` as observable on ``as_of``.

    ``events`` should already belong to a single user. Only ``settled`` and
    ``scheduled`` cash rows count as history (pending rows are reserved on
    the balance elsewhere, not treated as recurrence evidence). Credits only
    ever recur for ``category == "salary"``; every other credit category is
    skipped since refunds, windfalls, and investment proceeds never recur.
    """
    groups: dict[tuple[str, str], list[Event]] = {}
    for event in events:
        if event.amount is None or not event.is_cash:
            continue
        if event.status not in _HISTORY_STATUSES:
            continue
        if event.direction not in _CASH_DIRECTIONS:
            continue
        if event.direction == "credit" and event.category != "salary":
            continue
        key = (event.category, event.direction)
        groups.setdefault(key, []).append(event)

    results: list[RecurringSeries] = []
    for (category, direction), members in groups.items():
        if direction == "credit":
            latest = max(members, key=lambda e: (e.settlement_date, e.event_id))
            if _is_final_income(latest):
                continue
        series = _series_from_members(category, direction, members, as_of)
        if series is not None:
            results.append(series)
            continue
        # Salary can arrive as several interleaved streams (base pay on the
        # 15th, commission on the 24th, ...). Merged they look irregular; each
        # stream on its own may still be a clean series. Fall back to
        # per-description sub-groups for income only -- variable debit
        # categories legitimately span many descriptions and must stay merged.
        if direction != "credit":
            continue
        by_description: dict[str, list[Event]] = {}
        for event in members:
            by_description.setdefault(event.description, []).append(event)
        if len(by_description) < 2:
            continue
        for stream in by_description.values():
            series = _series_from_members(category, direction, stream, as_of)
            if series is not None:
                results.append(series)

    results.sort(key=lambda s: (s.direction, s.category))
    return tuple(results)


def _step_monthly(anchor: date, months_ahead: int) -> date:
    """``months_ahead`` months after ``anchor``, using the anchor's original
    day-of-month each time (clamped to the target month's length)."""
    total_month_index = anchor.month - 1 + months_ahead
    year = anchor.year + total_month_index // 12
    month = total_month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(anchor.day, last_day)
    return date(year, month, day)


def project(
    series: RecurringSeries, start: date, days: int = FORECAST_DAYS
) -> tuple[CashFlow, ...]:
    """Project ``series`` into concrete flows over ``[start, start+days-1]``.

    Occurrences are emitted strictly after ``series.anchor``, stepping by
    cadence. Occurrences that land before ``start`` are skipped (they are
    already past relative to the forecast window) but stepping continues
    until an occurrence lands beyond the window.
    """
    end = start + timedelta(days=days - 1)
    flows: list[CashFlow] = []
    step = 1
    while True:
        if series.cadence_days == CADENCE_MONTHLY:
            when = _step_monthly(series.anchor, step)
        else:
            when = series.anchor + timedelta(days=series.cadence_days * step)
        if when > end:
            break
        if when >= start:
            delta = series.amount if series.direction == "credit" else -series.amount
            flows.append(
                CashFlow(
                    when=when,
                    delta=delta,
                    label=series.label,
                    source_event_id=series.source_event_id,
                    is_recurring=True,
                )
            )
        step += 1
    return tuple(flows)


def project_all(
    series: Sequence[RecurringSeries], start: date, days: int = FORECAST_DAYS
) -> tuple[CashFlow, ...]:
    """Project every series in ``series`` and merge into one sorted timeline."""
    flows: list[CashFlow] = []
    for one in series:
        flows.extend(project(one, start, days))
    flows.sort(key=lambda f: (f.when, f.label, f.source_event_id or ""))
    return tuple(flows)
