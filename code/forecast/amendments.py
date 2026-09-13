"""Event normalisation, one-off flows, and message-amendment application.

Three pure steps that feed the forecast pipeline:

1. ``normalise_events`` turns raw ``Event`` rows into home-currency cash rows,
   resolving blank amounts from linked images and converting foreign
   currencies via ``Dataset.convert`` (with a bounded nearest-date fallback).
2. ``one_off_flows`` turns the non-recurring facts (pending / scheduled rows)
   in a normalised event list into ``CashFlow`` entries. Recurrence detection
   and projection belong to another module -- this module never produces or
   consumes ``RecurringSeries``.
3. ``relevant_amendments`` / ``apply_amendments`` filter untrusted message
   evidence down to structured, high-confidence ``Amendment`` facts and apply
   them to a list of flows.

All functions are pure: no globals, no I/O, no mutation of inputs. Inputs are
never mutated -- new records are built with ``dataclasses.replace``.
"""

from __future__ import annotations

import calendar
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence

from contracts import (
    Amendment,
    CashFlow,
    Dataset,
    Event,
    ExtractionCache,
    Message,
    Profile,
    RequestRecord,
    FORECAST_DAYS,
)
from forecast.series import flow_category, flow_label

MIN_CONFIDENCE = 0.6
RATE_SEARCH_DAYS = 31


# --------------------------------------------------------------------------
# Currency conversion helper
# --------------------------------------------------------------------------


def convert_amount(
    data: Dataset, amount: Decimal, frm: str, to: str, on: date
) -> Decimal | None:
    """Convert ``amount`` from ``frm`` to ``to`` as of date ``on``.

    Tries the exact-date rate first via ``Dataset.convert`` (which itself
    tries the direct pair then the inverse pair on the same date). Failing
    that, searches ``data.rates`` for the nearest date within
    ``RATE_SEARCH_DAYS`` days of ``on`` -- either direction of the currency
    pair, preferring the earlier date on ties -- and converts using that
    date's rate. Returns ``None`` (never raises) when no usable rate exists.
    """
    if frm == to:
        return amount
    try:
        return data.convert(amount, frm, to, on)
    except ValueError:
        pass

    best_date: date | None = None
    best_diff: int | None = None
    for rate_date, rf, rt in data.rates:
        if (rf, rt) != (frm, to) and (rf, rt) != (to, frm):
            continue
        diff = abs((rate_date - on).days)
        if diff > RATE_SEARCH_DAYS:
            continue
        if (
            best_diff is None
            or diff < best_diff
            or (diff == best_diff and rate_date < best_date)  # type: ignore[operator]
        ):
            best_diff = diff
            best_date = rate_date

    if best_date is None:
        return None
    try:
        return data.convert(amount, frm, to, best_date)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Step 1: normalise raw events into home-currency cash rows
# --------------------------------------------------------------------------


def normalise_events(
    events: Sequence[Event], profile: Profile, data: Dataset, cache: ExtractionCache
) -> tuple[Event, ...]:
    """Drop non-cash / failed / cancelled rows, resolve blank amounts from
    linked images, and convert every remaining row to ``profile.home_currency``.

    Never coerces a missing amount to zero: a row whose amount cannot be
    resolved (no image, a failing cache call, or no usable exchange rate) is
    dropped instead. Preserves input order and never mutates ``events``.
    """
    home = profile.home_currency
    result: list[Event] = []

    for event in events:
        if not event.is_cash:
            continue
        if event.status in ("failed", "cancelled"):
            continue

        amount = event.amount
        currency = event.currency

        if amount is None:
            image = data.images_by_event.get(event.event_id)
            if image is None:
                continue
            try:
                image_amount = cache.amount_for(image)
            except Exception:
                continue
            amount = image_amount.amount
            currency = image_amount.currency

        if currency != home:
            converted = convert_amount(data, amount, currency, home, event.settlement_date)
            if converted is None:
                converted = convert_amount(data, amount, currency, home, event.event_date)
            if converted is None:
                continue
            amount = converted

        result.append(dataclasses.replace(event, amount=amount, currency=home))

    return tuple(result)


# --------------------------------------------------------------------------
# Step 2: one-off flows from normalised, non-recurring facts
# --------------------------------------------------------------------------


def one_off_flows(
    events: Sequence[Event], start: date, days: int = FORECAST_DAYS
) -> tuple[CashFlow, ...]:
    """Turn pending and scheduled normalised events into one-off ``CashFlow``s.

    ``settled`` rows are history only (already reflected in the opening
    balance) and never produce a flow. Pending debits are reserved on day 0
    regardless of when they actually settle; pending credits never count
    until settled. Scheduled rows are placed on their settlement date,
    clamped forward to ``start`` if already due, and dropped if they fall
    outside the forecast window.
    """
    window_end = start + timedelta(days=days - 1)
    flows: list[CashFlow] = []

    for event in events:
        if event.status == "settled":
            continue
        if event.amount is None:
            # Defensive: normalise_events should already have dropped these.
            continue

        if event.status == "pending":
            if event.direction == "debit":
                flows.append(
                    CashFlow(
                        when=start,
                        delta=-event.amount,
                        label=flow_label(event.category, "pending"),
                        source_event_id=event.event_id,
                        is_recurring=False,
                    )
                )
            # pending credit: ignored until settled.

        elif event.status == "scheduled":
            if event.direction == "credit":
                if start <= event.settlement_date <= window_end:
                    flows.append(
                        CashFlow(
                            when=event.settlement_date,
                            delta=event.amount,
                            label=flow_label(event.category, "scheduled"),
                            source_event_id=event.event_id,
                            is_recurring=False,
                        )
                    )
            elif event.direction == "debit":
                when = start if event.settlement_date < start else event.settlement_date
                if start <= when <= window_end:
                    flows.append(
                        CashFlow(
                            when=when,
                            delta=-event.amount,
                            label=flow_label(event.category, "scheduled"),
                            source_event_id=event.event_id,
                            is_recurring=False,
                        )
                    )

    flows.sort(key=lambda f: (f.when, f.label, f.source_event_id or ""))
    return tuple(flows)


# --------------------------------------------------------------------------
# Step 3a: filter untrusted messages down to structured amendments
# --------------------------------------------------------------------------


def relevant_amendments(
    messages: Sequence[Message], request: RequestRecord, cache: ExtractionCache
) -> tuple[Amendment, ...]:
    """Extract and filter amendments relevant to ``request``.

    A message is relevant only if it belongs to the request's user and is
    either untargeted (``request_id is None``) or targets this exact request.
    Extraction failures drop the message rather than raising. Only actionable
    amendments meeting ``MIN_CONFIDENCE`` survive. Order is chronological by
    ``(sent_at, message_id)``.
    """
    pairs: list[tuple[Message, Amendment]] = []

    for message in messages:
        if message.user_id != request.user_id:
            continue
        if not (message.request_id is None or message.request_id == request.request_id):
            continue
        try:
            amendment = cache.amendment_for(message)
        except Exception:
            continue
        if amendment.actionable and amendment.confidence >= MIN_CONFIDENCE:
            pairs.append((message, amendment))

    pairs.sort(key=lambda pair: (pair[0].sent_at, pair[0].message_id))
    return tuple(amendment for _, amendment in pairs)


# --------------------------------------------------------------------------
# Step 3b: apply amendments to a list of flows
# --------------------------------------------------------------------------


def _clamped_date(year: int, month: int, day: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last_day))


def _is_salary_credit(flow: CashFlow) -> bool:
    return flow_category(flow) == "salary" and flow.delta > 0


def _apply_salary_change(
    flows: list[CashFlow], eff: date, new_amount: Decimal | None, pct: Decimal | None
) -> list[CashFlow]:
    if new_amount is None and pct is None:
        return flows
    updated: list[CashFlow] = []
    for f in flows:
        if _is_salary_credit(f) and f.when >= eff:
            delta = new_amount if new_amount is not None else f.delta * (1 + pct / 100)  # type: ignore[operator]
            updated.append(dataclasses.replace(f, delta=delta))
        else:
            updated.append(f)
    return updated


def _apply_salary_date_change(
    flows: list[CashFlow],
    eff: date,
    related_event_id: str | None,
    start: date,
    window_end: date,
) -> list[CashFlow]:
    candidates = [f for f in flows if _is_salary_credit(f) and f.when >= eff]
    if not candidates:
        return flows
    candidate_ids = {id(f) for f in candidates}

    single_target_id: int | None = None
    if related_event_id is not None:
        for f in candidates:
            if f.source_event_id == related_event_id and not f.is_recurring:
                single_target_id = id(f)
                break

    updated: list[CashFlow] = []
    for f in flows:
        if id(f) not in candidate_ids:
            updated.append(f)
            continue
        if single_target_id is not None:
            if id(f) != single_target_id:
                updated.append(f)
                continue
            new_when = eff
        else:
            new_when = _clamped_date(f.when.year, f.when.month, eff.day)
        if start <= new_when <= window_end:
            updated.append(dataclasses.replace(f, when=new_when))
        # else: dropped -- it left the forecast window.
    return updated


def _apply_expense_change(
    flows: list[CashFlow],
    eff: date,
    related_event_id: str,
    new_amount: Decimal | None,
    pct: Decimal | None,
) -> list[CashFlow]:
    if new_amount is None and pct is None:
        return flows
    updated: list[CashFlow] = []
    for f in flows:
        if f.source_event_id == related_event_id and f.when >= eff:
            delta = -new_amount if new_amount is not None else f.delta * (1 + pct / 100)  # type: ignore[operator]
            updated.append(dataclasses.replace(f, delta=delta))
        else:
            updated.append(f)
    return updated


def apply_amendments(
    flows: Sequence[CashFlow],
    amendments: Sequence[Amendment],
    profile: Profile,
    data: Dataset,
    start: date,
    days: int = FORECAST_DAYS,
) -> tuple[CashFlow, ...]:
    """Apply structured amendments to ``flows`` in order, later overriding
    earlier for the same target. Never mutates ``flows`` or ``amendments``.

    ``refund_status`` and ``dispute`` amendments never change flows: an
    unsettled credit never counts until it settles, and a disputed debit
    stays reserved -- the financially safer interpretation either way.
    """
    result: list[CashFlow] = list(flows)
    window_end = start + timedelta(days=days - 1)

    for amendment in amendments:
        if not amendment.actionable or amendment.confidence < MIN_CONFIDENCE:
            continue

        eff = amendment.effective_date or start

        new_amount = amendment.new_amount
        if (
            new_amount is not None
            and amendment.currency is not None
            and amendment.currency != profile.home_currency
        ):
            converted = convert_amount(
                data, new_amount, amendment.currency, profile.home_currency, eff
            )
            if converted is None:
                # No usable rate: the amendment can't be applied safely.
                continue
            new_amount = converted

        pct = amendment.pct_change

        if amendment.kind == "salary_change":
            result = _apply_salary_change(result, eff, new_amount, pct)
        elif amendment.kind == "salary_date_change":
            result = _apply_salary_date_change(
                result, eff, amendment.related_event_id, start, window_end
            )
        elif amendment.kind == "salary_end":
            result = [f for f in result if not (_is_salary_credit(f) and f.when >= eff)]
        elif amendment.kind == "expense_change":
            if amendment.related_event_id is None:
                continue
            result = _apply_expense_change(
                result, eff, amendment.related_event_id, new_amount, pct
            )
        # refund_status, dispute, no_op, ignore: no change to flows.

    result.sort(key=lambda f: (f.when, f.label, f.source_event_id or ""))
    return tuple(result)
