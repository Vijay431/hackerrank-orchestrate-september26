"""Parse the seven non-request CSVs in `dataset/` into the frozen dataclasses
defined in `contracts.py` and assemble the `Dataset` index object.

This module owns exactly one job: turn raw CSV text into typed, immutable
records with no lossy coercion. In particular:

* Money is always `Decimal`, built from the raw CSV string (never via
  `float`), so precision is exact.
* A blank `amount` cell in financial_events.csv becomes `None`, never
  `Decimal("0")` -- the true value lives in a linked image for those 16 rows,
  and treating it as zero would corrupt every downstream forecast.
* Dates parse to `datetime.date` via `date.fromisoformat`.
* Pipe-separated profile columns split into tuples; a blank cell is `()`.
* Blank optional fields (`linked_event_id`, `related_event_id`, `request_id`,
  `minimum_allowed_amount`, `payment_frequency_days`, `max_installment_months`)
  become `None`.

Import note: this repo runs `python3 code/main.py` from the repo root, which
puts `code/` (not the repo root) on `sys.path[0]`, so sibling modules import
each other as top-level names (`import contracts`), not as `code.contracts`.
A test harness invoked some other way might instead have the repo root on
`sys.path`. Both are supported below without editing any other file.

Design note on `Dataset.convert`: `contracts.Dataset` is a frozen dataclass
whose `convert` method body is `raise NotImplementedError` -- a placeholder,
not something this module may edit (contracts.py is frozen). The fix chosen
here is a concrete subclass, `_ConcreteDataset`, that declares no new fields
(so the dataclass-generated `__init__`, `__eq__`, and `__hash__` from
`Dataset` are inherited unchanged) and overrides only the `convert` method
with a real implementation delegating to the module-level `_convert`
function. `load_dataset` returns an instance of `_ConcreteDataset`, which
`isinstance(..., Dataset)` recognises as a `Dataset`, so every consumer typed
against `contracts.Dataset` works with no changes on their end.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date
from decimal import Decimal

try:  # running as `python3 code/main.py` -> code/ is on sys.path[0]
    from contracts import (
        Dataset,
        Event,
        ExchangeRate,
        ImageRef,
        LabelledRequest,
        Message,
        PaymentOption,
        Profile,
        RequestRecord,
    )
except ImportError:  # running with the repo root on sys.path instead
    from code.contracts import (
        Dataset,
        Event,
        ExchangeRate,
        ImageRef,
        LabelledRequest,
        Message,
        PaymentOption,
        Profile,
        RequestRecord,
    )


# --------------------------------------------------------------------------
# Small parsing helpers -- pure functions, no shared state.
# --------------------------------------------------------------------------


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _decimal(cell: str) -> Decimal:
    """Required money field. Built from the string, never a float."""
    return Decimal(cell)


def _optional_decimal(cell: str) -> Decimal | None:
    return None if cell == "" else Decimal(cell)


def _optional_int(cell: str) -> int | None:
    return None if cell == "" else int(cell)


def _optional_str(cell: str) -> str | None:
    return None if cell == "" else cell


def _split_pipe(cell: str) -> tuple[str, ...]:
    return tuple(cell.split("|")) if cell != "" else ()


def _parse_date(cell: str) -> date:
    return date.fromisoformat(cell)


# --------------------------------------------------------------------------
# Per-file loaders -- each returns records in file order.
# --------------------------------------------------------------------------


def _load_profiles(path: str) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    for row in _read_csv(path):
        profile = Profile(
            user_id=row["user_id"],
            home_currency=row["home_currency"],
            current_available_balance=_decimal(row["current_available_balance"]),
            minimum_balance_to_keep=_decimal(row["minimum_balance_to_keep"]),
            financial_priorities=_split_pipe(row["financial_priorities"]),
            expense_categories_to_protect=_split_pipe(
                row["expense_categories_to_protect"]
            ),
            expense_categories_user_is_willing_to_reduce=_split_pipe(
                row["expense_categories_user_is_willing_to_reduce"]
            ),
            expense_categories_user_is_willing_to_stop=_split_pipe(
                row["expense_categories_user_is_willing_to_stop"]
            ),
            payment_methods_user_will_consider=_split_pipe(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=_optional_int(row["max_installment_months"]),
        )
        profiles[profile.user_id] = profile
    return profiles


def _load_events(path: str) -> list[Event]:
    events: list[Event] = []
    for row in _read_csv(path):
        event_date = _parse_date(row["event_date"])
        # contracts.Event.settlement_date is a non-optional `date`, but 10
        # `investment_valuation` / `unrealized` / `non_cash` rows (mark-to-
        # market snapshots that never settle) carry a blank settlement_date
        # cell. contracts.py is frozen, so the field can't become `date |
        # None`; falling back to event_date is the least-misleading value
        # that still fits the type, and these rows are excluded from cash
        # flow anyway via status == "unrealized" / direction == "non_cash".
        settlement_cell = row["settlement_date"]
        settlement_date = (
            event_date if settlement_cell == "" else _parse_date(settlement_cell)
        )
        events.append(
            Event(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"],  # type: ignore[arg-type]
                description=row["description"],
                category=row["category"],
                direction=row["direction"],  # type: ignore[arg-type]
                amount=_optional_decimal(row["amount"]),
                currency=row["currency"],
                event_date=event_date,
                settlement_date=settlement_date,
                status=row["status"],  # type: ignore[arg-type]
                linked_event_id=_optional_str(row["linked_event_id"]),
                flexibility=row["flexibility"],  # type: ignore[arg-type]
                minimum_allowed_amount=_optional_decimal(
                    row["minimum_allowed_amount"]
                ),
            )
        )
    return events


def _load_exchange_rates(path: str) -> dict[tuple[date, str, str], Decimal]:
    rates: dict[tuple[date, str, str], Decimal] = {}
    for row in _read_csv(path):
        rate = ExchangeRate(
            rate_date=_parse_date(row["rate_date"]),
            from_currency=row["from_currency"],
            to_currency=row["to_currency"],
            rate=_decimal(row["rate"]),
        )
        key = (rate.rate_date, rate.from_currency, rate.to_currency)
        if key in rates and rates[key] != rate.rate:
            raise ValueError(
                f"Conflicting exchange rate rows for {key}: "
                f"{rates[key]} vs {rate.rate}"
            )
        rates[key] = rate.rate
    return rates


def _load_payment_options(path: str) -> list[PaymentOption]:
    options: list[PaymentOption] = []
    for row in _read_csv(path):
        options.append(
            PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"],
                payment_amount=_decimal(row["payment_amount"]),
                number_of_payments=int(row["number_of_payments"]),
                first_payment_date=_parse_date(row["first_payment_date"]),
                payment_frequency_days=_optional_int(row["payment_frequency_days"]),
                financing_fee=_decimal(row["financing_fee"]),
                total_payable_amount=_decimal(row["total_payable_amount"]),
            )
        )
    return options


def _load_messages(path: str) -> list[Message]:
    messages: list[Message] = []
    for row in _read_csv(path):
        messages.append(
            Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=_optional_str(row["request_id"]),
                related_event_id=_optional_str(row["related_event_id"]),
                sent_at=row["sent_at"],
                source_type=row["source_type"],
                message_text=row["message_text"],
            )
        )
    return messages


def _load_images(path: str) -> list[ImageRef]:
    images: list[ImageRef] = []
    for row in _read_csv(path):
        images.append(
            ImageRef(
                image_id=row["image_id"],
                user_id=row["user_id"],
                request_id=_optional_str(row["request_id"]),
                related_event_id=_optional_str(row["related_event_id"]),
            )
        )
    return images


# --------------------------------------------------------------------------
# Currency conversion -- module-level pure function, delegated to by
# _ConcreteDataset.convert (see the module docstring for why this can't be
# implemented directly on contracts.Dataset).
# --------------------------------------------------------------------------


def _convert(
    rates: "dict[tuple[date, str, str], Decimal]",
    amount: Decimal,
    frm: str,
    to: str,
    on: date,
) -> Decimal:
    if frm == to:
        return amount
    direct = rates.get((on, frm, to))
    if direct is not None:
        return amount * direct
    inverse = rates.get((on, to, frm))
    if inverse is not None:
        return amount / inverse
    raise ValueError(
        f"No exchange rate for {frm} -> {to} (or its inverse) on {on.isoformat()}"
    )


class _ConcreteDataset(Dataset):
    """`contracts.Dataset` with a working `convert`.

    Declares no new fields, so the frozen-dataclass `__init__` generated for
    `Dataset` is inherited unchanged; only the `convert` method is overridden.
    See the module docstring for why this subclass exists instead of editing
    contracts.py.
    """

    def convert(self, amount: Decimal, frm: str, to: str, on: date) -> Decimal:
        return _convert(self.rates, amount, frm, to, on)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_dataset(dataset_dir: str) -> Dataset:
    """Load every non-request CSV under `dataset_dir` and index it.

    Pure function: no globals, no mutation after construction. The returned
    `Dataset` (actually a `_ConcreteDataset`) is built entirely from local
    values and is safe to share read-only across threads.
    """
    profiles = _load_profiles(os.path.join(dataset_dir, "financial_profiles.csv"))
    events = _load_events(os.path.join(dataset_dir, "financial_events.csv"))
    rates = _load_exchange_rates(os.path.join(dataset_dir, "exchange_rates.csv"))
    options = _load_payment_options(
        os.path.join(dataset_dir, "request_payment_options.csv")
    )
    messages = _load_messages(os.path.join(dataset_dir, "messages.csv"))
    images = _load_images(os.path.join(dataset_dir, "images.csv"))

    events_by_user_lists: dict[str, list[Event]] = {}
    events_by_id: dict[str, Event] = {}
    for event in events:
        events_by_user_lists.setdefault(event.user_id, []).append(event)
        events_by_id[event.event_id] = event
    events_by_user: dict[str, tuple[Event, ...]] = {
        user_id: tuple(
            sorted(evs, key=lambda e: (e.settlement_date, e.event_id))
        )
        for user_id, evs in events_by_user_lists.items()
    }

    options_by_request_lists: dict[str, list[PaymentOption]] = {}
    for option in options:
        options_by_request_lists.setdefault(option.request_id, []).append(option)
    options_by_request: dict[str, tuple[PaymentOption, ...]] = {
        request_id: tuple(opts)
        for request_id, opts in options_by_request_lists.items()
    }

    messages_by_user_lists: dict[str, list[Message]] = {}
    for message in messages:
        messages_by_user_lists.setdefault(message.user_id, []).append(message)
    messages_by_user: dict[str, tuple[Message, ...]] = {
        user_id: tuple(sorted(msgs, key=lambda m: (m.sent_at, m.message_id)))
        for user_id, msgs in messages_by_user_lists.items()
    }

    images_by_event: dict[str, ImageRef] = {}
    for image in images:
        if image.related_event_id is not None:
            images_by_event[image.related_event_id] = image

    return _ConcreteDataset(
        profiles=profiles,
        events_by_user=events_by_user,
        events_by_id=events_by_id,
        options_by_request=options_by_request,
        messages_by_user=messages_by_user,
        images_by_event=images_by_event,
        rates=rates,
    )


# --------------------------------------------------------------------------
# JSON request readers
#
# `scripts/convert_json.py` emits these files once and they are committed;
# the runtime pipeline reads the JSON and never regenerates it.
#
# Amounts arrive as JSON numbers (floats). They are converted via
# Decimal(str(x)) -- never Decimal(float) -- so the value round-trips to the
# exact figure in the source CSV instead of a binary-float approximation.
# --------------------------------------------------------------------------


def _money_from_json(value) -> Decimal:
    """JSON numbers are floats; convert without binary-float error and drop the
    artefactual trailing zero.

    Decimal(str(15656000.0)) is Decimal("15656000.0"), which renders as
    "15656000.0" -- but the ground truth writes "15656000". Canonicalise here so
    every downstream consumer sees the same shape the dataset uses.
    """
    d = Decimal(str(value))
    if d == d.to_integral_value():
        return d.quantize(Decimal(1))
    return d.normalize()


def _request_from_obj(obj: dict) -> RequestRecord:
    return RequestRecord(
        request_id=obj["request_id"],
        user_id=obj["user_id"],
        request_date=date.fromisoformat(obj["request_date"]),
        request_type=obj["request_type"],
        requested_amount=_money_from_json(obj["requested_amount"]),
        desired_completion_date=date.fromisoformat(obj["desired_completion_date"]),
        allows_partial_payment=bool(obj["allows_partial_payment"]),
        request_text=obj["request_text"],
    )


def load_requests(json_path: str) -> tuple[RequestRecord, ...]:
    """Read an unlabelled request set (requests.json or requests_dev25.json)."""
    with open(json_path, encoding="utf-8") as fh:
        return tuple(_request_from_obj(o) for o in json.load(fh))


def load_labelled(json_path: str) -> tuple[LabelledRequest, ...]:
    """Read sample_requests.json: each record is a request plus its labels."""
    with open(json_path, encoding="utf-8") as fh:
        objs = json.load(fh)
    out = []
    for o in objs:
        out.append(
            LabelledRequest(
                request=_request_from_obj(o),
                amount_safe_to_pay=_money_from_json(o["amount_safe_to_pay"]),
                affordability_status=o["affordability_status"],
                recommended_payment_method=o["recommended_payment_method"],
                payment_plan=o["payment_plan"],
                earliest_date_for_full_payment=o["earliest_date_for_full_payment"],
                spending_changes_needed=o["spending_changes_needed"],
                decision_explanation=o["decision_explanation"],
            )
        )
    return tuple(out)


def load_requests_csv(csv_path: str) -> dict[str, RequestRecord]:
    """Read requests.csv directly, keyed by request_id.

    Used by tooling that must work straight from the CSV without depending on
    the generated JSON having been produced.
    """
    out: dict[str, RequestRecord] = {}
    for row in _read_csv(csv_path):
        out[row["request_id"]] = RequestRecord(
            request_id=row["request_id"],
            user_id=row["user_id"],
            request_date=_parse_date(row["request_date"]),
            request_type=row["request_type"],
            requested_amount=_decimal(row["requested_amount"]),
            desired_completion_date=_parse_date(row["desired_completion_date"]),
            allows_partial_payment=row["allows_partial_payment"].strip().lower() == "true",
            request_text=row["request_text"],
        )
    return out
