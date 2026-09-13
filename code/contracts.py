"""Frozen interfaces for the Buy or Wait? pipeline.

Every module codes against this file. It is written once, before implementation
fans out, and is never edited by an implementation agent: a silent change here
breaks every sibling module at once. If something in here is wrong or missing,
report it as a blocker instead of editing it.

Money is Decimal everywhere and is rounded only at the moment of output.
Dates are datetime.date everywhere; only the CSV/JSON boundary sees strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal, Mapping, Protocol, Sequence

# --------------------------------------------------------------------------
# Enumerations fixed by problem_statement.md
# --------------------------------------------------------------------------

AffordabilityStatus = Literal[
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
]

PaymentMethod = Literal[
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
]

EventStatus = Literal[
    "settled", "pending", "scheduled", "cancelled", "failed", "unrealized"
]

EventType = Literal[
    "expense",
    "subscription",
    "income",
    "debt_payment",
    "refund",
    "investment_purchase",
    "investment_sale",
    "investment_valuation",
]

# NOTE: "reducible_or_stoppable" is real and appears on 225 rows -- such an event
# may be EITHER reduced or stopped, subject to the user's category preferences.
Flexibility = Literal[
    "fixed", "reducible", "stoppable", "reducible_or_stoppable"
]

# NOTE: "non_cash" is real and appears on 10 rows, all investment_valuation /
# unrealized. It moves no money and must never touch the balance.
Direction = Literal["debit", "credit", "non_cash"]

# Amendment kinds the message extractor may emit. Deliberately closed: message
# text is untrusted, so anything it says must map onto one of these or be
# ignored. There is no free-text field by design.
AmendmentKind = Literal[
    "salary_change",
    "salary_date_change",
    "salary_end",
    "expense_change",
    "refund_status",
    "dispute",
    "no_op",
    "ignore",
]

FORECAST_DAYS = 90

OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)

# --------------------------------------------------------------------------
# Dataset records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    # Blank in the CSV means the user will not consider installments at all.
    max_installment_months: int | None


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: EventType
    description: str
    category: str
    direction: Direction
    # None when the CSV cell is blank -- the amount then lives in a linked
    # image. Never coerce this to Decimal("0"); problem_statement.md forbids it.
    amount: Decimal | None
    currency: str
    event_date: date
    # 10 investment_valuation rows have a blank settlement_date in the CSV;
    # loaders fall back to event_date for those. They are non-cash anyway.
    settlement_date: date
    status: EventStatus
    linked_event_id: str | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None

    @property
    def is_cash(self) -> bool:
        """False for rows that move no money. Unrealized investment valuations
        are bookkeeping only and must never reach the balance timeline."""
        return self.direction != "non_cash" and self.status != "unrealized"

    @property
    def signed_amount(self) -> Decimal | None:
        """Amount as a balance delta: credits positive, debits negative.

        Returns Decimal("0") for non-cash rows rather than negating them --
        treating a valuation as a debit would silently corrupt the forecast.
        """
        if not self.is_cash:
            return Decimal("0")
        if self.amount is None:
            return None
        return self.amount if self.direction == "credit" else -self.amount

    @property
    def can_stop(self) -> bool:
        return self.flexibility in ("stoppable", "reducible_or_stoppable")

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in ("reducible", "reducible_or_stoppable")


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None

    @property
    def path(self) -> str:
        return f"dataset/media/images/{self.image_id}.png"


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class LabelledRequest:
    """A sample_requests.csv row: a request plus its ground-truth answer."""

    request: RequestRecord
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


# --------------------------------------------------------------------------
# Extraction results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Amendment:
    """One structured fact extracted from an untrusted message."""

    message_id: str
    user_id: str
    kind: AmendmentKind
    effective_date: date | None
    new_amount: Decimal | None
    pct_change: Decimal | None
    currency: str | None
    related_event_id: str | None
    confidence: float

    @property
    def actionable(self) -> bool:
        return self.kind not in ("no_op", "ignore")


@dataclass(frozen=True)
class ImageAmount:
    image_id: str
    amount: Decimal
    currency: str
    confidence: float


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CashFlow:
    """A single projected movement on the balance timeline."""

    when: date
    delta: Decimal  # credits positive, debits negative
    label: str
    source_event_id: str | None
    is_recurring: bool


@dataclass(frozen=True)
class SpendingChange:
    """stop:<event_id> or reduce_to:<event_id>:<new_amount>."""

    action: Literal["stop", "reduce_to"]
    event_id: str
    new_amount: Decimal | None

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{self.new_amount}"


@dataclass(frozen=True)
class Forecast:
    """Daily balance series covering FORECAST_DAYS from request_date."""

    user_id: str
    start: date
    opening_balance: Decimal
    flows: tuple[CashFlow, ...]
    balances: tuple[Decimal, ...]  # one entry per day, index 0 == start

    def trough(self, frm: date | None = None) -> Decimal:
        raise NotImplementedError

    def balance_on(self, when: date) -> Decimal:
        raise NotImplementedError


@dataclass(frozen=True)
class Payment:
    when: date
    amount: Decimal


@dataclass(frozen=True)
class CandidatePlan:
    method: PaymentMethod
    payments: tuple[Payment, ...]
    spending_changes: tuple[SpendingChange, ...]
    payment_option_id: str | None
    total_paid: Decimal
    completes_request: bool

    def render_plan(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Decision:
    """The eight output columns for one request, pre-serialisation."""

    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


# --------------------------------------------------------------------------
# Shared, read-only dataset view handed to every worker
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    profiles: Mapping[str, Profile]
    events_by_user: Mapping[str, tuple[Event, ...]]
    events_by_id: Mapping[str, Event]
    options_by_request: Mapping[str, tuple[PaymentOption, ...]]
    messages_by_user: Mapping[str, tuple[Message, ...]]
    images_by_event: Mapping[str, ImageRef]
    rates: Mapping[tuple[date, str, str], Decimal]

    def convert(self, amount: Decimal, frm: str, to: str, on: date) -> Decimal:
        """Convert using the rate row for `on` and the stated direction."""
        raise NotImplementedError


class ExtractionCache(Protocol):
    """Read-through, thread-safe. Implementations must not call the API twice
    for the same key even under concurrent access."""

    def amendment_for(self, message: Message) -> Amendment: ...

    def amount_for(self, image: ImageRef) -> ImageAmount: ...


# --------------------------------------------------------------------------
# Module entry points
# --------------------------------------------------------------------------


def load_dataset(dataset_dir: str) -> Dataset:
    raise NotImplementedError


def load_requests(json_path: str) -> tuple[RequestRecord, ...]:
    raise NotImplementedError


def load_labelled(json_path: str) -> tuple[LabelledRequest, ...]:
    raise NotImplementedError


def build_forecast(
    request: RequestRecord,
    data: Dataset,
    cache: ExtractionCache,
) -> Forecast:
    raise NotImplementedError


def decide(
    request: RequestRecord,
    forecast: Forecast,
    data: Dataset,
) -> Decision:
    raise NotImplementedError


def explain(
    request: RequestRecord,
    plan: CandidatePlan,
    status: AffordabilityStatus,
    profile: Profile,
) -> str:
    raise NotImplementedError


def validate_decision(
    decision: Decision,
    request: RequestRecord,
    data: Dataset,
) -> Sequence[str]:
    """Return a list of contract violations; empty means valid."""
    raise NotImplementedError


def fallback_decision(request: RequestRecord, reason: str) -> Decision:
    """Safest legal answer, used when a worker fails. Never returns None --
    output.csv must always carry a row for every request."""
    raise NotImplementedError
