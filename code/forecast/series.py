"""Shared types for the forecast package.

Written by the controller before Phase 3 fans out and frozen like
``contracts.py``: recurrence.py produces ``RecurringSeries`` and labels its
projected flows with ``flow_label``; amendments.py labels one-off flows the
same way and targets flows by ``flow_category``. Do not edit inside a phase
agent -- report a needed change as BLOCKED instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from contracts import CashFlow, Direction, Flexibility

# Cadence classes. ``cadence_days`` is the nominal spacing; monthly series are
# projected on the anchor's day-of-month (clamped to month length), not by
# adding 30 days.
CADENCE_WEEKLY = 7
CADENCE_BIWEEKLY = 14
CADENCE_MONTHLY = 30

CADENCE_NAMES = {
    CADENCE_WEEKLY: "weekly",
    CADENCE_BIWEEKLY: "biweekly",
    CADENCE_MONTHLY: "monthly",
}

LABEL_SEP = "|"


@dataclass(frozen=True)
class RecurringSeries:
    """One recurring commitment or income stream inferred from history.

    ``amount`` is positive and already in the user's home currency (events are
    normalised before detection). ``anchor`` is the settlement date of the
    most recent known occurrence -- settled or scheduled -- and projection
    starts strictly after it so a scheduled row is never double counted.
    Every flow projected from a series carries ``source_event_id`` so that a
    later spending change (stop/reduce) can address the whole series by one id.
    """

    user_id: str
    category: str
    direction: Direction
    cadence_days: int
    amount: Decimal
    anchor: date
    source_event_id: str
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    is_fixed: bool
    event_ids: tuple[str, ...]  # member events, oldest first

    @property
    def cadence_name(self) -> str:
        return CADENCE_NAMES[self.cadence_days]

    @property
    def label(self) -> str:
        return flow_label(self.category, self.cadence_name)


def flow_label(category: str, tag: str) -> str:
    """``CashFlow.label`` convention: ``<category>|<tag>`` where tag is a
    cadence name for recurring flows or an event status for one-offs."""
    return f"{category}{LABEL_SEP}{tag}"


def flow_category(flow: CashFlow) -> str:
    return flow.label.split(LABEL_SEP, 1)[0]


def flow_tag(flow: CashFlow) -> str:
    parts = flow.label.split(LABEL_SEP, 1)
    return parts[1] if len(parts) == 2 else ""
