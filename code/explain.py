"""Explanation templating module for Buy or Wait? financial decisions."""

from __future__ import annotations

import calendar
from datetime import date
from decimal import Decimal
from typing import Literal, Mapping

from contracts import (
    AffordabilityStatus,
    CandidatePlan,
    Payment,
    Profile,
    RequestRecord,
    SpendingChange,
)


def fmt_amount(d: Decimal) -> str:
    """Format amount with thousands separators: integral -> "68,432"; else exactly 2dp "620.40", "15,952,906.67"."""
    # Convert to string with 2 decimal places
    formatted = format(d, ".2f")
    # Split into integer and decimal parts
    parts = formatted.split(".")
    integer_part = parts[0]
    decimal_part = parts[1]

    # Add thousands separators to integer part
    # Handle negative numbers
    is_negative = integer_part.startswith("-")
    if is_negative:
        integer_part = integer_part[1:]

    # Add commas
    integer_with_commas = ""
    for i, digit in enumerate(reversed(integer_part)):
        if i > 0 and i % 3 == 0:
            integer_with_commas = "," + integer_with_commas
        integer_with_commas = digit + integer_with_commas

    if is_negative:
        integer_with_commas = "-" + integer_with_commas

    # Check if decimal part is all zeros (integral value)
    if int(decimal_part) == 0:
        return integer_with_commas
    else:
        return f"{integer_with_commas}.{decimal_part}"


def fmt_date(d: date) -> str:
    """Format date as "8 August 2025" (no leading zero)."""
    day = d.day
    month_name = calendar.month_name[d.month]
    year = d.year
    return f"{day} {month_name} {year}"


def fmt_money(currency: str, d: Decimal) -> str:
    """Format as "INR 274,600" or "EUR 620.40"."""
    return f"{currency} {fmt_amount(d)}"


def explain(
    request: RequestRecord,
    plan: CandidatePlan,
    status: AffordabilityStatus,
    profile: Profile,
    *,
    descriptions: Mapping[str, str] | None = None,
    amount_safe_to_pay: Decimal | None = None,
) -> str:
    """Generate a decision explanation based on the mold selection rules."""
    C = profile.home_currency
    M = profile.minimum_balance_to_keep
    descriptions = descriptions or {}

    # Mold 1a: affordable_now + full_payment + no spending changes
    if (
        status == "affordable_now"
        and plan.method == "full_payment"
        and not plan.spending_changes
    ):
        A = plan.payments[0].amount
        return f"Pay {fmt_money(C, A)} today. This leaves at least {fmt_money(C, M)} available over the next 90 days."

    # Mold 2a: affordable_later + wait
    if status == "affordable_later" and plan.method == "wait":
        A = plan.payments[0].amount
        D = fmt_date(plan.payments[0].when)
        return f"Pay {fmt_money(C, A)} in full on {D}. Paying earlier would take the balance below the {fmt_money(C, M)} minimum."

    # Mold 3: affordable_with_plan + installments
    if status == "affordable_with_plan" and plan.method == "installments":
        N = len(plan.payments)
        A = plan.payments[0].amount
        D = fmt_date(plan.payments[0].when)
        return f"Use {N} installments of {fmt_money(C, A)}, starting {D}. This leaves at least {fmt_money(C, M)} available."

    # Mold 5: affordable_with_plan + full_payment + spending changes
    if (
        status == "affordable_with_plan"
        and plan.method == "full_payment"
        and plan.spending_changes
    ):
        # Build the prefix from spending changes
        changes_desc = []
        for change in plan.spending_changes:
            event_id = change.event_id
            if event_id in descriptions:
                desc = descriptions[event_id].lower()
            else:
                desc = f"the {event_id} expense"

            if change.action == "stop":
                changes_desc.append(f"stop the {desc}")
            elif change.action == "reduce_to":
                amount_str = fmt_money(C, change.new_amount)
                changes_desc.append(f"reduce the {desc} to {amount_str}")

        # Join changes with proper grammar
        if len(changes_desc) == 1:
            prefix = changes_desc[0]
        elif len(changes_desc) == 2:
            prefix = f"{changes_desc[0]} and {changes_desc[1]}"
        else:  # 3+
            prefix = ", ".join(changes_desc[:-1]) + f" and {changes_desc[-1]}"

        # Capitalize only the first letter
        prefix = prefix[0].upper() + prefix[1:]

        A = plan.payments[0].amount
        return f"{prefix}, then pay {fmt_money(C, A)} today. This leaves at least {fmt_money(C, M)} available."

    # Mold 6: affordable_with_plan + partial_payment
    if status == "affordable_with_plan" and plan.method == "partial_payment":
        A = plan.payments[0].amount
        B = plan.payments[1].amount
        D = fmt_date(plan.payments[1].when)
        return f"Pay {fmt_money(C, A)} today and the remaining {fmt_money(C, B)} on {D}. This completes the full request and keeps the {fmt_money(C, M)} minimum protected."

    # Mold 4b: not_affordable + not_recommended, with some amount safe today
    if (
        status == "not_affordable"
        and plan.method == "not_recommended"
        and amount_safe_to_pay is not None
        and amount_safe_to_pay >= request.requested_amount * Decimal("0.10")
    ):
        REQ = fmt_money(C, request.requested_amount)
        ASP = fmt_money(C, amount_safe_to_pay)
        return f"Do not proceed with the {REQ} request. Although {ASP} is available today, the full amount cannot be completed safely within 90 days."

    # Mold 4a: not_affordable + not_recommended (default)
    if status == "not_affordable" and plan.method == "not_recommended":
        D = fmt_date(request.desired_completion_date)
        return f"Do not make this payment by {D}. None of the available options keeps the {fmt_money(C, M)} minimum protected."

    # Fallback (should not reach here)
    return f"Unable to generate explanation for {request.request_id}."
