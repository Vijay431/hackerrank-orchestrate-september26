"""Output CSV writer for Buy or Wait? financial decisions."""

from __future__ import annotations

import csv
import os
import tempfile
from decimal import Decimal
from typing import Sequence

from contracts import OUTPUT_COLUMNS, Decision


def format_amount_plain(d: Decimal) -> str:
    """Format amount without thousands separators or exponents.

    Examples:
    - Decimal("25256") -> "25256"
    - Decimal("603.3") -> "603.3"
    - Decimal("87170.56") -> "87170.56"
    """
    # Use format with 'f' to avoid exponent notation
    formatted = format(d, "f")

    # Check if it's an integral value (no fractional part)
    if "." in formatted:
        # Strip trailing zeros after decimal point
        formatted = formatted.rstrip("0").rstrip(".")

    return formatted


def decision_row(decision: Decision) -> list[str]:
    """Convert a Decision to a list of strings in OUTPUT_COLUMNS order."""
    return [
        decision.request_id,
        format_amount_plain(decision.amount_safe_to_pay),
        decision.affordability_status,
        decision.recommended_payment_method,
        decision.payment_plan,
        decision.earliest_date_for_full_payment,
        decision.spending_changes_needed,
        decision.decision_explanation,
    ]


def write_rows(decisions: Sequence[Decision], output_path: str) -> int:
    """Write decisions to CSV file atomically.

    Returns the number of rows written (including header).
    """
    # Write to a temporary file first
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(output_path) or ".")
    try:
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(OUTPUT_COLUMNS)
            # Write data rows
            for decision in decisions:
                writer.writerow(decision_row(decision))

        # Atomic replace
        os.replace(temp_path, output_path)
    except:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise

    # Return row count: 1 header + len(decisions)
    return 1 + len(decisions)


def fill_template(
    decisions: Sequence[Decision],
    template_path: str,
    output_path: str | None = None,
) -> int:
    """Fill a template CSV with decision data.

    Reads a template with header + blank rows (each with request_id).
    Fills in the 7 output fields for each request_id.
    Writes atomically to output_path (default: template_path, in place).

    Raises ValueError if any template row has no decision or any decision
    has no template row, or if there are duplicates.
    """
    if output_path is None:
        output_path = template_path

    # Build a dict of decisions by request_id
    decisions_by_id = {d.request_id: d for d in decisions}

    # Check for duplicates
    if len(decisions_by_id) != len(decisions):
        raise ValueError("Duplicate request_ids in decisions")

    # Read template
    template_rows = []
    template_request_ids = set()
    with open(template_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            request_id = row.get("request_id", "").strip()
            if not request_id:
                raise ValueError("Template row has no request_id")
            if request_id in template_request_ids:
                raise ValueError(f"Duplicate request_id in template: {request_id}")
            template_request_ids.add(request_id)
            template_rows.append((request_id, row))

    # Check that every template row has a decision and vice versa
    decision_ids = set(decisions_by_id.keys())
    if template_request_ids != decision_ids:
        missing_in_decisions = template_request_ids - decision_ids
        missing_in_template = decision_ids - template_request_ids
        if missing_in_decisions:
            raise ValueError(f"Template has request_ids not in decisions: {missing_in_decisions}")
        if missing_in_template:
            raise ValueError(f"Decisions have request_ids not in template: {missing_in_template}")

    # Fill in the data
    filled_rows = []
    for request_id, template_row in template_rows:
        decision = decisions_by_id[request_id]
        filled_row = {
            "request_id": request_id,
            "amount_safe_to_pay": format_amount_plain(decision.amount_safe_to_pay),
            "affordability_status": decision.affordability_status,
            "recommended_payment_method": decision.recommended_payment_method,
            "payment_plan": decision.payment_plan,
            "earliest_date_for_full_payment": decision.earliest_date_for_full_payment,
            "spending_changes_needed": decision.spending_changes_needed,
            "decision_explanation": decision.decision_explanation,
        }
        filled_rows.append(filled_row)

    # Write to temp file, then replace
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(output_path) or ".")
    try:
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(filled_rows)

        # Atomic replace
        os.replace(temp_path, output_path)
    except:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise

    # Return row count: 1 header + len(filled_rows)
    return 1 + len(filled_rows)
