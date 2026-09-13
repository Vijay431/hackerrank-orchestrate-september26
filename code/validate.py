"""Contract validator for Buy or Wait? decisions.

Implements `validate_decision` against the frozen interfaces in `contracts.py`
and the rules in `problem_statement.md`. Also runnable as a CLI to validate an
entire predictions CSV:

    python code/validate.py <path-to-output.csv>

This module intentionally does NOT import `code/loaders.py` (owned by a
sibling agent and possibly incomplete/nonexistent at the time this file is
written). The CLI path loads just enough of `dataset/` itself, using only the
dataclasses from `contracts.py`, to build the `Dataset` and `RequestRecord`
objects `validate_decision` needs.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from contracts import (
    Dataset,
    Decision,
    Event,
    PaymentOption,
    Profile,
    RequestRecord,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

AFFORDABILITY_STATUSES = frozenset(
    {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
)
PAYMENT_METHODS = frozenset(
    {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
)

MAX_SPENDING_CHANGES = 3


# --------------------------------------------------------------------------
# Small parsing helpers (used both by validate_decision internals and by the
# CLI's own minimal CSV loading)
# --------------------------------------------------------------------------


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


def _parse_decimal(s: str) -> Decimal | None:
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _parse_plan(plan: str) -> tuple[list[tuple[date, Decimal]] | None, list[str]]:
    """Parse a payment_plan string into [(date, amount), ...].

    Returns (None, errors) if the string is structurally malformed (bad date,
    bad amount, or entries out of chronological order). Returns ([], []) for
    the literal string "none".
    """
    errors: list[str] = []
    if plan is None:
        return None, ["payment_plan is missing"]
    if plan == "none":
        return [], errors
    if plan == "":
        return None, ["payment_plan is empty (use 'none' for no payments)"]

    entries: list[tuple[date, Decimal]] = []
    parts = plan.split("|")
    malformed = False
    for part in parts:
        if ":" not in part:
            errors.append(f"payment_plan entry {part!r} is not '<date>:<amount>'")
            malformed = True
            continue
        date_str, amount_str = part.split(":", 1)
        d = _parse_date(date_str)
        amt = _parse_decimal(amount_str)
        if d is None:
            errors.append(f"payment_plan entry {part!r} has invalid date {date_str!r}")
            malformed = True
        if amt is None:
            errors.append(f"payment_plan entry {part!r} has invalid amount {amount_str!r}")
            malformed = True
        if d is not None and amt is not None:
            entries.append((d, amt))

    if malformed:
        return None, errors

    for i in range(1, len(entries)):
        if entries[i][0] <= entries[i - 1][0]:
            errors.append(
                "payment_plan entries are not in strictly chronological order: "
                f"{entries[i - 1][0].isoformat()} then {entries[i][0].isoformat()}"
            )
            return None, errors

    return entries, errors


def _plan_matches_option(
    entries: list[tuple[date, Decimal]], option: PaymentOption
) -> bool:
    if len(entries) != option.number_of_payments:
        return False
    if entries[0][0] != option.first_payment_date:
        return False
    for i, (d, amt) in enumerate(entries):
        if amt != option.payment_amount:
            return False
        if i == 0:
            continue
        if option.payment_frequency_days is None:
            return False
        expected_date = option.first_payment_date + timedelta(
            days=option.payment_frequency_days * i
        )
        if d != expected_date:
            return False
    return True


def _validate_spending_changes(
    sc: str, request: RequestRecord, data: Dataset
) -> list[str]:
    violations: list[str] = []
    if sc is None:
        return ["spending_changes_needed is missing"]
    if sc in ("none", ""):
        return violations

    parts = sc.split("|")
    if len(parts) > MAX_SPENDING_CHANGES:
        violations.append(
            f"spending_changes_needed has {len(parts)} entries, at most "
            f"{MAX_SPENDING_CHANGES} allowed"
        )

    profile = data.profiles.get(request.user_id)
    stop_ids: set[str] = set()
    reduce_ids: set[str] = set()

    for part in parts:
        if part.startswith("stop:"):
            event_id = part[len("stop:") :]
            if not event_id:
                violations.append(f"spending change {part!r} is missing an event_id")
                continue
            stop_ids.add(event_id)
            event = data.events_by_id.get(event_id)
            if event is None:
                violations.append(
                    f"spending change {part!r} references unknown event_id {event_id!r}"
                )
                continue
            if profile is not None:
                if event.category in profile.expense_categories_to_protect:
                    violations.append(
                        f"spending change 'stop:{event_id}' targets protected "
                        f"category {event.category!r}"
                    )
                if event.category not in profile.expense_categories_user_is_willing_to_stop:
                    violations.append(
                        f"spending change 'stop:{event_id}' category "
                        f"{event.category!r} is not in "
                        "expense_categories_user_is_willing_to_stop"
                    )
            # problem_statement.md: "Only recurring expenses marked as flexible
            # may be changed." Category permission alone is not enough -- 28
            # events in this dataset are flexibility="fixed" yet sit in a
            # category their user is willing to stop or reduce.
            if not event.can_stop:
                violations.append(
                    f"spending change 'stop:{event_id}' targets an event with "
                    f"flexibility={event.flexibility!r}, which cannot be stopped"
                )
        elif part.startswith("reduce_to:"):
            rest = part[len("reduce_to:") :]
            if ":" not in rest:
                violations.append(
                    f"spending change {part!r} is not 'reduce_to:<event_id>:<amount>'"
                )
                continue
            event_id, amount_str = rest.split(":", 1)
            if not event_id:
                violations.append(f"spending change {part!r} is missing an event_id")
                continue
            reduce_ids.add(event_id)
            new_amount = _parse_decimal(amount_str)
            if new_amount is None:
                violations.append(
                    f"spending change {part!r} has invalid amount {amount_str!r}"
                )
                continue
            event = data.events_by_id.get(event_id)
            if event is None:
                violations.append(
                    f"spending change {part!r} references unknown event_id {event_id!r}"
                )
                continue
            if profile is not None:
                if event.category in profile.expense_categories_to_protect:
                    violations.append(
                        f"spending change 'reduce_to:{event_id}' targets protected "
                        f"category {event.category!r}"
                    )
                if (
                    event.category
                    not in profile.expense_categories_user_is_willing_to_reduce
                ):
                    violations.append(
                        f"spending change 'reduce_to:{event_id}' category "
                        f"{event.category!r} is not in "
                        "expense_categories_user_is_willing_to_reduce"
                    )
            if not event.can_reduce:
                violations.append(
                    f"spending change 'reduce_to:{event_id}' targets an event "
                    f"with flexibility={event.flexibility!r}, which cannot be "
                    "reduced"
                )
            if (
                event.minimum_allowed_amount is not None
                and new_amount < event.minimum_allowed_amount
            ):
                violations.append(
                    f"spending change 'reduce_to:{event_id}' new amount "
                    f"{new_amount} is below minimum_allowed_amount "
                    f"{event.minimum_allowed_amount}"
                )
        else:
            violations.append(
                f"spending change entry {part!r} is not 'stop:<id>' or "
                "'reduce_to:<id>:<amount>'"
            )

    overlap = stop_ids & reduce_ids
    if overlap:
        violations.append(
            "event_id(s) appear in both a stop and a reduce action: "
            f"{sorted(overlap)}"
        )

    return violations


# --------------------------------------------------------------------------
# The contract function
# --------------------------------------------------------------------------


def validate_decision(
    decision: Decision,
    request: RequestRecord,
    data: Dataset,
) -> Sequence[str]:
    """Return a list of contract violations; empty means valid."""
    violations: list[str] = []

    # --- allowed enum values -------------------------------------------------
    if decision.affordability_status not in AFFORDABILITY_STATUSES:
        violations.append(
            f"affordability_status {decision.affordability_status!r} is not one "
            f"of {sorted(AFFORDABILITY_STATUSES)}"
        )
    if decision.recommended_payment_method not in PAYMENT_METHODS:
        violations.append(
            f"recommended_payment_method {decision.recommended_payment_method!r} "
            f"is not one of {sorted(PAYMENT_METHODS)}"
        )

    # --- amount_safe_to_pay bounds ------------------------------------------
    amt_safe = decision.amount_safe_to_pay
    if not isinstance(amt_safe, Decimal):
        parsed = _parse_decimal(str(amt_safe))
        if parsed is None:
            violations.append(f"amount_safe_to_pay {amt_safe!r} is not numeric")
            amt_safe = None
        else:
            amt_safe = parsed
    if amt_safe is not None:
        if amt_safe < 0 or amt_safe > request.requested_amount:
            violations.append(
                f"amount_safe_to_pay {amt_safe} is outside [0, "
                f"requested_amount={request.requested_amount}]"
            )

    # --- payment_plan structural checks --------------------------------------
    entries, plan_errors = _parse_plan(decision.payment_plan)
    violations.extend(plan_errors)

    # --- affordable_now <-> earliest_date_for_full_payment -------------------
    if decision.affordability_status == "affordable_now":
        if decision.earliest_date_for_full_payment != request.request_date.isoformat():
            violations.append(
                "affordable_now requires earliest_date_for_full_payment == "
                f"request_date ({request.request_date.isoformat()}), got "
                f"{decision.earliest_date_for_full_payment!r}"
            )

    # --- not_affordable <-> empty date / plan == none -------------------------
    if decision.affordability_status == "not_affordable":
        if decision.earliest_date_for_full_payment not in ("", None):
            violations.append(
                "not_affordable requires an empty earliest_date_for_full_payment, "
                f"got {decision.earliest_date_for_full_payment!r}"
            )
        if decision.payment_plan != "none":
            violations.append(
                "not_affordable requires payment_plan == 'none', got "
                f"{decision.payment_plan!r}"
            )

    # --- partial_payment rules -------------------------------------------------
    if decision.recommended_payment_method == "partial_payment":
        if decision.affordability_status != "affordable_with_plan":
            violations.append(
                "partial_payment requires affordability_status == "
                f"affordable_with_plan, got {decision.affordability_status!r}"
            )
        if not request.allows_partial_payment:
            violations.append(
                f"partial_payment recommended but request {request.request_id} "
                "has allows_partial_payment == false"
            )
        if entries is None:
            violations.append(
                "partial_payment: payment_plan is malformed, cannot validate "
                "payment structure"
            )
        else:
            if len(entries) != 2:
                violations.append(
                    f"partial_payment requires exactly two payments, got "
                    f"{len(entries)}"
                )
            else:
                (d1, a1), (d2, a2) = entries
                if d1 != request.request_date:
                    violations.append(
                        f"partial_payment first payment date {d1.isoformat()} != "
                        f"request_date {request.request_date.isoformat()}"
                    )
                if amt_safe is not None and a1 != amt_safe:
                    violations.append(
                        f"partial_payment first payment amount {a1} != "
                        f"amount_safe_to_pay {amt_safe}"
                    )
                ed = _parse_date(decision.earliest_date_for_full_payment)
                if ed is None:
                    violations.append(
                        "partial_payment: earliest_date_for_full_payment is "
                        f"missing or invalid ({decision.earliest_date_for_full_payment!r})"
                    )
                elif d2 != ed:
                    violations.append(
                        f"partial_payment second payment date {d2.isoformat()} != "
                        f"earliest_date_for_full_payment {ed.isoformat()}"
                    )
                if ed is not None and ed > request.desired_completion_date:
                    violations.append(
                        f"partial_payment second payment date {ed.isoformat()} is "
                        "after desired_completion_date "
                        f"{request.desired_completion_date.isoformat()}"
                    )
                if amt_safe is not None:
                    expected_second = request.requested_amount - amt_safe
                    if a2 != expected_second:
                        violations.append(
                            f"partial_payment second payment amount {a2} != "
                            f"requested_amount - amount_safe_to_pay "
                            f"({expected_second})"
                        )
                if a1 + a2 != request.requested_amount:
                    violations.append(
                        f"partial_payment payments sum to {a1 + a2}, expected "
                        f"requested_amount {request.requested_amount}"
                    )
                if amt_safe is not None and not (
                    Decimal(0) < amt_safe < request.requested_amount
                ):
                    violations.append(
                        "partial_payment requires 0 < amount_safe_to_pay < "
                        f"requested_amount, got {amt_safe}"
                    )

    # --- method eligibility against the user's stated preferences ----------
    # problem_statement.md, "Choosing Between Safe Plans": an immediate payment
    # method is eligible ONLY if it appears in payment_methods_user_will_consider;
    # `wait` is eligible only when the user accepts full_payment.
    profile = data.profiles.get(request.user_id)
    if profile is not None:
        considered = set(profile.payment_methods_user_will_consider)
        method = decision.recommended_payment_method
        if method in ("full_payment", "partial_payment", "installments"):
            if method not in considered:
                violations.append(
                    f"recommended_payment_method {method!r} is not in the user's "
                    f"payment_methods_user_will_consider {sorted(considered)}"
                )
        elif method == "wait" and "full_payment" not in considered:
            violations.append(
                "recommended_payment_method 'wait' requires the user to accept "
                f"full_payment, but they consider only {sorted(considered)}"
            )
        if method == "installments":
            if profile.max_installment_months is None:
                violations.append(
                    "installments recommended but max_installment_months is "
                    "blank, meaning the user will not consider installments"
                )
            elif entries is not None and len(entries) > profile.max_installment_months:
                violations.append(
                    f"installments plan has {len(entries)} payments, exceeding "
                    f"max_installment_months {profile.max_installment_months}"
                )

    # --- full_payment / wait structural rules ------------------------------
    # These two had no structural checks at all, so a garbled plan such as
    # "2024-03-03:1|2029-01-01:99999" validated cleanly as affordable_now.
    if decision.recommended_payment_method in ("full_payment", "wait"):
        method = decision.recommended_payment_method
        if entries is None:
            violations.append(f"{method}: payment_plan is malformed")
        elif len(entries) != 1:
            violations.append(
                f"{method} requires exactly one payment, got {len(entries)}"
            )
        else:
            when, amount = entries[0]
            if amount != request.requested_amount:
                violations.append(
                    f"{method} payment is {amount}, expected the full "
                    f"requested_amount {request.requested_amount}"
                )
            if method == "full_payment" and when != request.request_date:
                violations.append(
                    f"full_payment must be dated request_date "
                    f"{request.request_date}, got {when}"
                )
            if method == "wait":
                if when <= request.request_date:
                    violations.append(
                        f"wait payment date {when} must be after request_date "
                        f"{request.request_date}"
                    )
                if decision.earliest_date_for_full_payment and str(when) != (
                    decision.earliest_date_for_full_payment
                ):
                    violations.append(
                        f"wait payment date {when} must equal "
                        f"earliest_date_for_full_payment "
                        f"{decision.earliest_date_for_full_payment}"
                    )

    if (
        decision.affordability_status == "affordable_now"
        and amt_safe is not None
        and amt_safe != request.requested_amount
    ):
        violations.append(
            f"affordable_now requires amount_safe_to_pay == requested_amount, "
            f"got {amt_safe} vs {request.requested_amount}"
        )

    # --- installments rules ------------------------------------------------
    if decision.recommended_payment_method == "installments":
        if entries is None:
            violations.append(
                "installments: payment_plan is malformed, cannot validate "
                "against supplied options"
            )
        else:
            options = data.options_by_request.get(request.request_id, ())
            candidates = [o for o in options if o.payment_method == "installments"]
            if not any(_plan_matches_option(entries, o) for o in candidates):
                violations.append(
                    "installments payment_plan does not exactly match any "
                    f"supplied installment option for {request.request_id}"
                )

    # --- spending_changes_needed --------------------------------------------
    violations.extend(
        _validate_spending_changes(decision.spending_changes_needed, request, data)
    )

    # --- decision_explanation ------------------------------------------------
    if decision.decision_explanation is None or not decision.decision_explanation.strip():
        violations.append("decision_explanation is empty")

    return violations


# --------------------------------------------------------------------------
# Dataset loading for the CLI only.
#
# Delegates to `code/loaders.py`, the single source of truth for parsing.
# An earlier revision carried private CSV parsers here because loaders.py did
# not exist yet; keeping them would have meant two independent implementations
# of the same traps (blank amount != 0, non_cash direction, Decimal-from-string)
# that could silently drift apart.
# --------------------------------------------------------------------------

import loaders


def _build_minimal_dataset(dataset_dir: Path) -> Dataset:
    return loaders.load_dataset(str(dataset_dir))


def _load_requests(dataset_dir: Path) -> dict[str, RequestRecord]:
    """Both request sets: requests.csv (250 eval) and sample_requests.csv (25
    labelled). Predictions may be scored against either, so the CLI must know
    every request_id from both files."""
    out = loaders.load_requests_csv(str(dataset_dir / "requests.csv"))
    out.update(loaders.load_requests_csv(str(dataset_dir / "sample_requests.csv")))
    return out


def _load_profiles(dataset_dir: Path) -> dict[str, Profile]:
    """Kept as a named re-export: code/evaluation/main.py imports it."""
    return dict(loaders.load_dataset(str(dataset_dir)).profiles)


def _row_to_decision(row: dict[str, str]) -> Decision:
    amt_str = row["amount_safe_to_pay"].strip()
    amt = _parse_decimal(amt_str)
    return Decision(
        request_id=row["request_id"],
        amount_safe_to_pay=amt if amt is not None else Decimal("-1"),
        affordability_status=row["affordability_status"],  # type: ignore[arg-type]
        recommended_payment_method=row["recommended_payment_method"],  # type: ignore[arg-type]
        payment_plan=row["payment_plan"],
        earliest_date_for_full_payment=row["earliest_date_for_full_payment"],
        spending_changes_needed=row["spending_changes_needed"],
        decision_explanation=row["decision_explanation"],
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python code/validate.py <path-to-output.csv>", file=sys.stderr)
        return 2

    csv_path = Path(argv[0])
    if not csv_path.exists():
        print(f"error: no such file: {csv_path}", file=sys.stderr)
        return 2

    data = _build_minimal_dataset(DATASET_DIR)
    requests = _load_requests(DATASET_DIR)

    total = 0
    bad = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = {
            "request_id",
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
            "decision_explanation",
        } - set(reader.fieldnames or ())
        if missing_cols:
            print(f"error: {csv_path} is missing columns: {sorted(missing_cols)}")
            return 2

        for row in reader:
            total += 1
            request_id = row["request_id"]
            request = requests.get(request_id)
            if request is None:
                print(f"{request_id}: FAIL - unknown request_id (not in dataset/)")
                bad += 1
                continue
            decision = _row_to_decision(row)
            violations = validate_decision(decision, request, data)
            if violations:
                bad += 1
                print(f"{request_id}: FAIL ({len(violations)} violation(s))")
                for v in violations:
                    print(f"  - {v}")

    print(f"\n{total - bad}/{total} rows valid, {bad} row(s) with violations")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
