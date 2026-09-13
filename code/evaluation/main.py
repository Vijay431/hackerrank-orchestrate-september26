"""Scorer for Buy or Wait? predictions.

Two modes, selected by --set:

  --set samples   Score --predictions against the 25 labelled rows in
                   dataset/sample_requests.csv (the only accuracy signal we
                   have). Prints per-field exact-match rates, amount error
                   stats, an explanation-mold structural check, a per-request
                   diff table, and an affordability_status confusion matrix.

  --set dev        dataset/requests.csv has NO ground truth. Runs
                   validate_decision over every predicted row, summarizes
                   violations by category, compares the predicted class
                   distribution against the labelled-sample distribution
                   (flagging a degenerate result), and prints every decision
                   for eyeball review.

Both modes read predictions from --predictions <path> and never write to
dataset/output.csv. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contracts import Decision  # noqa: E402
from validate import (  # noqa: E402
    DATASET_DIR,
    _build_minimal_dataset,
    _load_profiles,
    _load_requests,
    _parse_decimal,
    _row_to_decision,
    validate_decision,
)

EXACT_MATCH_FIELDS = (
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)

REQUIRED_COLUMNS = {
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
}


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(
                f"error: {path} is missing required columns: {sorted(missing)}"
            )
        return list(reader)


def _load_predictions(path: Path) -> dict[str, dict[str, str]]:
    return {row["request_id"]: row for row in _read_csv_rows(path)}


def _load_sample_ground_truth() -> dict[str, dict[str, str]]:
    path = DATASET_DIR / "sample_requests.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["request_id"]: row for row in csv.DictReader(f)}


# --------------------------------------------------------------------------
# decision_explanation structural molds
#
# Derived by inspecting all 25 rows of dataset/sample_requests.csv. Each
# mold is (name, compiled regex). The regex's `min` named group, when
# present, is the currency amount the sentence claims is the protected
# minimum; molds with no `min` group (e.g. not_recommended_b) don't quote
# one, and the check is reported as "n/a" rather than pass/fail.
# --------------------------------------------------------------------------

_NUM = r"[\d,]+(?:\.\d+)?"
_CCY = r"[A-Z]{3}"

MOLDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "full_payment_leaves_at_least",
        re.compile(
            rf"^Pay {_CCY} {_NUM} today\. This leaves at least {_CCY} "
            rf"(?P<min>{_NUM}) available over the next 90 days\.$"
        ),
    ),
    (
        "full_payment_keeps_minimum",
        re.compile(
            rf"^Pay {_CCY} {_NUM} today\. This keeps the {_CCY} "
            rf"(?P<min>{_NUM}) minimum available over the next 90 days\.$"
        ),
    ),
    (
        "full_payment_with_spending_change",
        re.compile(
            rf"^(?:Stop|Reduce) the .+?, then pay {_CCY} {_NUM} today\. "
            rf"This leaves at least {_CCY} (?P<min>{_NUM}) available\.$"
        ),
    ),
    (
        "installments",
        re.compile(
            rf"^Use \d+ installments of {_CCY} {_NUM}, starting [^.]+\. "
            rf"This leaves at least {_CCY} (?P<min>{_NUM}) available\.$"
        ),
    ),
    (
        "wait_pay_in_full_on",
        re.compile(
            rf"^Pay {_CCY} {_NUM} in full on [^.]+\. Paying earlier would "
            rf"take the balance below the {_CCY} (?P<min>{_NUM}) minimum\.$"
        ),
    ),
    (
        "wait_until_then_pay",
        re.compile(
            rf"^Wait until [^,]+, then pay {_CCY} {_NUM} in full\. Paying "
            rf"sooner would put the {_CCY} (?P<min>{_NUM}) minimum at "
            rf"risk\.$"
        ),
    ),
    (
        "not_recommended_by_date",
        re.compile(
            rf"^Do not make this payment by [^.]+\. None of the available "
            rf"options keeps the {_CCY} (?P<min>{_NUM}) minimum "
            rf"protected\.$"
        ),
    ),
    (
        "not_recommended_amount_available",
        re.compile(
            rf"^Do not proceed with the {_CCY} {_NUM} request\. Although "
            rf"{_CCY} {_NUM} is available today, the full amount cannot be "
            rf"completed safely within 90 days\.$"
        ),
    ),
    (
        "partial_payment",
        re.compile(
            rf"^Pay {_CCY} {_NUM} today and the remaining {_CCY} {_NUM} on "
            rf"[^.]+\. This completes the full request and keeps the "
            rf"{_CCY} (?P<min>{_NUM}) minimum protected\.$"
        ),
    ),
)


def classify_explanation(text: str) -> tuple[str, Decimal | None]:
    """Return (mold_name, quoted_min_or_None). mold_name is 'unrecognized' if
    no mold matches."""
    text = (text or "").strip()
    for name, pattern in MOLDS:
        m = pattern.match(text)
        if m:
            groups = m.groupdict()
            min_str = groups.get("min")
            min_amt = _parse_decimal(min_str.replace(",", "")) if min_str else None
            return name, min_amt
    return "unrecognized", None


# --------------------------------------------------------------------------
# --set samples
# --------------------------------------------------------------------------


def _rel_error(expected: Decimal, got: Decimal | None) -> Decimal:
    if got is None:
        return Decimal("1")  # treat unparseable/missing as 100% error
    if expected == 0:
        return Decimal("0") if got == 0 else abs(got)
    return abs(got - expected) / abs(expected)


def run_samples(predictions_path: Path) -> int:
    predictions = _load_predictions(predictions_path)
    truth = _load_sample_ground_truth()
    profiles = _load_profiles(DATASET_DIR)

    field_matches: Counter[str] = Counter()
    total = len(truth)
    rel_errors: list[Decimal] = []
    worst: tuple[str, Decimal] | None = None
    mold_matches = 0
    confusion: Counter[tuple[str, str]] = Counter()
    diff_rows: list[str] = []
    missing_predictions: list[str] = []

    for request_id, exp in sorted(truth.items(), key=lambda kv: kv[0]):
        pred = predictions.get(request_id)
        if pred is None:
            missing_predictions.append(request_id)
            confusion[(exp["affordability_status"], "<missing>")] += 1
            continue

        row_diffs: list[str] = []
        for field in EXACT_MATCH_FIELDS:
            if exp[field] == pred.get(field, ""):
                field_matches[field] += 1
            else:
                row_diffs.append(
                    f"{field}: expected={exp[field]!r} got={pred.get(field, '')!r}"
                )

        confusion[(exp["affordability_status"], pred.get("affordability_status", ""))] += 1

        exp_amt = _parse_decimal(exp["amount_safe_to_pay"])
        got_amt = _parse_decimal(pred.get("amount_safe_to_pay", ""))
        assert exp_amt is not None
        err = _rel_error(exp_amt, got_amt)
        rel_errors.append(err)
        if worst is None or err > worst[1]:
            worst = (request_id, err)
        if got_amt != exp_amt:
            row_diffs.append(
                f"amount_safe_to_pay: expected={exp_amt} got={got_amt} "
                f"(rel_err={err:.2%})"
            )

        exp_mold, exp_min = classify_explanation(exp["decision_explanation"])
        got_mold, got_min = classify_explanation(pred.get("decision_explanation", ""))
        if exp_mold == got_mold:
            mold_matches += 1
        else:
            row_diffs.append(
                f"decision_explanation mold: expected={exp_mold!r} got={got_mold!r}"
            )

        if row_diffs:
            diff_rows.append(f"{request_id}:")
            diff_rows.extend(f"    {d}" for d in row_diffs)

    # decision_explanation min-quote check needs user_id, which isn't part of
    # the output columns above; re-run using the input columns present in
    # sample_requests.csv (it carries user_id as an input field).
    mold_min_checks = {"pass": 0, "fail": 0, "n/a": 0}
    for request_id, exp in truth.items():
        pred = predictions.get(request_id)
        if pred is None:
            continue
        user_id = exp.get("user_id", "")
        profile = profiles.get(user_id)
        got_mold, got_min = classify_explanation(pred.get("decision_explanation", ""))
        if got_min is None:
            mold_min_checks["n/a"] += 1
        elif profile is None:
            mold_min_checks["n/a"] += 1
        elif got_min == profile.minimum_balance_to_keep:
            mold_min_checks["pass"] += 1
        else:
            mold_min_checks["fail"] += 1

    scored = total - len(missing_predictions)

    print("=" * 72)
    print(f"SAMPLES SCORING  ({scored}/{total} requests had a prediction)")
    print("=" * 72)
    if missing_predictions:
        print(f"MISSING predictions for: {missing_predictions}")
    print()
    print("Exact match rate by field:")
    for field in EXACT_MATCH_FIELDS:
        rate = field_matches[field] / total if total else 0.0
        print(f"  {field:<32} {field_matches[field]:>3}/{total} ({rate:.1%})")
    print()
    if rel_errors:
        mean_err = sum(rel_errors) / len(rel_errors)
        print("amount_safe_to_pay:")
        print(f"  mean relative error:  {float(mean_err):.4%}")
        if worst:
            print(f"  worst relative error: {float(worst[1]):.4%}  ({worst[0]})")
    print()
    print(
        f"decision_explanation mold exact match: {mold_matches}/{total} "
        f"({mold_matches / total:.1%})" if total else "n/a"
    )
    print(
        "decision_explanation minimum-quote check "
        f"(quoted amount == user's minimum_balance_to_keep): "
        f"pass={mold_min_checks['pass']} fail={mold_min_checks['fail']} "
        f"n/a={mold_min_checks['n/a']}"
    )
    print()
    print("-" * 72)
    print("Per-request diffs (only requests with at least one differing field):")
    print("-" * 72)
    if diff_rows:
        print("\n".join(diff_rows))
    else:
        print("  (none — every scored field matched ground truth)")
    print()
    print("-" * 72)
    print("Confusion matrix: affordability_status (rows=expected, cols=got)")
    print("-" * 72)
    statuses = ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]
    extra_cols = sorted({c for (_, c) in confusion if c not in statuses})
    cols = statuses + extra_cols
    header = "expected \\ got".ljust(24) + "".join(c[:14].ljust(16) for c in cols)
    print(header)
    for r in statuses:
        row_str = r.ljust(24)
        for c in cols:
            row_str += str(confusion.get((r, c), 0)).ljust(16)
        print(row_str)

    return 0


# --------------------------------------------------------------------------
# --set dev
# --------------------------------------------------------------------------


def _categorize_violation(msg: str) -> str:
    lowered = msg.lower()
    if "affordability_status" in lowered and "is not one of" in lowered:
        return "invalid affordability_status"
    if "recommended_payment_method" in lowered and "is not one of" in lowered:
        return "invalid recommended_payment_method"
    # Method-specific prefixes are checked BEFORE the generic
    # "amount_safe_to_pay" substring: several partial_payment violations
    # mention that field and would otherwise be mis-triaged.
    if lowered.startswith("partial_payment"):
        return "partial_payment rule violation"
    if lowered.startswith("installments"):
        return "installments rule violation"
    if lowered.startswith("full_payment") or lowered.startswith("wait"):
        return "full_payment / wait structural violation"
    if "payment_methods_user_will_consider" in lowered or "max_installment_months" in lowered:
        return "method not eligible for this user"
    if "flexibility" in lowered:
        return "spending change targets an inflexible event"
    if "affordable_now requires" in lowered:
        return "affordable_now / earliest_date mismatch"
    if "not_affordable requires" in lowered:
        return "not_affordable / earliest_date or plan mismatch"
    if "amount_safe_to_pay" in lowered:
        return "amount_safe_to_pay out of range"
    if "chronological" in lowered or "payment_plan entry" in lowered or "payment_plan is" in lowered:
        return "payment_plan malformed"
    if "spending change" in lowered or "spending_changes_needed" in lowered:
        return "spending_changes_needed violation"
    if "decision_explanation is empty" in lowered:
        return "empty decision_explanation"
    if "unknown request_id" in lowered:
        return "unknown request_id"
    return "other"


def run_dev(predictions_path: Path) -> int:
    predictions = _load_predictions(predictions_path)
    requests = _load_requests(DATASET_DIR)
    data = _build_minimal_dataset(DATASET_DIR)
    sample_truth = _load_sample_ground_truth()

    total = len(predictions)
    violation_rows: list[tuple[str, list[str]]] = []
    category_counts: Counter[str] = Counter()

    # Coverage check. run_samples reports missing predictions; dev mode did not,
    # so a short output.csv (fewer than 250 rows) would be scored silently over
    # whatever subset happened to be present.
    expected_ids = {
        r["request_id"]
        for r in csv.DictReader(open(DATASET_DIR / "requests.csv", encoding="utf-8"))
    }
    predicted_ids = set(predictions)
    missing = expected_ids - predicted_ids
    extra = predicted_ids - expected_ids
    if missing:
        coverage = len(predicted_ids & expected_ids) / max(len(expected_ids), 1)
        # A deliberate dev subset is small by design; a truncated full run is
        # mostly complete. Only the latter is a problem worth shouting about.
        if coverage >= 0.5:
            print(
                f"WARNING: predictions cover {coverage:.0%} of requests.csv -- "
                f"{len(missing)} rows missing (e.g. {sorted(missing)[:5]}). "
                "This looks like a TRUNCATED full run."
            )
        else:
            print(
                f"NOTE: scoring a {len(predicted_ids)}-row subset of "
                f"{len(expected_ids)} requests.csv rows (expected for --set dev)"
            )
    if extra:
        print(
            f"NOTE: {len(extra)} predicted ids are not in requests.csv "
            f"(e.g. {sorted(extra)[:5]}) -- expected when scoring a dev subset"
        )
    if not missing and not extra:
        print(f"Coverage: all {len(expected_ids)} requests.csv rows have a prediction")

    for request_id, row in sorted(predictions.items()):
        request = requests.get(request_id)
        if request is None:
            violation_rows.append((request_id, [f"unknown request_id {request_id!r} (not in dataset/requests.csv)"]))
            category_counts["unknown request_id"] += 1
            continue
        decision = _row_to_decision(row)
        violations = list(validate_decision(decision, request, data))
        if violations:
            violation_rows.append((request_id, violations))
            for v in violations:
                category_counts[_categorize_violation(v)] += 1

    print("=" * 72)
    print(f"DEV SET VALIDATION  ({total} predicted rows checked)")
    print("=" * 72)
    print(f"Rows with >=1 contract violation: {len(violation_rows)}/{total}")
    print()
    if category_counts:
        print("Violations by category (cross-field coherence summary):")
        for cat, n in category_counts.most_common():
            print(f"  {n:>4}  {cat}")
    else:
        print("No contract violations found.")
    print()
    if violation_rows:
        print("-" * 72)
        print("Per-row violations:")
        print("-" * 72)
        for request_id, violations in violation_rows:
            print(f"{request_id}: {len(violations)} violation(s)")
            for v in violations:
                print(f"    - {v}")
    print()

    # Distribution comparison vs the labelled sample distribution.
    def _dist(rows: Iterable[dict], field: str) -> Counter[str]:
        return Counter(r.get(field, "") for r in rows)

    pred_status_dist = _dist(predictions.values(), "affordability_status")
    sample_status_dist = _dist(sample_truth.values(), "affordability_status")
    pred_method_dist = _dist(predictions.values(), "recommended_payment_method")
    sample_method_dist = _dist(sample_truth.values(), "recommended_payment_method")

    print("-" * 72)
    print("Distribution comparison: affordability_status")
    print("-" * 72)
    _print_dist_comparison(pred_status_dist, sample_status_dist, total, len(sample_truth))

    print()
    print("-" * 72)
    print("Distribution comparison: recommended_payment_method")
    print("-" * 72)
    _print_dist_comparison(pred_method_dist, sample_method_dist, total, len(sample_truth))

    degenerate = [
        (label, count / total)
        for label, count in pred_status_dist.items()
        if total and count / total > 0.8
    ] + [
        (label, count / total)
        for label, count in pred_method_dist.items()
        if total and count / total > 0.8
    ]
    print()
    if degenerate:
        print("WARNING: degenerate distribution detected (>80% one class):")
        for label, share in degenerate:
            print(f"  {label!r}: {share:.1%} of predictions")
    else:
        print("No degenerate (>80% one class) distribution detected.")

    print()
    print("-" * 72)
    print(f"All {total} predicted decisions (eyeball review):")
    print("-" * 72)
    header = (
        f"{'request_id':<14}{'status':<22}{'method':<16}"
        f"{'amount_safe_to_pay':<20}{'plan':<40}explanation"
    )
    print(header)
    for request_id, row in sorted(predictions.items()):
        plan = row.get("payment_plan", "")
        plan_short = plan if len(plan) <= 38 else plan[:35] + "..."
        expl = row.get("decision_explanation", "")
        expl_short = expl if len(expl) <= 60 else expl[:57] + "..."
        print(
            f"{request_id:<14}{row.get('affordability_status',''):<22}"
            f"{row.get('recommended_payment_method',''):<16}"
            f"{row.get('amount_safe_to_pay',''):<20}{plan_short:<40}{expl_short}"
        )

    return 0 if not violation_rows else 1


def _print_dist_comparison(
    pred_dist: Counter[str], sample_dist: Counter[str], pred_total: int, sample_total: int
) -> None:
    labels = sorted(set(pred_dist) | set(sample_dist))
    print(f"{'label':<24}{'predictions':<16}{'labelled sample':<16}")
    for label in labels:
        p_share = pred_dist.get(label, 0) / pred_total if pred_total else 0.0
        s_share = sample_dist.get(label, 0) / sample_total if sample_total else 0.0
        print(f"{label:<24}{p_share:<16.1%}{s_share:<16.1%}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? prediction scorer")
    parser.add_argument("--set", choices=["samples", "dev"], required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.predictions.exists():
        print(f"error: no such file: {args.predictions}", file=sys.stderr)
        return 2

    if args.set == "samples":
        return run_samples(args.predictions)
    return run_dev(args.predictions)


if __name__ == "__main__":
    raise SystemExit(main())
