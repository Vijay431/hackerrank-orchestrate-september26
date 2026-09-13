"""One-time CSV -> JSON converter for the Buy or Wait? dataset.

Run by hand from the repo root:

    python scripts/convert_json.py

Produces three files under dataset/json/:

    requests.json          -- all 250 rows of dataset/requests.csv
    sample_requests.json   -- all 25 rows of dataset/sample_requests.csv
                               (8 input columns + 7 ground-truth columns)
    requests_dev25.json    -- a 25-record stratified sample of requests.json,
                               drawn with random.Random(42), used as a cheap
                               dev subset that still exercises the hard paths
                               (image evidence, FX mismatch, partial-payment
                               mix, installment-eligibility mix, and spread
                               across request_type).

Deterministic and idempotent: running this script twice produces
byte-identical output files (json.dump(..., indent=2, sort_keys=True,
ensure_ascii=False) plus a fixed seed and a fully-ordered selection
algorithm with no reliance on set/dict iteration order for the final
output).

Standard library only.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
JSON_DIR = DATASET_DIR / "json"

REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"

REQUESTS_JSON = JSON_DIR / "requests.json"
SAMPLE_REQUESTS_JSON = JSON_DIR / "sample_requests.json"
DEV25_JSON = JSON_DIR / "requests_dev25.json"

REQUEST_INPUT_COLUMNS = (
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
)

SAMPLE_LABEL_COLUMNS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)

DEV25_SEED = 42
DEV25_SIZE = 25


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unrecognised boolean value: {raw!r}")


def _parse_amount(raw: str) -> float:
    return float(raw)


def load_requests_typed(csv_path: Path) -> list[dict]:
    """Read requests.csv (or sample_requests.csv) rows with typed values."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    typed_rows = []
    for row in rows:
        typed = {
            "request_id": row["request_id"],
            "user_id": row["user_id"],
            "request_date": row["request_date"],
            "request_type": row["request_type"],
            "requested_amount": _parse_amount(row["requested_amount"]),
            "desired_completion_date": row["desired_completion_date"],
            "allows_partial_payment": _parse_bool(row["allows_partial_payment"]),
            "request_text": row["request_text"],
        }
        if "amount_safe_to_pay" in row:
            typed["amount_safe_to_pay"] = _parse_amount(row["amount_safe_to_pay"])
            for col in SAMPLE_LABEL_COLUMNS:
                if col == "amount_safe_to_pay":
                    continue
                typed[col] = row[col]
        typed_rows.append(typed)
    return typed_rows


def write_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Dev-25 stratification
# ---------------------------------------------------------------------------


def _load_image_request_ids() -> set[str]:
    with IMAGES_CSV.open(newline="", encoding="utf-8") as fh:
        return {row["request_id"] for row in csv.DictReader(fh) if row["request_id"]}


def _load_currency_mismatch_users() -> set[str]:
    with FINANCIAL_PROFILES_CSV.open(newline="", encoding="utf-8") as fh:
        home_currency = {row["user_id"]: row["home_currency"] for row in csv.DictReader(fh)}

    mismatched: set[str] = set()
    with FINANCIAL_EVENTS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid = row["user_id"]
            currency = row["currency"]
            home = home_currency.get(uid)
            if home is not None and currency and currency != home:
                mismatched.add(uid)
    return mismatched


def _load_installment_eligible_users() -> set[str]:
    """Users whose max_installment_months is populated (non-blank)."""
    with FINANCIAL_PROFILES_CSV.open(newline="", encoding="utf-8") as fh:
        return {
            row["user_id"]
            for row in csv.DictReader(fh)
            if row["max_installment_months"].strip() != ""
        }


def _request_types_in_order(requests: list[dict]) -> list[str]:
    """Distinct request_type values, in first-seen order over requests.json."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for req in requests:
        rt = req["request_type"]
        if rt not in seen_set:
            seen_set.add(rt)
            seen.append(rt)
    return seen


def build_dev25(requests: list[dict]) -> tuple[list[dict], dict]:
    """Stratified 25-record sample of `requests`, seeded with random.Random(42).

    Quotas (all must be satisfied by the final 25, subject to availability):
      * as many distinct request_type values covered as possible
      * >= 2 requests with an image
      * >= 2 requests whose user has an event with a currency != home_currency
      * a mix of allows_partial_payment True/False (>= 2 of each)
      * a mix of users with populated vs blank max_installment_months
        (>= 2 of each)

    Implementation: shuffle request_ids with a fixed seed, then greedily walk
    the shuffled order picking any request that still helps satisfy an unmet
    quota; once all quotas are met, keep walking the same shuffled order to
    top up to exactly 25 records. The final list is sorted by request_id so
    the on-disk order does not depend on dict/set iteration order.
    """
    image_request_ids = _load_image_request_ids()
    mismatch_users = _load_currency_mismatch_users()
    installment_users = _load_installment_eligible_users()

    request_types = _request_types_in_order(requests)
    by_id = {r["request_id"]: r for r in requests}
    all_ids = [r["request_id"] for r in requests]

    order = list(all_ids)
    random.Random(DEV25_SEED).shuffle(order)

    # Mutable quota state.
    need_type = {t: 1 for t in request_types}
    need_image = 2
    need_mismatch = 2
    need_partial_true = 2
    need_partial_false = 2
    need_installment_pop = 2
    need_installment_blank = 2

    selected: list[str] = []
    selected_set: set[str] = set()

    def flags(rid: str):
        req = by_id[rid]
        uid = req["user_id"]
        return {
            "type": req["request_type"],
            "has_image": rid in image_request_ids,
            "has_mismatch": uid in mismatch_users,
            "partial": req["allows_partial_payment"],
            "installment_pop": uid in installment_users,
        }

    def quotas_remaining() -> bool:
        return (
            any(v > 0 for v in need_type.values())
            or need_image > 0
            or need_mismatch > 0
            or need_partial_true > 0
            or need_partial_false > 0
            or need_installment_pop > 0
            or need_installment_blank > 0
        )

    # Pass 1: greedily satisfy quotas in shuffle order.
    for rid in order:
        if len(selected) >= DEV25_SIZE:
            break
        if not quotas_remaining():
            break
        f = flags(rid)
        helps = False
        if need_type.get(f["type"], 0) > 0:
            helps = True
        if f["has_image"] and need_image > 0:
            helps = True
        if f["has_mismatch"] and need_mismatch > 0:
            helps = True
        if f["partial"] and need_partial_true > 0:
            helps = True
        if (not f["partial"]) and need_partial_false > 0:
            helps = True
        if f["installment_pop"] and need_installment_pop > 0:
            helps = True
        if (not f["installment_pop"]) and need_installment_blank > 0:
            helps = True

        if not helps:
            continue

        selected.append(rid)
        selected_set.add(rid)

        if need_type.get(f["type"], 0) > 0:
            need_type[f["type"]] = 0
        if f["has_image"]:
            need_image = max(0, need_image - 1)
        if f["has_mismatch"]:
            need_mismatch = max(0, need_mismatch - 1)
        if f["partial"]:
            need_partial_true = max(0, need_partial_true - 1)
        else:
            need_partial_false = max(0, need_partial_false - 1)
        if f["installment_pop"]:
            need_installment_pop = max(0, need_installment_pop - 1)
        else:
            need_installment_blank = max(0, need_installment_blank - 1)

    # Pass 2: top up to exactly DEV25_SIZE using the same shuffled order.
    for rid in order:
        if len(selected) >= DEV25_SIZE:
            break
        if rid in selected_set:
            continue
        selected.append(rid)
        selected_set.add(rid)

    selected = selected[:DEV25_SIZE]
    selected_sorted = sorted(selected)
    dev_records = [by_id[rid] for rid in selected_sorted]

    # Coverage summary computed from the FINAL selection (post top-up).
    covered_types = sorted({flags(rid)["type"] for rid in selected_sorted})
    n_image = sum(1 for rid in selected_sorted if flags(rid)["has_image"])
    n_mismatch = sum(1 for rid in selected_sorted if flags(rid)["has_mismatch"])
    n_partial_true = sum(1 for rid in selected_sorted if flags(rid)["partial"])
    n_partial_false = DEV25_SIZE - n_partial_true
    n_installment_pop = sum(1 for rid in selected_sorted if flags(rid)["installment_pop"])
    n_installment_blank = DEV25_SIZE - n_installment_pop

    summary = {
        "total_selected": len(selected_sorted),
        "distinct_request_types_total": len(request_types),
        "distinct_request_types_covered": len(covered_types),
        "covered_request_types": covered_types,
        "with_image": n_image,
        "with_currency_mismatch": n_mismatch,
        "allows_partial_true": n_partial_true,
        "allows_partial_false": n_partial_false,
        "installment_eligible_users": n_installment_pop,
        "installment_ineligible_users": n_installment_blank,
    }
    return dev_records, summary


def main() -> None:
    requests = load_requests_typed(REQUESTS_CSV)
    sample_requests = load_requests_typed(SAMPLE_REQUESTS_CSV)

    write_json(requests, REQUESTS_JSON)
    write_json(sample_requests, SAMPLE_REQUESTS_JSON)

    dev25, summary = build_dev25(requests)
    write_json(dev25, DEV25_JSON)

    print(f"Wrote {REQUESTS_JSON.relative_to(REPO_ROOT)}: {len(requests)} records")
    print(f"Wrote {SAMPLE_REQUESTS_JSON.relative_to(REPO_ROOT)}: {len(sample_requests)} records")
    print(f"Wrote {DEV25_JSON.relative_to(REPO_ROOT)}: {len(dev25)} records")
    print()
    print("dev25 coverage summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
