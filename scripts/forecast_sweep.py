"""Dev tool: compare the forecast engine's safe amount / earliest date with the
25 labelled samples. Read-only on the cache (fails loudly on a cache miss).

    .venv/bin/python scripts/forecast_sweep.py
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "code"))

from extraction import DiskExtractionCache  # noqa: E402
from forecast import build_context  # noqa: E402
from forecast.simulate import earliest_full_payment_date, safe_amount_on  # noqa: E402
from loaders import load_dataset, load_labelled  # noqa: E402


def _no_api() -> None:
    raise AssertionError("sweep must not call the API -- cache miss")


def main() -> None:
    data = load_dataset(os.path.join(ROOT, "dataset"))
    labelled = load_labelled(os.path.join(ROOT, "dataset", "json", "sample_requests.json"))
    cache = DiskExtractionCache(os.path.join(ROOT, "cache"), extractor_factory=_no_api, repo_root=ROOT)
    print(f"{'req':11}{'status':22}{'safe':>16}{'label':>16}{'rel%':>8}  {'earliest':11}{'label_e':11} amendments")
    hit = dhit = 0
    for lab in labelled:
        r = lab.request
        ctx = build_context(r, data, cache)
        minimum = ctx.profile.minimum_balance_to_keep
        safe = safe_amount_on(ctx.forecast, r.request_date, minimum, r.requested_amount)
        e = earliest_full_payment_date(ctx.forecast, r.requested_amount, minimum)
        es = e.isoformat() if e else ""
        if lab.amount_safe_to_pay:
            rel = (safe - lab.amount_safe_to_pay) / lab.amount_safe_to_pay * 100
        else:
            rel = Decimal(0) if safe == 0 else Decimal(999)
        hit += abs(rel) < 1
        dhit += es == lab.earliest_date_for_full_payment
        print(
            f"{r.request_id:11}{lab.affordability_status:22}{safe:>16.2f}"
            f"{lab.amount_safe_to_pay:>16.2f}{rel:>8.1f}  {es:11}"
            f"{lab.earliest_date_for_full_payment:11} {[a.kind for a in ctx.amendments]}"
        )
    print(f"amount within 1%: {hit}/25   earliest exact: {dhit}/25")


if __name__ == "__main__":
    main()
