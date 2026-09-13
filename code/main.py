"""Buy or Wait? -- entry point.

    python code/main.py [--set full|samples|dev] [--concurrency N] [--resume]

Loads every dataset CSV once, turns each request JSON record into a work unit,
runs them through the bounded batch runner (forecast -> decide -> validate),
and writes the eight output columns. ``--set full`` fills ``dataset/output.csv``
in place and generates ``code/evaluation/usage_report.md``; ``samples`` and
``dev`` write scratch predictions under ``cache/`` and invoke the scorer.

The LLM is only reached through the extraction cache; a warm cache makes the
whole run deterministic and free.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contracts import Dataset, Decision, RequestRecord  # noqa: E402
from decide import decide  # noqa: E402
from extraction import DiskExtractionCache, load_env  # noqa: E402
from forecast import build_context  # noqa: E402
from loaders import load_dataset, load_labelled, load_requests, load_requests_csv  # noqa: E402
from runner import BatchRunner  # noqa: E402
from validate import validate_decision  # noqa: E402
from writer import fill_template, write_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
JSON_DIR = DATASET_DIR / "json"
CACHE_DIR = REPO_ROOT / "cache"
USAGE_REPORT = REPO_ROOT / "code" / "evaluation" / "usage_report.md"

SET_FILES = {
    "full": JSON_DIR / "requests.json",
    "samples": JSON_DIR / "sample_requests.json",
    "dev": JSON_DIR / "requests_dev25.json",
}

# Indicative list prices (USD per 1M tokens) used only for the cost estimate in
# the usage report. Overridable from the environment; never a secret.
PRICE_PER_M = {
    "input": Decimal(os.environ.get("OPENAI_PRICE_INPUT_PER_M", "1.25")),
    "output": Decimal(os.environ.get("OPENAI_PRICE_OUTPUT_PER_M", "10.00")),
}


# --------------------------------------------------------------------------
# Per-request work unit
# --------------------------------------------------------------------------


def fallback_decision(request: RequestRecord, reason: str) -> Decision:
    """Safest legal row when a worker fails: nothing is safe, nothing is planned."""
    return Decision(
        request_id=request.request_id,
        amount_safe_to_pay=Decimal(0),
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment="",
        spending_changes_needed="none",
        decision_explanation=(
            f"Do not make this payment by {request.desired_completion_date.isoformat()}. "
            "The request could not be evaluated safely."
        ),
    )


def process(request: RequestRecord, data: Dataset, cache: DiskExtractionCache) -> Decision:
    ctx = build_context(request, data, cache)
    decision = decide(request, ctx, data)
    violations = validate_decision(decision, request, data)
    if violations:
        raise ValueError("; ".join(violations))
    return decision


def _serialize(decision: Decision) -> dict[str, Any]:
    d = decision.__dict__.copy()
    d["amount_safe_to_pay"] = str(decision.amount_safe_to_pay)
    return d


def _deserialize(obj: dict[str, Any]) -> Decision:
    obj = dict(obj)
    obj["amount_safe_to_pay"] = Decimal(obj["amount_safe_to_pay"])
    return Decision(**obj)


# --------------------------------------------------------------------------
# Usage report
# --------------------------------------------------------------------------


def write_usage_report(usage: dict[str, dict[str, Any]], n_requests: int,
                       elapsed: float, run_set: str, path: Path = USAGE_REPORT) -> None:
    calls = sum(u["calls"] for u in usage.values())
    tin = sum(u["input_tokens"] for u in usage.values())
    tout = sum(u["output_tokens"] for u in usage.values())
    total = tin + tout
    cost = (Decimal(tin) * PRICE_PER_M["input"] + Decimal(tout) * PRICE_PER_M["output"]) / Decimal(1_000_000)
    per_req = Decimal(n_requests) if n_requests else Decimal(1)
    lines = [
        "# Usage report",
        "",
        f"Run: `--set {run_set}` on {dt.datetime.now().astimezone().isoformat(timespec='seconds')}, "
        f"{n_requests} requests, {elapsed:.1f}s wall clock.",
        "",
        "The deterministic engine makes no model calls. The only LLM usage is the",
        "extraction of structured facts from `messages.csv` and the 16 images, which is",
        "cached on disk (`cache/amendments.json`, `cache/image_amounts.json`). Figures",
        "below are the calls actually issued by this run; a warm cache reports zero.",
        "",
        "| provider | model | calls | input tokens | output tokens | total tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for u in usage.values():
        lines.append(
            f"| {u['provider']} | {u['model']} | {u['calls']} | {u['input_tokens']} | "
            f"{u['output_tokens']} | {u['total_tokens']} |"
        )
    if not usage:
        lines.append("| openai | gpt-5.6-luna | 0 | 0 | 0 | 0 |")
    lines += [
        "",
        f"- Total model calls: **{calls}**",
        f"- Total tokens: **{total}** (input {tin}, output {tout})",
        f"- Average tokens per request: **{(Decimal(total) / per_req):.2f}**",
        f"- Estimated total cost: **USD {cost:.4f}** "
        f"(list price input {PRICE_PER_M['input']}/M, output {PRICE_PER_M['output']}/M)",
        f"- Estimated cost per request: **USD {(cost / per_req):.6f}**",
        "",
        "Cumulative cost of populating the cache before this run is recorded in",
        "`cache/usage_history.json` (appended by every run that issued calls).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_usage_history(usage: dict[str, dict[str, Any]], run_set: str) -> None:
    if not usage:
        return
    hist = CACHE_DIR / "usage_history.json"
    entries = json.loads(hist.read_text()) if hist.exists() else []
    entries.append({"at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "set": run_set, "usage": usage})
    hist.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _check_json_matches_csv(requests: tuple[RequestRecord, ...]) -> None:
    csv_ids = set(load_requests_csv(str(DATASET_DIR / "requests.csv")))
    json_ids = {r.request_id for r in requests}
    if csv_ids != json_ids:
        missing = sorted(csv_ids - json_ids)[:5]
        extra = sorted(json_ids - csv_ids)[:5]
        raise SystemExit(
            "dataset/json/requests.json does not match dataset/requests.csv "
            f"(missing {missing}, extra {extra}); regenerate with scripts/convert_json.py"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? decision pipeline")
    parser.add_argument("--set", choices=sorted(SET_FILES), default="full")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-score", action="store_true", help="skip the scorer for samples/dev")
    args = parser.parse_args(argv)

    load_env()
    started = time.time()

    data = load_dataset(str(DATASET_DIR))
    json_path = SET_FILES[args.set]
    if not json_path.exists():
        raise SystemExit(f"{json_path} missing; run scripts/convert_json.py")
    if args.set == "samples":
        requests = tuple(l.request for l in load_labelled(str(json_path)))
    else:
        requests = load_requests(str(json_path))
    if args.set == "full":
        _check_json_matches_csv(requests)

    cache = DiskExtractionCache(CACHE_DIR)
    runner: BatchRunner[RequestRecord, Decision] = BatchRunner(
        lambda r: process(r, data, cache),
        item_id=lambda r: r.request_id,
        on_error=lambda r, exc: _on_error(r, exc),
        checkpoint_dir=str(CACHE_DIR / "results" / args.set),
        concurrency=max(1, args.concurrency),
        resume=args.resume,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    decisions = runner.run(requests)
    elapsed = time.time() - started

    usage = cache.usage.to_dict() if cache.usage is not None else {}
    _append_usage_history(usage, args.set)

    if args.set == "full":
        n = fill_template(decisions, str(DATASET_DIR / "output.csv"))
        write_usage_report(usage, len(requests), elapsed, args.set)
        print(f"[main] wrote dataset/output.csv ({n - 1} rows) and {USAGE_REPORT.relative_to(REPO_ROOT)}",
              file=sys.stderr)
        return 0

    out = CACHE_DIR / f"predictions_{args.set}.csv"
    write_rows(decisions, str(out))
    print(f"[main] wrote {out.relative_to(REPO_ROOT)} ({len(decisions)} rows); "
          f"model calls this run: {sum(u['calls'] for u in usage.values())}", file=sys.stderr)
    if args.no_score:
        return 0
    return subprocess.call(
        [sys.executable, str(REPO_ROOT / "code" / "evaluation" / "main.py"),
         "--set", args.set, "--predictions", str(out)]
    )


def _on_error(request: RequestRecord, exc: BaseException) -> Decision:
    print(f"[main] {request.request_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return fallback_decision(request, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
