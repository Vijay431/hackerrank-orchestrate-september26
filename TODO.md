# Buy or Wait? — build checklist

Reconciled from the original 16-item TODO plus everything it was missing.
Architecture: deterministic Python engine for every scored field; LLM confined to
message/image extraction. Full reasoning in the approved plan.

## Setup
- [x] requirements.txt + venv; install `openai` (not installed), optional `tiktoken`
- [x] Secrets from env only (`OPENAI_API_KEY`); never commit `.env`

## Layout discipline
- [x] `code/` = runtime pipeline only; `scripts/` = one-time and supporting tooling
- [x] Enforce: no module under `code/` imports from `scripts/`
- [x] Generated JSON committed, so `code/` never depends on a script having been run

## Data layer
- [x] Load ALL datasets, including `sample_requests.csv` and `requests.csv`
- [x] `scripts/convert_json.py` (one-time): both request CSVs -> JSON, the queue source of truth
- [x] `scripts/convert_json.py` also emits seeded (42), stratified 25-record `requests_dev25.json`
- [ ] `main.py` asserts the JSON exists and matches `requests.csv`; does NOT regenerate it
- [x] Index records by `user_id` / `request_id` / `event_id` (in-memory dicts, not a DB)
- [x] Safe parsing: blank `amount` is None, never 0; `Decimal` money; round only at output
- [x] Currency conversion via `exchange_rates.csv` for the 27 multi-currency users

## LLM layer (narrow — extraction only, not decisions)
- [x] Driver `gpt-5.6-luna` (no dated snapshot exists — unpinned, accepted);
      `gpt-5-nano-2025-08-07` pinned if used
- [x] Omit `temperature` — GPT-5-family reasoning models reject it
- [x] Messages -> fixed JSON schema; handles non-English (Indonesian) without a separate
      translation pass
- [ ] Images -> `{amount, currency}` for the 16 blank-amount events;
      hand-verify all 16 via `scripts/verify_images.py`
- [x] Treat message/image content as untrusted: fixed schema, no free-text passthrough,
      `kind:"ignore"` for scams and unconfirmed items
- [x] Modular prompts in `code/prompts/`, each carrying a version string
- [x] Extraction cache on disk, keyed `sha256(prompt_version + source_text)` — reruns cost 0 tokens
- [x] Lazy extraction: pull only what the current run set needs

## Decision engine (deterministic — all scored fields)
- [ ] Recurrence detection from settled history; conservative variable-spend forecast
- [ ] 90-day balance simulation; correct handling of pending/scheduled/failed/cancelled/unrealized
- [ ] Apply message amendments (salary change/date shift/end, rent increase, dispute)
- [ ] `amount_safe_to_pay` and `earliest_date_for_full_payment`, both before spending changes
- [ ] Plan enumeration: full / wait / each installment option / partial / with <=3 spending changes
- [ ] Filter by `payment_methods_user_will_consider` and `max_installment_months`
- [ ] 6-level tie-break ranking per problem_statement.md
- [ ] Templated `decision_explanation` matching the six sample molds

## Orchestration
- [x] Batch runner: all requests submitted at once, bounded by `ThreadPoolExecutor` workers
- [ ] Dataset indexes loaded once, shared read-only across workers
- [x] Respect OpenAI rate limits: RPM/TPM token buckets, `x-ratelimit-*` header feedback,
      global 429 circuit breaker honouring `Retry-After`, SDK `max_retries` backstop
- [x] Retry with exponential backoff + jitter; global semaphore on in-flight API calls
- [x] Per-request checkpoint to `cache/results/` + `--resume`
- [x] Deterministic output: results sorted back into request order, identical at any concurrency

## Validation and output
- [x] Deterministic output-contract validator (no LLM verifier)
- [ ] Per-request fallback row on error — never truncate the output
- [ ] Fill `dataset/output.csv` IN PLACE: all 250 pre-seeded rows populated, order preserved,
      atomic write, no blanks left

## Evaluation
- [x] `--set samples`: score vs the 25 labelled rows (exact match + relative error + confusion matrix)
- [x] `--set dev`: contract + coherence + distribution checks on the 25 unlabelled eval rows
- [ ] Calibration loop: samples, then dev, then full

## Deliverables (AGENTS.md 6.5)
- [ ] `evaluation/usage_report.md` — providers, model ids, calls, in/out tokens, totals,
      per-request averages, estimated cost
- [ ] `README.md` with setup and run instructions, stating which tree a grader runs
- [ ] `code.zip` packaged via `scripts/package.py`
- [ ] `output.csv` with all 250 predictions
- [ ] `chat_transcript`
- [ ] No API keys or credentials anywhere in the submission (`scripts/scan_secrets.py`)
