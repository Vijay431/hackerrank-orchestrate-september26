# decision_explanation molds

Derived by normalising all 25 rows of `dataset/sample_requests.csv`. Placeholders:
`{C}` currency code, `{A}` an amount, `{M}` the user's `minimum_balance_to_keep`,
`{REQ}` `requested_amount`, `{ASP}` `amount_safe_to_pay`, `{D}` a date, `{N}` a count,
`{DESC}` a financial event's `description` lowercased.

Verified: in 23 of 25 samples the last quoted amount equals that user's
`minimum_balance_to_keep` exactly. The 2 exceptions are mold 4b, which quotes no minimum.

## 1. affordable_now + full_payment
- **1a** (request_01, request_16)
  `Pay {C} {A} today. This leaves at least {C} {M} available over the next 90 days.`
- **1b** (request_09)
  `Pay {C} {A} today. This keeps the {C} {M} minimum available over the next 90 days.`

## 2. affordable_later + wait
- **2a** (request_03, request_08, request_13, request_18, request_23)
  `Pay {C} {A} in full on {D}. Paying earlier would take the balance below the {C} {M} minimum.`
- **2b** (request_04)
  `Wait until {D}, then pay {C} {A} in full. Paying sooner would put the {C} {M} minimum at risk.`

## 3. affordable_with_plan + installments (request_02, _07, _12, _17, _22)
`Use {N} installments of {C} {A}, starting {D}. This leaves at least {C} {M} available.`

## 4. not_affordable + not_recommended
- **4a** (request_05, _10, _15, _20, _25)
  `Do not make this payment by {D}. None of the available options keeps the {C} {M} minimum protected.`
  `{D}` is `desired_completion_date`.
- **4b** (request_14, request_24) — used when some amount IS safe today but the full request
  cannot complete within 90 days
  `Do not proceed with the {C} {REQ} request. Although {C} {ASP} is available today, the full amount cannot be completed safely within 90 days.`

## 5. affordable_with_plan + full_payment, with spending changes
Prefix names the changes, then the same suffix.
- stop only (request_06): `Stop the {DESC}, then pay {C} {A} today. This leaves at least {C} {M} available.`
- reduce only (request_11): `Reduce the {DESC} to {C} {A}, then pay {C} {A} today. This leaves at least {C} {M} available.`
- both (request_21): `Stop the {DESC} and reduce the {DESC} to {C} {A}, then pay {C} {A} today. This leaves at least {C} {M} available.`

## 6. affordable_with_plan + partial_payment (request_19)
`Pay {C} {A} today and the remaining {C} {B} on {D}. This completes the full request and keeps the {C} {M} minimum protected.`

## Number and date formatting
- Thousands separators always: `25,256`, `15,952,906.67`, `60,496,000`.
- Integers print with no decimals (`68,432`); non-integers print to exactly 2dp
  (`620.40` for a stored `620.4`, `23.50`, `95,194.67`).
- Currency code precedes the amount, single space: `INR 274,600`.
- Dates are `D Month YYYY` with NO leading zero: `12 September 2024`, `8 August 2025`.
- `{DESC}` comes from the event's `description` column, lowercased
  (`Weekend food delivery` -> `the weekend food delivery`).

## Caveat for Gate 4
Molds 1 and 2 each have two variants with no visible selector in the inputs, so the
ground-truth generator chose between them by some rule we cannot see. Exact
character-match on all 25 is therefore NOT achievable from status alone. Target instead:
correct mold family + every substituted value correct. Scoring for this column is
"usefulness and consistency", not exact match, so variant choice is low-risk.
