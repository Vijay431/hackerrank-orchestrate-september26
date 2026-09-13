# Usage report

Run: `--set full` on 2026-09-13T16:55:54+05:30, 250 requests, 66.9s wall clock.

The deterministic engine makes no model calls. The only LLM usage is the
extraction of structured facts from `messages.csv` and the 16 images, which is
cached on disk (`cache/amendments.json`, `cache/image_amounts.json`). Figures
below are the calls actually issued by this run; a warm cache reports zero.

| provider | model | calls | input tokens | output tokens | total tokens |
|---|---|---:|---:|---:|---:|
| openai | gpt-5.6-luna | 208 | 270195 | 23602 | 293797 |

- Total model calls: **208**
- Total tokens: **293797** (input 270195, output 23602)
- Average tokens per request: **1175.19**
- Estimated total cost: **USD 0.5738** (list price input 1.25/M, output 10.00/M)
- Estimated cost per request: **USD 0.002295**

Cumulative cost of populating the cache before this run is recorded in
`cache/usage_history.json` (appended by every run that issued calls).
