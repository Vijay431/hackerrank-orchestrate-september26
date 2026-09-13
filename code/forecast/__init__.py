"""90-day balance forecast.

recurrence.py  -- infer recurring series from history and project them forward
amendments.py  -- normalise events (FX, image amounts, cash-state rules) and
                  apply message amendments
simulate.py    -- daily balance series, trough queries, safe-amount solvers

`build_forecast` (the contracts.py entry point) composes the three; it is
wired here once all three modules exist.
"""
