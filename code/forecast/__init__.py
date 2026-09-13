"""90-day balance forecast.

recurrence.py  -- infer recurring series from history and project them forward
amendments.py  -- normalise events (FX, image amounts, cash-state rules), emit
                  one-off flows, and apply message amendments
simulate.py    -- daily balance series, trough queries, safe-amount solvers
series.py      -- frozen shared types and the CashFlow label convention

``build_forecast`` (the contracts.py entry point) composes the three modules.
``build_context`` returns the same forecast plus every intermediate the
decision layer needs (normalised events, detected series, applied
amendments) so spending changes can be evaluated without recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from contracts import (
    Amendment,
    CashFlow,
    Dataset,
    Event,
    ExtractionCache,
    Forecast,
    Profile,
    RequestRecord,
    FORECAST_DAYS,
)
from forecast.amendments import (
    apply_amendments,
    normalise_events,
    one_off_flows,
    relevant_amendments,
)
from forecast.recurrence import detect_recurring, project_all
from forecast.series import RecurringSeries
from forecast.simulate import ConcreteForecast, simulate


@dataclass(frozen=True)
class ForecastContext:
    """Everything derived for one request, in the order it was derived."""

    request: RequestRecord
    profile: Profile
    events: tuple[Event, ...]
    series: tuple[RecurringSeries, ...]
    recurring_flows: tuple[CashFlow, ...]
    one_off_flows: tuple[CashFlow, ...]
    amendments: tuple[Amendment, ...]
    flows: tuple[CashFlow, ...]
    forecast: ConcreteForecast

    def resimulate(self, flows: tuple[CashFlow, ...]) -> ConcreteForecast:
        """Same opening balance and window, different flows (spending changes)."""
        return simulate(
            self.profile.user_id,
            self.profile.current_available_balance,
            flows,
            self.request.request_date,
            FORECAST_DAYS,
        )


def build_context(
    request: RequestRecord,
    data: Dataset,
    cache: ExtractionCache,
) -> ForecastContext:
    profile = data.profiles[request.user_id]
    start: date = request.request_date
    events = normalise_events(
        data.events_by_user.get(request.user_id, ()), profile, data, cache
    )
    series = detect_recurring(events, start)
    recurring = project_all(series, start, FORECAST_DAYS)
    one_offs = one_off_flows(events, start, FORECAST_DAYS)
    amendments = relevant_amendments(
        data.messages_by_user.get(request.user_id, ()), request, cache
    )
    flows = apply_amendments(
        tuple(recurring) + tuple(one_offs), amendments, profile, data, start, FORECAST_DAYS
    )
    forecast = simulate(
        profile.user_id, profile.current_available_balance, flows, start, FORECAST_DAYS
    )
    return ForecastContext(
        request=request,
        profile=profile,
        events=tuple(events),
        series=tuple(series),
        recurring_flows=tuple(recurring),
        one_off_flows=tuple(one_offs),
        amendments=tuple(amendments),
        flows=tuple(flows),
        forecast=forecast,
    )


def build_forecast(
    request: RequestRecord,
    data: Dataset,
    cache: ExtractionCache,
) -> Forecast:
    """contracts.build_forecast: the daily balance series for one request."""
    return build_context(request, data, cache).forecast


__all__ = ["ForecastContext", "build_context", "build_forecast"]
