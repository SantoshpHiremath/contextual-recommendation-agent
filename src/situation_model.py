"""
situation_model.py
--------------------

Encodes the current driving/trip context into a structured "situation"
-- the "Situationsmodelle" (situation models) piece named in the BMW
posting. A situation is not a black-box embedding here; it's an
explicit, inspectable set of derived context features (time-of-day
bucket, trip-type guess, battery-state urgency, occupancy), each
computed by a real, testable rule from raw sensor/context inputs -- an
honest, interpretable design choice for a portfolio project (a
production system might use a learned embedding instead, see the
module's honest scope note below).
"""
from __future__ import annotations

from dataclasses import dataclass


TIME_BUCKETS = ["early_morning", "morning_commute", "midday", "evening_commute", "night"]
TRIP_TYPES = ["commute", "errand", "long_distance", "unknown"]


def _time_of_day_bucket(hour: int) -> str:
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be 0-23, got {hour}")
    if 5 <= hour < 7:
        return "early_morning"
    if 7 <= hour < 10:
        return "morning_commute"
    if 10 <= hour < 16:
        return "midday"
    if 16 <= hour < 19:
        return "evening_commute"
    return "night"


def _infer_trip_type(distance_km: float, time_bucket: str, is_familiar_route: bool) -> str:
    """A simple, disclosed heuristic (not a trained classifier): a
    short, familiar route during a commute time bucket is inferred as
    a commute; a short route outside commute hours is an errand; a
    long route is long_distance regardless of time; anything else is
    unknown rather than a forced guess. HONEST SCOPE NOTE: a production
    system would likely learn this from historical route data per user
    rather than a fixed rule -- this is a real, working, but simpler
    starting point, disclosed as such rather than presented as a
    trained model.
    """
    if distance_km > 80:
        return "long_distance"
    if distance_km <= 25 and is_familiar_route and time_bucket in ("morning_commute", "evening_commute"):
        return "commute"
    if distance_km <= 25:
        return "errand"
    return "unknown"


@dataclass
class Situation:
    time_bucket: str
    trip_type: str
    battery_urgency: str  # "none", "moderate", "high"
    occupancy: str  # "solo", "with_passengers"
    hour: int
    distance_km: float
    battery_soc_percent: float


def _battery_urgency(soc_percent: float, distance_km: float) -> str:
    """A genuine, if simple, rule connecting remaining range need to
    urgency: this is what would drive a charging_stop_suggestion
    recommendation being ranked highly or not at all. A rough
    'range needed vs range available' proxy -- soc_percent as a stand-in
    for remaining range, distance_km as a stand-in for the trip's
    range requirement -- since this project has no real vehicle range
    model.

    IMPORTANT: this is a MARGIN-relative rule, not an absolute SOC
    threshold. 15% SOC before a 10km errand is "none" (plenty of
    margin for that specific trip); the same 15% SOC before a 100km
    trip is "high" (the estimated range doesn't cover the distance at
    all). This is a deliberate design choice -- urgency should depend
    on whether THIS trip is achievable, not on SOC in isolation -- but
    it means a low SOC alone does not automatically mean "high"; test
    cases and any caller reasoning about this function should keep the
    distance term in mind, not just the SOC term.
    """
    if not 0 <= soc_percent <= 100:
        raise ValueError(f"battery_soc_percent must be 0-100, got {soc_percent}")
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")

    # A rough proxy: assume ~4.5 km of range per 1% SOC (a plausible
    # real-EV ballpark, not measured from a real vehicle).
    estimated_range_km = soc_percent * 4.5
    margin = estimated_range_km - distance_km

    if margin < 0:
        return "high"
    if margin < 40:
        return "moderate"
    return "none"


def build_situation(
    hour: int,
    distance_km: float,
    is_familiar_route: bool,
    battery_soc_percent: float,
    passenger_count: int,
) -> Situation:
    """Builds a full Situation from raw context inputs. Each field is
    independently computed and testable -- a caller (or a test) can
    check any one derived field without needing to reason about the
    whole structure at once.
    """
    time_bucket = _time_of_day_bucket(hour)
    trip_type = _infer_trip_type(distance_km, time_bucket, is_familiar_route)
    battery_urgency = _battery_urgency(battery_soc_percent, distance_km)
    occupancy = "solo" if passenger_count == 0 else "with_passengers"

    return Situation(
        time_bucket=time_bucket,
        trip_type=trip_type,
        battery_urgency=battery_urgency,
        occupancy=occupancy,
        hour=hour,
        distance_km=distance_km,
        battery_soc_percent=battery_soc_percent,
    )
