"""Tests for src/situation_model.py -- deriving an explicit,
inspectable situation from raw context inputs."""
from __future__ import annotations

import pytest

from src.situation_model import build_situation


class TestTimeOfDayBucket:
    @pytest.mark.parametrize("hour,expected", [
        (5, "early_morning"), (6, "early_morning"),
        (7, "morning_commute"), (9, "morning_commute"),
        (10, "midday"), (15, "midday"),
        (16, "evening_commute"), (18, "evening_commute"),
        (19, "night"), (23, "night"), (0, "night"), (4, "night"),
    ])
    def test_hour_maps_to_expected_bucket(self, hour, expected):
        situation = build_situation(hour=hour, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        assert situation.time_bucket == expected

    def test_invalid_hour_raises(self):
        with pytest.raises(ValueError):
            build_situation(hour=24, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)


class TestTripTypeInference:
    def test_short_familiar_commute_hour_route_is_commute(self):
        situation = build_situation(hour=8, distance_km=15, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        assert situation.trip_type == "commute"

    def test_short_unfamiliar_route_is_not_commute(self):
        situation = build_situation(hour=8, distance_km=15, is_familiar_route=False, battery_soc_percent=50, passenger_count=0)
        assert situation.trip_type != "commute"

    def test_short_route_outside_commute_hours_is_errand(self):
        situation = build_situation(hour=13, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        assert situation.trip_type == "errand"

    def test_long_route_is_long_distance_regardless_of_time(self):
        situation = build_situation(hour=8, distance_km=150, is_familiar_route=True, battery_soc_percent=90, passenger_count=0)
        assert situation.trip_type == "long_distance"


class TestBatteryUrgency:
    def test_ample_margin_is_none(self):
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=80, passenger_count=0)
        assert situation.battery_urgency == "none"

    def test_insufficient_range_is_high(self):
        situation = build_situation(hour=10, distance_km=100, is_familiar_route=True, battery_soc_percent=10, passenger_count=0)
        assert situation.battery_urgency == "high"

    def test_low_soc_but_short_trip_is_not_automatically_high(self):
        # The core "margin, not absolute SOC" design point -- a
        # regression test for the exact case that looked surprising
        # during manual testing.
        situation = build_situation(hour=8, distance_km=10, is_familiar_route=True, battery_soc_percent=15, passenger_count=0)
        assert situation.battery_urgency != "high"

    def test_tight_margin_is_moderate(self):
        situation = build_situation(hour=10, distance_km=50, is_familiar_route=True, battery_soc_percent=15, passenger_count=0)
        assert situation.battery_urgency == "moderate"

    def test_invalid_soc_raises(self):
        with pytest.raises(ValueError):
            build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=150, passenger_count=0)

    def test_zero_distance_raises(self):
        with pytest.raises(ValueError):
            build_situation(hour=10, distance_km=0, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)


class TestOccupancy:
    def test_zero_passengers_is_solo(self):
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        assert situation.occupancy == "solo"

    def test_one_or_more_passengers_is_with_passengers(self):
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=2)
        assert situation.occupancy == "with_passengers"


class TestSituationRetainsRawInputs:
    def test_hour_distance_and_soc_are_retained_on_the_object(self):
        situation = build_situation(hour=8, distance_km=42.5, is_familiar_route=True, battery_soc_percent=33.3, passenger_count=1)
        assert situation.hour == 8
        assert situation.distance_km == 42.5
        assert situation.battery_soc_percent == 33.3
