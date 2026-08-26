"""Tests for src/recommendation_engine.py -- combining a user profile
and a situation into a ranked, context-sensitive recommendation list."""
from __future__ import annotations

from src.recommendation_engine import recommend
from src.situation_model import build_situation
from src.user_profile import UserProfile, cold_start_profile


class TestRecommend:
    def test_returns_at_most_top_n_recommendations(self):
        profile = cold_start_profile("u1")
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        recs = recommend(profile, situation, top_n=2)
        assert len(recs) <= 2

    def test_results_are_sorted_by_descending_score(self):
        profile = UserProfile(user_id="u1", category_scores={"music_suggestion": 5.0, "climate_adjustment": 1.0})
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        recs = recommend(profile, situation, top_n=6)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_high_battery_urgency_boosts_charging_stop_to_the_top(self):
        # The central "context changes the ranking" claim: a user with
        # NO particular preference for charging suggestions should
        # still see it ranked first when battery urgency is high.
        profile = cold_start_profile("u1")
        situation = build_situation(hour=10, distance_km=150, is_familiar_route=True, battery_soc_percent=10, passenger_count=0)
        assert situation.battery_urgency == "high"
        recs = recommend(profile, situation, top_n=1)
        assert recs[0].action_category == "charging_stop_suggestion"

    def test_same_profile_different_situation_gives_different_top_pick(self):
        # The core "context-sensitive" claim, demonstrated directly:
        # the SAME user profile produces a DIFFERENT top recommendation
        # depending on the situation.
        profile = cold_start_profile("u1")

        low_battery_long_trip = build_situation(hour=10, distance_km=150, is_familiar_route=True, battery_soc_percent=10, passenger_count=0)
        ample_battery_short_trip = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=80, passenger_count=0)

        recs_a = recommend(profile, low_battery_long_trip, top_n=1)
        recs_b = recommend(profile, ample_battery_short_trip, top_n=1)

        assert recs_a[0].action_category != recs_b[0].action_category

    def test_call_suggestion_is_excluded_with_passengers_present(self):
        profile = cold_start_profile("u1")
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=2)
        recs = recommend(profile, situation, top_n=10)
        categories = [r.action_category for r in recs]
        assert "call_suggestion" not in categories

    def test_call_suggestion_is_eligible_when_solo(self):
        profile = cold_start_profile("u1")
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        recs = recommend(profile, situation, top_n=10)
        categories = [r.action_category for r in recs]
        assert "call_suggestion" in categories

    def test_strong_user_preference_can_still_outrank_a_situational_boost(self):
        # A user who has strongly, consistently accepted music
        # suggestions should still rank it highly even in a situation
        # that gives it a slight penalty (with passengers present).
        profile = UserProfile(user_id="u1", category_scores={"music_suggestion": 10.0})
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=2)
        recs = recommend(profile, situation, top_n=1)
        assert recs[0].action_category == "music_suggestion"

    def test_cold_start_user_still_gets_recommendations(self):
        # No preference data at all should not crash or return nothing
        # -- situational relevance alone should still produce a ranked
        # list.
        profile = cold_start_profile("brand_new_user")
        situation = build_situation(hour=8, distance_km=15, is_familiar_route=True, battery_soc_percent=60, passenger_count=0)
        recs = recommend(profile, situation, top_n=3)
        assert len(recs) > 0

    def test_every_recommendation_has_a_human_readable_reason(self):
        profile = cold_start_profile("u1")
        situation = build_situation(hour=10, distance_km=10, is_familiar_route=True, battery_soc_percent=50, passenger_count=0)
        recs = recommend(profile, situation, top_n=6)
        for r in recs:
            assert isinstance(r.reason, str) and len(r.reason) > 0
