"""Tests for src/recommendation_tool.py -- the JSON-serializable tool
entry point wrapping profile + situation + recommendation together."""
from __future__ import annotations

import json

from src.recommendation_tool import get_contextual_recommendations


class TestGetContextualRecommendations:
    def test_returns_a_json_serializable_dict(self):
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=[], current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        # Should not raise -- confirms every value in the structure is
        # a plain JSON-compatible type, not e.g. a dataclass instance.
        json.dumps(result)

    def test_empty_history_produces_cold_start_flag(self):
        result = get_contextual_recommendations(
            user_id="new_user", interaction_history=[], current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        assert result["is_cold_start"] is True

    def test_nonempty_history_produces_non_cold_start_flag(self):
        history = [
            {"user_id": "u1", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 0},
        ]
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=history, current_time_index=1,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        assert result["is_cold_start"] is False

    def test_recommendations_list_is_present_and_nonempty(self):
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=[], current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        assert len(result["recommendations"]) > 0

    def test_situation_summary_reflects_the_inputs(self):
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=[], current_time_index=0,
            hour=8, distance_km=150, is_familiar_route=True,
            battery_soc_percent=10, passenger_count=0,
        )
        assert result["situation"]["battery_urgency"] == "high"
        assert result["situation"]["trip_type"] == "long_distance"

    def test_history_from_a_different_user_is_ignored(self):
        history = [
            {"user_id": "someone_else", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 0},
        ]
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=history, current_time_index=1,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        # u1 has no history of their own in this batch, even though the
        # history list is nonempty -- should still be treated as cold start.
        assert result["is_cold_start"] is True

    def test_each_recommendation_entry_has_expected_keys(self):
        result = get_contextual_recommendations(
            user_id="u1", interaction_history=[], current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        for rec in result["recommendations"]:
            assert set(rec.keys()) == {"action_category", "score", "reason"}
