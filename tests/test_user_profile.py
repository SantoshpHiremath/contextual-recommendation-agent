"""Tests for src/user_profile.py -- interaction-history-derived user
preference profiles with recency weighting."""
from __future__ import annotations

import pytest

from src.user_profile import (
    Interaction,
    build_user_profile,
    cold_start_profile,
)


class TestBuildUserProfile:
    def test_accepted_interactions_produce_positive_score(self):
        interactions = [Interaction("u1", "music_suggestion", True, 0)]
        profile = build_user_profile("u1", interactions, current_time_index=0)
        assert profile.category_scores["music_suggestion"] > 0

    def test_rejected_interactions_produce_negative_score(self):
        interactions = [Interaction("u1", "calendar_reminder", False, 0)]
        profile = build_user_profile("u1", interactions, current_time_index=0)
        assert profile.category_scores["calendar_reminder"] < 0

    def test_filters_to_only_the_requested_user(self):
        interactions = [
            Interaction("u1", "music_suggestion", True, 0),
            Interaction("u2", "music_suggestion", True, 0),
        ]
        profile = build_user_profile("u1", interactions, current_time_index=0)
        assert profile.n_interactions == 1

    def test_more_recent_interactions_weigh_more_than_older_ones(self):
        # Two users with the same net accept/reject count, but one has
        # their positive signal more recent -- recency weighting should
        # make the recent-positive user's score higher.
        recent_positive = [
            Interaction("u1", "music_suggestion", False, 0),
            Interaction("u1", "music_suggestion", True, 9),
        ]
        recent_negative = [
            Interaction("u1", "music_suggestion", True, 0),
            Interaction("u1", "music_suggestion", False, 9),
        ]
        profile_a = build_user_profile("u1", recent_positive, current_time_index=10)
        profile_b = build_user_profile("u1", recent_negative, current_time_index=10)
        assert profile_a.category_scores["music_suggestion"] > profile_b.category_scores["music_suggestion"]

    def test_categories_never_seen_are_absent_not_zero(self):
        interactions = [Interaction("u1", "music_suggestion", True, 0)]
        profile = build_user_profile("u1", interactions, current_time_index=0)
        assert "calendar_reminder" not in profile.category_scores

    def test_future_timestamp_raises(self):
        interactions = [Interaction("u1", "music_suggestion", True, 10)]
        with pytest.raises(ValueError):
            build_user_profile("u1", interactions, current_time_index=5)

    def test_top_categories_returns_highest_scoring_first(self):
        interactions = [
            Interaction("u1", "music_suggestion", True, 0),
            Interaction("u1", "music_suggestion", True, 1),
            Interaction("u1", "calendar_reminder", False, 0),
        ]
        profile = build_user_profile("u1", interactions, current_time_index=1)
        top = profile.top_categories(n=1)
        assert top[0][0] == "music_suggestion"

    def test_no_interactions_for_user_gives_empty_profile(self):
        interactions = [Interaction("u2", "music_suggestion", True, 0)]
        profile = build_user_profile("u1", interactions, current_time_index=0)
        assert profile.category_scores == {}
        assert profile.n_interactions == 0


class TestColdStartProfile:
    def test_has_no_category_scores(self):
        profile = cold_start_profile("brand_new_user")
        assert profile.category_scores == {}

    def test_has_zero_interactions(self):
        profile = cold_start_profile("brand_new_user")
        assert profile.n_interactions == 0

    def test_preserves_user_id(self):
        profile = cold_start_profile("brand_new_user")
        assert profile.user_id == "brand_new_user"
