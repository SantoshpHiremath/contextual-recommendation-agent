"""Tests for src/langchain_tool_wrapper.py -- the optional LangChain
@tool wrapper. Skips itself entirely if langchain_core isn't
installed, since the core project has no hard LangChain dependency."""
from __future__ import annotations

import json

import pytest

from src.langchain_tool_wrapper import (
    LANGCHAIN_AVAILABLE,
    _contextual_recommendations_impl,
)

pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain_core not installed")


class TestContextualRecommendationsImpl:
    def test_accepts_json_string_history_and_returns_json_string(self):
        result_str = _contextual_recommendations_impl(
            user_id="u1", interaction_history_json="[]", current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        assert isinstance(result_str, str)
        parsed = json.loads(result_str)
        assert "recommendations" in parsed

    def test_empty_string_history_is_treated_as_no_history(self):
        result_str = _contextual_recommendations_impl(
            user_id="u1", interaction_history_json="", current_time_index=0,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        parsed = json.loads(result_str)
        assert parsed["is_cold_start"] is True

    def test_nonempty_history_json_string_is_parsed_correctly(self):
        history = [{"user_id": "u1", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 0}]
        result_str = _contextual_recommendations_impl(
            user_id="u1", interaction_history_json=json.dumps(history), current_time_index=1,
            hour=8, distance_km=15, is_familiar_route=True,
            battery_soc_percent=60, passenger_count=0,
        )
        parsed = json.loads(result_str)
        assert parsed["is_cold_start"] is False


class TestLangChainToolRegistration:
    def test_tool_is_registered_as_a_langchain_tool(self):
        from src.langchain_tool_wrapper import contextual_recommendations_tool
        # A real LangChain @tool-wrapped function exposes .name and .func
        assert hasattr(contextual_recommendations_tool, "name")
        assert hasattr(contextual_recommendations_tool, "func")
