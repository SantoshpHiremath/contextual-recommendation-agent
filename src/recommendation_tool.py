"""
recommendation_tool.py
-------------------------

Wraps the recommendation engine as a callable "tool" with a stable,
serializable input/output shape -- the same tool-calling pattern used
by the agent in the sibling `rag-tool-agent-demo` project
(`@tool`-decorated functions an LLM agent can invoke) and the
`rag-tool-mcp-server` project (the same tools exposed over the Model
Context Protocol). This is deliberately kept dependency-light (no
LangChain/MCP import here) so this project's own test suite doesn't
require those packages to be installed -- but the function signature
and JSON-serializable return shape are designed to be trivially
wrapped by either pattern, which `run_pipeline.py` demonstrates
directly by wrapping it as a LangChain-style callable.
"""
from __future__ import annotations

from src.recommendation_engine import recommend
from src.situation_model import build_situation
from src.user_profile import Interaction, UserProfile, build_user_profile, cold_start_profile


def get_contextual_recommendations(
    user_id: str,
    interaction_history: list,
    current_time_index: int,
    hour: int,
    distance_km: float,
    is_familiar_route: bool,
    battery_soc_percent: float,
    passenger_count: int,
    top_n: int = 3,
) -> dict:
    """The single entry point an agent (or an MCP tool wrapper) would
    call: takes raw interaction history and raw situational context,
    builds both models internally, and returns a JSON-serializable
    dict -- not a Python dataclass -- so this is directly usable as a
    tool return value without extra marshalling code at the call site.

    `interaction_history` is a list of dicts (not Interaction objects)
    since that's the shape a real caller (an agent framework, an API
    request body) would actually hand this function -- converting to
    the internal Interaction dataclass happens inside, not pushed onto
    every caller.
    """
    interactions = [
        Interaction(
            user_id=row["user_id"],
            action_category=row["action_category"],
            accepted=row["accepted"],
            timestamp_index=row["timestamp_index"],
        )
        for row in interaction_history
    ]

    if interactions:
        profile = build_user_profile(user_id, interactions, current_time_index)
    else:
        profile = cold_start_profile(user_id)

    situation = build_situation(
        hour=hour,
        distance_km=distance_km,
        is_familiar_route=is_familiar_route,
        battery_soc_percent=battery_soc_percent,
        passenger_count=passenger_count,
    )

    recommendations = recommend(profile, situation, top_n=top_n)

    return {
        "user_id": user_id,
        "is_cold_start": profile.n_interactions == 0,
        "situation": {
            "time_bucket": situation.time_bucket,
            "trip_type": situation.trip_type,
            "battery_urgency": situation.battery_urgency,
            "occupancy": situation.occupancy,
        },
        "recommendations": [
            {"action_category": r.action_category, "score": round(r.score, 3), "reason": r.reason}
            for r in recommendations
        ],
    }
