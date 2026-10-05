"""
langchain_tool_wrapper.py
----------------------------

An OPTIONAL thin wrapper exposing get_contextual_recommendations() as a
LangChain @tool, in the same style as calculator_tool.py / rag_tool.py
in the sibling rag-tool-agent-demo project -- so this recommendation
engine plugs directly into that project's existing agent pattern
(TOOLS list, create_react_agent) rather than being a disconnected
standalone module.

Deliberately kept in its own file, imported nowhere else in this
project's own test suite: the core recommendation_tool.py has zero
LangChain dependency, so `pytest tests/` runs and passes without
langchain installed at all. This file is exercised separately (see
tests/test_langchain_tool_wrapper.py, which skips itself if langchain
isn't installed) and demonstrated in run_pipeline.py's optional agent
section.
"""
from __future__ import annotations

import json

from src.recommendation_tool import get_contextual_recommendations

try:
    from langchain_core.tools import tool as _langchain_tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    _langchain_tool = None


def _contextual_recommendations_impl(
    user_id: str,
    interaction_history_json: str,
    current_time_index: int,
    hour: int,
    distance_km: float,
    is_familiar_route: bool,
    battery_soc_percent: float,
    passenger_count: int,
) -> str:
    """The actual tool logic, taking interaction_history as a JSON
    string (not a Python list) because LLM agent frameworks typically
    pass tool arguments as flat, simple types the model can generate
    directly -- a nested list-of-dicts argument is harder for a model
    to produce reliably than a JSON string it can format as text.
    Returns a JSON string for the same reason, on the output side.
    """
    history = json.loads(interaction_history_json) if interaction_history_json else []
    result = get_contextual_recommendations(
        user_id=user_id,
        interaction_history=history,
        current_time_index=current_time_index,
        hour=hour,
        distance_km=distance_km,
        is_familiar_route=is_familiar_route,
        battery_soc_percent=battery_soc_percent,
        passenger_count=passenger_count,
    )
    return json.dumps(result)


if LANGCHAIN_AVAILABLE:
    contextual_recommendations_tool = _langchain_tool(_contextual_recommendations_impl)
else:
    # Fallback: calling code that checks LANGCHAIN_AVAILABLE
    # before use will never reach this, but leaving a plain-callable
    # stand-in (rather than None) means an accidental import doesn't
    # crash at import time, only at call time with a clear error.
    contextual_recommendations_tool = _contextual_recommendations_impl
