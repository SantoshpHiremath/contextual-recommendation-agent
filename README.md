# contextual-recommendation-agent

A tested context-sensitive recommendation system: a per-user preference profile learned from
interaction history, an explicit situation model derived from trip context, and a
recommendation engine that combines both into a ranked, explainable list of assistant actions.
It covers three building blocks of personalized, agentic systems: user profiles, situation
models, and recommendation systems.

## What it does

- **User profile** (`src/user_profile.py`): derives a per-category preference score from a
  list of accept/reject interaction events, with exponential recency weighting (recent
  behavior counts more than old behavior).
- **Situation model** (`src/situation_model.py`): derives time-of-day bucket, trip type,
  battery urgency, and occupancy from raw context inputs via small, individually testable
  rule-based functions, so every derived feature is inspectable.
- **Recommendation engine** (`src/recommendation_engine.py`): situational gating plus
  profile-weighted ranking. The same user profile produces a different full ranking depending
  on the situation.
- **Tool entry points**: a JSON-serializable, framework-agnostic tool function and an optional
  LangChain `@tool` wrapper, so the engine plugs into the agent pattern used in
  `rag-tool-agent-demo` and `rag-tool-mcp-server`.

## Scope

All data is synthetic: interaction histories and trip contexts are generated or hand-constructed
for demonstration and testing. The situation model and the recommendation scoring are
rule-based and interpretable, which suits a from-scratch project; a learned embedding or
ranker is a natural extension once interaction logs at larger scale are available. Agent-to-agent
(A2A) protocols are outside the scope of this project, which focuses on the
user-profile / situation-model / recommendation-system layer.

## Results

Running `python3 run_pipeline.py` shows two distinct effects:

1. **A cold-start user (no interaction history) gets a different top recommendation in every
   one of three tested situations**: `calendar_reminder` → `charging_stop_suggestion` →
   `navigation_suggestion`. This is pure situational relevance with no competing profile
   signal.
2. **A user with a strong, consistent preference (4/4 accepted music suggestions) keeps music
   as their #1 pick across the same three situations**, because that preference score is
   large enough to outrank the situational boosts in this design. A strong preference should
   be hard to override with a mild situational nudge. The full ranked list still shifts
   underneath the #1 pick in every situation: `charging_stop_suggestion` jumps to #2 only when
   battery margin is tight, and `call_suggestion` is excluded outright once a passenger is
   present. A dedicated test
   (`test_same_profile_different_situation_gives_different_top_pick`) confirms the top pick
   does change for a profile without a single dominant category.

## Tests

```
$ python3 -m pytest tests/ -v
============================== 57 passed in 0.26s ==============================
```

All 57 tests pass (53 core + 4 LangChain-wrapper tests). In a clean virtualenv with only
`pytest` installed (no LangChain), the 53 core tests pass and the 4 LangChain-specific tests
skip cleanly, so the core recommendation logic has no hard dependency on any agent framework.

## Project structure

```
src/
  user_profile.py            -- recency-weighted preference profile from interaction history
  situation_model.py           -- explicit, rule-based trip/context situation derivation
  recommendation_engine.py      -- combines profile + situation into a ranked list
  recommendation_tool.py         -- JSON-serializable tool entry point (framework-agnostic)
  langchain_tool_wrapper.py       -- optional @tool wrapper for LangChain agent integration
tests/
  test_user_profile.py
  test_situation_model.py
  test_recommendation_engine.py
  test_recommendation_tool.py
  test_langchain_tool_wrapper.py   -- self-skips if langchain_core isn't installed
run_pipeline.py                     -- end-to-end demo
```

## Running it

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```

## Notes

The project is designed to plug into the same agent pattern as `rag-tool-agent-demo` and
`rag-tool-mcp-server` (see `langchain_tool_wrapper.py`), so personalized, context-aware
recommendations can serve as a tool for an agent rather than a standalone system.

## Possible extensions

- Learn trip-type inference per user from historical route data.
- Replace the rule-based scoring with a learned ranker once interaction logs are available.
- Add agent-to-agent (A2A) interfaces on top of the tool entry point.
