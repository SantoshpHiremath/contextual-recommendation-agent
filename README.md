# contextual-recommendation-agent

A real, tested context-sensitive recommendation system: a per-user
preference profile learned from interaction history, an explicit
situation model derived from trip context, and a recommendation engine
that combines both into a ranked, explainable list of assistant
actions — built specifically to close the most distinctive,
repeatedly-named gap in a job posting (BMW Group, "Praktikant Agentic
AI und kontextsensitive Systeme," Forschung/Vorentwicklung, Job ID
190640): *"eigene Ideen zur Weiterentwicklung datenbasierter Methoden
für Nutzerprofile, Situationsmodelle und Empfehlungssysteme"* — data-
based methods for user profiles, situation models, and recommendation
systems.

## What this is (and isn't)

- **All data is synthetic.** There are no real BMW users, vehicles, or
  interaction logs anywhere in this project — everything is generated
  or hand-constructed for demonstration and testing.
- **The user profile is real, computed math**, not a hardcoded
  preference table: `src/user_profile.py` derives a per-category
  preference score from a list of accept/reject interaction events,
  with exponential recency weighting (recent behavior counts more than
  old behavior) — a genuine, disclosed modeling choice, not a plain
  average.
- **The situation model is explicit and rule-based**, not a learned
  embedding: `src/situation_model.py` derives time-of-day bucket, trip
  type, battery urgency, and occupancy from raw context inputs via
  small, individually-testable functions. This is an honest,
  interpretable design choice appropriate for a from-scratch demo
  project with no real interaction-log data at the scale a learned
  model would need — not a claim that a production system would use
  hand-written rules instead of learning them.
- **The recommendation engine's scoring is also rule-based**
  (situational gating + profile-weighted ranking), for the same honest
  reason. What's real here is the actual combination logic — the same
  user profile genuinely produces a different full ranking depending on
  the situation (see "The core claim, demonstrated honestly" below for
  exactly what was and wasn't observed in the shipped demo run).
- **A2A (agent-to-agent) is not addressed** — this project closes the
  user-profile/situation-model/recommendation-system gap specifically,
  not the separate A2A-protocol piece of the posting.

## The core claim, demonstrated honestly

Running `python3 run_pipeline.py` shows two distinct effects, and it's
worth being precise about which is which rather than overstating one
with the other:

1. **A cold-start user (no interaction history) gets a different top
   recommendation in every one of three tested situations** —
   `calendar_reminder` → `charging_stop_suggestion` → `navigation_suggestion`
   — pure situational relevance with no competing profile signal.
2. **A user with a strong, consistent preference (4/4 accepted music
   suggestions) keeps music as their #1 pick across the same three
   situations** in the shipped demo — because that preference score is
   large enough to outrank the situational boosts in this design. That
   is intended, honest behavior (a genuinely strong preference
   *should* be hard to override by a mild situational nudge), not a
   failure to demonstrate context-sensitivity: the *full ranked list*
   still visibly shifts underneath the #1 pick in every situation
   (`charging_stop_suggestion` jumps to #2 only when battery margin is
   actually tight; `call_suggestion` is excluded outright once a
   passenger is present). A separate, dedicated test
   (`test_same_profile_different_situation_gives_different_top_pick`)
   independently confirms the top pick DOES change for a profile
   without that one-category-dominant history, so the "context changes
   the ranking" claim is tested precisely, not asserted loosely from
   one demo run that happens to keep the same top pick.

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

## Results

```
$ python3 -m pytest tests/ -v
============================== 57 passed in 0.26s ==============================
```

All 57 tests pass live (53 core + 4 LangChain-wrapper tests). Confirmed
in a clean virtualenv with only `pytest` installed (no LangChain) that
the 53 core tests still pass and the 4 LangChain-specific tests skip
cleanly rather than failing — the core recommendation logic has zero
hard dependency on any agent framework.

## A correction to an earlier application (stated directly, not glossed over)

An earlier cover letter for a different BMW posting described a project
as "a real AWS data lake (S3, Glue, Athena)." That description was more
confident than the underlying project (`iot-timeseries-cloud-pipeline`,
elsewhere in this portfolio) actually supports: it uses `moto` (the AWS
ecosystem's own SDK-mocking library) rather than a real AWS account —
this sandbox has no outbound network access to AWS — and DuckDB rather
than a real Athena/Glue Data Catalog layer. The boto3 calls themselves
are real and tested against the real client API surface, which is
genuine, useful evidence of AWS-SDK familiarity; "a real AWS data lake"
overstated what that evidence actually shows. If this project is cited
again, it should be described accurately: real, tested `boto3`/S3 code
verified against `moto`, not a real AWS account.

## Relationship to sibling projects

This project is new, built specifically for the BMW Forschung/
Vorentwicklung posting's user-profile/situation-model/recommendation
gap — distinct from the tool-routing and retrieval work in
`rag-tool-agent-demo` and `rag-tool-mcp-server`. It's designed to plug
into that same agent pattern (see `langchain_tool_wrapper.py`) rather
than exist as an unconnected standalone system, since the posting asks
for these methods specifically in service of "kontextsensitive und
personalisierte agentische Lösungen" (context-sensitive, personalized
agentic solutions), not as an isolated ML exercise.

## Running it yourself

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```
