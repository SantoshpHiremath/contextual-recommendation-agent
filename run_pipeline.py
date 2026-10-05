"""
run_pipeline.py
-----------------

End-to-end demo: builds a user profile from synthetic interaction
history, derives a situation from raw trip context, and asks the
recommendation engine for a ranked list -- run across several different
situations for the SAME user, to demonstrate directly that context
changes the ranking (the actual point of combining a user profile with
a situation model, rather than running two independent systems).

Run with:
    python3 run_pipeline.py
"""
from __future__ import annotations

from src.recommendation_tool import get_contextual_recommendations

# A synthetic interaction history for one user: strongly prefers music
# suggestions, dislikes calendar reminders, is neutral on the rest.
INTERACTION_HISTORY = [
    {"user_id": "u_demo", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 0},
    {"user_id": "u_demo", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 3},
    {"user_id": "u_demo", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 6},
    {"user_id": "u_demo", "action_category": "music_suggestion", "accepted": True, "timestamp_index": 9},
    {"user_id": "u_demo", "action_category": "calendar_reminder", "accepted": False, "timestamp_index": 2},
    {"user_id": "u_demo", "action_category": "calendar_reminder", "accepted": False, "timestamp_index": 5},
    {"user_id": "u_demo", "action_category": "calendar_reminder", "accepted": False, "timestamp_index": 8},
]
CURRENT_TIME_INDEX = 10

SITUATIONS = [
    {
        "label": "Solo morning commute, ample battery, familiar route",
        "hour": 8, "distance_km": 15, "is_familiar_route": True,
        "battery_soc_percent": 70, "passenger_count": 0,
    },
    {
        "label": "Same user, long unfamiliar trip, low battery margin",
        "hour": 8, "distance_km": 180, "is_familiar_route": False,
        "battery_soc_percent": 15, "passenger_count": 0,
    },
    {
        "label": "Same user, midday errand, with a passenger",
        "hour": 13, "distance_km": 8, "is_familiar_route": False,
        "battery_soc_percent": 60, "passenger_count": 1,
    },
]


def main() -> None:
    print("=" * 70)
    print("contextual-recommendation-agent -- end-to-end demo")
    print("=" * 70)
    print("\nUser 'u_demo' interaction history: strongly prefers music")
    print("suggestions, consistently rejects calendar reminders.\n")

    for scenario in SITUATIONS:
        label = scenario.pop("label")
        result = get_contextual_recommendations(
            user_id="u_demo",
            interaction_history=INTERACTION_HISTORY,
            current_time_index=CURRENT_TIME_INDEX,
            top_n=3,
            **scenario,
        )
        print("-" * 70)
        print(f"Situation: {label}")
        print(f"  Derived: {result['situation']}")
        print("  Top recommendations:")
        for rec in result["recommendations"]:
            print(f"    {rec['action_category']}: {rec['score']}  ({rec['reason']})")
        print()

    print("=" * 70)
    print("Cold-start user (no interaction history at all), same three situations:")
    for scenario in SITUATIONS:
        result = get_contextual_recommendations(
            user_id="brand_new_user",
            interaction_history=[],
            current_time_index=0,
            top_n=1,
            hour=scenario["hour"], distance_km=scenario["distance_km"],
            is_familiar_route=scenario["is_familiar_route"],
            battery_soc_percent=scenario["battery_soc_percent"],
            passenger_count=scenario["passenger_count"],
        )
        top = result["recommendations"][0]
        print(f"  {result['situation']['battery_urgency']:>10} battery, {result['situation']['occupancy']:>16} -> top pick: {top['action_category']}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- Two different context-sensitivity effects are visible above, and")
    print("  it's worth being precise about which is which:")
    print("  (1) The cold-start user's TOP PICK changes across all three")
    print("      situations (calendar_reminder -> charging_stop_suggestion ->")
    print("      navigation_suggestion) -- pure situational relevance, no")
    print("      profile signal to compete with it.")
    print("  (2) 'u_demo' has a strong, consistent music preference (built from")
    print("      4 accepted music suggestions in their history) that is strong")
    print("      enough to stay the #1 pick across all three situations here --")
    print("      but the FULL ranked list still shifts underneath it every time")
    print("      (charging_stop_suggestion jumps to #2 only when battery is")
    print("      genuinely tight; call_suggestion is excluded outright once a")
    print("      passenger is present). A strong preference outranking a mild")
    print("      situational boost is the intended behavior of this")
    print("      design -- not every situation should override a user's clearly")
    print("      established preference -- and test_same_profile_different_")
    print("      situation_gives_different_top_pick in the test suite confirms")
    print("      the top pick DOES change for a user without that strong,")
    print("      one-category-dominant history.")
    print("- A cold-start user (zero interaction history) still gets sensible,")
    print("  situationally-relevant recommendations via a deliberate fallback, rather")
    print("  than a crash or a fabricated preference guess.")
    print("- See src/langchain_tool_wrapper.py for how this plugs into the same")
    print("  @tool-based agent pattern used in the sibling rag-tool-agent-demo")
    print("  project -- kept optional so this project's core test suite has zero")
    print("  hard LangChain dependency.")
    print("=" * 70)


if __name__ == "__main__":
    main()
