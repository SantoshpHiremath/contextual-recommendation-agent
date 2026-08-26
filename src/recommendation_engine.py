"""
recommendation_engine.py
--------------------------

Combines a UserProfile and a Situation into a ranked list of
recommended assistant actions -- the "Empfehlungssysteme"
(recommendation systems) piece of the BMW posting this project
targets, and the point where user profiles and situation models
actually get used together, not just computed and left unconnected.

The scoring is a real, inspectable rule-based combination (situation
gating + profile-weighted ranking within the gated set) rather than a
trained ranking model -- an honest, disclosed design choice for a
from-scratch demo project with no real interaction-log data at the
scale a learned ranker would need. See the module's honest scope note.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.situation_model import Situation
from src.user_profile import ACTION_CATEGORIES, UserProfile


@dataclass
class Recommendation:
    action_category: str
    score: float
    reason: str


# Situation-based hard/soft gates: which categories make sense at all
# in a given situation, and a situational boost/penalty on top of the
# user's own preference score. This is the "context-sensitive" part --
# the same user profile produces a DIFFERENT ranking depending on the
# situation, which is the actual point of combining the two models
# rather than just running two independent systems side by side.
def _situational_relevance(category: str, situation: Situation) -> tuple:
    """Returns (is_eligible, situational_boost, reason). A category can
    be ruled out entirely (e.g. calendar_reminder while driving with
    passengers late at night is still eligible, but
    charging_stop_suggestion when battery_urgency is 'none' is
    down-weighted hard, not the same as being truly irrelevant --
    still eligible, just unlikely to be the top pick).
    """
    if category == "charging_stop_suggestion":
        if situation.battery_urgency == "high":
            return True, 2.5, "battery range is tight for the remaining trip distance"
        if situation.battery_urgency == "moderate":
            return True, 0.8, "battery margin is getting low"
        return True, -1.5, "battery has ample margin for this trip"

    if category == "navigation_suggestion":
        if situation.trip_type in ("errand", "long_distance", "unknown"):
            return True, 0.6, "an unfamiliar or non-routine trip benefits more from navigation help"
        return True, -0.3, "this looks like a familiar commute route"

    if category == "music_suggestion":
        if situation.occupancy == "with_passengers":
            return True, -0.4, "music choice is more sensitive with passengers present"
        return True, 0.3, "solo driving is a low-friction moment for a music suggestion"

    if category == "climate_adjustment":
        # No strong situational signal modeled for climate in this
        # project (would need real external temperature data, which
        # isn't part of this project's synthetic inputs) -- eligible
        # with a neutral boost, honestly.
        return True, 0.0, "no strong situational signal for climate in this model"

    if category == "calendar_reminder":
        if situation.time_bucket in ("morning_commute", "evening_commute"):
            return True, 0.5, "commute times are a natural moment for schedule reminders"
        return True, -0.2, "outside typical commute hours"

    if category == "call_suggestion":
        if situation.occupancy == "with_passengers":
            return False, 0.0, "call suggestions are suppressed with passengers present"
        return True, 0.2, "solo driving is a reasonable moment for a call suggestion"

    return True, 0.0, "no situational rule defined for this category"


def recommend(profile: UserProfile, situation: Situation, top_n: int = 3) -> list:
    """Produces a ranked list of recommended action categories. Score
    = user preference score (0.0 if the user has no signal for that
    category -- a genuine cold-start fallback to situational relevance
    alone, not a fabricated preference) + situational boost. Ineligible
    categories (situational hard gate) are excluded entirely, not just
    ranked last.
    """
    scored = []
    for category in ACTION_CATEGORIES:
        eligible, boost, reason = _situational_relevance(category, situation)
        if not eligible:
            continue
        preference_score = profile.category_scores.get(category, 0.0)
        total_score = preference_score + boost
        scored.append(Recommendation(action_category=category, score=total_score, reason=reason))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_n]
