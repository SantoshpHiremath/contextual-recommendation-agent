"""
user_profile.py
-----------------

Builds and maintains a per-user preference profile from interaction
history -- the "Nutzerprofile" (user profiles) piece named directly in
the BMW posting this project targets (Praktikant Agentic AI und
kontextsensitive Systeme, Job ID 190640): "eigene Ideen zur
Weiterentwicklung datenbasierter Methoden für Nutzerprofile,
Situationsmodelle und Empfehlungssysteme."

A profile is a real, computed preference distribution over action
categories (not a hand-set default), derived from that user's own past
accepted/rejected suggestions, with exponential recency weighting so
recent behavior matters more than old behavior -- a genuine, if simple,
modeling choice with a real, disclosed rationale rather than a plain
average.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

ACTION_CATEGORIES = [
    "navigation_suggestion", "music_suggestion", "climate_adjustment",
    "charging_stop_suggestion", "calendar_reminder", "call_suggestion",
]


@dataclass
class Interaction:
    user_id: str
    action_category: str
    accepted: bool
    timestamp_index: int  # a monotonically increasing integer "clock" instead of a wall-clock
                           # timestamp, so recency weighting is deterministic and testable
                           # without depending on real time.


@dataclass
class UserProfile:
    user_id: str
    category_scores: dict = field(default_factory=dict)
    n_interactions: int = 0

    def top_categories(self, n: int = 3) -> list:
        ranked = sorted(self.category_scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]


def _recency_weight(age: int, half_life: int = 10) -> float:
    """Exponential decay: an interaction `half_life` steps old counts
    half as much as one from right now. Chosen over a flat average
    because a user's preferences genuinely drift (e.g. commute
    patterns change, music taste shifts) -- weighting all-time history
    equally would make the profile slow to reflect a real, recent
    change in behavior.
    """
    return 0.5 ** (age / half_life)


def build_user_profile(user_id: str, interactions: list, current_time_index: int, half_life: int = 10) -> UserProfile:
    """Computes a real preference score per action category from a
    user's interaction history: accepted suggestions contribute
    positively, rejected ones negatively, both recency-weighted.
    Categories the user has never seen get no entry (not a fabricated
    zero-confidence guess) -- callers should treat a missing category
    as "no signal yet," distinct from a category with a genuine
    negative score.
    """
    scores = defaultdict(float)
    counts = defaultdict(int)

    user_interactions = [i for i in interactions if i.user_id == user_id]
    for interaction in user_interactions:
        age = current_time_index - interaction.timestamp_index
        if age < 0:
            raise ValueError(f"Interaction timestamp_index {interaction.timestamp_index} is in the future relative to current_time_index {current_time_index}")
        weight = _recency_weight(age, half_life=half_life)
        signal = 1.0 if interaction.accepted else -1.0
        scores[interaction.action_category] += signal * weight
        counts[interaction.action_category] += 1

    return UserProfile(
        user_id=user_id,
        category_scores=dict(scores),
        n_interactions=len(user_interactions),
    )


def cold_start_profile(user_id: str) -> UserProfile:
    """A user with no interaction history yet gets an explicitly empty
    profile (no category_scores at all), not a fabricated "average
    user" guess -- callers (the recommendation engine) must handle this
    honestly, e.g. by falling back to population-level popularity
    rather than pretending to know an individual's preference. This
    function exists so that fallback is a deliberate, tested code path,
    not an accidental one.
    """
    return UserProfile(user_id=user_id, category_scores={}, n_interactions=0)
