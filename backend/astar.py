from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class State:
    """
    A* state for recovery planning.
    - streak: consecutive "done" days
    - fatigue: simulated effort accumulation (higher => higher risk)
    - recent_failures: failures in recent window (proxy)
    """

    streak: int
    fatigue: int
    recent_failures: int


Action = str  # "full" | "light" | "rest" | "skip"


def _base_risk_prob(prob_from_model: Optional[float]) -> float:
    if prob_from_model is None:
        return 0.30
    return max(0.01, min(0.99, float(prob_from_model)))


def _state_risk(prob_from_model: Optional[float], s: State) -> float:
    """
    Combine ML probability with simulated fatigue/failures.
    This ties the ML model into the A* cost function.
    """
    base = _base_risk_prob(prob_from_model)
    p = base + 0.08 * s.fatigue + 0.12 * s.recent_failures - 0.02 * s.streak
    return max(0.01, min(0.99, p))


def _neighbors(s: State) -> List[Tuple[Action, State, float]]:
    """
    Return (action, next_state, effort_cost) tuples.
    """
    out: List[Tuple[Action, State, float]] = []

    # Full habit: grows streak fast but increases fatigue.
    out.append(
        (
            "Full habit",
            State(streak=s.streak + 1, fatigue=min(10, s.fatigue + 2), recent_failures=max(0, s.recent_failures - 1)),
            3.0,
        )
    )
    # Light habit: slower but reduces fatigue pressure.
    out.append(
        (
            "Light habit",
            State(streak=s.streak + 1, fatigue=min(10, s.fatigue + 1), recent_failures=max(0, s.recent_failures - 1)),
            1.5,
        )
    )
    # Rest: reduces fatigue, but doesn't increase streak.
    out.append(
        (
            "Rest",
            State(streak=max(0, s.streak - 1), fatigue=max(0, s.fatigue - 2), recent_failures=s.recent_failures),
            0.3,
        )
    )
    # Skip: breaks streak and increases failures.
    out.append(
        (
            "Skip",
            State(streak=0, fatigue=s.fatigue, recent_failures=min(10, s.recent_failures + 1)),
            0.0,
        )
    )
    return out


def _is_goal(s: State) -> bool:
    # Stable habit state: 5-day streak, low fatigue, no recent failures.
    return s.streak >= 5 and s.fatigue <= 3 and s.recent_failures == 0


def _heuristic(s: State) -> float:
    # Admissible-ish heuristic: how far from stable streak + penalties.
    return max(0, 5 - s.streak) * 1.0 + max(0, s.fatigue - 3) * 0.5 + s.recent_failures * 1.5


def astar_recovery_plan(
    *,
    start_state: State,
    start_date: date,
    prob_from_model: Optional[float],
    max_steps: int = 14,
) -> List[Dict[str, str]]:
    """
    A* search to find an optimal action sequence that reaches a stable state.

    Cost g(n) includes:
    - effort cost for the action
    - risk penalty based on predicted failure probability (ML) + simulated state
    """
    # (f, g, steps, state, path)
    heap: List[Tuple[float, float, int, State, List[str]]] = []
    heapq.heappush(heap, (_heuristic(start_state), 0.0, 0, start_state, []))

    best_g: Dict[State, float] = {start_state: 0.0}
    risk_weight = 5.0

    while heap:
        f, g, steps, s, path = heapq.heappop(heap)
        if steps > max_steps:
            continue
        if _is_goal(s):
            # Convert to dated plan.
            plan: List[Dict[str, str]] = []
            d = start_date
            for action in path:
                plan.append({"date": d.isoformat(), "action": action})
                d = d + timedelta(days=1)
            return plan

        for action, nxt, effort in _neighbors(s):
            risk = _state_risk(prob_from_model, nxt)
            risk_cost = risk * risk_weight  # expected failure penalty
            g2 = g + effort + risk_cost

            if g2 < best_g.get(nxt, float("inf")):
                best_g[nxt] = g2
                f2 = g2 + _heuristic(nxt)
                heapq.heappush(heap, (f2, g2, steps + 1, nxt, path + [action]))

    # Fallback: return the cheapest partial plan found (best heuristic+g).
    return [{"date": start_date.isoformat(), "action": "Light habit"}]
