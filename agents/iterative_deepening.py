"""Iterative deepening with a shared search deadline."""

from dataclasses import dataclass, field
import time

from constants import MAX_PLIES
from helpers import Move
from agents.alpha_beta import SearchStats, alpha_beta
from agents.evaluation import terminal_utility
from agents.game_state import StateTimeout, iter_legal_moves, terminal
from agents.transposition_table import TranspositionTable


@dataclass
class SearchResult:
    """Root result; iterations stores completed (depth, move, value) tuples."""
    move: Move | None = None
    completed_depth: int = 0
    value: int | None = None
    nodes: int = 0
    elapsed: float = 0.0
    status: str = "fallback"
    iterations: list = field(default_factory=list)


def search(state, deadline):
    """Search the official course variant until the shared deadline.

    deadline is an absolute perf_counter time supplied by the caller.
    elapsed measures this call. Status is terminal, fallback, or completed;
    completed means at least one depth finished, not that the game is solved.
    """
    started = time.perf_counter()

    # Bounded preparation also handles an already-expired deadline.
    finished, winner, _ = terminal(state)
    if finished:
        return SearchResult(
            value=terminal_utility(state, state.player, winner),
            elapsed=time.perf_counter() - started, status="terminal"
        )

    result = SearchResult(move=next(iter_legal_moves(state)))
    stats = SearchStats()
    tt = TranspositionTable()
    preferred_move = None

    for depth in range(1, MAX_PLIES - state.ply + 1):
        if time.perf_counter() >= deadline:
            break

        try:
            value, move, _ = alpha_beta(
                state, depth, -float("inf"), float("inf"),
                deadline=deadline, preferred_move=preferred_move, stats=stats,
                tt=tt
            )
        except StateTimeout:
            break

        # Commit only a complete, on-time root search.
        if time.perf_counter() >= deadline:
            break
        result.move = move
        result.completed_depth = depth
        result.value = value
        result.status = "completed"
        result.iterations.append((depth, move, value))
        preferred_move = move

    result.nodes = stats.nodes
    result.elapsed = time.perf_counter() - started
    return result
