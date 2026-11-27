"""Iterative deepening with a shared search deadline."""

from dataclasses import dataclass, field
import time

from constants import MAX_PLIES
from helpers import Move
from agents.alpha_beta import SearchConfig, SearchStats, alpha_beta
from agents.game_state import StateTimeout, iter_legal_moves
from agents.move_ordering import SearchFunctions
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
    iteration_elapsed: list = field(default_factory=list)
    optimal_moves: tuple = ()
    preferred_used: bool = False
    stop_reason: str | None = None
    interrupted_depth: int | None = None


def validate_max_depth(max_depth):
    if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0):
        raise ValueError("max_depth must be None or a nonnegative integer")


def search(state, deadline, *, max_depth=None, preferred_move=None, collect_ties=False,
           config=SearchConfig(), calls=None, tt=None):
    """Commit only complete root iterations, using one TT for this online step."""
    validate_max_depth(max_depth)
    started = time.perf_counter()
    if calls is None:
        calls = SearchFunctions()
    # Resolve terminal and fallback states even if the search deadline has passed.
    finished, winner, _ = calls.terminal(state)
    if finished:
        return SearchResult(value=calls.utility(state, state.player, winner),
                            elapsed=time.perf_counter() - started, status="terminal", stop_reason="terminal")

    result = SearchResult(move=next(iter_legal_moves(state), None))
    stats = calls.counters if calls.counters is not None else SearchStats()
    if config.tt and tt is None:
        tt = TranspositionTable(config.tt_capacity)
    limit = MAX_PLIES - state.ply
    result.stop_reason = "remaining_ply_limit"
    if max_depth is not None and max_depth < limit:
        limit = max_depth
        result.stop_reason = "depth_limit"

    def mark_preferred_used():
        result.preferred_used = True

    for depth in range(1, limit + 1):
        if time.perf_counter() >= deadline:
            result.stop_reason = "deadline_before_iteration"
            break
        root_scores = [] if collect_ties or not config.alpha_beta else None
        try:
            value, move, _ = alpha_beta(
                state, depth, -float("inf"), float("inf"), deadline=deadline,
                preferred_move=preferred_move, stats=stats, tt=tt, config=config,
                calls=calls, root_scores=root_scores,
                on_preferred_child=mark_preferred_used if depth == 1 and preferred_move is not None else None)
        except StateTimeout:
            result.stop_reason = "deadline_during_iteration"
            result.interrupted_depth = depth
            if calls.counters is not None:
                calls.counters.interrupted_iterations += 1
            break

        finished_at = time.perf_counter()
        if finished_at >= deadline:
            result.stop_reason = "deadline_during_iteration"
            result.interrupted_depth = depth
            if calls.counters is not None:
                calls.counters.interrupted_iterations += 1
            break
        result.move = move
        result.completed_depth = depth
        result.value = value
        result.status = "completed"
        result.iterations.append((depth, move, value))
        result.iteration_elapsed.append(finished_at - started)
        if root_scores is not None:
            result.optimal_moves = tuple(m for m, score in root_scores if score == value)
        preferred_move = move

    result.nodes = stats.nodes
    result.elapsed = time.perf_counter() - started
    return result
