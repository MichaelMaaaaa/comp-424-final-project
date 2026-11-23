"""Fixed-depth Negamax with Alpha-Beta pruning."""

from dataclasses import dataclass
import time

from agents.evaluation import evaluate, terminal_utility
from agents.game_state import StateTimeout, is_legal_move, terminal
from agents.move_ordering import order_moves
from agents.transposition_table import EXACT, LOWER, UPPER, encode_state


@dataclass
class SearchStats:
    """Count recursive calls across rounds, including interrupted work."""
    nodes: int = 0


def alpha_beta(state, depth, alpha=-float("inf"),
               beta=float("inf"), deadline=None, preferred_move=None, stats=None,
               tt=None):
    """Return (value, best_move, nodes) from state.player's perspective.

    A full window gives the exact value at this depth. The deadline is an
    absolute perf_counter time; StateTimeout propagates to the caller.
    The preferred move applies only at this node.
    The node count measures recursive search calls.
    Optional stats also counts calls interrupted by a timeout.
    Optional tt reuses values only at the same remaining depth.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if alpha >= beta:
        raise ValueError("alpha must be less than beta")

    if stats is not None:
        stats.nodes += 1

    # terminal() also checks the deadline at recursive entry.
    finished, winner, _ = terminal(state, deadline)
    if finished:
        return terminal_utility(state, state.player, winner), None, 1
    if depth == 0:
        return evaluate(state, state.player), None, 1

    alpha_start, beta_start = alpha, beta
    if tt is not None:
        key = encode_state(state)
        entry = tt.get(key, depth)
        if entry is not None and is_legal_move(state, entry.best_move):
            if entry.flag == EXACT:
                return entry.value, entry.best_move, 1
            if entry.flag == LOWER and entry.value >= beta:
                return entry.value, entry.best_move, 1
            if entry.flag == UPPER and entry.value <= alpha:
                return entry.value, entry.best_move, 1

        # A non-cutting bound does not tighten the search window.
        if preferred_move is None:
            preferred_move = tt.get_move(key)

    best_value = -float("inf")
    best_move = None
    nodes = 1

    for move, child in order_moves(state, preferred_move, deadline):
        child_value, _, child_nodes = alpha_beta(
            child, depth - 1, -beta, -alpha, deadline, stats=stats, tt=tt
        )
        value = -child_value
        nodes += child_nodes

        # An equal cutoff bound does not establish a genuine tie.
        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, best_value)
        if alpha >= beta:
            break

    if tt is not None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateTimeout("search deadline exceeded before caching")
        if best_value <= alpha_start:
            flag = UPPER
        elif best_value >= beta_start:
            flag = LOWER
        else:
            flag = EXACT
        tt.store(key, depth, best_value, flag, best_move)

    return best_value, best_move, nodes
