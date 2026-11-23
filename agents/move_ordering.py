"""Move ordering for Alpha-Beta search."""

import time

from constants import BOARD_SIZE
from agents.evaluation import evaluate, terminal_utility
from agents.game_state import StateTimeout, apply_move, legal_moves, terminal


def move_code(move):
    """Return (source, destination, size), using -1 for reserve moves."""
    source = -1
    if move.src is not None:
        source = move.src[0] * BOARD_SIZE + move.src[1]
    destination = move.dest[0] * BOARD_SIZE + move.dest[1]
    return source, destination, move.size


def order_moves(state, preferred_move=None, deadline=None):
    """Return (move, child) pairs for a nonterminal state.

    A legal preferred move comes first; an illegal hint is ignored.
    """
    scored_moves = []
    preferred = None

    for move in legal_moves(state, deadline):
        child = apply_move(state, move)
        finished, winner, _ = terminal(child, deadline)

        if move == preferred_move:
            preferred = (move, child)
            continue

        # Use the perspective of the player choosing the move.
        if finished:
            score = terminal_utility(child, state.player, winner)
        else:
            score = evaluate(child, state.player)
        scored_moves.append((score, move, child))

    scored_moves.sort(key=lambda item: (-item[0], move_code(item[1])))

    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    for _, move, child in scored_moves:
        ordered.append((move, child))

    if deadline is not None and time.perf_counter() >= deadline:
        raise StateTimeout("move-ordering deadline exceeded")
    return ordered
