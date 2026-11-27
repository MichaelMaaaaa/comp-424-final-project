"""Move ordering for Alpha-Beta search."""

import time

from constants import BOARD_SIZE
from agents.evaluation import evaluate, terminal_utility
from agents.game_state import (StateTimeout, apply_move, is_legal_move,
                               iter_legal_moves, legal_moves, terminal)


class SearchFunctions:
    """The same game operations serve ordering and recursion. Time them only in diagnostics."""

    def __init__(self, evaluator=evaluate, utility=terminal_utility, counters=None):
        self.evaluate = evaluator
        self.utility = utility
        self.terminal = terminal
        self.child = apply_move
        self.counters = counters
        if counters is None:
            return

        def measured(function, count, seconds):
            def call(*args):
                setattr(counters, count, getattr(counters, count) + 1)
                started = time.perf_counter()
                result = function(*args)
                elapsed = time.perf_counter() - started
                setattr(counters, seconds, getattr(counters, seconds) + elapsed)
                return result
            return call

        self.evaluate = measured(evaluator, "evaluation_calls", "evaluation_seconds")
        self.terminal = measured(terminal, "terminal_calls", "terminal_seconds")
        self.child = measured(apply_move, "successors", "successor_seconds")


def move_code(move):
    """Return (source, destination, size), using -1 for reserve moves."""
    source = -1
    if move.src is not None:
        source = move.src[0] * BOARD_SIZE + move.src[1]
    destination = move.dest[0] * BOARD_SIZE + move.dest[1]
    return source, destination, move.size


def order_moves(state, preferred_move=None, deadline=None, *, calls=None, prepared=None, skip=None):
    """Return eagerly scored (move, child) pairs, with a legal preferred move first."""
    if calls is None:
        calls = SearchFunctions()
    scored_moves = []
    preferred = None
    for move in legal_moves(state, deadline):
        if move == skip:
            continue
        child = calls.child(state, move)
        outcome = calls.terminal(child, deadline)

        if move == preferred_move:
            preferred = (move, child)
            if prepared is not None:
                prepared[move] = (outcome, None)
            continue

        if outcome[0]:
            score = calls.utility(child, state.player, outcome[1])
        else:
            score = calls.evaluate(child, state.player)
        scored_moves.append((score, move, child))
        if prepared is not None:
            # The zero-sum leaf score is needed from the child's perspective.
            prepared[move] = (outcome, None if outcome[0] else -score)

    scored_moves.sort(key=lambda item: (-item[0], move_code(item[1])))
    if calls.counters is not None:
        calls.counters.sorted_children += len(scored_moves)
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    for _, move, child in scored_moves:
        ordered.append((move, child))

    if deadline is not None and time.perf_counter() >= deadline:
        raise StateTimeout("move-ordering deadline exceeded")
    return ordered


def search_children(state, preferred, deadline, config, calls, prepared):
    if not config.ordering:
        for move in iter_legal_moves(state, deadline):
            yield move, calls.child(state, move)
        return
    skip = None
    if config.preferred_first and preferred is not None and is_legal_move(state, preferred):
        child = calls.child(state, preferred)
        outcome = calls.terminal(child, deadline)
        if prepared is not None:
            prepared[preferred] = (outcome, None)
        yield preferred, child
        skip = preferred
    yield from order_moves(state, preferred, deadline, calls=calls, prepared=prepared, skip=skip)
