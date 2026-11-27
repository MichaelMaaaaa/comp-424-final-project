"""Original fixed-depth Negamax, shared by the component and optimization variants."""

from dataclasses import dataclass
from functools import lru_cache
import time

from constants import BOARD_SIZE, MAX_PLIES
from agents.game_state import StateTimeout, apply_pass, is_legal_move, must_pass
from agents.move_ordering import SearchFunctions, search_children
from agents.symmetry import (TRANSFORMS, inverse_transform_move, transform_cell,
                             transform_move)
from agents.transposition_table import EXACT, LOWER, UPPER, encode_state


@dataclass(frozen=True)
class SearchConfig:
    alpha_beta: bool = True
    ordering: bool = True
    tt: bool = True
    preferred_first: bool = False
    reuse_results: bool = False
    key_mode: str = "tuple"
    symmetry: bool = False
    tt_capacity: int = 50000

    def __post_init__(self):
        for name in ("alpha_beta", "ordering", "tt", "preferred_first", "reuse_results", "symmetry"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.key_mode not in ("tuple", "compact"):
            raise ValueError("key_mode must be tuple or compact")
        if isinstance(self.tt_capacity, bool) or not isinstance(self.tt_capacity, int) or self.tt_capacity < 0:
            raise ValueError("tt_capacity must be a nonnegative integer")
        if not self.ordering and (self.preferred_first or self.reuse_results):
            raise ValueError("ordering optimizations require ordering")
        if not self.tt and (self.symmetry or self.key_mode != "tuple"):
            raise ValueError("key optimizations require TT")


def _pack_search_key(state, board=None):
    if not 0 <= state.ply <= MAX_PLIES:
        raise ValueError("compact TT keys require an official-range ply")
    packed = 0
    for stack in state.board if board is None else board:
        for owner in stack:
            packed = (packed << 2) | owner
    for row in state.reserves[1:]:
        for count in row:
            packed = (packed << 2) | count
    packed = (packed << 2) | state.player
    packed = (packed << 7) | state.ply
    packed = (packed << 1) | int(state.allow_moves)
    packed = (packed << 1) | (state.win_length - 3)
    return packed.to_bytes(15, "big")


@lru_cache(maxsize=16)
def obstacle_stabilizers(obstacle):
    """Use B's transform API; only transforms fixing THIS obstacle are allowed."""
    result = []
    for transform in TRANSFORMS:
        if transform_cell(obstacle, transform) != obstacle:
            continue
        permutation = [0] * (BOARD_SIZE * BOARD_SIZE)
        for index in range(BOARD_SIZE * BOARD_SIZE):
            row, col = transform_cell(divmod(index, BOARD_SIZE), transform)
            permutation[row * BOARD_SIZE + col] = index
        result.append((transform, tuple(permutation)))
    return tuple(result)


def search_key(state, config):
    """Return the complete key and its transform. Symmetric keys require evaluation
    scores to be invariant under every obstacle-preserving transform.
    """
    if not config.symmetry:
        return (_pack_search_key(state) if config.key_mode == "compact" else encode_state(state)), 0
    best = chosen = None
    for transform, permutation in obstacle_stabilizers(state.obstacle):
        board = tuple(state.board[index] for index in permutation)
        if config.key_mode == "compact":
            key = _pack_search_key(state, board)
        else:
            key = (board, state.reserves, state.player, state.obstacle, state.ply,
                   state.allow_moves, state.win_length)
        if best is None or key < best:
            best, chosen = key, transform
    return best, chosen


@dataclass
class SearchStats:
    """Count recursive calls across rounds, including interrupted work."""
    nodes: int = 0


def alpha_beta(state, depth, alpha=-float("inf"), beta=float("inf"), deadline=None,
               preferred_move=None, stats=None, tt=None, *, config=SearchConfig(),
               calls=None, prepared=None, root_scores=None, on_preferred_child=None):
    """Return (value, best_move, nodes). A supplied root_scores list requests exact ties."""
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if alpha >= beta:
        raise ValueError("alpha must be less than beta")
    if calls is None:
        calls = SearchFunctions()
    counters = calls.counters
    if stats is not None:
        stats.nodes += 1

    # terminal() checks the deadline at recursive entry, including depth zero.
    if prepared is None:
        outcome = calls.terminal(state, deadline)
    else:
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateTimeout("search deadline exceeded")
        outcome = prepared[0]
    if outcome[0]:
        if counters is not None:
            counters.terminal_returns += 1
        return calls.utility(state, state.player, outcome[1]), None, 1
    if depth == 0:
        if counters is not None:
            counters.depth_returns += 1
        if prepared is not None and prepared[1] is not None:
            return prepared[1], None, 1
        return calls.evaluate(state, state.player), None, 1
    if not state.allow_moves and must_pass(state, deadline):
        value, _, nodes = alpha_beta(apply_pass(state), depth - 1, -beta, -alpha,
                                     deadline, stats=stats, tt=tt, config=config, calls=calls)
        return -value, None, nodes + 1

    if not config.tt:
        tt = None
    alpha_start, beta_start = alpha, beta
    if tt is not None:
        if counters is not None:
            started = time.perf_counter()
        key, transform = search_key(state, config)
        if counters is not None:
            counters.key_seconds += time.perf_counter() - started
            counters.tt_probes += 1
            started = time.perf_counter()
        entry = tt.get(key, depth)
        if counters is not None:
            counters.tt_seconds += time.perf_counter() - started
        if entry is not None and entry.best_move is not None:
            move = inverse_transform_move(entry.best_move, transform) if transform else entry.best_move
            if is_legal_move(state, move):
                if counters is not None:
                    counters.tt_hits += 1
                if root_scores is None:
                    if entry.flag == EXACT:
                        if counters is not None:
                            counters.tt_exact_returns += 1
                        return entry.value, move, 1
                    if ((entry.flag == LOWER and entry.value >= beta)
                            or (entry.flag == UPPER and entry.value <= alpha)):
                        if counters is not None:
                            counters.tt_bound_returns += 1
                        return entry.value, move, 1

        # The original never tightens the window on a non-cutting bound.
        if config.ordering and preferred_move is None:
            if counters is not None:
                started = time.perf_counter()
            preferred_move = tt.get_move(key)
            if counters is not None:
                counters.tt_seconds += time.perf_counter() - started
            if preferred_move is not None:
                if transform:
                    preferred_move = inverse_transform_move(preferred_move, transform)
                if counters is not None and is_legal_move(state, preferred_move):
                    counters.tt_hint_hits += 1

    best_value, best_move, nodes = -float("inf"), None, 1
    reuse = {} if config.reuse_results else None
    prune = config.alpha_beta and root_scores is None
    children = search_children(state, preferred_move, deadline, config, calls, reuse)
    for index, (move, child) in enumerate(children):
        if on_preferred_child is not None and config.ordering and index == 0 and move == preferred_move:
            on_preferred_child()
        lo, hi = (-beta, -alpha) if prune else (-float("inf"), float("inf"))
        child_value, _, child_nodes = alpha_beta(
            child, depth - 1, lo, hi, deadline, stats=stats, tt=tt, config=config,
            calls=calls, prepared=reuse[move] if reuse is not None else None)
        value = -child_value
        nodes += child_nodes
        if root_scores is not None:
            root_scores.append((move, value))
        if value > best_value:
            best_value, best_move = value, move

        if prune:
            alpha = max(alpha, best_value)
            if alpha >= beta:
                if counters is not None:
                    counters.cutoffs += 1
                    counters.first_child_cutoffs += int(index == 0)
                    counters.preferred_cutoffs += int(config.ordering and index == 0 and move == preferred_move)
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
        stored_move = transform_move(best_move, transform) if transform else best_move
        if counters is not None:
            started = time.perf_counter()
        tt.store(key, depth, best_value, flag, stored_move)
        if counters is not None:
            counters.tt_seconds += time.perf_counter() - started
            if (key, depth) in tt.entries:
                counters.tt_stores += 1
            else:
                counters.tt_capacity_rejections += 1
    return best_value, best_move, nodes
