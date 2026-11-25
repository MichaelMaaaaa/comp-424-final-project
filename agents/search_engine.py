"""Configurable search for experiment modules 0-2; no position-book access."""

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import math
import sys
import time

from constants import BOARD_SIZE, MAX_PLIES
from agents.evaluation import evaluate, evaluate_e0, terminal_utility
from agents.game_state import (StateTimeout, apply_move, apply_pass, is_legal_move,
                               iter_legal_moves, legal_moves, must_pass, terminal)
from agents.symmetry import (TRANSFORMS, inverse_transform_move, transform_cell,
                             transform_move)
from agents.transposition_table import (EXACT, LOWER, UPPER, TranspositionTable,
                                        encode_state)


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


def component_configs():
    return {
        "C0": SearchConfig(alpha_beta=False, ordering=False, tt=False),
        "C1": SearchConfig(ordering=False, tt=False),
        "C2": SearchConfig(tt=False),
        "C3": SearchConfig(),
    }


def default_evaluator(config=None):
    """Compatibility adapter; F may supply make_evaluator(config) instead.

    Only the original E0/E1 are implemented here. Parameter experiments belong
    to evaluation.py; unknown parameters fail rather than silently using E1.
    """
    config = dict(config or {"evaluation_id": "E1"})
    name = config.get("evaluation_id", "E1")
    strength = config.get("lambda", 0 if name == "E0" else 1)
    sizes = tuple(config.get("q", (1, 20, 100)))
    if name == "E0" and strength == 0:
        return evaluate_e0
    if name == "E1" and strength == 1 and sizes == (1, 20, 100):
        return evaluate
    raise ValueError("custom evaluation requires F's make_evaluator(config) provider")


def action_code(move):
    src = -1 if move.src is None else move.src[0] * BOARD_SIZE + move.src[1]
    return src, move.dest[0] * BOARD_SIZE + move.dest[1], move.size


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
    """Return (complete key, original-to-key transform); no book canonicalizer."""
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
class SearchCounters:
    nodes: int = 0
    successors: int = 0
    terminal_calls: int = 0
    evaluation_calls: int = 0
    cutoffs: int = 0
    first_child_cutoffs: int = 0
    preferred_cutoffs: int = 0
    sorted_children: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_exact_returns: int = 0
    tt_bound_returns: int = 0
    tt_hint_hits: int = 0
    tt_stores: int = 0
    tt_capacity_rejections: int = 0
    key_seconds: float = 0.0
    tt_seconds: float = 0.0
    terminal_returns: int = 0
    depth_returns: int = 0
    interrupted_iterations: int = 0
    successor_seconds: float = 0.0
    terminal_seconds: float = 0.0
    evaluation_seconds: float = 0.0


@dataclass
class EngineResult:
    move: object = None
    value: object = None
    completed_depth: int = 0
    elapsed: float = 0.0
    status: str = "fallback"
    iterations: list = field(default_factory=list)
    optimal_moves: tuple = ()
    statistics: dict = field(default_factory=dict)
    solved_remaining: bool = False


class SearchEngine:
    def __init__(self, config=None, evaluator=None, utility=None, diagnostics=False):
        self.config = config if isinstance(config, SearchConfig) else SearchConfig(**(config or {}))
        self.evaluator = evaluator or evaluate
        self.utility = utility or terminal_utility
        self.diagnostics = diagnostics
        self.counters = SearchCounters()
        self.tt = TranspositionTable(self.config.tt_capacity)
        self._root_scores = []

    def _check(self, deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateTimeout("search deadline exceeded")

    def _terminal(self, state, deadline):
        if self.diagnostics:
            self.counters.terminal_calls += 1
            tick = time.perf_counter()
        result = terminal(state, deadline)
        if self.diagnostics:
            self.counters.terminal_seconds += time.perf_counter() - tick
        return result

    def _evaluate(self, state):
        if self.diagnostics:
            self.counters.evaluation_calls += 1
            tick = time.perf_counter()
        result = self.evaluator(state, state.player)
        if self.diagnostics:
            self.counters.evaluation_seconds += time.perf_counter() - tick
        return result

    def _child(self, state, move):
        if self.diagnostics:
            self.counters.successors += 1
            tick = time.perf_counter()
        result = apply_move(state, move)
        if self.diagnostics:
            self.counters.successor_seconds += time.perf_counter() - tick
        return result

    def _children(self, state, preferred, deadline):
        config = self.config
        if not config.ordering:
            for move in iter_legal_moves(state, deadline):
                yield move, self._child(state, move), None
            return
        early = preferred is not None and config.preferred_first and is_legal_move(state, preferred)
        if early:
            child = self._child(state, preferred)
            outcome = self._terminal(child, deadline)
            yield preferred, child, (outcome, None) if config.reuse_results else None
        scored = []
        delayed = None
        for move in iter_legal_moves(state, deadline):
            self._check(deadline)
            if early and move == preferred:
                continue
            child = self._child(state, move)
            outcome = self._terminal(child, deadline)
            if move == preferred:
                delayed = (move, child, (outcome, None) if config.reuse_results else None)
                continue
            value = self.utility(child, child.player, outcome[1]) if outcome[0] else self._evaluate(child)
            prepared = (outcome, value if not outcome[0] else None) if config.reuse_results else None
            scored.append((-value, action_code(move), move, child, prepared))
            if self.diagnostics:
                self.counters.sorted_children += 1
        scored.sort(key=lambda item: (-item[0], item[1]))
        self._check(deadline)
        if delayed is not None:
            yield delayed
        for _, _, move, child, prepared in scored:
            self._check(deadline)
            yield move, child, prepared

    def _search(self, state, depth, alpha, beta, deadline, preferred=None, prepared=None,
                root=False, collect_ties=False):
        self._check(deadline)
        if self.diagnostics:
            self.counters.nodes += 1
        outcome = prepared[0] if prepared is not None else self._terminal(state, deadline)
        if outcome[0]:
            if self.diagnostics:
                self.counters.terminal_returns += 1
            return self.utility(state, state.player, outcome[1]), None
        if depth == 0:
            if self.diagnostics:
                self.counters.depth_returns += 1
            return (prepared[1] if prepared is not None and prepared[1] is not None
                    else self._evaluate(state)), None
        if not state.allow_moves and must_pass(state, deadline):
            value, _ = self._search(apply_pass(state), depth - 1, -beta, -alpha, deadline)
            return -value, None
        alpha0, beta0 = alpha, beta
        key, transform = None, 0
        if self.config.tt:
            tick = time.perf_counter() if self.diagnostics else 0
            key, transform = search_key(state, self.config)
            if self.diagnostics:
                self.counters.key_seconds += time.perf_counter() - tick
                self.counters.tt_probes += 1
            tick = time.perf_counter() if self.diagnostics else 0
            entry = self.tt.get(key, depth)
            hint = self.tt.get_move(key)
            if self.diagnostics:
                self.counters.tt_seconds += time.perf_counter() - tick
            move = None if entry is None else entry.best_move
            if move is not None and transform:
                move = inverse_transform_move(move, transform)
            if entry is not None and is_legal_move(state, move):
                if self.diagnostics:
                    self.counters.tt_hits += 1
                # Root tie collection must establish every root action's exact value.
                if not (root and collect_ties):
                    if entry.flag == EXACT:
                        if self.diagnostics:
                            self.counters.tt_exact_returns += 1
                        return entry.value, move
                    if ((entry.flag == LOWER and entry.value >= beta)
                            or (entry.flag == UPPER and entry.value <= alpha)):
                        if self.diagnostics:
                            self.counters.tt_bound_returns += 1
                        return entry.value, move
            if self.config.ordering and preferred is None and hint is not None:
                if transform:
                    hint = inverse_transform_move(hint, transform)
                if is_legal_move(state, hint):
                    preferred = hint
                    if self.diagnostics:
                        self.counters.tt_hint_hits += 1
        if not self.config.ordering:
            preferred = None
        best, best_move = -math.inf, None
        for index, (move, child, reuse) in enumerate(self._children(state, preferred, deadline)):
            full = not self.config.alpha_beta or (root and collect_ties)
            lo, hi = (-math.inf, math.inf) if full else (-beta, -alpha)
            value, _ = self._search(child, depth - 1, lo, hi, deadline, prepared=reuse)
            value = -value
            if root and collect_ties:
                self._root_scores.append((move, value))
            if value > best:
                best, best_move = value, move
            if self.config.alpha_beta and not (root and collect_ties):
                alpha = max(alpha, best)
                if alpha >= beta:
                    if self.diagnostics:
                        self.counters.cutoffs += 1
                        self.counters.first_child_cutoffs += int(index == 0)
                        self.counters.preferred_cutoffs += int(index == 0 and move == preferred)
                    break
        self._check(deadline)
        if self.config.tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta0 else EXACT
            canonical_move = transform_move(best_move, transform) if transform else best_move
            at_capacity = len(self.tt.entries) >= self.config.tt_capacity and (key, depth) not in self.tt.entries
            tick = time.perf_counter() if self.diagnostics else 0
            self.tt.store(key, depth, best, flag, canonical_move)
            if self.diagnostics:
                self.counters.tt_seconds += time.perf_counter() - tick
                self.counters.tt_capacity_rejections += int(at_capacity)
                self.counters.tt_stores += int(not at_capacity)
        return best, best_move

    def statistics(self):
        if not self.diagnostics:
            return {"enabled": False}
        # Approximate retained Python-object memory, measured only in diagnostics.
        seen = set()
        def retained_bytes(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            result = sys.getsizeof(value)
            if isinstance(value, dict):
                result += sum(retained_bytes(k) + retained_bytes(v) for k, v in value.items())
            elif isinstance(value, (tuple, list)):
                result += sum(retained_bytes(v) for v in value)
            elif hasattr(value, "__dict__"):
                result += retained_bytes(vars(value))
            return result
        return dict(enabled=True, **asdict(self.counters), tt_entries=len(self.tt.entries),
                    approximate_tt_bytes=retained_bytes((self.tt.entries, self.tt.best_moves)))

    def fixed_depth(self, state, depth, deadline=None, preferred=None, collect_ties=False,
                    clear_tt=True):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        started = time.perf_counter()
        if clear_tt:
            self.tt.clear()
            self.counters = SearchCounters()
        result = EngineResult()
        self._root_scores = []
        ties = collect_ties or not self.config.alpha_beta
        try:
            value, move = self._search(state, depth, -math.inf, math.inf, deadline,
                                       preferred=preferred, root=True, collect_ties=ties)
            self._check(deadline)
        except StateTimeout:
            if self.diagnostics:
                self.counters.interrupted_iterations += 1
            result.status = "incomplete"
        else:
            result.value, result.move = value, move
            result.completed_depth = depth
            result.status = "completed"
            if ties:
                result.optimal_moves = tuple(m for m, v in self._root_scores if v == value)
            result.solved_remaining = depth >= MAX_PLIES - state.ply
        result.elapsed = time.perf_counter() - started
        result.statistics = self.statistics()
        return result

    def search(self, state, deadline, max_depth=None, preferred=None, collect_ties=False,
               tie_rng=None):
        """Iterative deepening; commit only complete, on-time root iterations."""
        started = time.perf_counter()
        self.tt.clear()
        self.counters = SearchCounters()
        # Bounded setup provides a legal fallback even for an expired deadline.
        outcome = terminal(state)
        if outcome[0]:
            return EngineResult(value=self.utility(state, state.player, outcome[1]), status="terminal",
                                elapsed=time.perf_counter() - started, statistics=self.statistics())
        fallback = next(iter_legal_moves(state), None)
        result = EngineResult(move=fallback)
        limit = MAX_PLIES - state.ply if max_depth is None else min(max_depth, MAX_PLIES - state.ply)
        for depth in range(1, limit + 1):
            if time.perf_counter() >= deadline:
                break
            iteration = self.fixed_depth(state, depth, deadline, preferred, collect_ties, clear_tt=False)
            if iteration.status != "completed":
                break
            result.move, result.value = iteration.move, iteration.value
            result.completed_depth, result.status = depth, "completed"
            result.optimal_moves = iteration.optimal_moves
            result.iterations.append({"depth": depth, "move": iteration.move,
                                      "value": iteration.value, "elapsed": time.perf_counter() - started})
            preferred = iteration.move if self.config.ordering else None
        if tie_rng is not None and result.optimal_moves:
            result.move = tie_rng.choice(result.optimal_moves)
        result.elapsed = time.perf_counter() - started
        result.statistics = self.statistics()
        result.solved_remaining = result.completed_depth >= MAX_PLIES - state.ply
        return result
