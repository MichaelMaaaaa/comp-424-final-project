"""Experiment interfaces to the original fixed-depth and iterative search modules."""

from dataclasses import asdict, dataclass, field
import sys
import time

from constants import MAX_PLIES
from agents.alpha_beta import SearchConfig, alpha_beta, obstacle_stabilizers, search_key
from agents.evaluation import evaluate, make_evaluator, terminal_utility
from agents.game_state import StateTimeout
from agents.iterative_deepening import search
from agents.move_ordering import SearchFunctions, move_code as action_code
from agents.transposition_table import TranspositionTable


def component_configs():
    return {
        "C0": SearchConfig(alpha_beta=False, ordering=False, tt=False),
        "C1": SearchConfig(ordering=False, tt=False),
        "C2": SearchConfig(tt=False),
        "C3": SearchConfig(),
    }


def default_evaluator(config=None):
    return make_evaluator(config)


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
    preferred_used: bool = False
    stop_reason: str | None = None
    interrupted_depth: int | None = None


class SearchEngine:
    """Inject a fixed evaluator with opposite scores for the two players.
    With symmetry=True, its score must also be invariant under transforms fixing the obstacle.
    """

    def __init__(self, config=None, evaluator=None, utility=None, diagnostics=False):
        self.config = config if isinstance(config, SearchConfig) else SearchConfig(**(config or {}))
        self.evaluator = evaluate if evaluator is None else evaluator
        self.utility = terminal_utility if utility is None else utility
        self.diagnostics = diagnostics
        self.counters = SearchCounters()
        self.tt = TranspositionTable(self.config.tt_capacity)

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
        counters = self.counters if self.diagnostics else None
        root_scores = [] if collect_ties or not self.config.alpha_beta else None
        result = EngineResult()
        try:
            value, move, _ = alpha_beta(
                state, depth, deadline=deadline, preferred_move=preferred, stats=counters,
                tt=self.tt if self.config.tt else None, config=self.config,
                calls=SearchFunctions(self.evaluator, self.utility, counters), root_scores=root_scores)
            if deadline is not None and time.perf_counter() >= deadline:
                raise StateTimeout("fixed-depth search returned late")
        except StateTimeout:
            if counters is not None:
                counters.interrupted_iterations += 1
            result.status = "incomplete"
            result.stop_reason = "deadline_during_iteration"
            result.interrupted_depth = depth
        else:
            result.value, result.move = value, move
            result.completed_depth = depth
            result.status = "completed"
            result.stop_reason = "fixed_depth"
            if root_scores is not None:
                result.optimal_moves = tuple(m for m, score in root_scores if score == value)
            result.solved_remaining = depth >= MAX_PLIES - state.ply
        result.elapsed = time.perf_counter() - started
        result.statistics = self.statistics()
        return result

    def search(self, state, deadline, max_depth=None, preferred=None, collect_ties=False,
               tie_rng=None):
        started = time.perf_counter()
        self.tt = TranspositionTable(self.config.tt_capacity)
        self.counters = SearchCounters()
        calls = SearchFunctions(self.evaluator, self.utility, self.counters if self.diagnostics else None)
        result = search(state, deadline, max_depth=max_depth, preferred_move=preferred,
                        collect_ties=collect_ties, config=self.config, calls=calls,
                        tt=self.tt if self.config.tt else None)
        move = result.move
        if tie_rng is not None and result.optimal_moves:
            move = tie_rng.choice(result.optimal_moves)
        iterations = [dict(depth=depth, move=m, value=value, elapsed=elapsed)
                      for (depth, m, value), elapsed in zip(result.iterations, result.iteration_elapsed)]
        return EngineResult(move=move, value=result.value, completed_depth=result.completed_depth,
                            elapsed=time.perf_counter() - started, status=result.status, iterations=iterations,
                            optimal_moves=result.optimal_moves, statistics=self.statistics(),
                            preferred_used=result.preferred_used,
                            stop_reason=result.stop_reason, interrupted_depth=result.interrupted_depth,
                            solved_remaining=(result.status == "completed"
                                              and result.completed_depth >= MAX_PLIES - state.ply))
