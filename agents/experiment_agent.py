"""Shared online entry point. F injects evaluation; B injects book decisions."""

from dataclasses import asdict
import json
import math
import time

from agents.agent import Agent
from agents.game_state import GameState, StateTimeout, is_legal_move
from agents.search_engine import SearchConfig, SearchEngine, default_evaluator
from helpers import get_valid_moves


class ExperimentAgent(Agent):
    def __init__(self, variant=None, evaluation_factory=None, book_factory=None):
        super().__init__()
        self.variant = dict(variant or {"variant_id": "S0_E1_N", "search": {},
                                       "evaluation": {"evaluation_id": "E1"}, "book": {"mode": "N"}})
        self.name = self.variant["variant_id"]
        budget = self.variant.get("step_budget", 1.6)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
            raise ValueError("step_budget must be finite and nonnegative")
        if budget > 1.6:
            raise ValueError("online experiment budget must not exceed 1.6 seconds")
        self.budget = budget
        self.call_count = 0
        self.last_info = None
        factory = evaluation_factory or default_evaluator
        self.engine = SearchEngine(SearchConfig(**self.variant.get("search", {})),
                                   evaluator=factory(self.variant.get("evaluation")), diagnostics=False)
        book_config = self.variant.get("book", {"mode": "N"})
        self.book_hook = book_factory(book_config, self.variant) if book_factory is not None else None
        if book_config.get("mode", "N") != "N" and self.book_hook is None:
            raise ValueError("a non-N book mode requires B's book_factory(config, variant)")

    def _report(self, info, started):
        info["processing_before_report_s"] = time.perf_counter() - started
        self.last_info = info
        if self.variant.get("telemetry", False):
            print("EXP_STEP " + json.dumps(info, separators=(",", ":")), flush=True)

    def step(self, board, reserves, player, opponent):
        started = time.perf_counter()
        # Keep reporting/return overhead inside the same whole-step budget.
        deadline = started + max(0.0, self.budget - min(0.002, self.budget / 10))
        ply = 2 * self.call_count + player - 1
        self.call_count += 1
        moves = get_valid_moves(board, reserves, player)
        fallback = moves[0]
        info = {"variant_id": self.name, "ply": ply, "completed_depth": None,
                "status": "single_move", "book_queried": False, "book_matched": False, "book_action": "none"}
        if len(moves) == 1:
            self._report(info, started)
            return fallback
        state = GameState.from_observation(board, reserves, player, ply)
        preferred = None
        if self.book_hook is not None and time.perf_counter() < deadline:
            info["book_queried"] = True
            try:
                decision = self.book_hook(state, tuple(moves), deadline)
            except StateTimeout:
                decision = {}
            info.update(book_matched=bool(decision.get("matched", False)),
                        book_version=decision.get("version"), book_metadata=decision.get("metadata", {}))
            direct = decision.get("direct_move")
            if direct is not None and is_legal_move(state, direct) and time.perf_counter() < deadline:
                info.update(status="book", book_action="direct")
                self._report(info, started)
                return direct
            hint = decision.get("preferred_move")
            if hint is not None and is_legal_move(state, hint):
                preferred = hint
                info["book_action"] = "hint"
        result = self.engine.search(state, deadline, self.variant.get("max_depth"), preferred)
        info.update(status=result.status, completed_depth=result.completed_depth,
                    value=result.value, search_elapsed_s=result.elapsed,
                    iterations=[{"depth": r["depth"], "value": r["value"], "elapsed": r["elapsed"]}
                                for r in result.iterations], solved_remaining=result.solved_remaining)
        self._report(info, started)
        return result.move if result.move in moves else fallback
