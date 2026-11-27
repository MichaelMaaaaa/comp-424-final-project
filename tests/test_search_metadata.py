"""Search exits and telemetry use the existing clocks, moves and raw status."""
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from agents import iterative_deepening as deep
from agents.experiment_agent import ExperimentAgent
from agents.game_state import GameState, initial_state, legal_moves, StateTimeout, to_arrays
from agents.search_engine import SearchEngine
from helpers import Move


class SearchMetadataTests(unittest.TestCase):
    def test_iteration_exit_and_last_completed_depth(self):
        state = initial_state()
        cases = [
            (2, 0, [0, 1, 1], [], "deadline_before_iteration", None, 0),
            (2, 0, [0, 0, 1], [StateTimeout()], "deadline_during_iteration", 1, 0),
            (2, 0, [0, 0, .2, .2, 1, 1], [(7, None, None)] * 2, "deadline_during_iteration", 2, 1),
            (2, 0, [0, 0, .2, 1, 1], [(7, None, None)], "deadline_before_iteration", None, 1),
            (1, 0, [0, 0, .2, .2], [(7, None, None)], "depth_limit", None, 1),
            (0, 0, [0, 0], [], "depth_limit", None, 0),
            (1, 99, [0, 0, .2, .2], [(7, None, None)], "remaining_ply_limit", None, 1),
            (None, 100, [0, 0], [], "terminal", None, 0),
        ]
        for maximum, ply, clock, results, reason, interrupted, completed in cases:
            with self.subTest(reason=reason, ply=ply, maximum=maximum), \
                    patch.object(deep.time, "perf_counter", side_effect=clock), \
                    patch.object(deep, "alpha_beta", side_effect=results):
                result = deep.search(replace(state, ply=ply), 1, max_depth=maximum)
                self.assertEqual((getattr(result, "stop_reason", None), getattr(result, "interrupted_depth", None),
                                  result.completed_depth), (reason, interrupted, completed))

    def test_engine_exposes_fixed_and_iterative_metadata_without_counters(self):
        engine = SearchEngine()
        state = initial_state()
        results = [(engine.fixed_depth(state, 0), "fixed_depth", None),
                   (engine.fixed_depth(state, 2, deadline=0), "deadline_during_iteration", 2),
                   (engine.search(state, deadline=0), "deadline_before_iteration", None)]
        for result, reason, depth in results:
            self.assertEqual(result.statistics, {"enabled": False})
            self.assertEqual((getattr(result, "stop_reason", None), result.interrupted_depth), (reason, depth))

    def test_agent_emits_bypasses_and_search_exits(self):
        board, reserves = to_arrays(initial_state())
        for label, budget, mode in [("single_move", 1, "N"), ("book_direct", 1, "D"),
                                    ("deadline_before_search", 0, "N"), ("depth_limit", 1, "N")]:
            agent = ExperimentAgent({"variant_id": "test", "max_depth": 0, "step_budget": budget,
                                     "book": {"mode": mode}},
                                    book_factory=lambda *_: lambda s, moves, t: {"matched": True,
                                                                                "direct_move": moves[0]})
            from helpers import get_valid_moves
            moves = get_valid_moves(board, reserves, 1)
            output = StringIO()
            with patch("agents.experiment_agent.get_valid_moves", return_value=moves[:1] if label == "single_move" else moves), redirect_stdout(output):
                agent.step(board, reserves, 1, 2)
            payload = json.loads(output.getvalue().removeprefix("EXP_STEP "))
            self.assertEqual(payload.get("stop_reason"), label)
            self.assertIsNone(payload["interrupted_depth"])
            self.assertEqual(payload["completed_depth"], None if label in ("single_move", "book_direct") else 0)

    def test_agent_runtime_pause_retains_completed_result_inside_whole_step_budget(self):
        state = initial_state((1, 1))
        board, reserves = to_arrays(state)
        hint = legal_moves(state)[-1]
        convert, search_depth = GameState.from_observation, deep.alpha_beta
        for budget, setup_s, completed_at, pause_s, report_s in (
                (1.6, .1, .5, .020, .005), (.1, .01, .085, .004, .001)):
            for mode in ("N", "D", "H"):
                with self.subTest(budget=budget, mode=mode):
                    started = now = 10.0
                    book_deadline = None

                    def timed_conversion(*args):
                        nonlocal now
                        now += setup_s
                        return convert(*args)

                    def lookup(node, moves, deadline):
                        nonlocal now, book_deadline
                        now += setup_s
                        book_deadline = deadline
                        return ({"matched": True, "preferred_move": hint}
                                if mode == "H" else {"matched": False})

                    def delayed_root(node, depth, *args, **kwargs):
                        nonlocal now
                        if mode != "N":
                            self.assertEqual(kwargs["deadline"], book_deadline)
                        # Complete real depth one, then consume the remaining search
                        # time plus a runtime pause after the next cooperative check.
                        now = (started + completed_at if depth == 1
                               else kwargs["deadline"] + pause_s)
                        return search_depth(node, depth, *args, **kwargs)

                    class DelayedOutput(StringIO):
                        def flush(self):
                            nonlocal now
                            now += report_s
                            super().flush()

                    agent = ExperimentAgent(
                        {"variant_id": "runtime_pause", "step_budget": budget,
                         "max_depth": 2, "book": {"mode": mode}},
                        evaluation_factory=lambda _: lambda node, player: 0,
                        book_factory=lambda *_: lookup)
                    output = DelayedOutput()
                    with patch("time.perf_counter", side_effect=lambda: now), \
                            patch.object(GameState, "from_observation", side_effect=timed_conversion), \
                            patch.object(deep, "alpha_beta", side_effect=delayed_root), redirect_stdout(output):
                        move = agent.step(board, reserves, 1, 2)
                    payload = json.loads(output.getvalue().removeprefix("EXP_STEP "))
                    self.assertEqual(move, hint if mode == "H" else Move(None, (0, 0), 0))
                    self.assertEqual((payload["status"], payload["completed_depth"], payload["value"],
                                      payload["stop_reason"], payload["interrupted_depth"]),
                                     ("completed", 1, 0, "deadline_during_iteration", 2))
                    self.assertEqual([row["depth"] for row in payload["iterations"]], [1])
                    self.assertEqual(payload["book_action"], "hint" if mode == "H" else "none")
                    self.assertLessEqual(now - started, budget)
