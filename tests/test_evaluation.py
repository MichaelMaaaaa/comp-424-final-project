from dataclasses import replace
import unittest
from unittest.mock import patch

from agents import evaluation
from agents.experiment_agent import ExperimentAgent
from agents.game_state import apply_move, initial_state, to_arrays
from agents.search_engine import SearchEngine
from book_experiments.collect import online_engine
from helpers import Move


def covered_position():
    state = replace(initial_state(), win_length=4)
    for dest, size in [((0, 1), 0), ((0, 1), 1), ((1, 2), 2), ((2, 0), 1), ((2, 1), 0)]:
        state = apply_move(state, Move(None, dest, size))
    return state


class FalseyCallable:
    def __init__(self, function):
        self.function = function

    def __len__(self):
        return 0

    def __call__(self, *args):
        return self.function(*args)


class EvaluationTests(unittest.TestCase):
    def test_supplied_falsey_evaluator_is_used_in_search_and_offline(self):
        state = initial_state()
        expected_move = Move(None, (1, 2), 0)
        leaf = apply_move(state, expected_move)
        evaluator = FalseyCallable(evaluation.evaluate_e0)
        offline_config = {"search": {}, "evaluation": {"evaluation_id": "E1"}}
        for label, engine in (("direct", SearchEngine(evaluator=evaluator)),
                              ("offline", online_engine(offline_config, evaluator=evaluator))):
            with self.subTest(entry=label):
                fixed = engine.fixed_depth(state, 1)
                iterative = engine.search(state, float("inf"), max_depth=1)
                self.assertEqual((fixed.value, fixed.move, iterative.value, iterative.move,
                                  engine.fixed_depth(leaf, 0).value),
                                 (3000, expected_move, 3000, expected_move, -3000))

    def test_supplied_falsey_factory_and_factory_result_are_used_online(self):
        state = initial_state()
        variant = {"variant_id": "override", "evaluation": {"evaluation_id": "E1"},
                   "max_depth": 1, "telemetry": False}
        for label, factory in (("false-valued factory", FalseyCallable(lambda _: evaluation.evaluate_e0)),
                               ("false-valued evaluator", lambda _: FalseyCallable(evaluation.evaluate_e0))):
            with self.subTest(entry=label), patch("time.perf_counter", return_value=0):
                agent = ExperimentAgent(variant, evaluation_factory=factory)
                move = agent.step(*to_arrays(state), 1, 2)
                self.assertEqual((agent.last_info["completed_depth"], agent.last_info["value"], move),
                                 (1, 3000, Move(None, (1, 2), 0)))

    def test_supplied_falsey_terminal_utility_is_used_in_search(self):
        state = initial_state()
        for cell in ((1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3)):
            state = apply_move(state, Move(None, cell, 2 if cell[1] < 2 else 1))

        def utility(state, player, winner):
            return 0 if winner == 0 else 42 if player == winner else -42

        engine = SearchEngine(utility=FalseyCallable(utility))
        fixed = engine.fixed_depth(state, 1)
        iterative = engine.search(state, float("inf"))
        self.assertEqual((fixed.value, fixed.move, iterative.value, iterative.move, iterative.status),
                         (-42, None, -42, None, "terminal"))

    def test_default_and_reserved_evaluators_match_originals(self):
        state = covered_position()
        configs = [
            (None, evaluation.evaluate),
            ({}, evaluation.evaluate),
            ({"evaluation_id": "E1"}, evaluation.evaluate),
            ({"evaluation_id": "E1", "lambda": 1, "q": [1, 20, 100]}, evaluation.evaluate),
            ({"evaluation_id": "E0"}, evaluation.evaluate_e0),
            ({"evaluation_id": "E0", "lambda": 0, "q": [100, 1, 100]}, evaluation.evaluate_e0),
        ]
        positions = [initial_state(), state, apply_move(state, Move((0, 1), (0, 2), 1))]
        for config, original in configs:
            evaluator = evaluation.make_evaluator(config)
            for position in positions:
                for player in (1, 2):
                    with self.subTest(config=config, ply=position.ply, player=player):
                        self.assertEqual(evaluator(position, player), original(position, player))

    def test_formula_uses_only_top_pieces_and_unblocked_lines(self):
        state = covered_position()
        self.assertEqual(state.board[1], (1, 2, 0))
        without_hidden_piece = replace(state, board=state.board[:1] + ((0, 2, 0),) + state.board[2:])
        # P1 controls 3 + 3 lines and P2 controls 1 + 1. P1 shows L/S, P2 shows M/M.
        for config, score in [
            ({"evaluation_id": "E0"}, 4000),
            ({"evaluation_id": "E1"}, 4061),
            ({"lambda": 4, "q": [1, 20, 100]}, 4244),
            ({"lambda": 4, "q": [40, 3, 100]}, 4536),
        ]:
            with self.subTest(config=config):
                evaluator = evaluation.make_evaluator(config)
                self.assertEqual(evaluator(state, 1), score)
                self.assertEqual(evaluator(state, 2), -score)
                self.assertEqual(evaluator(without_hidden_piece, 1), score)

    def test_evaluation_parameters_and_factory_capture_immutable_weights(self):
        state = covered_position()
        self.assertEqual(evaluation.evaluation_parameters(), (1, (1, 20, 100)))
        self.assertEqual(evaluation.evaluation_parameters({"evaluation_id": "E0"}), (0, (1, 20, 100)))
        config = {"evaluation_id": "custom", "lambda": 4, "q": [40, 3, 100]}
        self.assertEqual(evaluation.evaluation_parameters(config), (4, (40, 3, 100)))
        evaluator = evaluation.make_evaluator(config)
        config["q"][0] = 100
        config["lambda"] = 0
        self.assertEqual(evaluator(state, 1), 4536)

    def test_invalid_evaluation_configs_fail_at_factory_creation(self):
        invalid = [
            [], {"evaluation_id": None}, {"evaluation_id": " "}, {"evaluation_id": 1},
            {"evaluation_id": "custom"}, {"evaluation_id": "custom", "lambda": 1},
            {"evaluation_id": "custom", "q": [1, 20, 100]},
            {"lambda": 1}, {"q": [1, 20, 100]}, {"lamba": 1},
            {"evaluation_id": "E0", "lambda": 1},
            {"evaluation_id": "E1", "lambda": 0},
            {"evaluation_id": "E1", "q": [1, 50, 100]},
        ]
        invalid.extend({"lambda": strength, "q": [1, 20, 100]}
                       for strength in (True, 1.0, -1, 11, float("nan"), float("inf"), "1"))
        invalid.extend({"lambda": 1, "q": sizes} for sizes in (
            None, "123", [1, 100], [1, 20, 100, 100], [True, 20, 100], [1, 20.0, 100],
            [1, float("nan"), 100], [1, float("inf"), 100], [0, 20, 100], [1, 101, 100], [1, 20, 99],
        ))
        for config in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "evaluation|lambda|q"):
                evaluation.make_evaluator(config)

    def test_direct_agent_uses_configured_leaf_and_ordering_scores(self):
        state = replace(initial_state(), win_length=4)
        config = {"evaluation_id": "custom", "lambda": 4, "q": (100, 2, 100)}
        engine = ExperimentAgent({"variant_id": "custom", "evaluation": config}).engine
        first = Move(None, (1, 2), 0)
        visited_leaves = []
        evaluator = engine.evaluator
        def record(child, player):
            if player == child.player:
                visited_leaves.append(child.board)
            return evaluator(child, player)
        engine.evaluator = record
        selected = engine.fixed_depth(state, 1)
        self.assertEqual(selected.move, first)
        self.assertEqual(visited_leaves[0], apply_move(state, first).board)
        result = engine.fixed_depth(apply_move(state, first), 0)
        self.assertEqual(result.value, -3400)

    def test_offline_book_engine_uses_configured_evaluation(self):
        config = {"evaluation_id": "custom", "lambda": 4, "q": [40, 3, 100]}
        engine = online_engine({"search": {}, "evaluation": config})
        self.assertEqual(engine.fixed_depth(covered_position(), 0).value, -4536)

    def test_terminal_utility_takes_precedence_at_ply_cap(self):
        state = covered_position()
        config = {"lambda": 10, "q": [100, 100, 100]}
        engine = ExperimentAgent({"variant_id": "custom", "evaluation": config}).engine
        winning_row = ((1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1))
        won = replace(state, board=state.board[:4] + winning_row + state.board[8:], ply=100)
        self.assertEqual(engine.fixed_depth(replace(won, player=1), 0).value, 99900)
        self.assertEqual(engine.fixed_depth(replace(won, player=2), 0).value, -99900)
        self.assertEqual(engine.fixed_depth(replace(state, ply=100), 0).value, 0)
