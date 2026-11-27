from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from agents.alpha_beta import alpha_beta
from agents.evaluation import evaluate, evaluate_e0
from agents.experiment_agent import ExperimentAgent
from agents.game_state import (GameState, StateTimeout, apply_move, initial_state,
                               is_legal_move, legal_moves, terminal, to_arrays)
from agents.move_ordering import SearchFunctions
from agents.search_engine import (SearchConfig, SearchCounters, SearchEngine, component_configs,
                                  obstacle_stabilizers, search_key)
from agents.symmetry import transform_move, transform_state
from agents.transposition_table import EXACT, LOWER, UPPER
from baselines.ab_baseline_agent import evaluate as baseline_evaluate
from constants import MEDIUM
from helpers import Move
from runners import load_agent_class
from search_experiments.build import build_agent, build_opponent
from search_experiments.common import (REPO_ROOT, append_jsonl, atomic_json, digest, file_hash, read_jsonl,
                                        normalize_variant, repair_jsonl_tail, restore_state, state_record)
from search_experiments import exp02_search
from search_experiments.measure import compare_performance, validation_passes
from search_experiments.runner import (make_manifest, paired_difference, run_pairs,
                                       sample_positions, summarize_games, validate_manifest)


def random_state(plies=12, obstacle=(1, 1), seed=424):
    rng = random.Random(seed)
    state = initial_state(obstacle, seed % 3)
    for _ in range(plies):
        if terminal(state)[0]:
            break
        state = apply_move(state, rng.choice(legal_moves(state)))
    return state


class SearchCorrectnessTests(unittest.TestCase):
    def test_terminal_diagnostics_count_setup_and_recursive_calls(self):
        state = initial_state()
        for workload in ("fixed_depth", "search"):
            with self.subTest(workload=workload):
                checked = []

                def observed_terminal(node, deadline=None):
                    checked.append((node, deadline))
                    return terminal(node, deadline)

                def run(engine, deadline):
                    if workload == "fixed_depth":
                        return engine.fixed_depth(state, 1, deadline)
                    return engine.search(state, deadline, 1)

                deadline = float("inf")
                with patch("agents.move_ordering.terminal", side_effect=observed_terminal), \
                        patch("agents.iterative_deepening.terminal", side_effect=observed_terminal, create=True):
                    detailed = run(SearchEngine(diagnostics=True), deadline)
                self.assertEqual(detailed.status, "completed")
                self.assertEqual(detailed.statistics["terminal_calls"], len(checked))
                self.assertEqual(checked[0], (state, None if workload == "search" else deadline))
                plain = run(SearchEngine(), deadline)
                self.assertEqual((plain.status, plain.move, plain.value, plain.completed_depth),
                                 (detailed.status, detailed.move, detailed.value, detailed.completed_depth))
                self.assertEqual(plain.statistics, {"enabled": False})

    def test_c0_c3_and_original_agree_at_fixed_depth(self):
        for state in [initial_state((0, 0), 0), random_state(), random_state(18, (0, 1), 7)]:
            reference = SearchEngine(component_configs()["C0"]).fixed_depth(state, 2)
            self.assertTrue(reference.optimal_moves)
            for config in component_configs().values():
                result = SearchEngine(config).fixed_depth(state, 2)
                self.assertEqual(result.value, reference.value)
                self.assertIn(result.move, reference.optimal_moves)
            value, move, _ = alpha_beta(state, 2)
            self.assertEqual(value, reference.value)
            self.assertIn(move, reference.optimal_moves)

    def test_s1_s4_are_exact_for_e0_and_e1(self):
        state = random_state(10, (2, 2), 31)
        for evaluator in (evaluate_e0, evaluate):
            reference = SearchEngine(component_configs()["C0"], evaluator).fixed_depth(state, 2)
            config = SearchConfig()
            for change in [{"preferred_first": True}, {"reuse_results": True},
                           {"key_mode": "compact"}, {"symmetry": True}]:
                config = replace(config, **change)
                result = SearchEngine(config, evaluator).fixed_depth(state, 2)
                self.assertEqual(result.value, reference.value)
                self.assertIn(result.move, reference.optimal_moves)

    def test_depth_three_exact_reference_at_ply_99(self):
        state = replace(random_state(13), ply=99, player=2)
        reference = SearchEngine(component_configs()["C0"]).fixed_depth(state, 3)
        for config in component_configs().values():
            result = SearchEngine(config).fixed_depth(state, 3)
            self.assertEqual(result.value, reference.value)
            self.assertIn(result.move, reference.optimal_moves)

    def test_complete_root_ties_use_full_child_windows(self):
        state = random_state(6)
        reference = SearchEngine(component_configs()["C0"]).fixed_depth(state, 2)
        result = SearchEngine().fixed_depth(state, 2, collect_ties=True)
        self.assertEqual(result.value, reference.value)
        self.assertEqual(set(result.optimal_moves), set(reference.optimal_moves))

    def test_unsorted_search_disables_preferred_promotion(self):
        state = initial_state()
        moves = legal_moves(state)
        constant = lambda state, player: 0
        result = SearchEngine(component_configs()["C1"], constant).fixed_depth(state, 1, preferred=moves[-1])
        self.assertEqual(result.move, moves[0])
        ordered = SearchEngine(component_configs()["C2"], constant).fixed_depth(state, 1, preferred=moves[-1])
        self.assertEqual(ordered.move, moves[-1])

    def test_s1_preferred_cutoff_avoids_other_successors(self):
        state = initial_state()
        preferred = legal_moves(state)[-1]
        counts = []
        for early in (False, True):
            counters = SearchCounters()
            alpha_beta(state, 1, -math.inf, -1_000_000, preferred_move=preferred,
                       config=SearchConfig(tt=False, preferred_first=early),
                       calls=SearchFunctions(counters=counters))
            counts.append(counters.successors)
        self.assertGreater(counts[0], 1)
        self.assertEqual(counts[1], 1)

    def test_s2_reuses_terminal_and_leaf_results(self):
        state = random_state(5)
        results = [SearchEngine(SearchConfig(preferred_first=True, reuse_results=flag),
                                diagnostics=True).fixed_depth(state, 2) for flag in (False, True)]
        self.assertEqual(results[0].value, results[1].value)
        self.assertLess(results[1].statistics["terminal_calls"], results[0].statistics["terminal_calls"])
        self.assertLess(results[1].statistics["evaluation_calls"], results[0].statistics["evaluation_calls"])

    def test_different_depth_entry_is_only_a_hint(self):
        state = initial_state()
        engine = SearchEngine()
        key, _ = search_key(state, engine.config)
        engine.tt.store(key, 1, 99999, EXACT, legal_moves(state)[0])
        result = engine.fixed_depth(state, 2, clear_tt=False)
        self.assertEqual(result.value, 0)

    def test_noncutting_bounds_do_not_tighten_window(self):
        state = initial_state()
        for flag, value in [(LOWER, -100), (UPPER, 100)]:
            engine = SearchEngine()
            key, _ = search_key(state, engine.config)
            engine.tt.store(key, 2, value, flag, legal_moves(state)[0])
            windows = []
            def spy(node, depth, alpha, beta, *args, **kwargs):
                if node.ply == state.ply + 1 and not windows:
                    windows.append((alpha, beta))
                return alpha_beta(node, depth, alpha, beta, *args, **kwargs)
            with patch("agents.alpha_beta.alpha_beta", side_effect=spy):
                result, _, _ = alpha_beta(state, 2, -1000, 1000, tt=engine.tt)
            self.assertEqual(windows[0], (-1000, 1000))
            self.assertEqual(result, 0)

    def test_matching_bounds_cut_without_expansion(self):
        state = initial_state()
        for depth, flag, value, alpha, beta in [(1, LOWER, 3000, -1000, 1000), (2, UPPER, 100, 500, 1000)]:
            engine = SearchEngine(diagnostics=True)
            key, _ = search_key(state, engine.config)
            engine.tt.store(key, depth, value, flag, legal_moves(state)[0])
            score, _, _ = alpha_beta(state, depth, alpha, beta, tt=engine.tt,
                                     calls=SearchFunctions(counters=engine.counters))
            self.assertEqual(score, value)
            self.assertEqual(engine.counters.successors, 0)
            self.assertEqual(engine.counters.tt_bound_returns, 1)

    def test_interrupted_root_is_not_cached_or_committed(self):
        state = initial_state()
        def interrupted(state, player):
            raise StateTimeout("injected interruption")
        engine = SearchEngine(evaluator=interrupted)
        result = engine.fixed_depth(state, 2)
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.value)
        key, _ = search_key(state, engine.config)
        self.assertIsNone(engine.tt.get(key, 2))

    def test_timeout_retains_last_complete_iteration(self):
        state = initial_state()
        engine = SearchEngine()
        def stop_at_two(node, depth, *args, **kwargs):
            if depth >= 2:
                raise StateTimeout("second iteration interrupted")
            return alpha_beta(node, depth, *args, **kwargs)
        with patch("agents.iterative_deepening.alpha_beta", side_effect=stop_at_two):
            result = engine.search(state, time.perf_counter() + 10, max_depth=3)
        self.assertEqual(result.completed_depth, 1)
        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.move, result.iterations[0]["move"])
        self.assertEqual(result.value, result.iterations[0]["value"])

    def test_expired_deadline_has_legal_fallback(self):
        state = initial_state()
        result = SearchEngine().search(state, time.perf_counter() - 1)
        self.assertEqual(result.completed_depth, 0)
        self.assertIsNone(result.value)
        self.assertTrue(is_legal_move(state, result.move))

    def test_remaining_depth_cap_is_solved_and_stops(self):
        state = replace(initial_state(), ply=98)
        result = SearchEngine().search(state, time.perf_counter() + 2)
        self.assertEqual(result.completed_depth, 2)
        self.assertTrue(result.solved_remaining)

    def test_zero_capacity_tt_and_separate_counters(self):
        state = initial_state()
        timed = SearchEngine(SearchConfig(tt_capacity=0)).fixed_depth(state, 2)
        detailed_engine = SearchEngine(SearchConfig(tt_capacity=0), diagnostics=True)
        detailed = detailed_engine.fixed_depth(state, 2)
        self.assertEqual(timed.value, detailed.value)
        self.assertEqual(timed.statistics, {"enabled": False})
        self.assertEqual(detailed.statistics["tt_entries"], 0)
        self.assertGreater(detailed.statistics["tt_capacity_rejections"], 0)
        self.assertGreater(detailed.statistics["approximate_tt_bytes"], 0)


class KeyAndHookTests(unittest.TestCase):
    def test_stabilizer_sizes_and_obstacle_invariance(self):
        for row in range(4):
            for col in range(4):
                values = obstacle_stabilizers((row, col))
                corner = row in (0, 3) and col in (0, 3)
                interior = row in (1, 2) and col in (1, 2)
                self.assertEqual(len(values), 2 if corner or interior else 1)

    def test_compact_key_retains_hidden_reserves_player_ply_and_rules(self):
        state = random_state()
        config = SearchConfig(key_mode="compact")
        board = list(state.board)
        location = next(i for i, stack in enumerate(board) if stack[0] != 3)
        stack = list(board[location])
        stack[0] = 1 if stack[0] != 1 else 2
        board[location] = tuple(stack)
        reserves = list(state.reserves)
        counts = list(reserves[1])
        counts[0] = (counts[0] + 1) % 4
        reserves[1] = tuple(counts)
        variants = [state, replace(state, board=tuple(board)), replace(state, reserves=tuple(reserves)),
                    replace(state, player=3-state.player), replace(state, ply=state.ply + 1),
                    replace(state, allow_moves=False), replace(state, win_length=3)]
        keys = [search_key(s, config)[0] for s in variants]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertTrue(all(len(key) == 15 for key in keys))

    def test_symmetry_tt_maps_action_back_to_query_coordinates(self):
        state = random_state(8, (0, 0), 73)
        config = SearchConfig(key_mode="compact", symmetry=True)
        engine = SearchEngine(config)
        original = engine.fixed_depth(state, 2)
        reference = SearchEngine(component_configs()["C0"]).fixed_depth(state, 2)
        for transform, _ in obstacle_stabilizers(state.obstacle):
            other = transform_state(state, transform)
            self.assertEqual(search_key(state, config)[0], search_key(other, config)[0])
            actual = engine.fixed_depth(other, 2, clear_tt=False)
            self.assertEqual(actual.value, original.value)
            self.assertTrue(is_legal_move(other, actual.move))
            self.assertIn(actual.move, [transform_move(m, transform) for m in reference.optimal_moves])

    def test_evaluator_injection(self):
        state = initial_state()
        result = SearchEngine(evaluator=lambda state, player: 17).fixed_depth(state, 0)
        self.assertEqual(result.value, 17)

    def test_book_direct_returns_and_call_counter_always_advances(self):
        config = {"variant_id": "book_hook_test", "book": {"mode": "D"}, "telemetry": False}
        def factory(book_config, variant):
            return lambda state, moves, deadline: {"matched": True, "version": "test",
                                                   "direct_move": moves[-1]}
        agent = ExperimentAgent(config, book_factory=factory)
        state = initial_state()
        board, reserves = to_arrays(state)
        with patch.object(agent.engine, "search", side_effect=AssertionError("book should return first")):
            for ply in (0, 2):
                returned = agent.step(board, reserves, 1, 2)
                self.assertTrue(is_legal_move(state, returned))
                self.assertEqual(agent.last_info["ply"], ply)
                self.assertIsNone(agent.last_info["completed_depth"])
        self.assertEqual(agent.call_count, 2)

    def test_book_hint_is_passed_to_search_and_illegal_direct_ignored(self):
        state = initial_state()
        hint = legal_moves(state)[-1]
        config = {"variant_id": "hint", "book": {"mode": "H"}, "max_depth": 1, "telemetry": False}
        def factory(book_config, variant):
            return lambda state, moves, deadline: {"matched": True, "preferred_move": hint,
                                                   "direct_move": Move(None, state.obstacle, 2)}
        agent = ExperimentAgent(config, evaluation_factory=lambda _: lambda state, player: 0, book_factory=factory)
        returned = agent.step(*to_arrays(state), 1, 2)
        self.assertEqual(returned, hint)
        self.assertEqual(agent.last_info["book_action"], "hint")
        with self.assertRaises(ValueError):
            ExperimentAgent({"variant_id": "missing", "book": {"mode": "D"}})

    def test_book_hint_lookup_can_finish_after_search_deadline(self):
        state = initial_state()
        moves = legal_moves(state)
        now = 0

        def lookup(state, moves, deadline):
            nonlocal now
            now = deadline
            return {"matched": True, "preferred_move": moves[-1]}

        config = {"variant_id": "late_hint", "book": {"mode": "H"}, "max_depth": 1, "telemetry": False}
        agent = ExperimentAgent(config, book_factory=lambda config, variant: lookup)
        with patch("time.perf_counter", side_effect=lambda: now):
            self.assertEqual(agent.step(*to_arrays(state), 1, 2), moves[0])
        self.assertEqual(agent.last_info["completed_depth"], 0)
        self.assertTrue(agent.last_info["book_queried"])
        self.assertTrue(agent.last_info["book_matched"])
        self.assertEqual(agent.last_info["book_action"], "none")

    def test_original_submission_not_changed_by_builder(self):
        original_hash = file_hash(REPO_ROOT / "agents/student_agent.py")
        with tempfile.TemporaryDirectory() as tmp:
            path, metadata = build_agent({"variant_id": "built", "max_depth": 1, "telemetry": False},
                                         Path(tmp)/"student_agent.py")
            source = path.read_text()
            self.assertNotIn("from agents.game_state", source)
            self.assertNotIn("from agents.search_engine", source)
            cls = load_agent_class(str(path))
            agent = cls()
            self.assertTrue(is_legal_move(initial_state(), agent.step(*to_arrays(initial_state()), 1, 2)))
            self.assertLess(metadata["bytes"], 2_000_000)
        self.assertEqual(original_hash, file_hash(REPO_ROOT / "agents/student_agent.py"))

    def test_seeded_random_and_greedy_artifacts_embed_real_rng(self):
        for kind in ("random", "greedy"):
            with tempfile.TemporaryDirectory() as tmp:
                path, meta = build_opponent(kind, Path(tmp)/"student_agent.py", seed=1234)
                cls = load_agent_class(str(path))
                first, second = cls(), cls()
                self.assertEqual(int(first.rng.integers(1000000)), int(second.rng.integers(1000000)))
                self.assertEqual(meta["opponent_rng_seed"], 1234)

    def test_ab_opponent_build_rejects_invalid_budgets_before_replacing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, metadata = build_opponent("ab", Path(tmp)/"student_agent.py", budget=0)
            meta_path = path.with_suffix(".meta.json")
            saved = path.read_bytes(), meta_path.read_bytes()
            state = initial_state()
            self.assertEqual(metadata["budget"], 0)
            self.assertEqual(load_agent_class(str(path))().step(*to_arrays(state), 1, 2), legal_moves(state)[0])
            for budget in (False, True, "0.01", None, -1, math.nan, math.inf, -math.inf):
                with self.subTest(budget=budget):
                    with self.assertRaisesRegex(ValueError, "AB.*budget"):
                        build_opponent("ab", path, budget=budget)
                    self.assertEqual((path.read_bytes(), meta_path.read_bytes()), saved)
            _, valid = build_opponent("ab", path, budget=3.0)
            self.assertEqual(valid["budget"], 3.0)

    def test_builder_inlines_configured_evaluation_and_book_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"
            shutil.copytree(REPO_ROOT/"agents", root/"agents", ignore=shutil.ignore_patterns("__pycache__"))
            (root/"agents/test_book.py").write_text('''def make_book_hook(book_config, variant):
    def lookup(state, moves, deadline):
        return {"matched": True, "version": "fixture", "direct_move": moves[-1]}
    return lookup
''')
            variant = {"variant_id": "integration",
                       "evaluation": {"evaluation_id": "custom", "lambda": 5, "q": [1, 50, 100]},
                       "book": {"mode": "D", "provider": "agents.test_book"}}
            path, _ = build_agent(variant, Path(tmp)/"student_agent.py", root=root)
            agent = load_agent_class(str(path))()
            state = apply_move(initial_state(), Move(None, (1, 2), MEDIUM))
            self.assertEqual(agent.engine.fixed_depth(state, 0).value, -3250)
            self.assertEqual(agent.step(*to_arrays(state), 2, 1), legal_moves(state)[-1])

    def test_local_baseline_discards_entire_obstacle_line(self):
        board, reserves = to_arrays(initial_state((0, 0)))
        board[0, 1, 0] = 1
        self.assertEqual(baseline_evaluate(board, 1), 2)


class HarnessTests(unittest.TestCase):
    def test_variant_hash_normalization_is_idempotent(self):
        variant = {"variant_id": "S0_E1_N"}
        once = normalize_variant(variant)
        self.assertEqual(once, normalize_variant(once))

    def test_manifest_is_complete_disjoint_stratified_and_reproducible(self):
        first = make_manifest(123)
        second = make_manifest(123)
        self.assertEqual(first, second)
        rows = validate_manifest(first)
        self.assertEqual([sum(row["split"] == split for row in rows) for split in ("D0", "D1", "T")], [12, 12, 24])

    def test_development_runner_rejects_T_before_playing(self):
        row = next(r for r in make_manifest()["rows"] if r["split"] == "T")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pairs({"variant_id": "no_leak"}, [row], tmp)

    def test_state_serialization_round_trip_and_validation(self):
        state = random_state()
        self.assertEqual(restore_state(json.loads(json.dumps(state_record(state)))), state)
        bad = json.loads(json.dumps(state_record(state)))
        bad["board"][1][0] = 1.7
        with self.assertRaises(ValueError):
            restore_state(bad)
        bad = json.loads(json.dumps(state_record(state)))
        bad["obstacle"] = [0, 0]
        with self.assertRaises(ValueError):
            restore_state(bad)

    def test_no_ranking_for_incomplete_games_and_forfeits_count_as_loss(self):
        row = {"game_id": "a", "config_id": "x", "candidate_seat": 1, "status": "adjudicated",
               "outcome": "L", "score": 0.0, "technical_failure": True}
        summary = summarize_games([row], ["a", "b"])
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["L"], 1)
        self.assertFalse(summary["eligible"])
        self.assertEqual(summary["missing_game_ids"], ["b"])
        legacy_win = dict(row, outcome="W", score=1.0, technical_failure=False)
        legacy = summarize_games([legacy_win], ["a"])
        self.assertEqual((legacy["W"], legacy["score"], legacy["complete"]), (1, 1.0, True))
        self.assertFalse(legacy["eligible"])

    def test_pair_difference_clusters_both_seats(self):
        old = {"complete": True, "paired_scores": {"a": 0.5, "b": 0.5}}
        new = {"complete": True, "paired_scores": {"a": 1.0, "b": 0.0}}
        result = paired_difference(new, old)
        self.assertEqual(result["mean"], 0)
        self.assertEqual((result["improved"], result["worse"]), (1, 1))
        self.assertIn("configuration", result["sampling_unit"])

    def test_sampling_records_missing_groups_without_padding(self):
        state = initial_state((0, 0), 0)
        game = {"status": "adjudicated", "split": "D0", "game_id": "one", "config_id": "one",
                "variant_id": "S0", "obstacle": (0, 0), "removed_size": 0,
                "turns": [{"state": state_record(state), "ply": 0, "move": {}, "actor_role": "candidate"}]}
        result = sample_positions([game], seed=42)
        self.assertEqual(result["count"], 1)
        self.assertEqual(sum(row["missing"] for row in result["coverage"]), 53)

    def test_sampler_reassigns_a_shared_state_to_fill_the_stratum(self):
        row = next(row for row in validate_manifest(make_manifest(0)) if row["config_id"] == "r3c3_L")
        self.assertEqual(row["split"], "D0")
        prefix = (Move(None, (0, 0), 0), Move(None, (1, 0), 0), Move(None, (0, 1), 0),
                  Move(None, (1, 2), 0), Move(None, (0, 2), 1), Move(None, (2, 2), 1),
                  Move(None, (2, 0), 0), Move(None, (1, 3), 0), Move(None, (2, 3), 1),
                  Move(None, (3, 1), 1))
        state = initial_state(row["obstacle"], row["removed_size"])
        before_moves = []
        for move in prefix:
            before_moves.append((state, move))
            state = apply_move(state, move)
        next_move = Move(None, (3, 2), 2)
        alternative = apply_move(state, next_move)
        early_win, later_win = Move(None, (0, 3), 2), Move(None, (1, 1), 2)
        self.assertEqual(terminal(apply_move(state, early_win)), (True, 1, "line"))
        self.assertEqual(terminal(apply_move(alternative, later_win)), (True, 2, "line"))

        def game(seat, ending):
            turns = [{"state": state_record(before), "ply": before.ply,
                      "move": {"src": move.src, "dest": move.dest, "size": move.size},
                      "actor_role": "candidate" if before.player == seat else "opponent"}
                     for before, move in [*before_moves, *ending]]
            return dict(row, status="adjudicated", variant_id="S0", game_id=f"S0__r3c3_L__P{seat}",
                        candidate_seat=seat, turns=turns)

        long_game = game(1, [(state, next_move), (alternative, later_win)])
        short_game = game(2, [(state, early_win)])
        both_short = game(1, [(state, early_win)])
        records = [long_game, short_game]
        for label, games, selected in (("feasible", records, 2),
                                      ("insufficient", [both_short, short_game, short_game], 1)):
            with self.subTest(case=label):
                sampled = sample_positions(games, seed=5)
                self.assertEqual(sampled, sample_positions(games, seed=5))
                middle = next(c for c in sampled["coverage"] if c["stage"] == "middle"
                              and c["removed_size"] == 2 and c["obstacle_class"] == "corner")
                self.assertEqual((middle["available_games"], middle["selected"], middle["missing"]),
                                 (2, selected, 2 - selected))
                actual = {p["source_game"]: p["state"]["ply"] for p in sampled["positions"]
                          if p["stage"] == "middle"}
                if label == "feasible":
                    self.assertEqual(actual, {"S0__r3c3_L__P1": 11, "S0__r3c3_L__P2": 10})
                else:
                    self.assertEqual(list(actual.values()), [10])
                self.assertEqual(len({digest(p["state"]) for p in sampled["positions"]}), sampled["count"])
        self.assertEqual(sample_positions(records, seed=5, per_group=0)["count"], 0)

    def test_interrupted_tail_repair_preserves_flushed_records_for_retry(self):
        records = [{"complete": 1}, {"complete": 2}]
        original_open = Path.open

        def interrupted_open(target, mode="r", *args, **kwargs):
            stream = original_open(target, mode, *args, **kwargs)
            if target == path and any(flag in mode for flag in "wa+"):
                stream.close()
                raise KeyboardInterrupt
            return stream

        with tempfile.TemporaryDirectory() as tmp:
            for label in ("partial_row", "missing_newline"):
                with self.subTest(tail=label):
                    path = Path(tmp)/f"{label}.jsonl"
                    for row in records:
                        append_jsonl(path, row)
                    complete = path.read_bytes()
                    if label == "partial_row":
                        with path.open("ab") as stream:
                            stream.write(b'{"incomplete":')
                    else:
                        with path.open("r+b") as stream:
                            stream.truncate(len(complete) - 1)
                    before = path.read_bytes()
                    with patch.object(Path, "open", interrupted_open), self.assertRaises(KeyboardInterrupt):
                        repair_jsonl_tail(path)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(read_jsonl(path), records)
                    repair_jsonl_tail(path)
                    self.assertEqual(path.read_bytes(), complete)

    def test_truncated_log_tail_can_resume_without_losing_complete_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"record.jsonl"
            path.write_text('{"valid":1}\n{"partial":')
            self.assertEqual(read_jsonl(path), [{"valid": 1}])
            repair_jsonl_tail(path)
            self.assertEqual(path.read_text(), '{"valid":1}\n')
            path.write_text('{"valid":2}')
            repair_jsonl_tail(path)
            self.assertEqual(read_jsonl(path), [{"valid": 2}])

    def test_config_and_key_validation(self):
        for options in [{"tt_capacity": -1}, {"tt_capacity": True}, {"key_mode": "hash"},
                        {"ordering": False, "preferred_first": True}, {"tt": False, "symmetry": True}]:
            with self.assertRaises(ValueError):
                SearchConfig(**options)

    def test_unretained_change_does_not_leak_into_later_candidates(self):
        position = {"position_id": "fixture", "state": state_record(initial_state()), "split": "D0"}
        payload = {"positions_hash": digest([position]), "execution_tier": "formal", "reference_complete": True}
        results = [False, True, False, False]
        def performance(records, parent, candidate, baseline=None):
            return {"retain_candidate": results.pop(0), "budget_valid": True}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(exp02_search, "load_positions", return_value=([position],payload)), \
                 patch.object(exp02_search, "exact_references", return_value={}), \
                 patch.object(exp02_search, "measure", return_value=([],{})), \
                 patch.object(exp02_search, "validation_passes", return_value=True), \
                 patch.object(exp02_search, "compare_performance", side_effect=performance):
                selected = exp02_search.run({"positions":"fixture", "run_confirmation":False},tmp,smoke=True)
        self.assertEqual(selected["S_diag"], "S2")
        second = selected["decisions"][1]
        self.assertEqual(second["parent_variant"], "S0")
        self.assertFalse(second["search"]["preferred_first"])
        self.assertTrue(second["search"]["reuse_results"])
        fourth = selected["decisions"][3]
        self.assertEqual(fourth["parent_variant"], "S2")
        self.assertEqual(fourth["search"]["key_mode"], "tuple")
        self.assertIsNone(selected["S_star"])

    def test_performance_order_must_be_consistent_three_times(self):
        records = []
        for repetition, depths in enumerate([(2,3),(3,2),(2,3)]):
            for name, depth in zip(("parent","candidate"),depths):
                records.append({"variant_id":name,"position_id":"x","workload":"online",
                    "repetition":repetition,"near_cap":False,"solved_remaining":False,
                    "completed_depth":depth,"budget_ok":True,"hard_limit_ok":True})
        result = compare_performance(records,"parent","candidate")
        self.assertFalse(result["consistent"])
        self.assertFalse(result["retain_candidate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
