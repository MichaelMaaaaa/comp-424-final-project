"""Checks against the frozen uploaded search and the executable experiment adapter."""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from agents import student_agent as uploaded
from agents.alpha_beta import alpha_beta
from agents.evaluation import evaluate, evaluate_e0
from agents.experiment_agent import ExperimentAgent
from agents.game_state import StateTimeout, apply_move, initial_state, legal_moves, to_arrays
from agents.iterative_deepening import search
from agents.move_ordering import SearchFunctions
from agents.search_engine import SearchConfig, SearchCounters, SearchEngine, component_configs, search_key
from agents.transposition_table import EXACT, TranspositionTable, encode_state
from book_experiments.analysis import ANALYSIS_FILES
from helpers import Move
from runners import load_agent_class
from search_experiments.build import build_agent
from search_experiments.measure import measure_online_step


class OriginalSearchTests(unittest.TestCase):
    def test_s0_matches_uploaded_order_perspectives_value_and_nodes(self):
        state = initial_state()
        for preferred in (None, legal_moves(state)[-1], Move(None, state.obstacle, 0)):
            def recording(trace):
                def score(child, player):
                    trace.append((child.board, player))
                    return evaluate(child, player)
                return score

            original_calls, adapted_calls = [], []
            with patch.object(uploaded, 'evaluate', recording(original_calls)):
                want = uploaded.alpha_beta(state, 2, preferred_move=preferred,
                                           tt=uploaded.TranspositionTable())
            engine = SearchEngine(evaluator=recording(adapted_calls), diagnostics=True)
            got = engine.fixed_depth(state, 2, preferred=preferred)
            self.assertEqual((got.value, got.move, got.statistics['nodes']), want)
            self.assertTrue(adapted_calls == original_calls, 'ordering or evaluation perspective differs')

    def test_uploaded_complete_iterations_and_fresh_online_table(self):
        state = replace(initial_state(), ply=98)
        original = uploaded.search(state, float('inf'))
        legacy = search(state, float('inf'))
        engine = SearchEngine(diagnostics=True)
        table = engine.tt
        with patch.object(engine, 'statistics', wraps=engine.statistics) as snapshot:
            got = engine.search(state, float('inf'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertIsNot(table, engine.tt)
        self.assertEqual(legacy.iterations, original.iterations)
        self.assertEqual([(r['depth'], r['move'], r['value']) for r in got.iterations],
                         original.iterations)
        self.assertEqual((got.move, got.statistics['nodes']), (original.move, original.nodes))
        table = engine.tt
        again = engine.search(state, 0)
        self.assertIsNot(table, engine.tt)
        self.assertFalse(engine.tt.entries)
        self.assertEqual((again.move, again.completed_depth, again.optimal_moves),
                         (legal_moves(state)[0], 0, ()))

    def test_direct_searches_ignore_supplied_table_when_tt_is_disabled(self):
        state = initial_state()
        config = SearchConfig(tt=False)
        def run(kind, table, counters):
            calls = SearchFunctions(counters=counters)
            if kind == "fixed":
                return alpha_beta(state, 1, tt=table, config=config, calls=calls)
            result = search(state, float("inf"), max_depth=1, tt=table, config=config, calls=calls)
            return result.value, result.move, result.nodes

        for kind in ("fixed", "iterative"):
            reference = run(kind, None, SearchCounters())
            for warm in (False, True):
                with self.subTest(kind=kind, warm=warm):
                    table = TranspositionTable()
                    if warm:
                        alpha_beta(state, 1, tt=table)
                    before = dict(table.entries), dict(table.best_moves)
                    counters = SearchCounters()
                    self.assertEqual(run(kind, table, counters), reference)
                    self.assertEqual((counters.tt_probes, counters.tt_stores), (0, 0))
                    self.assertEqual((table.entries, table.best_moves), before)

    def test_h_hint_is_replaced_by_first_completed_move(self):
        from agents import iterative_deepening
        state = initial_state()
        preferred = legal_moves(state)[0]
        observed = []
        def record(*args, **kwargs):
            observed.append(kwargs['preferred_move'])
            return original(*args, **kwargs)
        original = iterative_deepening.alpha_beta
        with patch.object(iterative_deepening, 'alpha_beta', side_effect=record):
            got = SearchEngine().search(state, float('inf'), max_depth=2, preferred=preferred)
        first = got.iterations[0]['move']
        self.assertNotEqual(first, preferred)
        self.assertEqual(observed, [preferred, first])

    def test_timeout_during_order_or_recursion_or_before_caching(self):
        state = initial_state()
        move_count = len(legal_moves(state))
        for phase, player, ordinal in [('ordering', 1, 3), ('recursion', 2, 2),
                                       ('before cache', 2, move_count)]:
            def run(frozen):
                now = [0]
                calls = []
                def expire(child, perspective):
                    calls.append(perspective)
                    if perspective == player and calls.count(player) == ordinal:
                        now[0] = 2
                    return evaluate(child, perspective)
                with patch('time.perf_counter', side_effect=lambda: now[0]):
                    if frozen:
                        tt = uploaded.TranspositionTable()
                        stats = uploaded.SearchStats()
                        with patch.object(uploaded, 'evaluate', expire), self.assertRaises(uploaded.StateTimeout):
                            uploaded.alpha_beta(state, 1, deadline=1, tt=tt, stats=stats)
                        return calls, stats.nodes, tt.get(encode_state(state), 1)
                    engine = SearchEngine(evaluator=expire, diagnostics=True)
                    result = engine.fixed_depth(state, 1, deadline=1)
                    self.assertEqual(result.status, 'incomplete', phase)
                    return calls, result.statistics['nodes'], engine.tt.get(encode_state(state), 1)
            self.assertEqual(run(False), run(True), phase)

    def test_late_root_result_is_rejected_by_iteration_controller(self):
        from agents import iterative_deepening
        state = initial_state()
        def run(frozen):
            module = uploaded if frozen else iterative_deepening
            original = module.alpha_beta
            now = [0]
            def stop(node, depth, *args, **kwargs):
                result = original(node, depth, *args, **kwargs)
                if node is state and depth == 2:
                    now[0] = 2
                return result
            with patch.object(module, 'alpha_beta', side_effect=stop), \
                    patch('time.perf_counter', side_effect=lambda: now[0]):
                return uploaded.search(state, 1) if frozen else SearchEngine().search(state, 1, 2)
        want, got = run(True), run(False)
        self.assertEqual((got.completed_depth, got.move, got.value), (1, want.move, want.value))

    def test_exact_ties_override_root_tt_and_survive_interrupted_next_depth(self):
        state = initial_state()
        moves = legal_moves(state)
        first_root_board = None
        def stop_at_grandchild(child, player):
            nonlocal first_root_board
            if child.ply == state.ply + 2:
                player_one = tuple(tuple(owner == 1 for owner in stack) for stack in child.board)
                if first_root_board is None:
                    first_root_board = player_one
                elif player_one != first_root_board:
                    raise StateTimeout('second root action interrupted')
            return 0
        engine = SearchEngine(evaluator=stop_at_grandchild)
        key, _ = search_key(state, engine.config)
        engine.tt.store(key, 1, 500, EXACT, moves[-1])
        result = engine.fixed_depth(state, 1, clear_tt=False, collect_ties=True)
        self.assertEqual((result.value, set(result.optimal_moves)), (0, set(moves)))
        result = engine.search(state, float('inf'), 2, collect_ties=True, tie_rng=random.Random(63))
        self.assertEqual((result.completed_depth, set(result.optimal_moves)), (1, set(moves)))
        self.assertEqual(result.move, random.Random(63).choice(result.optimal_moves))

    def test_each_independent_candidate_against_exact_minimax(self):
        state = initial_state((1, 1))
        for evaluator in (evaluate, evaluate_e0):
            exact = SearchEngine(component_configs()['C0'], evaluator).fixed_depth(state, 2)
            for change in (dict(preferred_first=True), dict(reuse_results=True),
                           dict(key_mode='compact'), dict(symmetry=True)):
                got = SearchEngine(SearchConfig(**change), evaluator).fixed_depth(state, 2)
                self.assertEqual(got.value, exact.value)
                self.assertIn(got.move, exact.optimal_moves)

    def test_n_never_constructs_or_imports_unused_provider(self):
        state = initial_state()
        variant = {'variant_id': 'S0_E1_N', 'book': {'mode': 'N', 'provider': 'agents.absent'},
                   'max_depth': 1, 'telemetry': False}
        def forbidden(*args):
            self.fail('N mode invoked a book factory')
        agent = ExperimentAgent(variant, book_factory=forbidden)
        move = agent.step(*to_arrays(state), 1, 2)
        self.assertFalse(agent.last_info['book_queried'])
        self.assertEqual(measure_online_step(state, variant)['move'], move)
        with tempfile.TemporaryDirectory() as tmp:
            path, meta = build_agent(variant, Path(tmp) / 'student_agent.py')
            self.assertNotIn('agents.absent', meta['source_modules'])
            for module in ('agents.alpha_beta', 'agents.move_ordering', 'agents.iterative_deepening'):
                self.assertIn(module, meta['source_files'])
                self.assertIn(module.replace('.', '/') + '.py', ANALYSIS_FILES)
            built = load_agent_class(str(path))()
            self.assertEqual(built.step(*to_arrays(state), 1, 2), move)
            self.assertEqual(built.last_info['completed_depth'], 1)

    def test_generated_weights_change_search_move(self):
        state = replace(initial_state(), win_length=4)
        with tempfile.TemporaryDirectory() as tmp:
            for weights, size in [([1, 50, 100], 2), ([100, 2, 100], 0)]:
                variant = {'variant_id': 'S0_custom_N', 'max_depth': 1, 'telemetry': False,
                           'evaluation': {'evaluation_id': 'custom', 'lambda': 4, 'q': weights}}
                path, _ = build_agent(variant, Path(tmp) / 'student_agent.py')
                built = load_agent_class(str(path))()
                got = built.step(*to_arrays(state), 1, 2)
                self.assertEqual((got.size, got.dest, built.last_info['value']), (size, (1, 2), 3400))

    def test_a_cutoff_equal_to_best_is_not_a_true_root_tie(self):
        state = initial_state()
        best, false_tie = legal_moves(state)[:2]
        # A small P1 placement at (0,1) draws against every reply. At (0,2)
        # only the first reply draws. Every other P1 placement loses outright.
        def score(node, player):
            board = node.board
            draw = board[1][0] == 1 or (board[2][0] == 1 and board[1][0] == 2)
            return 0 if draw else (50 if player == 2 else -50)
        config = component_configs()['C1']
        child = apply_move(state, false_tie)
        bound, _, _ = alpha_beta(child, 1, -float('inf'), 0, config=config,
                                 calls=SearchFunctions(score))
        self.assertEqual(bound, 0)
        result = SearchEngine(config, score).fixed_depth(state, 2, collect_ties=True)
        self.assertEqual((result.value, result.optimal_moves), (0, (best,)))

    def test_s2_checks_deadline_when_reusing_a_leaf(self):
        now = [0]
        def expire_at_entry(node, depth, *args, **kwargs):
            now[0] = 2
            return alpha_beta(node, depth, *args, **kwargs)
        engine = SearchEngine(SearchConfig(tt=False, reuse_results=True), diagnostics=True)
        with patch('agents.alpha_beta.alpha_beta', side_effect=expire_at_entry), \
                patch('time.perf_counter', side_effect=lambda: now[0]):
            result = engine.fixed_depth(initial_state(), 1, deadline=1)
        self.assertEqual((result.status, result.statistics['nodes'], result.statistics['depth_returns']),
                         ('incomplete', 2, 0))

    def test_s4_discards_a_numeric_entry_with_missing_action(self):
        state = apply_move(initial_state(), Move(None, (0, 1), 0))
        engine = SearchEngine(SearchConfig(symmetry=True))
        key, transform = search_key(state, engine.config)
        self.assertNotEqual(transform, 0)
        engine.tt.store(key, 1, 123456, EXACT, None)
        result = engine.fixed_depth(state, 1, clear_tt=False)
        expected = uploaded.alpha_beta(state, 1)
        self.assertEqual((result.value, result.move), expected[:2])

    def test_expired_whole_step_uses_framework_fallback(self):
        state = initial_state()
        agent = ExperimentAgent({'variant_id': 'zero', 'step_budget': 0})
        with patch.object(agent.engine, 'search', side_effect=AssertionError('already expired')):
            move = agent.step(*to_arrays(state), 1, 2)
        self.assertEqual((move, agent.last_info['completed_depth'], agent.last_info['status']),
                         (legal_moves(state)[0], 0, 'fallback'))
