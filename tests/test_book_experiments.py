import base64
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from agents.book_policy import (book_build_hook, book_check_key, book_pack_key, book_pack_move,
                                book_unpack_key, book_unpack_move)
from agents.experiment_agent import ExperimentAgent
from agents.game_state import (GameState, StateTimeout, apply_move, initial_state, is_legal_move,
                               legal_moves, terminal, to_arrays)
from agents.symmetry import (TRANSFORMS, canonicalize, decode_book_state, encode_book_state,
                             inverse_transform_move, transform_move, transform_state)
from book_experiments import exp05_book
from book_experiments.analysis import screen_record
from book_experiments.books import (book_table, book_variant, encode_layout, expand_entries,
                                    load_current_book, lookup, make_book, move_record, subset_book,
                                    write_provider)
from book_experiments.candidates import by_ply, common_count, initial_keys, select_by_stage, stage_quotas
from book_experiments.collect import collection_plan
from book_experiments.common import rank
from book_experiments.coverage import game_metrics, replay_coverage
from helpers import Move
from runners import load_agent_class
from search_experiments.build import build_agent
from search_experiments.common import REPO_ROOT, normalize_variant, state_record

LAYOUTS = [dict(symmetry=s, key=k, compression=c) for s, k, c in
           (("d4", "compact", "zlib"), ("expanded", "compact", "zlib"),
            ("d4", "tuple", "zlib"), ("d4", "compact", "none"))]


def random_states(count, seed=0):
    rng, states = random.Random(seed), []
    while len(states) < count:
        state = initial_state((rng.randrange(4), rng.randrange(4)), rng.randrange(3))
        for _ in range(rng.randrange(1, 25)):
            if terminal(state)[0]:
                break
            state = apply_move(state, rng.choice(legal_moves(state)))
        if not terminal(state)[0]:
            states.append(state)
    return states


def canonical_entries(states, pick=0):
    """One canonical entry per distinct class; the move is legal in canonical coordinates."""
    entries = {}
    for state in states:
        key, _ = canonicalize(state)
        moves = legal_moves(decode_book_state(key))
        entries.setdefault(key, moves[pick % len(moves)])
    return sorted(entries.items())


def board_state(pieces, player=1, ply=8, obstacle=(3, 3)):
    board = np.zeros((4, 4, 3), dtype=np.int8)
    board[obstacle] = 3
    for row, col, size, owner in pieces:
        board[row, col, size] = owner
    reserves = np.array([[0, 0, 0], [3, 3, 3], [3, 3, 3]], dtype=np.int8)
    return GameState.from_observation(board, reserves, player, ply)


def analysis_record(state, move, moves_by_depth=None, analysis_id="a"):
    """Analysis record in canonical coordinates for a move given in the state's coordinates."""
    key, transform = canonicalize(state)
    rows = moves_by_depth or [move]
    return {"analysis_id": analysis_id, "key": list(key), "value": 0,
            "move": move_record(transform_move(move, transform)), "completed_depth": len(rows),
            "iterations": [{"depth": d, "move": move_record(transform_move(m, transform)), "value": 0,
                            "elapsed": 0.0} for d, m in enumerate(rows, start=1)]}


class SymmetryAndPackingTests(unittest.TestCase):
    def test_transforms_round_trip_for_cells_and_moves(self):
        for state in random_states(6, seed=1):
            for move in legal_moves(state):
                for transform in TRANSFORMS:
                    self.assertEqual(inverse_transform_move(transform_move(move, transform), transform), move)

    def test_canonical_key_is_invariant_and_stored_move_maps_back_legally(self):
        for state in random_states(8, seed=2):
            key, _ = canonicalize(state)
            canonical_move = legal_moves(decode_book_state(key))[-1]
            for transform in TRANSFORMS:
                actual = transform_state(state, transform)
                actual_key, chosen = canonicalize(actual)
                self.assertEqual(actual_key, key)
                self.assertTrue(is_legal_move(actual, inverse_transform_move(canonical_move, chosen)))

    def test_key_and_move_packing_round_trip(self):
        for state in random_states(10, seed=3):
            key = canonicalize(state)[0]
            self.assertEqual(book_unpack_key(book_pack_key(key)), key)
            for move in legal_moves(state):
                self.assertEqual(book_unpack_move(book_pack_move(move)), move)
        with self.assertRaises(ValueError):
            book_check_key(key[:-1] + (128,))

    def test_current_book_round_trips_to_the_same_bytes(self):
        entries, version = load_current_book()
        self.assertEqual(len(entries), 486)
        data, layout = encode_layout(entries, "d4", "compact", "none")
        ours = base64.b64decode("".join(data))
        source = (REPO_ROOT / "agents/student_agent.py").read_text()
        import ast
        blob = next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "BOOK_DATA")
        theirs = zlib.decompress(base64.b64decode(blob))
        chunks = lambda raw: {raw[i:i + 17] for i in range(0, len(raw), 17)}
        self.assertEqual(chunks(ours), chunks(theirs))


class RepresentationTests(unittest.TestCase):
    def test_all_layouts_return_identical_legal_moves_and_misses(self):
        symmetric = initial_state((0, 0), 0)  # fixed by the diagonal reflection
        states = [symmetric] + random_states(12, seed=4)
        entries = canonical_entries(states, pick=3)
        tables = [book_table(make_book(entries, "unit", layout)) for layout in LAYOUTS]
        for state in states:
            for transform in TRANSFORMS:
                actual = transform_state(state, transform)
                answers = {lookup(table, actual) for table in tables}
                self.assertEqual(len(answers), 1)
                self.assertTrue(is_legal_move(actual, answers.pop()))
        miss = apply_move(states[1], legal_moves(states[1])[0])
        if canonicalize(miss)[0] not in dict(entries):
            self.assertEqual({lookup(table, miss) for table in tables}, {None})

    def test_expansion_merges_self_symmetric_orientations(self):
        state = initial_state((0, 0), 0)
        entries = canonical_entries([state])
        expanded = expand_entries(entries)
        self.assertLess(len(expanded), 8)
        table = book_table(make_book(entries, "unit"))
        for raw, move in expanded:
            self.assertEqual(lookup(table, decode_book_state(raw)), move)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.state = random_states(1, seed=5)[0]
        self.book = make_book(canonical_entries([self.state]), "unit")
        self.data, self.layout = tuple(self.book["data"]), self.book["layout"]
        self.expected = lookup(book_table(self.book), self.state)

    def hook(self, mode, **extra):
        return book_build_hook(self.data, self.layout, dict(mode=mode, **extra), {})

    def test_modes(self):
        self.assertIsNone(self.hook("N"))
        moves = tuple(legal_moves(self.state))
        direct = self.hook("D")(self.state, moves, float("inf"))
        hint = self.hook("H")(self.state, moves, float("inf"))
        self.assertEqual((direct["direct_move"], direct["preferred_move"]), (self.expected, None))
        self.assertEqual((hint["direct_move"], hint["preferred_move"]), (None, self.expected))
        self.assertTrue(direct["matched"] and hint["matched"])
        self.assertGreaterEqual(direct["metadata"]["query_seconds"], 0)
        miss = apply_move(self.state, moves[0])
        self.assertFalse(self.hook("D")(miss, tuple(legal_moves(miss)), float("inf"))["matched"])

    def test_illegal_entry_is_not_matched(self):
        key, transform = canonicalize(self.state)
        bad = Move(None, transform_move(Move(None, self.state.obstacle, 0), transform).dest, 0)
        book = make_book([(key, bad)], "unit")
        hook = book_build_hook(tuple(book["data"]), book["layout"], {"mode": "D"}, {})
        decision = hook(self.state, tuple(legal_moves(self.state)), float("inf"))
        self.assertFalse(decision["matched"])
        self.assertTrue(decision["metadata"]["invalid_entry"])

    def test_version_hash_and_deadline_checks(self):
        with self.assertRaises(ValueError):
            self.hook("D", version="other")
        with self.assertRaises(ValueError):
            book_build_hook(("A" + self.data[0][1:],) + self.data[1:], self.layout, {"mode": "D"}, {})
        with self.assertRaises(ValueError):
            self.hook("X")
        with self.assertRaises(StateTimeout):
            self.hook("D")(self.state, tuple(legal_moves(self.state)), 0.0)

    def test_agent_uses_direct_and_hint(self):
        state = initial_state((0, 1), 1)
        book = make_book(canonical_entries([state], pick=5), "unit")
        expected = lookup(book_table(book), state)
        factory = lambda config, variant: book_build_hook(tuple(book["data"]), book["layout"], config, variant)
        agent = ExperimentAgent({"variant_id": "d", "book": {"mode": "D"}, "telemetry": False}, book_factory=factory)
        with patch.object(agent.engine, "search", side_effect=AssertionError("D must not search on a hit")):
            self.assertEqual(agent.step(*to_arrays(state), 1, 2), expected)
        self.assertEqual(agent.last_info["book_action"], "direct")
        agent = ExperimentAgent({"variant_id": "h", "book": {"mode": "H"}, "max_depth": 1, "telemetry": False},
                                book_factory=factory)
        real, seen = agent.engine.search, []
        def spy(state, deadline, max_depth=None, preferred=None, **kwargs):
            seen.append(preferred)
            return real(state, deadline, max_depth, preferred, **kwargs)
        with patch.object(agent.engine, "search", side_effect=spy):
            agent.step(*to_arrays(state), 1, 2)
        self.assertEqual(seen, [expected])
        self.assertEqual(agent.last_info["book_action"], "hint")


class CandidateTests(unittest.TestCase):
    @staticmethod
    def pool(early=20, middle=20, late=20):
        rows = [{"key": list(k), "ply": 0, "frequency": 4, "stage": "early", "initial": True} for k in initial_keys()]
        for stage, count, ply in (("early", early, 5), ("middle", middle, 15), ("late", late, 40)):
            rows += [{"key": [stage == "late", ply, i], "ply": ply, "frequency": i % 7, "stage": stage, "initial": False}
                     for i in range(count)]
        return rows

    def test_nine_initial_classes_and_b0_order(self):
        self.assertEqual(len(initial_keys()), 9)
        rows = [{"key": [2], "ply": 1, "frequency": 1}, {"key": [1], "ply": 1, "frequency": 3},
                {"key": [0], "ply": 0, "frequency": 1}]
        self.assertEqual([r["key"] for r in by_ply(rows)], [[0], [1], [2]])

    def test_stage_quotas_keep_initial_states_in_the_early_quota(self):
        selected = select_by_stage(self.pool(), 30)
        self.assertEqual(stage_quotas(30), {"early": 10, "middle": 10, "late": 10})
        stages = [r["stage"] for r in selected]
        self.assertEqual((stages.count("early"), stages.count("middle"), stages.count("late")), (10, 10, 10))
        self.assertEqual(sum(r["initial"] for r in selected), 9)

    def test_shortage_is_filled_in_rotation_and_small_nc_keeps_initial_states(self):
        selected = select_by_stage(self.pool(late=2), 30)
        stages = [r["stage"] for r in selected]
        self.assertEqual(len(selected), 30)
        self.assertEqual(stages.count("late"), 2)
        self.assertEqual(stages.count("early") - 10, stages.count("middle") - 10)
        small = select_by_stage(self.pool(), 12)
        self.assertEqual(sum(r["initial"] for r in small), 9)
        self.assertEqual([r["stage"] for r in small[9:]], ["early", "middle", "late"])
        with self.assertRaises(ValueError):
            common_count(8, [self.pool()])

    def test_collection_plan_balances_seats(self):
        rows = [{"config_id": f"c{i}", "split": "D0", "obstacle": [0, i % 4], "removed_size": i % 3,
                 "collection_seed": i} for i in range(12)]
        plan = collection_plan(rows, {"b0_selfplay_per_config": 4, "b1_selfplay_per_config": 2})
        greedy = [g["our_seat"] for g in plan if g["kind"] == "greedy"]
        ab = [g["our_seat"] for g in plan if g["kind"] == "ab"]
        self.assertEqual((greedy.count(1), ab.count(1)), (6, 6))
        self.assertTrue(all(g != a for g, a in zip(greedy, ab)))
        self.assertEqual(sum("B1" in g["pools"] for g in plan if g["kind"] == "self"), 24)
        self.assertEqual(sum("B0" in g["pools"] for g in plan), 48)


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.win = board_state([(0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 1, 1), (2, 0, 0, 2), (2, 1, 0, 2)])
        three = [(1, 0, 0, 2), (1, 1, 0, 2), (1, 2, 1, 2)]
        self.uncover = board_state(three + [(1, 3, 0, 2), (1, 3, 2, 1)])
        self.threat = board_state(three + [(0, 0, 0, 1)])

    def test_rules(self):
        win = Move(None, (0, 3), 2)
        self.assertEqual(screen_record(analysis_record(self.win, win), "a")["reason"], "immediate_win")
        loss = Move((1, 3), (3, 0), 2)
        self.assertEqual(screen_record(analysis_record(self.uncover, loss, [loss] * 4), "a")["reason"],
                         "immediate_loss")
        careless, block = Move(None, (3, 0), 0), Move(None, (1, 3), 2)
        self.assertEqual(screen_record(analysis_record(self.threat, careless, [careless] * 4), "a")["reason"],
                         "opponent_immediate_win")
        self.assertEqual(screen_record(analysis_record(self.threat, block, [block] * 2), "a")["reason"],
                         "insufficient_depth")
        self.assertEqual(screen_record(analysis_record(self.threat, block, [careless, block, block]), "a")["reason"],
                         "stable_and_safe")
        self.assertEqual(screen_record(analysis_record(self.threat, block, [block, careless, block]), "a")["reason"],
                         "unstable_move")
        self.assertEqual(screen_record(analysis_record(self.threat, block, [block] * 3), "other")["reason"],
                         "version_mismatch")
        illegal = Move(None, (3, 3), 0)
        self.assertEqual(screen_record(analysis_record(self.threat, illegal, [illegal] * 3), "a")["reason"],
                         "illegal_move")


class ExportCoverageAndRankTests(unittest.TestCase):
    def test_export_puts_accepted_initial_states_first(self):
        keys = [canonicalize(s)[0] for s in random_states(6, seed=6)] + initial_keys()[:2]
        rows = [{"key": list(k), "ply": k[-1], "stage": "early", "initial": k in initial_keys(),
                 "frequency": 1 + i % 3} for i, k in enumerate(keys)]
        records = {tuple(r["key"]): {"completed_depth": 3 + i % 2, "value": 0,
                                     "move": move_record(legal_moves(decode_book_state(r["key"]))[0])}
                   for i, r in enumerate(rows)}
        screening = {k: {"accepted": True, "reason": "stable_and_safe"} for k in records}
        selected = {scheme: rows for scheme in exp05_book.SCHEMES}
        with tempfile.TemporaryDirectory() as tmp, patch.object(exp05_book, "save_book",
                                                                 return_value={"provider": "x", "path": "x"}):
            books, export = exp05_book.export_books(selected, records, screening, "unit", 5, Path(tmp))
        order = [tuple(e["key"]) for e in books["B0"]["entries"]]
        self.assertEqual(order[:2], sorted(initial_keys()[:2]))
        rest = sorted((r for r in rows if not r["initial"]),
                      key=lambda r: (-r["frequency"], -records[tuple(r["key"])]["completed_depth"], r["key"]))
        self.assertEqual(order[2:], [tuple(r["key"]) for r in rest[:3]])
        self.assertEqual(export["B0"]["exported"], 5)
        self.assertEqual(len(subset_book(books["B0"], 3, "unit_small")["entries"]), 3)

    def test_replay_and_game_metrics(self):
        hit, other = random_states(2, seed=7)
        table = book_table(make_book(canonical_entries([hit]), "unit"))
        turn = lambda state, search: {"actor_role": "candidate", "move": {"src": None, "dest": [0, 0], "size": 0},
                                      "ply": state.ply, "player": state.player, "state": state_record(state),
                                      "search": search}
        game = {"status": "adjudicated", "game_id": "g", "obstacle": [0, 0], "candidate_seat": 1,
                "turns": [turn(hit, {"book_queried": True, "book_matched": True, "book_action": "direct",
                                     "status": "book", "completed_depth": None}),
                          turn(other, {"book_queried": True, "book_matched": False, "book_action": "none",
                                       "status": "completed", "completed_depth": 3}),
                          turn(other, {"book_queried": True, "book_matched": True, "book_action": "hint",
                                       "status": "completed", "completed_depth": 0}),
                          dict(turn(other, None), actor_role="opponent")],
                "referee": {"moves": [{"player": 1, "agent_time": 0.5}, {"player": 2, "agent_time": 1.0}],
                            "time": {"1": {"init": 0.25}}, "max_rss_mb": {"1": 40.0}}}
        coverage = replay_coverage(table, [game])
        self.assertEqual((coverage["queryable"], coverage["matched"]), (3, 1))
        self.assertEqual(sum(v["queryable"] for v in coverage["by_obstacle_class"].values()), 3)
        metrics = game_metrics([game])
        self.assertEqual((metrics["queries"], metrics["matches"], metrics["direct_returns"], metrics["hint_only"]),
                         (3, 2, 1, 1))
        self.assertEqual((metrics["depth_not_applicable"], metrics["depth_zero"], metrics["mean_completed_depth"]),
                         (1, 1, 1.5))
        self.assertEqual((metrics["step_time_max"], metrics["init_time_max"], metrics["max_rss_mb"]), (0.5, 0.25, 40.0))

    def test_rank_uses_score_then_the_simpler_option(self):
        summaries = {"N": {"eligible": True, "score": 0.5}, "D": {"eligible": True, "score": 0.5},
                     "H": {"eligible": True, "score": 0.6}, "X": {"eligible": False, "score": 1.0}}
        self.assertEqual(rank(summaries, ["N", "D", "H", "X"]), ["H", "N", "D"])


class BuildTests(unittest.TestCase):
    def test_generated_provider_is_inlined_and_returns_the_book_move(self):
        state = initial_state((3, 2), 1)  # an edge cell, the same class as (0, 1)
        book = make_book(canonical_entries([initial_state((0, 1), 1)], pick=7), "unit_build")
        expected = lookup(book_table(book), state)
        self.assertIsNotNone(expected)
        online = normalize_variant({"variant_id": "S0_E1_N", "max_depth": 1, "telemetry": False})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            shutil.copytree(REPO_ROOT / "agents", root / "agents",
                            ignore=shutil.ignore_patterns("__pycache__", "books"))
            write_provider(book, root)
            path, meta = build_agent(book_variant(online, book, "D"), Path(tmp) / "student_agent.py", root=root)
            source = path.read_text()
            self.assertNotIn("from agents.books", source)
            self.assertNotIn("from agents.book_policy", source)
            self.assertNotIn("open(", source)
            self.assertLess(meta["bytes"], 2_000_000)
            agent = load_agent_class(str(path))()
            self.assertEqual(agent.step(*to_arrays(state), 1, 2), expected)
            self.assertEqual(agent.last_info["status"], "book")


if __name__ == "__main__":
    unittest.main()
