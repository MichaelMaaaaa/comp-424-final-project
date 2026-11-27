"""Step 4–6 handoff regressions with real runner persistence and one-ply games."""

from contextlib import contextmanager, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from agents.game_state import apply_move, initial_state, legal_moves
from agents.symmetry import decode_book_state
from book_experiments import common as book_common, exp05_book, exp06_book_mode, self_test
from book_experiments.books import book_entries, load_book, make_book, move_record
from helpers import Move
from search_experiments import common, runner
from search_experiments.build import build_agent
from tests import test_experiment_contracts as fixtures
from world import GameRecord


@contextmanager
def fail_first_replace(path, error):
    """Interrupt exactly the first attempted replacement of the published status."""
    original = os.replace
    armed = True

    def replace(temporary, target):
        nonlocal armed
        if target == path and armed:
            armed = False
            raise error("initial publication failed")
        return original(temporary, target)

    with patch.object(os, "replace", replace):
        yield


class RecordedEndingTests(unittest.TestCase):
    def test_both_seats_require_the_recorded_board_ending_in_selection_and_final_reports(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["config_id"] == "r0c2_S")
        variant = common.normalize_variant({"variant_id": "recorded_line", "step_budget": 0.012, "max_depth": 2})
        opponent = {"kind": "ab", "budget": 0.006}

        def transcript(seed, first, second, **rules):
            record = GameRecord(seed, row["obstacle"], row["removed_size"], rules,
                                {"1": {"agent": first.name}, "2": {"agent": second.name}})
            state = initial_state(row["obstacle"], row["removed_size"])
            turns, times = [], {1: [], 2: []}
            # Alternating placements fill row 1 for P1 while P2 has only three pieces in row 2.
            for ply, (lane, column, size) in enumerate(((1, 0, 0), (2, 0, 0), (1, 1, 1), (2, 1, 1),
                                                        (1, 2, 2), (2, 2, 2), (1, 3, 0))):
                move = Move(None, (lane, column), size)
                record.add(ply, state.player, move, 0.004, 0.005, [])
                times[state.player].append(0.004)
                turns.append(dict(state=common.state_record(state), ply=ply, player=state.player,
                                  move=dict(record.moves[-1])))
                state = apply_move(state, move)
            record.finish(1, "line")
            return SimpleNamespace(obstacle=row["obstacle"], removed=row["removed_size"], turns=turns,
                                   step_times=times, play=lambda: record)

        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            games = directory / "games.jsonl"
            with patch.object(runner, "SnapshotWorld", transcript), patch.object(
                    runner, "SubprocessRunner", lambda path, **kw: SimpleNamespace(name=Path(path).parent.name)):
                runner.run_pairs(variant, [row], directory, opponent=opponent, execution_tier="smoke")
            original = games.read_bytes()
            for strict in (True, False):
                records, evidence = book_common.validate_match_evidence(
                    games, variant, [row], opponent, "smoke", require_eligible=strict)
                self.assertEqual([(r["candidate_seat"], r["outcome"]) for r in records], [(1, "W"), (2, "L")])
                self.assertEqual(evidence["summary"]["score"], 0.5)

            for edit, seat in (("missing last move", 1), ("wrong reason", 1), ("wrong winner", 2)):
                games.write_bytes(original)
                rows = common.read_jsonl(games)
                record = next(r for r in rows if r["candidate_seat"] == seat)
                referee = record["referee"]
                # Keep derived counts, timing, seat score and the saved summary consistent.
                if edit == "missing last move":
                    referee["moves"].pop()
                    referee["plies"] -= 1
                    record["turns"].pop()
                elif edit == "wrong reason":
                    referee["reason"] = "ply_cap"
                else:
                    referee["winner"] = 2
                    record.update(outcome="W", score=1.0)
                games.write_text("".join(common.canonical_json(r) + "\n" for r in rows))
                with patch.object(runner, "SubprocessRunner", side_effect=AssertionError("replayed a saved pair")):
                    runner.run_pairs(variant, [row], directory, opponent=opponent, execution_tier="smoke")
                for strict in (True, False):
                    with self.subTest(edit=edit, strict=strict), self.assertRaisesRegex(ValueError, "ending"):
                        book_common.validate_match_evidence(games, variant, [row], opponent, "smoke",
                                                            require_eligible=strict)


class HandoffFixture(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.label = f"handoff_{uuid4().hex[:8]}"
        self.addCleanup(self.remove_providers)
        # These branch fixtures deliberately stop after a few moves and can prescribe a score.
        # RecordedEndingTests separately exercises the production ending check on a complete transcript.
        self.opening_endings = {(0, "ply_cap")}

        def check_opening_ending(state, winner, reason):
            self.assertIn(state.ply, (1, 2, 4))
            self.assertIn((winner, reason), self.opening_endings)

        self.enterContext(patch.object(book_common, "_check_board_ending", check_opening_ending))
        self.manifest = runner.make_manifest()
        self.rows = runner.validate_manifest(self.manifest)
        self.reference = self.root / "d1_reference" / "games.jsonl"
        self.config = {"manifest": self.root / "manifest.json", "online_config": self.root / "online_config.json",
                       "d1_reference_games": self.reference, "positions": self.root / "positions.json",
                       "selected_book": self.root / "p5/selected_book.json",
                       "smoke": {"max_configs": 1, "book_label": self.label}}
        common.atomic_json(self.config["manifest"], self.manifest)

    def remove_providers(self):
        for path in (common.REPO_ROOT / "agents/books").glob(f"{self.label}_*.py"):
            path.unlink()

    def prepare(self, smoke, plies=1):
        self.tier = "smoke" if smoke else "formal"
        self.opp = {"kind": "ab", "budget": 0.006 if smoke else 1.5}
        self.online = common.normalize_variant({"variant_id": "handoff_N",
            **({"step_budget": 0.012, "max_depth": 2} if smoke else {})})
        self.code_hash = common.code_manifest()["code_hash"]
        self.manifest_hash = self.manifest["manifest_hash"]
        self.wrapper = {"status": "smoke_only" if smoke else "selected", "execution_tier": self.tier,
                        "formal_selection": not smoke, "code_hash": self.code_hash,
                        "manifest_hash": self.manifest_hash, "variant": self.online}
        common.atomic_json(self.config["online_config"], self.wrapper)
        d0 = [row for row in self.rows if row["split"] == "D0"]
        d1 = [row for row in self.rows if row["split"] == "D1"]
        self.d0, self.d1 = (d0[:1], d1[:1]) if smoke else (d0, d1)
        with self.opening_games(plies):
            runner.run_pairs(self.online, self.d1, self.reference.parent, opponent=self.opp,
                             execution_tier=self.tier)
        self.positions = runner.sample_positions(common.read_jsonl(self.reference))["positions"]
        self.write_positions()

    def change_recommendation(self, path):
        book = load_book(path)
        entries = list(book_entries(book))
        key, old = entries[0]
        other = next(move for move in legal_moves(decode_book_state(key)) if move != old)
        entries[0] = (key, other)
        common.atomic_json(path, make_book(entries, book["label"]))

    def write_positions(self):
        common.atomic_json(self.config["positions"], {"positions": self.positions,
            "positions_hash": common.digest(self.positions), "manifest_hash": self.manifest_hash,
            "execution_tier": self.tier, "reference_complete": True, "reference_eligible": True,
            "reference_code_hash": self.code_hash})

    @contextmanager
    def opening_games(self, plies=1, max_strikes=None):
        with fixtures.ManifestAndSelectionTests()._opening_pairs(), redirect_stdout(StringIO()):
            referee = runner.SnapshotWorld
            original_play = referee.play

            def play(world):
                world.max_plies = plies
                if max_strikes is not None:
                    world.max_strikes = max_strikes
                return original_play(world)

            with patch.object(referee, "play", play):
                yield

    def write_games(self, path, records):
        path.write_text("".join(common.canonical_json(row) + "\n" for row in records))


class EvidenceTests(HandoffFixture):
    def test_formal_d1_only_admits_exact_runner_evidence(self):
        self.prepare(smoke=False)
        records, evidence = book_common.reference_games(self.reference, self.online, self.config)
        self.assertEqual([(row["config_id"], row["candidate_seat"]) for row in records],
                         [(row["config_id"], seat) for row in self.d1 for seat in (1, 2)])
        self.assertEqual(evidence["summary"]["adjudicated"], 24)
        self.assertEqual(evidence["variant"], self.online)
        for name in ("plan", "games", "summary"):
            self.assertEqual(evidence[f"{name}_hash"], common.file_hash(evidence[f"{name}_path"]))

    def test_rejects_changed_plan_roster_variant_opponent_or_raw_rows(self):
        self.prepare(smoke=False)
        plan_path = self.reference.with_name("plan.json")
        summary_path = self.reference.with_name("summary.json")
        files = [plan_path, self.reference, summary_path]
        saved = {path: path.read_bytes() for path in files}

        def change_plan(field, value, refresh_digest=True):
            plan = common.load_json(plan_path)
            plan["request"][field] = value
            if refresh_digest:
                plan["request_hash"] = common.digest(plan["request"])
            common.atomic_json(plan_path, plan)

        def change_records(edit):
            records = common.read_jsonl(self.reference)
            with self.reference.open("w") as output:
                for row in edit(records):
                    output.write(common.canonical_json(row) + "\n")

        edits = {
            "missing plan": lambda: plan_path.unlink(),
            "wrong digest": lambda: change_plan("strict", False, False),
            "wrong tier": lambda: change_plan("execution_tier", "smoke"),
            "wrong machine": lambda: change_plan("machine", {}),
            "wrong code": lambda: change_plan("code_hash", "old"),
            "non-strict": lambda: change_plan("strict", False),
            "partial roster": lambda: change_plan("configurations", self.d1[:-1]),
            "mixed D0": lambda: change_plan("configurations", self.d1[:-1] + self.d0[:1]),
            "mixed T": lambda: change_plan("configurations", self.d1[:-1] +
                                          [next(row for row in self.rows if row["split"] == "T")]),
            "wrong variant name": lambda: change_plan("variant", common.normalize_variant(
                dict(self.online, variant_id="imposter"))),
            "wrong budget": lambda: change_plan("variant", common.normalize_variant(
                dict(self.online, step_budget=0.02))),
            "wrong depth": lambda: change_plan("variant", common.normalize_variant(
                dict(self.online, max_depth=2))),
            "wrong opponent": lambda: change_plan("opponent", {"kind": "greedy", "budget": 1.5}),
            "duplicate": lambda: change_records(lambda rows: rows + rows[-1:]),
            "partial evidence": lambda: change_records(lambda rows: rows[:-1]),
            "wrong trajectory setup": lambda: change_records(lambda rows: [
                dict(rows[0], turns=[dict(rows[0]["turns"][0], state={})]), *rows[1:]]),
            "wrong seed": lambda: change_records(lambda rows: rows[:-1] +
                                                 [dict(rows[-1], opponent_rng_seed=1)]),
            "wrong row machine": lambda: change_records(lambda rows: rows[:-1] + [dict(rows[-1], machine={})]),
            "wrong candidate source": lambda: change_records(lambda rows: rows[:-1] +
                                                            [dict(rows[-1], candidate_source_hash="elsewhere")]),
            "wrong opponent artifact": lambda: change_records(lambda rows: rows[:-1] +
                [dict(rows[-1], opponent=dict(rows[-1]["opponent"], artifact_hash="elsewhere"))]),
            "technical": lambda: change_records(lambda rows: rows[:-1] +
                                                  [dict(rows[-1], technical_failure=True)]),
            "over budget": lambda: change_records(lambda rows: rows[:-1] +
                                                    [dict(rows[-1], candidate_max_step_s=2.)]),
            "forged summary": lambda: common.atomic_json(summary_path,
                dict(common.load_json(summary_path), score=1.)),
        }
        for case, edit in edits.items():
            with self.subTest(case=case):
                edit()
                try:
                    with self.assertRaises(ValueError):
                        book_common.reference_games(self.reference, self.online, self.config)
                finally:
                    for path, data in saved.items():
                        path.write_bytes(data)

    def test_formal_d1_rejects_erased_opponent_only_trajectory(self):
        self.prepare(smoke=False)
        records = common.read_jsonl(self.reference)
        row = next(row for row in records if row["candidate_seat"] == 2)
        self.assertEqual(row["referee"]["plies"], 1)
        self.assertIsNone(row["candidate_max_step_s"])
        row["referee"].update(plies=0, moves=[])
        row["turns"] = []
        self.write_games(self.reference, records)
        with self.assertRaises(ValueError):
            book_common.reference_games(self.reference, self.online, self.config)

    def test_d1_requires_contiguous_legal_moves_and_matching_snapshots(self):
        self.prepare(smoke=True, plies=2)
        records, _ = book_common.reference_games(self.reference, self.online, self.config, smoke=True)
        self.assertEqual([len(row["turns"]) for row in records], [2, 2])
        saved = self.reference.read_bytes()

        def gap(row):
            row["referee"]["moves"][1]["ply"] = 3
            row["turns"][1]["move"]["ply"] = 3
            row["turns"][1]["ply"] = row["turns"][1]["state"]["ply"] = 3

        def incompatible_move(row):
            before = common.restore_state(row["turns"][0]["state"])
            other = move_record(legal_moves(before)[1])
            row["referee"]["moves"][0].update(other)
            row["turns"][0]["move"].update(other)

        def pass_instead_of_move(row):
            passed = {"ply": 0, "player": 1, "pass": True}
            row["referee"]["moves"][0] = passed
            row["turns"][0]["move"] = passed

        for edit, seat in ((gap, 1), (incompatible_move, 1), (pass_instead_of_move, 2)):
            with self.subTest(edit=edit.__name__):
                self.reference.write_bytes(saved)
                records = common.read_jsonl(self.reference)
                edit(next(row for row in records if row["candidate_seat"] == seat))
                self.write_games(self.reference, records)
                with self.assertRaises(ValueError):
                    book_common.reference_games(self.reference, self.online, self.config, smoke=True)

    def test_requires_current_wrapper_and_preserves_frozen_smoke_budget(self):
        self.prepare(smoke=True)
        self.assertEqual(book_common.load_online(self.config, True), self.online)
        variants = (self.online, dict(self.wrapper, status="pending"), dict(self.wrapper, formal_selection=0),
                    dict(self.wrapper, code_hash="stale"), dict(self.wrapper, manifest_hash="different"),
                    dict(self.wrapper, variant=dict(self.online, config_hash="bad")))
        for variant in variants:
            with self.subTest(variant=variant):
                common.atomic_json(self.config["online_config"], variant)
                with self.assertRaises(ValueError):
                    book_common.load_online(self.config, True)
        common.atomic_json(self.config["online_config"], self.wrapper)
        with self.assertRaises(ValueError):
            book_common.load_online(self.config, False)
        for field, value in (("step_budget", 0.02), ("max_depth", 3)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                book_common.load_online(dict(self.config, smoke={"max_configs": 1, field: value}), True)
        for limit in (True, 0, -1, 1.5, None, 12, 100):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                book_common.manifest_rows(dict(self.config, smoke={"max_configs": limit}), "D1", True)
        with self.assertRaises(ValueError):
            book_common.reference_games(self.reference, self.online,
                                        dict(self.config, smoke={"max_configs": 2}), True)

    def test_smoke_self_test_supplies_matching_wrapper_and_reference(self):
        output = self.root / "smoke"

        class SmokeSetupComplete(Exception):
            pass

        def command(argv, log_path, **kwargs):
            if "search_experiments.exp00_prepare" in argv:
                common.atomic_json(output / "prepare" / "manifest.json", self.manifest)
            if "book_experiments.exp05_book" in argv:
                config = dict(self.config, online_config=argv[argv.index("--online-config") + 1],
                              manifest=argv[argv.index("--manifest") + 1])
                variant = book_common.load_online(config, True)
                games = argv[argv.index("--references") + 1]
                records, evidence = book_common.reference_games(games, variant, config, smoke=True)
                self.assertEqual((len(records), evidence["summary"]["execution_tier"]), (2, "smoke"))
                raise SmokeSetupComplete

        with patch.object(self_test, "run_command", side_effect=command), \
                patch.object(sys, "argv", ["self_test", "--output", str(output)]), self.opening_games():
            with self.assertRaises(SmokeSetupComplete):
                self_test.main()


class AnalysisSettingsTests(HandoffFixture):
    def test_invalid_effective_analysis_settings_are_rejected_before_binding(self):
        self.prepare(smoke=False)
        cases = [("budget", v) for v in (0, -1, True, "1", None, float("nan"), float("inf"), 10.01)]
        cases += [("max_depth", v) for v in (0, -1, True, 1.5)]
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key, value=value):
                output = self.root / str(index)
                with patch.object(exp05_book, "_run", return_value=None), self.assertRaises(ValueError):
                    exp05_book.run(dict(self.config, analysis={key: value}), output)
                self.assertFalse((output / "request.json").exists())
                self.assertFalse((output / "selected_book.json").exists())

        self.reference = self.root / "smoke_ref" / "games.jsonl"
        self.config["d1_reference_games"] = self.reference
        self.prepare(smoke=True)
        config = dict(self.config, analysis={"budget": 1}, smoke=dict(self.config["smoke"], analysis={"budget": 0}))
        with patch.object(exp05_book, "_run", return_value=None), self.assertRaises(ValueError):
            exp05_book.run(config, self.root / "invalid-smoke", smoke=True)

    def test_positive_smoke_budget_can_produce_screened_fallbacks(self):
        self.prepare(smoke=True)
        config = dict(self.config, smoke=dict(self.config["smoke"], analysis={"budget": 1e-100}))
        output = self.root / "short-budget"
        with patch.object(exp05_book, "compare_representations", return_value={}), patch.object(
                exp05_book, "run_collection", return_value=([], {"complete": True})):
            result = exp05_book.run(config, output, smoke=True)
        self.assertEqual((result["status"], result["K"], result["formal_selection"]), ("no_common_book", 0, False))
        records = common.read_jsonl(output / "analysis" / "records.jsonl")
        self.assertEqual({(r["status"], r["completed_depth"]) for r in records}, {("fallback", 0)})

    def test_valid_shallow_analysis_then_initial_publication_failure(self):
        self.prepare(smoke=False)
        config = dict(self.config, analysis={"budget": 10.0, "max_depth": 1})
        with patch.object(exp05_book, "compare_representations", return_value={}), patch.object(
                exp05_book, "run_collection", return_value=([], {"complete": True})):
            for error, status in ((KeyboardInterrupt, "pending"), (OSError, "blocked")):
                with self.subTest(error=error):
                    output = self.root / error.__name__
                    path = output / "selected_book.json"
                    result = exp05_book.run(config, output)
                    self.assertEqual((result["status"], result["K"], result["formal_selection"]),
                                     ("no_common_book", 0, True))
                    analyses = common.read_jsonl(output / "analysis" / "records.jsonl")
                    self.assertEqual({r["completed_depth"] for r in analyses}, {1})
                    with fail_first_replace(path, error), self.assertRaises(error):
                        exp05_book.run(config, output)
                    result = common.load_json(path)
                    self.assertEqual((result["status"], result["formal_selection"], result["K"]),
                                     (status, False, None))
                    self.assertEqual(common.load_json(output / "book_summary.json")["status"], status)


class StageTests(HandoffFixture):
    def setUp(self):
        super().setUp()
        self.prepare(smoke=True)
        self.p5, self.p6 = self.root / "p5", self.root / "p6"

        def step_timing(state, variant):
            return {"elapsed": 0.001, "completed_depth": 1, "book_queried": variant["book"]["mode"] == "D",
                    "book_matched": False, "book_metadata": {}, "budget_ok": True}

        patcher = patch.object(exp06_book_mode, "measure_online_step", side_effect=step_timing)
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextmanager
    def book_work(self, *, collection_complete=True, screening="accept"):
        def analysis(keys, online, identity, output):
            records = {}
            for key in keys:
                state = decode_book_state(key)
                move = move_record(legal_moves(state)[0])
                depth = {"reject": 0, "fallback_at_depth_one": 1, "shallow": 1}.get(screening, 3)
                entries = [{"depth": index, "move": move, "value": 1, "elapsed": 0.01}
                           for index in range(1, depth + 1)]
                if screening == "invalid":
                    entries.pop()
                if screening == "illegal_iteration":
                    entries[1]["move"] = dict(move, dest=list(state.obstacle))
                records[key] = {"key": list(key), "analysis_id": identity["analysis_id"], "move": move,
                                "iterations": entries, "completed_depth": depth, "elapsed": 0.01,
                                "value": 1, "termination": "deadline",
                                "status": "completed" if depth and screening != "fallback_at_depth_one" else "fallback"}
            return records

        with patch.object(exp05_book, "run_collection", return_value=([], {"complete": collection_complete})), \
                patch.object(exp05_book, "analyze_candidates", side_effect=analysis), \
                patch.object(exp05_book, "compare_representations", return_value={}), self.opening_games():
            yield

    def select_book(self):
        with self.book_work():
            return exp05_book.run(self.config, self.p5, smoke=True)

    def final_roles(self):
        return [self.p6 / name for name in ("final_variant.json", "book_comparison_variant.json")]

    def assert_unusable(self, result, status):
        self.assertEqual(result["status"], status)
        self.assertIsNone(result["variant"])
        self.assertFalse(result["formal_selection"])
        self.assertFalse(any(path.exists() for path in self.final_roles()))
        self.assertEqual(common.load_json(self.p6 / "final_config.json"), result)

    def test_p5_publishes_actual_descriptor_and_rejects_changed_accepted_inputs(self):
        result = self.select_book()
        self.assertEqual(result["status"], "selected")
        descriptor = result["selected_book"]
        book = load_book(result["book_file"])
        self.assertEqual(descriptor["variant_D"], result["variant_D"])
        for key in ("hash", "version", "entries"):
            self.assertEqual(descriptor[key], book["layout"][key])
            self.assertEqual(descriptor[key], result[key])
        self.assertEqual(descriptor["entries"], result["k_star"])
        self.assertEqual(result["online_config_hash"], common.file_hash(self.config["online_config"]))
        previous = (self.p5 / "selected_book.json").read_bytes()
        with self.book_work(), self.assertRaisesRegex(ValueError, "new output directory"):
            exp05_book.run(dict(self.config, export={"sizes": [9]}), self.p5, smoke=True)
        self.assertEqual((self.p5 / "selected_book.json").read_bytes(), previous)

    def test_p5_exposes_no_stale_selection_after_interruption_or_error(self):
        self.select_book()
        for error, expected in ((KeyboardInterrupt, "pending"), (RuntimeError, "blocked")):
            with self.subTest(error=error), self.book_work(), patch.object(exp05_book, "run_collection", side_effect=error):
                with self.assertRaises(error):
                    exp05_book.run(self.config, self.p5, smoke=True)
            result = common.load_json(self.config["selected_book"])
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["formal_selection"])
            self.assertIsNone(result["selected_book"])
            self.assertIsNone(result["variant_D"])

    def test_p5_incomplete_collection_or_required_match_cannot_select(self):
        with self.book_work(collection_complete=False):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["formal_selection"])
        self.assertIsNone(result["selected_book"])

        def incomplete(online, record, rows, output, opponent, tier):
            return {"complete": False, "eligible": True, "score": 1.0}, {}

        for case in ("source", "size"):
            self.p5 = self.root / f"p5-{case}"
            original = exp05_book.play_book

            def match(*args):
                output = args[3]
                if (case == "source" and output.name == "source_B2") or (
                    case == "size" and output.name.startswith("size_")
                ):
                    return incomplete(*args)
                return original(*args)

            def coverage(table, rows):
                return {"matched_turns": list(range(len(table[0])))}

            with self.book_work(), patch.object(exp05_book, "play_book", side_effect=match), \
                    patch.object(exp05_book, "replay_coverage", side_effect=coverage):
                result = exp05_book.run(dict(self.config, export={"sizes": [1]}), self.p5, smoke=True)
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["selected_book"])

    def test_p5_rechecks_eligible_source_and_size_logs_on_resume(self):
        config = dict(self.config, export={"sizes": [1]})

        def coverage(table, rows):
            return {"matched_turns": list(range(len(table[0])))}

        for relative in ("source_B2", "size_B0_k1"):
            with self.subTest(match=relative):
                self.p5 = self.root / relative
                with self.book_work(), patch.object(exp05_book, "replay_coverage", side_effect=coverage):
                    result = exp05_book.run(config, self.p5, smoke=True)
                    self.assertEqual((result["status"], result["source"], result["k_star"]),
                                     ("selected", "B0", 1))
                    log = self.p5 / "games" / relative / "games.jsonl"
                    rows = common.read_jsonl(log)
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        row.update(machine={}, score=1.0)
                    self.write_games(log, rows)
                    with self.assertRaisesRegex(ValueError, "invalid match evidence"):
                        exp05_book.run(config, self.p5, smoke=True)
                saved = common.load_json(self.p5 / "selected_book.json")
                self.assertEqual(saved["status"], "blocked")
                self.assertIsNone(saved["selected_book"])

    def test_fallback_with_a_completed_legal_iteration_blocks_the_empty_decision(self):
        with self.book_work(screening="fallback_at_depth_one"):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual((result["status"], result["reason"], result["K"]),
                         ("blocked", "invalid_analysis", None))
        self.assert_unusable(exp06_book_mode.run(self.config, self.p6, smoke=True), "blocked")
        with self.book_work(screening="shallow"):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual((result["status"], result["K"]), ("no_common_book", 0))

    def test_step6_checks_the_step5_tied_export_before_request_binding(self):
        selection = self.select_book()
        self.assertTrue(selection["source_tie"])
        path = Path(selection["second_source_book_file"])
        original = path.read_bytes()
        self.change_recommendation(path)
        with patch.object(exp06_book_mode, "bind_request", side_effect=AssertionError("bound invalid export")), \
                self.assertRaises(ValueError):
            exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertFalse((self.p6 / "request.json").exists())
        path.write_bytes(original)
        self.assertEqual(selection["second_source_book"]["book_file_sha256"], common.file_hash(path))
        del selection["second_source_book"]
        common.atomic_json(self.config["selected_book"], selection)
        with patch.object(exp06_book_mode, "bind_request", side_effect=AssertionError("bound invalid export")), \
                self.assertRaises(ValueError):
            exp06_book_mode.run(self.config, self.p6, smoke=True)

    def test_overhead_positions_must_match_the_actual_development_setup(self):
        self.select_book()
        dev = [row for row in self.rows if row["split"] == "D0"][-1]
        held_out = next(row for row in self.rows if row["split"] == "T")
        with self.opening_games(2):
            directory = self.root / "outside_smoke_match_roster"
            runner.run_pairs(self.online, [dev], directory, opponent=self.opp, execution_tier=self.tier)
        records = common.read_jsonl(directory / "games.jsonl")
        after_opening = [dict(record, turns=record["turns"][1:]) for record in records]
        fixed = runner.sample_positions(after_opening)["positions"][0]
        self.assertEqual(fixed["state"]["ply"], 1)
        self.assertNotIn(dev["config_id"], [row["config_id"] for row in self.d0])
        self.positions = [fixed]
        self.write_positions()
        with self.opening_games():
            self.assertEqual(exp06_book_mode.run(self.config, self.p6, smoke=True)["status"], "selected")
        t_state = common.state_record(initial_state(held_out["obstacle"], held_out["removed_size"]))
        invalid = [dict(fixed, config_id=held_out["config_id"], removed_size=held_out["removed_size"],
                        obstacle_class=held_out["obstacle_class"], state=t_state),
                   dict(fixed, split="D1"), dict(fixed, state=t_state),
                   dict(fixed, state=common.state_record(initial_state(dev["obstacle"], (dev["removed_size"] + 1) % 3)))]
        for index, wrong in enumerate(invalid):
            with self.subTest(index=index):
                self.positions = [wrong]
                self.write_positions()
                directory = self.root / f"bad_positions_{index}"
                with patch.object(exp06_book_mode, "bind_request", side_effect=AssertionError("bound T position")), \
                        self.assertRaises(ValueError):
                    exp06_book_mode.run(self.config, directory, smoke=True)
                self.assertFalse((directory / "request.json").exists())

    def test_illegal_completed_iteration_does_not_establish_empty_intersection(self):
        with self.book_work(screening="illegal_iteration"):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual((result["status"], result["reason"], result["K"]),
                         ("blocked", "invalid_analysis", None))
        self.assert_unusable(exp06_book_mode.run(self.config, self.p6, smoke=True), "blocked")

    def test_p5_invalid_analysis_and_nonpositive_sizes_cannot_yield_k0(self):
        with self.book_work(screening="invalid"):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["K"])
        for name, invalid in (("export", {"size_reference": 0}), ("export", {"sizes": [-1]}),
                              ("candidates", {"nc": 0})):
            with self.subTest(invalid=invalid), self.book_work(), self.assertRaises(ValueError):
                exp05_book.run(dict(self.config, **{name: invalid}), self.root / "invalid-size", smoke=True)

    def test_legitimate_empty_intersection_yields_n_only_with_no_book_control(self):
        with self.book_work(screening="reject"):
            result = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual((result["status"], result["K"]), ("no_common_book", 0))
        comparison_file = self.final_roles()[1]
        common.atomic_json(comparison_file, self.online)
        with patch.object(exp06_book_mode, "run_pairs", side_effect=AssertionError("extra campaign")):
            final = exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertEqual((final["status"], final["mode"], final["reason"]), ("N_only", "N", "no_common_book"))
        self.assertEqual(final["variant"], self.online)
        self.assertFalse(final["formal_selection"])
        self.assertIsNone(final["book_comparison_variant"])
        self.assertIsNone(final["book_comparison_file"])
        self.assertFalse(comparison_file.exists())
        self.assertEqual(final["development_evidence"], {})
        self.assertEqual(final["d1_reference"]["games_hash"], common.file_hash(self.reference))
        self.assertEqual(final["selected_book_hash"], common.file_hash(self.config["selected_book"]))

    def test_k0_publication_cleans_up_on_interruption_or_error(self):
        with self.book_work(screening="reject"):
            decision = exp05_book.run(self.config, self.p5, smoke=True)
        self.assertEqual((decision["status"], decision["K"]), ("no_common_book", 0))
        real_write = exp06_book_mode.atomic_json

        for error, status in ((KeyboardInterrupt, "pending"), (RuntimeError, "blocked")):
            with self.subTest(error=error):
                original = exp06_book_mode.run(self.config, self.p6, smoke=True)
                self.assertEqual(original["status"], "N_only")

                def fail(path, payload):
                    if path.name == "mode_summary.json" and payload["status"] == "N_only":
                        raise error("publication failed")
                    return real_write(path, payload)

                with self.assertRaises(error), patch.object(exp06_book_mode, "atomic_json", side_effect=fail):
                    exp06_book_mode.run(self.config, self.p6, smoke=True)
                self.assert_unusable(common.load_json(self.p6 / "final_config.json"), status)

    def test_n_has_a_buildable_full_development_book_control(self):
        self.select_book()
        with self.opening_games():
            final = exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertEqual((final["status"], final["mode"]), ("selected", "N"))
        self.assertFalse(final["formal_selection"])
        self.assertIsNone(final["book_file"])
        control = final["book_comparison_variant"]
        self.assertEqual(control["book"]["mode"], "D")
        self.assertEqual(common.load_json(self.final_roles()[1]), control)
        self.assertEqual(load_book(final["book_comparison_file"])["layout"]["hash"], final["book_comparison_hash"])
        evidence = final["development_evidence"][control["variant_id"]]
        self.assertEqual(evidence["variant"], control)
        self.assertEqual(evidence["summary"]["adjudicated"], 4)
        self.assertEqual(evidence["summary"]["eligible"], True)
        self.assertEqual([row["split"] for row in common.load_json(evidence["plan_path"])["request"]["configurations"]],
                         ["D0", "D1"])
        _, built = build_agent(control, self.root / "control.py")
        self.assertEqual(built["artifact_hash"], common.load_json(evidence["plan_path"])["request"]["candidate_artifact_hash"])

    def test_n_blocks_with_only_d1_book_control_and_waits_for_required_matches(self):
        self.select_book()
        original = exp06_book_mode.run_pairs

        for name, fault, expected in (("mode_incomplete", ("d1", "H"), "pending"),
                                      ("dev_incomplete", ("dev", "D"), "pending"),
                                      ("dev_ineligible", ("dev", "book"), "blocked")):
            with self.subTest(name=name):
                self.p6 = self.root / name
                for path in self.final_roles():
                    common.atomic_json(path, self.online)

                def match(variant, rows, output, **kwargs):
                    result = original(variant, rows, output, **kwargs)
                    if output.parent.name == fault[0] and (
                        output.name == fault[1] or fault[1] == "book" and output.name in ("D", "H")
                    ):
                        result["eligible"] = False
                        if name.endswith("incomplete"):
                            result["complete"] = False
                    return result

                with self.opening_games(), patch.object(exp06_book_mode, "run_pairs", side_effect=match):
                    result = exp06_book_mode.run(self.config, self.p6, smoke=True)
                self.assert_unusable(result, expected)

    def test_pending_decision_can_finish_but_changed_bound_request_cannot_overwrite(self):
        with self.book_work(collection_complete=False):
            exp05_book.run(self.config, self.p5, smoke=True)
        for path in self.final_roles():
            common.atomic_json(path, self.online)
        with patch.object(exp06_book_mode, "run_pairs", side_effect=AssertionError("premature games")):
            self.assert_unusable(exp06_book_mode.run(self.config, self.p6, smoke=True), "pending")
        self.select_book()
        with self.opening_games():
            final = exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertEqual(final["status"], "selected")
        selection = common.load_json(self.config["selected_book"])
        common.atomic_json(self.config["selected_book"], dict(selection, note="changed accepted decision"))
        with self.assertRaisesRegex(ValueError, "new output directory"):
            exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertEqual(common.load_json(self.p6 / "final_config.json"), final)

    def test_interruption_and_error_remove_stale_roles_on_admitted_request(self):
        self.select_book()
        with self.opening_games():
            exp06_book_mode.run(self.config, self.p6, smoke=True)
        for error, expected in ((KeyboardInterrupt, "pending"), (RuntimeError, "blocked")):
            with self.subTest(error=error), patch.object(exp06_book_mode, "run_pairs", side_effect=error):
                with self.assertRaises(error):
                    exp06_book_mode.run(self.config, self.p6, smoke=True)
                self.assert_unusable(common.load_json(self.p6 / "final_config.json"), expected)

    def test_h_second_source_requires_confirmation_and_preserves_actual_book_evidence(self):
        selection = self.select_book()
        self.assertTrue(selection["source_tie"])
        real_pairs = runner.run_pairs

        def matches(variant, rows, output, **kwargs):
            if variant["book"]["mode"] != "H":
                return real_pairs(variant, rows, output, **kwargs)

            # Use a real one-ply referee and control only the adjudicated score.
            class ScoredReferee(runner.SnapshotWorld):
                def play(self):
                    game = super().play()
                    candidate_seat = 1 if self.runner_1.name == "candidate" else 2
                    if output.name == "H_second" or candidate_seat == 1:
                        game.finish(candidate_seat, "line")
                    return game

            with patch.object(runner, "SnapshotWorld", ScoredReferee):
                summary = real_pairs(variant, rows, output, **kwargs)
            if output.name == "H_second" and output.parent.name == "dev" and output.is_relative_to(pending):
                summary.update(complete=False, eligible=False)
            return summary

        self.opening_endings.update(((1, "line"), (2, "line")))
        pending = self.root / "second-pending"
        with self.opening_games(), patch.object(exp06_book_mode, "run_pairs", side_effect=matches):
            result = exp06_book_mode.run(self.config, pending, smoke=True)
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["variant"])
            self.assertFalse((pending / "final_variant.json").exists())
            result = exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.assertEqual((result["status"], result["mode"]), ("selected", "H"))
        self.assertEqual(result["book_file"], selection["second_source_book_file"])
        self.assertEqual(result["book_hash"], load_book(result["book_file"])["layout"]["hash"])
        self.assertIsNone(result["book_comparison_variant"])
        self.assertFalse(self.final_roles()[1].exists())
        evidence = result["development_evidence"][result["variant"]["variant_id"]]
        self.assertEqual(evidence["summary"]["adjudicated"], 4)
        self.assertEqual(evidence["summary"]["score"], 1.0)

    def test_rejects_t_positions_and_tampered_selected_book(self):
        selection = self.select_book()
        original = self.positions
        self.positions = [dict(self.positions[0], split="T")]
        self.write_positions()
        with self.assertRaises(ValueError):
            exp06_book_mode.run(self.config, self.p6, smoke=True)
        self.positions = original
        self.write_positions()
        path = selection["book_file"]
        book = common.load_json(path)
        book["entries"] = book["entries"][:-1]
        common.atomic_json(path, book)
        with self.assertRaises(ValueError):
            exp06_book_mode.run(self.config, self.p6, smoke=True)


if __name__ == "__main__":
    unittest.main()
