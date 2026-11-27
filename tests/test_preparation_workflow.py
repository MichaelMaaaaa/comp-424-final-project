"""Preparation orchestration tests. Stage doubles never launch games."""
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search_experiments import common, exp07_final, runner


def forbid(*args, **kwargs):
    raise AssertionError("unexpected live game or validator")


def fail_official(report, games, rules, in_process):
    report.add("FAIL", "controlled official failure; no games launched")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ("validate_submission.dynamic_phase", "search_experiments.exp07_final.dynamic_phase",
                       "book_experiments.collect.play_collection_game", "world.World.play", "world.World.step"):
            self.stack.enter_context(patch(target, side_effect=forbid))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.out = self.root / "out"

    def module(self, name):
        try:
            return import_module("search_experiments." + name)
        except ModuleNotFoundError as error:
            self.fail(str(error))

    def acceptance(self):
        module = self.module("acceptance")
        from book_experiments.books import write_provider

        def temporary_provider(record):
            target = common.REPO_ROOT.joinpath(*record["provider"].split(".")).with_suffix(".py")
            if not target.exists():
                self.addCleanup(target.unlink, missing_ok=True)
            return write_provider(record)

        self.stack.enter_context(patch.object(module, "write_provider", side_effect=temporary_provider))
        return module

    def test_pipeline_binds_paths_tier_and_stops_at_published_status(self):
        pipeline = self.module("smoke_pipeline")
        cfg = common.load_json(common.REPO_ROOT / "search_experiments/configs/smoke_pipeline.json")
        calls = []
        def produce(name, config, output):
            calls.append(name)
            self.assertEqual(output, self.out / name)
            if name == "prepare":
                self.assertEqual(config["smoke"]["step_budget"], 0.1)
                common.atomic_json(output / "manifest.json", runner.make_manifest())
            else:
                self.assertEqual(config["positions"], str(self.out / "prepare/positions.json"))
            common.atomic_json(output / "observations.json", {"name": name})

        def check(name, config, directory):
            if name == "components":
                return "blocked", "component correctness"
            return "complete", None

        with patch.object(pipeline, "_execute", produce), patch.object(pipeline, "_check_stage", check):
            result = pipeline.run(cfg, self.out)
            self.assertEqual((result["status"], calls), ("blocked", ["prepare", "components"]))
            self.assertIs(result["formal_selection"], False)
            self.assertEqual(result["stages"]["search"]["status"], "pending")
            calls.clear()
            pipeline.run(cfg, self.out)
            self.assertEqual(calls, ["components"])
            with patch.object(pipeline, "machine_info", return_value={"host": "changed"}):
                with self.assertRaisesRegex(ValueError, "new output"):
                    pipeline.run(cfg, self.out)
            with patch.object(pipeline, "code_identity", return_value={"code_hash": "changed"}):
                with self.assertRaisesRegex(ValueError, "new output"):
                    pipeline.run(cfg, self.out)
            with self.assertRaises(ValueError):
                pipeline.run(dict(cfg, sample_seed=7), self.out)
        self.assertEqual(common.load_json(self.out / "configs/report.json")["diagnostic_dirs"],
                         [{"directory": str(self.out / "components/components"),
                           "positions": str(self.out / "prepare/positions.json"), "max_positions": 1}])
        for name in ("evaluation", "search", "recheck", "book", "book_mode", "final"):
            stage = common.load_json(self.out / "configs" / (name+".json"))
            self.assertEqual((stage["smoke"]["step_budget"], stage["smoke"]["max_depth"],
                              stage["smoke"]["opponent_budget"]), (0.1, 1, 0.006))
            self.assertEqual(stage["manifest"], str(self.out / "prepare/manifest.json"))

    def test_smoke_handoff_reads_artifact_even_if_producer_returned(self):
        pipeline = self.module("smoke_pipeline")
        path = self.out / "evaluation_selection.json"
        for status in ("pending", "blocked"):
            common.atomic_json(path, dict(status="smoke_only", diagnostic_status=status,
                                          execution_tier="smoke", formal_selection=False, reason="evidence"))
            self.assertEqual(pipeline._check_stage("evaluation", {}, self.out), (status, "evidence"))
        common.atomic_json(path, dict(status="selected", execution_tier="formal", formal_selection=True))
        with self.assertRaisesRegex(ValueError, "tier"):
            pipeline._check_stage("evaluation", {}, self.out)

    def test_controlled_probe_records_real_mode_semantics(self):
        pipeline = self.module("smoke_pipeline")
        result = pipeline.controlled_probe(self.out)
        self.assertIs(result["modes"]["N"]["book_queried"], False)
        self.assertEqual(result["modes"]["D"]["book_action"], "direct")
        self.assertEqual(result["modes"]["H"]["book_action"], "hint")
        self.assertEqual(result["modes"]["D"]["book_hash"], result["modes"]["H"]["book_hash"])
        self.assertTrue((self.out / "controlled_probe.json").is_file())

    def test_completed_stages_are_reused_and_changed_evidence_stays_blocked(self):
        pipeline = self.module("smoke_pipeline")
        cfg = common.load_json(pipeline.PROFILE)
        calls = []
        def produce(name, config, output):
            calls.append(name)
            common.atomic_json(output / "observations.json", {"stage": name})
            if name == "prepare":
                common.atomic_json(output / "manifest.json", runner.make_manifest())
            if name == "report":
                common.atomic_json(output / "report_summary.json", {"files": {"results.csv": str(output / "results.csv")}})
            if name == "book":
                common.atomic_json(output / "selected_book.json", {"status": "no_common_book", "K": 0, "selected_book": None})
        with patch.object(pipeline, "_execute", produce), patch.object(pipeline, "_check_stage", return_value=("complete", None)):
            complete = pipeline.run(cfg, self.out)
            self.assertEqual(calls, ["prepare", "components", "search", "evaluation", "recheck", "book", "book_mode", "final", "report"])
            self.assertEqual(complete["report_paths"]["results.csv"], str(self.out / "report/results.csv"))
            self.assertEqual(complete["book_identity"], {"status": "no_common_book", "K": 0, "selected_book": None})
            calls.clear()
            self.assertEqual(pipeline.run(cfg, self.out)["status"], "complete")
            common.atomic_json(self.out / "components/observations.json", {"changed": True})
            for _ in range(2):
                self.assertEqual(pipeline.run(cfg, self.out)["status"], "blocked")
            self.assertEqual(calls, [])

    def test_held_out_game_in_smoke_evidence_is_rejected(self):
        pipeline = self.module("smoke_pipeline")
        for label in ("T", "D0"):
            directory = self.out / label
            common.append_jsonl(directory / "games.jsonl", {"split": label, "config_id": "held_out", "game_id": "bad"})
            with self.assertRaisesRegex(ValueError, "development"):
                pipeline._evidence(directory, {"held_out": "T"})

    def test_preflight_rejects_wrong_runtime_and_held_out_reservation_before_gate(self):
        acceptance = self.acceptance()
        manifest = runner.make_manifest()
        path = self.root / "manifest.json"
        common.atomic_json(path, manifest)
        with patch.object(exp07_final, "_official_gate", side_effect=forbid), \
                patch.object(acceptance, "runtime", return_value={"python": "3.12.0", "numpy": "1.26.4", "matplotlib": "3.8.4"}):
            with self.assertRaisesRegex(ValueError, "3.11"):
                acceptance.run({"manifest": str(path)}, self.out)
        next(row for row in manifest["rows"] if row["setup_seed"] == 0)["split"] = "T"
        manifest["manifest_hash"] = common.digest({k: v for k, v in manifest.items() if k!="manifest_hash"})
        common.atomic_json(path, manifest)
        with patch.object(exp07_final, "_official_gate", side_effect=forbid):
            with self.assertRaisesRegex(ValueError, "reserved"):
                acceptance.run({"manifest": str(path)}, self.out)

    def test_preflight_freezes_all_bytes_before_official_and_blocks_strict(self):
        acceptance = self.acceptance()
        common.atomic_json(self.root / "manifest.json", runner.make_manifest())
        config = {"manifest": str(self.root / "manifest.json")}
        def official(path, artifact, binding):
            self.assertEqual(len(list((self.out / "candidates").glob("*/student_agent.py"))), 3)
            saved = dict(binding, status="passed", reason=None, report={"verdict": "PASS", "dynamic": [
                {"max_step": 1.0, "init_time": 0.5, "max_rss_mb": 30}]})
            common.atomic_json(path, saved)
            return saved
        def strict(variant, rows, opp, tier, directory, binding):
            self.assertEqual([(r["split"], r["setup_seed"]) for r in rows], [("D0", 0)])
            self.assertEqual((tier, variant["step_budget"], variant.get("max_depth"), opp["budget"]),
                             ("formal", 1.6, None, 1.5))
            return dict(binding, status="blocked", reason="strict AB preflight ineligible",
                        summary={"complete": True, "eligible": False}, games_path=str(directory / "games.jsonl"))
        with patch.object(exp07_final, "_official_gate", side_effect=official), \
                patch.object(exp07_final, "_strict_preflight", side_effect=strict):
            result = acceptance.run(config, self.out)
        self.assertEqual((result["purpose"], result["status"], result["candidate_ranking_allowed"]),
                         ("preflight", "blocked", False))
        for mode in "NDH":
            entry = result["artifacts"][mode]
            self.assertEqual(Path(entry["path"]).stat().st_size, entry["bytes"])
            self.assertEqual(common.file_hash(entry["path"]), entry["artifact_hash"])
        with patch.object(exp07_final, "_official_gate", side_effect=forbid):
            changed = self.root / "another.json"
            common.atomic_json(changed, runner.make_manifest(42))
            with self.assertRaisesRegex(ValueError, "new output"):
                acceptance.run({"manifest": str(changed)}, self.out)
            path = Path(result["artifacts"]["D"]["path"])
            path.write_bytes(b"changed")
            self.assertEqual(acceptance.run(config, self.out)["status"], "blocked")
            path.unlink()
            self.assertEqual(acceptance.run(config, self.out)["status"], "blocked")

    def test_preflight_recovers_book_interrupted_before_initial_checkpoint(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        config = {"manifest": str(manifest)}
        book_path = self.out / "representative_book.json"

        def interrupt_book(path, value):
            if Path(path) == book_path:
                raise KeyboardInterrupt("before initial book checkpoint")
            return common.atomic_json(path, value)

        with patch.object(acceptance, "atomic_json", side_effect=interrupt_book):
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run(config, self.out)
        saved = common.load_json(self.out / "acceptance_summary.json")
        self.assertEqual(saved["status"], "pending")
        self.assertFalse(book_path.exists())
        self.assertTrue(all(entry[key] is None for entry in saved["artifacts"].values()
                            for key in ("static", "official", "strict")))
        with patch.object(exp07_final, "dynamic_phase", side_effect=fail_official), \
                patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
            resumed = acceptance.run(config, self.out)
        self.assertEqual(resumed["reason"], "N: official validator failed")
        self.assertEqual(resumed["artifacts"]["N"]["static"]["status"], "passed")
        frozen = common.load_json(self.out / "final_plan.json")["request"]
        self.assertEqual(common.digest(common.load_json(book_path)), frozen["representative_book_hash"])

    def test_preflight_recovers_interrupted_initial_candidate_installation(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        config = {"manifest": str(manifest)}
        missing = self.out / "candidates/preflight_S0_E1_D/student_agent.py"
        install = exp07_final._install_candidate

        def interrupt_install(staged, target, metadata):
            if target == missing:
                raise KeyboardInterrupt("before initial candidate installation")
            return install(staged, target, metadata)

        with patch.object(exp07_final, "_install_candidate", side_effect=interrupt_install):
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run(config, self.out)
            self.assertTrue((self.out / "final_plan.json").is_file())
            self.assertFalse((self.out / "acceptance_summary.json").exists())
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run(config, self.out)
        saved = common.load_json(self.out / "acceptance_summary.json")
        self.assertEqual(saved["status"], "pending")
        self.assertFalse(missing.exists())
        with patch.object(exp07_final, "dynamic_phase", side_effect=fail_official), \
                patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
            resumed = acceptance.run(config, self.out)
        self.assertEqual(resumed["reason"], "N: official validator failed")
        frozen = common.load_json(self.out / "final_plan.json")["request"]["artifacts"]
        for entry in resumed["artifacts"].values():
            path = Path(entry["path"])
            metadata = frozen[entry["variant"]["variant_id"]]
            self.assertEqual((common.file_hash(path), path.stat().st_size),
                             (metadata["artifact_hash"], metadata["bytes"]))
            self.assertEqual(common.load_json(path.with_suffix(".meta.json")), metadata)

    def test_preflight_initial_recovery_preserves_changed_existing_files(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        config = {"manifest": str(manifest)}
        install = exp07_final._install_candidate
        for label, relative, reason in (
                ("book", "representative_book.json", "saved representative book differs"),
                ("agent", "candidates/preflight_S0_E1_N/student_agent.py", "artifact bytes differ"),
                ("metadata", "candidates/preflight_S0_E1_D/student_agent.meta.json", "artifact metadata differs")):
            with self.subTest(file=label):
                output = self.out / label
                missing = output / "candidates/preflight_S0_E1_D/student_agent.py"

                def interrupt_install(staged, target, metadata):
                    if target == missing:
                        raise KeyboardInterrupt("before initial candidate installation")
                    return install(staged, target, metadata)

                with patch.object(exp07_final, "_install_candidate", side_effect=interrupt_install):
                    with self.assertRaises(KeyboardInterrupt):
                        acceptance.run(config, output)
                self.assertFalse((output / "acceptance_summary.json").exists())
                changed = output / relative
                common.atomic_json(changed, {"changed": True})
                changed_hash = common.file_hash(changed)
                with patch.object(exp07_final, "dynamic_phase", side_effect=fail_official), \
                        patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
                    for _ in range(2):
                        result = acceptance.run(config, output)
                        self.assertEqual(result["status"], "blocked")
                        self.assertIn(reason, result["reason"])
                        self.assertEqual(common.file_hash(changed), changed_hash)
                if label == "metadata":
                    self.assertFalse(missing.exists())

    def test_preflight_persists_gate_boundary_before_first_static_result(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        checkpoints = []

        def interrupt_static(artifact):
            checkpoints.append(common.load_json(self.out / "acceptance_summary.json"))
            raise KeyboardInterrupt("before first static result")

        with patch.object(exp07_final, "static_phase", side_effect=interrupt_static):
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run({"manifest": str(manifest)}, self.out)
        at_gate, = checkpoints
        interrupted = common.load_json(self.out / "acceptance_summary.json")
        for checkpoint in (at_gate, interrupted):
            self.assertEqual(checkpoint.get("phase"), "gates_started")
            self.assertTrue(all(entry[key] is None for entry in checkpoint["artifacts"].values()
                                for key in ("static", "official", "strict")))

    def test_preflight_missing_files_stay_blocked_after_gate_start_without_results(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        config = {"manifest": str(manifest)}
        with patch.object(exp07_final, "static_phase", side_effect=KeyboardInterrupt("no static result")):
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run(config, self.out)
        for relative, reason in (
                ("representative_book.json", "saved representative book differs"),
                ("candidates/preflight_S0_E1_D/student_agent.py", "artifact bytes differ"),
                ("candidates/preflight_S0_E1_D/student_agent.meta.json", "artifact metadata differs")):
            with self.subTest(file=relative):
                path = self.out / relative
                original = path.read_bytes()
                path.unlink()
                with patch.object(exp07_final, "dynamic_phase", side_effect=fail_official), \
                        patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
                    for _ in range(2):
                        result = acceptance.run(config, self.out)
                        self.assertEqual(result["status"], "blocked")
                        self.assertIn(reason, result["reason"])
                        self.assertFalse(path.exists())
                path.write_bytes(original)

    def test_preflight_legacy_summary_cannot_recreate_missing_book(self):
        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        config = {"manifest": str(manifest)}
        with patch.object(exp07_final, "static_phase", side_effect=KeyboardInterrupt("no static result")):
            with self.assertRaises(KeyboardInterrupt):
                acceptance.run(config, self.out)
        summary_path = self.out / "acceptance_summary.json"
        legacy = common.load_json(summary_path)
        legacy.pop("phase", None)
        common.atomic_json(summary_path, legacy)
        book_path = self.out / "representative_book.json"
        book_path.unlink()
        with patch.object(exp07_final, "dynamic_phase", side_effect=fail_official), \
                patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
            for _ in range(2):
                result = acceptance.run(config, self.out)
                self.assertIn("saved representative book differs", result["reason"])
                self.assertFalse(book_path.exists())

    def test_official_report_drift_stays_blocked_after_one_failed_resume(self):
        acceptance = self.acceptance()
        common.atomic_json(self.root / "manifest.json", runner.make_manifest())
        config = {"manifest": str(self.root / "manifest.json")}
        def gate(path, artifact, binding):
            saved = dict(binding, status="blocked", reason="official failure", report={"verdict": "FAIL", "dynamic": []})
            common.atomic_json(path, saved)
            return saved
        with patch.object(exp07_final, "_official_gate", side_effect=gate):
            result = acceptance.run(config, self.out)
        Path(result["artifacts"]["N"]["official"]["path"]).write_text("{}")
        with patch.object(exp07_final, "_official_gate", side_effect=forbid):
            for _ in range(2):
                self.assertIn("evidence changed", acceptance.run(config, self.out)["reason"])

    def test_preflight_propagates_pending_and_failed_official(self):
        acceptance = self.acceptance()
        common.atomic_json(self.root / "manifest.json", runner.make_manifest())
        for status in ("pending", "blocked"):
            def gate(path, artifact, binding):
                saved = dict(binding, status=status, reason="official evidence", report={"verdict": "FAIL", "dynamic": []})
                common.atomic_json(path, saved)
                return saved
            with patch.object(exp07_final, "_official_gate", side_effect=gate), \
                    patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
                result = acceptance.run({"manifest": str(self.root / "manifest.json")}, self.out / status)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["artifacts"]["N"]["official"]["measurements"])

    def test_strict_resources_use_returned_calls_and_observed_initialization(self):
        from helpers import Move
        from world import GameRecord

        acceptance = self.acceptance()
        # The fourth candidate strike ends the game before its move can be applied.
        for label, seat, winner, reason, calls, init, expected in (
                ("fourth_strike", 1, 2, "too_many_strikes", [0.09, 0.08, 0.0965432, 2.1256789],
                 0.3, (2.1256789, 0.3)),
                ("unstarted", 2, 2, "init_failed", [], 0.0, (None, None)),
                ("failed_init_unknown", 1, 2, "init_timeout", [], 0.0, (None, None)),
                ("failed_init_measured", 2, 1, "init_failed", [], 0.25, (None, 0.25)),
                ("observed_zero", 1, 0, "ply_cap", [0.0], 0.0, (0.0, 0.0))):
            with self.subTest(label=label):
                game = GameRecord(0, (0, 0), 0, common.OFFICIAL_RULES, {})
                game.init_time[seat] = init
                applied = calls
                if reason == "too_many_strikes":
                    game.strikes[seat] = 4
                    applied = calls[:-1]
                for i, elapsed in enumerate(applied):
                    game.add(game.plies, seat, Move(None, (1, i), 0), elapsed, elapsed,
                             ["illegal_move"] if reason == "too_many_strikes" else [])
                    if reason == "too_many_strikes":
                        game.add(game.plies, 2, Move(None, (2, i), 0), 0.01, 0.01, [])
                game.finish(winner, reason)
                world = SimpleNamespace(turns=[], step_times={seat: calls})
                base = dict(game_id=label, candidate_seat=seat)
                row = runner._adjudicated_row(base, world, game, seat, 1.6)
                path = self.root / label / "games.jsonl"
                common.append_jsonl(path, row)
                raw_hash = common.file_hash(path)
                resources = acceptance._strict_resources(path)
                sample, = resources["adjudications"]
                for measurements in (sample, resources["candidate_maxima"]):
                    self.assertEqual((measurements["max_step"], measurements["init_time"]), expected)
                self.assertEqual(common.file_hash(path), raw_hash)
                self.assertEqual(sample["outcome"], row["outcome"])
                self.assertEqual(row["candidate_budget_failure"], label == "fourth_strike")

    def test_official_summary_distinguishes_default_and_observed_zero_steps(self):
        from tests.test_final_experiment import dynamic_result

        acceptance = self.acceptance()
        manifest = self.root / "manifest.json"
        common.atomic_json(manifest, runner.make_manifest())
        for label, plies, passes, expected in (
                ("load_failure", [], 0, None),
                ("first_call_forfeit", [0, 1, 0, 1], 0, None),
                ("only_pass", [2, 1, 0, 1], 1, None),
                ("observed_zero", [0, 3, 0, 1], 0, 0.0)):
            with self.subTest(label=label):
                group = dynamic_result()
                group["games"] = [dict(game, plies=n, outcome="L", reason="crash")
                                  for game, n in zip(group["games"], plies)]
                group.update(D=0, L=len(plies), forfeits=len(plies), passes=passes,
                             max_step=0.0, mean_step=0.0)
                if not plies:
                    group.update(load_error="could not start", init_time=None, max_rss_mb=None)

                def dynamic(report, games, rules, in_process):
                    report.dynamic.append(group)
                    report.add("FAIL", "controlled load failure or forfeit")

                with patch.object(exp07_final, "dynamic_phase", side_effect=dynamic), \
                        patch.object(exp07_final, "_strict_preflight", side_effect=forbid):
                    result = acceptance.run({"manifest": str(manifest)}, self.out / label)
                official = result["artifacts"]["N"]["official"]
                self.assertEqual((result["status"], official["status"], official["verdict"]),
                                 ("blocked", "blocked", "FAIL"))
                raw = common.load_json(official["path"])["report"]["dynamic"][0]
                self.assertEqual((raw["max_step"], raw["mean_step"]), (0.0, 0.0))
                self.assertEqual(official["measurements"], dict(max_step=expected, mean_step=expected,
                                 init_time=raw["init_time"], max_rss_mb=raw["max_rss_mb"]))

    def test_representative_gates_and_summary_on_all_modes_and_identical_resume(self):
        from agents.game_state import initial_state, legal_moves
        from agents.symmetry import canonicalize, decode_book_state
        from book_experiments.books import make_book
        from tests.test_final_experiment import dynamic_result

        acceptance = self.acceptance()
        common.atomic_json(self.root / "manifest.json", runner.make_manifest())
        key, _ = canonicalize(initial_state())
        book = make_book([(key, legal_moves(decode_book_state(key))[0])], "fixture")
        common.atomic_json(self.root / "input_book.json", book)
        config = {"manifest": str(self.root / "manifest.json"), "book": str(self.root / "input_book.json")}
        generated = []
        def dynamic(report, games, rules, in_process):
            report.dynamic.append(dynamic_result())
        def strict(variant, rows, opponent, tier, directory, binding):
            path = directory / "games.jsonl"
            if not path.exists():
                generated.append(variant["book"]["mode"])
                for seat in (1, 2):
                    common.append_jsonl(path, dict(game_id=f"seat{seat}", candidate_seat=seat, status="adjudicated",
                        outcome="D", candidate_max_step_s=0.4*seat, referee={"reason": "ply_cap", "winner": 0,
                            "time": {str(seat): {"max": 0.4*seat, "init": 0.25}},
                                            "max_rss_mb": {str(seat): 40+seat}}))
            return dict(binding, status="passed", reason=None, games_path=str(path),
                        summary={"planned": 2, "adjudicated": 2, "complete": True, "eligible": True})
        with patch.object(exp07_final, "dynamic_phase", side_effect=dynamic) as calls, \
                patch.object(exp07_final, "_strict_preflight", side_effect=strict):
            for _ in range(2):
                result = acceptance.run(config, self.out)
                self.assertEqual(result["status"], "passed", result["reason"])
                for mode in "NDH":
                    entry = result["artifacts"][mode]
                    self.assertEqual(entry["official"]["verdict"], "PASS")
                    self.assertEqual(entry["official"]["measurements"]["max_step"], 1.0)
                    self.assertEqual(entry["official"]["measurements"]["mean_step"], 0.25)
                    self.assertEqual(entry["strict"]["resources"]["candidate_maxima"],
                                     {"max_step": 0.8, "init_time": 0.25, "max_rss_mb": 42})
                self.assertEqual(result["artifacts"]["H"]["variant"]["book_hash"], book["layout"]["hash"])
                self.assertEqual(result["artifacts"]["D"]["variant"]["book_hash"], book["layout"]["hash"])
            self.assertEqual(calls.call_count, 3)
        self.assertEqual(generated, ["N", "D", "H"])
