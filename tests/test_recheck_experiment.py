from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agents.game_state import initial_state, legal_moves
from agents.search_engine import SearchConfig
from book_experiments import common as book
from search_experiments import common, exp02_search, exp03_evaluation, exp04_recheck as experiment, measure, runner
from tests import test_experiment_contracts as fixtures


class RecheckExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "recheck"
        manifest = runner.make_manifest()
        self.manifest_hash = manifest["manifest_hash"]
        self.code_hash = common.code_manifest()["code_hash"]
        self.config = {"manifest": str(self.root / "manifest.json"),
                       "positions": str(self.root / "positions.json"),
                       "search_selection": str(self.root / "search" / "search_selection.json"),
                       "evaluation_selection": str(self.root / "evaluation" / "evaluation_selection.json")}
        self.search_path, self.evaluation_path = (Path(self.config[key]) for key in
                                                  ("search_selection", "evaluation_selection"))
        common.atomic_json(self.config["manifest"], manifest)
        self.rows = runner.validate_manifest(common.load_json(self.config["manifest"]))
        self.development = [row for row in self.rows if row["split"] in ("D0", "D1")]
        self.d1 = [row for row in self.rows if row["split"] == "D1"]
        row = next(row for row in self.rows if row["split"] == "D0")
        self.position = {"position_id": "opening", "state": common.state_record(
            initial_state(row["obstacle"], row["removed_size"])),
            "obstacle_class": row["obstacle_class"], "split": "D0"}
        self.positions_payload = {"positions": [self.position], "positions_hash": common.digest([self.position]),
                                  "execution_tier": "formal", "reference_complete": True,
                                  "reference_eligible": True, "reference_code_hash": self.code_hash,
                                  "manifest_hash": self.manifest_hash}
        common.atomic_json(self.config["positions"], self.positions_payload)
        self.write_upstream()

    def write_upstream(self, *, star="S2_L2_Q2_N", alternate="S2_E0_N", same_search=False):
        base = common.normalize_variant(exp02_search._match_variant(
            "S2", SearchConfig(reuse_results=True), "S1_E1_N"))
        other = common.normalize_variant(exp02_search._match_variant(
            "S0", SearchConfig(reuse_results=same_search)))
        search = {"execution_tier": "formal", "status": "selected", "code_hash": self.code_hash,
                  "manifest_hash": self.manifest_hash, "positions_hash": self.positions_payload["positions_hash"],
                  "S_diag": "S2", "S_star": "S2", "S_alt": "S0", "variants": {"S2": base, "S0": other},
                  "development": {"S2": self.match_summary(base, self.development, "wins")}}
        common.atomic_json(self.search_path, search)
        common.atomic_json(self.search_path.with_name("selected_search.json"), base)
        common.atomic_json(self.search_path.with_name("alternate_search.json"), other)
        variants = {"S2_E0_N": common.normalize_variant(dict(base, variant_id="S2_E0_N",
                    parent_variant="S2_E1_N", evaluation={"evaluation_id": "E0"})), "S2_E1_N": base}
        for name, q in (("S2_L2_Q0_N", [1, 20, 100]), ("S2_L2_Q2_N", [1, 80, 100])):
            variants[name] = common.normalize_variant(dict(base, variant_id=name, parent_variant="S2_E1_N",
                evaluation={"evaluation_id": name.removeprefix("S2_").removesuffix("_N"), "lambda": 2, "q": q}))
        p3 = {"execution_tier": "formal", "status": "selected", "formal_selection": True,
              "code_hash": self.code_hash, "manifest_hash": self.manifest_hash,
              "source_search_variant": base, "effective_search_variant": base,
              "search_selection_hash": common.file_hash(self.search_path), "variants": variants,
              "E_star": star, "E_alt": alternate,
              "stages": {"confirmation": {name: self.match_summary(variants[name], self.development, outcome)
                                          for name, outcome in ((alternate, "ties"), (star, "wins"))}},
              "d0_games": {}, "development_games": {}}
        common.atomic_json(self.evaluation_path, p3)
        for name, filename in ((star, "selected_evaluation.json"), (alternate, "alternate_evaluation.json")):
            common.atomic_json(self.evaluation_path.with_name(filename), variants[name])

    def match_summary(self, variant, rows, outcome, path=None, fault=None):
        name = variant["variant_id"]
        request = {"variant": variant, "configurations": rows, "execution_tier": "formal", "strict": True,
                   "code_hash": self.code_hash, "manifest_hash": self.manifest_hash,
                   "candidate_artifact_hash": "controlled-artifact", "candidate_source_hash": "controlled-source"}
        request_hash = common.digest(request)
        schedule = [{"game_id": f"{name}__{row['config_id']}__P{seat}", "config": row, "seat": seat}
                    for row in rows for seat in (1, 2)]
        score, result = {"wins": (1., "W"), "ties": (0.5, "D"), "losses": (0., "L"),
                         "failure": (1., "W"), "pending": (1., "W")}[outcome]
        records = [{"game_id": game["game_id"], "config_id": game["config"]["config_id"],
                    "candidate_seat": game["seat"], "split": game["config"]["split"],
                    "setup_seed": game["config"]["setup_seed"], "obstacle": game["config"]["obstacle"],
                    "removed_size": game["config"]["removed_size"], "request_hash": request_hash,
                    "variant_id": name, "config_hash": variant["config_hash"],
                    "parent_variant": variant["parent_variant"], "book_hash": variant["book_hash"],
                    "code_hash": self.code_hash, "manifest_hash": self.manifest_hash,
                    "execution_tier": "formal", "strict": True, "candidate_artifact_hash": "controlled-artifact",
                    "candidate_source_hash": "controlled-source", "score": score, "outcome": result,
                    "status": "adjudicated", "technical_failure": outcome == "failure",
                    "opponent_technical_failure": False, "candidate_budget_failure": False} for game in schedule]
        if outcome == "pending":
            records.pop()
        summary = runner.summarize_games(records, [game["game_id"] for game in schedule])
        summary.update(request_hash=request_hash, variant_id=name, execution_tier="formal",
                       code_hash=self.code_hash, formal_ranking_allowed=summary["eligible"])
        if path is not None:
            if fault == "plan":
                request["variant"] = common.normalize_variant(dict(variant, parent_variant="incorrect_parent"))
            if fault == "seat":
                records[-1]["candidate_seat"] = 1
            if fault == "variant":
                records[-1]["config_hash"] = "another candidate"
            if fault == "split":
                records[-1]["split"] = "D0"
            if fault == "duplicate":
                records.append(records[-1])
            common.atomic_json(path / "plan.json", {"request": request, "request_hash": request_hash,
                                                   "games": schedule})
            for row in records:
                common.append_jsonl(path / "games.jsonl", row)
            common.atomic_json(path / "summary.json", summary)
        return summary

    def run_controlled(self, *, speed=(4, 4, 4), timing_fault=None, scores=None, d1_fault=None):
        scores = scores or {}
        calls = {}
        star_is_e1 = common.load_json(self.evaluation_path)["E_star"] == "S2_E1_N"

        def measurements(positions, configs, depths, evaluation, output, refs, **settings):
            self.assertEqual(settings["execution_tier"], "formal")
            self.assertEqual(settings["online_budget"], 1.6)
            self.assertEqual(depths, [2, 3])
            is_e1 = evaluation["evaluation_id"] == "E1"
            calls[evaluation["evaluation_id"]] = tuple(configs)
            position = positions[0]["position_id"]
            records = [fixtures.fixed(position, name, depth, workload, rep)
                       for name in configs for depth in depths for workload in ("cold", "iterative")
                       for rep in (0, 1, 2, -1)]
            for index, name in enumerate(configs):
                selected_speed = (2, 2, 2) if not index else (speed if star_is_e1 or not is_e1 else (8, 8, 8))
                for rep, depth in enumerate(selected_speed):
                    records.append(fixtures.online(position, name, rep, depth))
            if timing_fault == "partial_coverage":
                records[3].update(status="timeout", checked=False)
            if timing_fault == "wrong_E1" and is_e1:
                records[3]["correct"] = False
            if timing_fault == "missing_fixed" and not is_e1:
                records[0]["checked"] = False
            if not is_e1 and timing_fault == "missing_online":
                records = [row for row in records if row["workload"] != "online"]
            if not is_e1 and timing_fault == "over_budget":
                records[-1]["budget_ok"] = False
            return records, measure.summarize_measurements(records, configs, depths, positions, diagnostics=True)

        def matches(variant, rows, output, *, opponent, execution_tier, reuse_paths):
            self.assertEqual(opponent, {"kind": "ab", "budget": 1.5})
            self.assertEqual(execution_tier, "formal")
            d1 = output.name == "d1_reference"
            self.assertEqual(rows, self.d1 if d1 else self.development)
            calls.setdefault("matches", []).append((variant["variant_id"], len(rows), tuple(reuse_paths)))
            score = scores.get("D1" if d1 else variant["variant_id"], "ties")
            if d1 and score == "ties" and d1_fault is None:
                with fixtures.ManifestAndSelectionTests()._opening_pairs(), redirect_stdout(StringIO()):
                    return runner.run_pairs(variant, rows, output, opponent=opponent,
                                            execution_tier=execution_tier, reuse_paths=reuse_paths)
            return self.match_summary(variant, rows, score, output, d1_fault if d1 else None)

        with patch.object(experiment, "exact_references", return_value={}), \
                patch.object(experiment, "measure", side_effect=measurements), \
                patch.object(experiment, "run_pairs", side_effect=matches):
            return experiment.run(self.config, self.output), calls

    def test_cross_product_preserves_original_roles_and_e1_speed_cannot_trigger_switch(self):
        result, calls = self.run_controlled(speed=(1, 3, 1))
        self.assertEqual(calls["E1"], ("S2_E1_N", "S0_E1_N"))
        self.assertEqual(calls["L2_Q2"], ("S2_L2_Q2_N", "S0_L2_Q2_N"))
        self.assertEqual(result["roles"], {"S_star": {"E1": "S2_E1_N", "E_star": "S2_L2_Q2_N"},
                                           "S_alt": {"E1": "S0_E1_N", "E_star": "S0_L2_Q2_N"}})
        self.assertEqual(result["variants"]["S2_E1_N"]["parent_variant"], "S1_E1_N")
        self.assertEqual(result["variants"]["S0_L2_Q2_N"]["parent_variant"], "S0_E1_N")
        self.assertFalse(result["performance"]["consistent"])
        self.assertFalse(result["search_switched"])
        self.assertEqual([(name, count) for name, count, _ in calls["matches"]], [("S2_L2_Q2_N", 12)])
        self.assert_handoff(result, "S2_L2_Q2_N")

    def test_numerical_aliases_share_measurements_without_renaming_roles(self):
        for star, same_search, expected in (("S2_E1_N", False, 2), ("S2_L2_Q2_N", True, 2),
                                            ("S2_E1_N", True, 1)):
            with self.subTest(star=star, same_search=same_search):
                self.output = self.root / f"{star}-{same_search}"
                self.write_upstream(star=star, same_search=same_search)
                result, calls = self.run_controlled(speed=(1, 1, 1))
                self.assertEqual(sum(len(group["variants"]) for group in result["measurements"].values()), expected)
                self.assertEqual(result["roles"]["S_star"]["E_star"], star)
                self.assertEqual(result["roles"]["S_alt"]["E1"], "S0_E1_N")
                if same_search:
                    self.assertEqual(result["measurement_roles"]["S_alt"]["E1"], "S2_E1_N")
                    self.assertEqual(result["performance"]["reason"], "identical_search")
                self.assertEqual(len(calls["matches"]), 1)
                self.assert_handoff(result, star)

    def test_wrong_or_insufficient_correctness_and_budget_evidence_block(self):
        for fault in ("wrong_E1", "missing_fixed", "missing_online", "over_budget", "identical_missing_online"):
            with self.subTest(fault=fault):
                self.output = self.root / fault
                self.write_upstream(same_search=fault == "identical_missing_online")
                fault = fault.removeprefix("identical_")
                result, calls = self.run_controlled(timing_fault=fault)
                self.assertEqual(result["status"], "blocked")
                self.assertFalse(result["formal_selection"])
                self.assertNotIn("matches", calls)
                self.assertFalse((self.output / "online_config.json").exists())

    def test_common_coverage_can_pass_without_every_diagnostic(self):
        result, calls = self.run_controlled(speed=(1, 1, 1), timing_fault="partial_coverage")
        self.assertEqual(result["status"], "selected")
        self.assertTrue(all(not group["summary"]["verification_complete"] for group in result["measurements"].values()))
        self.assertEqual(len(calls["matches"]), 1)

    def test_faster_alternate_losing_tying_or_failing_matches_keeps_current(self):
        for current, other in (("wins", "losses"), ("ties", "ties"), ("ties", "failure")):
            with self.subTest(current=current, other=other):
                self.output = self.root / other
                result, calls = self.run_controlled(scores={"S2_L2_Q2_N": current, "S0_L2_Q2_N": other})
                self.assertTrue(result["performance"]["retain_candidate"])
                self.assertFalse(result["search_switched"])
                self.assertNotIn("evaluation_recheck", result["stages"])
                self.assertEqual([(name, count) for name, count, _ in calls["matches"]],
                                 [("S2_L2_Q2_N", 24), ("S0_L2_Q2_N", 24), ("S2_L2_Q2_N", 12)])
                self.assert_handoff(result, "S2_L2_Q2_N")

    def test_faster_alternate_winning_switches_once_and_evaluation_ties_prefer_e0(self):
        for current, other in (("losses", "ties"), ("failure", "losses")):
            with self.subTest(current=current):
                self.output = self.root / current
                scores = {"S2_L2_Q2_N": current, "S0_L2_Q2_N": other, "S0_E0_N": other}
                result, calls = self.run_controlled(scores=scores)
                self.assertTrue(result["search_switched"])
                self.assertEqual(set(result["stages"]["evaluation_recheck"]), {"S0_L2_Q2_N", "S0_E0_N"})
                self.assertEqual(result["evaluation_recheck_roles"], {"E_star": "S0_L2_Q2_N", "E_alt": "S0_E0_N"})
                self.assertEqual([(name, count) for name, count, _ in calls["matches"]],
                                 [("S2_L2_Q2_N", 24), ("S0_L2_Q2_N", 24), ("S0_E0_N", 24), ("S0_E0_N", 12)])
                self.assertIn(self.output / "development" / "S0_E0_N" / "games.jsonl", calls["matches"][-1][2])
                self.assert_handoff(result, "S0_E0_N")

    def test_e1_and_custom_creation_order_break_recheck_ties(self):
        for other, expected in (("S2_E1_N", "S0_E1_N"), ("S2_L2_Q0_N", "S0_L2_Q0_N")):
            with self.subTest(other=other):
                self.output = self.root / other
                self.write_upstream(alternate=other)
                result, _ = self.run_controlled(scores={"S2_L2_Q2_N": "losses"})
                self.assert_handoff(result, expected)

    def test_switch_with_aliased_evaluations_schedules_no_extra_development_games(self):
        self.write_upstream(alternate="S2_L2_Q2_N")
        result, calls = self.run_controlled(scores={"S2_L2_Q2_N": "losses"})
        self.assertTrue(result["search_switched"])
        self.assertEqual([(name, count) for name, count, _ in calls["matches"]],
                         [("S2_L2_Q2_N", 24), ("S0_L2_Q2_N", 24), ("S0_L2_Q2_N", 12)])
        self.assertEqual(set(result["stages"]["evaluation_recheck"]), {"S0_L2_Q2_N"})
        self.assert_handoff(result, "S0_L2_Q2_N")

    def test_required_pending_and_rejected_matches_invalidate_online_config(self):
        for case, scores, expected in (
                ("search_pending", {"S0_L2_Q2_N": "pending"}, "pending"),
                ("search_rejected", {"S2_L2_Q2_N": "failure", "S0_L2_Q2_N": "failure"}, "blocked"),
                ("evaluation_pending", {"S2_L2_Q2_N": "losses", "S0_E0_N": "pending"}, "pending"),
                ("d1_pending", {"D1": "pending"}, "pending"), ("d1_rejected", {"D1": "failure"}, "blocked")):
            with self.subTest(case=case):
                self.output = self.root / case
                common.atomic_json(self.output / "online_config.json", {"stale": True})
                result, _ = self.run_controlled(scores=scores)
                self.assertEqual(result["status"], expected)
                self.assertFalse(result["formal_selection"])
                self.assertFalse((self.output / "online_config.json").exists())

    def test_d1_plan_and_all_adjudicated_seats_must_match_exact_frozen_candidate(self):
        for fault in ("plan", "seat", "variant", "split", "duplicate"):
            with self.subTest(fault=fault):
                self.output = self.root / fault
                result, _ = self.run_controlled(speed=(1, 1, 1), d1_fault=fault)
                self.assertEqual(result["status"], "blocked")
                self.assertFalse((self.output / "online_config.json").exists())

    def test_provenance_guard_never_changes_an_accepted_output(self):
        result, _ = self.run_controlled(speed=(1, 1, 1))
        saved = {path: path.read_bytes() for path in self.output.rglob("*.json")}
        p3 = common.load_json(self.evaluation_path)
        edits = [(self.evaluation_path, {"execution_tier": "smoke", "formal_selection": False,
                                         "status": "smoke_only", "diagnostic_status": "selected"}),
                 (self.evaluation_path, {"status": "pending"}),
                 (self.evaluation_path, {"code_hash": "stale"}),
                 (self.evaluation_path, {"manifest_hash": "different"}),
                 (self.evaluation_path, {"search_selection_hash": "different"}),
                 (self.evaluation_path, {"source_search_variant": p3["variants"]["S2_E0_N"]}),
                 (self.evaluation_path, {"stages": {"confirmation": {}}}),
                 (self.evaluation_path.with_name("selected_evaluation.json"), {"evaluation": {"evaluation_id": "E1"}}),
                 (self.search_path, {"code_hash": "stale"}),
                 (Path(self.config["positions"]), {"manifest_hash": "different"}),
                 (Path(self.config["positions"]), {"reference_code_hash": "stale"}),
                 (Path(self.config["positions"]), {"execution_tier": "smoke"})]
        alternate_positions = [dict(self.position, position_id="other")]
        edits.append((Path(self.config["positions"]), {"positions": alternate_positions,
                                                        "positions_hash": common.digest(alternate_positions)}))
        for path, change in edits:
            with self.subTest(path=path.name, change=change):
                original = common.load_json(path)
                common.atomic_json(path, dict(original, **change))
                with patch.object(experiment, "measure", side_effect=AssertionError("started work")):
                    with self.assertRaises(ValueError):
                        experiment.run(self.config, self.output)
                self.assertEqual({path: path.read_bytes() for path in saved}, saved)
                common.atomic_json(path, original)
        with patch.object(experiment, "measure", side_effect=AssertionError("started work")):
            with self.assertRaises(ValueError):
                experiment.run(dict(self.config, guard_seconds=61), self.output)
        self.assertEqual(common.load_json(self.output / "recheck_selection.json"), result)

    def assert_handoff(self, result, name):
        path = self.output / "online_config.json"
        config = dict(self.config, online_config=path)
        variant = book.load_online(config, smoke=False)
        schedule = common.load_json(self.output / "d1_reference" / "plan.json")
        # Match the one-ply cap of the controlled referee that produced these rows.
        with patch("agents.game_state.MAX_PLIES", 1):
            records, check = book.reference_games(self.output / "d1_reference" / "games.jsonl", variant, config)
        self.assertEqual([(row["config_id"], row["candidate_seat"]) for row in records],
                         [(row["config_id"], seat) for row in self.d1 for seat in (1, 2)])
        self.assertEqual(check["variant"]["variant_id"], name)
        self.assertEqual(check["summary"]["adjudicated"], 24)
        self.assertTrue(check["summary"]["formal_ranking_allowed"])
        self.assertEqual(variant, schedule["request"]["variant"])
        self.assertEqual(variant, result["chosen_variant"])
        self.assertEqual(common.load_json(path), {"variant": variant, "status": "selected",
                         "execution_tier": "formal", "formal_selection": True,
                         "code_hash": self.code_hash, "manifest_hash": self.manifest_hash})
        self.assertEqual(common.load_json(self.output / "recheck_selection.json"), result)

    def test_bounded_smoke_step2_p3_p4_and_book_handoff_resumes_exactly(self):
        common.atomic_json(self.config["positions"], dict(self.positions_payload, execution_tier="smoke"))

        # Keep real exact/fixed measurements and the official one-ply referee.
        # Fix only online timing and the subprocess replies so the fixture is deterministic.
        def timing(state, variant, **kwargs):
            depth = 2 if variant["variant_id"] in ("S0", "S0_E1_N") else 1
            return {"move": legal_moves(state)[0], "value": 0, "status": "completed",
                    "completed_depth": depth, "elapsed": 0.001, "budget_ok": True, "hard_limit_ok": True}

        with fixtures.ManifestAndSelectionTests()._opening_pairs() as openings, \
                patch.object(measure, "measure_online_step", side_effect=timing), redirect_stdout(StringIO()):
            exp02_search.run(self.config | {"smoke": {"max_positions": 1, "depths": [1],
                             "guard_seconds": 2, "online_budget": 0.03}}, self.search_path.parent, smoke=True)
            p3 = exp03_evaluation.run(self.config, self.evaluation_path.parent, smoke=True)
            config_path = self.root / "recheck-config.json"
            common.atomic_json(config_path, self.config)
            argv = ["exp04_recheck", "--config", str(config_path), "--output", str(self.output), "--smoke"]
            with patch.object(sys, "argv", argv):
                experiment.main()
            result = common.load_json(self.output / "recheck_selection.json")
            paths = list(self.output.rglob("*.jsonl"))
            before = {path: path.read_bytes() for path in paths}
            self.assertEqual(len(openings), 22)
            with patch.object(measure.SearchEngine, "fixed_depth", side_effect=AssertionError("fixed rerun")), \
                    patch.object(measure, "measure_online_step", side_effect=AssertionError("online rerun")):
                self.assertEqual(experiment.run(self.config, self.output, smoke=True), result)
            self.assertEqual(len(openings), 22)
            self.assertEqual({path: path.read_bytes() for path in paths}, before)

        self.assertEqual((p3["status"], result["status"], result["diagnostic_status"]),
                         ("smoke_only", "smoke_only", "selected"))
        self.assertFalse(p3["formal_selection"])
        self.assertFalse(result["formal_selection"])
        config_path = self.output / "online_config.json"
        self.assertFalse(common.load_json(config_path)["formal_selection"])
        book_config = dict(self.config, online_config=config_path, smoke={"max_configs": 1})
        online = book.load_online(book_config, smoke=True)
        self.assertEqual((online["variant_id"], online["step_budget"], online["max_depth"]), ("S0_E1_N", 0.012, 2))
        d1 = self.output / "d1_reference"
        # Match the one-ply cap of the controlled referee that produced these rows.
        with patch("agents.game_state.MAX_PLIES", 1):
            records, check = book.reference_games(d1 / "games.jsonl", online, book_config, smoke=True)
        self.assertEqual(online, common.load_json(d1 / "plan.json")["request"]["variant"])
        self.assertEqual(check["variant"]["variant_id"], "S0_E1_N")
        self.assertEqual(check["summary"]["adjudicated"], 2)
        self.assertFalse(check["summary"]["formal_ranking_allowed"])
        self.assertEqual([(row["config_id"], row["candidate_seat"]) for row in records],
                         [(self.d1[0]["config_id"], 1), (self.d1[0]["config_id"], 2)])
        previous = common.read_jsonl(p3["development_games"]["S0_E1_N"])
        self.assertEqual([row["reused_from_request_hash"] for row in records], [previous[0]["request_hash"]] * 2)
        self.assertTrue(all(len(row["turns"]) == 1 for row in records))
        self.assertTrue(any(path.name == "measurements.jsonl" for path in paths))
        self.assertEqual(len(common.read_jsonl(self.output / "measurements" / "E1" / "measurements.jsonl")), 22)


if __name__ == "__main__":
    unittest.main()
