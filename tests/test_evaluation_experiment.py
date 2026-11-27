from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agents.evaluation import evaluation_parameters
from agents.search_engine import SearchConfig
from search_experiments import common, development, exp02_search, exp03_evaluation as experiment, runner
from tests import test_experiment_contracts as fixtures


class EvaluationExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "evaluation"
        self.search = self.root / "search" / "search_selection.json"
        self.manifest = runner.make_manifest()
        self.code_hash = common.code_manifest()["code_hash"]
        self.config = {"manifest": self.root / "manifest.json", "search_selection": self.search}
        common.atomic_json(self.config["manifest"], self.manifest)
        self.rows = runner.validate_manifest(common.load_json(self.config["manifest"]))
        self.variants = {
            "S0": exp02_search._match_variant("S0", SearchConfig()),
            "S2": exp02_search._match_variant("S2", SearchConfig(reuse_results=True), "S1_E1_N"),
        }
        self.base = common.normalize_variant(self.variants["S2"])
        summary = {"status": "selected", "execution_tier": "formal", "code_hash": self.code_hash,
                   "manifest_hash": self.manifest["manifest_hash"], "S_diag": "S2", "S_star": "S2",
                   "S_alt": "S0", "variants": self.variants}
        summary["development"] = {
            "S2": self.match_summary(self.base, self.development_rows, "formal", "wins"),
            "S0": self.match_summary(self.variants["S0"], self.development_rows, "formal", "ties"),
        }
        self.write_search(summary)

    @property
    def development_rows(self):
        return [row for row in self.rows if row["split"] in ("D0", "D1")]

    def write_search(self, summary):
        common.atomic_json(self.search, summary)
        for role, filename in (("S_star", "selected_search.json"), ("S_alt", "alternate_search.json")):
            path = self.search.with_name(filename)
            if summary[role] is None:
                path.unlink(missing_ok=True)
            else:
                common.atomic_json(path, summary["variants"][summary[role]])

    def match_summary(self, variant, rows, tier, result):
        plan = [f"{variant['variant_id']}__{row['config_id']}__P{seat}" for row in rows for seat in (1, 2)]
        score, outcome = {"wins": (1.0, "W"), "ties": (0.5, "D"), "losses": (0.0, "L"),
                          "pending": (1.0, "W"), "failure": (1.0, "W")}[result]
        records = [{"game_id": game_id, "config_id": row["config_id"], "candidate_seat": seat,
                    "score": score, "outcome": outcome, "status": "adjudicated",
                    "technical_failure": result == "failure", "opponent_technical_failure": False,
                    "candidate_budget_failure": False}
                   for game_id, (row, seat) in zip(plan, ((row, seat) for row in rows for seat in (1, 2)))]
        if result == "pending":
            records.pop()
        summary = runner.summarize_games(records, plan)
        summary.update(request_hash=common.digest(plan), variant_id=variant["variant_id"],
                       execution_tier=tier, code_hash=self.code_hash,
                       formal_ranking_allowed=tier == "formal" and summary["eligible"])
        return summary

    def run_controlled(self, d0_scores, confirmation_scores=None):
        calls = []
        confirmation_scores = confirmation_scores or {}

        def run_pairs(variant, rows, output, *, opponent, execution_tier, reuse_paths):
            stage = output.parent.name
            self.assertEqual(variant, common.normalize_variant(variant))
            self.assertEqual(execution_tier, "formal")
            self.assertEqual(opponent, {"kind": "ab", "budget": 1.5})
            self.assertEqual(variant["step_budget"], 1.6)
            self.assertEqual(rows, self.development_rows if stage == "development" else
                             [row for row in self.rows if row["split"] == "D0"])
            name = variant["variant_id"]
            scores = confirmation_scores if stage == "development" else d0_scores
            summary = self.match_summary(variant, rows, execution_tier, scores.get(name, "losses"))
            log = output / "games.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            for row in rows:
                for seat in (1, 2):
                    common.append_jsonl(log, {"config_id": row["config_id"], "candidate_seat": seat})
            calls.append((stage, variant, summary, tuple(reuse_paths)))
            return summary

        with patch.object(experiment, "run_pairs", side_effect=run_pairs):
            result = experiment.run(self.config, self.output)
        return result, calls

    def test_e0_d0_leader_still_screens_eight_candidates_and_confirmation_reranks(self):
        upstream = self.search.parent / "development" / "S2" / "games.jsonl"
        extra = self.root / "earlier.jsonl"
        for path in (upstream, extra):
            common.append_jsonl(path, {"fixture": True})
        self.config["reuse_paths"] = [str(extra)]
        result, calls = self.run_controlled(
            {"S2_E0_N": "wins", "S2_L2_Q0_N": "ties"},
            {"S2_E1_N": "wins", "S2_E0_N": "ties"},
        )
        d0_calls = [call for call in calls if call[0] == "d0"]
        self.assertEqual([variant["variant_id"] for _, variant, _, _ in d0_calls], [
            "S2_E0_N", "S2_E1_N", "S2_L2_Q0_N", "S2_L5_Q0_N", "S2_L10_Q0_N",
            "S2_L2_Q1_N", "S2_L2_Q2_N", "S2_L2_Q3_N",
        ])
        self.assertEqual(sum(summary["planned"] for _, _, summary, _ in d0_calls), 192)
        self.assertTrue(all(set(paths) == {upstream, extra} for _, _, _, paths in d0_calls))
        self.assertEqual(result["nonzero_lambdas"], [2, 1])
        self.assertEqual(result["best_q"], "Q0")
        self.assertEqual(result["interaction_reason"], "best_q_is_q0")
        self.assertEqual(set(result["stages"]["interaction"]), {"S2_E1_N", "S2_L2_Q0_N"})
        self.assertEqual(result["new_candidate"], "S2_L2_Q0_N")
        self.assertEqual([call[1]["variant_id"] for call in calls[8:]],
                         ["S2_E1_N", "S2_E0_N", "S2_L2_Q0_N"])
        for _, variant, match, paths in calls[8:]:
            name = variant["variant_id"]
            self.assertEqual(match["planned"], 48)
            self.assertEqual(set(paths), {upstream, extra, self.output / "d0" / name / "games.jsonl"})
            self.assertEqual(Path(result["development_games"][name]), self.output / "development" / name / "games.jsonl")
        self.assertEqual(result["variants"]["S2_E1_N"], self.base)
        self.assertEqual(result["variants"]["S2_E1_N"]["parent_variant"], "S1_E1_N")
        self.assertEqual((result["E_star"], result["E_alt"]), ("S2_E1_N", "S2_E0_N"))
        self.assertEqual(result["status"], "selected")
        self.assertTrue(result["formal_selection"])
        self.assertEqual(result["source_search_variant"], self.base)
        self.assertEqual(result["effective_search_variant"], self.base)
        self.assertEqual(result["code_hash"], self.code_hash)
        self.assertEqual(result["manifest_hash"], self.manifest["manifest_hash"])
        self.assertEqual(result["search_selection_hash"], common.file_hash(self.search))
        self.assertEqual(common.load_json(self.output / "evaluation_selection.json"), result)
        self.assertEqual(common.load_json(self.output / "selected_evaluation.json"), self.base)
        self.assertEqual(common.load_json(self.output / "alternate_evaluation.json"), result["variants"]["S2_E0_N"])

    def test_q2_leader_adds_one_interaction_using_the_original_search_parent(self):
        result, calls = self.run_controlled({"S2_E1_N": "ties", "S2_L2_Q0_N": "ties",
                                              "S2_L2_Q2_N": "wins",
                                              "S2_L1_Q2_N": "wins"})
        self.assertEqual(result["nonzero_lambdas"], [1, 2])
        self.assertEqual(result["best_q"], "Q2")
        d0 = [variant for stage, variant, _, _ in calls if stage == "d0"]
        self.assertEqual(len(d0), 9)
        self.assertEqual(d0[-1]["variant_id"], "S2_L2_Q2_N")
        self.assertEqual(d0[-1]["parent_variant"], self.base["variant_id"])
        self.assertEqual(evaluation_parameters(d0[-1]["evaluation"]), (2, (1, 80, 100)))
        self.assertEqual(result["interaction_reason"], "added_second_lambda_best_q")
        self.assertEqual(result["new_candidate"], "S2_L1_Q2_N")

    def test_pending_d0_and_confirmation_never_publish_roles(self):
        for name, location, stage in (("S2_E1_N", "d0", "lambda"),
                                      ("S2_L1_Q2_N", "d0", "q"),
                                      ("S2_L2_Q2_N", "d0", "interaction"),
                                      ("S2_E1_N", "development", "confirmation")):
            with self.subTest(stage=stage):
                self.output = self.root / stage
                for filename in ("selected_evaluation.json", "alternate_evaluation.json"):
                    common.atomic_json(self.output / filename, self.base)
                d0 = {"S2_L1_Q2_N": "wins"}
                confirmation = {"S2_E0_N": "wins"}
                if location == "d0":
                    d0[name] = "pending"
                else:
                    confirmation[name] = "pending"
                result, calls = self.run_controlled(d0, confirmation)
                self.assertEqual(result["status"], "pending")
                self.assertEqual(result["pending_stage"], stage)
                self.assertFalse(result["formal_selection"])
                for role, filename in (("E_star", "selected_evaluation.json"),
                                       ("E_alt", "alternate_evaluation.json")):
                    self.assertIsNone(result[role])
                    self.assertFalse((self.output / filename).exists())
                if stage != "confirmation":
                    self.assertFalse(any(call[0] == "development" for call in calls))

    def test_no_eligible_new_d0_candidate_blocks_nonzero_screen(self):
        result, calls = self.run_controlled({f"S2_L{strength}_Q0_N": "failure" for strength in (2, 5, 10)})
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "fewer_than_two_eligible_nonzero_lambdas")
        self.assertIsNone(result["new_candidate"])
        self.assertIsNone(result["E_star"])

    def test_technically_failed_confirmation_cannot_outrank_single_eligible_candidate(self):
        result, _ = self.run_controlled({}, {"S2_E1_N": "failure", "S2_L2_Q0_N": "failure"})
        self.assertEqual((result["E_star"], result["E_alt"]), ("S2_E0_N", "S2_E0_N"))
        self.assertEqual(result["E_alt_reason"], "no_eligible_development_alternative")
        self.assertEqual(common.load_json(self.output / "alternate_evaluation.json"),
                         common.load_json(self.output / "selected_evaluation.json"))

    def test_non_interaction_q_winner_can_be_the_best_new_candidate(self):
        result, _ = self.run_controlled({"S2_L1_Q1_N": "wins", "S2_L1_Q2_N": "wins"})
        self.assertEqual(result["best_q"], "Q1")
        self.assertEqual(result["new_candidate"], "S2_L1_Q1_N")
        self.assertEqual((result["E_star"], result["E_alt"]), ("S2_E1_N", "S2_E0_N"))

    def test_evaluation_aliases_preserve_e1_object_and_make_e0_canonical(self):
        self.assertEqual(development.evaluation_variant(self.base, {"evaluation_id": "alias", "lambda": 1,
                                                                    "q": [1, 20, 100]}), self.base)
        changed = development.evaluation_variant(self.base, {"lambda": 0, "q": [20, 50, 100]})
        self.assertEqual(changed["variant_id"], "S2_E0_N")
        self.assertEqual(changed["parent_variant"], "S2_E1_N")
        self.assertEqual(changed["evaluation"], {"evaluation_id": "E0"})
        self.assertEqual(development.evaluation_variant(changed, {"evaluation_id": "E0"}), changed)
        custom = development.evaluation_variant(self.base, {"lambda": 2, "q": [1, 50, 100]})
        self.assertEqual(custom["evaluation"], {"evaluation_id": "L2_Q1", "lambda": 2, "q": [1, 50, 100]})
        self.assertEqual(development.evaluation_variant(custom, dict(custom["evaluation"], evaluation_id="another")),
                         custom)

    def test_upstream_provenance_and_roles_are_checked_before_touching_output(self):
        original = self.run_controlled({})[0]
        existing = self.output / "evaluation_selection.json"
        saved = existing.read_bytes()
        selection = common.load_json(self.search)
        for changed in (dict(selection, status="smoke_only", execution_tier="smoke", S_star=None),
                        dict(selection, status="awaiting_development_confirmation", S_star=None),
                        dict(selection, code_hash="old"), dict(selection, manifest_hash="other"),
                        dict(selection, development={"S2": dict(selection["development"]["S2"], code_hash="old")}),
                        dict(selection, development={"S2": dict(selection["development"]["S2"], complete=False)})):
            with self.subTest(change=changed):
                self.write_search(changed)
                self.assert_rejected(experiment, self.root / "new")
                self.assert_rejected(experiment, self.output)
        self.write_search(selection)
        common.atomic_json(self.search.with_name("selected_search.json"), self.variants["S0"])
        self.assert_rejected(experiment, self.output)
        self.write_search(dict(selection, unrelated_valid_metadata="changed upstream bytes"))
        self.assert_rejected(experiment, self.output)
        common.atomic_json(self.config["manifest"], runner.make_manifest(314))
        self.assert_rejected(experiment, self.output)
        self.assertEqual(existing.read_bytes(), saved)
        self.assertEqual(common.load_json(existing), original)

    def test_incomplete_sibling_variant_is_rejected_even_when_defaults_match(self):
        partial = dict(self.variants["S2"])
        partial.pop("evaluation")
        common.atomic_json(self.search.with_name("selected_search.json"), partial)
        self.assert_rejected(experiment, self.root / "new")

    def assert_rejected(self, experiment, output):
        with patch.object(experiment, "run_pairs", side_effect=AssertionError("scheduled games")):
            with self.assertRaises(ValueError):
                experiment.run(self.config, output)
        if output != self.output:
            self.assertFalse(output.exists())

    def test_one_ply_smoke_runs_both_splits_reuses_d0_and_does_no_games_on_exact_restart(self):
        summary = common.load_json(self.search)
        self.write_search(dict(summary, status="smoke_only", execution_tier="smoke", S_star=None, development={}))
        with fixtures.ManifestAndSelectionTests()._opening_pairs() as openings:
            result = experiment.run(self.config, self.output, smoke=True)
            self.assertEqual(len(openings), 22)
            saved_logs = {path: path.read_bytes() for path in self.output.glob("*/*/games.jsonl")}
            self.assertEqual(experiment.run(self.config, self.output, smoke=True), result)
            self.assertEqual(len(openings), 22)
            self.assertEqual({path: path.read_bytes() for path in saved_logs}, saved_logs)
        self.assertEqual(result["status"], "smoke_only")
        self.assertEqual(result["diagnostic_status"], "selected")
        self.assertFalse(result["formal_selection"])
        self.assertEqual(result["source_search_variant"], self.base)
        effective = result["effective_search_variant"]
        self.assertEqual(effective["variant_id"], self.base["variant_id"])
        self.assertEqual(effective["step_budget"], 0.012)
        self.assertEqual(effective["max_depth"], 2)
        self.assertEqual(result["variants"][effective["variant_id"]], effective)
        expected_rows = {split: next(row["config_id"] for row in self.rows if row["split"] == split)
                         for split in ("D0", "D1")}
        for name, path in result["development_games"].items():
            raw = common.read_jsonl(path)
            self.assertEqual({(row["config_id"], row["candidate_seat"]) for row in raw},
                             {(config_id, seat) for config_id in expected_rows.values() for seat in (1, 2)})
            d0 = common.read_jsonl(result["d0_games"][name])
            self.assertEqual({row["candidate_seat"] for row in d0}, {1, 2})
            self.assertEqual([row.get("reused_from_request_hash") for row in raw if row["split"] == "D0"],
                             [d0[0]["request_hash"]] * 2)
            self.assertTrue(all("reused_from_request_hash" not in row for row in raw if row["split"] == "D1"))
        self.assertEqual(common.load_json(self.output / "selected_evaluation.json"), effective)


if __name__ == "__main__":
    unittest.main()
