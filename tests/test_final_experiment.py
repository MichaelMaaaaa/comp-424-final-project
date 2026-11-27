"""Step 7 regressions using real admissions and controlled, one-ply pair records."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.game_state import initial_state, legal_moves
from agents.symmetry import canonicalize, decode_book_state
from book_experiments import common as book_common
from book_experiments.books import book_variant, make_book, save_book
from helpers import get_valid_moves
from runners import AgentLoadError, StepResult
from search_experiments import common, exp07_final as final, runner
from search_experiments.development import effective_search_variant, evaluation_variant
from tests import test_book_handoff


def dynamic_result():
    seats = [(0, 1), (0, 2), (1, 1), (1, 2)]
    return {"rules": {"allow_moves": True, "win_length": 4},
            "games": [dict(seed=seed, seat=seat, outcome="D", reason="ply_cap", plies=1,
                           strikes=0, log_bytes=0) for seed, seat in seats],
            "W": 0, "D": 4, "L": 0, "max_step": 1.0, "mean_step": 0.25, "strikes": {},
            "notes": {}, "forfeits": 0, "passes": 0, "init_time": 1.0, "max_rss_mb": 70.0,
            "load_error": None, "examples": {}}


class FinalTests(test_book_handoff.HandoffFixture):
    def setUp(self):
        super().setUp()
        self.config.update(search_selection=self.root / "search_selection.json",
                           recheck_selection=self.root / "recheck_selection.json",
                           final_config=self.root / "final_config.json")
        self.output = self.root / "final"

    def freeze(self, *, smoke=True, mode="N", empty=False, aliases=False, switched=True):
        self.prepare(smoke)
        baseline = common.normalize_variant({"variant_id": "S0_E1_N"})
        alternative = common.normalize_variant({"variant_id": "S1_E1_N", "parent_variant": "S0_E1_N",
                                                 "search": {} if aliases else {"ordering": False}})
        search = {"execution_tier": self.tier, "status": "smoke_only" if smoke else "selected",
                  "code_hash": self.code_hash, "manifest_hash": self.manifest_hash,
                  "variants": {"S0": baseline, "S1": alternative}, "S_star": "S1", "S_diag": "S1", "S_alt": "S0",
                  "development": {"S1": {"complete": True, "eligible": True, "planned": 48, "adjudicated": 48,
                                         "execution_tier": self.tier, "formal_ranking_allowed": not smoke,
                                         "code_hash": self.code_hash, "variant_id": "S1_E1_N"}}}
        common.atomic_json(self.config["search_selection"], search)
        common.atomic_json(self.root / "selected_search.json", alternative)
        common.atomic_json(self.root / "alternate_search.json", baseline)
        effective = self.config["smoke"] | {"step_budget": 0.012, "max_depth": 2} if smoke else {}
        self.v0 = effective_search_variant(baseline, effective)
        current = effective_search_variant(alternative, effective)
        self.v1 = self.v0 if switched else current
        if aliases:
            self.online = self.v1
        else:
            self.online = evaluation_variant(self.v1, {"evaluation_id": "E0"})
        self.wrapper["variant"] = self.online
        common.atomic_json(self.config["online_config"], self.wrapper)
        self.reference = self.root / "recheck_d1" / "games.jsonl"
        self.config["d1_reference_games"] = self.reference
        with self.opening_games():
            d1_summary = runner.run_pairs(self.online, self.d1, self.reference.parent,
                                          opponent=self.opp, execution_tier=self.tier)
        _, reference = book_common.reference_games(self.reference, self.online, self.config, smoke)
        recheck = {**{key: value for key, value in self.wrapper.items() if key != "variant"},
                   "search_selection_hash": common.file_hash(self.config["search_selection"]),
                   "chosen_variant": self.online, "source_search_variant": alternative,
                   "alternate_search_variant": baseline,
                   "effective_search_variants": {"S_star": current, "S_alt": self.v0},
                   "search_switched": switched,
                   "roles": {"S_star": {"E1": current["variant_id"]}, "S_alt": {"E1": self.v0["variant_id"]}},
                   "variants": {v["variant_id"]: v for v in (current, self.v0, self.online)},
                   "d1_reference": {"summary": d1_summary, "games_path": str(self.reference),
                                    "plan_path": str(self.reference.with_name("plan.json"))}}
        if smoke:
            recheck["diagnostic_status"] = "selected"
        common.atomic_json(self.config["recheck_selection"], recheck)
        identity = book_common.handoff_identity(self.config, self.online, self.tier, self.opp, self.manifest_hash)
        step5 = dict(identity, status="no_common_book" if empty else "selected", formal_selection=not smoke,
                     K=0 if empty else 1, selected_book=None, variant_D=None, d1_reference=reference,
                     reference_games=str(self.reference))
        frozen = dict(identity, status="N_only" if empty else "selected", reason="no_common_book" if empty else None,
                     formal_selection=not smoke, selected_book_path=str(self.config["selected_book"]),
                     selected_book_status=step5["status"], K=step5["K"], mode=mode, variant=self.online,
                     book_file=None, book_hash=None, book_comparison_variant=None, book_comparison_file=None,
                     book_comparison_hash=None, development_evidence={}, d1_reference=reference)
        self.v3 = None
        if not empty:
            state = initial_state()
            key, _ = canonicalize(state)
            book = make_book([(key, legal_moves(decode_book_state(key))[0])], f"{self.label}_b0_k1")
            self.book_file = self.root / "book.json"
            save_book(book, self.book_file)
            _, desc = book_common.selected_book_descriptor(self.book_file, self.online, "B0", 1, 1)
            step5.update(desc, selected_book=desc, source_tie=False, second_source_book_file=None, second_source_book=None)
            self.v3 = common.normalize_variant(book_variant(self.online, book, "D" if mode == "N" else mode))
            if mode == "N":
                frozen.update(book_comparison_variant=self.v3, book_comparison_file=str(self.book_file),
                             book_comparison_hash=book["layout"]["hash"])
            else:
                frozen.update(variant=self.v3, book_file=str(self.book_file), book_hash=book["layout"]["hash"])
            with self.opening_games():
                for variant in (self.online, self.v3):
                    directory = self.root / "dev" / variant["variant_id"]
                    runner.run_pairs(variant, self.d0 + self.d1, directory, opponent=self.opp,
                                     execution_tier=self.tier)
                    _, desc = book_common.validate_match_evidence(directory / "games.jsonl", variant,
                                                                 self.d0 + self.d1, self.opp, self.tier)
                    frozen["development_evidence"][variant["variant_id"]] = desc
        common.atomic_json(self.config["selected_book"], step5)
        frozen["selected_book_hash"] = common.file_hash(self.config["selected_book"])
        common.atomic_json(self.config["final_config"], frozen)

    def change(self, config_key, **changes):
        path = self.config[config_key]
        value = common.load_json(path)
        common.atomic_json(path, dict(value, **changes))

    @contextmanager
    def controlled_final(self, *, dynamic=None, pair=None, rss=70.0, random_candidate_fault=None):
        real_pairs = runner.run_pairs
        calls = []

        def play(variant, rows, output, **options):
            calls.append((variant, rows, Path(output), options))
            return (pair or real_pairs)(variant, rows, output, **options)

        def official(report, games, rules, in_process):
            self.assertEqual((games, rules, in_process), (4, {"allow_moves": True, "win_length": 4}, False))
            if dynamic:
                dynamic(report)
            else:
                report.dynamic.append(dynamic_result())

        def fake_runner(spec, **kwargs):
            fault = random_candidate_fault if Path(spec).parts[-4:-1] == (
                self.online["variant_id"], "random", "candidate") else None

            step_count = 0

            def start(seat, rules):
                if fault == "init_failed":
                    raise AgentLoadError("controlled init failure")

            def step(snapshot, player, opponent, hard_limit):
                nonlocal step_count
                step_count += 1
                if fault == "infrastructure_error":
                    raise RuntimeError("controlled runner transport failure")
                if fault == "crash":
                    return StepResult("crash", None, 0.0, 0.0, "controlled process exit")
                if fault == "too_many_strikes":
                    elapsed = 0.002 if step_count == 1 else 0.03
                    return StepResult("error", None, elapsed, elapsed, "controlled agent exception")
                elapsed = 0.03 if fault == "over_budget" else 0.002
                return StepResult("ok", get_valid_moves(*snapshot, player)[0], elapsed, elapsed, None)

            return SimpleNamespace(name=Path(spec).parent.name, start=start, step=step,
                                   stop=lambda: None, stats=lambda: {"init_time": 0.01, "max_rss_mb": rss})

        plies = 4 if random_candidate_fault == "too_many_strikes" else 2 if random_candidate_fault else 1
        with self.opening_games(plies, max_strikes=1), \
                patch.object(runner, "SubprocessRunner", side_effect=fake_runner), \
                patch.object(final, "dynamic_phase", side_effect=official), \
                patch.object(final, "run_pairs", side_effect=play), \
                patch.object(final, "require_formal_runtime"):
            yield calls

    def test_formal_uses_exact_roles_and_t_roster_for_frozen_submission(self):
        self.freeze(smoke=False, mode="H")
        with self.controlled_final() as calls:
            result = final.run(self.config, self.output)
        self.assertEqual(result["status"], "complete")
        plan = common.load_json(self.output / "final_plan.json")
        self.assertEqual(plan["request"]["roles"], {"V0": self.v0, "V1": self.v1,
                                                    "V2": self.online, "V3": self.v3})
        held_out = [row for row in self.rows if row["split"] == "T"]
        self.assertEqual(len(held_out), 24)
        expected = [(self.online, "ab"), (self.v0, "ab"), (self.v3, "ab"),
                    (self.v3, "random"), (self.v3, "greedy")]
        preflight, held_out_calls = calls[:3], calls[3:]
        self.assertEqual([options["allow_test"] for _, _, _, options in preflight], [False] * 3)
        self.assertTrue(all([r["setup_seed"] for r in rows] == [0, 1] for _, rows, _, _ in preflight))
        self.assertEqual([(v, options["opponent"]["kind"]) for v, _, _, options in held_out_calls], expected)
        self.assertEqual(result["submission"]["variant_id"], self.v3["variant_id"])
        for variant, rows, directory, options in held_out_calls:
            self.assertEqual(common.canonical_json(rows), common.canonical_json(held_out))
            self.assertTrue(options["strict"] and options["allow_test"])
            records = common.read_jsonl(directory / "games.jsonl")
            self.assertEqual(len(records), 48)
            self.assertEqual([r["opponent_rng_seed"] for r in records],
                             [r["opponent_rng_seed"] ^ (seat * 0x9E3779B1) for r in held_out for seat in (1, 2)])
            self.assertEqual({r["candidate_artifact_hash"] for r in records},
                             {options["expected_candidate_artifact_hash"]})
        submission = result["submission"]
        self.assertEqual(common.file_hash(submission["path"]), submission["artifact_hash"])
        self.assertEqual(Path(submission["path"]).name, "student_agent.py")
        self.assertEqual(submission["artifact_hash"],
                         plan["request"]["artifacts"][self.v3["variant_id"]]["artifact_hash"])

    def test_smoke_only_development_and_aliases_preserve_v2(self):
        self.freeze(aliases=True, empty=True, switched=False)
        with self.controlled_final() as calls:
            result = final.run(self.config, self.output, smoke=True)
        plan = common.load_json(self.output / "final_plan.json")["request"]
        self.assertEqual(plan["role_candidates"], {role: self.online["variant_id"] for role in ("V0", "V1", "V2")})
        self.assertEqual(plan["candidates"], [self.online])
        self.assertEqual(plan["roles"]["V0"]["variant_id"], "S0_E1_N")
        self.assertEqual(plan["roles"]["V2"]["variant_id"], "S1_E1_N")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["formal_selection"])
        self.assertEqual(result["execution_tier"], "smoke")
        self.assertEqual(len(calls), 4)
        for variant, rows, _, options in calls:
            self.assertEqual(variant, self.online)
            self.assertEqual((variant["step_budget"], variant["max_depth"]), (0.012, 2))
            self.assertEqual([r["split"] for r in rows], ["D0", "D1"])
            self.assertFalse(options["allow_test"])
            if options["opponent"]["kind"] == "ab":
                self.assertEqual(options["opponent"]["budget"], 0.006)
        self.assertEqual(common.canonical_json([rows for _, rows, _, _ in calls[1:]]),
                         common.canonical_json([self.d0 + self.d1] * 3))

    def test_omitted_smoke_roster_uses_one_row_per_development_split(self):
        self.freeze(aliases=True, empty=True)
        config = {k: v for k, v in self.config.items() if k != "smoke"}
        request = final._admit_inputs(config, smoke=True)
        self.assertEqual([(r["split"], r["config_id"]) for r in request["final_rows"]],
                         [("D0", "r2c2_S"), ("D1", "r0c2_S")])
        self.assertEqual(request["smoke_settings"]["max_configs"], 1)
        for invalid in (None, False, 0, 12):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                final._admit_inputs(dict(config, smoke={"max_configs": invalid}), smoke=True)

    def test_failed_initial_publication_invalidates_an_earlier_complete_result(self):
        self.freeze(aliases=True, empty=True)
        path = self.output / "final_summary.json"
        for error, status in ((KeyboardInterrupt, "pending"), (OSError, "blocked")):
            with self.subTest(error=error), self.controlled_final():
                self.assertEqual(final.run(self.config, self.output, smoke=True)["status"], "complete")
                with test_book_handoff.fail_first_replace(path, error), self.assertRaises(error):
                    final.run(self.config, self.output, smoke=True)
                summary = common.load_json(path)
                self.assertEqual((summary["status"], summary["formal_selection"]), (status, False))
                self.assertIsNone(summary["submission"])

    def test_final_plan_and_summary_explain_empty_book_omission(self):
        self.freeze(aliases=True, empty=True)
        with self.controlled_final():
            summary = final.run(self.config, self.output, smoke=True)
        plan = common.load_json(self.output / "final_plan.json")["request"]
        for payload in (plan, summary):
            self.assertEqual(payload.get("v3_omission"), {"reason": "no_common_book", "K": 0})

    def test_linked_empty_decision_rejects_contradictory_selection_fields(self):
        self.freeze(empty=True)
        for change in ({"book_file": "nonempty.json"}, {"k_star": 1}, {"source_tie": True}):
            path = self.config["selected_book"]
            original = path.read_bytes()
            with self.subTest(change=change):
                self.change("selected_book", **change)
                self.change("final_config", selected_book_hash=common.file_hash(path))
                with self.assertRaises(ValueError):
                    final._admit_inputs(self.config, True)
                path.write_bytes(original)

    def test_final_rejects_swapped_or_unfingerprinted_second_export(self):
        self.freeze(mode="H")
        key, _ = canonicalize(initial_state())
        second = self.root / "second.json"
        common.atomic_json(second, make_book([(key, legal_moves(decode_book_state(key))[0])], f"{self.label}_b1_k1"))
        _, descriptor = book_common.selected_book_descriptor(second, self.online, "B1", 1, 1)
        self.change("selected_book", source_tie=True, second_source="B1", second_source_book_file=str(second),
                    second_source_book=descriptor)
        self.change("final_config", selected_book_hash=common.file_hash(self.config["selected_book"]))
        self.assertIn("V3", final._admit_inputs(self.config, True)["roles"])
        original = second.read_bytes()
        self.change_recommendation(second)
        with self.assertRaises(ValueError):
            final._admit_inputs(self.config, True)
        second.write_bytes(original)
        step5 = common.load_json(self.config["selected_book"])
        del step5["second_source_book"]
        common.atomic_json(self.config["selected_book"], step5)
        self.change("final_config", selected_book_hash=common.file_hash(self.config["selected_book"]))
        with self.assertRaises(ValueError):
            final._admit_inputs(self.config, True)

    def test_final_resume_rejects_inconsistent_saved_results_for_each_opponent(self):
        self.freeze(aliases=True, empty=True)
        with self.controlled_final():
            result = final.run(self.config, self.output, smoke=True)
        matches = result["matches"][self.online["variant_id"]]
        for kind, changes in (("ab", {"score": 1.0, "outcome": "W"}),
                              ("random", {"opponent_rng_seed": -99}),
                              ("greedy", {"candidate_artifact_hash": "wrong", "machine": {}})):
            with self.subTest(kind=kind):
                path = Path(matches[kind]["games_path"])
                original = path.read_bytes()
                rows = common.read_jsonl(path)
                self.assertEqual(rows[0]["referee"]["winner"], 0)
                rows[0].update(changes)
                self.write_games(path, rows)
                with self.controlled_final(), self.assertRaises(ValueError):
                    final.run(self.config, self.output, smoke=True)
                self.assertEqual(common.load_json(self.output / "final_summary.json")["status"], "blocked")
                path.write_bytes(original)

        plan_path = Path(matches["ab"]["plan_path"])
        plan = common.load_json(plan_path)
        plan["games"].pop()
        common.atomic_json(plan_path, plan)
        with self.controlled_final(), self.assertRaises(ValueError):
            final.run(self.config, self.output, smoke=True)
        self.assertEqual(common.load_json(plan_path), plan)

    def test_final_resume_requires_persisted_participant_bytes_and_plan(self):
        self.freeze(aliases=True, empty=True)
        with self.controlled_final():
            result = final.run(self.config, self.output, smoke=True)
        match = result["matches"][self.online["variant_id"]]["greedy"]
        row = common.read_jsonl(match["games_path"])[0]
        artifact = Path(match["games_path"]).parent / "opponents" / row["game_id"] / "student_agent.py"
        candidate = Path(match["games_path"]).parent / "candidate" / "student_agent.py"
        for stored in (artifact, candidate):
            with self.subTest(path=stored.name, directory=stored.parent.name):
                original = stored.read_bytes()
                stored.write_bytes(original + b"# changed participant bytes\n")
                with self.controlled_final(), self.assertRaises(ValueError):
                    final.run(self.config, self.output, smoke=True)
                stored.write_bytes(original)
        Path(match["plan_path"]).unlink()
        with self.controlled_final(), self.assertRaises(ValueError):
            final.run(self.config, self.output, smoke=True)

    def test_final_admits_genuine_technical_losses_and_pending_retryable_attempts(self):
        self.freeze(aliases=True, empty=True)
        for fault in ("init_failed", "crash", "infrastructure_error"):
            with self.subTest(fault=fault):
                directory = self.output / fault
                with self.controlled_final(random_candidate_fault=fault):
                    result = final.run(self.config, directory, smoke=True)
                match = result["matches"][self.online["variant_id"]]["random"]
                rows = common.read_jsonl(match["games_path"])
                self.assertEqual(len(rows), 4)
                self.assertFalse(match["summary"]["eligible"])
                if fault == "infrastructure_error":
                    self.assertEqual(result["status"], "pending")
                    self.assertTrue(all(r["status"] == fault and r["retryable"] for r in rows))
                    self.assertEqual(match["summary"]["adjudicated"], 0)
                else:
                    self.assertEqual(result["status"], "blocked")
                    self.assertTrue(all(r["referee"]["reason"] == fault and r["outcome"] == "L" for r in rows))
                    self.assertEqual(match["summary"]["technical_failures"], 4)
                    self.assertEqual(match["summary"]["L"], 4)
                    self.assertEqual(match["summary"]["score"], 0.0)
                    if fault == "crash":
                        self.assertEqual({r["referee"]["plies"] for r in rows}, {0, 1})
                        self.assertTrue(all(r["turns"][-1]["move"] is None for r in rows))
                    else:
                        self.assertTrue(all(r["turns"] == [] for r in rows))
                self.assertEqual(match.get("games_hash"), common.file_hash(match["games_path"]))
                with self.controlled_final():
                    resumed = final.run(self.config, directory, smoke=True)
                self.assertEqual(resumed["status"], "complete" if fault == "infrastructure_error" else "blocked")

    def test_final_counts_measured_candidate_budget_failures_including_a_forfeited_step(self):
        self.freeze(aliases=True, empty=True)
        for fault, technical, lost in (("over_budget", 0, 0), ("too_many_strikes", 4, 4)):
            with self.subTest(fault=fault), self.controlled_final(random_candidate_fault=fault):
                result = final.run(self.config, self.output / fault, smoke=True)
                self.assertEqual(result["status"], "blocked")
                match = result["matches"][self.online["variant_id"]]["random"]
                self.assertEqual(match["summary"]["candidate_budget_failures"], 4)
                self.assertEqual(match["summary"]["technical_failures"], technical)
                self.assertEqual(match["summary"]["L"], lost)
                for row in common.read_jsonl(match["games_path"]):
                    self.assertEqual(row["candidate_max_step_s"], 0.03)
                    if fault == "too_many_strikes":
                        self.assertEqual(row["referee"]["time"][str(row["candidate_seat"])]["max"], 0.002)
                        self.assertEqual(row["turns"][-1]["failure_reason"], fault)

    def test_n_submission_has_a_full_development_book_control(self):
        self.freeze(switched=False)
        with self.controlled_final() as calls:
            result = final.run(self.config, self.output, smoke=True)
        self.assertEqual(result["submission"]["variant_id"], self.online["variant_id"])
        plan = common.load_json(self.output / "final_plan.json")["request"]
        self.assertEqual(plan["roles"]["V1"], self.v1)
        self.assertEqual(plan["roles"]["V3"], self.v3)
        self.assertIn((self.v3, "ab"), [(v, opts["opponent"]["kind"]) for v, _, directory, opts in calls
                                       if "matches" in directory.parts])

    def test_unfrozen_and_wrong_provenance_are_rejected_before_play(self):
        self.freeze(empty=True)
        cases = [
            ("search_selection", {"execution_tier": "formal"}),
            ("recheck_selection", {"diagnostic_status": "pending"}),
            ("recheck_selection", {"search_selection_hash": "stale"}),
            ("recheck_selection", {"chosen_variant": self.v0}),
            ("final_config", {"status": "blocked"}),
            ("final_config", {"formal_selection": True}),
            ("final_config", {"execution_tier": "formal"}),
            ("final_config", {"code_hash": "stale"}),
            ("final_config", {"book_code_hash": "stale"}),
            ("final_config", {"manifest_hash": "other"}),
            ("final_config", {"machine": {}}),
            ("final_config", {"online_config_hash": "stale"}),
            ("final_config", {"K": False}),
        ]
        for key, changes in cases:
            with self.subTest(changes=changes):
                path = self.config[key]
                saved = path.read_bytes()
                self.change(key, **changes)
                try:
                    with self.controlled_final() as calls, self.assertRaises(ValueError):
                        final.run(self.config, self.output, smoke=True)
                    self.assertEqual(calls, [])
                finally:
                    path.write_bytes(saved)

    def test_book_control_rejects_d1_only_or_corrupted_evidence(self):
        self.freeze()
        frozen = common.load_json(self.config["final_config"])
        evidence = frozen["development_evidence"][self.v3["variant_id"]]
        directory = self.root / "book_d1"
        with self.opening_games():
            runner.run_pairs(self.v3, self.d1, directory, opponent=self.opp, execution_tier=self.tier)
        _, d1 = book_common.validate_match_evidence(directory / "games.jsonl", self.v3, self.d1,
                                                     self.opp, self.tier)
        self.change("final_config", development_evidence=dict(frozen["development_evidence"],
                                                              **{self.v3["variant_id"]: d1}))
        with self.controlled_final() as calls, self.assertRaisesRegex(ValueError, "invalid match evidence"):
            final.run(self.config, self.output, smoke=True)
        self.assertFalse(calls)
        self.change("final_config", **frozen)
        games = Path(evidence["games_path"])
        games.write_text(games.read_text() + "invalid\n")
        with self.controlled_final() as calls, self.assertRaisesRegex(ValueError, "invalid match evidence"):
            final.run(self.config, self.output, smoke=True)
        self.assertFalse(calls)

    def test_book_variant_and_export_must_match_the_linked_selection(self):
        self.freeze(mode="H")
        original = self.book_file.read_bytes()
        invalid = common.load_json(self.book_file)
        invalid["layout"]["version"] = "wrong"
        common.atomic_json(self.book_file, invalid)
        with self.controlled_final() as calls, self.assertRaisesRegex(ValueError, "invalid exported book"):
            final.run(self.config, self.output, smoke=True)
        self.assertFalse(calls)
        self.book_file.write_bytes(original)
        self.change("final_config", variant=common.normalize_variant(dict(self.v3, max_depth=3)))
        with self.controlled_final() as calls, self.assertRaises(ValueError):
            final.run(self.config, self.output, smoke=True)
        self.assertFalse(calls)

    def test_official_pass_is_blocked_for_strikes_and_excess_resources(self):
        self.freeze(smoke=False, empty=True)
        for change in ({"strikes": {"bad_move": 1}}, {"forfeits": 1}, {"max_step": 2.1},
                       {"init_time": 31}, {"max_rss_mb": 500}):
            with self.subTest(change=change):
                def weak_pass(report):
                    report.add("WARN", "an official PASS may still have warnings")
                    report.dynamic.append(dict(dynamic_result(), **change))

                with self.controlled_final(dynamic=weak_pass) as calls:
                    result = final.run(self.config, self.output / next(iter(change)))
                self.assertEqual(result["status"], "blocked")
                self.assertFalse(result["formal_selection"])
                self.assertFalse(any(options["allow_test"] for _, _, _, options in calls))

    def test_resume_preserves_completed_official_verdicts_and_recovers_missing_text(self):
        self.freeze(aliases=True, empty=True)
        for expected in ("blocked", "passed"):
            with self.subTest(expected=expected):
                directory = self.output / expected

                def verdict(report):
                    result = dynamic_result()
                    if expected == "blocked":
                        result["strikes"] = {"soft_timeout": 1}
                        result["games"][0]["strikes"] = 1
                        result["max_step"] = 2.1
                        report.add("WARN", "candidate incurred a soft timeout")
                    report.dynamic.append(result)

                with self.controlled_final(dynamic=verdict):
                    first = final.run(self.config, directory, smoke=True)
                path = directory / "validation" / self.online["variant_id"] / "official.json"
                self.assertEqual(common.load_json(path)["status"], expected)
                readable = path.with_suffix(".txt")
                original, text = path.read_bytes(), readable.read_text()
                for remove_text in (False, True):
                    with self.subTest(remove_text=remove_text):
                        if remove_text:
                            readable.unlink()
                        with self.controlled_final(dynamic=lambda report: self.fail("replayed completed verdict")) as calls:
                            resumed = final.run(self.config, directory, smoke=True)
                        self.assertEqual((resumed["status"], resumed["reason"]), (first["status"], first["reason"]))
                        self.assertEqual(path.read_bytes(), original)
                        self.assertEqual(readable.read_text(), text)
                        if expected == "blocked":
                            self.assertEqual(calls, [])

    def test_completed_aggregate_survives_an_exception_before_validator_findings(self):
        self.freeze(aliases=True, empty=True)
        for error, fault in ((KeyboardInterrupt, "strike"), (OSError, "strike"), (OSError, "load")):
            with self.subTest(error=error, fault=fault):
                directory = self.output / (error.__name__ + fault)

                def interrupted_after_games(report):
                    result = dynamic_result()
                    if fault == "load":
                        result.update(load_error="candidate failed to load", games=[], D=0)
                    else:
                        result["strikes"] = {"soft_timeout": 1}
                        result["games"][0]["strikes"] = 1
                    report.dynamic.append(result)
                    raise error("after aggregate, before final findings")

                with self.controlled_final(dynamic=interrupted_after_games), self.assertRaises(error):
                    final.run(self.config, directory, smoke=True)
                path = directory / "validation" / self.online["variant_id"] / "official.json"
                original_report = common.load_json(path)["report"]
                with self.controlled_final(dynamic=lambda report: self.fail("replayed a finished batch")) as calls:
                    resumed = final.run(self.config, directory, smoke=True)
                saved = common.load_json(path)
                self.assertEqual((saved["status"], resumed["status"]), ("blocked", "blocked"))
                self.assertEqual(saved["reason"], "official validator reported a load failure, strike or forfeit")
                self.assertEqual(saved["report"], original_report)
                self.assertEqual(calls, [])

    def test_unfinished_official_batch_retries_after_interrupt_or_infrastructure_error(self):
        self.freeze(aliases=True, empty=True)
        for error, status, partial in ((KeyboardInterrupt, "pending", False), (OSError, "blocked", False),
                                       (OSError, "blocked", True)):
            with self.subTest(error=error, partial=partial):
                directory = self.output / (error.__name__ + str(partial))

                def unfinished(report):
                    report.add("WARN", "incomplete batch")
                    if partial:
                        result = dynamic_result()
                        result.update(games=result["games"][:2], D=2)
                        report.dynamic.append(result)
                    raise error("simulated interruption")

                with self.controlled_final(dynamic=unfinished), self.assertRaises(error):
                    final.run(self.config, directory, smoke=True)
                path = directory / "validation" / self.online["variant_id"] / "official.json"
                checkpoint = common.load_json(path)
                self.assertEqual(checkpoint["status"], status)
                self.assertEqual(len(checkpoint["report"]["dynamic"]), int(partial))
                with self.controlled_final():
                    resumed = final.run(self.config, directory, smoke=True)
                self.assertEqual(resumed["status"], "complete")
                self.assertEqual(common.load_json(path)["status"], "passed")

    def test_pending_validator_and_failed_preflight_block_all_t_matches(self):
        self.freeze(smoke=False, empty=True, switched=False)

        def incomplete_last(report):
            if Path(report.path).parent.name != self.v1["variant_id"]:
                report.dynamic.append(dynamic_result())

        with self.controlled_final(dynamic=incomplete_last) as calls:
            result = final.run(self.config, self.output)
        self.assertEqual(result["status"], "pending")
        self.assertEqual([variant for variant, _, _, _ in calls], [self.online, self.v0])
        self.assertFalse(any(options["allow_test"] for _, _, _, options in calls))
        with self.controlled_final(rss=500.0) as calls:
            result = final.run(self.config, self.output)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(calls)
        self.assertFalse(any(options["allow_test"] for _, _, _, options in calls))

    def test_altered_submission_blocks_resume_and_preserves_exact_plan(self):
        self.freeze(smoke=False, empty=True)
        with self.controlled_final():
            result = final.run(self.config, self.output)
        plan = self.output / "final_plan.json"
        saved_plan = plan.read_bytes()
        submission = Path(result["submission"]["path"])
        submission.write_bytes(submission.read_bytes() + b"# altered\n")
        with self.controlled_final() as calls, self.assertRaisesRegex(ValueError, "artifact"):
            final.run(self.config, self.output)
        self.assertFalse(any(options["allow_test"] for _, _, _, options in calls))
        self.assertEqual(common.load_json(self.output / "final_summary.json")["status"], "blocked")
        self.assertEqual(plan.read_bytes(), saved_plan)
        self.assertTrue(submission.read_bytes().endswith(b"# altered\n"))

    def test_changed_input_cannot_overwrite_a_bound_final_request(self):
        self.freeze(empty=True)
        with self.controlled_final():
            final.run(self.config, self.output, smoke=True)
        plan = self.output / "final_plan.json"
        summary = self.output / "final_summary.json"
        saved = plan.read_bytes(), summary.read_bytes()
        self.change("final_config", unused="changed exact upstream bytes")
        with self.controlled_final() as calls, self.assertRaisesRegex(ValueError, "new output directory"):
            final.run(self.config, self.output, smoke=True)
        self.assertEqual(calls, [])
        self.assertEqual((plan.read_bytes(), summary.read_bytes()), saved)
