"""Step 8 table definitions and controlled Step 7 admission. Never launch a participant."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.game_state import initial_state
from agents.search_engine import component_configs
from book_experiments.common import validate_match_evidence
from search_experiments import common, exp07_final, measure, runner
from tests import test_final_experiment
import runners
import validate_submission
from world import World


def record(name, config, seat, score):
    return dict(game_id=f"{name}__{config}__P{seat}", variant_id=name, candidate_seat=seat,
                config_id=config, split="D0", obstacle=[0, 0], removed_size=0, score=score,
                outcome={0: "L", 0.5: "D", 1: "W"}[score], technical_failure=False,
                opponent_technical_failure=False, candidate_budget_failure=False,
                candidate_max_step_s=None, status="adjudicated", turns=[],
                referee=dict(plies=0, moves=[], winner=0, reason="ply_cap", time={}, max_rss_mb={}))


class Reducers(unittest.TestCase):
    def test_pairs_require_both_seats_and_full_roster_but_keep_losses(self):
        from search_experiments import reporting
        config_ids = ["a", "b"]
        groups = {}
        for name, scores in [("old", [0, .5, 1, 1]), ("new", [1, 1, .5, 0])]:
            records = [record(name, config, seat, scores[i*2+seat-1])
                       for i, config in enumerate(config_ids) for seat in (1, 2)]
            records[-1]["technical_failure"] = name == "new"
            groups[name] = reporting.result_summary(name, "ab", config_ids, records, 1)
        result = groups["new"]
        self.assertEqual([result[k] for k in ("W", "D", "L", "score", "eligible", "infrastructure_attempts")],
                         [2, 1, 1, .625, False, 1])
        roles = {"V0": "old", "V1": "new", "V2": "new"}
        rows = reporting.contrasts(roles, groups, config_ids, 20261125)
        self.assertEqual(rows[0]["per_config"], {"a": .75, "b": -.75})
        self.assertEqual((rows[0]["mean"], rows[0]["bootstrap_config_ci95"]), (0, [-.75, .75]))
        self.assertEqual(rows, reporting.contrasts(roles, groups, config_ids, 20261125))
        self.assertEqual(rows[0]["n_configs"], 2)
        self.assertTrue(rows[1]["alias"])
        self.assertEqual(rows[1]["mean"], 0)
        self.assertEqual(rows[2]["status"], "omitted")
        groups["new"] = reporting.result_summary("new", "ab", config_ids, records[:-1], 1)
        self.assertIsNone(groups["new"]["score"])
        partial = reporting.contrasts(roles, groups, config_ids, 20261125)[0]
        self.assertEqual((partial["per_config"], partial["n_configs"]), ({"a": .75}, 1))
        self.assertEqual((partial["status"], partial["complete"], partial["mean"], partial["bootstrap_config_ci95"]),
                         ("pending", False, None, None))
        for bad in ([*records, records[0]], [dict(records[0], config_id="elsewhere")]):
            with self.assertRaises(ValueError):
                reporting.result_summary("new", "ab", config_ids, bad, 0)

    def test_turn_metrics_keep_bypasses_missingness_and_referee_time(self):
        from search_experiments import reporting
        game = record("v", "a", 1, 0)
        no_query = dict(status="completed", stop_reason="depth_limit", completed_depth=0,
                        book_queried=False, book_matched=False, book_action="none", search_elapsed_s=7)
        payloads = [no_query, dict(no_query, status="book", stop_reason="book_direct", completed_depth=None,
                                   book_queried=True, book_matched=True, book_action="direct"),
                    dict(no_query, completed_depth=3, book_queried=True, book_matched=True, book_action="hint"),
                    dict(no_query, book_queried=True, book_matched=True),
                    dict(no_query, book_queried=True), None, {"status":"completed"}, {"completed_depth":None},
                    dict(no_query, stop_reason="deadline_before_search"),
                    {"completed_depth":0, "status":"fallback"}, {"book_queried": True}]
        for i, payload in enumerate(payloads):
            move = dict(player=1, src=None, dest=[1, 2], size=1, agent_time=.2)
            game["turns"].append(dict(state=common.state_record(initial_state()), ply=i*2, player=1,
                                      actor_role="candidate", search=payload, move=move))
            game["referee"]["moves"].append(move)
        game["candidate_max_step_s"] = .4
        turns = reporting.turn_rows(game, "ab")
        observed = reporting.live_metrics([game["game_id"]], turns)
        self.assertEqual([observed[k] for k in ("candidate_calls", "query_observed", "queries",
                         "match_observed_queries", "matches", "direct_returns", "hints")], [11, 7, 5, 4, 3, 1, 1])
        self.assertEqual(observed["hit_rate"], .75)
        self.assertEqual([t["depth_status"] for t in turns[5:10]],
                         ["missing_payload", "missing_field", "null", "not_applicable", "unknown_search_entry"])
        self.assertEqual([turns[i]["completed_depth"] for i in (0, 1, 8, 9)], [0, None, None, None])
        self.assertIsNone(turns[6]["stop_reason"])
        self.assertEqual(turns[8]["raw_completed_depth"], 0)
        self.assertEqual([t["step_s"] for t in turns], [.2]*11)
        self.assertEqual(reporting.game_row(game, "ab", turns)["candidate_max_step_s"], .4)
        missing = reporting.live_metrics([], [])
        self.assertIsNone(missing["queries"])
        self.assertIsNone(missing["direct_returns"])
        self.assertIsNone(reporting.turn_metrics(turns[5:8])["depth_zero"])
        self.assertIsNone(reporting.live_metrics(["game"], [turns[0]])["hit_rate"])
        game.update(candidate_seat=2)
        game["referee"].update(winner=2, reason="init_failed", time={"2":{"init":0.0,"max":0.0}})
        self.assertIsNone(reporting.game_row(game, "ab", [])["init_s"])

    def test_full_pre_turn_repetition_and_applied_actions(self):
        from search_experiments import reporting
        game = record("v", "a", 1, .5)
        state = json.loads(common.canonical_json(common.state_record(initial_state())))
        state["board"][1] = [1, 2, 0]
        moves = [dict(src=None, dest=[1,1], size=2), dict(src=[0,1], dest=[1,1], size=2),
                 {"pass":True}, None]
        states = [deepcopy(state) for _ in range(8)]
        states[1]["board"][1][0] = 2
        states[2]["reserves"][1][0] += 1
        states[3]["reserves"][2][0] += 1
        states[4]["player"] = 2
        states[5]["allow_moves"] = False
        for i, s in enumerate(states):
            s["ply"] = i
            move = moves[i] if i < 4 else moves[0]
            actor = s["player"]
            if move is not None:
                move = dict(move, player=actor)
                game["referee"]["moves"].append(move)
            game["turns"].append(dict(state=s, ply=i, player=actor, actor_role="candidate" if actor==1 else "opponent",
                                      move=move))
        turns = reporting.turn_rows(game, "ab")
        self.assertEqual(sum(t["repeated_position"] for t in turns), 2)
        self.assertEqual(sum(t["repeated_action"] is True for t in turns), 3)
        self.assertEqual([t["action_type"] for t in turns[:4]], ["placement","relocation","pass","failed"])
        game["referee"].update(winner=0, reason="ply_cap", plies=100)
        self.assertTrue(reporting.game_row(game, "ab", turns)["draw_100_ply"])
        game["referee"].update(winner=1, reason="line")
        self.assertFalse(reporting.game_row(game, "ab", turns)["draw_100_ply"])


class DiagnosticTests(unittest.TestCase):
    def test_only_enabled_independent_diagnostics_and_read_only_missing_directory(self):
        from search_experiments import exp08_report
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            absent = directory / "absent"
            rows, source = exp08_report.load_diagnostics(absent, "smoke")
            self.assertEqual((rows, source["status"], absent.exists()), ([], "missing_directory", False))
            positions = [{"state": common.state_record(initial_state()), "position_id": "p",
                          "obstacle_class": "corner", "split":"D0"}]
            records, _ = measure.measure(positions, {"C1":component_configs()["C1"]}, [0],
                                         {"evaluation_id":"E1"}, directory, {}, workloads=["cold"],
                                         online_budget=0, execution_tier="smoke")
            self.assertEqual({r.get("stop_reason") for r in records}, {"fixed_depth","deadline_before_search"})
            diagnostic = records[-1]
            del diagnostic["statistics"]["nodes"]
            records[0]["statistics"]["nodes"] = 999
            log = directory / "measurements.jsonl"
            def save(items):
                log.write_text("".join(common.canonical_json(r)+"\n" for r in items))
            save(records)
            rows, source = exp08_report.load_diagnostics(directory, "smoke")
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["nodes"])
            self.assertEqual(rows[0]["cutoffs"], 0)
            _, proven = exp08_report.load_diagnostics(directory, "smoke", positions)
            self.assertEqual((proven["complete"], proven["expected_cases"]), (True, 1))
            save(records[:-1])
            _, incomplete = exp08_report.load_diagnostics(directory, "smoke", positions)
            self.assertEqual(incomplete["missing_case_ids"], [diagnostic["case_id"]])
            save(records)
            with self.assertRaises(ValueError):
                exp08_report.load_diagnostics(directory, "formal")
            self.assertIsNone(source["complete"])
            self.assertEqual(rows[0]["measurements_hash"], common.file_hash(log))
            for bad in ([*records, diagnostic], [dict(diagnostic, instrumented=False)],
                        [dict(diagnostic, case_id="stale")], [dict(diagnostic, statistics={"enabled":False})]):
                save(bad)
                with self.assertRaises(ValueError):
                    exp08_report.load_diagnostics(directory, "smoke")


class ControlledReport(unittest.TestCase):
    def setUp(self):
        guards = self.enterContext(ExitStack())
        def forbidden(*args, **kwargs):
            raise AssertionError("real participant or official dynamic validation")
        play = World.play
        def capped(world):
            self.assertTrue(1 <= world.max_plies <= 4)
            self.assertTrue(all(isinstance(r, SimpleNamespace) for r in (world.runner_1, world.runner_2)))
            return play(world)
        for target, field in ((runners.SubprocessRunner,"start"), (runners.InProcessRunner,"start"),
                              (validate_submission,"dynamic_phase"), (exp07_final,"dynamic_phase")):
            guards.enter_context(patch.object(target, field, forbidden))
        guards.enter_context(patch.object(World,"play", capped))
        self.fixture = test_final_experiment.FinalTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def produce(self, fault=None, empty=True):
        f = self.fixture
        f.freeze(aliases=True, empty=empty)
        with f.controlled_final(random_candidate_fault=fault):
            return exp07_final.run(f.config, f.output, smoke=True)

    def report(self, smoke=True, plots=False, diagnostic_dirs=()):
        from search_experiments import exp08_report
        f = self.fixture
        with patch.object(runner, "run_pairs", side_effect=AssertionError("report replayed games")):
            return exp08_report.run({"final_summary":str(f.output/"final_summary.json"), "plots":plots,
                                     "diagnostic_dirs":diagnostic_dirs},
                                    f.root/"report", smoke=smoke)

    def test_diagnostic_source_identity_across_directories(self):
        self.produce()
        root = self.fixture.root
        original, distinct = root / "original", root / "distinct"
        positions = [dict(state=common.state_record(initial_state()), position_id="p",
                          obstacle_class="corner", split="D0")]
        for directory in (original, distinct):
            measure.measure(positions, {"C1": component_configs()["C1"]}, [0], {"evaluation_id": "E1"},
                            directory, {}, workloads=["cold"], execution_tier="smoke")
        missing = [root / "absent1", root / "absent2"]
        result = self.report(diagnostic_dirs=[original, distinct, *missing])
        first, second, *absent = result["diagnostic_sources"]
        self.assertEqual(first["request_hash"], second["request_hash"])
        self.assertNotEqual(first["measurements_hash"], second["measurements_hash"])
        self.assertEqual([s["status"] for s in absent], ["missing_directory", "missing_directory"])
        csv_path = Path(result["files"]["diagnostics.csv"])
        with csv_path.open() as handle:
            exported = [row["measurements_hash"] for row in csv.DictReader(handle)]
        self.assertEqual(exported, [first["measurements_hash"], second["measurements_hash"]])
        copied = copytree(original, root / "copied")
        with self.assertRaisesRegex(ValueError, "duplicate diagnostic"):
            self.report(diagnostic_dirs=[original, copied])
        self.assertEqual(common.file_hash(csv_path), result["file_hashes"]["diagnostics.csv"])

    def test_complete_smoke_and_checkpoint_admission(self):
        summary = self.produce()
        f = self.fixture
        result = self.report(plots=True)
        self.assertEqual((result["status"], result["execution_tier"], result["formal_results"]), ("complete","smoke",False))
        self.assertEqual(len(result["results"]), 3)
        self.assertTrue(any(row["dimension"] == "stage/reserves_total/actor_role" for row in result["turn_strata"]))
        self.assertTrue(all(r["adjudicated"] == 4 for r in result["results"]))
        self.assertEqual(result["coverage"]["d1_replay"]["reason"], "no_common_book")
        for name in ("results.csv","paired_differences.csv","game_strata.csv","turn_strata.csv","turns.csv",
                     "diagnostics.csv","book_metrics.csv","report_summary.json","scores.png","differences.png"):
            self.assertTrue((f.root/"report"/name).is_file(), name)
        with self.assertRaises(ValueError):
            self.report(smoke=False)
        path = f.output / "final_summary.json"
        edited = deepcopy(summary)
        name = f.online["variant_id"]
        del edited["matches"][name]["greedy"]
        common.atomic_json(path, dict(edited, status="pending"))
        pending = self.report()
        group = next(r for r in pending["results"] if r["opponent"] == "greedy")
        self.assertEqual((group["status"], group["planned"], group["score"]), ("pending",4,None))
        for mutation in ("missing_complete","wrong_hash","unscheduled","official"):
            bad = deepcopy(edited if mutation == "missing_complete" else summary)
            if mutation == "wrong_hash":
                bad["matches"][name]["ab"]["games_hash"] = "stale"
            elif mutation == "unscheduled":
                bad["matches"][name]["bogus"] = bad["matches"][name]["ab"]
            elif mutation == "official":
                bad["validation"][name]["official_path"] = str(path)
            common.atomic_json(path, bad)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.report()
        common.atomic_json(path, summary)
        self.assertEqual(self.report()["status"], "complete")
        # An internally rehashed envelope still must agree with the exact manifest rows.
        plan_path = f.output/"final_plan.json"
        frozen = common.load_json(plan_path)
        for changed in (lambda r: r[1].update(split="T"), lambda r: r.__setitem__(1, r[0])):
            plan = deepcopy(frozen)
            changed(plan["request"]["final_rows"])
            plan["request_hash"] = common.digest(plan["request"])
            common.atomic_json(plan_path, plan)
            common.atomic_json(path, dict(summary, request_hash=plan["request_hash"]))
            with self.assertRaisesRegex(ValueError, "final roster"):
                self.report()
        common.atomic_json(plan_path, frozen)
        common.atomic_json(path, summary)
        artifact = f.output/"matches"/name/"ab"/"candidate"/"student_agent.py"
        artifact.write_text("different")
        with self.assertRaises(ValueError):
            self.report()

    def test_pending_contrast_exports_observed_configurations_only(self):
        final = self.produce()
        f = self.fixture
        match = final["matches"][f.online["variant_id"]]["ab"]
        path = Path(match["games_path"])
        records = common.read_jsonl(path)
        f.write_games(path, records[:-1])
        stats = runner.summarize_games(records[:-1], [r["game_id"] for r in records])
        common.atomic_json(match["summary_path"], dict(match["summary"], **stats, formal_ranking_allowed=False))
        _, match = validate_match_evidence(path, f.online, f.d0 + f.d1, f.opp, "smoke", require_eligible=False)
        final["matches"][f.online["variant_id"]]["ab"] = match
        common.atomic_json(f.output/"final_summary.json", dict(final, status="pending"))
        report = self.report()
        with Path(report["files"]["paired_differences.csv"]).open(newline="") as handle:
            contrast = [r for r in csv.DictReader(handle) if r["new_role"] == "V1"]
        self.assertEqual([(r["level"], r["config_id"], r["difference"]) for r in contrast],
                         [("contrast", "N/A", "N/A"), ("configuration", f.d0[0]["config_id"], "0.0")])
        self.assertTrue(all(r["bootstrap_config_ci95"] == "N/A" for r in contrast))

    def test_published_official_failure_must_match_raw_verdict_and_reason(self):
        f = self.fixture
        f.freeze(aliases=True, empty=True)
        def failed(report):
            report.dynamic.append(dict(test_final_experiment.dynamic_result(), strikes={"soft_timeout": 1}))
        with f.controlled_final(dynamic=failed):
            final = exp07_final.run(f.config, f.output, smoke=True)
        self.assertEqual((final["status"], self.report()["status"]), ("blocked", "blocked"))
        gate = final["validation"][f.online["variant_id"]]
        official = Path(gate["official_path"])
        saved = common.load_json(official)
        passing = dict(saved["report"], dynamic=[test_final_experiment.dynamic_result()])
        summary = f.output/"final_summary.json"
        for raw, status, reason in ((passing, "blocked", saved["reason"]),
                                    (saved["report"], "impossible", saved["reason"]),
                                    (saved["report"], "blocked", "unrecorded failure")):
            common.atomic_json(official, dict(saved, report=raw, status=status, reason=reason))
            gate.update(official_hash=common.file_hash(official), official_status=status)
            common.atomic_json(summary, final)
            with self.subTest(status=status, reason=reason), self.assertRaisesRegex(ValueError, "official.*verdict"):
                self.report()
        # Interrupted raw gate output can exist before the producer publishes a descriptor.
        common.atomic_json(official, dict(saved, status="pending", reason="official dynamic validation interrupted"))
        common.atomic_json(summary, dict(final, status="pending", validation={}))
        self.assertEqual(self.report()["status"], "pending")

    def test_technical_loss_and_retry_do_not_disappear(self):
        self.produce("infrastructure_error")
        self.assertEqual(self.report()["status"], "pending")
        f = self.fixture
        with f.controlled_final(random_candidate_fault="too_many_strikes"):
            exp07_final.run(f.config, f.output, smoke=True)
        blocked = self.report()
        group = next(r for r in blocked["results"] if r["opponent"] == "random")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual([group[k] for k in ("eligible","L","score","infrastructure_attempts","adjudicated")],
                         [False,4,0.0,4,4])

    def test_active_d1_replay_is_separate_from_missing_live_telemetry(self):
        opening = next(r for r in self.fixture.rows if r["split"] == "D1")
        with patch.object(test_final_experiment, "initial_state", return_value=initial_state(opening["obstacle"], opening["removed_size"])):
            self.produce(empty=False)
        report = self.report()
        replay = report["coverage"]["d1_replay"]
        self.assertEqual(replay["status"], "available")
        self.assertGreater(replay["matched"], 0)
        self.assertTrue(all(m["query_observed"] == 0 for m in report["coverage"]["live"]))
