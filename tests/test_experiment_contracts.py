import copy
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from functools import partial
from io import StringIO
from itertools import product
from pathlib import Path
import os
import py_compile
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import agents
from agents.experiment_agent import ExperimentAgent
from agents.evaluation import evaluate_e0
from agents.game_state import GameState, apply_move, initial_state, legal_moves, to_arrays
from agents.iterative_deepening import search
from agents.search_engine import SearchConfig
from helpers import Move, get_valid_moves, new_game
from runners import AgentLoadError, StepResult, SubprocessRunner, load_agent_class
from search_experiments import build, common, exp00_prepare, exp01_components, exp02_search, measure, runner
from world import GameRecord


def fixed(position, name, depth, workload, repetition):
    return dict(case_id=f"{position}-{name}-{depth}-{workload}-{repetition}", position_id=position,
                variant_id=name, target_depth=depth, workload=workload, repetition=repetition,
                instrumented=repetition == -1, status="completed", target_reached=True, checked=True,
                correct=True, elapsed=0.1, near_cap=False, solved_remaining=False)


def online(position, name, rep, depth):
    return dict(position_id=position, variant_id=name, workload="online", repetition=rep,
                completed_depth=depth, near_cap=False, solved_remaining=False, budget_ok=True,
                hard_limit_ok=True, instrumented=False)


def referee_fixture(seed, *args, **kwargs):
    _, _, obstacle, removed = new_game(seed)
    game = GameRecord(seed, obstacle, removed, {}, {}, winner=0, reason="ply_cap")
    return SimpleNamespace(obstacle=obstacle, removed=removed, turns=[], step_times={1: [], 2: []}, play=lambda: game)


class EvidenceTests(unittest.TestCase):
    names = ("C0", "C1")
    workloads = ("cold", "iterative")
    depths = (2, 3)
    positions = [{"position_id": "a"}, {"position_id": "b"}]

    def rows(self):
        return [fixed(p, n, d, w, rep) for p, n, d, w, rep in
                product(("a", "b"), self.names, self.depths, self.workloads, range(3))
                if (n, w) != ("C0", "iterative")]

    def summary(self, rows):
        return measure.summarize_measurements(rows, self.names, self.depths, self.positions)

    def test_measurement_resume_requires_the_same_variant_order(self):
        positions = [{"position_id": "opening", "state": common.state_record(initial_state()),
                      "obstacle_class": "corner", "split": "D0"}]
        evaluation = {"evaluation_id": "E1"}
        original = {"A": SearchConfig(), "B": SearchConfig(ordering=False), "C": SearchConfig(tt=False)}
        reversed_order = dict(reversed(list(original.items())))
        expected = [["A", "B", "C"], ["C", "B", "A"], ["B", "C", "A"]]

        def interrupt(path, row):
            common.append_jsonl(path, row)
            raise KeyboardInterrupt("after first timed case")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder/"interrupted"
            references = measure.exact_references(positions, [1], evaluation, folder/"references")

            def run(configs, directory=output):
                return measure.measure(positions, configs, [1], evaluation, directory, references,
                                       workloads=("cold",), diagnostics=False, execution_tier="smoke")

            def saved_files():
                paths = [output/name for name in ("request.json", "measurements.jsonl", "summary.json", "timings.csv")]
                return {path.name: path.read_bytes() for path in paths if path.is_file()}

            def assert_reordered_rejected():
                saved = saved_files()
                with self.assertRaisesRegex(ValueError, "measurement request changed"):
                    run(reversed_order)
                self.assertEqual(saved_files(), saved)

            with patch.object(measure, "append_jsonl", side_effect=interrupt), \
                    self.assertRaisesRegex(KeyboardInterrupt, "after first timed case"):
                run(original)
            log = output/"measurements.jsonl"
            prefix = log.read_bytes()
            self.assertEqual([row["variant_id"] for row in common.read_jsonl(log)], ["A"])
            assert_reordered_rejected()

            actual_calls = []
            fixed_depth = measure.SearchEngine.fixed_depth

            def observe(engine, *args, **kwargs):
                actual_calls.append(engine.config)
                return fixed_depth(engine, *args, **kwargs)

            with patch.object(measure.SearchEngine, "fixed_depth", observe):
                resumed, summary = run(original)
            self.assertTrue(summary["correct"] and summary["verification_complete"])
            self.assertTrue(log.read_bytes().startswith(prefix))
            self.assertEqual(actual_calls, [original[name] for names in expected for name in names][1:])
            self.assertEqual([[row["variant_id"] for row in resumed if row["repetition"] == i] for i in range(3)],
                             expected)
            assert_reordered_rejected()

            fresh, _ = run(reversed_order, folder/"fresh")
            self.assertEqual([row["variant_id"] for row in fresh if row["repetition"] == 0], ["C", "B", "A"])
            self.assertNotEqual(fresh[0]["request_hash"], resumed[0]["request_hash"])

    def test_development_measure_rejects_held_out_positions_before_work(self):
        rows = runner.validate_manifest(runner.make_manifest())
        positions = []
        for split in ("D0", "D1", "T"):
            row = next(r for r in rows if r["split"] == split)
            board, reserves, _, _ = new_game(row["setup_seed"])
            positions.append({"position_id": row["config_id"], "split": split,
                              "obstacle_class": row["obstacle_class"],
                              "state": common.state_record(GameState.from_observation(board, reserves, 1, 0))})
        evaluation = {"evaluation_id": "E1"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            references = measure.exact_references(positions, [1], evaluation, directory/"references")
            self.assertEqual(len(references), 3)
            self.assertTrue(all(row["status"] == "completed" for row in references.values()))

            def run(selected, output, tier):
                return measure.measure(selected, {"S0": SearchConfig()}, [1], evaluation, output, references,
                                       workloads=("cold",), diagnostics=False, execution_tier=tier)

            existing = directory/"development"
            records, summary = run(positions[:2], existing, "smoke")
            self.assertEqual(len(records), 6)
            self.assertTrue(summary["correct"] and summary["evidence_sufficient"])
            saved = {p: p.read_bytes() for p in existing.iterdir()}
            for tier in ("formal", "smoke"):
                fresh = directory/tier
                for output in (existing, fresh):
                    with self.subTest(tier=tier, output=output.name), \
                            patch.object(measure, "resolve_evaluator", wraps=measure.resolve_evaluator) as resolve:
                        with self.assertRaisesRegex(ValueError, "held-out T"):
                            run(positions[2:], output, tier)
                        resolve.assert_not_called()
                        self.assertFalse(fresh.exists())
                        self.assertEqual({p: p.read_bytes() for p in existing.iterdir()}, saved)

    def test_standalone_measure_cli_requires_explicit_smoke_for_smoke_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = directory/"config.json"
            common.atomic_json(config, {"positions": "experiments/a_examples/prepare/positions.json",
                                       "depths": [1], "variants": {"S0": {}}, "online_budget": 0.03})
            for smoke in (False, True):
                with self.subTest(smoke=smoke):
                    output = directory/("smoke" if smoke else "default")
                    command = [sys.executable, "-m", "search_experiments.measure", "--config", str(config),
                               "--output", str(output)]
                    if smoke:
                        command.append("--smoke")
                    result = subprocess.run(command, cwd=common.REPO_ROOT, text=True, capture_output=True)
                    if not smoke:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("complete, eligible formal reference games", result.stderr)
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        timing = output/"measurements"
                        request = common.load_json(timing/"request.json")["request"]
                        records = common.read_jsonl(timing/"measurements.jsonl")
                        self.assertEqual((request["execution_tier"], request["online_budget"]), ("smoke", 0.03))
                        self.assertEqual({row["execution_tier"] for row in records}, {"smoke"})
                        self.assertEqual({row["measurement"] for row in records}, {"timing", "diagnostic"})
                        self.assertTrue(any(row["workload"] == "online" for row in records))

    def test_standalone_measure_formal_cli_checks_provenance_and_online_budget(self):
        positions = [{"position_id": "opening", "state": common.state_record(initial_state()),
                      "obstacle_class": "corner", "split": "D0"}]
        payload = {"positions": positions, "positions_hash": common.digest(positions), "execution_tier": "formal",
                   "reference_complete": True, "reference_eligible": True,
                   "reference_code_hash": common.code_manifest()["code_hash"]}
        cases = (("missing", "complete, eligible formal reference games"),
                 ("ineligible", "complete, eligible formal reference games"),
                 ("stale", "reference game code changed"), ("short_budget", "online budget of 1.6"),
                 ("qualified_fixed_depth", None))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config, source = directory/"config.json", directory/"positions.json"
            for case, error in cases:
                with self.subTest(case=case):
                    candidate = dict(payload)
                    settings = {"positions": str(source), "depths": [1], "variants": {"S0": {}}}
                    if case == "missing":
                        candidate.pop("reference_complete")
                    elif case == "ineligible":
                        candidate["reference_eligible"] = False
                    elif case == "stale":
                        candidate["reference_code_hash"] = "old source"
                    elif case == "short_budget":
                        settings["online_budget"] = 0.03
                    common.atomic_json(source, candidate)
                    common.atomic_json(config, settings)
                    output = directory/case
                    command = [sys.executable, "-m", "search_experiments.measure", "--config", str(config),
                               "--output", str(output)]
                    result = subprocess.run(command, cwd=common.REPO_ROOT, text=True, capture_output=True)
                    if error is not None:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(error, result.stderr)
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        timing = output/"measurements"
                        request = common.load_json(timing/"request.json")["request"]
                        records = common.read_jsonl(timing/"measurements.jsonl")
                        self.assertEqual(request["execution_tier"], "formal")
                        self.assertEqual({row["execution_tier"] for row in records}, {"formal"})
                        self.assertEqual({row["workload"] for row in records}, {"cold", "iterative"})
                        self.assertTrue(common.load_json(timing/"summary.json")["correct"])

    def test_empty_missing_group_and_duplicate_repeat_cannot_pass(self):
        rows = self.rows()
        missing = [r for r in rows if (r["workload"], r["target_depth"], r["repetition"]) !=
                   ("iterative", 3, 2)]
        incomplete = [dict(r, checked=False, correct=None, target_reached=False) for r in rows]
        for evidence in ([], incomplete, missing + missing[-1:]):
            with self.subTest(count=len(evidence)):
                summary = self.summary(evidence)
                self.assertFalse(summary["correct"])
                self.assertTrue(summary["no_mismatches"])
                self.assertFalse(summary["verification_complete"])
                self.assertFalse(summary["evidence_sufficient"])
                self.assertTrue(summary["empty_groups"])

    def test_partial_common_cases_and_wrong_outside_intersection(self):
        rows = [r for r in self.rows() if r["position_id"] == "a"]
        summary = self.summary(rows)
        self.assertTrue(summary["correct"])
        self.assertFalse(summary["verification_complete"])
        self.assertEqual(summary["common_by_workload"]["iterative"], [["a", 2], ["a", 3]])
        refs = {(p, d): {"status": "completed" if p == "a" else "timeout"}
                for p, d in product(("a", "b"), self.depths)}
        def passes():
            return measure.validation_passes(rows, refs, self.positions, self.depths, self.names, self.workloads)
        self.assertTrue(passes())
        refs[("a", 3)]["status"] = "timeout"
        self.assertFalse(passes())
        refs[("a", 3)]["status"] = "completed"
        rows.append(dict(fixed("b", "C1", 2, "iterative", -1), correct=False))
        self.assertFalse(passes())
        self.assertFalse(self.summary(rows)["correct"])

    def test_missing_whole_candidate_workload_and_diagnostics(self):
        rows = self.rows()
        refs = {(p, d): {"status": "completed"} for p, d in product(("a", "b"), self.depths)}
        self.assertTrue(self.summary(rows)["verification_complete"])
        self.assertFalse(measure.summarize_measurements(rows, self.names, self.depths, self.positions,
                                                       diagnostics=True)["verification_complete"])
        self.assertFalse(measure.validation_passes(rows, refs, self.positions, self.depths,
                                                   ("C0", "C1", "C2"), self.workloads))
        cold = [r for r in rows if r["workload"] == "cold"]
        self.assertFalse(measure.validation_passes(cold, refs, self.positions, self.depths,
                                                  self.names, self.workloads))
        self.assertTrue(measure.validation_passes(cold, refs, self.positions, self.depths,
                                                 self.names, ("cold",)))

    def test_performance_uses_same_states_and_checks_both_budgets(self):
        rows = [online(str(rep), name, rep, depth) for rep in range(3)
                for name, depth in (("S0", 1), ("S1", 2))]
        self.assertFalse(measure.compare_performance(rows, "S0", "S1")["retain_candidate"])
        self.assertIsNone(exp02_search._relative_to_s0(rows, "S1")["mean_depth"])
        for r in rows:
            r["position_id"] = "same"
        self.assertTrue(measure.compare_performance(rows, "S0", "S1")["retain_candidate"])
        rows[0]["budget_ok"] = False
        self.assertFalse(measure.compare_performance(rows, "S0", "S1")["budget_valid"])

    def test_tie_break_requires_verified_common_fixed_rows(self):
        rows = [online("p", name, rep, 2) for name, rep in product(("S0", "S1"), range(3))]
        times = [dict(fixed("p", name, 3, "iterative", rep), elapsed=0.1 if name == "S1" else 1)
                 for name, rep in product(("S0", "S1"), range(3))]
        self.assertTrue(measure.compare_performance(rows + times, "S0", "S1")["retain_candidate"])
        times[-1]["correct"] = False
        self.assertFalse(measure.compare_performance(rows + times, "S0", "S1")["retain_candidate"])
        self.assertIsNone(exp02_search._relative_to_s0(rows + times, "S1")["fixed_ratio_to_S0"])
        self.assertFalse(measure.compare_performance(rows + times, "S0", "S1")["consistent"])


    def test_measurement_resume_requires_the_same_exact_answers(self):
        evaluation = {"evaluation_id": "E1"}
        positions = [{"position_id": "p", "state": common.state_record(initial_state()),
                      "obstacle_class": "corner"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            refs = measure.exact_references(positions, [1], evaluation, output/"refs")
            def run(refs, directory):
                return measure.measure(positions, {"S0": SearchConfig()}, [1], evaluation, directory,
                                       refs, workloads=("cold",), diagnostics=False)[1]
            self.assertTrue(run(refs, output/"timing")["correct"])
            cached = common.read_jsonl(output/"timing"/"measurements.jsonl")
            ref = refs["p", 1]
            for change in ({"value": ref["value"] + 1}, {"optimal_moves": []}, {"status": "timeout"}):
                different = {("p", 1): dict(ref, **change)}
                with self.subTest(change=change):
                    self.assertFalse(run(different, output/next(iter(change)))["correct"])
                    with self.assertRaises(ValueError):
                        run(different, output/"timing")
            self.assertFalse(run({("p", 1): {"status": "timeout"}}, output/"no_answer")["correct"])
            # Ignore incidental wall-clock differences, including on-disk Move serialization.
            replayed = measure.exact_references(positions, [1], evaluation, output/"refs")
            replayed["p", 1]["elapsed"] += 1
            self.assertTrue(run(replayed, output/"timing")["correct"])
            self.assertEqual(common.read_jsonl(output/"timing"/"measurements.jsonl"), cached)

    def test_step1_writes_insufficient_evidence_for_an_empty_reference_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            positions = [{"position_id": "a", "state": common.state_record(initial_state()),
                          "split": "D0", "obstacle_class": "corner"}]
            source = output/"positions.json"
            common.atomic_json(source, {"positions": positions, "positions_hash": common.digest(positions)})
            for name, settings in (("no_positions", {"max_positions": 0, "depths": [1]}),
                                   ("no_depths", {"depths": []})):
                with self.subTest(name=name):
                    directory = output/name
                    result = exp01_components.run({"positions": source, "smoke": settings}, directory, smoke=True)
                    self.assertFalse(result["evidence_sufficient"])
                    self.assertFalse(result["correct"])
                    self.assertEqual(common.load_json(directory/"components_summary.json"), result)
                    log = directory/result["exact_reference"]["path"]/"measurements.jsonl"
                    self.assertEqual(log.read_bytes(), b"")
                    self.assertEqual(result["exact_reference"]["measurements_hash"], common.file_hash(log))

    def test_duplicate_performance_cases_cannot_reverse_retention(self):
        names = ("S0", "S1")
        rows = [online(p, n, r, 2) for p, n, r in product(("clean", "duplicate"), names, range(3))]
        for p, n, r in product(("clean", "duplicate"), names, range(3)):
            candidate_time = 1.1 if p == "clean" else 0.1
            rows.append(dict(fixed(p, n, 3, "iterative", r), elapsed=1 if n == "S0" else candidate_time))
        extra = next(r for r in rows if r["position_id"] == "duplicate" and r["workload"] == "iterative")
        for duplicate in (extra, dict(extra, target_reached=False, checked=False, correct=None)):
            with self.subTest(fixed_status=duplicate["target_reached"]):
                compared = measure.compare_performance(rows + [duplicate], *names)
                self.assertFalse(compared["retain_candidate"])
                self.assertEqual(compared["fixed_positions"], ["clean"])
                self.assertEqual(exp02_search._relative_to_s0(rows + [duplicate], "S1")["fixed_ratio_to_S0"], 1.1)
        online_rows = [online(p, n, r, 6 if (p, n) == ("duplicate", "S1") else 2)
                       for p, n, r in product(("clean", "duplicate"), names, range(3))]
        extra = next(r for r in online_rows if r["position_id"] == "duplicate")
        for duplicate in (extra, dict(extra, completed_depth=None)):
            with self.subTest(online_depth=duplicate["completed_depth"]):
                compared = measure.compare_performance(online_rows + [duplicate], *names)
                self.assertFalse(compared["retain_candidate"])
                self.assertEqual(compared["depth_positions"], ["clean"])
                self.assertEqual(exp02_search._relative_to_s0(online_rows + [duplicate], "S1")["mean_depth"], 2)


class VariantTests(unittest.TestCase):
    def _add_book_fixture(self, root, name, source):
        shutil.copytree(common.REPO_ROOT/"agents", root/"agents", ignore=shutil.ignore_patterns("__pycache__"))
        (root/"agents"/f"{name}.py").write_text(source, encoding="utf-8")
        package_path = patch.object(agents, "__path__", [*agents.__path__, str(root/"agents")])
        package_path.start()
        self.addCleanup(package_path.stop)
        module_name = f"agents.{name}"
        self.addCleanup(sys.modules.pop, module_name, None)
        return module_name

    def test_sparse_variant_is_canonical_for_generated_and_measured_provider(self):
        state = initial_state()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"
            provider = self._add_book_fixture(root, "test_sparse_book_provider", '''
def make_book_hook(book_config, variant):
    ordering = variant.get("search", {}).get("ordering", False)
    def lookup(state, moves, deadline):
        return {"matched": True, "direct_move": moves[-1] if ordering else None,
                "metadata": {"variant": variant}}
    return lookup
''')
            variant = {"variant_id": "sparse", "telemetry": False, "max_depth": 0,
                       "book": {"mode": "D", "provider": provider}}
            original = copy.deepcopy(variant)
            path, metadata = build.build_agent(variant, Path(tmp)/"student_agent.py", root=root)
            generated = load_agent_class(str(path))()
            move = generated.step(*to_arrays(state), 1, 2)
            measured = measure.measure_online_step(state, variant)
            self.assertEqual(variant, original)
            self.assertEqual(measured["move"], legal_moves(state)[-1])
            self.assertEqual(measured["move"], move)
            self.assertEqual(measured["status"], generated.last_info["status"])
            self.assertEqual(measured["status"], "book")
            self.assertEqual(measured["book_metadata"], generated.last_info["book_metadata"])
            self.assertEqual(measured["book_metadata"]["variant"], metadata["variant"])
            self.assertTrue(measured["book_metadata"]["variant"]["search"]["ordering"])

    def test_hint_telemetry_tracks_the_first_preferred_root_child(self):
        state = initial_state()
        fallback, preferred = legal_moves(state)[0], legal_moves(state)[-1]
        preferred_child = apply_move(state, preferred)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"
            provider = self._add_book_fixture(root, "test_hint_book_provider", '''
def make_book_hook(book_config, variant):
    def lookup(state, moves, deadline):
        if variant.get("fixture_miss", False):
            return {"matched": False}
        return {"matched": True, "preferred_move": moves[-1]}
    return lookup
''')
            for label, options, expire_on, expected_depth, expected_action in (
                    ("zero_depth", {"max_depth": 0}, None, 0, "none"),
                    ("disabled_ordering", {"search": {"ordering": False}}, None, 1, "none"),
                    ("expires_before_root_child", {}, 1, 0, "none"),
                    ("expires_after_root_child", {}, 2, 0, "hint"),
                    ("expires_after_preferred_first_child", {"search": {"preferred_first": True}}, 2, 0, "hint"),
                    ("completed_search", {}, None, 1, "hint"),
                    ("miss_with_internal_preference", {"fixture_miss": True, "max_depth": 2}, None, 2, "none")):
                with self.subTest(case=label):
                    variant = {"variant_id": label, "max_depth": 1, "telemetry": False,
                               "book": {"mode": "H", "provider": provider}} | options
                    path, _ = build.build_agent(variant, Path(tmp)/label/"student_agent.py", root=root)
                    generated = load_agent_class(str(path))()

                    def check(run):
                        now = 0
                        visited = []

                        def evaluator(child, player):
                            nonlocal now
                            visited.append((child.board, player))
                            if player == expire_on:
                                now = 2
                            return 0

                        with patch("time.perf_counter", side_effect=lambda: now):
                            info = run(evaluator)
                        if expire_on is not None:
                            self.assertEqual(now, 2)
                        if expire_on == 2:
                            self.assertEqual(visited[-1], (preferred_child.board, 2))
                        self.assertTrue(info["book_queried"])
                        self.assertEqual(info["book_matched"], not options.get("fixture_miss", False))
                        self.assertEqual(info["completed_depth"], expected_depth)
                        self.assertEqual(info["book_action"], expected_action)
                        if not expected_depth:
                            self.assertEqual(info["move"], fallback)
                            self.assertEqual(info["status"], "fallback")
                        return info

                    def generated_step(evaluator):
                        generated.engine.evaluator = evaluator
                        move = generated.step(*to_arrays(state), 1, 2)
                        return dict(generated.last_info, move=move)

                    built_info = check(generated_step)
                    measured = check(lambda evaluator: measure.measure_online_step(
                        state, variant, evaluation_factory=lambda _: evaluator))
                    self.assertEqual((built_info["move"], built_info["status"]),
                                     (measured["move"], measured["status"]))

    def test_provider_aliases_keep_real_e1_and_allow_canonical_reexports(self):
        state = initial_state((0, 3))
        for move in (Move(None, (1, 2), 0), Move(None, (1, 2), 1),
                     Move(None, (2, 1), 1), Move(None, (0, 1), 0)):
            state = apply_move(state, move)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"
            provider = self._add_book_fixture(root, "test_safe_book_aliases", '''
from agents.evaluation import evaluate_e0 as line_score
import agents.evaluation as fixture_evaluation
from agents.alpha_beta import SearchConfig as BookSearchConfig
from agents.search_engine import SearchConfig as BookSearchConfig
from agents.move_ordering import move_code as provider_move_code
from agents.search_engine import action_code as provider_move_code

def make_book_hook(book_config, variant):
    def lookup(state, moves, deadline):
        return {"metadata": {"line_score": line_score(state, state.player),
                             "module_score": fixture_evaluation.evaluate_e0(state, state.player),
                             "ordering": BookSearchConfig().ordering,
                             "code": list(provider_move_code(moves[0]))}}
    return lookup
''')
            variant = {"variant_id": "safe_aliases", "max_depth": 1, "telemetry": False,
                       "evaluation": {"evaluation_id": "E1"}, "book": {"mode": "H", "provider": provider}}
            measured = measure.measure_online_step(state, variant)
            path, _ = build.build_agent(variant, Path(tmp)/"student_agent.py", root=root)
            agent = load_agent_class(str(path))()
            agent.call_count = state.ply // 2
            self.assertEqual(agent.step(*to_arrays(state), 1, 2), measured["move"])
            self.assertEqual((agent.last_info["value"], measured["value"]), (3119, 3119))
            self.assertEqual(agent.last_info["book_metadata"], measured["book_metadata"])
            self.assertEqual(agent.last_info["book_metadata"]["line_score"], evaluate_e0(state, 1))

    def test_builder_rejects_import_conflicts_and_unsupported_forms_before_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"source"
            provider = self._add_book_fixture(root, "test_conflicting_book_aliases", "")
            modules = root/"agents"
            (modules/"alias_dependency.py").write_text(
                "from agents.evaluation import evaluate_e0 as provider_score\n", encoding="utf-8")
            (modules/"fixture_package").mkdir()
            destination = Path(tmp)/"student_agent.py"
            variant = {"variant_id": "e1_control", "telemetry": False, "max_depth": 1}
            build.build_agent(variant, destination, root=root)
            metadata = destination.with_suffix(".meta.json")
            saved = destination.read_bytes(), metadata.read_bytes()
            active = dict(variant, book={"mode": "H", "provider": provider})
            for label, imports, error in (
                    ("definition then alias", "from agents.evaluation import evaluate_e0 as evaluate\n", "collision.*evaluate"),
                    ("alias then definition", "from agents.alias_dependency import provider_score as helper_score\n"
                     "def provider_score(state, player):\n    return 0\n", "collision.*provider_score"),
                    ("two aliases", "from agents.alias_dependency import provider_score as helper_score\n"
                     "from agents.evaluation import evaluate as provider_score\n", "collision.*provider_score"),
                    ("external import alias", "from math import sqrt as evaluate\n", "collision.*evaluate"),
                    ("relative import", "from .evaluation import evaluate_e0 as line_score\n", "relative"),
                    ("package from import", "from agents import evaluation\n", "package"),
                    ("package import", "import agents as development_agents\n", "package"),
                    ("nested package import", "from agents.fixture_package import data\n", "package")):
                with self.subTest(case=label):
                    (modules/"test_conflicting_book_aliases.py").write_text(
                        imports + "\ndef make_book_hook(book_config, variant):\n"
                        "    return lambda state, moves, deadline: {}\n", encoding="utf-8")
                    try:
                        with self.assertRaisesRegex(ValueError, error):
                            build.build_agent(active, destination, root=root)
                    except FileNotFoundError as exc:
                        self.fail(f"builder tried to inline a package as a module: {exc}")
                    self.assertEqual((destination.read_bytes(), metadata.read_bytes()), saved)
            agent = load_agent_class(str(destination))()
            state = initial_state()
            self.assertEqual(agent.step(*to_arrays(state), 1, 2), Move(None, (1, 2), 2))

    def test_nested_development_import_cannot_change_published_evaluator(self):
        sources = {
            "conditional": '''
BOOK_DIAGNOSTICS = True
if BOOK_DIAGNOSTICS:
    from agents.evaluation import evaluate_e0 as evaluate

def make_book_hook(book_config, variant):
    return lambda state, moves, deadline: {"metadata": {"line_score": evaluate(state, state.player)}}
''',
            "function": '''
def make_book_hook(book_config, variant):
    import agents.evaluation as scoring
    return lambda state, moves, deadline: {"metadata": {"line_score": scoring.evaluate_e0(state, state.player)}}
''',
        }
        state = initial_state()
        variant = {"variant_id": "e1_control", "max_depth": 1, "telemetry": False}
        with tempfile.TemporaryDirectory() as tmp:
            for scope, source in sources.items():
                with self.subTest(scope=scope):
                    directory = Path(tmp)/scope
                    root = directory/"source"
                    provider = self._add_book_fixture(root, f"test_{scope}_book_import", source)
                    destination, _ = build.build_agent(variant, directory/"student_agent.py", root=root)
                    metadata = destination.with_suffix(".meta.json")
                    saved = destination.read_bytes(), metadata.read_bytes()
                    active = dict(variant, book={"mode": "H", "provider": provider})
                    measured = measure.measure_online_step(state, active)
                    self.assertEqual(measured["value"], 3100)
                    self.assertEqual(measured["book_metadata"]["line_score"], evaluate_e0(state, 1))
                    with self.assertRaisesRegex(ValueError, "direct module-level declaration"):
                        build.build_agent(active, destination, root=root)
                    self.assertEqual((destination.read_bytes(), metadata.read_bytes()), saved)
                    preserved = load_agent_class(str(destination))()
                    self.assertEqual(preserved.step(*to_arrays(state), 1, 2), measured["move"])
                    self.assertEqual(preserved.last_info["value"], 3100)

    def test_rebuild_invalidates_strict_child_cache_with_parent_prefix_override(self):
        board, reserves, _, _ = new_game(0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for case, inherited in (("default", None), ("inherited", str(output/"env_cache"))):
                with self.subTest(case=case), patch.dict(os.environ), \
                        patch.object(sys, "pycache_prefix", str(output/"parent_cache")):
                    if inherited is None:
                        os.environ.pop("PYTHONPYCACHEPREFIX", None)
                    else:
                        os.environ["PYTHONPYCACHEPREFIX"] = inherited
                    path = output/case/"student_agent.py"
                    variant = {"variant_id": "fixture", "max_depth": 1, "telemetry": True}
                    build.build_agent(dict(variant, evaluation={"lambda": 2, "q": [1, 20, 100]}), path)
                    size = path.stat().st_size
                    stamp = 1700000000
                    os.utime(path, (stamp, stamp))
                    # Compile exactly where a spawned child with the inherited environment looks.
                    with patch.object(sys, "pycache_prefix", inherited):
                        cache = Path(py_compile.compile(str(path), doraise=True,
                                     invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
                    unrelated = cache.with_name("unrelated.pyc")
                    unrelated.write_bytes(b"preserved")
                    build.build_agent(dict(variant, evaluation={"lambda": 4, "q": [1, 20, 100]}), path)
                    os.utime(path, (stamp, stamp))
                    self.assertEqual(path.stat().st_size, size)
                    child = SubprocessRunner(str(path), log_dir=str(output/case/"logs"))
                    try:
                        child.start(1, {"allow_moves": True, "win_length": 4})
                        reply = child.step((board, reserves), 1, 2, child.limits["hard_limit"])
                        self.assertEqual(reply.kind, "ok", reply)
                    finally:
                        child.stop()
                    self.assertEqual(runner._telemetry(child.log_path)[0]["value"], 3400)
                    self.assertEqual(unrelated.read_bytes(), b"preserved")

    def test_same_stamp_generated_replacements_execute_the_new_source(self):
        board, reserves, _, _ = new_game(0)
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "fixture", "max_depth": 1, "telemetry": False}
        def candidate(strength):
            return dict(variant, evaluation={"lambda": strength, "q": [1, 20, 100]})
        def value(path):
            agent = load_agent_class(str(path))()
            agent.step(board, reserves, 1, 2)
            return agent.last_info["value"]
        with tempfile.TemporaryDirectory() as tmp:
            for route in ("build", "promote", "opponent"):
                with self.subTest(route=route):
                    directory = Path(tmp)/route
                    path = directory/"candidate/student_agent.py"
                    if route == "opponent":
                        build.build_opponent("random", path, seed=1)
                    else:
                        build.build_agent(candidate(2), path)
                    size = path.stat().st_size
                    stamp = 1700000000
                    os.utime(path, (stamp, stamp))
                    cache = py_compile.compile(str(path), doraise=True,
                                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
                    self.assertTrue(Path(cache).is_file())
                    if route == "opponent":
                        first = load_agent_class(str(path))().step(board, reserves, 1, 2)
                        control = directory/"fresh/student_agent.py"
                        build.build_opponent("random", control, seed=2)
                        second = load_agent_class(str(control))().step(board, reserves, 1, 2)
                        self.assertNotEqual(first, second)
                        build.build_opponent("random", path, seed=2)
                    else:
                        self.assertEqual(value(path), 3200)
                        if route == "build":
                            build.build_agent(candidate(4), path)
                        else:
                            with patch.object(runner, "SnapshotWorld", side_effect=referee_fixture):
                                runner.run_pairs(candidate(4), [row], directory)
                    os.utime(path, (stamp, stamp))
                    self.assertEqual(path.stat().st_size, size)
                    if route == "opponent":
                        self.assertEqual(load_agent_class(str(path))().step(board, reserves, 1, 2), second)
                    else:
                        self.assertEqual(value(path), 3400)

    def test_telemetry_default_matches_behavior_key_across_agent_entry_points(self):
        base = {"variant_id": "telemetry", "step_budget": 0}
        enabled = dict(base, telemetry=True)
        disabled = dict(base, telemetry=False)
        self.assertEqual(common.variant_behavior_key(base), common.variant_behavior_key(enabled))
        self.assertNotEqual(common.variant_behavior_key(base), common.variant_behavior_key(disabled))
        state = initial_state()
        step_args = (*to_arrays(state), state.player, 3 - state.player)
        with tempfile.TemporaryDirectory() as tmp:
            for label, variant, expected in (("omitted", base, True), ("enabled", enabled, True),
                                              ("disabled", disabled, False)):
                path, _ = build.build_agent(variant, Path(tmp)/label/"student_agent.py")
                direct = ExperimentAgent(variant)
                generated = load_agent_class(str(path))()
                for entry, run_step in (("direct", lambda: direct.step(*step_args)),
                                        ("generated", lambda: generated.step(*step_args)),
                                        ("measurement", lambda: measure.measure_online_step(state, variant))):
                    captured = StringIO()
                    with self.subTest(telemetry=label, entry=entry), redirect_stdout(captured):
                        run_step()
                        self.assertEqual(captured.getvalue().startswith("EXP_STEP "), expected)

    def test_original_objects_aliases_and_strict_hash(self):
        for strength, preset in ((0, "E0"), (1, "E1")):
            base = {"variant_id": preset, "evaluation": {"evaluation_id": preset}}
            alias = {"variant_id": "V2", "parent_variant": "E*", "max_depth": None, "search": {},
                     "evaluation": {"evaluation_id": "chosen", "lambda": strength,
                                    "q": [1, 20 if strength else 50, 100]},
                     "book": {"mode": "N", "hash": "inactive", "provider": "unused"}}
            self.assertTrue(hasattr(common, "deduplicate_variants"))
            kept, roles = common.deduplicate_variants([base, alias])
            self.assertEqual(roles, {preset: preset, "V2": preset})
            self.assertEqual(len(kept), 1)
            self.assertIs(kept[0], base)
            self.assertNotEqual(common.normalize_variant(base)["config_hash"],
                                common.normalize_variant(alias)["config_hash"])
            with self.assertRaises(ValueError):
                common.deduplicate_variants([base, dict(base, step_budget=0.8)])

    def test_active_book_dedup_keeps_provider_visible_evaluation_fields(self):
        state = initial_state()
        moves = legal_moves(state)
        base = {"variant_id": "preset", "max_depth": 0, "telemetry": False,
                "evaluation": {"evaluation_id": "E1"}}
        explicit = dict(base, variant_id="explicit", evaluation={"evaluation_id": "E1", "lambda": 1,
                                                               "q": [1, 20, 100]})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            provider = self._add_book_fixture(directory/"source", "test_evaluation_identity_book", '''
def make_book_hook(book_config, variant):
    explicit = "lambda" in variant["evaluation"]
    def lookup(state, moves, deadline):
        action = "direct_move" if book_config["mode"] == "D" else "preferred_move"
        return {"matched": True, action: moves[-1] if explicit else moves[0]}
    return lookup
''')
            for mode in ("D", "H"):
                with self.subTest(mode=mode):
                    book = {"mode": mode, "provider": provider}
                    preset, expanded = dict(base, book=book), dict(explicit, book=book)
                    equivalent = dict(preset, variant_id="equivalent", parent_variant="E*")
                    kept, aliases = common.deduplicate_variants([preset, equivalent, expanded])
                    self.assertEqual(aliases, {"preset": "preset", "equivalent": "preset", "explicit": "explicit"})
                    first, second = kept
                    self.assertIs(first, preset)
                    self.assertIs(second, expanded)
                    with self.assertRaisesRegex(ValueError, "lambda"):
                        common.variant_behavior_key(dict(expanded, evaluation={"lambda": -1, "q": [1, 20, 100]}))

            for variant, expected in ((base, moves[0]), (explicit, moves[-1])):
                active = dict(variant, book={"mode": "D", "provider": provider})
                path, _ = build.build_agent(active, directory/variant["variant_id"]/"student_agent.py",
                                            root=directory/"source")
                generated = load_agent_class(str(path))()
                with patch("time.perf_counter", return_value=0):
                    self.assertEqual(generated.step(*to_arrays(state), 1, 2), expected)
                    measured = measure.measure_online_step(state, active)
                self.assertEqual(measured["move"], expected)
                self.assertEqual(generated.last_info["book_action"], measured["book_action"])
                self.assertEqual(measured["book_action"], "direct")

    def test_equivalent_budgets_depth_ceilings_and_inactive_table_capacity(self):
        for first, second in (({"step_budget": 1}, {"step_budget": 1.0}),
                              ({"step_budget": -0.0}, {"step_budget": 0}),
                              ({}, {"max_depth": 100}), ({"max_depth": None}, {"max_depth": 101}),
                              ({"search": {"tt": False, "tt_capacity": 0}}, {"search": {"tt": False}})):
            a, b = dict(first, variant_id="a"), dict(second, variant_id="b")
            with self.subTest(first=first, second=second):
                kept, roles = common.deduplicate_variants([a, b])
                self.assertEqual(roles, {"a": "a", "b": "a"})
                self.assertIs(kept[0], a)
                self.assertNotEqual(common.normalize_variant(a)["config_hash"], common.normalize_variant(b)["config_hash"])
                active = {"mode": "H", "provider": "agents.books.fixture"}
                self.assertNotEqual(common.variant_behavior_key(dict(a, book=active)),
                                    common.variant_behavior_key(dict(b, book=active)))
        for first, second in (({"step_budget": True}, {"step_budget": 1}),
                              ({"step_budget": "1"}, {"step_budget": 1}),
                              ({"max_depth": 99}, {}),
                              ({"search": {"tt_capacity": 1}}, {})):
            with self.subTest(first=first, second=second):
                self.assertNotEqual(common.variant_behavior_key(dict(first, variant_id="a")),
                                    common.variant_behavior_key(dict(second, variant_id="a")))

    def test_invalid_max_depth_is_rejected_before_build_agent_or_search(self):
        near_cap = replace(initial_state(), ply=98)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"student_agent.py"
            for depth in (2.0, 100.0, True, -1, "2"):
                variant = {"variant_id": "depth", "max_depth": depth}
                for entry, run in (("normalize", lambda: common.normalize_variant(variant)),
                                   ("behavior", lambda: common.variant_behavior_key(variant)),
                                   ("build", lambda: build.build_agent(variant, path)),
                                   ("agent", lambda: ExperimentAgent(variant)),
                                   ("direct_search", lambda: search(near_cap, float("inf"), max_depth=depth))):
                    with self.subTest(depth=depth, entry=entry), self.assertRaisesRegex(ValueError, "max_depth"):
                        run()
            self.assertFalse(path.exists())
        for cap in (None, 0, 1, 100, 101):
            with self.subTest(cap=cap):
                result = search(near_cap, float("inf"), max_depth=cap)
                self.assertEqual(result.completed_depth, cap if cap in (0, 1) else 2)

    def test_distinguish_active_switches_and_reject_unsupported_rules(self):
        base = {"variant_id": "x", "book": {"mode": "D", "provider": "p", "hash": "a", "version": 1}}
        self.assertTrue(hasattr(common, "variant_behavior_key"))
        key = common.variant_behavior_key(base)
        for book in ({"mode": "H"}, {"provider": "q"}, {"hash": "b"}, {"version": 2}):
            self.assertNotEqual(key, common.variant_behavior_key(dict(base, book=dict(base["book"], **book))))
        for change in ({"search": {"preferred_first": True}}, {"step_budget": 1.2}, {"max_depth": 2},
                       {"telemetry": False}, {"future_option": 1}):
            self.assertNotEqual(key, common.variant_behavior_key(dict(base, **change)))
        self.assertEqual(key, common.variant_behavior_key(dict(base, rules={"allow_moves": True})))
        with self.assertRaises(ValueError):
            common.variant_behavior_key(dict(base, rules={"allow_moves": False}))


class ManifestAndSelectionTests(unittest.TestCase):
    @contextmanager
    def _opening_pairs(self):
        starts = []

        class OpeningReferee(runner.SnapshotWorld):
            def __init__(self, seed, *args, **kwargs):
                starts.append(seed)
                super().__init__(seed, *args, max_plies=1, **kwargs)

        def fake_runner(spec, **kwargs):
            def step(snapshot, player, opponent, hard_limit):
                return StepResult("ok", get_valid_moves(*snapshot, player)[0], 0.01, 0.01, None)

            return SimpleNamespace(name=Path(spec).parent.name, start=lambda seat, rules: None,
                                   step=step, stop=lambda: None, stats=dict)

        with patch.object(runner, "SnapshotWorld", OpeningReferee), \
                patch.object(runner, "SubprocessRunner", side_effect=fake_runner):
            yield starts

    def test_pair_admission_rejects_duplicate_configuration_ids_before_building(self):
        rows = runner.validate_manifest(runner.make_manifest())
        d0 = next(r for r in rows if r["split"] == "D0")
        d1 = next(r for r in rows if r["split"] == "D1")
        variant = {"variant_id": "unique_schedule"}
        with tempfile.TemporaryDirectory() as tmp, self._opening_pairs() as starts:
            existing = Path(tmp)/"valid"
            selected = [d1, d0]
            result = runner.run_pairs(variant, selected, existing)
            self.assertEqual((result["planned"], result["adjudicated"], result["complete"]), (4, 4, True))
            scheduled = [(r["config_id"], r["candidate_seat"]) for r in common.read_jsonl(existing/"games.jsonl")]
            self.assertEqual(scheduled, [(r["config_id"], seat) for r in selected for seat in (1, 2)])
            self.assertEqual(runner.run_pairs(variant, selected, existing), result)
            expected_starts = [r["setup_seed"] for r in selected for _ in (1, 2)]
            self.assertEqual(starts, expected_starts)
            saved = {p: p.read_bytes() for p in existing.rglob("*") if p.is_file()}
            for label, duplicate in (("exact", d0), ("conflicting", dict(d1, config_id=d0["config_id"]))):
                fresh = Path(tmp)/label
                for destination in (existing, fresh):
                    with self.subTest(case=label, output=destination.name), \
                            patch.object(runner, "build_agent", wraps=runner.build_agent) as candidate, \
                            patch.object(runner, "build_opponent", wraps=runner.build_opponent) as opponent:
                        with self.assertRaisesRegex(ValueError, "duplicate config_id"):
                            runner.run_pairs(variant, [d0, duplicate], destination)
                        candidate.assert_not_called()
                        opponent.assert_not_called()
                        self.assertEqual(starts, expected_starts)
                        self.assertFalse(fresh.exists())
                        self.assertEqual({p: p.read_bytes() for p in existing.rglob("*") if p.is_file()}, saved)

    def test_pairs_reject_a_different_expected_artifact_before_replacing_candidate(self):
        row = next(row for row in runner.validate_manifest(runner.make_manifest()) if row["split"] == "D0")
        variant = {"variant_id": "validated_bytes"}
        with tempfile.TemporaryDirectory() as tmp, self._opening_pairs() as starts:
            output = Path(tmp) / "pairs"
            runner.run_pairs(variant, [row], output)
            saved = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
            start_count = len(starts)
            with self.assertRaisesRegex(ValueError, "candidate artifact"):
                runner.run_pairs(variant, [row], output, expected_candidate_artifact_hash="0" * 64)
            self.assertEqual(len(starts), start_count)
            self.assertEqual({path: path.read_bytes() for path in output.rglob("*") if path.is_file()}, saved)
            actual = common.file_hash(output / "candidate" / "student_agent.py")
            self.assertTrue(runner.run_pairs(
                variant, [row], output, expected_candidate_artifact_hash=actual)["complete"])
            self.assertEqual(len(starts), start_count)

    def test_pair_admission_checks_opening_before_building_or_reusing_records(self):
        rows = runner.validate_manifest(runner.make_manifest())
        d0 = next(r for r in rows if r["split"] == "D0")
        d1 = next(r for r in rows if r["split"] == "D1" and r["obstacle"] != d0["obstacle"])
        variant = {"variant_id": "opening_identity"}
        with tempfile.TemporaryDirectory() as tmp, self._opening_pairs() as starts:
            directory = Path(tmp)
            source = directory/"source"
            self.assertTrue(runner.run_pairs(variant, [d0], source)["eligible"])
            log = source/"games.jsonl"
            saved = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            for field, changed in (("obstacle", d1["obstacle"]), ("removed_size", (d0["removed_size"] + 1) % 3)):
                wrong = dict(d0, **{field: changed})
                for reuse in ((), (log,)):
                    fresh = directory/f"invalid_{field}_{bool(reuse)}"
                    with self.subTest(field=field, reuse=bool(reuse)), \
                            patch.object(runner, "build_agent", wraps=runner.build_agent) as candidate, \
                            patch.object(runner, "read_jsonl", wraps=runner.read_jsonl) as records:
                        with self.assertRaisesRegex(ValueError, "initial configuration differs"):
                            runner.run_pairs(variant, [wrong], fresh, reuse_paths=reuse)
                        with self.assertRaisesRegex(ValueError, "initial configuration differs"):
                            runner.run_pairs(variant, [wrong], source, reuse_paths=reuse)
                        candidate.assert_not_called()
                        records.assert_not_called()
                        self.assertEqual(starts, [d0["setup_seed"]] * 2)
                        self.assertFalse(fresh.exists())
                        self.assertEqual({p: p.read_bytes() for p in source.rglob("*") if p.is_file()}, saved)

            extended = directory/"extended"
            summary = runner.run_pairs(variant, [d0, d1], extended, reuse_paths=[log])
            self.assertEqual((summary["adjudicated"], summary["eligible"]), (4, True))
            reused = [r for r in common.read_jsonl(extended/"games.jsonl") if "reused_from_request_hash" in r]
            self.assertEqual([(r["config_id"], r["candidate_seat"]) for r in reused],
                             [(d0["config_id"], 1), (d0["config_id"], 2)])
            self.assertEqual(starts, [d0["setup_seed"]] * 2 + [d1["setup_seed"]] * 2)

    def test_formal_pair_admission_enforces_candidate_and_ab_budgets(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "fixed_budgets", "step_budget": 1.6}
        with tempfile.TemporaryDirectory() as tmp, self._opening_pairs() as starts:
            directory = Path(tmp)
            valid = directory/"formal"
            formal = runner.run_pairs(variant, [row], valid, opponent={"kind": "ab", "budget": 1.5})
            self.assertEqual((formal["complete"], formal["eligible"], formal["formal_ranking_allowed"]),
                             (True, True, True))
            saved = {p: p.read_bytes() for p in valid.rglob("*") if p.is_file()}
            for budget, ab_budget, error in ((0.2, 1.5, "candidate.*1.6"), (1.6, 0.01, "AB.*1.5"),
                                             (1.6, -0.01, "AB.*1.5")):
                invalid = directory/f"invalid_{budget}_{ab_budget}"
                for output in (valid, invalid):
                    with self.subTest(candidate=budget, ab=ab_budget, output=output.name), \
                            patch.object(runner, "build_agent", wraps=runner.build_agent) as candidate, \
                            patch.object(runner, "build_opponent", wraps=runner.build_opponent) as opponent:
                        with self.assertRaisesRegex(ValueError, error):
                            runner.run_pairs(dict(variant, step_budget=budget), [row], output,
                                             opponent={"kind": "ab", "budget": ab_budget})
                        candidate.assert_not_called()
                        opponent.assert_not_called()
                        self.assertEqual(starts, [row["setup_seed"]] * 2)
                        self.assertFalse(invalid.exists())
                        self.assertEqual({p: p.read_bytes() for p in valid.rglob("*") if p.is_file()}, saved)

            smoke_path = directory/"smoke"
            smoke = runner.run_pairs(dict(variant, step_budget=0.2), [row], smoke_path, execution_tier="smoke",
                                     opponent={"kind": "ab", "budget": 0.01})
            self.assertEqual((smoke["execution_tier"], smoke["eligible"], smoke["formal_ranking_allowed"]),
                             ("smoke", True, False))
            for record in common.read_jsonl(smoke_path/"games.jsonl"):
                self.assertEqual((record["candidate_step_budget"], record["opponent"]["budget"]), (0.2, 0.01))
            random = runner.run_pairs(variant, [row], directory/"random",
                                      opponent={"kind": "random", "budget": 0.01})
            self.assertTrue(random["formal_ranking_allowed"])

    @contextmanager
    def _short_referee(self, log_path=None):
        trace = SimpleNamespace(fault=None, starts=[], stops=[])
        original_step = runner.SnapshotWorld.step

        def fake_runner(spec, **kwargs):
            candidate = Path(spec).parent.name == "candidate"
            seat = None
            moves = stops = 0

            def start(player, rules):
                nonlocal seat
                seat = player
                if candidate:
                    trace.starts.append(seat)
                    if seat == 1 and trace.fault == "startup":
                        raise OSError("temporary startup error")

            def step(snapshot, player, opponent, hard_limit):
                nonlocal moves
                moves += 1
                if candidate and seat == 1:
                    if trace.fault == "in_play":
                        raise KeyboardInterrupt("cancelled before adjudication")
                    if moves == 2:
                        return StepResult("hard_timeout", None, hard_limit, hard_limit, None)
                elapsed = 1.6000001 if candidate and seat == 1 else 0.01
                return StepResult("ok", get_valid_moves(*snapshot, player)[0], elapsed, elapsed, None)

            def stop():
                nonlocal stops
                stops += 1
                trace.stops.append((candidate, stops))
                if (candidate and seat == 1 and moves == 2
                        and (trace.fault, stops) in (("first_stop", 1), ("second_stop", 2))):
                    raise KeyboardInterrupt("cancelled during runner cleanup")

            def stats():
                return {"log_path": str(log_path) if log_path and candidate and seat == 1 else None}

            return SimpleNamespace(name="candidate" if candidate else "opponent", start=start, step=step,
                                   stop=stop, stats=stats)

        def short_step(world):
            original_step(world)
            if not world.over and world.runner_1.name == "opponent":
                world.finish(0, "ply_cap")

        with patch.object(runner, "SubprocessRunner", side_effect=fake_runner), \
                patch.object(runner.SnapshotWorld, "step", short_step):
            yield trace

    def _assert_saved_referee_loss(self, record):
        self.assertEqual((record["status"], record["outcome"], record["score"]), ("adjudicated", "L", 0.0))
        self.assertTrue(record["technical_failure"])
        self.assertFalse(record["opponent_technical_failure"])
        self.assertEqual((record["referee"]["winner"], record["referee"]["reason"]), (2, "hard_timeout"))
        self.assertEqual(record["candidate_max_step_s"], 1.6000001)
        self.assertTrue(record["candidate_budget_failure"])

    def test_known_referee_loss_survives_telemetry_and_cleanup_failures(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "recovery", "step_budget": 1.6}
        original_read = Path.read_text
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp)/"telemetry.txt"
            telemetry.write_text('EXP_STEP {"ply":0,"status":"completed"}\n', encoding="utf-8")
            for fault, error in (("telemetry_eio", OSError("telemetry EIO")),
                                 ("telemetry_interrupt", KeyboardInterrupt("cancelled during telemetry")),
                                 ("telemetry_exit", SystemExit(13)), ("second_stop", None),
                                 ("first_stop", KeyboardInterrupt("cancelled during runner cleanup"))):
                output = Path(tmp)/fault

                def fail_read(path, *args, **kwargs):
                    if path == telemetry and fault.startswith("telemetry_"):
                        raise error
                    return original_read(path, *args, **kwargs)

                with self.subTest(fault=fault), self._short_referee(telemetry) as trace:
                    trace.fault = fault
                    with patch.object(Path, "read_text", fail_read):
                        if isinstance(error, (KeyboardInterrupt, SystemExit)):
                            with self.assertRaises(type(error)) as cancelled:
                                runner.run_pairs(variant, [row], output)
                            self.assertEqual(cancelled.exception.args, error.args)
                        else:
                            try:
                                first = runner.run_pairs(variant, [row], output)
                            except KeyboardInterrupt:
                                self.fail("runner was stopped again after official cleanup")
                            self.assertEqual((first["adjudicated"], first["L"], first["eligible"]), (2, 1, False))
                    records = common.read_jsonl(output/"games.jsonl")
                    self.assertTrue(records, "cancelled after an official result but no row was saved")
                    first = records[0]
                    self._assert_saved_referee_loss(first)
                    self.assertEqual(first["turns"][0]["actor_role"], "candidate")
                    if fault.startswith("telemetry_"):
                        self.assertIn(str(error), first["telemetry_error"])
                    if fault == "first_stop":
                        self.assertTrue(any(not candidate for candidate, _ in trace.stops))
                        self.assertIn("cancelled during runner cleanup", first["post_play_error"])
                    if fault == "second_stop":
                        self.assertTrue(all(stops == 1 for _, stops in trace.stops))
                        self.assertEqual(first["turns"][0]["search"]["status"], "completed")
                    trace.fault = None
                    resumed = runner.run_pairs(variant, [row], output)
                    self.assertEqual(trace.starts, [1, 2])
                    self.assertEqual((resumed["adjudicated"], resumed["L"], resumed["eligible"]), (2, 1, False))
                    self.assertEqual(common.read_jsonl(output/"games.jsonl")[0], first)

    def test_pending_startup_retries_and_interrupted_durable_loss_is_reused(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "recovery", "step_budget": 1.6}

        def interrupt_after_append(path, record):
            common.append_jsonl(path, record)
            raise KeyboardInterrupt("cancelled after fsync")

        with tempfile.TemporaryDirectory() as tmp, self._short_referee() as trace:
            output = Path(tmp)
            trace.fault = "startup"
            first = runner.run_pairs(variant, [row], output)
            self.assertEqual((first["adjudicated"], first["eligible"]), (1, False))
            self.assertIn((False, 1), trace.stops)
            log = output/"games.jsonl"
            prefix = log.read_bytes()
            pending, completed = common.read_jsonl(log)
            self.assertEqual((pending["status"], pending["retryable"]), ("infrastructure_error", True))
            self.assertEqual(completed["candidate_seat"], 2)

            trace.fault = "in_play"
            with self.assertRaisesRegex(KeyboardInterrupt, "before adjudication"):
                runner.run_pairs(variant, [row], output)
            self.assertEqual(log.read_bytes(), prefix)

            trace.fault = None
            with patch.object(runner, "append_jsonl", interrupt_after_append), \
                    self.assertRaisesRegex(KeyboardInterrupt, "after fsync"):
                runner.run_pairs(variant, [row], output)
            durable = log.read_bytes()
            self.assertTrue(durable.startswith(prefix))
            loss = common.read_jsonl(log)[-1]
            self._assert_saved_referee_loss(loss)
            self.assertEqual(loss["attempt"], 2)
            self.assertEqual(common.load_json(output/"summary.json"), first)
            resumed = runner.run_pairs(variant, [row], output)
            self.assertEqual(trace.starts, [1, 2, 1, 1])
            self.assertEqual(log.read_bytes(), durable)
            self.assertEqual((resumed["adjudicated"], resumed["L"], resumed["eligible"]), (2, 1, False))

    def test_match_resume_requires_the_same_reuse_source_priority(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "reuse_priority", "telemetry": False}
        starts = []
        time_out = False

        class OnePlyWorld(runner.SnapshotWorld):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, max_plies=1, **kwargs)

        def fake_runner(spec, **kwargs):
            def step(snapshot, player, opponent, hard_limit):
                if time_out:
                    return StepResult("hard_timeout", None, hard_limit, hard_limit, None)
                return StepResult("ok", get_valid_moves(*snapshot, player)[0], 0.01, 0.01, None)

            return SimpleNamespace(name=spec, start=lambda seat, rules: starts.append(seat),
                                   step=step, stats=dict, stop=lambda: None)

        def interrupt_after_first(path, record):
            common.append_jsonl(path, record)
            raise KeyboardInterrupt("after first reused seat")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "SubprocessRunner", side_effect=fake_runner), \
                patch.object(runner, "SnapshotWorld", OnePlyWorld):
            directory = Path(tmp)
            first, second = directory/"A", directory/"B"
            a = runner.run_pairs(variant, [row], first)
            time_out = True
            b = runner.run_pairs(variant, [row], second)
            first_log, second_log = first/"games.jsonl", second/"games.jsonl"
            self.assertEqual([r["outcome"] for r in common.read_jsonl(first_log)], ["D", "D"])
            self.assertEqual([r["outcome"] for r in common.read_jsonl(second_log)], ["L", "W"])
            self.assertEqual(a["request_hash"], b["request_hash"])
            starts_at_reuse = list(starts)

            output = directory/"interrupted"
            log = output/"games.jsonl"
            ordered = [first_log, second_log]
            reversed_sources = [second_log, first_log]

            def saved_files():
                return {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}

            def assert_reordered_rejected():
                before = saved_files()
                with self.assertRaisesRegex(ValueError, "use a new output directory"):
                    runner.run_pairs(variant, [row], output, reuse_paths=reversed_sources)
                self.assertEqual(saved_files(), before)

            with patch.object(runner, "append_jsonl", side_effect=interrupt_after_first), \
                    self.assertRaisesRegex(KeyboardInterrupt, "after first reused seat"):
                runner.run_pairs(variant, [row], output, reuse_paths=ordered)
            prefix = log.read_bytes()
            self.assertEqual([r["outcome"] for r in common.read_jsonl(log)], ["D"])
            assert_reordered_rejected()

            resumed = runner.run_pairs(variant, [row], output, reuse_paths=ordered)
            self.assertTrue(log.read_bytes().startswith(prefix))
            self.assertEqual([r["outcome"] for r in common.read_jsonl(log)], ["D", "D"])
            self.assertEqual((resumed["complete"], resumed["eligible"], resumed["score"]), (True, True, 0.5))
            assert_reordered_rejected()

            fresh = directory/"fresh"
            reversed_run = runner.run_pairs(variant, [row], fresh, reuse_paths=reversed_sources)
            self.assertEqual([r["outcome"] for r in common.read_jsonl(fresh/"games.jsonl")], ["L", "W"])
            self.assertFalse(reversed_run["eligible"])
            self.assertNotEqual(resumed["request_hash"], reversed_run["request_hash"])
            self.assertEqual(starts, starts_at_reuse)

    def test_reserves_validator_games_before_random_allocation(self):
        for seed in (10, 123, 20261125):
            manifest = runner.make_manifest(seed)
            rows = runner.validate_manifest(manifest)
            for setup_seed, expected_split, config_id in ((0, "D0", "r3c1_M"), (1, "D1", "r1c3_M")):
                row, = [r for r in rows if r["setup_seed"] == setup_seed]
                self.assertEqual((row["config_id"], row["split"]), (config_id, expected_split))
            self.assertIn("validator_reservations", manifest)
            self.assertEqual(manifest, runner.make_manifest(seed))
            original = copy.deepcopy(manifest)
            for mutation in ("class", "id", "policy", "seed", "swapped_reservation"):
                manifest = copy.deepcopy(original)
                if mutation == "class":
                    first, other = manifest["rows"][0:2]
                    first["obstacle_class"], other["obstacle_class"] = other["obstacle_class"], first["obstacle_class"]
                elif mutation == "id":
                    manifest["rows"][0]["config_id"] = "incorrect"
                elif mutation == "policy":
                    manifest["validator_reservations"] = []
                elif mutation == "seed":
                    manifest["rows"][0]["setup_seed"] = 0
                else:
                    a, b = sorted((r for r in manifest["rows"] if r["setup_seed"] in (0, 1)),
                                  key=lambda r: r["setup_seed"])
                    a["split"], b["split"] = b["split"], a["split"]
                manifest["manifest_hash"] = common.digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
                with self.assertRaises(ValueError):
                    runner.validate_manifest(manifest)

    def test_resume_rejects_nested_provider_artifact_drift_and_preserves_evidence(self):
        rows = runner.validate_manifest(runner.make_manifest())
        d0 = next(r for r in rows if r["split"] == "D0")
        d1 = next(r for r in rows if r["split"] == "D1")
        variant = {"variant_id": "hint", "book": {"mode": "H", "provider": "agents.books.fixture"}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp/"source"
            for relative in common.code_manifest()["files"]:
                destination = root/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.REPO_ROOT/relative, destination)
            provider = root/"agents/books/fixture.py"
            provider.parent.mkdir()
            provider.write_text('''def make_book_hook(book_config, variant):
    def lookup(state, moves, deadline):
        return {"matched": True, "preferred_move": moves[0]}
    return lookup
''')
            with patch.object(common, "REPO_ROOT", root), \
                    patch.object(runner, "build_agent", partial(build.build_agent, root=root)), \
                    patch.object(runner, "SnapshotWorld", side_effect=referee_fixture) as referee:
                code_hash = common.code_manifest()["code_hash"]
                output = tmp/"first"
                original = runner.run_pairs(variant, [d0], output)
                self.assertTrue(original["eligible"])
                saved = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
                log = output/"games.jsonl"
                referee.reset_mock()
                self.assertEqual(runner.run_pairs(variant, [d0], output), original)
                referee.assert_not_called()
                self.assertEqual(log.read_bytes(), saved[log])

                subset = tmp/"subset_reuse"
                self.assertTrue(runner.run_pairs(variant, [d0, d1], subset, reuse_paths=[log])["eligible"])
                reused = [row for row in common.read_jsonl(subset/"games.jsonl") if "reused_from_request_hash" in row]
                self.assertEqual([row["config_id"] for row in reused], [d0["config_id"]] * 2)
                self.assertEqual(referee.call_count, 2)

                disabled = dict(variant, book=dict(variant["book"], mode="N"))
                n_result = runner.run_pairs(disabled, [d0], tmp/"no_book")
                provider.write_text(provider.read_text() + "# source-only change\n")
                _, comment_build = build.build_agent(variant, tmp/"comment"/"student_agent.py", root=root)
                old_meta = common.load_json(output/"candidate/student_agent.meta.json")
                self.assertEqual(comment_build["artifact_hash"], old_meta["artifact_hash"])
                self.assertNotEqual(comment_build["source_files"], old_meta["source_files"])
                referee.reset_mock()
                self.assertEqual(runner.run_pairs(disabled, [d0], tmp/"no_book"), n_result)
                referee.assert_not_called()
                with self.subTest(change="source_only_resume"):
                    with self.assertRaisesRegex(ValueError, "new output directory"):
                        runner.run_pairs(variant, [d0], output)
                    self.assertEqual({path: path.read_bytes() for path in output.rglob("*") if path.is_file()}, saved)
                cross = tmp/"source_only_cross"
                runner.run_pairs(variant, [d0], cross, reuse_paths=[log])
                with self.subTest(change="source_only_import"):
                    self.assertTrue(all("reused_from_request_hash" not in row
                                        for row in common.read_jsonl(cross/"games.jsonl")))

                provider.write_text(provider.read_text().replace("moves[0]", "moves[-1]"))
                self.assertEqual(common.code_manifest()["code_hash"], code_hash)
                _, changed = build.build_agent(variant, tmp/"changed"/"student_agent.py", root=root)
                self.assertNotEqual(changed["artifact_hash"], common.read_jsonl(log)[0]["candidate_artifact_hash"])
                referee.reset_mock()
                with self.assertRaisesRegex(ValueError, "new output directory"):
                    runner.run_pairs(variant, [d0], output)
                referee.assert_not_called()
                self.assertEqual({path: path.read_bytes() for path in output.rglob("*") if path.is_file()}, saved)

    def test_changed_machine_environment_prevents_resume_and_seat_import(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        variant = {"variant_id": "fixture"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
                patch.object(runner, "SnapshotWorld", side_effect=referee_fixture):
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.pop(key, None)
            output = Path(tmp)/"first"
            original = runner.run_pairs(variant, [row], output)
            machine = common.load_json(output/"plan.json")["request"]["machine"]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                self.assertIsNone(machine["thread_environment"][key])
                self.assertEqual(machine["strict_thread_environment"][key], "1")
            self.assertEqual(runner.run_pairs(variant, [row], output), original)
            # A refused promotion must preserve even a cached candidate module.
            py_compile.compile(str(output/"candidate/student_agent.py"), doraise=True)
            saved = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
            for name, changed in (
                    ("host", patch.object(common.platform, "node", return_value="other-lab-host")),
                    ("python", patch.object(common.sys, "executable", "/other/bin/python")),
                    ("prefix", patch.object(common.sys, "prefix", "/other/env")),
                    ("base", patch.object(common.sys, "base_prefix", "/other/base")),
                    ("omp", patch.dict(os.environ, OMP_NUM_THREADS="2")),
                    ("mkl", patch.dict(os.environ, MKL_NUM_THREADS="2")),
                    ("blas", patch.dict(os.environ, OPENBLAS_NUM_THREADS="2")),
                    ("parent", patch.dict(os.environ, OPENBLAS_NUM_THREADS="1"))):
                with changed:
                    with self.subTest(name=name, mode="resume"):
                        with self.assertRaisesRegex(ValueError, "new output directory"):
                            runner.run_pairs(variant, [row], output)
                        self.assertEqual({p: p.read_bytes() for p in output.rglob("*") if p.is_file()}, saved)
                    with self.subTest(name=name, mode="import"):
                        destination = Path(tmp)/name
                        runner.run_pairs(variant, [row], destination, reuse_paths=[output/"games.jsonl"])
                        rows = common.read_jsonl(destination/"games.jsonl")
                        self.assertEqual(len(rows), 2)
                        self.assertTrue(all("reused_from_request_hash" not in r for r in rows))

    def test_registry_source_changes_formal_code_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in set(common.code_manifest()["files"]) | {"store.py"}:
                destination = root/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.REPO_ROOT/relative, destination)
            with patch.object(common, "REPO_ROOT", root):
                original = common.code_manifest()["code_hash"]
                registry = root/"store.py"
                registry.write_text(registry.read_text() + "# changed registry source\n")
                self.assertNotEqual(common.code_manifest()["code_hash"], original)

    def test_formal_reuse_requires_full_manifest_even_if_the_scheduled_row_matches(self):
        original = runner.make_manifest()
        other = copy.deepcopy(original)
        for split, replacement in (("D1", "T"), ("T", "D1")):
            # Choose from original to swap two different nonreserved cells in the same stratum.
            source = next(i for i, r in enumerate(original["rows"]) if r["split"] == split
                          and r["obstacle_class"] == "corner" and r["removed_size"] == 0)
            other["rows"][source]["split"] = replacement
        other["manifest_hash"] = common.digest({k: v for k, v in other.items() if k != "manifest_hash"})
        original_rows = runner.validate_manifest(original)
        other_rows = runner.validate_manifest(other)
        d0 = next(r for r in original_rows if r["split"] == "D0")
        d1 = next(r for r in original_rows if r["split"] == "D1")
        other_d0 = next(r for r in other_rows if r["config_id"] == d0["config_id"])
        self.assertNotEqual(original["manifest_hash"], other["manifest_hash"])
        self.assertNotIn("manifest_hash", original["rows"][0])
        variant = {"variant_id": "fixture"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "SnapshotWorld",
                                                                side_effect=referee_fixture) as referee:
            output = Path(tmp)
            runner.run_pairs(variant, [d0], output/"first")
            log = output/"first"/"games.jsonl"
            for label, rows, expected in (("same", [d0, d1], 2), ("different", [other_d0], 2)):
                with self.subTest(label=label):
                    referee.reset_mock()
                    runner.run_pairs(variant, rows, output/label, reuse_paths=[log])
                    self.assertEqual(referee.call_count, expected)
            records = common.read_jsonl(log)
            legacy = output/"legacy.jsonl"
            for row in records:
                common.append_jsonl(legacy, {k: v for k, v in row.items() if k != "manifest_hash"})
            referee.reset_mock()
            runner.run_pairs(variant, [d0], output/"from_legacy", reuse_paths=[legacy])
            with self.subTest(label="legacy"):
                self.assertEqual(referee.call_count, 2)
            self.assertTrue(all(r.get("manifest_hash") == original["manifest_hash"] for r in records))
            self.assertEqual(common.load_json(output/"first"/"plan.json")["request"]["manifest_hash"],
                             original["manifest_hash"])
            with self.assertRaises(ValueError):
                runner.run_pairs(variant, [original["rows"][0]], output/"unbound", allow_test=True)
            with self.assertRaises(ValueError):
                runner.run_pairs(variant, [d1, other_d0], output/"mixed")

    def test_candidate_budget_uses_unrounded_agent_time_and_preserves_referee_outcomes(self):
        row = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        with tempfile.TemporaryDirectory() as tmp:
            for case, elapsed in (("below", 1.599), ("equal", 1.6), ("above", 1.6000001), ("custom", 0.21),
                                  ("pass", 1.9), ("init", 0), ("crash", 0), ("strike", 0)):
                variant = {"variant_id": "fixture", "step_budget": 0.2 if case == "custom" else 1.6}
                tier = "smoke" if case == "custom" else "formal"
                def fake_runner(spec, **kwargs):
                    candidate = Path(spec).parent.name == "candidate"
                    def start(seat, rules):
                        if candidate and case == "init":
                            raise AgentLoadError("fixture init failure")
                    def step(snapshot, player, opponent, hard_limit):
                        kind = case if candidate and case == "crash" else "ok"
                        move = None if candidate and case == "strike" else get_valid_moves(*snapshot, player)[0]
                        return StepResult(kind, move, elapsed if candidate else 1.9, 2.5, None)
                    return SimpleNamespace(name="candidate" if candidate else "opponent", start=start, step=step,
                                           stats=dict, stop=lambda: None)

                def two_turns(world):
                    seat = 1 if world.runner(1).name == "candidate" else 2
                    with patch("world.must_pass", side_effect=lambda b, r, p, **kw: case == "pass" and p == seat):
                        world.step()
                        world.step()
                    if not world.over:
                        world.finish(seat, "line")
                    return world.record

                directory = Path(tmp)/case
                with self.subTest(case=case), patch.object(runner, "SubprocessRunner", side_effect=fake_runner), \
                        patch.object(runner.SnapshotWorld, "play", two_turns):
                    result = runner.run_pairs(variant, [row], directory, execution_tier=tier)
                    self.assertTrue(result["complete"])
                    self.assertEqual(result["execution_tier"], tier)
                    self.assertEqual(result["adjudicated"], 2)
                    won = case not in ("init", "crash")
                    self.assertEqual((result["W"], result["score"]), (2 if won else 0, 1.0 if won else 0.0))
                    self.assertEqual(result["paired_scores"], {row["config_id"]: float(won)})
                    eligible = case in ("below", "equal", "pass")
                    self.assertEqual((result["eligible"], result["formal_ranking_allowed"]), (eligible, eligible))
                    failed = case in ("above", "custom")
                    self.assertEqual(result["candidate_budget_failures"], 2 if failed else 0)
                    for record in common.read_jsonl(directory/"games.jsonl"):
                        expected_max = None if case in ("pass", "init", "crash") else elapsed
                        self.assertEqual(record["candidate_max_step_s"], expected_max)
                        self.assertEqual(record["candidate_budget_failure"], failed)
                    self.assertEqual(runner.run_pairs(variant, [row], directory, execution_tier=tier), result)
                    if case == "above":
                        imported = runner.run_pairs(variant, [row], Path(tmp)/"import", reuse_paths=[directory/"games.jsonl"])
                        self.assertFalse(imported["eligible"])
                        self.assertEqual(imported["candidate_budget_failures"], 2)

    def test_ineligible_formal_games_stay_in_denominator(self):
        self.assertFalse(runner.summarize_games([], [])["eligible"])
        config = next(r for r in runner.validate_manifest(runner.make_manifest()) if r["split"] == "D0")
        _, _, obs, removed = new_game(config["setup_seed"])
        game = GameRecord(config["setup_seed"], obs, removed, {}, {}, winner=2, reason="hard_timeout")
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "SnapshotWorld") as referee:
            referee.return_value.turns = []
            referee.return_value.step_times = {1: [], 2: []}
            referee.return_value.obstacle = obs
            referee.return_value.removed = removed
            referee.return_value.play.return_value = game
            result = runner.run_pairs({"variant_id": "fixture"}, [config], tmp)
            self.assertTrue(result["complete"])
            self.assertEqual((result["W"], result["L"], result["score"]), (1, 1, 0.5))
            self.assertFalse(result["eligible"])
            self.assertFalse(result["formal_ranking_allowed"])
            records = common.read_jsonl(Path(tmp)/"games.jsonl")
            derived = Path(tmp)/"copied.jsonl"
            for row in records:
                common.append_jsonl(derived, row)
            referee.reset_mock()
            runner.run_pairs({"variant_id": "fixture"}, [config], Path(tmp)/"exact", reuse_paths=[derived])
            referee.assert_not_called()
            for field, change in (("code_hash", "old"), ("machine", {}), ("setup_seed", -1),
                                  ("execution_tier", "smoke"), ("config_hash", "different-budget")):
                for row in records:
                    common.append_jsonl(Path(tmp)/field, dict(row, **{field: change}))
                referee.reset_mock()
                runner.run_pairs({"variant_id": "fixture"}, [config], Path(tmp)/("run_" + field),
                                 reuse_paths=[Path(tmp)/field])
                self.assertEqual(referee.call_count, 2)

    def test_no_game_manifest_freeze_and_position_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp00_prepare.prepare({"manifest_seed": 11, "run_reference": False}, tmp)
            self.assertFalse((Path(tmp) / "reference").exists())
            with self.assertRaises(ValueError):
                exp00_prepare.prepare({"manifest_seed": 12, "run_reference": False}, tmp)
            result = dict(complete=True, eligible=False, request_hash="r",
                          code_hash=common.code_manifest()["code_hash"])
            old_game = {"request_hash": "r", "status": "adjudicated", "split": "D0", "game_id": "old",
                        "config_id": "r0c0_S", "variant_id": "S0_E1_N", "obstacle": [0, 0], "removed_size": 0,
                        "turns": [{"move": {}, "ply": 0, "state": common.state_record(initial_state()),
                                   "actor_role": "candidate"}]}
            common.append_jsonl(Path(tmp)/"reference"/"games.jsonl", old_game)
            with patch.object(exp00_prepare, "run_pairs", return_value=result):
                exp00_prepare.prepare({"manifest_seed": 11}, tmp)
            payload = common.load_json(Path(tmp) / "positions.json")
            self.assertEqual(payload["positions"], [])
            self.assertIs(payload.get("reference_eligible"), False)
            self.assertEqual(payload.get("reference_code_hash"), result["code_hash"])

    def test_formal_rejects_unqualified_reference_and_component_evidence_up_front(self):
        with tempfile.TemporaryDirectory() as tmp:
            pos = [{"position_id": "a", "state": common.state_record(initial_state()), "split": "D0"}]
            current = common.code_manifest()["code_hash"]
            payload = dict(positions=pos, positions_hash=common.digest(pos), execution_tier="formal",
                           reference_complete=True, reference_eligible=True, reference_code_hash=current)
            config = dict(positions=str(Path(tmp)/"p.json"), components_summary=str(Path(tmp)/"c.json"),
                          run_confirmation=False, depths=[2, 3])
            components = dict(correct=True, execution_tier="formal", positions_hash=payload["positions_hash"],
                              code_hash=current, evidence_sufficient=True,
                              common_by_workload={"cold": [["a", 2], ["a", 3]], "iterative": [["a", 2], ["a", 3]]})
            common.atomic_json(config["components_summary"], components)
            for module in (exp01_components, exp02_search):
                for update in ({"reference_complete": False}, {"reference_eligible": False},
                               {"reference_code_hash": "wrong"}):
                    common.atomic_json(config["positions"], dict(payload, **update))
                    with patch.object(module, "exact_references", side_effect=AssertionError("started search")):
                        with self.subTest(module=module.__name__, update=update), self.assertRaises(ValueError):
                            module.run(config, tmp)
            common.atomic_json(config["positions"], payload)
            for update in ({"evidence_sufficient": False}, {"common_by_workload": {"cold": [["a", 2]]}}):
                common.atomic_json(config["components_summary"], dict(components, **update))
                with patch.object(exp02_search, "exact_references", side_effect=AssertionError("started search")):
                    with self.subTest(update=update), self.assertRaises(ValueError):
                        exp02_search.run(config, tmp)

    def formal_config(self, output, positions=None):
        manifest = runner.make_manifest()
        if positions is None:
            positions = [{"position_id": "a", "state": common.state_record(initial_state()),
                          "split": "D0", "obstacle_class": "corner"}]
        common.atomic_json(output / "manifest.json", manifest)
        common.atomic_json(output / "positions.json", {
            "positions": positions, "positions_hash": common.digest(positions), "execution_tier": "formal",
            "reference_complete": True, "reference_eligible": True,
            "reference_code_hash": common.code_manifest()["code_hash"], "manifest_hash": manifest["manifest_hash"]})
        config = dict(positions=output/"positions.json", components_summary=output/"step1/components_summary.json",
                      manifest=output/"manifest.json", depths=[1], online_budget=1.6)
        exp01_components.run(config, output/"step1")
        return config

    @staticmethod
    def timed_step(state, variant, **kwargs):
        depth = 3 if variant["variant_id"] == "S0" else 2
        return dict(value=0, move=None, status="completed", completed_depth=depth, elapsed=1.59,
                    budget_ok=True, hard_limit_ok=True, solved_remaining=False)

    def test_measurements_reject_orphan_mixed_and_inconsistent_requests_without_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            positions = [{"position_id": "a", "state": common.state_record(initial_state()),
                          "obstacle_class": "corner"}]
            evaluation = {"evaluation_id": "E1"}
            refs = measure.exact_references(positions, [1], evaluation, output/"refs")
            def timing(path, tier):
                return measure.measure(positions, {"S0": SearchConfig()}, [1], evaluation, path, refs,
                                       online_budget=0.001 if tier == "smoke" else 1.6, execution_tier=tier)
            with patch.object(measure, "measure_online_step", side_effect=self.timed_step):
                for tier in ("smoke", "formal"):
                    timing(output/tier, tier)
                smoke_log = (output/"smoke/measurements.jsonl").read_bytes()
                for change in ("orphan", "malformed_orphan", "sidecar_body", "all_smoke", "one_smoke", "legacy"):
                    with self.subTest(change=change):
                        directory = output/change
                        shutil.copytree(output/"formal", directory)
                        log, sidecar = directory/"measurements.jsonl", directory/"request.json"
                        if change in ("orphan", "malformed_orphan"):
                            sidecar.unlink()
                            if change == "malformed_orphan":
                                log.write_bytes(b'{"partial"')
                        elif change == "sidecar_body":
                            saved = common.load_json(sidecar)
                            saved["request"]["online_budget"] = 0.01
                            common.atomic_json(sidecar, saved)
                        elif change == "all_smoke":
                            log.write_bytes(smoke_log)
                        elif change == "one_smoke":
                            log.write_bytes(log.read_bytes() + smoke_log.splitlines()[0] + b'\n{"partial"')
                        else:
                            rows = common.read_jsonl(log)
                            rows[-1].pop("request_hash", None)
                            log.write_text("".join(common.canonical_json(r) + "\n" for r in rows))
                        saved = {p: p.read_bytes() for p in directory.iterdir()}
                        with self.assertRaises(ValueError):
                            timing(directory, "formal")
                        self.assertEqual({p: p.read_bytes() for p in directory.iterdir()}, saved)
                directory = output/"formal"
                rows = common.read_jsonl(directory/"measurements.jsonl")
                identity = common.load_json(directory/"request.json")["request_hash"]
                self.assertEqual({r.get("request_hash") for r in rows}, {identity})
                # Preserve the completed prefix, repair its partial tail and finish the request.
                prefix = common.canonical_json(rows[0]) + "\n"
                (directory/"measurements.jsonl").write_text(prefix + '{"partial"')
                resumed, summary = timing(directory, "formal")
                self.assertTrue(summary["correct"])
                self.assertEqual(resumed[0], rows[0])
                self.assertEqual({r["case_id"] for r in resumed}, {r["case_id"] for r in rows})
                (directory/"measurements.jsonl").unlink()
                self.assertTrue(timing(directory, "formal")[1]["correct"])
                empty = output/"empty"
                empty.mkdir()
                (empty/"measurements.jsonl").touch()
                self.assertTrue(timing(empty, "formal")[1]["correct"])

    def test_exact_reference_rows_belong_to_the_saved_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = self.formal_config(output)
            positions = common.load_json(config["positions"])["positions"]
            summary = common.load_json(config["components_summary"])
            ref_dir = output/"step1/references"
            log = ref_dir/"measurements.jsonl"
            row, = common.read_jsonl(log)
            row["request_hash"] = "foreign"
            log.write_text(common.canonical_json(row) + "\n")
            # Even a descriptor for these exact bytes cannot turn a foreign row into an E1 answer.
            summary["exact_reference"]["measurements_hash"] = common.file_hash(log)
            with self.subTest(path="step2"), self.assertRaises(ValueError):
                measure.load_component_references(config["components_summary"], summary["exact_reference"], positions, [1])
            before = log.read_bytes()
            with self.subTest(path="step1"), self.assertRaises(ValueError):
                measure.exact_references(positions, [1], {"evaluation_id": "E1"}, ref_dir)
            self.assertEqual(log.read_bytes(), before)

    def test_step2_uses_step1_exact_answers_including_incomplete_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            positions = [{"position_id": name, "state": common.state_record(initial_state(obstacle)),
                          "split": "D0", "obstacle_class": "corner"}
                         for name, obstacle in (("a", (0, 0)), ("b", (3, 3)))]
            search = measure.SearchEngine.fixed_depth
            def partial_reference(engine, state, depth, deadline=None, **kwargs):
                if kwargs.get("collect_ties") and state.obstacle == (3, 3):
                    deadline = 0
                return search(engine, state, depth, deadline, **kwargs)
            with patch.object(measure.SearchEngine, "fixed_depth", partial_reference):
                config = self.formal_config(output, positions)
            self.assertEqual(common.load_json(config["components_summary"])["exact_reference_incomplete"], 1)
            def e0_only(positions, depths, evaluation, *args):
                self.assertEqual(evaluation, {"evaluation_id": "E0"}, "step 2 recomputed E1")
                return measure.exact_references(positions, depths, evaluation, *args)
            with patch.object(exp02_search, "exact_references", side_effect=e0_only), \
                    patch.object(measure, "measure_online_step", side_effect=self.timed_step):
                result = exp02_search.run(dict(config, run_confirmation=False), output/"step2")
            evidence = result["decisions"][0]["evidence"]
            self.assertEqual(evidence["E1"]["common_by_workload"]["cold"], [["a", 1]])
            self.assertEqual(evidence["E0"]["common_by_workload"]["cold"], [["a", 1], ["b", 1]])
            self.assertFalse((output/"step2/references_E1").exists())

    def test_formal_step2_rejects_missing_or_changed_step1_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = self.formal_config(output)
            directory = output/"step1/references"
            for filename in ("request.json", "measurements.jsonl"):
                path = directory/filename
                original = path.read_bytes()
                for change in ("removed", "changed"):
                    if change == "removed":
                        path.unlink()
                    elif filename == "request.json":
                        request = common.load_json(path)
                        request["request"]["depths"] = [9]
                        common.atomic_json(path, request)
                    else:
                        row, = common.read_jsonl(path)
                        row["value"] += 1
                        path.write_text(common.canonical_json(row) + "\n")
                    with self.subTest(filename=filename, change=change), patch.object(exp02_search,
                            "exact_references", side_effect=AssertionError("searched before validating Step 1")):
                        with self.assertRaisesRegex(ValueError, "reference"):
                            exp02_search.run(config, output/"step2")
                    path.write_bytes(original)
            summary = common.load_json(config["components_summary"])
            summary.pop("exact_reference", None)
            common.atomic_json(config["components_summary"], summary)
            with patch.object(exp02_search, "exact_references", side_effect=AssertionError("invented reference")):
                with self.assertRaisesRegex(ValueError, "reference"):
                    exp02_search.run(config, output/"step2")

    def test_interrupted_diagnostic_rerun_preserves_published_selected_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = self.formal_config(folder)
            output = folder/"step2"
            summary_path = output/"search_selection.json"
            selected_path = output/"selected_search.json"
            match = {"complete": True, "eligible": True, "score": 0.5}

            def publish(path, value):
                if path == summary_path and value["status"] == "selected":
                    self.assertTrue(selected_path.is_file(), "selected summary preceded its role")
                    if value["S_alt"] is not None:
                        self.assertTrue((output/"alternate_search.json").is_file())
                common.atomic_json(path, value)

            with patch.object(exp02_search, "run_pairs", return_value=match), \
                    patch.object(measure, "measure_online_step", side_effect=self.timed_step), \
                    patch.object(exp02_search, "atomic_json", side_effect=publish):
                selected = exp02_search.run(config, output)
            self.assertEqual(selected["status"], "selected")
            saved_summary, saved_role = summary_path.read_bytes(), selected_path.read_bytes()
            diagnostic = dict(config, run_confirmation=False)

            def interrupt(path, value):
                if path == summary_path:
                    raise KeyboardInterrupt("before selection publication")
                common.atomic_json(path, value)

            with patch.object(exp02_search, "run_pairs", side_effect=AssertionError("replayed games")), \
                    patch.object(measure, "measure_online_step", side_effect=AssertionError("reran timings")), \
                    patch.object(measure.SearchEngine, "fixed_depth", side_effect=AssertionError("reran fixed search")):
                with patch.object(exp02_search, "atomic_json", side_effect=interrupt), \
                        self.assertRaisesRegex(KeyboardInterrupt, "before selection publication"):
                    exp02_search.run(diagnostic, output)
                self.assertEqual(summary_path.read_bytes(), saved_summary)
                self.assertTrue(selected_path.exists(), "published selection lost its selected role")
                self.assertEqual(selected_path.read_bytes(), saved_role)
                pending = exp02_search.run(diagnostic, output)
            self.assertEqual(pending["status"], "awaiting_development_confirmation")
            self.assertIsNone(pending["S_star"])
            self.assertEqual(common.load_json(summary_path), pending)
            self.assertFalse(selected_path.exists())

    def test_step2_smoke_cli_checks_explicit_manifest_before_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = runner.make_manifest()
            row = next(r for r in runner.validate_manifest(manifest) if r["split"] == "D0")
            state = initial_state(row["obstacle"], row["removed_size"])
            positions = [{"position_id": row["config_id"], "state": common.state_record(state),
                          "obstacle_class": row["obstacle_class"], "split": "D0"}]
            common.atomic_json(folder/"positions.json", {"positions": positions,
                "positions_hash": common.digest(positions), "manifest_hash": manifest["manifest_hash"]})
            common.atomic_json(folder/"manifest.json", manifest)
            common.atomic_json(folder/"other.json", runner.make_manifest(314))
            common.atomic_json(folder/"corrupt.json", dict(manifest, rows=manifest["rows"][:-1]))
            common.atomic_json(folder/"config.json", {"manifest": str(folder/"manifest.json"),
                "smoke": {"max_positions": 1, "depths": [1], "online_budget": 0.001}})
            command = [sys.executable, "-m", "search_experiments.exp02_search", "--smoke",
                       "--config", str(folder/"config.json"), "--positions", str(folder/"positions.json")]
            for manifest_name, error in (("missing.json", "FileNotFoundError"),
                                         ("other.json", "same frozen manifest"),
                                         ("corrupt.json", "each of the 48 configurations"), ("manifest.json", None)):
                with self.subTest(manifest=manifest_name):
                    output = folder/Path(manifest_name).stem
                    result = subprocess.run(command + ["--manifest", str(folder/manifest_name), "--output", str(output)],
                                            cwd=common.REPO_ROOT, text=True, capture_output=True)
                    if error is None:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(common.load_json(output/"search_selection.json")["status"], "smoke_only")
                        self.assertFalse((output/"development").exists())
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(error, result.stderr)
                        self.assertFalse(list(output.glob("**/*")))

    def test_step2_formal_diagnostics_checks_supplied_manifest_and_allows_omission(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = self.formal_config(folder)
            config["run_confirmation"] = False
            other = folder/"other.json"
            common.atomic_json(other, runner.make_manifest(314))
            with patch.object(exp02_search, "exact_references", side_effect=AssertionError("started timed work")):
                with self.assertRaisesRegex(ValueError, "same frozen manifest"):
                    exp02_search.run(dict(config, manifest=other), folder/"bad")
                with self.assertRaises(FileNotFoundError):
                    exp02_search.run(dict(config, manifest=folder/"missing.json"), folder/"missing")
            config.pop("manifest")
            with self.assertRaisesRegex(ValueError, "confirmation requires the frozen manifest"):
                exp02_search.run(dict(config, run_confirmation=True), folder/"confirmation")
            with patch.object(exp02_search, "run_pairs", side_effect=AssertionError("started games")), \
                    patch.object(measure, "measure_online_step", side_effect=self.timed_step):
                result = exp02_search.run(config, folder/"diagnostic")
            self.assertEqual(result["status"], "awaiting_development_confirmation")

    def test_formal_online_budget_is_validated_before_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = self.formal_config(output)
            for budget in (None, 0, 0.1, float("nan"), float("inf"), 2.0):
                with self.subTest(budget=budget), patch.object(exp02_search, "exact_references",
                        side_effect=AssertionError("started search with invalid budget")):
                    with self.assertRaisesRegex(ValueError, "budget"):
                        exp02_search.run(dict(config, online_budget=budget), output/"step2")

    def test_formal_performance_readiness_and_discarded_alternate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = self.formal_config(output)
            for case in ("missing_baseline", "baseline_overrun", "finalist_overrun", "unstable", "discarded_overrun"):
                counts = {}
                def timed(state, variant, **kwargs):
                    name = variant["variant_id"]
                    count = counts.get(name, 0)
                    counts[name] = count + 1
                    row = self.timed_step(state, variant)
                    if case in ("missing_baseline", "finalist_overrun", "discarded_overrun") and name == "S1":
                        row["completed_depth"] = 4
                    if case == "missing_baseline" and name == "S0" and count == 3:
                        row["completed_depth"] = None
                    if (case, name, count) in (("baseline_overrun", "S0", 5),
                                                ("finalist_overrun", "S1", 5), ("discarded_overrun", "S1", 1)):
                        row.update(budget_ok=False, hard_limit_ok=False, elapsed=2.01)
                    if case == "unstable" and name == "S1" and count == 1:
                        row["completed_depth"] = 4
                    return row

                destination = output/case
                match = dict(complete=True, eligible=True, score=0.5, paired_scores={"a": 0.5})
                with self.subTest(case=case), patch.object(measure, "measure_online_step", side_effect=timed), \
                        patch.object(exp02_search, "run_pairs", return_value=match):
                    result = exp02_search.run(config, destination)
                    if case in ("missing_baseline", "baseline_overrun", "finalist_overrun"):
                        self.assertEqual(result["status"], "blocked_performance")
                        self.assertNotIn("development", result)
                        self.assertFalse((destination/"selected_search.json").exists())
                        self.assertFalse((destination/"alternate_search.json").exists())
                    else:
                        self.assertEqual(result["status"], "selected")
                        self.assertEqual(result["S_star"], "S0")
                        if case == "discarded_overrun":
                            self.assertNotEqual(result["S_alt"], "S1")

    def test_formal_confirmation_blocked_for_missing_or_wrong_step2_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = self.formal_config(output)
            with patch.object(measure, "measure_online_step", side_effect=self.timed_step), \
                    patch.object(exp02_search, "run_pairs", side_effect=AssertionError("started formal games")):
                missing = exp02_search.run(dict(config, guard_seconds=0), output/"missing")
                self.assertEqual(missing["status"], "blocked_correctness")
                self.assertIsNone(missing["S_star"])
                self.assertTrue(missing["decisions"][0]["evidence"]["E0"]["empty_groups"])
                self.assertFalse((output/"missing"/"selected_search.json").exists())

                # Reuse actual depth-one measurements, with one logged failed diagnostic.
                cached = output/"cached"
                exp02_search.run(dict(config, run_confirmation=False), cached)
                log = cached/"S4"/"performance_E1"/"measurements.jsonl"
                rows = common.read_jsonl(log)
                wrong = next(r for r in rows if r["instrumented"] and r["checked"])
                wrong["correct"] = False
                log.write_text("".join(common.canonical_json(row) + "\n" for row in rows))
                failed = exp02_search.run(config, cached)
                self.assertEqual(failed["status"], "blocked_correctness")
                self.assertIsNone(failed["S_star"])
                self.assertIn(wrong["case_id"], failed["decisions"][3]["evidence"]["E1"]["incorrect_case_ids"])
                self.assertFalse((cached/"selected_search.json").exists())

    def test_development_selection_preserves_identity_and_excludes_failed_alt(self):
        self.assertTrue(hasattr(exp02_search, "_confirm_development"))
        variants = {"S0": exp02_search._match_variant("S0", SearchConfig()),
                    "S1": exp02_search._match_variant("S1", SearchConfig(preferred_first=True), "S0_E1_N")}
        for failed in ("S0", "S1", None):
            actual = []
            def paired(variant, *args, **kwargs):
                name = variant["variant_id"].split("_")[0]
                actual.append(variant)
                return dict(complete=True, eligible=name != failed, score=1 if name == "S0" else 0.5,
                            paired_scores={"a": 1 if name == "S0" else 0.5})
            with tempfile.TemporaryDirectory() as tmp, patch.object(exp02_search, "run_pairs", side_effect=paired):
                selection = {"S_diag": "S1", "S_alt": "S0", "S_star": None}
                exp02_search._confirm_development(selection, variants, [], Path(tmp), [])
                exp02_search._save_selection(selection, variants, Path(tmp))
                star = "S1" if failed == "S0" else "S0"
                self.assertEqual(selection["S_star"], star)
                self.assertEqual(selection["S_alt"], star if failed else "S1")
                self.assertTrue(selection["S_alt_reason"])
                self.assertIs(actual[1], variants["S1"])
                self.assertEqual(common.load_json(Path(tmp)/"selected_search.json"), variants[star])
                self.assertEqual(common.load_json(Path(tmp)/"alternate_search.json"), variants[selection["S_alt"]])
        for complete in (False, True):
            with tempfile.TemporaryDirectory() as tmp, patch.object(exp02_search, "run_pairs", return_value={
                    "complete": complete, "eligible": False}):
                selection = {"S_diag": "S1", "S_alt": "S0", "S_star": None}
                exp02_search._confirm_development(selection, variants, [], Path(tmp), [])
                self.assertIsNone(selection["S_star"])
                self.assertIsNone(selection["S_alt"])
