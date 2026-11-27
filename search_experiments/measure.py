"""Exact references, alternating timed workloads and separate diagnostic runs."""

from collections import defaultdict
from dataclasses import asdict
import csv
import importlib
from pathlib import Path
import statistics
import time

from agents.experiment_agent import ExperimentAgent
from agents.game_state import to_arrays
from agents.search_engine import SearchConfig, SearchEngine, component_configs
from search_experiments.common import (append_jsonl, atomic_json, code_manifest, digest,
    file_hash, json_default, load_json, machine_info, normalize_variant, read_jsonl, repair_jsonl_tail,
    resolve_evaluator, restore_move, restore_state, validate_positions)


def alternate_order(names, repetition):
    names = list(names)
    if repetition % 3 == 1:
        return list(reversed(names))
    if repetition % 3 == 2 and names:
        return names[1:] + names[:1]
    return names


def load_positions(path, limit=None, require_formal=False):
    payload = load_json(path)
    if require_formal:
        if (payload.get("execution_tier") != "formal" or not payload.get("reference_complete")
                or not payload.get("reference_eligible")):
            raise ValueError("formal measurements require complete, eligible formal reference games")
        if payload.get("reference_code_hash") != code_manifest()["code_hash"]:
            raise ValueError("reference game code changed; regenerate the fixed positions")
    positions = validate_positions(payload["positions"])
    if digest(payload["positions"]) != payload["positions_hash"]:
        raise ValueError("fixed position hash mismatch")
    if limit is not None:
        positions = positions[:limit]
    return positions, payload


def _case_store(output, request):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "request.json"
    identity = digest(request)
    log = output / "measurements.jsonl"
    if path.exists():
        saved = load_json(path)
        if (not isinstance(saved, dict) or saved.get("request_hash") != identity
                or digest(saved.get("request")) != identity):
            raise ValueError("measurement request changed; use a new output directory")
    elif log.exists() and log.stat().st_size:
        raise ValueError("measurement log has no request; use a new output directory")
    records = read_jsonl(log)
    if any(not isinstance(row, dict) or row.get("request_hash") != identity for row in records):
        raise ValueError("measurement row belongs to another or unknown request")
    if not path.exists():
        atomic_json(path, {"request_hash": identity, "request": request})
    repair_jsonl_tail(log)
    return log, records, {r["case_id"]: r for r in records}, identity


def exact_references(positions, depths, evaluation, output, guard_seconds=60):
    request = {"positions": digest(positions), "depths": depths, "evaluation": evaluation,
               "guard_seconds": guard_seconds, "code": code_manifest(), "machine": machine_info()}
    log, records, done, request_hash = _case_store(output, request)
    if not log.exists():
        log.touch()  # Empty requests still have a reference log for the summary digest.
    cfg = component_configs()["C0"]
    evaluator = resolve_evaluator(evaluation)
    for position in positions:
        for depth in depths:
            case_id = digest([position["position_id"], depth, evaluation, "reference"])
            if case_id in done:
                continue
            state = restore_state(position["state"])
            engine = SearchEngine(cfg, evaluator=evaluator)
            result = engine.fixed_depth(state, depth, time.perf_counter() + guard_seconds,
                                        collect_ties=True)
            row = {"case_id": case_id, "request_hash": request_hash, "position_id": position["position_id"],
                   "depth": depth, "evaluation": evaluation, "status": result.status, "value": result.value,
                   "optimal_moves": result.optimal_moves, "elapsed": result.elapsed}
            append_jsonl(log, row)
            records.append(row)
    return {(row["position_id"], row["depth"]): row for row in records}


def load_component_references(summary_path, descriptor, positions, depths):
    """Load the E1 answers saved with Step 1, including searches that timed out."""
    if not isinstance(descriptor, dict) or descriptor.get("path") != "references":
        raise ValueError("missing Step 1 exact reference descriptor; regenerate Step 1")
    directory = Path(summary_path).parent / descriptor["path"]
    log = directory / "measurements.jsonl"
    try:
        saved = load_json(directory / "request.json")
        log_hash = file_hash(log)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("missing or invalid Step 1 reference files; regenerate Step 1") from exc
    request = saved.get("request", {})
    evaluation = {"evaluation_id": "E1"}
    if (saved.get("request_hash") != digest(request)
            or saved["request_hash"] != descriptor.get("request_hash")
            or log_hash != descriptor.get("measurements_hash")
            or request.get("positions") != digest(positions)
            or request.get("evaluation") != evaluation or request.get("code") != code_manifest()
            or not set(depths).issubset(request.get("depths", []))):
        raise ValueError("Step 1 reference identity or contents changed; regenerate Step 1")

    expected = {(position["position_id"], depth) for position in positions for depth in request["depths"]}
    references = {}
    for row in read_jsonl(log):
        key = row["position_id"], row["depth"]
        if (key not in expected or key in references or row.get("evaluation") != evaluation
                or row.get("request_hash") != saved["request_hash"]
                or row.get("case_id") != digest([*key, evaluation, "reference"])):
            raise ValueError("unexpected or duplicate Step 1 reference case")
        references[key] = row
    if references.keys() != expected:
        raise ValueError("Step 1 reference cases are missing; regenerate Step 1")
    return references


def check_reference(result, reference, target_reached=True):
    if reference is None or reference["status"] != "completed" or not target_reached:
        return {"checked": False, "correct": None}
    optimal = [restore_move(record) for record in reference["optimal_moves"]]
    return {"checked": True, "correct": result.value == reference["value"] and result.move in optimal,
            "value_equal": result.value == reference["value"], "move_optimal": result.move in optimal}


def measure_online_step(state, variant, evaluation_factory=None, book_factory=None):
    """Whole-step timer reusable by F's combinations and B's N/D/H experiments."""
    variant = normalize_variant(variant)
    if evaluation_factory is None:
        evaluation_factory = lambda config: resolve_evaluator(config)
    book_config = variant["book"]
    provider = book_config.get("provider")
    if book_config.get("mode", "N") != "N" and book_factory is None and provider:
        book_factory = importlib.import_module(provider).make_book_hook
    initialized = time.perf_counter()
    agent = ExperimentAgent(variant, evaluation_factory=evaluation_factory, book_factory=book_factory)
    object_init = time.perf_counter() - initialized
    agent.call_count = (state.ply - state.player + 1) // 2
    if 2 * agent.call_count + state.player - 1 != state.ply:
        raise ValueError("online position player/ply parity is invalid")
    board, reserves = to_arrays(state)
    started = time.perf_counter()
    move = agent.step(board, reserves, state.player, 3 - state.player)
    elapsed = time.perf_counter() - started
    return dict(agent.last_info, move=move, elapsed=elapsed, object_init_s=object_init,
                budget_ok=elapsed <= variant["step_budget"], hard_limit_ok=elapsed < 2.0)


def measure(positions, configs, depths, evaluation, output, references, *, repeats=3,
            guard_seconds=60, workloads=("cold", "iterative"), online_budget=None,
            diagnostics=True, execution_tier="formal"):
    if any(position.get("split") == "T" for position in positions):
        raise ValueError("held-out T positions must not enter development selection")
    if repeats != 3:
        raise ValueError("the experiment plan requires exactly three alternating repetitions")
    cfgs = {key: value if isinstance(value, SearchConfig) else SearchConfig(**value)
            for key, value in configs.items()}
    reference_answers = []
    for position in positions:
        for depth in depths:
            pid = position["position_id"]
            ref = references.get((pid, depth))
            answer = None if ref is None else {k: ref.get(k) for k in ("status", "value", "optimal_moves")}
            reference_answers.append([pid, depth, answer])
    request = {"positions": digest(positions), "configs": cfgs, "variant_order": list(cfgs), "depths": depths,
               "reference_answers_hash": digest(reference_answers),
               "evaluation": evaluation, "repeats": repeats, "guard_seconds": guard_seconds,
               "workloads": workloads, "online_budget": online_budget, "diagnostics": diagnostics,
               "code": code_manifest(), "machine": machine_info(), "execution_tier": execution_tier}
    log, records, done, request_hash = _case_store(output, request)
    evaluator = resolve_evaluator(evaluation)

    def perform(position, name, depth, workload, repetition, instrumented):
        case_id = digest([position["position_id"], name, depth, workload, repetition, instrumented])
        if case_id in done:
            return
        state = restore_state(position["state"])
        target = depth if workload == "cold" else min(depth, 100 - state.ply) if depth is not None else None
        before = time.perf_counter()
        if workload == "online":
            if instrumented:
                raise ValueError("online instrumentation is measured separately at fixed depth")
            variant = {"variant_id": name, "search": asdict(cfgs[name]), "evaluation": evaluation,
                       "book": {"mode": "N"}, "step_budget": online_budget, "telemetry": False}
            info = measure_online_step(state, variant, evaluation_factory=lambda _: evaluator)
            row = {"value": info.get("value"), "move": info["move"], "status": info["status"],
                   "stop_reason": info.get("stop_reason"), "interrupted_depth": info.get("interrupted_depth"),
                   "completed_depth": info["completed_depth"], "elapsed": info["elapsed"],
                   "statistics": {"enabled": False}, "correct": None, "checked": False,
                   "budget_ok": info["budget_ok"], "hard_limit_ok": info["hard_limit_ok"],
                   "solved_remaining": info.get("solved_remaining", False)}
        else:
            engine = SearchEngine(cfgs[name], evaluator=evaluator, diagnostics=instrumented)
            deadline = time.perf_counter() + guard_seconds
            result = (engine.fixed_depth(state, depth, deadline) if workload == "cold"
                      else engine.search(state, deadline, max_depth=depth))
            reached = result.status == "completed" and result.completed_depth >= target
            row = {"value": result.value, "move": result.move, "status": result.status,
                   "stop_reason": result.stop_reason, "interrupted_depth": result.interrupted_depth,
                   "completed_depth": result.completed_depth, "elapsed": result.elapsed,
                   "statistics": result.statistics, "target_reached": reached,
                   "iterations": result.iterations, "solved_remaining": result.solved_remaining}
            row.update(check_reference(result, references.get((position["position_id"], depth)), reached))
        row.update(case_id=case_id, request_hash=request_hash, position_id=position["position_id"], variant_id=name,
                   evaluation=evaluation, target_depth=depth, effective_target_depth=target,
                   workload=workload, repetition=repetition, instrumented=instrumented,
                   measurement="diagnostic" if instrumented else "timing",
                   ply=state.ply, near_cap=state.ply >= 90,
                   obstacle_class=position["obstacle_class"], execution_tier=execution_tier)
        append_jsonl(log, row)
        records.append(row)
        done[case_id] = row

    for repetition in range(repeats):
        for position in positions:
            for workload in workloads:
                for depth in depths:
                    for name in alternate_order(cfgs, repetition):
                        if name == "C0" and workload == "iterative":
                            continue
                        perform(position, name, depth, workload, repetition, False)
            if online_budget is not None:
                for name in alternate_order(cfgs, repetition):
                    perform(position, name, None, "online", repetition, False)
    if diagnostics:
        for position in positions:
            for workload in workloads:
                for depth in depths:
                    for name in cfgs:
                        if name == "C0" and workload == "iterative":
                            continue
                        perform(position, name, depth, workload, -1, True)
    summary = summarize_measurements(records, list(cfgs), depths, positions, workloads, diagnostics, references)
    atomic_json(Path(output) / "summary.json", summary)
    export_timing_csv(records, Path(output) / "timings.csv")
    return records, summary


def _verified(row):
    return (row["status"] == "completed" and row.get("target_reached")
            and row.get("checked") and row.get("correct"))


def summarize_measurements(records, names, depths, positions, workloads=("cold", "iterative"),
                           diagnostics=False, references=None):
    # Group rows by the planned case, including its repetition. A diagnostic
    # cannot stand in for a missing timed search.
    cases = defaultdict(list)
    for row in records:
        if row["workload"] != "online":
            cases[(row["workload"], row["target_depth"], row["position_id"], row["variant_id"],
                   row["repetition"], row["instrumented"])].append(row)
    verified = {key: rows[0] for key, rows in cases.items() if len(rows) == 1 and _verified(rows[0])}
    position_ids = [p["position_id"] for p in positions]
    errors = [r["case_id"] for r in records if r.get("checked") and not r.get("correct")]
    incomplete = [r["case_id"] for r in records if r["workload"] != "online" and not _verified(r)]
    common = {workload: [] for workload in workloads}
    empty_groups, aggregates = [], []
    full_coverage = True
    group_count = 0
    for workload in workloads:
        compared = [n for n in names if (n, workload) != ("C0", "iterative")]
        if not compared:
            continue
        for depth in depths:
            group_count += 1
            common_ids = [
                pid for pid in position_ids
                if (references is None or references.get((pid, depth), {}).get("status") == "completed")
                and all((workload, depth, pid, name, rep, False) in verified
                        for name in compared for rep in range(3))]
            common[workload].extend([pid, depth] for pid in common_ids)
            if len(common_ids) != len(position_ids):
                full_coverage = False
            if diagnostics and not all((workload, depth, pid, name, -1, True) in verified
                                       for pid in position_ids for name in compared):
                full_coverage = False
            if not common_ids:
                empty_groups.append({"workload": workload, "depth": depth,
                                     "reason": "no common position with three verified timing repetitions"})
                continue
            for name in compared:
                values = [verified[workload, depth, pid, name, rep, False]["elapsed"]
                          for pid in common_ids for rep in range(3)]
                aggregates.append({"variant_id": name, "workload": workload, "depth": depth,
                                   "count": len(values), "median_s": statistics.median(values),
                                   "min_s": min(values), "max_s": max(values)})
    sufficient = group_count > 0 and not empty_groups
    return {"correct": sufficient and not errors, "no_mismatches": not errors,
            "evidence_sufficient": sufficient, "empty_groups": empty_groups,
            "incorrect_case_ids": errors, "incomplete_case_ids": incomplete,
            "verification_complete": sufficient and full_coverage and not errors,
            "common_completed_position_depths": common.get("cold", []),
            "common_by_workload": common, "aggregates": aggregates}


def export_timing_csv(records, path):
    fields = ["position_id", "variant_id", "workload", "target_depth", "effective_target_depth",
              "repetition", "measurement", "elapsed", "completed_depth", "status", "correct",
              "target_reached", "near_cap", "budget_ok", "stop_reason", "interrupted_depth"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    per_position = []
    groups = {}
    for record in records:
        if record["instrumented"] or record["workload"] == "online":
            continue
        key = (record["position_id"], record["variant_id"], record["workload"], record["target_depth"])
        groups.setdefault(key, []).append(record)
    for (position, variant, workload, depth), rows in groups.items():
        complete = (len(rows) == 3 and {r["repetition"] for r in rows} == {0, 1, 2}
                    and all(_verified(r) for r in rows))
        times = [r["elapsed"] for r in rows]
        per_position.append({"position_id": position, "variant_id": variant, "workload": workload,
            "target_depth": depth, "complete_three": complete,
            "median_s": statistics.median(times) if complete else None,
            "min_s": min(times) if complete else None, "max_s": max(times) if complete else None})
    with Path(path).with_name("per_position_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["position_id", "variant_id", "workload", "target_depth",
                                        "complete_three", "median_s", "min_s", "max_s"])
        writer.writeheader()
        writer.writerows(per_position)


def validation_passes(records, refs, positions, depths, names, workloads):
    return summarize_measurements(records, names, depths, positions, workloads, references=refs)["correct"]


def budget_failures(records):
    return {row["variant_id"] for row in records if row["workload"] == "online"
            and not row.get("instrumented") and not (row["budget_ok"] and row["hard_limit_ok"])}


def performance_samples(records, parent, candidate, baseline=None):
    """Use common online observations and pairwise verified fixed-time ratios."""
    names = (parent, candidate)
    online_names = set(names)
    if baseline is not None:
        online_names.add(baseline)
    selection_depth = max((r["target_depth"] for r in records if r["workload"] == "iterative"
                           and r["variant_id"] in names), default=None)
    online_cases, fixed_cases = defaultdict(list), defaultdict(list)
    budget_valid = not budget_failures(records).intersection(names)
    for row in records:
        if row["variant_id"] not in online_names or row.get("instrumented"):
            continue
        key = (row["position_id"], row["repetition"], row["variant_id"])
        if row["workload"] == "online":
            online_cases[key].append(row)
        elif row["workload"] == "iterative" and row["target_depth"] == selection_depth:
            fixed_cases[key].append(row)

    def unique_rows(cases):
        for key, rows in cases.items():
            if len(rows) == 1 and not rows[0]["near_cap"] and not rows[0]["solved_remaining"]:
                yield key, rows[0]

    online = {key: row["completed_depth"] for key, row in unique_rows(online_cases)
              if row["completed_depth"] is not None}
    fixed = {key: row["elapsed"] for key, row in unique_rows(fixed_cases) if _verified(row)}

    def common_positions(samples, required_names):
        return sorted(pid for pid in {key[0] for key in samples}
                      if all((pid, rep, name) in samples for rep in range(3) for name in required_names))

    depths = {(pid, rep): {name: online[pid, rep, name] for name in names}
              for pid in common_positions(online, online_names) for rep in range(3)}
    ratios = {(pid, rep): fixed[pid, rep, candidate] / fixed[pid, rep, parent]
              for pid in common_positions(fixed, names) for rep in range(3)}
    return depths, ratios, budget_valid


def compare_performance(records, parent, candidate, baseline=None):
    depths, ratios, budget_valid = performance_samples(records, parent, candidate, baseline)
    rankings, repetitions = [], []
    for repetition in range(3):
        sample = [values for (pid, rep), values in depths.items() if rep == repetition]
        means = {name: statistics.mean(r[name] for r in sample) if sample else None for name in (parent, candidate)}
        fixed = [r for (pid, rep), r in ratios.items() if rep == repetition]
        ratio = statistics.median(fixed) if fixed else None
        if not sample:
            winner = "unresolved"
        elif means[candidate] != means[parent]:
            winner = candidate if means[candidate] > means[parent] else parent
        elif ratio is None:
            winner = "unresolved"
        elif ratio < 1:
            winner = candidate
        else:
            winner = parent
        rankings.append(winner)
        repetitions.append({"repetition": repetition, "mean_depth": means,
                            "fixed_time_ratio_candidate_over_parent": ratio, "winner": winner,
                            "depth_position_count": len(sample)})
    retained = budget_valid and rankings == [candidate] * 3
    sufficient = "unresolved" not in rankings
    reason = "candidate_stably_better" if retained else "keep_parent"
    if not sufficient:
        reason = "no_common_online_observations" if not depths else "no_verified_fixed_tiebreak"
    return {"retain_candidate": retained, "budget_valid": budget_valid, "evidence_sufficient": sufficient,
            "consistent": sufficient and len(set(rankings)) == 1,
            "depth_positions": sorted({pid for pid, rep in depths}),
            "fixed_positions": sorted({pid for pid, rep in ratios}),
            "repetitions": repetitions, "reason": reason}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shared fixed-state measurements for A/F/B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    positions, _ = load_positions(config["positions"], require_formal=not args.smoke)
    online_budget = config.get("online_budget")
    if not args.smoke and online_budget is not None and online_budget != 1.6:
        raise ValueError("formal measurements require an online budget of 1.6 seconds")
    output = Path(args.output)
    evaluation = config.get("evaluation", {"evaluation_id": "E1"})
    depths = config.get("depths", [2, 3])
    guard = config.get("guard_seconds", 60)
    refs = exact_references(positions, depths, evaluation, output / "references", guard)
    _, summary = measure(positions, config["variants"], depths, evaluation, output / "measurements", refs,
                         guard_seconds=guard, online_budget=online_budget,
                         execution_tier="smoke" if args.smoke else "formal")
    print("correct:", summary["correct"])


if __name__ == "__main__":
    main()
