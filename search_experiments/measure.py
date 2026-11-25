"""Exact references, alternating timed workloads and separate diagnostic runs."""

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
    json_default, load_json, machine_info, read_jsonl, repair_jsonl_tail, resolve_evaluator,
    restore_move, restore_state, validate_positions)


def alternate_order(names, repetition):
    names = list(names)
    if repetition % 3 == 1:
        return list(reversed(names))
    if repetition % 3 == 2 and names:
        return names[1:] + names[:1]
    return names


def load_positions(path, limit=None):
    payload = load_json(path)
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
    if path.exists() and load_json(path)["request_hash"] != identity:
        raise ValueError("measurement request changed; use a new output directory")
    atomic_json(path, {"request_hash": identity, "request": request})
    log = output / "measurements.jsonl"
    repair_jsonl_tail(log)
    records = read_jsonl(log)
    return log, records, {r["case_id"]: r for r in records}


def exact_references(positions, depths, evaluation, output, guard_seconds=60):
    request = {"positions": digest(positions), "depths": depths, "evaluation": evaluation,
               "guard_seconds": guard_seconds, "code": code_manifest(), "machine": machine_info()}
    log, records, done = _case_store(output, request)
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
            row = {"case_id": case_id, "position_id": position["position_id"], "depth": depth,
                   "evaluation": evaluation, "status": result.status, "value": result.value,
                   "optimal_moves": result.optimal_moves, "elapsed": result.elapsed}
            append_jsonl(log, row)
            records.append(row)
    return {(row["position_id"], row["depth"]): row for row in records}


def check_reference(result, reference, target_reached=True):
    if reference is None or reference["status"] != "completed" or not target_reached:
        return {"checked": False, "correct": None}
    optimal = [restore_move(record) for record in reference["optimal_moves"]]
    return {"checked": True, "correct": result.value == reference["value"] and result.move in optimal,
            "value_equal": result.value == reference["value"], "move_optimal": result.move in optimal}


def measure_online_step(state, variant, evaluation_factory=None, book_factory=None):
    """Whole-step timer reusable by F's combinations and B's N/D/H experiments."""
    if evaluation_factory is None:
        evaluation_factory = lambda config: resolve_evaluator(config)
    provider = variant.get("book", {}).get("provider")
    if book_factory is None and provider:
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
                budget_ok=elapsed <= variant.get("step_budget", 1.6), hard_limit_ok=elapsed < 2.0)


def measure(positions, configs, depths, evaluation, output, references, *, repeats=3,
            guard_seconds=60, workloads=("cold", "iterative"), online_budget=None,
            diagnostics=True, execution_tier="formal"):
    if repeats != 3:
        raise ValueError("the experiment plan requires exactly three alternating repetitions")
    cfgs = {key: value if isinstance(value, SearchConfig) else SearchConfig(**value)
            for key, value in configs.items()}
    request = {"positions": digest(positions), "configs": cfgs, "depths": depths,
               "evaluation": evaluation, "repeats": repeats, "guard_seconds": guard_seconds,
               "workloads": workloads, "online_budget": online_budget, "diagnostics": diagnostics,
               "code": code_manifest(), "machine": machine_info(), "execution_tier": execution_tier}
    log, records, done = _case_store(output, request)
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
                   "completed_depth": result.completed_depth, "elapsed": result.elapsed,
                   "statistics": result.statistics, "target_reached": reached,
                   "iterations": result.iterations, "solved_remaining": result.solved_remaining}
            row.update(check_reference(result, references.get((position["position_id"], depth)), reached))
        row.update(case_id=case_id, position_id=position["position_id"], variant_id=name,
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
    summary = summarize_measurements(records, list(cfgs), depths, positions)
    atomic_json(Path(output) / "summary.json", summary)
    export_timing_csv(records, Path(output) / "timings.csv")
    return records, summary


def summarize_measurements(records, names, depths, positions):
    timed = [r for r in records if not r["instrumented"] and r["workload"] != "online"]
    common_by_workload = {"cold": [], "iterative": []}
    for workload in common_by_workload:
        compared_names = [name for name in names if not (workload == "iterative" and name == "C0")]
        for position in positions:
            for depth in depths:
                rows = [r for r in timed if r["position_id"] == position["position_id"]
                        and r["target_depth"] == depth and r["workload"] == workload]
                if rows and all(any(r["variant_id"] == name and r["repetition"] == repetition
                                   and r.get("target_reached") and r.get("correct") for r in rows)
                                for name in compared_names for repetition in range(3)):
                    common_by_workload[workload].append([position["position_id"], depth])
    errors = [r["case_id"] for r in records if r.get("checked") and not r.get("correct")]
    missing = [r["case_id"] for r in records if r["workload"] != "online" and not r.get("target_reached")]
    aggregates = []
    for name in names:
        for workload in ("cold", "iterative"):
            for depth in depths:
                selected = [r for r in timed if r["variant_id"] == name and r["workload"] == workload
                            and r["target_depth"] == depth and r.get("target_reached")
                            and [r["position_id"], depth] in common_by_workload[workload]]
                times = [r["elapsed"] for r in selected]
                if times:
                    aggregates.append({"variant_id": name, "workload": workload, "depth": depth,
                        "count": len(times), "median_s": statistics.median(times),
                        "min_s": min(times), "max_s": max(times)})
    return {"correct": not errors, "incorrect_case_ids": errors, "incomplete_case_ids": missing,
            "verification_complete": not missing and all(r.get("checked") for r in timed),
            "common_completed_position_depths": common_by_workload["cold"],
            "common_by_workload": common_by_workload, "aggregates": aggregates,
            "note": "Cross-version timing comparisons must use complete common position-depths; diagnostic rows are separate."}


def export_timing_csv(records, path):
    fields = ["position_id", "variant_id", "workload", "target_depth", "effective_target_depth",
              "repetition", "measurement", "elapsed", "completed_depth", "status", "correct",
              "target_reached", "near_cap", "budget_ok"]
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
        complete = len(rows) == 3 and all(r.get("target_reached") and r.get("correct") for r in rows)
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


def validation_passes(records, refs, positions, depths):
    expected = {(p["position_id"], d) for p in positions for d in depths}
    available = {key for key, ref in refs.items() if ref["status"] == "completed"}
    # No exact reference means unverified, never a passing optimization.
    if not expected.issubset(available):
        return False
    for key in expected:
        rows = [r for r in records if r["position_id"] == key[0] and r["target_depth"] == key[1]
                and r["workload"] == "cold"]
        if not rows or any(not r.get("target_reached") or not r.get("correct") for r in rows):
            return False
    return True


def compare_performance(records, parent, candidate):
    rankings = []
    repetitions = []
    invalid = any(r["variant_id"] == candidate and r["workload"] == "online"
                  and (not r["budget_ok"] or not r["hard_limit_ok"]) for r in records)
    selection_depth = max((r["target_depth"] for r in records if r["workload"] == "iterative"), default=None)
    for repetition in range(3):
        online = {(r["position_id"], r["variant_id"]): r for r in records
                  if r["workload"] == "online" and r["repetition"] == repetition}
        ids = sorted({key[0] for key in online})
        usable = [key for key in ids if (key, parent) in online and (key, candidate) in online
                  and all(not online[(key, name)]["near_cap"] and not online[(key, name)]["solved_remaining"]
                          and online[(key, name)]["completed_depth"] is not None for name in (parent, candidate))]
        means = {name: statistics.mean(online[(key, name)]["completed_depth"] for key in usable)
                 if usable else None for name in (parent, candidate)}
        fixed = {(r["position_id"], r["target_depth"], r["variant_id"]): r for r in records
                 if r["workload"] == "iterative" and not r["instrumented"]
                 and r["repetition"] == repetition and r.get("target_reached")
                 and r["target_depth"] == selection_depth and not r["near_cap"] and not r["solved_remaining"]}
        ratios = [fixed[(key, depth, candidate)]["elapsed"] / fixed[(key, depth, parent)]["elapsed"]
                  for key, depth, name in fixed if name == parent and (key, depth, candidate) in fixed]
        ratio = statistics.median(ratios) if ratios else None
        if means[parent] is None:
            winner = "unresolved"
        elif means[candidate] != means[parent]:
            winner = candidate if means[candidate] > means[parent] else parent
        elif ratio is None or ratio == 1:
            winner = parent
        else:
            winner = candidate if ratio < 1 else parent
        rankings.append(winner)
        repetitions.append({"repetition": repetition, "mean_depth": means,
                            "fixed_time_ratio_candidate_over_parent": ratio, "winner": winner,
                            "depth_position_count": len(usable)})
    retained = not invalid and rankings == [candidate] * 3
    return {"retain_candidate": retained, "budget_valid": not invalid,
            "consistent": len(set(rankings)) == 1, "repetitions": repetitions,
            "reason": "candidate_stably_better" if retained else "keep_parent"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shared fixed-state measurements for A/F/B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    positions, _ = load_positions(config["positions"])
    output = Path(args.output)
    evaluation = config.get("evaluation", {"evaluation_id": "E1"})
    depths = config.get("depths", [2, 3])
    guard = config.get("guard_seconds", 60)
    refs = exact_references(positions, depths, evaluation, output / "references", guard)
    _, summary = measure(positions, config["variants"], depths, evaluation, output / "measurements", refs,
                         guard_seconds=guard, online_budget=config.get("online_budget"))
    print("correct:", summary["correct"])


if __name__ == "__main__":
    main()
