"""Sequential S1-S4 retention and full-development match confirmation."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import statistics

from agents.search_engine import SearchConfig
from search_experiments.common import atomic_json, code_manifest, load_json
from search_experiments.measure import (compare_performance, exact_references, load_positions,
                                        measure, validation_passes)
from search_experiments.runner import paired_difference, run_pairs, validate_manifest

OPTIMIZATIONS = [("S1", {"preferred_first": True}), ("S2", {"reuse_results": True}),
                 ("S3", {"key_mode": "compact"}), ("S4", {"symmetry": True})]


def _relative_to_s0(records, candidate):
    online = {(r["position_id"], r["repetition"], r["variant_id"]): r for r in records
              if r["workload"] == "online"}
    samples = {}
    for (position, repetition, name), row in online.items():
        baseline = online.get((position, repetition, "S0"))
        if name == candidate and baseline is not None and all(
                not r["near_cap"] and not r["solved_remaining"] and r["completed_depth"] is not None
                for r in (row, baseline)):
            samples[f"{position}:{repetition}"] = row["completed_depth"]
    selection_depth = max((r["target_depth"] for r in records if r["workload"] == "iterative"), default=None)
    fixed = {(r["position_id"], r["target_depth"], r["repetition"], r["variant_id"]): r
             for r in records if r["workload"] == "iterative" and not r["instrumented"]
             and r.get("target_reached") and r["target_depth"] == selection_depth
             and not r["near_cap"] and not r["solved_remaining"]}
    ratios = [record["elapsed"] / fixed[(position, depth, repetition, "S0")]["elapsed"]
              for (position, depth, repetition, name), record in fixed.items()
              if name == candidate and (position, depth, repetition, "S0") in fixed]
    return {"mean_depth": statistics.mean(samples.values()) if samples else None, "depth_samples": samples,
            "fixed_ratio_to_S0": statistics.median(ratios) if ratios else None}


def _match_variant(name, cfg, parent=None):
    return {"variant_id": f"{name}_E1_N", "parent_variant": parent,
            "search": asdict(cfg), "evaluation": {"evaluation_id": "E1"},
            "book": {"mode": "N"}, "step_budget": 1.6, "telemetry": True}


def run(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = config.get("smoke", {}) if smoke else config
    positions, payload = load_positions(config["positions"], settings.get("max_positions") if smoke else None)
    if not smoke and payload.get("execution_tier") != "formal":
        raise ValueError("formal selection requires formal reference-game positions")
    if not smoke and not payload.get("reference_complete"):
        raise ValueError("formal optimization requires complete reference games")
    if not smoke:
        if not config.get("components_summary"):
            raise ValueError("formal step 2 requires step 1's components_summary")
        components = load_json(config["components_summary"])
        if not components.get("correct") or components.get("execution_tier") != "formal":
            raise ValueError("fix failing search components before formal step 2")
        if components.get("positions_hash") != payload["positions_hash"]:
            raise ValueError("component and optimization experiments must use the same fixed position set")
        if components.get("code_hash") != code_manifest()["code_hash"]:
            raise ValueError("code changed after component verification; rerun with the frozen version")
    depths = settings.get("depths", [2, 3])
    guard = settings.get("guard_seconds", 60)
    online_budget = settings.get("online_budget", 1.6)
    refs = {name: exact_references(positions, depths, {"evaluation_id": name},
                                  output / f"references_{name}", guard) for name in ("E0", "E1")}
    parent, parent_config = "S0", SearchConfig()
    candidates = {"S0": parent_config}
    decisions, candidate_metrics = [], []
    for name, change in OPTIMIZATIONS:
        candidate = replace(parent_config, **change)
        compared = {"S0": SearchConfig(), parent: parent_config, name: candidate}
        e0_records, _ = measure(positions, compared, depths, {"evaluation_id": "E0"},
            output / name / "correctness_E0", refs["E0"], guard_seconds=guard,
            workloads=("cold",), diagnostics=False, execution_tier="smoke" if smoke else "formal")
        e1_records, _ = measure(positions, compared, depths, {"evaluation_id": "E1"},
            output / name / "performance_E1", refs["E1"], guard_seconds=guard,
            online_budget=online_budget, execution_tier="smoke" if smoke else "formal")
        correct = (validation_passes(e0_records, refs["E0"], positions, depths)
                   and validation_passes(e1_records, refs["E1"], positions, depths))
        comparison = compare_performance(e1_records, parent, name)
        keep = correct and comparison["retain_candidate"]
        decision = {"variant_id": name, "parent_variant": parent, "changed": change,
                    "search": asdict(candidate), "correct_E0_E1": correct,
                    "retained": keep, "performance": comparison}
        candidates[name] = candidate
        metrics = _relative_to_s0(e1_records, name)
        candidate_metrics.append(dict(variant_id=name, correct=correct,
                                      budget_valid=comparison["budget_valid"], **metrics))
        decisions.append(decision)
        atomic_json(output / name / "decision.json", decision)
        if keep:
            parent, parent_config = name, candidate
    comparable = [r for r in candidate_metrics if r["correct"] and r["budget_valid"]]
    common_cases = set.intersection(*(set(r["depth_samples"]) for r in comparable)) if comparable else set()
    for metrics in candidate_metrics:
        samples = metrics.pop("depth_samples")
        metrics["mean_depth"] = (statistics.mean(samples[key] for key in common_cases)
                                 if common_cases and common_cases.issubset(samples) else None)
        metrics["common_depth_cases"] = len(common_cases)
    eligible = [r for r in candidate_metrics if r["correct"] and r["budget_valid"]
                and r["mean_depth"] is not None and r["fixed_ratio_to_S0"] is not None]
    eligible.sort(key=lambda r: (-r["mean_depth"], r["fixed_ratio_to_S0"], list(candidates).index(r["variant_id"])))
    alt = "S0" if parent != "S0" else eligible[0]["variant_id"] if eligible else "S0"
    selection = {"execution_tier": "smoke" if smoke else "formal", "positions_hash": payload["positions_hash"],
                 "S_diag": parent, "S_diag_config": asdict(parent_config),
                 "S_star": None, "S_alt": alt, "S_alt_config": asdict(candidates[alt]),
                 "decisions": decisions, "metrics_relative_to_S0": candidate_metrics,
                 "status": "smoke_only" if smoke else "awaiting_development_confirmation"}
    if not smoke and config.get("run_confirmation", True):
        manifest = load_json(config["manifest"])
        development = [r for r in validate_manifest(manifest) if r["split"] in ("D0", "D1")]
        summaries = {}
        for name in dict.fromkeys(("S0", parent)):
            ancestor = next((d["parent_variant"] + "_E1_N" for d in decisions if d["variant_id"] == name), None)
            summaries[name] = run_pairs(_match_variant(name, candidates[name], ancestor), development,
                output / "development" / name, opponent={"kind": "ab", "budget": 1.5},
                reuse_paths=[config["reference_games"]] if config.get("reference_games") else [])
        selection["development"] = summaries
        if all(summary["complete"] for summary in summaries.values()):
            valid = [name for name in summaries if summaries[name]["eligible"]]
            # Score first; an exact tie retains the original, simpler S0.
            valid.sort(key=lambda name: (-summaries[name]["score"], name != "S0"))
            if valid:
                star = valid[0]
                selection.update(S_star=star, S_star_config=asdict(candidates[star]), status="selected")
                if parent != "S0":
                    alt = "S0" if star == parent else parent
                    selection.update(S_alt=alt, S_alt_config=asdict(candidates[alt]))
                if parent != "S0":
                    selection["paired_difference"] = paired_difference(summaries[parent], summaries["S0"])
    atomic_json(output / "search_selection.json", selection)
    if selection["S_star"] is not None:
        atomic_json(output / "selected_search.json", _match_variant(selection["S_star"],
                    SearchConfig(**selection["S_star_config"])))
    atomic_json(output / "alternate_search.json", _match_variant(selection["S_alt"],
                SearchConfig(**selection["S_alt_config"])))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions")
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.positions:
        config["positions"] = args.positions
    if args.manifest:
        config["manifest"] = args.manifest
    selection = run(config, args.output, args.smoke)
    print("status:", selection["status"], "S_diag:", selection["S_diag"], "S_star:", selection["S_star"])


if __name__ == "__main__":
    main()
