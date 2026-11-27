"""Sequential S1-S4 retention and full-development match confirmation."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import statistics

from agents.search_engine import SearchConfig
from search_experiments.common import atomic_json, code_manifest, load_json
from search_experiments.measure import (budget_failures, compare_performance, exact_references,
    load_component_references, load_positions, measure, performance_samples, validation_passes)
from search_experiments.runner import paired_difference, run_pairs, validate_manifest

OPTIMIZATIONS = [("S1", {"preferred_first": True}), ("S2", {"reuse_results": True}),
                 ("S3", {"key_mode": "compact"}), ("S4", {"symmetry": True})]


def _relative_to_s0(records, candidate):
    depths, ratios, budget_valid = performance_samples(records, "S0", candidate)
    return {"mean_depth": statistics.mean(row[candidate] for row in depths.values()) if depths else None,
            "fixed_ratio_to_S0": statistics.median(ratios.values()) if ratios else None,
            "depth_samples": {key: row[candidate] for key, row in depths.items()},
            "ratio_samples": ratios, "baseline_budget_valid": budget_valid}


def _match_variant(name, cfg, parent=None):
    return {"variant_id": f"{name}_E1_N", "parent_variant": parent,
            "search": asdict(cfg), "evaluation": {"evaluation_id": "E1"},
            "book": {"mode": "N"}, "step_budget": 1.6, "telemetry": True}


def _confirm_development(selection, variants, development, output, reuse_paths):
    parent = selection["S_diag"]
    summaries = {}
    for name in dict.fromkeys(("S0", parent)):
        summaries[name] = run_pairs(variants[name], development, output / "development" / name,
                                    opponent={"kind": "ab", "budget": 1.5}, reuse_paths=reuse_paths)
    selection["development"] = summaries
    if not all(summary["complete"] for summary in summaries.values()):
        selection.update(S_alt=None, S_alt_config=None, S_alt_reason="development_incomplete")
        return
    valid = sorted((name for name, summary in summaries.items() if summary["eligible"]),
                   key=lambda name: (-summaries[name]["score"], name != "S0"))
    if not valid:
        selection.update(status="no_eligible_development_candidate", S_alt=None, S_alt_config=None,
                         S_alt_reason="no_eligible_development_candidate")
        return
    star = valid[0]
    selection.update(S_star=star, S_star_config=variants[star]["search"], status="selected")
    if parent != "S0":
        alt = valid[1] if len(valid) > 1 else star
        selection.update(S_alt=alt, S_alt_config=variants[alt]["search"],
                         S_alt_reason="other_eligible_development_candidate" if alt != star
                         else "no_eligible_development_alternative")
        if len(valid) == 2:
            selection["paired_difference"] = paired_difference(summaries[parent], summaries["S0"])


def _save_selection(selection, variants, output):
    summary = output / "search_selection.json"
    selected = selection["status"] == "selected"
    if not selected:
        atomic_json(summary, selection)
    for role, filename in (("S_star", "selected_search.json"), ("S_alt", "alternate_search.json")):
        if selection[role] is None:
            (output / filename).unlink(missing_ok=True)
        else:
            atomic_json(output / filename, variants[selection[role]])
    if selected:
        atomic_json(summary, selection)


def run(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = config.get("smoke", {}) if smoke else config
    online_budget = settings.get("online_budget", 1.6)
    if not smoke and online_budget != 1.6:
        raise ValueError("formal step 2 requires an online budget of 1.6 seconds")
    positions, payload = load_positions(config["positions"], settings.get("max_positions") if smoke else None,
                                        require_formal=not smoke)
    depths = settings.get("depths", [2, 3])
    development = None
    if not smoke:
        if not config.get("components_summary"):
            raise ValueError("formal step 2 requires step 1's components_summary")
        components = load_json(config["components_summary"])
        if (not components.get("correct") or not components.get("evidence_sufficient")
                or components.get("execution_tier") != "formal"):
            raise ValueError("fix failing search components before formal step 2")
        if components.get("positions_hash") != payload["positions_hash"]:
            raise ValueError("component and optimization experiments must use the same fixed position set")
        if components.get("code_hash") != code_manifest()["code_hash"]:
            raise ValueError("code changed after component verification; rerun with the frozen version")
        ids = {p["position_id"] for p in positions}
        for workload in ("cold", "iterative"):
            cases = components.get("common_by_workload", {}).get(workload, [])
            if not all(any(pid in ids and depth == d for pid, depth in cases) for d in depths):
                raise ValueError("step 1 lacks common evidence at a requested depth/workload")
    confirm = not smoke and config.get("run_confirmation", True)
    manifest_path = config.get("manifest")
    if manifest_path is not None:
        manifest = load_json(manifest_path)
        rows = validate_manifest(manifest)
        if manifest["manifest_hash"] != payload.get("manifest_hash"):
            raise ValueError("development and reference positions must use the same frozen manifest")
        if confirm:
            development = [row for row in rows if row["split"] in ("D0", "D1")]
    elif confirm:
        raise ValueError("formal step 2 confirmation requires the frozen manifest")
    guard = settings.get("guard_seconds", 60)
    if smoke:
        e1 = exact_references(positions, depths, {"evaluation_id": "E1"}, output / "references_E1", guard)
    else:
        e1 = load_component_references(config["components_summary"], components.get("exact_reference"),
                                       positions, depths)
    refs = {"E1": e1, "E0": exact_references(positions, depths, {"evaluation_id": "E0"},
                                              output / "references_E0", guard)}
    parent, parent_config = "S0", SearchConfig()
    variants = {"S0": _match_variant("S0", parent_config)}
    decisions, candidate_metrics = [], []
    failed_budgets = set()
    for name, change in OPTIMIZATIONS:
        candidate = replace(parent_config, **change)
        compared = {"S0": SearchConfig(), parent: parent_config, name: candidate}
        e0_records, e0_summary = measure(positions, compared, depths, {"evaluation_id": "E0"},
            output / name / "correctness_E0", refs["E0"], guard_seconds=guard,
            workloads=("cold",), diagnostics=False, execution_tier="smoke" if smoke else "formal")
        e1_records, e1_summary = measure(positions, compared, depths, {"evaluation_id": "E1"},
            output / name / "performance_E1", refs["E1"], guard_seconds=guard,
            online_budget=online_budget, execution_tier="smoke" if smoke else "formal")
        correct = (validation_passes(e0_records, refs["E0"], positions, depths, compared, ("cold",))
                   and validation_passes(e1_records, refs["E1"], positions, depths, compared, ("cold", "iterative")))
        failed_budgets.update(budget_failures(e1_records))
        comparison = compare_performance(e1_records, parent, name, baseline="S0")
        keep = correct and comparison["retain_candidate"]
        evidence_fields = ("no_mismatches", "evidence_sufficient", "empty_groups",
                           "incorrect_case_ids", "common_by_workload")
        evidence = {evaluation: {k: v for k, v in summary.items() if k in evidence_fields}
                    for evaluation, summary in (("E0", e0_summary), ("E1", e1_summary))}
        decision = {"variant_id": name, "parent_variant": parent, "changed": change,
                    "search": asdict(candidate), "correct_E0_E1": correct, "evidence": evidence,
                    "retained": keep, "performance": comparison}
        variants[name] = _match_variant(name, candidate, variants[parent]["variant_id"])
        metrics = _relative_to_s0(e1_records, name)
        metrics.update(variant_id=name, correct=correct,
                       budget_valid=comparison["budget_valid"] and metrics["baseline_budget_valid"])
        candidate_metrics.append(metrics)
        decisions.append(decision)
        atomic_json(output / name / "decision.json", decision)
        if keep:
            parent, parent_config = name, candidate
    comparable = [r for r in candidate_metrics if r["correct"] and r["budget_valid"]
                  and r["variant_id"] not in failed_budgets
                  and r["depth_samples"] and r["ratio_samples"]]
    # A diagnostic alternate is ranked on the same observations across candidates.
    for samples_key, metric_key, count_key, aggregate in (
            ("depth_samples", "mean_depth", "common_depth_cases", statistics.mean),
            ("ratio_samples", "fixed_ratio_to_S0", "common_fixed_cases", statistics.median)):
        cases = set.intersection(*(set(r[samples_key]) for r in comparable)) if comparable else set()
        for metrics in candidate_metrics:
            samples = metrics.pop(samples_key)
            used = cases if cases.issubset(samples) else set()
            metrics[metric_key] = aggregate(samples[key] for key in sorted(used)) if used else None
            metrics[count_key] = len(used)
    eligible = [r for r in comparable if r["mean_depth"] is not None and r["fixed_ratio_to_S0"] is not None]
    eligible.sort(key=lambda r: (-r["mean_depth"], r["fixed_ratio_to_S0"], list(variants).index(r["variant_id"])))
    alt = "S0" if parent != "S0" or not eligible else eligible[0]["variant_id"]
    selection = {"execution_tier": "smoke" if smoke else "formal", "positions_hash": payload["positions_hash"],
                 "code_hash": code_manifest()["code_hash"], "manifest_hash": payload.get("manifest_hash"),
                 "variants": variants,
                 "S_diag": parent, "S_diag_config": asdict(parent_config), "S_star": None,
                 "S_alt": alt, "S_alt_config": variants[alt]["search"],
                 "S_alt_reason": ("baseline_for_confirmation" if parent != "S0" else
                                  "qualified_diagnostic_alternative" if eligible else "no_qualified_alternative"),
                 "budget_failures": sorted(failed_budgets),
                 "decisions": decisions, "metrics_relative_to_S0": candidate_metrics,
                 "status": "smoke_only" if smoke else "awaiting_development_confirmation"}
    finalist_budget_failures = failed_budgets.intersection(("S0", parent))
    if not smoke and not all(d["correct_E0_E1"] for d in decisions):
        selection.update(status="blocked_correctness", S_alt=None, S_alt_config=None,
                         S_alt_reason="blocked_correctness")
    elif not smoke and (finalist_budget_failures
                        or not all(d["performance"]["evidence_sufficient"] for d in decisions)):
        selection.update(status="blocked_performance", S_alt=None, S_alt_config=None,
                         S_alt_reason="finalist_budget_failure" if finalist_budget_failures
                         else "missing_performance_evidence")
    elif development is not None:
        reuse_paths = [config["reference_games"]] if config.get("reference_games") else []
        _confirm_development(selection, variants, development, output, reuse_paths)
    _save_selection(selection, variants, output)
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
    if args.manifest is not None:
        config["manifest"] = args.manifest
    selection = run(config, args.output, args.smoke)
    print("status:", selection["status"], "S_diag:", selection["S_diag"], "S_star:", selection["S_star"])


if __name__ == "__main__":
    main()
