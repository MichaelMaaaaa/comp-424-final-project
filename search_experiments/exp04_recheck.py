"""Recheck search and evaluation combinations, then freeze the D1 no-book reference."""

import argparse
from pathlib import Path

from search_experiments.common import (atomic_json, code_manifest, deduplicate_variants, digest,
                                       file_hash, load_json, machine_info, normalize_variant, read_jsonl)
from search_experiments.development import (bind_request, effective_search_variant, evaluation_key,
    evaluation_variant, load_development_rows, load_search_selection, match_settings, rank_matches, reuse_sources)
from search_experiments.measure import budget_failures, compare_performance, exact_references, load_positions, measure
from search_experiments.runner import run_pairs


def _load_evaluation_selection(path, code_hash, manifest_hash, search_hash, source, effective, smoke, dev_count):
    selection = load_json(path)
    tier = selection.get("execution_tier")
    formal = (tier == "formal" and selection.get("status") == "selected"
              and selection.get("formal_selection") is True)
    provisional = (smoke and tier == "smoke" and selection.get("status") == "smoke_only"
                   and selection.get("diagnostic_status") == "selected" and selection.get("formal_selection") is False)
    if not (formal or provisional):
        raise ValueError("Step 3 requires a selected formal result or an explicitly requested selected smoke result")
    base = source if formal else effective
    if (selection.get("code_hash") != code_hash or selection.get("manifest_hash") != manifest_hash
            or selection.get("search_selection_hash") != search_hash
            or selection.get("source_search_variant") != source or selection.get("effective_search_variant") != base):
        raise ValueError("Step 3 requires current code and the same frozen manifest and Step 2 search")

    variants = selection.get("variants", {})
    confirmation = selection.get("stages", {}).get("confirmation", {})
    count = 48 if formal else dev_count
    roles = {}
    for name, filename in (("E_star", "selected_evaluation.json"), ("E_alt", "alternate_evaluation.json")):
        label = selection.get(name)
        variant = variants.get(label)
        role_path = path.with_name(filename)
        saved = load_json(role_path) if role_path.is_file() else None
        if (not isinstance(variant, dict) or "evaluation" not in variant
                or variant.get("variant_id") != label or saved != variant
                or evaluation_variant(base, variant["evaluation"]) != variant):
            raise ValueError(f"Step 3 selection lacks the original candidate in {filename}")
        summary = confirmation.get(label, {})
        if (summary.get("variant_id") != label or summary.get("code_hash") != code_hash
                or summary.get("execution_tier") != tier or not summary.get("complete")
                or not summary.get("eligible") or summary.get("planned") != count
                or summary.get("adjudicated") != count or (formal and not summary.get("formal_ranking_allowed"))):
            raise ValueError(f"Step 3 {name} requires complete eligible confirmation for current code")
        roles[name] = variant
    return selection, roles


def _publish(output, selection, status, reason=None):
    smoke = selection["execution_tier"] == "smoke"
    selection.update(status="smoke_only" if smoke else status, reason=reason,
                     formal_selection=not smoke and status == "selected")
    if smoke:
        selection["diagnostic_status"] = status
    summary = output / "recheck_selection.json"
    online = output / "online_config.json"
    if status != "selected":
        atomic_json(summary, selection)
        online.unlink(missing_ok=True)
    else:
        wrapper = {key: selection[key] for key in ("status", "execution_tier", "formal_selection",
                                                   "code_hash", "manifest_hash")}
        atomic_json(online, dict(wrapper, variant=selection["chosen_variant"]))
        atomic_json(summary, selection)
    return selection


def _measure_combinations(selection, positions, output, depths, guard, online_budget):
    variants, aliases = deduplicate_variants(selection["variants"].values())
    selection["measurement_roles"] = {
        role: {evaluation: aliases[name] for evaluation, name in names.items()}
        for role, names in selection["roles"].items()}
    groups = {}
    for variant in variants:
        groups.setdefault(evaluation_key(variant["evaluation"]), []).append(variant)
    star_id = selection["roles"]["S_star"]["E_star"]
    star_key = evaluation_key(selection["variants"][star_id]["evaluation"])
    star_records = []
    for key, group in groups.items():
        evaluation = group[0]["evaluation"]
        tag = evaluation["evaluation_id"]
        configs = {variant["variant_id"]: variant["search"] for variant in group}
        references_path, measures_path = output / "references" / tag, output / "measurements" / tag
        refs = exact_references(positions, depths, evaluation, references_path, guard)
        records, summary = measure(positions, configs, depths, evaluation, measures_path, refs,
                                   guard_seconds=guard, online_budget=online_budget,
                                   execution_tier=selection["execution_tier"])
        selection["measurements"][tag] = {"evaluation": evaluation, "variants": list(configs),
            "references_path": str(references_path), "measurements_path": str(measures_path),
            "summary": summary, "budget_failures": sorted(budget_failures(records))}
        if key == star_key:
            star_records = records
    return star_records


def _star_performance(records, positions, current, alternate):
    if current != alternate:
        return compare_performance(records, current, alternate)

    # A shared search needs three actual budget observations per position. It has no speed comparison.
    online = [(row["position_id"], row["repetition"]) for row in records if row["workload"] == "online"
              and row["variant_id"] == current and not row["instrumented"]]
    expected = {(position["position_id"], repeat) for position in positions for repeat in range(3)}
    return {"retain_candidate": False, "budget_valid": current not in budget_failures(records),
            "evidence_sufficient": len(online) == len(expected) and set(online) == expected,
            "reason": "identical_search"}


def _d1_matches_variant(directory, variant, rows, summary, tier, code_hash, manifest_hash):
    plan_path = directory / "plan.json"
    if not plan_path.is_file():
        return False
    plan = load_json(plan_path)
    request = plan["request"]
    request_hash = summary.get("request_hash")
    request_fields = {"variant": variant, "configurations": rows, "execution_tier": tier, "strict": True,
                      "code_hash": code_hash, "manifest_hash": manifest_hash}
    if (digest(request) != request_hash or plan.get("request_hash") != request_hash
            or any(request.get(key) != value for key, value in request_fields.items())):
        return False

    paired_seats = [(f"{variant['variant_id']}__{row['config_id']}__P{seat}", row, seat)
                    for row in rows for seat in (1, 2)]
    if (summary.get("planned") != len(paired_seats) or summary.get("adjudicated") != len(paired_seats)
            or summary.get("variant_id") != variant["variant_id"] or summary.get("execution_tier") != tier
            or summary.get("code_hash") != code_hash
            or plan.get("games") != [{"game_id": name, "config": row, "seat": seat} for name, row, seat in paired_seats]):
        return False

    records = [row for row in read_jsonl(directory / "games.jsonl") if row.get("status") == "adjudicated"]
    by_id = {row["game_id"]: row for row in records}
    if len(records) != len(paired_seats) or set(by_id) != {name for name, _, _ in paired_seats}:
        return False
    identity = {key: variant[key] for key in ("variant_id", "parent_variant", "config_hash", "book_hash")}
    identity.update(request_hash=request_hash, manifest_hash=manifest_hash, code_hash=code_hash,
                    execution_tier=tier, strict=True, candidate_artifact_hash=request["candidate_artifact_hash"],
                    candidate_source_hash=request["candidate_source_hash"])
    for name, row, seat in paired_seats:
        expected = {key: row[key] for key in ("config_id", "split", "setup_seed", "obstacle", "removed_size")}
        expected.update(identity, candidate_seat=seat)
        if any(by_id[name].get(key) != value for key, value in expected.items()):
            return False
    return True


def run(config, output, smoke=False):
    tier, opponent, smoke_settings = match_settings(config, smoke)
    manifest_path = Path(config["manifest"]).resolve()
    search_path = Path(config["search_selection"]).resolve()
    eval_path = Path(config["evaluation_selection"]).resolve()
    positions_path = Path(config["positions"]).resolve()
    manifest_hash, _, d1_rows, dev_rows = load_development_rows(manifest_path, smoke_settings.get("max_configs"))
    search, source, alternate = load_search_selection(search_path, manifest_hash, smoke)
    selected = effective_search_variant(source, smoke_settings)
    alternate_effective = effective_search_variant(alternate, smoke_settings)
    code_hash, search_hash = code_manifest()["code_hash"], file_hash(search_path)
    evaluation, evaluation_roles = _load_evaluation_selection(
        eval_path, code_hash, manifest_hash, search_hash, source, selected, smoke, 2 * len(dev_rows))

    settings = smoke_settings if smoke else config
    depths = settings.get("depths", [1] if smoke else [2, 3])
    guard = settings.get("guard_seconds", 2 if smoke else 60)
    online_budget = settings.get("online_budget", 0.03 if smoke else 1.6)
    if not smoke and online_budget != 1.6:
        raise ValueError("formal step 4 requires an online budget of 1.6 seconds")
    limit = settings.get("max_positions", 1) if smoke else None
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("smoke max_positions must be a positive integer")
    positions, position_set = load_positions(positions_path, limit, require_formal=not smoke)
    if (position_set.get("manifest_hash") != manifest_hash
            or position_set["positions_hash"] != search.get("positions_hash")):
        raise ValueError("Step 4 requires Step 2's fixed positions and the same frozen manifest")

    input_paths = [search_path, search_path.with_name("alternate_search.json"), eval_path,
                   eval_path.with_name("selected_evaluation.json"), eval_path.with_name("alternate_evaluation.json"),
                   positions_path]
    if search["execution_tier"] == "formal":
        input_paths.append(search_path.with_name("selected_search.json"))
    input_hashes = {str(path): file_hash(path) for path in input_paths}
    upstream_games = [search_path.parent / "development" / name / "games.jsonl"
                      for name in search.get("development", {})]
    sources = reuse_sources([*config.get("reuse_paths", []), *upstream_games,
                             *evaluation.get("development_games", {}).values(),
                             *evaluation.get("d0_games", {}).values()])
    reuse_paths = [Path(item["path"]) for item in sources if item["sha256"] is not None]
    output = Path(output).resolve()
    bind_request(output, {"stage": "recheck", "code_hash": code_hash, "execution_tier": tier,
                          "machine": machine_info(), "manifest_path": str(manifest_path),
                          "manifest_hash": manifest_hash, "input_files": input_hashes,
                          "selected_search": selected, "alternate_search": alternate_effective,
                          "evaluation_roles": evaluation_roles, "smoke_settings": smoke_settings,
                          "measurement_settings": {"depths": depths, "guard_seconds": guard,
                                                   "online_budget": online_budget, "max_positions": limit},
                          "opponent": opponent, "reuse_sources": sources})

    selection = {"execution_tier": tier, "code_hash": code_hash, "manifest_hash": manifest_hash,
                 "search_selection_hash": search_hash, "evaluation_selection_hash": input_hashes[str(eval_path)],
                 "positions_hash": position_set["positions_hash"], "source_search_variant": source,
                 "alternate_search_variant": alternate, "effective_search_variants": {"S_star": selected,
                     "S_alt": alternate_effective}, "upstream_evaluation_roles": evaluation_roles,
                 "variants": {}, "roles": {}, "measurements": {}, "performance": None, "stages": {},
                 "search_switched": False, "search_switch_reason": None,
                 "evaluation_recheck_reason": "search_unchanged", "evaluation_recheck_roles": None,
                 "chosen_variant": None,
                 "d1_reference": None, "pending_stage": None}
    variants = selection["variants"]
    for role, base in selection["effective_search_variants"].items():
        selection["roles"][role] = {}
        for evaluation_role, evaluator in (("E1", base["evaluation"]), ("E_star", evaluation_roles["E_star"]["evaluation"])):
            candidate = evaluation_variant(base, evaluator)
            name = candidate["variant_id"]
            if name in variants and variants[name] != candidate:
                raise ValueError(f"conflicting definitions for variant {name}")
            variants[name] = candidate
            selection["roles"][role][evaluation_role] = name

    records = _measure_combinations(selection, positions, output, depths, guard, online_budget)
    if not all(group["summary"]["correct"] and group["summary"]["evidence_sufficient"]
               for group in selection["measurements"].values()):
        return _publish(output, selection, "blocked", "combination_correctness_failed")
    current, other = (selection["roles"][role]["E_star"] for role in ("S_star", "S_alt"))
    current_case, other_case = (selection["measurement_roles"][role]["E_star"] for role in ("S_star", "S_alt"))
    performance = selection["performance"] = _star_performance(records, positions, current_case, other_case)
    selection["search_switch_reason"] = performance["reason"]
    if not performance["budget_valid"]:
        return _publish(output, selection, "blocked", "star_evaluation_budget_failure")
    if not performance["evidence_sufficient"]:
        return _publish(output, selection, "blocked", "star_evaluation_performance_incomplete")

    chosen = current
    development_summaries, development_logs = {}, {}

    def confirm(stage, names):
        summaries = selection["stages"][stage] = {}
        for name in names:
            if name not in development_summaries:
                directory = output / "development" / name
                development_summaries[name] = run_pairs(variants[name], dev_rows, directory,
                    opponent=opponent, execution_tier=tier, reuse_paths=reuse_paths)
                development_logs[name] = directory / "games.jsonl"
            summaries[name] = development_summaries[name]
        if not all(summary["complete"] for summary in summaries.values()):
            selection["pending_stage"] = stage
            return None
        return rank_matches(summaries, names)

    if performance["retain_candidate"]:
        stage = "search_confirmation"
        ranked = confirm(stage, [current, other])
        if not ranked:
            selection["search_switch_reason"] = f"{stage}_incomplete" if ranked is None else "no_eligible_search"
            return _publish(output, selection, "pending" if ranked is None else "blocked",
                            selection["search_switch_reason"])
        chosen = ranked[0]
        selection["search_switched"] = chosen == other
        selection["search_switch_reason"] = "alternate_won_development" if chosen == other else "current_kept_on_development"

    if selection["search_switched"]:
        stage = "evaluation_recheck"
        candidate = evaluation_variant(alternate_effective, evaluation_roles["E_alt"]["evaluation"])
        candidate_id = candidate["variant_id"]
        variants[candidate_id] = candidate
        selection["evaluation_recheck_roles"] = {"E_star": chosen, "E_alt": candidate_id}
        tie_order = {evaluation_key(variant["evaluation"]): index
                     for index, variant in enumerate(evaluation["variants"].values())}
        baseline_order = {evaluation_key({"evaluation_id": "E1"}): -2,
                          evaluation_key({"evaluation_id": "E0"}): -1}
        tie_order.update(baseline_order)
        names = sorted(dict.fromkeys((chosen, candidate_id)),
                       key=lambda name: tie_order[evaluation_key(variants[name]["evaluation"])])
        ranked = confirm(stage, names)
        if not ranked:
            selection["evaluation_recheck_reason"] = f"{stage}_incomplete" if ranked is None else "no_eligible_evaluation"
            return _publish(output, selection, "pending" if ranked is None else "blocked",
                            selection["evaluation_recheck_reason"])
        if len(names) == 1:
            selection["evaluation_recheck_reason"] = "identical_evaluation"
        elif ranked[0] == candidate_id:
            selection["evaluation_recheck_reason"] = "alternate_evaluation_won_development"
        else:
            selection["evaluation_recheck_reason"] = "star_evaluation_kept_on_development"
        chosen = ranked[0]

    variant = selection["chosen_variant"] = normalize_variant(variants[chosen])
    d1_dir = output / "d1_reference"
    matches = run_pairs(variant, d1_rows, d1_dir, opponent=opponent, execution_tier=tier,
                        reuse_paths=list(dict.fromkeys([*development_logs.values(), *reuse_paths])))
    selection["d1_reference"] = {"summary": matches, "plan_path": str(d1_dir / "plan.json"),
                                 "games_path": str(d1_dir / "games.jsonl")}
    if not matches["complete"]:
        selection["pending_stage"] = "d1_reference"
        return _publish(output, selection, "pending", "d1_reference_incomplete")
    if not matches["eligible"]:
        return _publish(output, selection, "blocked", "d1_reference_ineligible")
    if not _d1_matches_variant(d1_dir, variant, d1_rows, matches, tier, code_hash, manifest_hash):
        return _publish(output, selection, "blocked", "d1_reference_identity_mismatch")
    return _publish(output, selection, "selected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    selection = run(load_json(args.config), args.output, args.smoke)
    chosen = selection["chosen_variant"]
    print(selection["status"], "candidate", chosen["variant_id"] if chosen else None)


if __name__ == "__main__":
    main()
