"""Screen evaluation parameters on D0, then confirm finalists on D0 and D1."""

import argparse
from pathlib import Path

from search_experiments.common import (atomic_json, code_manifest, deduplicate_variants, file_hash,
                                       load_json, machine_info)
from search_experiments.development import (Q_VALUES, bind_request, effective_search_variant,
    evaluation_key, evaluation_variant, load_development_rows, load_search_selection, match_settings,
    rank_matches, reuse_sources)
from search_experiments.runner import run_pairs

LAMBDA_VALUES = (0, 1, 2, 5, 10)


def _publish(output, selection, status, reason=None):
    smoke = selection["execution_tier"] == "smoke"
    selection.update(status="smoke_only" if smoke else status,
                     formal_selection=not smoke and status == "selected", reason=reason)
    if smoke:
        selection["diagnostic_status"] = status
    summary = output / "evaluation_selection.json"
    if status != "selected":
        atomic_json(summary, selection)
    for role, filename in (("E_star", "selected_evaluation.json"), ("E_alt", "alternate_evaluation.json")):
        path = output / filename
        if status == "selected":
            atomic_json(path, selection["variants"][selection[role]])
        else:
            path.unlink(missing_ok=True)
    if status == "selected":
        atomic_json(summary, selection)
    return selection


def run(config, output, smoke=False):
    tier, opponent, smoke_settings = match_settings(config, smoke)
    manifest_path = Path(config["manifest"]).resolve()
    manifest_hash, d0_rows, _, dev_rows = load_development_rows(manifest_path, smoke_settings.get("max_configs"))
    search_path = Path(config["search_selection"]).resolve()
    upstream, source, alternate = load_search_selection(search_path, manifest_hash, smoke)
    base = effective_search_variant(source, smoke_settings)
    upstream_files = [search_path, search_path.with_name("alternate_search.json")]
    if upstream["execution_tier"] == "formal":
        upstream_files.append(search_path.with_name("selected_search.json"))
    upstream_hashes = {str(path): file_hash(path) for path in upstream_files}
    upstream_games = [search_path.parent / "development" / name / "games.jsonl"
                      for name in upstream.get("development", {})]
    sources = reuse_sources([*config.get("reuse_paths", []), *upstream_games])
    reuse_paths = [Path(entry["path"]) for entry in sources if entry["sha256"] is not None]
    output = Path(output).resolve()
    code_hash = code_manifest()["code_hash"]
    search_hash = upstream_hashes[str(search_path)]
    bind_request(output, {"stage": "evaluation", "execution_tier": tier, "code_hash": code_hash,
                          "machine": machine_info(), "manifest_path": str(manifest_path),
                          "manifest_hash": manifest_hash, "search_selection_files": upstream_hashes,
                          "source_search_variant": source, "alternate_search_variant": alternate,
                          "effective_search_variant": base, "opponent": opponent,
                          "smoke_settings": smoke_settings, "reuse_sources": sources})

    selection = {"execution_tier": tier, "code_hash": code_hash, "manifest_hash": manifest_hash,
                 "source_search_variant": source, "effective_search_variant": base,
                 "search_selection_hash": search_hash, "variants": {}, "stages": {},
                 "d0_games": {}, "development_games": {}, "nonzero_lambdas": [], "best_q": None,
                 "interaction_reason": None, "new_candidate": None, "new_candidate_reason": None,
                 "E_star": None, "E_alt": None, "E_alt_reason": None, "pending_stage": None}
    variants, stages, d0 = selection["variants"], selection["stages"], {}

    def candidate(strength, q=0):
        return evaluation_variant(base, {"lambda": strength, "q": Q_VALUES[q]})

    def screen(stage, candidates):
        summaries = stages[stage] = {}
        for variant in candidates:
            name = variant["variant_id"]
            if name not in d0:
                directory = output / "d0" / name
                d0[name] = run_pairs(variant, d0_rows, directory, opponent=opponent,
                                     execution_tier=tier, reuse_paths=reuse_paths)
                variants[name] = variant
                selection["d0_games"][name] = str(directory / "games.jsonl")
            summaries[name] = d0[name]
        return summaries

    def pending(stage):
        selection["pending_stage"] = stage
        return _publish(output, selection, "pending", f"{stage}_incomplete")

    strengths = {strength: candidate(strength) for strength in LAMBDA_VALUES}
    results = screen("lambda", strengths.values())
    if not all(summary["complete"] for summary in results.values()):
        return pending("lambda")
    nonzero = {variant["variant_id"]: strength for strength, variant in strengths.items() if strength}
    chosen = [nonzero[name] for name in rank_matches(results, nonzero)[:2]]
    selection["nonzero_lambdas"] = chosen
    if len(chosen) < 2:
        return _publish(output, selection, "blocked", "fewer_than_two_eligible_nonzero_lambdas")
    best, second = chosen

    qs = [candidate(best, q) for q in range(len(Q_VALUES))]
    results = screen("q", qs)
    if not all(summary["complete"] for summary in results.values()):
        return pending("q")
    q_by_id = {variant["variant_id"]: q for q, variant in enumerate(qs)}
    best_q = q_by_id[rank_matches(results, q_by_id)[0]]
    selection["best_q"] = f"Q{best_q}"
    selection["interaction_reason"] = "added_second_lambda_best_q" if best_q else "best_q_is_q0"
    results = screen("interaction", (candidate(strength, q) for strength in chosen for q in (0, best_q)))
    if not all(summary["complete"] for summary in results.values()):
        return pending("interaction")

    e1, e0 = strengths[1], strengths[0]
    baseline_keys = {evaluation_key(variant["evaluation"]) for variant in (e1, e0)}
    new_ids = [name for name, variant in variants.items() if evaluation_key(variant["evaluation"]) not in baseline_keys]
    best_new = rank_matches(d0, new_ids)
    finalists = [e1, e0]
    if best_new:
        selection.update(new_candidate=best_new[0], new_candidate_reason="best_eligible_new_d0_evaluation")
        finalists.append(variants[best_new[0]])
    else:
        selection["new_candidate_reason"] = "no_eligible_new_d0_evaluation"
    finalists, _ = deduplicate_variants(finalists)
    confirmation = stages["confirmation"] = {}
    for variant in finalists:
        name = variant["variant_id"]
        directory = output / "development" / name
        d0_log = Path(selection["d0_games"][name])
        confirmation[name] = run_pairs(variant, dev_rows, directory, opponent=opponent,
                                       execution_tier=tier, reuse_paths=[d0_log, *reuse_paths])
        selection["development_games"][name] = str(directory / "games.jsonl")
    if not all(summary["complete"] for summary in confirmation.values()):
        return pending("confirmation")

    ranked = rank_matches(confirmation, confirmation)
    if not ranked:
        return _publish(output, selection, "blocked", "no_eligible_development_candidate")
    star = ranked[0]
    alternate = ranked[1] if len(ranked) > 1 else star
    selection.update(E_star=star, E_alt=alternate,
                     E_alt_reason="other_eligible_development_candidate" if alternate != star
                     else "no_eligible_development_alternative")
    return _publish(output, selection, "selected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    selection = run(load_json(args.config), args.output, args.smoke)
    print(selection["status"], "E*", selection["E_star"], "E_alt", selection["E_alt"])


if __name__ == "__main__":
    main()
