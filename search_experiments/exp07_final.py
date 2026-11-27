"""Test the frozen variants on the final roster and publish the validated submission."""

import argparse
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

from book_experiments.books import book_variant, write_provider
from book_experiments.common import (handoff_identity, load_online, reference_games, validate_empty_book,
                                     validate_match_evidence, validate_selected_exports)
from constants import INIT_TIME_LIMIT, MEMORY_LIMIT_MB, STEP_TIME_LIMIT
from search_experiments.build import build_agent, clear_generated_bytecode
from search_experiments.common import (SCHEMA_VERSION, atomic_json, deduplicate_variants, digest, file_hash,
                                       load_json, normalize_variant)
from search_experiments.development import (Q_VALUES, effective_search_variant, evaluation_key,
                                            evaluation_variant, load_search_selection, match_settings)
from search_experiments.runner import TECHNICAL_REASONS, run_pairs, validate_manifest
from validate_submission import Finding, Report, dynamic_phase, format_report, static_phase


VALIDATOR_RULES = {"allow_moves": True, "win_length": 4}
VALIDATOR_PAIRS = [(0, 1), (0, 2), (1, 1), (1, 2)]
SUBMISSION = "student_agent.py"
BOOK_FIELDS = ("book_file", "book_hash")
CONTROL_FIELDS = ("book_comparison_variant", "book_comparison_file", "book_comparison_hash")


def require_formal_runtime():
    if sys.version_info[:2] != (3, 11) or np.__version__ != "1.26.4":
        raise ValueError("formal Step 7 requires Python 3.11 and NumPy 1.26.4")


def _require_fields(payload, expected, label):
    if any(key not in payload or digest(payload[key]) != digest(value) for key, value in expected.items()):
        raise ValueError(f"{label} differs from the frozen inputs")


def _search_roles(paths, online, identity, settings, smoke):
    search_path = paths["search_selection"]
    search, star, alternate = load_search_selection(search_path, identity["manifest_hash"], smoke)
    if search["execution_tier"] != identity["execution_tier"]:
        raise ValueError("Step 2 and Step 7 require the same execution tier")
    baseline = effective_search_variant(search["variants"]["S0"], settings)
    if (baseline["variant_id"] != "S0_E1_N" or baseline["book"].get("mode") != "N"
            or baseline["book_hash"] is not None or evaluation_key(baseline["evaluation"]) != (1, Q_VALUES[0])
            or baseline["step_budget"] != online["step_budget"]):
        raise ValueError("Step 2 S0 must be the frozen E1/N baseline")
    recheck = load_json(paths["recheck_selection"])
    expected = {key: identity[key] for key in ("code_hash", "manifest_hash", "execution_tier")}
    expected.update(status="smoke_only" if smoke else "selected", formal_selection=not smoke,
                    search_selection_hash=file_hash(search_path), chosen_variant=online,
                    source_search_variant=star, alternate_search_variant=alternate)
    if smoke:
        expected["diagnostic_status"] = "selected"
    _require_fields(recheck, expected, "Step 4 recheck selection")
    if type(recheck["search_switched"]) is not bool:
        raise ValueError("Step 4 must identify its final search branch")
    effective = {role: effective_search_variant(variant, settings)
                 for role, variant in (("S_star", star), ("S_alt", alternate))}
    _require_fields(recheck, {"effective_search_variants": effective}, "Step 4 effective searches")
    branch = "S_alt" if recheck["search_switched"] else "S_star"
    v1_id = recheck["roles"][branch]["E1"]
    v1 = recheck["variants"][v1_id]
    _require_fields(recheck["variants"], {v1_id: effective[branch], online["variant_id"]: online},
                    "Step 4 variant roster")
    if (v1_id != v1["variant_id"] or v1 != effective[branch]
            or evaluation_variant(v1, online["evaluation"]) != online):
        raise ValueError("Step 4 E1 role and online variant must use the same frozen search branch")
    return {"V0": baseline, "V1": v1, "V2": online}, recheck["d1_reference"]


def _book_roles(step5, final, online, identity, dev_rows):
    if step5["status"] == "no_common_book":
        validate_empty_book(step5)
        _require_fields(final, dict.fromkeys((*BOOK_FIELDS, *CONTROL_FIELDS)) |
                        {"status": "N_only", "reason": "no_common_book", "mode": "N",
                         "variant": online, "development_evidence": {}}, "Step 6 N-only selection")
        return None, {}

    if step5["status"] != "selected" or final["status"] != "selected":
        raise ValueError("Step 5 and Step 6 require successful selections")
    main, descriptor, second, second_desc = validate_selected_exports(step5, online)
    exports = {descriptor["book_file"]: (main, descriptor)}
    if second_desc is not None:
        exports[second_desc["book_file"]] = (second, second_desc)

    mode = final["mode"]
    if mode == "N":
        _require_fields(final, dict.fromkeys(BOOK_FIELDS) | {"variant": online}, "Step 6 N submission")
        active = final["book_comparison_variant"]
        active_path, active_hash = final["book_comparison_file"], final["book_comparison_hash"]
    elif mode in ("D", "H"):
        _require_fields(final, dict.fromkeys(CONTROL_FIELDS), "Step 6 book submission")
        active = final["variant"]
        active_path, active_hash = final["book_file"], final["book_hash"]
    else:
        raise ValueError("Step 6 must select an N, D or H submission")
    if not isinstance(active, dict) or active_path not in exports:
        raise ValueError("V3 requires an active book linked to the Step 5 export")
    book, selected = exports[active_path]
    book_mode = active["book"]["mode"]
    if (active_hash != selected["hash"] or book_mode not in ("D", "H")
            or (mode != "N" and mode != book_mode)
            or (active_path != descriptor["book_file"] and (mode != "H" or book_mode != "H"))
            or digest(normalize_variant(book_variant(online, book, book_mode))) != digest(active)):
        raise ValueError("V3 must match the exact exported book and frozen online settings")
    write_provider(book)

    for variant in (online, active) if mode == "N" else (active,):
        saved = final["development_evidence"][variant["variant_id"]]
        _, admitted = validate_match_evidence(saved["games_path"], variant, dev_rows,
                                              identity["opponent"], identity["execution_tier"])
        if digest(saved) != digest(admitted):
            raise ValueError("Step 6 full-development evidence differs from the frozen descriptor")
    return active, {path: desc["book_file_sha256"] for path, (_, desc) in exports.items()}


def _admit_inputs(config, smoke):
    path_keys = ("manifest", "search_selection", "recheck_selection", "online_config", "selected_book", "final_config")
    paths = {key: Path(config[key]).resolve() for key in path_keys}
    tier, opponent, settings = match_settings(config, smoke)
    limit = settings.get("max_configs")
    if smoke and (type(limit) is not int or not 1 <= limit < 12):
        raise ValueError("smoke max_configs must select 1 through 11 configurations per development split")
    if smoke:
        config = dict(config, smoke=settings)
    manifest = load_json(paths["manifest"])
    rows = validate_manifest(manifest)
    d0, d1 = ([row for row in rows if row["split"] == split][:limit] for split in ("D0", "D1"))
    online = load_online(config, smoke)
    identity = handoff_identity(config, online, tier, opponent, manifest["manifest_hash"])
    roles, recheck_reference = _search_roles(paths, online, identity, settings, smoke)
    _, reference = reference_games(recheck_reference["games_path"], online, config, smoke)
    _require_fields(recheck_reference, {key: reference[key] for key in ("games_path", "plan_path", "summary")},
                    "Step 4 D1 reference")
    step5, final = (load_json(paths[key]) for key in ("selected_book", "final_config"))
    _require_fields(step5, dict(identity, formal_selection=not smoke, d1_reference=reference,
                                reference_games=reference["games_path"]), "Step 5 selection")
    _require_fields(final, dict(identity, formal_selection=not smoke, d1_reference=reference,
                               selected_book_path=str(paths["selected_book"]),
                               selected_book_hash=file_hash(paths["selected_book"]),
                               selected_book_status=step5["status"], K=step5["K"]), "Step 6 selection")
    active, book_hashes = _book_roles(step5, final, online, identity, d0 + d1)
    if active is not None:
        roles["V3"] = active
    candidates, aliases = deduplicate_variants([online, roles["V0"], roles["V1"]] +
                                               ([active] if active is not None else []))
    role_candidates = {role: aliases[value["variant_id"]] for role, value in roles.items()}
    submitted = aliases[final["variant"]["variant_id"]]
    for variant in candidates:
        name = variant["variant_id"]
        if Path(name).name != name or name in (".", "..") or "\x00" in name:
            raise ValueError("invalid final candidate directory name")
    match_rows = d0 + d1 if smoke else [row for row in rows if row["split"] == "T"]
    reserved = {row["setup_seed"]: row for row in rows if row["split"] in ("D0", "D1")}
    matches = [{"candidate": variant["variant_id"], "opponent": opponent} for variant in candidates]
    matches += [{"candidate": submitted, "opponent": {"kind": kind, "budget": opponent["budget"]}}
                for kind in ("random", "greedy")]
    search_path = paths["search_selection"]
    inputs = [*paths.values(), search_path.with_name("alternate_search.json")]
    if not smoke:
        inputs.append(search_path.with_name("selected_search.json"))
    return {"stage": "final", **identity, "formal_selection": not smoke,
            "inputs": {str(path): file_hash(path) for path in inputs} | book_hashes,
            "d1_reference": reference, "development_evidence": final["development_evidence"],
            "roles": roles, "role_candidates": role_candidates, "candidates": candidates,
            "v3_omission": {"reason": final["reason"], "K": step5["K"]} if active is None else None,
            "submitted_candidate": submitted, "preflight_rows": [reserved[0], reserved[1]],
            "final_rows": match_rows, "smoke_settings": settings, "matches": matches}


def _candidate_path(output, name):
    return output / "candidates" / name / SUBMISSION


def _check_artifact(path, metadata):
    if (not path.is_file() or file_hash(path) != metadata["artifact_hash"]
            or path.stat().st_size != metadata["bytes"]):
        raise ValueError(f"artifact bytes differ from the frozen build: {path}")


def _copy_artifact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        shutil.copyfile(source, staged)
        staged.replace(target)
    clear_generated_bytecode(target)


def _install_candidate(staged, target, metadata):
    _copy_artifact(staged, target)
    atomic_json(target.with_suffix(".meta.json"), metadata)


def _bind_plan(output, request):
    path = output / "final_plan.json"
    if path.exists():
        plan = load_json(path)
        frozen = plan["request"]
        if (digest(frozen) != plan["request_hash"]
                or digest({key: value for key, value in frozen.items() if key != "artifacts"}) != digest(request)):
            raise ValueError("final request changed; use a new output directory")
        return plan

    with TemporaryDirectory() as temporary:
        staging = Path(temporary)
        artifacts = {}
        for variant in request["candidates"]:
            name = variant["variant_id"]
            _, artifacts[name] = build_agent(variant, _candidate_path(staging, name))
        frozen = dict(request, artifacts=artifacts)
        plan = {"schema": SCHEMA_VERSION, "request_hash": digest(frozen), "request": frozen}
        atomic_json(path, plan)
        for name, metadata in artifacts.items():
            _install_candidate(_candidate_path(staging, name), _candidate_path(output, name), metadata)
    return plan


def _ensure_candidate(output, variant, metadata):
    path = _candidate_path(output, variant["variant_id"])
    meta_path = path.with_suffix(".meta.json")
    # A plan may have been checkpointed just before the artifact was copied.
    if not path.exists() or not meta_path.exists():
        if path.exists():
            _check_artifact(path, metadata)
        with TemporaryDirectory() as temporary:
            built, actual = build_agent(variant, Path(temporary) / SUBMISSION)
            if digest(actual) != digest(metadata):
                raise ValueError("rebuilt artifact differs from the frozen final plan")
            _install_candidate(built, path, metadata)
    _check_artifact(path, metadata)
    if digest(load_json(meta_path)) != digest(metadata):
        raise ValueError(f"artifact metadata differs from the frozen build: {path}")
    return path


def _resource_failure(step, init, rss):
    for label, value, limit, inclusive in (("step", step, STEP_TIME_LIMIT, True),
                                           ("init", init, INIT_TIME_LIMIT, True),
                                           ("RSS", rss, MEMORY_LIMIT_MB, False)):
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or value > limit or (value == limit and not inclusive)):
            return f"{label} limit or missing measurement ({value})"
    return None


def _official_status(report):
    if report["verdict"] != "PASS":
        return "blocked", "official validator failed"
    groups = report["dynamic"]
    if not groups:
        return "pending", "official dynamic validation incomplete"
    if len(groups) != 1 or digest(groups[0]["rules"]) != digest(VALIDATOR_RULES):
        return "blocked", "official validator used unexpected rules"
    result = groups[0]
    games = result["games"]
    if (result["load_error"] is not None or result["forfeits"] != 0 or any(result["strikes"].values())
            or any(game["strikes"] != 0 or game["reason"] in TECHNICAL_REASONS for game in games)):
        return "blocked", "official validator reported a load failure, strike or forfeit"
    seats = [(game["seed"], game["seat"]) for game in games]
    if seats != VALIDATOR_PAIRS[:len(seats)] or any(
            game["outcome"] not in ("W", "D", "L") or type(game["plies"]) is not int or game["plies"] <= 0
            or not game["reason"] for game in games):
        return "blocked", "official validator reported unexpected games"
    if len(games) != len(VALIDATOR_PAIRS):
        return "pending", "official dynamic validation incomplete"
    if any(type(result[outcome]) is not int or result[outcome] != sum(g["outcome"] == outcome for g in games)
           for outcome in ("W", "D", "L")):
        return "blocked", "official validator game and aggregate counts disagree"
    failure = _resource_failure(result["max_step"], result["init_time"], result["max_rss_mb"])
    return ("blocked", failure) if failure else ("passed", None)


def _save_report(path, report, identity, status, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {**identity, "status": status, "reason": reason, "report": report.to_dict()}
    atomic_json(path, saved)
    path.with_suffix(".txt").write_text(format_report(report) + "\n", encoding="utf-8")
    return saved


def _official_gate(path, artifact, binding):
    if path.exists():
        saved = load_json(path)
        _require_fields(saved, binding, "saved official validation")
        result = saved["report"]
        _require_fields(result, {"path": str(artifact)}, "saved official artifact")
        status, reason = _official_status(result)
        saved_verdict = (saved["status"], saved["reason"]) == (status, reason)
        # The validator publishes the aggregate before cleanup and formatting its findings.
        # A load failure is the one ordinary way that batch can finish early.
        groups = result["dynamic"]
        finished_batch = len(groups) == 1 and (groups[0]["load_error"] is not None
                                                or len(groups[0]["games"]) == len(VALIDATOR_PAIRS))
        if status != "pending" and (saved_verdict or finished_batch):
            if not saved_verdict:
                saved.update(status=status, reason=reason)
                atomic_json(path, saved)
            readable = path.with_suffix(".txt")
            if not readable.exists():
                report = Report(result["path"], findings=[Finding(**f) for f in result["findings"]],
                                dynamic=result["dynamic"], class_name=result["class_name"])
                readable.write_text(format_report(report) + "\n", encoding="utf-8")
            return saved
    report = static_phase(str(artifact))
    if report.verdict == "PASS":
        try:
            dynamic_phase(report, 4, dict(VALIDATOR_RULES), in_process=False)
        except (KeyboardInterrupt, SystemExit):
            _save_report(path, report, binding, "pending", "official dynamic validation interrupted")
            raise
        except Exception as error:
            _save_report(path, report, binding, "blocked", f"official dynamic validation error: {error}")
            raise
    status, reason = _official_status(report.to_dict())
    return _save_report(path, report, binding, status, reason)


def _pair_paths(directory):
    return {f"{kind}_path": str(directory / filename) for kind, filename in
            (("plan", "plan.json"), ("games", "games.jsonl"), ("summary", "summary.json"))}


def _strict_preflight(variant, rows, opponent, tier, directory, binding):
    summary = run_pairs(variant, rows, directory, opponent=opponent, strict=True, execution_tier=tier,
                        allow_test=False, expected_candidate_artifact_hash=binding["artifact_hash"])
    result = {**binding, **_pair_paths(directory), "summary": summary}
    if not summary["complete"]:
        return dict(result, status="pending", reason="strict AB preflight incomplete")
    if not summary["eligible"]:
        return dict(result, status="blocked", reason="strict AB preflight ineligible")
    records, evidence = validate_match_evidence(directory / "games.jsonl", variant, rows, opponent, tier)
    for record in records:
        seat = str(record["candidate_seat"])
        referee = record["referee"]
        timing = referee["time"][seat]
        failure = _resource_failure(timing["max"], timing["init"], referee["max_rss_mb"][seat])
        if failure:
            return dict(result, status="blocked", reason=f"{record['game_id']}: {failure}", evidence=evidence)
    return dict(result, status="passed", reason=None, evidence=evidence)


def _prepare_submission(output, source, metadata, binding):
    submission = output / "submission" / SUBMISSION
    if not submission.exists():
        _copy_artifact(source, submission)
    _check_artifact(submission, metadata)
    report = static_phase(str(submission))
    status = "passed" if report.verdict == "PASS" else "blocked"
    reason = None if status == "passed" else "submission copy failed static validation"
    report_path = submission.with_name("static_validation.json")
    _save_report(report_path, report, binding, status, reason)
    _check_artifact(submission, metadata)
    return {**binding, "path": str(submission), "bytes": metadata["bytes"], "status": status, "reason": reason,
            "static_validation_path": str(report_path)}


def run(config, output, smoke=False):
    """Admit frozen inputs, validate exact artifact bytes, then run the fixed final schedule."""
    if not smoke:
        require_formal_runtime()
    output = Path(output).resolve()
    try:
        request = _admit_inputs(config, smoke)
    except (AttributeError, KeyError, OSError, TypeError) as error:
        raise ValueError(f"invalid frozen Step 7 input: {error}") from error
    plan = _bind_plan(output, request)
    request = plan["request"]
    tier = request["execution_tier"]
    submission_name = request["submitted_candidate"]
    submission_meta = request["artifacts"][submission_name]
    variants = {v["variant_id"]: v for v in request["candidates"]}
    summary_path = output / "final_summary.json"
    summary = {"schema": SCHEMA_VERSION, "request_hash": plan["request_hash"], "execution_tier": tier,
               "plan_path": str(output / "final_plan.json"), "role_candidates": request["role_candidates"],
               "v3_omission": request["v3_omission"], "validation": {}, "submission": None, "matches": {}}

    def publish(status, reason=None):
        payload = dict(summary, status=status, reason=reason, formal_selection=not smoke and status == "complete")
        atomic_json(summary_path, payload)
        return payload

    def binding(name):
        return {"request_hash": plan["request_hash"], "execution_tier": tier, "formal_selection": False,
                "variant_id": name, "artifact_hash": request["artifacts"][name]["artifact_hash"]}

    try:
        publish("pending", "validation started")
        for name, variant in variants.items():
            artifact = _ensure_candidate(output, variant, request["artifacts"][name])
            directory = output / "validation" / name
            official_path = directory / "official.json"
            official = _official_gate(official_path, artifact, binding(name))
            summary["validation"][name] = {"official_path": str(official_path),
                                            "official_hash": file_hash(official_path),
                                            "official_status": official["status"]}
            if official["status"] != "passed":
                return publish(official["status"], f"{name}: {official['reason']}")
            _check_artifact(artifact, request["artifacts"][name])
            strict = _strict_preflight(variant, request["preflight_rows"], request["opponent"], tier,
                                       directory / "strict_ab", binding(name))
            atomic_json(directory / "strict_preflight.json", strict)
            summary["validation"][name]["strict_preflight"] = strict
            if strict["status"] != "passed":
                return publish(strict["status"], f"{name}: {strict['reason']}")
            publish("pending", "validation in progress")

        submission = summary["submission"] = _prepare_submission(
            output, _candidate_path(output, submission_name), submission_meta, binding(submission_name))
        if submission["status"] != "passed":
            return publish("blocked", submission["reason"])
        submission_path = Path(submission["path"])
        publish("pending", "final matches in progress")
        for match in request["matches"]:
            _check_artifact(submission_path, submission_meta)
            name, opponent = match["candidate"], match["opponent"]
            directory = output / "matches" / name / opponent["kind"]
            run_pairs(variants[name], request["final_rows"], directory, opponent=opponent, strict=True,
                      execution_tier=tier, allow_test=not smoke,
                      expected_candidate_artifact_hash=request["artifacts"][name]["artifact_hash"])
            _, evidence = validate_match_evidence(directory / "games.jsonl", variants[name], request["final_rows"],
                                                  opponent, tier, require_eligible=False)
            summary["matches"].setdefault(name, {})[opponent["kind"]] = evidence
            publish("pending", "final matches in progress")
        _check_artifact(submission_path, submission_meta)
        results = [match["summary"] for candidate in summary["matches"].values() for match in candidate.values()]
        if not all(result["complete"] for result in results):
            return publish("pending", "final matches incomplete")
        if not all(result["eligible"] for result in results):
            return publish("blocked", "final matches contain technical or candidate-budget failures")
        return publish("complete")
    except (KeyboardInterrupt, SystemExit):
        publish("pending", "final testing interrupted")
        raise
    except Exception as error:
        publish("blocked", f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run(load_json(args.config), args.output, smoke=args.smoke)
    print(summary["execution_tier"], summary["status"], summary["reason"] or "", flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
