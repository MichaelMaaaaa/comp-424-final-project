"""Admit the frozen Step 7 campaign without launching agents or modifying source evidence."""

from pathlib import Path
import hashlib
import json

from book_experiments.books import book_table, book_variant
from book_experiments.common import (code_identity, validate_empty_book, validate_match_evidence,
                                     validate_selected_exports)
from book_experiments.coverage import replay_coverage
from search_experiments.common import (SCHEMA_VERSION, code_manifest, deduplicate_variants, digest, file_hash,
                                       load_json, machine_info, normalize_variant)
from search_experiments.development import match_settings
from search_experiments.exp07_final import _official_status, _resource_failure, _search_roles
from search_experiments.runner import validate_manifest


def _expect(actual, expected, label):
    if digest(actual) != digest(expected):
        raise ValueError(f"{label} differs from frozen Step 7 evidence")


def _fields(record, expected, label):
    _expect({key: record[key] for key in expected}, expected, label)


def _one_input(inputs, field):
    matches = [(path, value) for path, value in inputs.items() if field in value]
    if len(matches) != 1:
        raise ValueError(f"one frozen Step 7 input with {field} is required")
    return matches[0]


def _frozen_roster(req, smoke):
    inputs = {}
    for path, sha in req["inputs"].items():
        _expect(file_hash(path), sha, f"input bytes at {path}")
        inputs[path] = load_json(path)
    _, manifest = _one_input(inputs, "validator_reservations")
    roster = validate_manifest(manifest)
    _expect(manifest["manifest_hash"], req["manifest_hash"], "manifest")
    tier, opponent, settings = match_settings({"opponent":req["opponent"], "smoke":req["smoke_settings"]}, smoke)
    _fields(req, dict(stage="final", execution_tier=tier, formal_selection=not smoke, opponent=opponent,
                      code_hash=code_manifest()["code_hash"], book_code_hash=code_identity()["code_hash"],
                      machine=machine_info(), smoke_settings=settings), "final request")
    limit = settings.get("max_configs")
    if smoke and (type(limit) is not int or not 1 <= limit < 12):
        raise ValueError("invalid smoke final roster size")
    d0, d1 = ([r for r in roster if r["split"] == split][:limit] for split in ("D0", "D1"))
    _expect(req["final_rows"], d0+d1 if smoke else [r for r in roster if r["split"]=="T"], "final roster")
    reserved = {r["setup_seed"]:r for r in roster if r["split"] in ("D0","D1")}
    _expect(req["preflight_rows"], [reserved[0], reserved[1]], "preflight roster")

    online_path = req["online_config_path"]
    online = req["online_variant"]
    _expect(req["inputs"][online_path], req["online_config_hash"], "online input hash")
    _fields(inputs[online_path], dict(variant=online, code_hash=req["code_hash"], execution_tier=tier,
                                      manifest_hash=req["manifest_hash"], formal_selection=not smoke,
                                      status="smoke_only" if smoke else "selected"), "online variant")
    recheck_path, recheck = _one_input(inputs, "search_switched")
    search_path, _ = _one_input(inputs, "S_star")
    paths = {"search_selection":Path(search_path), "recheck_selection":Path(recheck_path)}
    roles, reference = _search_roles(paths, online, req, settings, smoke)
    _fields(req["d1_reference"], {k:reference[k] for k in ("games_path", "plan_path", "summary")}, "D1 reference")

    _, final = _one_input(inputs, "selected_book_path")
    step5_path = final["selected_book_path"]
    step5 = inputs[step5_path]
    identity = {k:req[k] for k in ("execution_tier", "code_hash", "book_code_hash", "manifest_hash", "machine",
                                   "opponent", "online_variant", "online_config_path", "online_config_hash",
                                   "formal_selection", "d1_reference")}
    _fields(final, dict(identity, selected_book_hash=req["inputs"][step5_path],
                        selected_book_status=step5["status"], K=step5["K"],
                        development_evidence=req["development_evidence"]), "Step 6 handoff")
    _fields(step5, identity, "Step 5 handoff")
    book = book_path = None
    if step5["status"] == "no_common_book":
        validate_empty_book(step5)
        _fields(final, dict(reason="no_common_book", status="N_only", mode="N", variant=online), "empty book")
        _expect(req["v3_omission"], {"K":0, "reason":"no_common_book"}, "V3 omission")
    else:
        _fields(step5, {"status":"selected"}, "active book selection")
        _fields(final, {"status":"selected"}, "active book decision")
        main, main_desc, second, second_desc = validate_selected_exports(step5, online)
        exports = {main_desc["book_file"]:main}
        if second_desc:
            exports[second_desc["book_file"]] = second
        mode = final["mode"]
        if mode == "N":
            active, book_path = final["book_comparison_variant"], final["book_comparison_file"]
            book_hash = final["book_comparison_hash"]
            _expect(final["variant"], online, "submitted N")
        elif mode in ("D", "H"):
            active, book_path, book_hash = final["variant"], final["book_file"], final["book_hash"]
            _expect(active["book"]["mode"], mode, "active mode")
        else:
            raise ValueError("unexpected frozen book mode")
        book = exports[book_path]
        _expect(req["inputs"][book_path], file_hash(book_path), "export bytes")
        _expect(book_hash, book["layout"]["hash"], "active book hash")
        if active["book"]["mode"] not in ("D", "H"):
            raise ValueError("invalid active book mode")
        _expect(active, normalize_variant(book_variant(online, book, active["book"]["mode"])), "V3")
        roles["V3"] = active
        _expect(req["v3_omission"], None, "V3 presence")
    candidates, aliases = deduplicate_variants([roles[k] for k in ("V2","V0","V1","V3") if k in roles])
    names = [v["variant_id"] for v in candidates]
    if any(Path(n).name != n or n in (".","..") for n in names):
        raise ValueError("invalid final candidate name")
    submission = aliases[final["variant"]["variant_id"]]
    matches = [dict(candidate=n, opponent=opponent) for n in names]
    matches.extend(dict(candidate=submission, opponent=dict(opponent, kind=k)) for k in ("random", "greedy"))
    _fields(req, dict(roles=roles, role_candidates={r:aliases[v["variant_id"]] for r,v in roles.items()},
                      candidates=candidates, submitted_candidate=submission, matches=matches), "role schedule")
    _expect(sorted(req["artifacts"]), sorted(names), "candidate artifact roster")
    return d1, book, book_path


def _artifact(path, meta):
    _expect(file_hash(path), meta["artifact_hash"], f"artifact at {path}")
    _expect(path.stat().st_size, meta["bytes"], f"artifact size at {path}")


def _match(directory, variant, rows, opp, tier, meta):
    games = directory / "games.jsonl"
    records, evidence = validate_match_evidence(games, variant, rows, opp, tier, require_eligible=False)
    _artifact(directory/"candidate"/"student_agent.py", meta)
    _fields(load_json(evidence["plan_path"])["request"], {"candidate_artifact_hash":meta["artifact_hash"]},
            "pair artifact")
    return records, evidence


def _gates(root, summary, req):
    complete = summary["status"] == "complete"
    validation = summary["validation"]
    if set(validation) - set(req["artifacts"]):
        raise ValueError("unscheduled validation")
    all_passed = True
    for variant in req["candidates"]:
        name = variant["variant_id"]
        meta = req["artifacts"][name]
        binding = dict(variant_id=name, artifact_hash=meta["artifact_hash"], request_hash=summary["request_hash"],
                       execution_tier=req["execution_tier"], formal_selection=False)
        artifact = root/"candidates"/name/"student_agent.py"
        _fields(meta, dict(variant=variant, provenance=code_manifest()), "candidate metadata")
        if artifact.exists() or name in validation or complete:
            _artifact(artifact, meta)
            _expect(load_json(artifact.with_suffix(".meta.json")), meta, "saved build metadata")
        gate = validation.get(name)
        if gate is None:
            all_passed = False
            continue
        directory = root/"validation"/name
        official = directory/"official.json"
        _fields(gate, {"official_path":str(official), "official_hash":file_hash(official)}, "official descriptor")
        saved = load_json(official)
        _fields(saved, dict(binding, status=gate["official_status"]), "official binding")
        _expect(saved["report"]["path"], str(artifact), "official artifact path")
        _expect((saved["status"], saved["reason"]), _official_status(saved["report"]), "official verdict")
        strict = gate.get("strict_preflight")
        if strict is None:
            all_passed = False
            continue
        _expect(saved["status"], "passed", "official precondition")
        _expect(strict, load_json(directory/"strict_preflight.json"), "strict preflight checkpoint")
        _fields(strict, binding, "strict preflight binding")
        records, evidence = _match(directory/"strict_ab", variant, req["preflight_rows"], req["opponent"],
                                   req["execution_tier"], meta)
        _fields(strict, {k:evidence[k] for k in ("plan_path","games_path","summary_path","summary")}, "strict pair")
        status = "pending" if not evidence["summary"]["complete"] else "blocked"
        if evidence["summary"]["eligible"]:
            _expect(strict["evidence"], evidence, "strict eligible evidence")
            status = "passed"
            for record in records:
                seat = str(record["candidate_seat"])
                referee = record["referee"]
                timing = referee["time"][seat]
                if _resource_failure(timing["max"], timing["init"], referee["max_rss_mb"][seat]):
                    status = "blocked"
                    break
        _expect(strict["status"], status, "strict gate status")
        all_passed = all_passed and status == "passed"
    submission = summary["submission"]
    if submission is not None:
        _expect(all_passed, True, "submission preconditions")
        name = req["submitted_candidate"]
        meta = req["artifacts"][name]
        artifact = root/"submission"/"student_agent.py"
        static = artifact.with_name("static_validation.json")
        binding = dict(variant_id=name, artifact_hash=meta["artifact_hash"], request_hash=summary["request_hash"],
                       execution_tier=req["execution_tier"], formal_selection=False)
        _fields(submission, dict(binding, path=str(artifact), bytes=meta["bytes"], static_validation_path=str(static)),
                "submission descriptor")
        _artifact(artifact, meta)
        saved = load_json(static)
        _fields(saved, dict(binding, status=submission["status"], reason=submission["reason"]), "submission validation")
        _expect(saved["report"]["path"], str(artifact), "static artifact path")
        _expect(submission["status"], "passed" if saved["report"]["verdict"] == "PASS" else "blocked", "static status")
    if complete or summary["matches"]:
        if submission is None or submission["status"] != "passed":
            raise ValueError("published final matches require a validated submission")


def admit_campaign(summary_path, smoke):
    path = Path(summary_path).resolve()
    root = path.parent
    summary_bytes = path.read_bytes()
    summary = json.loads(summary_bytes)
    plan_path = root/"final_plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    req = plan["request"]
    tier = "smoke" if smoke else "formal"
    _fields(plan, dict(schema=SCHEMA_VERSION, request_hash=digest(req)), "plan envelope")
    if summary["status"] not in ("pending","blocked","complete"):
        raise ValueError("unknown final campaign status")
    _fields(summary, dict(schema=SCHEMA_VERSION, request_hash=plan["request_hash"], plan_path=str(plan_path),
                          execution_tier=tier, role_candidates=req["role_candidates"], v3_omission=req["v3_omission"],
                          formal_selection=not smoke and summary["status"]=="complete"), "final summary")
    d1, book, book_path = _frozen_roster(req, smoke)
    _gates(root, summary, req)
    expected = {(m["candidate"],m["opponent"]["kind"]) for m in req["matches"]}
    published = summary["matches"]
    if set(published) - set(req["artifacts"]) or any(
            (n,k) not in expected for n in published for k in published[n]):
        raise ValueError("unscheduled final match descriptor")
    groups = []
    for match in req["matches"]:
        name, opponent = match["candidate"], match["opponent"]
        kind = opponent["kind"]
        saved = published.get(name, {}).get(kind)
        if saved is None:
            records, attempts = [], None
        else:
            directory = root/"matches"/name/kind
            meta = req["artifacts"][name]
            records, evidence = _match(directory, meta["variant"], req["final_rows"], opponent, tier, meta)
            _expect(saved, evidence, "final match descriptor")
            # The admission helper verified every strict JSONL attempt, including retriable errors.
            data = Path(evidence["games_path"]).read_bytes()
            _expect(hashlib.sha256(data).hexdigest(), evidence["games_hash"], "admitted attempts")
            attempts = sum(json.loads(line)["status"] == "infrastructure_error" for line in data.splitlines() if line.strip())
        if summary["status"] == "complete" and (saved is None or not saved["summary"]["eligible"]):
            raise ValueError("complete campaign requires every eligible scheduled group")
        groups.append((name, kind, records, attempts, saved))
    if book is None:
        replay = dict(status="unavailable", reason=req["v3_omission"]["reason"], K=0, queryable=None, matched=None, rate=None)
    else:
        ref = req["d1_reference"]
        if not all(Path(ref[k]).is_file() for k in ("games_path","plan_path","summary_path")):
            replay = dict(status="unavailable", reason="missing_d1_reference", queryable=None, matched=None, rate=None)
        else:
            records, evidence = validate_match_evidence(ref["games_path"], req["online_variant"], d1, req["opponent"], tier)
            _expect(evidence, ref, "D1 reference")
            replay = dict(replay_coverage(book_table(book), records), status="available", games=len(records),
                          d1_reference=evidence, book_path=book_path, book_file_sha256=file_hash(book_path),
                          book_hash=book["layout"]["hash"])
    fingerprints = dict(final_summary_path=str(path), final_summary_hash=hashlib.sha256(summary_bytes).hexdigest(),
                        final_plan_path=str(plan_path), final_plan_hash=hashlib.sha256(plan_bytes).hexdigest(),
                        request_hash=plan["request_hash"], inputs=req["inputs"])
    return summary, req, groups, replay, fingerprints
