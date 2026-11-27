"""Validate three representative single-file agents on reserved development openings."""
import argparse
from importlib.metadata import version
import math
from pathlib import Path
import platform
import subprocess
import sys

from book_experiments.books import book_variant, load_current_book, make_book, write_provider
from book_experiments.common import code_identity, validate_book_file
from search_experiments import exp07_final as final
from search_experiments.common import (OFFICIAL_RULES, REPO_ROOT, SOURCE_COMMIT, atomic_json, digest,
                                       file_hash, load_json, machine_info, normalize_variant, read_jsonl)
from search_experiments.reporting import candidate_start_resources
from search_experiments.runner import validate_manifest

PREFLIGHT = {"purpose": "preflight", "candidate_ranking_allowed": False, "formal_selection": False}


def runtime():
    return {"python": platform.python_version(), "numpy": version("numpy"), "matplotlib": version("matplotlib"),
            "executable": sys.executable}


def _request(config, root):
    if set(config) - {"manifest", "book"}:
        raise ValueError("acceptance takes only manifest and optional book inputs")
    environment = runtime()
    if (environment["python"].split(".")[:2] != ["3", "11"] or environment["numpy"] != "1.26.4"
            or environment["matplotlib"] != "3.8.4"):
        raise ValueError("preflight requires Python 3.11, NumPy 1.26.4 and matplotlib 3.8.4")
    manifest_path = Path(config["manifest"]).resolve()
    manifest = load_json(manifest_path)
    rows = validate_manifest(manifest)
    reservations = [next(r for r in rows if all(r[k] == v for k, v in wanted.items()))
                    for wanted in manifest["validator_reservations"]]
    if [(r["setup_seed"], r["split"]) for r in reservations] != [(0, "D0"), (1, "D1")]:
        raise ValueError("official development reservations are required")
    supplied = config.get("book")
    source = Path(supplied).resolve() if supplied else REPO_ROOT / "agents/student_agent.py"
    book_source = {"path": str(source), "sha256": file_hash(source),
                   "kind": "supplied_export" if supplied else "original_embedded"}
    label = "preflight_" + digest([str(root), book_source])[:16]
    if supplied:
        record = validate_book_file(source)
        book = dict(record, label=label, provider="agents.books." + label)
    else:
        entries, book_source["original_version"] = load_current_book(source)
        book = make_book(entries, label)
    if not book["entries"]:
        raise ValueError("representative D/H preflight needs a nonempty book")
    baseline = normalize_variant(dict(variant_id="preflight_S0_E1_N", step_budget=1.6, max_depth=None))
    candidates = [baseline]
    for mode in "DH":
        candidates.append(normalize_variant(dict(book_variant(baseline, book, mode), variant_id="preflight_S0_E1_" + mode)))
    head = None
    if (REPO_ROOT / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    request = dict(PREFLIGHT, stage="representative_acceptance", execution_tier="formal", output=str(root),
                   runtime=environment, machine=machine_info(), sources=code_identity(), source_commit=SOURCE_COMMIT,
                   checkout_commit=head, manifest_path=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                   manifest=manifest, official_rows=reservations, strict_rows=[reservations[0]],
                   rules=OFFICIAL_RULES, opponent={"kind": "ab", "budget": 1.5},
                   book_source=book_source, representative_book_hash=digest(book), book_layout=book["layout"],
                   candidates=candidates)
    return request, book


def _strict_resources(path):
    samples = []
    for row in read_jsonl(path):
        if row.get("status") != "adjudicated":
            continue
        seat = row["candidate_seat"]
        init, rss = candidate_start_resources(row["referee"], seat)
        samples.append({"game_id": row["game_id"], "candidate_seat": seat,
                        "outcome": row["outcome"], "max_step": row.get("candidate_max_step_s"),
                        "init_time": init, "max_rss_mb": rss})
    maxima = {}
    for key in ("max_step", "init_time", "max_rss_mb"):
        values = [s[key] for s in samples]
        valid = values and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)
        maxima[key] = max(values) if valid else None
    return {"adjudications": samples, "candidate_maxima": maxima}


def _official_resources(groups):
    if not groups:
        return None
    group = groups[0]
    result = {k: group.get(k) for k in ("max_step", "init_time", "max_rss_mb", "mean_step")}
    # Seat 1 takes the extra ply in odd-length games. Only non-pass candidate plies have timings.
    candidate_plies = sum((game["plies"] + (game["seat"] == 1)) // 2 for game in group.get("games", []))
    passes = group.get("passes")
    if passes is None or candidate_plies <= passes:
        result.update(max_step=None, mean_step=None)
    return result


def run(config, output):
    root = Path(output).resolve()
    request, book = _request(config, root)
    plan_path = root / "final_plan.json"
    summary_path = root / "acceptance_summary.json"
    if root.exists() and any(root.iterdir()) and not plan_path.exists():
        raise ValueError("unbound preflight directory; use a new output directory")
    # Refuse changed inputs before recreating even a generated provider.
    if not plan_path.exists():
        write_provider(book)
    plan = final._bind_plan(root, request)
    frozen = plan["request"]
    book_path = root / "representative_book.json"
    summary = dict(PREFLIGHT, status="pending", phase="initializing", execution_tier="formal", plan_path=str(plan_path),
                   request_hash=plan["request_hash"], sources=request["sources"], source_commit=SOURCE_COMMIT,
                   checkout_commit=request["checkout_commit"], manifest_hash=request["manifest"]["manifest_hash"],
                   runtime=request["runtime"], machine=request["machine"], book_source=request["book_source"],
                   book_layout=book["layout"], representative_book_path=str(book_path), artifacts={})
    for candidate in frozen["candidates"]:
        name = candidate["variant_id"]
        mode = candidate["book"]["mode"]
        metadata = frozen["artifacts"][name]
        summary["artifacts"][mode] = dict(variant=candidate, path=str(final._candidate_path(root, name)),
            artifact_hash=metadata["artifact_hash"], bytes=metadata["bytes"], static=None, official=None, strict=None)

    def publish(status, reason):
        summary.update(status=status, reason=reason)
        atomic_json(summary_path, summary)
        return summary

    previous = load_json(summary_path) if summary_path.exists() else None
    if previous is not None and previous["request_hash"] != plan["request_hash"]:
        raise ValueError("acceptance summary differs from the bound request; use a new output directory")
    if previous is not None:
        # Legacy summaries cannot prove that initialization was interrupted before a gate.
        summary["phase"] = "initializing" if previous.get("phase") == "initializing" else "gates_started"
        for mode, entry in summary["artifacts"].items():
            for key in ("static", "official", "strict"):
                entry[key] = previous["artifacts"][mode][key]
                if entry[key] is not None:
                    summary["phase"] = "gates_started"
    try:
        initializing = summary["phase"] == "initializing"
        if book_path.exists() or not initializing:
            if not book_path.is_file() or digest(load_json(book_path)) != digest(book):
                raise ValueError("saved representative book differs from preflight binding")
        else:
            atomic_json(book_path, book)
        write_provider(book)
        # A crash during initial installation can leave a plan before some candidates exist.
        # Once gates start, missing or changed artifacts invalidate the recorded evidence.
        for entry in summary["artifacts"].values():
            candidate = entry["variant"]
            metadata = frozen["artifacts"][candidate["variant_id"]]
            artifact = Path(entry["path"])
            meta_path = artifact.with_suffix(".meta.json")
            if not initializing:
                final._check_artifact(artifact, metadata)
            if meta_path.exists() or not initializing:
                if not meta_path.is_file() or digest(load_json(meta_path)) != digest(metadata):
                    raise ValueError(f"artifact metadata differs from the frozen build: {artifact}")
            final._ensure_candidate(root, candidate, metadata)
        for mode, entry in summary["artifacts"].items():
            for key in ("static", "official", "strict"):
                prior = entry[key]
                if prior and prior["status"] != "pending":
                    for path, sha in prior["file_hashes"].items():
                        if file_hash(path) != sha:
                            raise ValueError(f"saved {mode} {key} evidence changed")
        # Persist the boundary before even the first static call can be interrupted.
        summary["phase"] = "gates_started"
        publish("pending", "representative validation started")
        for mode, entry in summary["artifacts"].items():
            candidate = entry["variant"]
            artifact = Path(entry["path"])
            name = candidate["variant_id"]
            binding = dict(PREFLIGHT, request_hash=plan["request_hash"], execution_tier="formal",
                           variant_id=name, artifact_hash=entry["artifact_hash"])
            directory = root / "validation" / mode
            static_path = directory / "static.json"
            if entry["static"] is None:
                static_report = final.static_phase(str(artifact))
                status = "passed" if static_report.verdict == "PASS" else "blocked"
                saved = final._save_report(static_path, static_report, binding, status,
                                           None if status == "passed" else "static validation failed")
                entry["static"] = dict(status=status, reason=saved["reason"], verdict=static_report.verdict,
                                       path=str(static_path), file_hashes={str(static_path): file_hash(static_path)})
            if entry["static"]["status"] != "passed":
                return publish("blocked", f"{mode}: static validation failed")
            official_path = directory / "official.json"
            official = final._official_gate(official_path, artifact, binding)
            measurements = _official_resources(official["report"]["dynamic"])
            entry["official"] = dict(status=official["status"], reason=official["reason"], path=str(official_path),
                                     verdict=official["report"]["verdict"], measurements=measurements,
                                     file_hashes={str(official_path): file_hash(official_path)})
            if official["status"] != "passed":
                return publish(official["status"], f"{mode}: {official['reason']}")
            publish("pending", f"{mode}: strict AB preflight")
            final._check_artifact(artifact, frozen["artifacts"][name])
            strict = final._strict_preflight(candidate, frozen["strict_rows"], frozen["opponent"], "formal",
                                             directory / "strict_ab", binding)
            strict_path = directory / "strict_preflight.json"
            atomic_json(strict_path, strict)
            paths = [strict_path, *[Path(strict[k]) for k in ("plan_path", "games_path", "summary_path") if k in strict]]
            entry["strict"] = dict(status=strict["status"], reason=strict["reason"], path=str(strict_path),
                                   games_path=strict["games_path"], summary=strict["summary"],
                                   resources=_strict_resources(strict["games_path"]),
                                   file_hashes={str(p): file_hash(p) for p in paths if p.is_file()})
            if strict["status"] != "passed":
                return publish(strict["status"], f"{mode}: {strict['reason']}")
            final._check_artifact(artifact, frozen["artifacts"][name])
            publish("pending", f"{mode}: gates passed")
        return publish("passed", None)
    except (KeyboardInterrupt, SystemExit):
        publish("pending", "representative validation interrupted")
        raise
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        return publish("blocked", f"{type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--book")
    args = parser.parse_args()
    result = run({"manifest": args.manifest, "book": args.book}, args.output)
    print(result["status"], result["reason"] or "", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
