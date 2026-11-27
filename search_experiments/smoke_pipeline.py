"""Run Steps 0–8 serially on one diagnostic development configuration per split."""
import argparse
import json
from pathlib import Path

from agents.book_policy import book_build_hook
from agents.game_state import initial_state, legal_moves
from agents.symmetry import canonicalize, decode_book_state
from book_experiments import exp05_book, exp06_book_mode
from book_experiments.books import book_variant, make_book
from book_experiments.common import code_identity, validate_match_evidence
from search_experiments import (exp00_prepare, exp01_components, exp02_search, exp03_evaluation,
                                exp04_recheck, exp07_final, exp08_report)
from search_experiments.common import (REPO_ROOT, atomic_json, digest, file_hash, load_json,
                                       machine_info, normalize_variant)
from search_experiments.development import bind_request, effective_search_variant, load_search_selection
from search_experiments.measure import measure_online_step
from search_experiments.runner import make_manifest, validate_manifest


STAGES = {
    "prepare": (exp00_prepare.prepare, "preparation.json"),
    "components": (exp01_components.run, "components_summary.json"),
    "search": (exp02_search.run, "search_selection.json"),
    "evaluation": (exp03_evaluation.run, "evaluation_selection.json"),
    "recheck": (exp04_recheck.run, "recheck_selection.json"),
    "book": (exp05_book.run, "selected_book.json"),
    "book_mode": (exp06_book_mode.run, "final_config.json"),
    "final": (exp07_final.run, "final_summary.json"),
    "report": (exp08_report.run, "report_summary.json"),
}
PROFILE = REPO_ROOT / "search_experiments/configs/smoke_pipeline.json"


def _configs(profile, root):
    inputs = {"manifest": "prepare/manifest.json", "positions": "prepare/positions.json",
              "reference_games": "prepare/reference/games.jsonl", "components_summary": "components/components_summary.json",
              "search_selection": "search/search_selection.json", "evaluation_selection": "evaluation/evaluation_selection.json",
              "online_config": "recheck/online_config.json", "recheck_selection": "recheck/recheck_selection.json",
              "d1_reference_games": "recheck/d1_reference/games.jsonl", "selected_book": "book/selected_book.json",
              "final_config": "book_mode/final_config.json", "final_summary": "final/final_summary.json"}
    configs = {}
    for name in STAGES:
        package = "book_experiments" if name in ("book", "book_mode") else "search_experiments"
        cfg = load_json(REPO_ROOT / package / "configs" / (name + ".json"))
        cfg.update({key: str(root / path) for key, path in inputs.items() if key in cfg})
        if "reuse_paths" in cfg:
            cfg["reuse_paths"] = [str(root / inputs["reference_games"])]
        if name != "report":
            cfg["smoke"] = dict(profile["match"])
        if name in ("components", "search", "recheck"):
            cfg["smoke"].update(profile["diagnostics"])
        if name == "prepare":
            cfg.update(manifest_seed=profile["manifest_seed"], sample_seed=profile["sample_seed"], run_reference=True)
        if name == "book":
            label = "smoke_" + digest([str(root), profile])[:16]
            cfg["smoke"].update(profile["book"], book_label=label)
        if name == "book_mode":
            cfg["smoke"]["overhead"] = profile["overhead"]
        if name == "report":
            cfg["diagnostic_dirs"] = [{"directory": str(root / "components/components"),
                                       "positions": str(root / inputs["positions"]),
                                       "max_positions": profile["diagnostics"]["max_positions"]}]
        configs[name] = cfg
    return configs


def controlled_probe(directory):
    row = next(r for r in make_manifest()["rows"] if r["setup_seed"] == 0 and r["split"] == "D0")
    state = initial_state(tuple(row["obstacle"]), row["removed_size"])
    key, _ = canonicalize(state)
    move = legal_moves(decode_book_state(key))[-1]
    book = make_book([(key, move)], "probe")
    base = normalize_variant(dict(variant_id="controlled_S0_E1_N", max_depth=1, step_budget=0.1, telemetry=False))
    modes = {}
    for mode in "NDH":
        variant = base if mode == "N" else normalize_variant(dict(book_variant(base, book, mode), telemetry=False))
        def factory(config, candidate):
            if mode == "N":
                raise ValueError("N constructed a book hook")
            return book_build_hook(book["data"], book["layout"], config, candidate)
        measured = measure_online_step(state, variant, book_factory=factory)
        action = {"N": "none", "D": "direct", "H": "hint"}[mode]
        if (measured["move"] not in legal_moves(state) or measured["book_action"] != action
                or measured["book_queried"] is not (mode != "N") or measured["book_matched"] is not (mode != "N")
                or (mode == "D" and measured["completed_depth"] is not None)):
            raise ValueError(f"controlled {mode} book action failed")
        modes[mode] = {key: measured[key] for key in
                       ("book_queried", "book_matched", "book_action", "completed_depth", "move")}
        modes[mode]["book_hash"] = variant["book_hash"]
    result = dict(purpose="controlled_probe", config_id=row["config_id"], split="D0",
                  canonical_key=list(key), canonical_move=move, modes=modes)
    atomic_json(Path(directory) / "controlled_probe.json", result)
    return result


def _execute(name, config, directory):
    STAGES[name][0](config, directory, smoke=True)


def _check_stage(name, config, directory):
    path = directory / STAGES[name][1]
    data = load_json(path)
    if data.get("execution_tier") != "smoke":
        raise ValueError(f"{name} requires smoke-tier output")
    if name not in ("prepare", "components", "search"):
        field = "formal_results" if name == "report" else "formal_selection"
        if data.get(field) is not False:
            raise ValueError(f"{name} lacks an explicit non-formal result")
    status = data.get("diagnostic_status", data.get("status"))
    if status in ("pending", "blocked"):
        return status, data.get("reason")
    if name == "prepare":
        if not data["reference_complete"]:
            return "pending", "reference incomplete"
        if not data["reference_eligible"] or data["positions_count"] < 1:
            return "blocked", "ineligible reference or empty position sample"
        rows = validate_manifest(load_json(directory / "manifest.json"))
        row = next(r for r in rows if r["split"] == "D0")
        settings = config["smoke"]
        variant = normalize_variant(dict(variant_id="S0_E1_N", step_budget=settings["step_budget"],
                                         max_depth=settings["max_depth"]))
        validate_match_evidence(directory / "reference/games.jsonl", variant, [row],
                                {"kind": "ab", "budget": settings["opponent_budget"]}, "smoke")
    elif name == "components":
        if not (data["correct"] and data["verification_complete"] and data["exact_reference_incomplete"] == 0):
            return "blocked", "component verification incomplete or incorrect"
    elif name == "search":
        _, *roles = load_search_selection(path, load_json(config["manifest"])["manifest_hash"], smoke=True)
        decisions = data["decisions"]
        if (data["S_star"] is not None or [d["variant_id"] for d in decisions] != ["S1", "S2", "S3", "S4"]
                or not all(d["correct_E0_E1"] for d in decisions)):
            return "blocked", "search diagnostic handoff lacks complete correctness evidence"
        for role in roles:
            variant = effective_search_variant(role, config["smoke"])
            if (variant["step_budget"], variant["max_depth"]) != (0.100, 1):
                return "blocked", "unexpected effective match settings"
    elif name in ("evaluation", "recheck"):
        if (data["status"], status) != ("smoke_only", "selected"):
            return "blocked", "diagnostic selection incomplete"
    elif name == "book":
        if status not in ("selected", "no_common_book"):
            return "blocked", "book selection incomplete"
    elif name == "book_mode":
        if status not in ("selected", "N_only"):
            return "blocked", "book mode selection incomplete"
    elif status != "complete":
        return "blocked", f"{name} incomplete"
    if name == "report":
        if [(s["status"], s.get("complete")) for s in data["diagnostic_sources"]] != [("observed", True)]:
            return "blocked", "component diagnostic evidence missing"
        for filename, expected_hash in data["file_hashes"].items():
            if Path(data["files"][filename]) != directory / filename or file_hash(directory / filename) != expected_hash:
                raise ValueError("report file identity mismatch")
    return "complete", None


def _evidence(directory, splits):
    # Include raw checkpoints so a completed stage can be reused without repeating measurements.
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[str(path.relative_to(directory))] = file_hash(path)
            if path.name == "games.jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        game = json.loads(line)
                        if game["split"] not in ("D0", "D1") or splits.get(game["config_id"]) != game["split"]:
                            raise ValueError(f"non-development game in {path}")
    return hashes


def run(config, output):
    root = Path(output).resolve()
    if digest(config) != digest(load_json(PROFILE)):
        raise ValueError("smoke pipeline requires the fixed preparation profile")
    configs = _configs(config, root)
    if root.exists() and any(root.iterdir()) and not (root / "request.json").exists():
        raise ValueError("unbound smoke directory; use a new output directory")
    manifest = make_manifest(config["manifest_seed"])
    manifest_hash = manifest["manifest_hash"]
    splits = {row["config_id"]: row["split"] for row in manifest["rows"]}
    request = dict(execution_tier="smoke", formal_selection=False, output=str(root), manifest_hash=manifest_hash,
                   profile=config, configs=configs, sources=code_identity(), machine=machine_info())
    bind_request(root, request)
    path = root / "pipeline_summary.json"
    summary = dict(request_hash=digest(request), execution_tier="smoke", formal_selection=False,
                   manifest_hash=manifest_hash, sources=request["sources"], machine=request["machine"],
                   report_paths={}, controlled_probe=None, book_identity=None, stages={})
    if path.exists():
        summary = load_json(path)
        if summary["request_hash"] != digest(request):
            raise ValueError("pipeline summary differs from the bound request; use a new output directory")
    for name, cfg in configs.items():
        config_path = root / "configs" / (name + ".json")
        if config_path.exists() and load_json(config_path) != cfg:
            raise ValueError(f"stage config changed: {config_path}; use a new output directory")
        atomic_json(config_path, cfg)
        summary["stages"].setdefault(name, dict(config=str(config_path), output=str(root / name),
                                               artifact=str(root / name / STAGES[name][1]), status="pending", reason="not started"))

    def publish(status, reason):
        summary.update(status=status, reason=reason)
        atomic_json(path, summary)
        return summary

    for name, cfg in configs.items():
        stage = summary["stages"][name]
        directory = root / name
        try:
            if "evidence_hashes" in stage:
                if _evidence(directory, splits) != stage["evidence_hashes"]:
                    raise ValueError(f"saved {name} evidence changed; use a new output directory")
                stage.update(status="complete", reason=None)
                continue
            stage.update(status="pending", reason="stage started")
            publish("pending", f"{name} in progress")
            _execute(name, cfg, directory)
            stage["status"], stage["reason"] = _check_stage(name, cfg, directory)
            if stage["status"] != "complete":
                return publish(stage["status"], stage["reason"])
            if name == "prepare" and load_json(directory / "manifest.json")["manifest_hash"] != manifest_hash:
                raise ValueError("prepared manifest differs from the bound manifest")
            if name == "book":
                selection = load_json(directory / STAGES[name][1])
                summary["book_identity"] = {k: selection[k] for k in ("status", "K", "selected_book")}
            if name == "report":
                summary["report_paths"] = load_json(directory / STAGES[name][1])["files"]
            stage["evidence_hashes"] = _evidence(directory, splits)
            publish("pending", f"{name} complete")
        except (KeyboardInterrupt, SystemExit):
            stage.update(status="pending", reason="interrupted")
            publish("pending", f"{name} interrupted")
            raise
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            stage.update(status="blocked", reason=f"{type(error).__name__}: {error}")
            return publish("blocked", stage["reason"])
    try:
        probe_path = root / "controlled/controlled_probe.json"
        if summary["controlled_probe"]:
            if file_hash(probe_path) != summary["controlled_probe"]["sha256"]:
                raise ValueError("controlled probe evidence changed")
        else:
            controlled_probe(probe_path.parent)
            summary["controlled_probe"] = {"path": str(probe_path), "sha256": file_hash(probe_path)}
    except (OSError, ValueError) as error:
        return publish("blocked", str(error))
    return publish("complete", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(load_json(args.config), args.output)
    print(result["status"], result["reason"] or "", flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
