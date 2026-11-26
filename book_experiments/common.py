"""Inputs, identities and the unified selection rule shared by steps 5 and 6."""

from pathlib import Path

from search_experiments.common import (REPO_ROOT, code_manifest, digest, file_hash, load_json,
                                        normalize_variant, read_jsonl)
from search_experiments.runner import validate_manifest

CORE_FIELDS = ("search", "evaluation", "step_budget", "max_depth", "book")


def code_identity():
    """A's code manifest plus B's experiment modules."""
    files = dict(code_manifest()["files"])
    for path in sorted((REPO_ROOT / "book_experiments").glob("*.py")):
        files[str(path.relative_to(REPO_ROOT))] = file_hash(path)
    return {"files": files, "code_hash": digest(files)}


def smoke_settings(config, smoke):
    return dict(config.get("smoke", {})) if smoke else {}


def load_online(config, smoke):
    """F's fixed online configuration: the no-book variant, or {"variant": ...}.

    Search and evaluation parameters are read from this file, never fixed in
    B's code. Smoke runs only shrink the step budget and depth.
    """
    data = load_json(config["online_config"])
    variant = dict(data.get("variant", data))
    if variant.get("book", {"mode": "N"}).get("mode", "N") != "N":
        raise ValueError("the online configuration must be the no-book (N) variant")
    if smoke:
        settings = smoke_settings(config, smoke)
        variant["step_budget"] = settings.get("step_budget", 0.012)
        variant["max_depth"] = settings.get("max_depth", 2)
    return normalize_variant(variant)


def manifest_rows(config, split, smoke):
    if split not in ("D0", "D1"):
        raise ValueError("steps 5 and 6 use only the D0/D1 development configurations")
    rows = [row for row in validate_manifest(load_json(config["manifest"])) if row["split"] == split]
    if smoke:
        rows = rows[:smoke_settings(config, smoke).get("max_configs", 1)]
    return rows


def opponent(config, smoke):
    result = dict(config.get("opponent", {"kind": "ab", "budget": 1.5}))
    if smoke:
        result["budget"] = smoke_settings(config, smoke).get("opponent_budget", 0.006)
    return result


def reference_games(path, online):
    """F's D1 no-book trajectories; their core parameters must equal the online variant's."""
    path = Path(path)
    records = [r for r in read_jsonl(path) if r.get("status") == "adjudicated"]
    if not records:
        raise ValueError(f"no adjudicated reference games in {path}")
    plan = path.parent / "plan.json"
    check = {"verified": False, "reason": "plan.json not found next to the reference games"}
    if plan.exists():
        variant = load_json(plan)["request"]["variant"]
        differing = [name for name in CORE_FIELDS if digest(variant.get(name)) != digest(online.get(name))]
        if differing:
            raise ValueError(f"reference games differ from the online configuration in {differing}")
        check = {"verified": True, "reference_variant_id": variant["variant_id"]}
    return records, check


def rank(summaries, order):
    """Unified rule: complete and free of technical failures; score; then the earlier, simpler option."""
    valid = [name for name in order if summaries.get(name) and summaries[name].get("eligible")]
    return sorted(valid, key=lambda name: (-summaries[name]["score"], order.index(name)))


def games_of(summary, output):
    return [r for r in read_jsonl(Path(output) / "games.jsonl") if r.get("request_hash") == summary["request_hash"]]
