"""Shared variant and input handling for development selection."""

from pathlib import Path

from agents.evaluation import evaluation_parameters
from search_experiments.common import (atomic_json, code_manifest, digest, file_hash, load_json,
                                       normalize_variant)
from search_experiments.runner import validate_manifest

Q_VALUES = ((1, 20, 100), (1, 50, 100), (1, 80, 100), (20, 50, 100))
SMOKE_DEFAULTS = {"max_configs": 1, "step_budget": 0.012, "opponent_budget": 0.006, "max_depth": 2}


def evaluation_key(evaluation):
    strength, weights = evaluation_parameters(evaluation)
    return strength, weights if strength else None


def evaluation_variant(base, evaluation):
    base = normalize_variant(base)
    strength, weights = evaluation_key(evaluation)
    if (strength, weights) == evaluation_key(base["evaluation"]):
        return base

    if strength == 0 or (strength, weights) == (1, Q_VALUES[0]):
        name = f"E{strength}"
        settings = {"evaluation_id": name}
    else:
        name = f"L{strength}_Q{Q_VALUES.index(weights)}"
        settings = {"evaluation_id": name, "lambda": strength, "q": weights}
    base_id = base["variant_id"]
    tag = base["evaluation"].get("evaluation_id", "E1")
    family = base_id.removesuffix(f"_{tag}_N")
    return normalize_variant(dict(base, variant_id=f"{family}_{name}_N", parent_variant=base_id,
                                  evaluation=settings, book={"mode": "N"}))


def _search_role(path, variants, label, filename=None):
    variant = variants.get(label)
    required = {"variant_id", "parent_variant", "search", "evaluation", "book", "step_budget", "telemetry"}
    if not isinstance(variant, dict) or not required <= variant.keys():
        raise ValueError(f"Step 2 selection lacks a complete variant for {label}")
    variant = normalize_variant(variant)
    if variant["book"].get("mode") != "N" or evaluation_key(variant["evaluation"]) != (1, Q_VALUES[0]):
        raise ValueError("Step 2 search roles must use E1 with book disabled")
    if filename:
        role_path = path.with_name(filename)
        role = load_json(role_path) if role_path.is_file() else None
        if not isinstance(role, dict) or not required <= role.keys() or normalize_variant(role) != variant:
            raise ValueError(f"Step 2 selection disagrees with {filename}")
    return variant


def load_search_selection(path, manifest_hash, smoke=False):
    """Read the current Step 2 result and return its two original normalized variants."""
    path = Path(path)
    selection = load_json(path)
    tier, status = selection.get("execution_tier"), selection.get("status")
    formal = tier == "formal" and status == "selected"
    if not formal and not (smoke and tier == "smoke" and status == "smoke_only"):
        raise ValueError("Step 2 requires a selected formal result or an explicitly requested smoke result")
    current_code = code_manifest()["code_hash"]
    if selection.get("code_hash") != current_code:
        raise ValueError("Step 2 selection requires current code")
    if selection.get("manifest_hash") != manifest_hash:
        raise ValueError("Step 2 selection requires the same frozen manifest")

    variants = selection.get("variants", {})
    label = selection.get("S_star" if formal else "S_diag")
    star = _search_role(path, variants, label, "selected_search.json" if formal else None)
    alternate = _search_role(path, variants, selection.get("S_alt"), "alternate_search.json")
    if formal:
        summary = selection.get("development", {}).get(label, {})
        if (not summary.get("complete") or not summary.get("eligible") or summary.get("planned") != 48
                or summary.get("adjudicated") != 48 or summary.get("execution_tier") != "formal"
                or not summary.get("formal_ranking_allowed") or summary.get("code_hash") != current_code
                or summary.get("variant_id") != star["variant_id"]):
            raise ValueError("Step 2 winner requires complete eligible formal development results for current code")
        if any(variant["step_budget"] != 1.6 for variant in (star, alternate)):
            raise ValueError("formal Step 2 variants require a 1.6 second budget")
    return selection, star, alternate


def match_settings(config, smoke=False):
    """Return execution tier, effective AB opponent, and smoke settings if requested."""
    opponent = {"kind": "ab", "budget": 1.5, **config.get("opponent", {})}
    if opponent["kind"] != "ab":
        raise ValueError("development selection uses the local AB opponent")
    if not smoke:
        if opponent["budget"] != 1.5:
            raise ValueError("formal development requires a 1.5 second AB budget")
        return "formal", opponent, {}

    settings = {**SMOKE_DEFAULTS, **config.get("smoke", {})}
    if not (0 < settings["step_budget"] < 1.6 and 0 < settings["opponent_budget"] < 1.5):
        raise ValueError("smoke budgets must be positive and shorter than formal budgets")
    return "smoke", dict(opponent, budget=settings["opponent_budget"]), settings


def load_development_rows(manifest_path, max_configs=None):
    manifest = load_json(manifest_path)
    rows = validate_manifest(manifest)
    if max_configs is not None and (type(max_configs) is not int or max_configs < 1):
        raise ValueError("smoke max_configs must be a positive integer")
    d0, d1 = ([row for row in rows if row["split"] == split][:max_configs] for split in ("D0", "D1"))
    development_ids = {row["config_id"] for row in d0 + d1}
    development = [row for row in rows if row["config_id"] in development_ids]
    return manifest["manifest_hash"], d0, d1, development


def effective_search_variant(variant, smoke_settings):
    if smoke_settings:
        variant = dict(variant, step_budget=smoke_settings["step_budget"], max_depth=smoke_settings["max_depth"])
    return normalize_variant(variant)


def bind_request(output, request):
    path = Path(output) / "request.json"
    if path.exists():
        if digest(load_json(path)) != digest(request):
            raise ValueError("development selection request changed; use a new output directory")
    else:
        atomic_json(path, request)


def reuse_sources(paths):
    """Bind ordered log inputs, including paths that are currently absent."""
    paths = dict.fromkeys(Path(path).resolve() for path in paths)
    return [{"path": str(path), "sha256": file_hash(path) if path.is_file() else None} for path in paths]


def rank_matches(summaries, order):
    eligible = [name for name in order if summaries[name]["complete"] and summaries[name]["eligible"]]
    return sorted(eligible, key=lambda name: -summaries[name]["score"])
