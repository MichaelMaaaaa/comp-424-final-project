"""Freeze 48 configurations, run S0 reference pairs, and sample fixed positions."""

import argparse
import csv
from pathlib import Path

from search_experiments.common import atomic_json, load_json, read_jsonl
from search_experiments.runner import make_manifest, run_pairs, sample_positions, validate_manifest


def prepare(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    desired = make_manifest(config.get("manifest_seed", 20261125))
    if manifest_path.exists() and load_json(manifest_path) != load_json_value(desired):
        raise ValueError("an existing manifest is frozen; use another output directory for a new seed")
    atomic_json(manifest_path, desired)
    rows = validate_manifest(desired)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["config_id", "obstacle_row", "obstacle_col", "obstacle_class", "removed_size",
                  "removed_name", "split", "setup_seed", "opponent_rng_seed", "collection_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if not config.get("run_reference", True):
        return {"manifest": str(manifest_path), "reference": "not_requested"}
    selected = [row for row in rows if row["split"] == "D0"]
    if smoke:
        selected = selected[:config.get("smoke", {}).get("max_configs", 1)]
    variant = {"variant_id": "S0_E1_N", "parent_variant": None, "search": {},
               "evaluation": {"evaluation_id": "E1"}, "book": {"mode": "N"},
               "step_budget": 1.6, "telemetry": True}
    opponent = {"kind": "ab", "budget": 1.5}
    if smoke:
        variant["step_budget"] = config.get("smoke", {}).get("step_budget", 0.01)
        variant["max_depth"] = config.get("smoke", {}).get("max_depth", 2)
        opponent["budget"] = config.get("smoke", {}).get("opponent_budget", 0.005)
    summary = run_pairs(variant, selected, output / "reference", opponent=opponent,
                        execution_tier="smoke" if smoke else "formal")
    records = read_jsonl(output / "reference" / "games.jsonl")
    samples = sample_positions(records, config.get("sample_seed", 424))
    samples["reference_request_hash"] = summary["request_hash"]
    samples["reference_complete"] = summary["complete"]
    samples["execution_tier"] = "smoke" if smoke else "formal"
    atomic_json(output / "positions.json", samples)
    atomic_json(output / "preparation.json", {"manifest_hash": desired["manifest_hash"],
        "reference_complete": summary["complete"], "positions_count": samples["count"],
        "target": samples["target"], "missing_strata": [row for row in samples["coverage"] if row["missing"]],
        "execution_tier": samples["execution_tier"]})
    return summary


def load_json_value(value):
    import json
    return json.loads(json.dumps(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true", help="small self-test; never a formal selection")
    args = parser.parse_args()
    print(prepare(load_json(args.config), args.output, args.smoke))


if __name__ == "__main__":
    main()
