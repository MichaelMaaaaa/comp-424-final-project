"""C0-C3 exact-value checks, cold searches and iterative-deepening costs."""

import argparse
from pathlib import Path

from agents.search_engine import component_configs
from search_experiments.common import atomic_json, code_manifest, file_hash, load_json, machine_info
from search_experiments.measure import exact_references, load_positions, measure


def run(config, output, smoke=False):
    output = Path(output)
    settings = config.get("smoke", {}) if smoke else config
    positions, payload = load_positions(config["positions"], settings.get("max_positions") if smoke else None,
                                        require_formal=not smoke)
    depths = settings.get("depths", [2, 3])
    guard = settings.get("guard_seconds", 60)
    evaluation = {"evaluation_id": "E1"}
    references = exact_references(positions, depths, evaluation, output / "references", guard)
    _, summary = measure(positions, component_configs(), depths, evaluation, output / "components", references,
                         guard_seconds=guard, execution_tier="smoke" if smoke else "formal")
    summary.update(execution_tier="smoke" if smoke else "formal",
                   positions_hash=payload["positions_hash"], positions_used=len(positions),
                   code_hash=code_manifest()["code_hash"], machine=machine_info(),
                   exact_reference={"path": "references",
                                    "request_hash": load_json(output/"references/request.json")["request_hash"],
                                    "measurements_hash": file_hash(output/"references/measurements.jsonl")},
                   exact_reference_incomplete=sum(r["status"] != "completed" for r in references.values()))
    atomic_json(output / "components_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", help="override fixed positions path")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.positions:
        config["positions"] = args.positions
    result = run(config, args.output, args.smoke)
    print("correct:", result["correct"], "common cases:", len(result["common_completed_position_depths"]))


if __name__ == "__main__":
    main()
