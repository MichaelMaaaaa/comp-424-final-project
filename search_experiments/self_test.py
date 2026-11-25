"""Run A's small handoff checks; these outputs are not formal experiment results."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from agents.search_engine import SearchConfig
from search_experiments.build import build_agent
from search_experiments.common import (REPO_ROOT, atomic_json, file_hash, load_json,
                                        machine_info, run_command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = file_hash(REPO_ROOT / "agents/student_agent.py")
    commands = [
        ("unit_tests.txt", [sys.executable, "-m", "unittest", "tests.test_search_experiments", "-v"]),
        ("prepare.txt", [sys.executable, "-m", "search_experiments.exp00_prepare", "--config",
             "search_experiments/configs/prepare.json", "--output", str(output / "prepare"), "--smoke"]),
        ("components.txt", [sys.executable, "-m", "search_experiments.exp01_components", "--config",
             "search_experiments/configs/components.json", "--positions", str(output / "prepare/positions.json"),
             "--output", str(output / "components"), "--smoke"]),
        ("search.txt", [sys.executable, "-m", "search_experiments.exp02_search", "--config",
             "search_experiments/configs/search.json", "--positions", str(output / "prepare/positions.json"),
             "--manifest", str(output / "prepare/manifest.json"), "--output", str(output / "search"), "--smoke"]),
    ]
    for label, command in commands:
        print("Running", label, flush=True)
        run_command(command, output / label)
    # Exercise resume: exact completed games must not be played or appended twice.
    game_log = output / "prepare/reference/games.jsonl"
    log_before = file_hash(game_log)
    run_command(commands[1][1], output / "resume.txt")
    if file_hash(game_log) != log_before:
        raise AssertionError("resume duplicated completed games")
    config = SearchConfig(preferred_first=True, reuse_results=True, key_mode="compact", symmetry=True)
    variant = {"variant_id": "A_selftest_S4_E1_N", "search": asdict(config),
               "evaluation": {"evaluation_id": "E1"}, "book": {"mode": "N"},
               "step_budget": 0.03, "max_depth": 2, "telemetry": True}
    path, _ = build_agent(variant, output / "generated/student_agent.py")
    print("Running official sandbox validator", flush=True)
    validation = run_command([sys.executable, "validate_submission.py", str(path), "--json",
                              str(output / "validation.json")], output / "validation.txt")
    if "RESULT: PASS" not in validation:
        raise AssertionError("validator did not pass")
    components = load_json(output / "components/components_summary.json")
    selection = load_json(output / "search/search_selection.json")
    if not components["correct"] or not components["verification_complete"]:
        raise AssertionError("component correctness was not fully verified")
    if any(not d["correct_E0_E1"] for d in selection["decisions"]):
        raise AssertionError("an optimization did not pass exact E0/E1 references")
    if file_hash(REPO_ROOT / "agents/student_agent.py") != before:
        raise AssertionError("the original student_agent.py changed")
    summary = {"status": "PASS", "execution_tier": "smoke", "machine": machine_info(),
               "source_student_agent_sha256": before, "original_submission_unchanged": True,
               "resume_does_not_duplicate": True, "components_correct": True,
               "optimizations_correct_E0_E1": True, "official_validator": "PASS",
               "formal_search_selected": False,
               "note": "A's code handoff and small tests only; F runs the full formal 0-8 sequence."}
    atomic_json(output / "SELF_TEST_SUMMARY.json", summary)
    print("SELF TEST: PASS", flush=True)


if __name__ == "__main__":
    main()
