"""Run B's small handoff checks; these outputs are not formal experiment results.

Steps 0 and 4 have not produced formal outputs yet, so the self-test makes its
own small stand-ins: A's smoke manifest and positions, and a smoke D1 no-book
run of the template online variant in place of F's reference trajectories.
"""

import argparse
from pathlib import Path
import sys

from book_experiments.books import book_variant, load_book
from book_experiments.common import load_online, manifest_rows, opponent
from search_experiments.build import build_agent
from search_experiments.common import REPO_ROOT, atomic_json, file_hash, load_json, machine_info, run_command
from search_experiments.runner import run_pairs

ONLINE_TEMPLATE = "search_experiments/configs/variant.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(REPO_ROOT):
        # Relative paths keep machine-specific folders out of the recorded outputs.
        output = output.relative_to(REPO_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    before = file_hash(REPO_ROOT / "agents/student_agent.py")
    print("Running unit_tests.txt", flush=True)
    run_command([sys.executable, "-m", "unittest", "tests.test_book_experiments", "-v"], output / "unit_tests.txt")
    print("Running prepare.txt (A's smoke manifest and positions)", flush=True)
    run_command([sys.executable, "-m", "search_experiments.exp00_prepare", "--config",
                 "search_experiments/configs/prepare.json", "--output", str(output / "prepare"), "--smoke"],
                output / "prepare.txt")

    book_config = load_json(REPO_ROOT / "book_experiments/configs/book.json")
    book_config.update(online_config=ONLINE_TEMPLATE, manifest=str(output / "prepare/manifest.json"))
    online = load_online(book_config, smoke=True)
    print("Running a smoke D1 no-book reference in place of F's step 4", flush=True)
    reference = run_pairs(online, manifest_rows(book_config, "D1", True), output / "d1_reference",
                          opponent=opponent(book_config, True), execution_tier="smoke")
    if not reference["complete"]:
        raise AssertionError("smoke D1 reference games did not finish")

    common = ["--online-config", ONLINE_TEMPLATE, "--manifest", str(output / "prepare/manifest.json"),
              "--references", str(output / "d1_reference/games.jsonl"), "--smoke"]
    step5 = [sys.executable, "-m", "book_experiments.exp05_book", "--config", "book_experiments/configs/book.json",
             "--output", str(output / "book")] + common
    step6 = [sys.executable, "-m", "book_experiments.exp06_book_mode", "--config",
             "book_experiments/configs/book_mode.json", "--selected-book", str(output / "book/selected_book.json"),
             "--positions", str(output / "prepare/positions.json"), "--output", str(output / "book_mode")] + common
    print("Running book.txt (step 5 smoke)", flush=True)
    run_command(step5, output / "book.txt")
    # A rerun must reuse every finished collection game and analysis record.
    logs = [output / "book/collection/games.jsonl", output / "book/analysis/records.jsonl"]
    hashes = [file_hash(path) for path in logs]
    print("Running book_resume.txt", flush=True)
    run_command(step5, output / "book_resume.txt")
    if [file_hash(path) for path in logs] != hashes:
        raise AssertionError("resume repeated finished collection or analysis work")
    print("Running book_mode.txt (step 6 smoke)", flush=True)
    run_command(step6, output / "book_mode.txt")

    selection = load_json(output / "book/selected_book.json")
    final = load_json(output / "book_mode/final_config.json")
    checks = {"selection_status": selection["status"], "final_mode": final["mode"]}
    if selection["status"] != "selected":
        raise AssertionError(f"smoke step 5 did not select a book: {selection['status']}")
    # Build the book (D) variant and the final variant and pass both through the official validator.
    book = load_book(selection["book_file"])
    builds = {"book_D": book_variant(online, book, "D"), "final": final["variant"]}
    for name, variant in builds.items():
        path, meta = build_agent(variant, output / "generated" / name / "student_agent.py")
        print(f"Running official validator on {name}", flush=True)
        text = run_command([sys.executable, "validate_submission.py", str(path), "--json",
                            str(output / f"validation_{name}.json")], output / f"validation_{name}.txt")
        if "RESULT: PASS" not in text:
            raise AssertionError(f"validator did not pass the {name} build")
        checks[f"{name}_bytes"] = meta["bytes"]
    if file_hash(REPO_ROOT / "agents/student_agent.py") != before:
        raise AssertionError("the original student_agent.py changed")
    summary = {"status": "PASS", "execution_tier": "smoke", "machine": machine_info(), **checks,
               "source_student_agent_sha256": before, "original_submission_unchanged": True,
               "resume_does_not_repeat_work": True, "official_validator": "PASS",
               "formal_book_selected": False,
               "note": "B's code handoff and small tests only; F runs the full formal 0-8 sequence."}
    atomic_json(output / "SELF_TEST_SUMMARY.json", summary)
    scrub_paths(output)
    print("SELF TEST: PASS", flush=True)


def scrub_paths(output):
    """Record paths relative to the repository root in the text outputs."""
    prefix = str(REPO_ROOT) + "/"
    for path in Path(output).rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".csv"):
            text = path.read_text(encoding="utf-8", errors="strict")
            if prefix in text:
                path.write_text(text.replace(prefix, ""), encoding="utf-8")


if __name__ == "__main__":
    main()
