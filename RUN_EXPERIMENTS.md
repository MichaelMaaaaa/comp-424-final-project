# Running the COMP 424 experiments

Run commands from the repository root. Preparation consists of a serial smoke chain and three representative artifact checks. Both use development openings. The frozen manifest also lists the 24 held-out configurations for the eventual formal final.

## Environment

Use the same interpreter, CPU affinity and thread settings for a run and its recovery. Allow one campaign or test process at a time. A strict match runs its two players in isolated subprocesses and asks them for moves in turn.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys, numpy, matplotlib; print(sys.version, numpy.__version__, matplotlib.__version__)'
```

The target is Python 3.11.x, NumPy 1.26.4 and matplotlib 3.8.4. The representative acceptance command checks the installed versions before building or validating candidates.

On Linux, use `taskset -pc $$` to list the CPUs allowed for your shell. Choose one listed core. For example, if CPU 0 is allowed, prefix each campaign command with `taskset -c 0`, as in `taskset -c 0 python -m search_experiments.smoke_pipeline ...`. Use the same core and thread settings for recovery.

## Whole-step timing

Every online variant uses one absolute deadline established at step entry. Legal-move generation, observation conversion, book lookup, ordering and search all consume the same whole-step allowance. The common wrapper reserves `min(0.05, step_budget / 10)` seconds for runtime pauses around cooperative deadline checks and reporting/return overhead. A 1.6-second step therefore gives these operations an internal deadline 1.55 seconds after entry. Short budgets reserve at most ten percent; a zero budget remains zero.

This common reserve applies to N, D and H and every search configuration. It reduces effective search time and can change completed depths or returned moves. The search algorithms, evaluator, book decision rules, nominal configuration budgets and acceptance gates stay unchanged. The reserve is a conservative engineering allowance; it does not establish a worst-case bound on Python runtime or scheduling delays. Whole-step timing still has to pass the recorded acceptance checks.

## Preparation commands

Choose unused directories for the smoke chain and the representative preflight.

```sh
python -m search_experiments.smoke_pipeline --config search_experiments/configs/smoke_pipeline.json --output experiments/preparation_smoke
python -m search_experiments.acceptance --manifest experiments/preparation_smoke/prepare/manifest.json --output experiments/representative_preflight
```

Smoke runs real Steps 0 through 8 in order. It binds the seeds, machine, source hashes, generated configs and paths in `request.json`. Each diagnostic match uses the first configuration of its required split, both seats, a 0.100-second candidate at depth one and a 0.006-second AB opponent. Fixed-depth comparisons use one position at depth one. Online diagnostics retain each producer’s search workload with a 0.100-second budget. Step 2 emits its original source variants. Step 3 applies the fixed match budget and depth to those variants. The book collection and screening settings are in the checked-in smoke profile.

`pipeline_summary.json` lists each producer’s config, output, status, reason and evidence hashes. Read `report/report_summary.json` for game outcomes, optional book controls, measured hit counts and links to CSVs and figures. The one-position N/D/H check writes `controlled/controlled_probe.json`. It exercises a known canonical book entry independently of the campaign logs. Smoke selections are diagnostic and have no formal ranking authority.

Representative preflight builds S0/E1 in N, D and H modes at 1.6 seconds with no depth cap. The default D and H artifacts share the original embedded book’s entries. To exercise another exported book, pass `--book /path/to/exported_book.json` and choose a fresh preflight directory. The command validates the export’s entries, layout and data hash. Each artifact gets its own official static check and complete default four-game official check on reserved seeds 0 and 1, both seats. A strict AB pair then uses reserved D0 seed 0 in both seats against the 1.5-second baseline. The strict runner uses the formal execution tier to enforce timing and eligibility. Its enclosing purpose is `preflight` and `candidate_ranking_allowed` is false.

Read `acceptance_summary.json` for artifact SHA-256 values, sizes, validator measurements, strict outcomes, resource maxima and exact raw evidence paths. `final_plan.json` in the representative directory freezes the manifest, candidates, build metadata, machine, dependencies and source hashes. The exact artifacts selected in a future formal campaign still require Step 7 validation.

## Recovery

Repeat the identical command with the original directory after an interruption. Review the top-level summary and its stage reason. A completed smoke stage is reused after checking its evidence hashes. Pending stages resume from their own request-bound checkpoints. Step 5 collection games are in `book/collection/games.jsonl` and offline key records are in `book/analysis/records.jsonl`. The representative validator reuses completed official verdicts and the strict runner reuses completed adjudications. A pending official batch can require rerunning that single batch.

A completed smoke rerun can skip all nine unchanged stages. Controlled interrupted-prefix tests exercise recovery inside pending producers, including preservation of previously recorded games and offline analyses.

Use a fresh directory for changed code, machine, roster, budgets, settings, book, input path or bound artifact bytes. Preserve any blocked run’s files for diagnosis. Timing-dependent smoke decisions can block later stages. Keep the recorded reason and measurements with the run.

## Formal campaign commands

The following commands describe a separate future campaign. Step 0 freezes the 12 D0, 12 D1 and 24 T configurations and samples reference positions. For a roster-only export, use a copy of the prepare config with `run_reference` set to `false`. Use a separate directory for that export.

| Step | Reads | Produces |
| --- | --- | --- |
| 0 | Manifest/sample seeds | Manifest JSON and CSV, D0 reference games, fixed positions |
| 1 | Fixed positions | Exact references, component correctness and timing |
| 2 | Manifest, positions, components, reference games | Confirmed search and alternate search |
| 3 | Manifest and search roles | Evaluation screening and confirmed evaluation roles |
| 4 | Search/evaluation roles, manifest, positions | Rechecked no-book online config and D1 reference |
| 5 | Online config, manifest and D1 reference | Collection and analysis logs, exported books and source/size decision |
| 6 | Step 5 decision, online config, manifest, D1 reference, positions | Final mode and any required comparison variant |
| 7 | Frozen search, recheck, online, book and mode handoffs, manifest | Exact artifacts, validation and strict final matches on T |
| 8 | Final summary and independent component diagnostics | Verified report, raw-evidence links, CSVs and figures |

```sh
python -m search_experiments.exp00_prepare --config search_experiments/configs/prepare.json --output experiments/a_run/prepare
python -m search_experiments.exp01_components --config search_experiments/configs/components.json --output experiments/a_run/components
python -m search_experiments.exp02_search --config search_experiments/configs/search.json --output experiments/a_run/search
python -m search_experiments.exp03_evaluation --config search_experiments/configs/evaluation.json --output experiments/f_run/evaluation
python -m search_experiments.exp04_recheck --config search_experiments/configs/recheck.json --output experiments/f_run/recheck
python -m book_experiments.exp05_book --config book_experiments/configs/book.json --output experiments/b_run/book
python -m book_experiments.exp06_book_mode --config book_experiments/configs/book_mode.json --output experiments/b_run/book_mode
python -m search_experiments.exp07_final --config search_experiments/configs/final.json --output experiments/f_run/final
python -m search_experiments.exp08_report --config search_experiments/configs/report.json --output experiments/f_run/report
```

Inspect the stage artifact and eligibility before starting the next formal command. Some producers exit normally after publishing a pending or blocked decision. Formal development comparisons use D0 and D1. Step 7 alone admits held-out final matches. Keep preflight and smoke directories outside the formal input and reuse paths.
