"""Export Step 8 results from saved final evidence and separate diagnostic measurements."""

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agents.search_engine import SearchCounters
from search_experiments import reporting
from search_experiments.common import atomic_json, canonical_json, code_manifest, digest, file_hash, load_json, machine_info
from search_experiments.report_evidence import admit_campaign


COUNTERS = [f.name for f in fields(SearchCounters)] + ["tt_entries", "approximate_tt_bytes"]
CASE_FIELDS = ("case_id", "request_hash", "execution_tier", "position_id", "variant_id", "evaluation", "workload",
               "target_depth", "effective_target_depth", "completed_depth", "status", "target_reached", "checked",
               "correct", "stop_reason", "interrupted_depth", "elapsed", "ply", "near_cap", "obstacle_class")


def load_diagnostics(directory, tier, positions=None):
    """Positions, if provided, are the exact list used in the measurement request."""
    directory = Path(directory).resolve()
    request_path, log = directory/"request.json", directory/"measurements.jsonl"
    source = dict(directory=str(directory), request_path=str(request_path), measurements_path=str(log),
                  complete=None, expected_cases=None, missing_case_ids=None)
    if not directory.exists():
        return [], dict(source, status="missing_directory")
    if not request_path.exists() or not log.exists():
        return [], dict(source, status="missing_request_or_log")
    data = request_path.read_bytes()
    saved = json.loads(data)
    req, fingerprint = saved["request"], saved["request_hash"]
    if (digest(req) != fingerprint or req["code"] != code_manifest() or req["machine"] != machine_info()
            or req["execution_tier"] != tier):
        raise ValueError(f"diagnostic request identity mismatch at {directory}")
    data_hash = hashlib.sha256(data).hexdigest()
    raw = log.read_bytes()
    source.update(request_hash=fingerprint, request_file_hash=data_hash, measurements_hash=hashlib.sha256(raw).hexdigest(),
                  positions_hash=req["positions"], evaluation=req["evaluation"], diagnostics_enabled=req["diagnostics"])
    expected = None
    if positions is not None:
        ids = [p["position_id"] for p in positions]
        if digest(positions) != req["positions"] or len(set(ids)) != len(ids):
            raise ValueError("diagnostic positions differ from the frozen measurement request")
        expected = {digest([pid, name, d, w, -1, True]) for pid in ids for name in req["configs"]
                    for d in req["depths"] for w in req["workloads"] if (name,w) != ("C0","iterative")
                    and req["diagnostics"]}
    rows, seen = [], set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid, name, depth, workload, rep, instrumented = (
            r[k] for k in ("position_id","variant_id","target_depth","workload","repetition","instrumented"))
        case = digest([pid, name, depth, workload, rep, instrumented])
        if (case != r["case_id"] or case in seen or r["request_hash"] != fingerprint or r["execution_tier"] != tier
                or r["evaluation"] != req["evaluation"] or name not in req["configs"] or type(instrumented) is not bool
                or type(rep) is not int or (positions is not None and pid not in ids)):
            raise ValueError(f"invalid or duplicate measurement case in {log}")
        seen.add(case)
        online = workload == "online"
        if (online and (req["online_budget"] is None or depth is not None)
                or not online and (workload not in req["workloads"] or depth not in req["depths"]
                                   or (name,workload) == ("C0","iterative"))):
            raise ValueError("measurement differs from its requested workload")
        if not instrumented:
            if r["measurement"] != "timing" or not 0 <= rep < req["repeats"] or r["statistics"]["enabled"] is not False:
                raise ValueError("invalid timing row in diagnostic source")
            continue
        if (r["measurement"] != "diagnostic" or rep != -1 or online or req["diagnostics"] is not True
                or workload not in ("cold","iterative") or r["statistics"]["enabled"] is not True):
            raise ValueError("invalid independently instrumented diagnostic")
        target = depth if workload == "cold" else min(depth, 100-r["ply"])
        if r["effective_target_depth"] != target:
            raise ValueError("diagnostic effective target differs from the requested depth")
        counters = {k:r["statistics"].get(k) for k in COUNTERS}
        rows.append({**{k:r.get(k) for k in CASE_FIELDS}, **counters,
                     **{k:source[k] for k in ("request_path","measurements_path","request_file_hash","measurements_hash",
                                              "positions_hash")},
                     "completed_target":r["status"] == "completed" and r["completed_depth"] >= target
                                        and r["target_reached"] is True,
                     "missing_counters":[k for k, value in counters.items() if value is None]})
    source.update(status="observed" if rows else "no_diagnostic_rows", observed_cases=len(rows),
                  completeness_reason="positions list unavailable" if expected is None else None)
    if expected is not None:
        missing = expected - {r["case_id"] for r in rows}
        source.update(complete=not missing, expected_cases=len(expected), missing_case_ids=sorted(missing))
    return rows, source


def _write_csv(path, rows, columns):
    keys = dict.fromkeys(columns)
    for row in rows:
        keys.update(dict.fromkeys(row))
    def cell(value):
        if value is None:
            return "N/A"
        return canonical_json(value) if isinstance(value, (list,tuple,dict)) else value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(keys))
        writer.writeheader()
        for row in rows:
            writer.writerow({key:cell(row.get(key)) for key in keys})


def _plots(directory, tier, results, comparisons):
    from matplotlib.figure import Figure
    names = []
    for filename, title, labels, values in (
        ("scores.png", "Full-roster score",
         [f"{r['candidate']} / {r['opponent']}" + ("" if r["eligible"] else " (ineligible)")
          for r in results if r["score"] is not None],
         [r["score"] for r in results if r["score"] is not None]),
        ("differences.png", "Mean paired score difference vs AB",
         [f"{r['new_role']} − {r['old_role']}" + ("" if r["new_eligible"] and r["old_eligible"] else " (ineligible)")
          for r in comparisons if r["complete"]],
         [r["mean"] for r in comparisons if r["complete"]]),
    ):
        if not values:
            continue
        figure = Figure(figsize=(8, max(2.6, len(values)*.55+1.2)), layout="constrained")
        ax = figure.subplots()
        ax.barh(labels, values)
        ax.set_title(f"{tier.capitalize()} — {title}")
        ax.set_xlim((0,1) if filename == "scores.png" else (-1,1))
        ax.axvline(0, linewidth=.7, color="black")
        figure.savefig(directory/filename, dpi=120)
        names.append(filename)
    return names


def run(config, output, smoke=False):
    tier = "smoke" if smoke else "formal"
    try:
        final, req, groups, replay, fingerprints = admit_campaign(config["final_summary"], smoke)
    except (KeyError, AttributeError, IndexError, TypeError, OSError) as error:
        raise ValueError(f"invalid Step 7 evidence: {error}") from error
    output = Path(output).resolve()
    old_path = output/"report_summary.json"
    if old_path.exists() and load_json(old_path)["input_fingerprints"]["request_hash"] != fingerprints["request_hash"]:
        raise ValueError("use a new report directory for a different final plan")
    config_ids = [row["config_id"] for row in req["final_rows"]]
    results, turns, games, game_strata, turn_strata, live, book_rows = [], [], [], [], [], [], []
    descriptors = {}
    for candidate, opponent, records, attempts, saved in groups:
        identity = dict(candidate=candidate, opponent=opponent,
                        book_mode=req["artifacts"][candidate]["variant"]["book"]["mode"])
        group_games, group_turns = [], []
        for record in records:
            these_turns = reporting.turn_rows(record, opponent)
            group_turns.extend(these_turns)
            group_games.append(reporting.game_row(record, opponent, these_turns))
        result = reporting.result_summary(candidate, opponent, config_ids, records, attempts)
        result.update(book_mode=identity["book_mode"], roles=[k for k,v in req["role_candidates"].items() if v==candidate],
                      evidence_status="verified" if saved else "missing_descriptor",
                      resources_and_games=reporting.game_metrics(group_games), search_and_turns=reporting.turn_metrics(group_turns))
        results.append(result)
        identity["group_eligible"] = result["eligible"]
        for game in group_games:
            game["group_eligible"] = result["eligible"]
        descriptors.setdefault(candidate,{})[opponent] = saved
        games.extend(group_games)
        turns.extend(group_turns)
        for row in reporting.strata(group_games, ("obstacle_class","removed_size","candidate_seat","reason","split","config_id"),
                                    reporting.game_metrics):
            game_strata.append(dict(identity, **row))
        for row in reporting.strata(group_turns, ("stage", "reserves_total", "actor_role", "reserves_actor",
                                                     ("stage", "reserves_total", "actor_role")),
                                    reporting.turn_metrics):
            turn_strata.append(dict(identity, **row))
        game_ids = [r["game_id"] for r in group_games]
        metrics = dict(identity, **reporting.live_metrics(game_ids, group_turns))
        live.append(metrics)
        book_rows.append(dict(metrics, cohort="final_live", stage=None))
        for phase in ("early","middle","late"):
            segment = [r for r in group_turns if r["stage"]==phase]
            active_games = {r["game_id"] for r in segment}
            book_rows.append(dict(identity, cohort="final_live", stage=phase,
                                  **reporting.live_metrics(active_games, segment)))
    for key in ("by_stage","by_obstacle_class"):
        for value, counts in replay.get(key,{}).items():
            book_rows.append(dict(cohort="D1_N_replay", dimension=key, value=value, **counts))
    book_rows.append(dict(cohort="D1_N_replay", **{k:v for k,v in replay.items()
                                                    if k not in ("by_stage","by_obstacle_class","matched_turns","d1_reference")}))
    seed = config.get("bootstrap_seed", 20261125)
    if type(seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    comparisons = reporting.contrasts(req["role_candidates"], {r["candidate"]:r for r in results if r["opponent"]=="ab"},
                                      config_ids, seed)
    differences = []
    for comparison in comparisons:
        common = {k:v for k,v in comparison.items() if k!="per_config"}
        differences.append(dict(common, level="contrast", config_id=None, difference=comparison["mean"]))
        differences.extend(dict(common, level="configuration", config_id=config, difference=diff)
                           for config,diff in comparison["per_config"].items())
    sources, diagnostics = [], []
    seen_dirs = set()
    seen_logs = set()
    for source in config.get("diagnostic_dirs", []):
        # A plain directory is sufficient to report observed cases. An exact planned list enables a missing-case count.
        directory = Path(source["directory"] if isinstance(source,dict) else source).resolve()
        if directory in seen_dirs:
            raise ValueError("duplicate diagnostic directory")
        seen_dirs.add(directory)
        positions = None
        if isinstance(source, dict):
            positions = load_json(source["positions"])["positions"]
            positions = positions[:source.get("max_positions")]
        try:
            rows, detail = load_diagnostics(directory, tier, positions)
        except (KeyError, AttributeError, TypeError, OSError) as error:
            raise ValueError(f"invalid diagnostic source at {directory}: {error}") from error
        if "measurements_hash" in detail:
            identity = (detail["request_hash"], detail["measurements_hash"])
            if identity in seen_logs:
                raise ValueError(f"duplicate diagnostic source at {directory}")
            seen_logs.add(identity)
        sources.append(detail)
        diagnostics.extend(rows)
    tables = {
        "results.csv":(results, ["candidate","opponent","roles","status","eligible","planned","adjudicated","W","D","L","score"]),
        "paired_differences.csv":(differences, ["new_role","old_role","level","config_id","difference","bootstrap_config_ci95"]),
        "game_strata.csv":(game_strata, ["candidate","opponent","dimension","value","games","W","D","L","observed_mean_score"]),
        "turn_strata.csv":(turn_strata, ["candidate","opponent","dimension","value","games","turns"]),
        "turns.csv":(turns, ["candidate","opponent","game_id","ply","actor_role","action_type","step_s","completed_depth"]),
        "diagnostics.csv":(diagnostics, [*CASE_FIELDS,*COUNTERS,"missing_counters"]),
        "book_metrics.csv":(book_rows, ["cohort","candidate","opponent","stage","games","query_observed","queries","matches","hit_rate"]),
    }
    summary = dict(status=final["status"], reason=final["reason"], execution_tier=tier, formal_results=final["formal_selection"],
                   role_candidates=req["role_candidates"], roles=req["roles"], v3_omission=req["v3_omission"], submitted_candidate=req["submitted_candidate"],
                   input_fingerprints=fingerprints, match_evidence=descriptors, results=results, paired_differences=comparisons,
                   validation=final["validation"], submission=final["submission"],
                   games=games, game_strata=game_strata, turn_strata=turn_strata, coverage=dict(d1_replay=replay, live=live),
                   diagnostic_sources=sources, missing_data=dict(
                       pending_groups=[dict(candidate=r["candidate"],opponent=r["opponent"],missing_game_ids=r["missing_game_ids"])
                                       for r in results if not r["complete"]],
                       telemetry=[dict(candidate=r["candidate"], opponent=r["opponent"], candidate_calls=r["candidate_calls"],
                                       missing_payloads=r["candidate_calls"]-r["telemetry_observed"], query_missing=r["query_missing"])
                                  for r in live],
                       diagnostics="see diagnostic_sources" if sources else "no independent source supplied"),
                   trajectory_scope="Saved pre-turn snapshots, including a saved failed turn. Repetitions count occurrences after the first per game.")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staging = Path(tmp)
        for name, (rows, columns) in tables.items():
            _write_csv(staging/name, rows, columns)
        plots = _plots(staging,tier,results,comparisons) if config.get("plots",True) else []
        names = [*tables,*plots]
        summary["files"] = {name:str(output/name) for name in [*names,"report_summary.json"]}
        summary["file_hashes"] = {name:file_hash(staging/name) for name in names}
        atomic_json(staging/"report_summary.json", summary)
        for name in names:
            (staging/name).replace(output/name)
        for unused in {"scores.png","differences.png"}-set(plots):
            (output/unused).unlink(missing_ok=True)
        (staging/"report_summary.json").replace(old_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(load_json(args.config), args.output, args.smoke)
    print(report["execution_tier"], report["status"], report["files"]["report_summary.json"])


if __name__ == "__main__":
    main()
