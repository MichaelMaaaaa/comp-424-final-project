"""Step 6 (B): compare book modes N/D/H with the selected book; fix the final configuration.

N does not query the book. D returns a legal book move directly. H passes it
to the search as the first iteration's root preferred move. D and H use the
same book bytes, fallback and whole-step timer. Inputs: step 5's
selected_book.json, F's online configuration and D1 no-book games, A's manifest
and fixed development positions.
"""

import argparse
from pathlib import Path
import statistics

from agents.game_state import legal_moves, terminal
from book_experiments.books import book_table, book_variant, load_book, lookup
from book_experiments.common import (games_of, load_online, manifest_rows, opponent, rank,
                                     reference_games)
from book_experiments.coverage import game_metrics
from search_experiments.common import (SCHEMA_VERSION, atomic_json, load_json, normalize_variant,
                                        restore_state)
from search_experiments.measure import measure_online_step
from search_experiments.runner import paired_difference, run_pairs

MODES = ("N", "D", "H")


def _brief(summary):
    keys = ("complete", "eligible", "W", "D", "L", "score", "technical_failures", "request_hash")
    return {key: summary.get(key) for key in keys}


def all_miss_overhead(positions_path, book, variants, repeats=3, max_positions=None):
    """Same development states, all missing the book: query on (D) versus no query (N)."""
    table = book_table(book)
    states = []
    for position in load_json(positions_path)["positions"]:
        state = restore_state(position["state"])
        if terminal(state)[0]:
            continue
        moves = legal_moves(state)
        move = lookup(table, state)
        if len(moves) > 1 and (move is None or move not in moves):
            states.append((position["position_id"], state))
    states = states[:max_positions] if max_positions else states
    quiet = {mode: dict(variants[mode], telemetry=False) for mode in ("N", "D")}
    rows = []
    for repetition in range(repeats):
        for position_id, state in states:
            for mode in (("N", "D") if repetition % 2 == 0 else ("D", "N")):
                info = measure_online_step(state, quiet[mode])
                rows.append({"position_id": position_id, "repetition": repetition, "mode": mode,
                             "elapsed": info["elapsed"], "completed_depth": info["completed_depth"],
                             "book_queried": info["book_queried"], "book_matched": info["book_matched"],
                             "query_seconds": (info.get("book_metadata") or {}).get("query_seconds"),
                             "budget_ok": info["budget_ok"]})
    summary = {}
    for mode in ("N", "D"):
        mine = [r for r in rows if r["mode"] == mode]
        depths = [r["completed_depth"] for r in mine if r["completed_depth"] is not None]
        queries = [r["query_seconds"] for r in mine if r["query_seconds"] is not None]
        summary[mode] = {"measurements": len(mine), "any_match": any(r["book_matched"] for r in mine),
                         "median_step_seconds": statistics.median(r["elapsed"] for r in mine) if mine else None,
                         "mean_completed_depth": statistics.mean(depths) if depths else None,
                         "median_query_seconds": statistics.median(queries) if queries else None,
                         "max_query_seconds": max(queries) if queries else None}
    return {"positions": len(states), "repeats": repeats, "summary": summary, "rows": rows}


def run(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tier = "smoke" if smoke else "formal"
    selection = load_json(config["selected_book"])
    if not smoke and selection.get("execution_tier") != "formal":
        raise ValueError("a formal step 6 needs a formal step-5 selection")
    online = load_online(config, smoke)
    if selection["online_variant"]["config_hash"] != online["config_hash"]:
        raise ValueError("step 5 used a different online configuration")
    _, reference_check = reference_games(config["d1_reference_games"], online)
    d0, d1 = manifest_rows(config, "D0", smoke), manifest_rows(config, "D1", smoke)
    opp = opponent(config, smoke)
    result = {"schema": SCHEMA_VERSION, "execution_tier": tier, "formal_selection": not smoke,
              "online_variant_id": online["variant_id"], "reference_check": reference_check, "opponent": opp}
    if selection["status"] != "selected":
        # Without a qualifying non-empty book only N remains, and V3 is omitted.
        final = {"status": "N_only", "reason": selection["status"], "mode": "N", "variant": online,
                 "book_comparison_variant": None, "execution_tier": tier}
        atomic_json(output / "final_config.json", final)
        atomic_json(output / "final_variant.json", online)
        result.update(status="N_only", reason=selection["status"])
        atomic_json(output / "mode_summary.json", result)
        return final
    book = load_book(selection["book_file"])
    if book["layout"]["hash"] != selection["hash"]:
        raise ValueError("selected book file does not match step 5's recorded hash")
    variants = {"N": online, "D": book_variant(online, book, "D"), "H": book_variant(online, book, "H")}
    if normalize_variant(variants["D"])["config_hash"] != normalize_variant(selection["variant_D"])["config_hash"]:
        raise ValueError("the D variant differs from the one step 5 played")
    reuse = {"N": [config["d1_reference_games"]], "D": [selection["d_games"]], "H": []}

    # 6.1.1 N/D/H on D1, reusing identical N and D games.
    d1_runs = {mode: run_pairs(variants[mode], d1, output / "d1" / mode, opponent=opp, execution_tier=tier,
                               reuse_paths=reuse[mode]) for mode in MODES}
    result["d1"] = {mode: dict(_brief(d1_runs[mode]), metrics=game_metrics(games_of(d1_runs[mode], output / "d1" / mode)))
                    for mode in MODES}

    # 6.1.2 keep N and the better of D/H on D1 (both on a tie); complete D0+D1 for them.
    keep = ["N"]
    books_ok = [m for m in ("D", "H") if d1_runs[m].get("eligible")]
    if books_ok:
        best = max(d1_runs[m]["score"] for m in books_ok)
        keep += [m for m in books_ok if d1_runs[m]["score"] == best]
    candidates = dict(variants)
    dev, dev_paths = {}, {}
    for mode in keep:
        dev_paths[mode] = output / "dev" / mode
        dev[mode] = run_pairs(variants[mode], d0 + d1, dev_paths[mode], opponent=opp, execution_tier=tier,
                              reuse_paths=[output / "d1" / mode / "games.jsonl"] + reuse[mode])
    order = rank(dev, ["N", "D", "H"])

    # 6.1.3 if H wins and step 5's top two sources tied in D mode, try the runner-up under H at k*.
    if order and order[0] == "H" and selection.get("source_tie") and selection.get("second_source_book_file"):
        other = load_book(selection["second_source_book_file"])
        candidates["H_second"] = book_variant(online, other, "H")
        trial = run_pairs(candidates["H_second"], d1, output / "d1" / "H_second", opponent=opp, execution_tier=tier)
        result["d1"]["H_second"] = dict(_brief(trial), metrics=game_metrics(games_of(trial, output / "d1" / "H_second")))
        if trial.get("eligible") and trial["score"] > d1_runs["H"]["score"]:
            dev_paths["H_second"] = output / "dev" / "H_second"
            dev["H_second"] = run_pairs(candidates["H_second"], d0 + d1, dev_paths["H_second"], opponent=opp,
                                        execution_tier=tier, reuse_paths=[output / "d1" / "H_second" / "games.jsonl"])
            order = rank(dev, ["N", "D", "H", "H_second"])
    result["development"] = {mode: dict(_brief(dev[mode]), metrics=game_metrics(games_of(dev[mode], dev_paths[mode])))
                             for mode in dev}
    result["development_ranking"] = order
    if "N" in dev:
        result["paired_difference_vs_N"] = {m: paired_difference(dev[m], dev["N"]) for m in dev if m != "N"}

    # 6.2 cost when every query misses: identical states and search, query on versus off.
    overhead = config.get("overhead", {})
    if smoke:
        overhead = dict(overhead, **config.get("smoke", {}).get("overhead", {}))
    result["all_miss_overhead"] = all_miss_overhead(config["positions"], book, variants,
                                                    overhead.get("repeats", 3), overhead.get("max_positions"))
    if not order:
        result["status"] = "blocked_technical_failure"
        atomic_json(output / "mode_summary.json", result)
        return result

    # Final configuration; if N wins, the best non-empty book variant is kept for step 7 (V3).
    chosen = order[0]
    comparison = None
    if chosen == "N":
        with_book = [m for m in order if m != "N"] or [m for m in rank(d1_runs, ["D", "H"])]
        comparison = normalize_variant(candidates[with_book[0]]) if with_book else None
    final_book = None if chosen == "N" else (other if chosen == "H_second" else book)
    final = {"status": "selected", "execution_tier": tier, "formal_selection": not smoke, "mode": chosen[0],
             "variant": normalize_variant(candidates[chosen]),
             "book_file": None if final_book is None else (selection["second_source_book_file"]
                                                           if chosen == "H_second" else selection["book_file"]),
             "book_hash": None if final_book is None else final_book["layout"]["hash"],
             "book_comparison_variant": comparison,
             "hashes": {"online_config": online["config_hash"],
                        "final_config": normalize_variant(candidates[chosen])["config_hash"]}}
    atomic_json(output / "final_config.json", final)
    atomic_json(output / "final_variant.json", final["variant"])
    if comparison is not None:
        atomic_json(output / "book_comparison_variant.json", comparison)
    result.update(status="selected", mode=final["mode"], selected=chosen)
    atomic_json(output / "mode_summary.json", result)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-book", help="override: step 5's selected_book.json")
    parser.add_argument("--online-config", help="override: F's fixed online configuration (step 4)")
    parser.add_argument("--manifest", help="override: A's manifest.json (step 0)")
    parser.add_argument("--references", help="override: F's D1 no-book games.jsonl (step 4)")
    parser.add_argument("--positions", help="override: A's fixed development positions.json (step 0)")
    parser.add_argument("--smoke", action="store_true", help="small self-test; never a formal selection")
    args = parser.parse_args()
    config = load_json(args.config)
    for name, value in (("selected_book", args.selected_book), ("online_config", args.online_config),
                        ("manifest", args.manifest), ("d1_reference_games", args.references),
                        ("positions", args.positions)):
        if value:
            config[name] = value
    final = run(config, args.output, args.smoke)
    print("status:", final["status"], "mode:", final.get("mode"))


if __name__ == "__main__":
    main()
