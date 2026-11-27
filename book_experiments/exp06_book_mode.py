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

from agents.game_state import initial_state, legal_moves
from book_experiments.books import book_table, book_variant, lookup, write_provider
from book_experiments.common import (games_of, handoff_identity, load_online, manifest_rows, opponent,
                                     rank, reference_games, validate_empty_book, validate_match_evidence,
                                     validate_selected_exports)
from book_experiments.coverage import game_metrics
from search_experiments.common import (SCHEMA_VERSION, atomic_json, digest, file_hash, load_json,
                                       normalize_variant, restore_state)
from search_experiments.development import bind_request, load_development_rows
from search_experiments.measure import load_positions, measure_online_step
from search_experiments.runner import paired_difference, run_pairs

MODES = ("N", "D", "H")


def _brief(summary):
    keys = ("complete", "eligible", "W", "D", "L", "score", "technical_failures", "request_hash")
    return {key: summary.get(key) for key in keys}


def all_miss_overhead(positions, book, variants, repeats=3, max_positions=None):
    """Same development states, all missing the book: query on (D) versus no query (N)."""
    table = book_table(book)
    states = []
    for position in positions:
        state = restore_state(position["state"])
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


def _publish(output, final, result, status, reason=None, **roles):
    formal = final["execution_tier"] == "formal" and status in ("selected", "N_only")
    state = dict(status=status, reason=reason, formal_selection=formal)
    payload = {**final, **state, **roles}
    for name, variant in (("final_variant.json", payload["variant"]),
                          ("book_comparison_variant.json", payload["book_comparison_variant"])):
        path = output / name
        if variant is None:
            path.unlink(missing_ok=True)
        else:
            atomic_json(path, variant)
    atomic_json(output / "final_config.json", payload)
    atomic_json(output / "mode_summary.json", dict(result, **state))
    return payload


def _check_overhead_positions(positions, manifest_path):
    _, _, _, development = load_development_rows(manifest_path)
    by_config = {row["config_id"]: row for row in development}
    for position in positions:
        row = by_config.get(position.get("config_id"))
        if row is None or any(position.get(key) != row[key] for key in ("split", "removed_size", "obstacle_class")):
            raise ValueError("overhead position does not match a frozen development configuration")
        state = restore_state(position["state"])
        start = initial_state(row["obstacle"], row["removed_size"])
        if state.obstacle != start.obstacle:
            raise ValueError("overhead position has the wrong obstacle")
        # Placing or moving a piece preserves each player's total stock, including covered pieces.
        for player in (1, 2):
            stock = tuple(count + sum(stack[size] == player for stack in state.board)
                          for size, count in enumerate(state.reserves[player]))
            if stock != start.reserves[player]:
                raise ValueError("overhead position has the wrong piece inventory for its configuration")


def run(config, output, smoke=False):
    output = Path(output).resolve()
    tier = "smoke" if smoke else "formal"
    online = load_online(config, smoke)
    opp = opponent(config, smoke)
    d0, d1 = manifest_rows(config, "D0", smoke), manifest_rows(config, "D1", smoke)
    _, reference = reference_games(config["d1_reference_games"], online, config, smoke)
    identity = handoff_identity(config, online, tier, opp, d1[0]["manifest_hash"])

    step5_path = Path(config["selected_book"]).resolve()
    selection = load_json(step5_path)
    status = selection.get("status")
    formal = not smoke and status in ("selected", "no_common_book")
    if (any(key not in selection or digest(selection[key]) != digest(value) for key, value in identity.items())
            or selection.get("formal_selection") is not formal or selection.get("d1_reference") != reference):
        raise ValueError("Step 5 selection requires the same code, tier, manifest, machine and online/D1 inputs")
    final = {"schema": SCHEMA_VERSION, **identity, "selected_book_path": str(step5_path),
             "selected_book_hash": file_hash(step5_path), "selected_book_status": status, "K": selection.get("K"),
             "mode": None, "variant": None, "book_file": None, "book_hash": None,
             "book_comparison_variant": None, "book_comparison_file": None, "book_comparison_hash": None,
             "development_evidence": {}, "d1_reference": reference}
    result = {"schema": SCHEMA_VERSION, **identity, "online_variant_id": online["variant_id"],
              "reference_check": reference, "selected_book_status": status}
    if status not in ("selected", "no_common_book"):
        if (output / "request.json").exists():
            raise ValueError("development selection request changed; use a new output directory")
        return _publish(output, final, result, "pending" if status == "pending" else "blocked",
                        f"step5_{status or 'missing_status'}")

    manifest = Path(config["manifest"]).resolve()
    request = {"stage": "book_mode", **identity, "manifest_path": str(manifest),
               "manifest_file_hash": file_hash(manifest), "d0": d0, "d1_reference": reference,
               "selected_book_path": final["selected_book_path"], "selected_book_hash": final["selected_book_hash"]}
    if status == "no_common_book":
        validate_empty_book(selection)
    else:
        book, descriptor, other, other_descriptor = validate_selected_exports(selection, online)
        d_games = Path(selection["d_games"]).resolve()
        d_inputs = (d_games, d_games.with_name("plan.json"), d_games.with_name("summary.json"))
        positions_path = Path(config["positions"]).resolve()
        positions, payload = load_positions(positions_path, require_formal=not smoke)
        if payload.get("execution_tier") != tier or payload.get("manifest_hash") != identity["manifest_hash"]:
            raise ValueError("Step 6 overhead requires same-tier positions from the frozen development manifest")
        _check_overhead_positions(positions, manifest)
        overhead = dict(config.get("overhead", {}))
        if smoke:
            overhead.update(config.get("smoke", {}).get("overhead", {}))
        request.update(book=descriptor, second_book=other_descriptor, overhead=overhead,
                       inputs={str(path): file_hash(path) for path in (*d_inputs, positions_path)})
    bind_request(output, request)
    try:
        _publish(output, final, result, "pending", "book_mode_started")
        if status == "no_common_book":
            return _publish(output, final, result, "N_only", "no_common_book", mode="N", variant=online)
        write_provider(book)
        validate_match_evidence(d_games, descriptor["variant_D"], d1, opp, tier)
        return _run(output, selection, final, result, book, other, other_descriptor, positions, overhead, d0, d1,
                    {"N": [reference["games_path"]], "D": [d_games], "H": []})
    except (KeyboardInterrupt, SystemExit):
        _publish(output, final, result, "pending", "book_mode_interrupted")
        raise
    except Exception as error:
        _publish(output, final, result, "blocked", f"{type(error).__name__}: {error}")
        raise


def _run(output, selection, final, result, book, other, other_descriptor, positions, overhead, d0, d1, reuse):
    online = final["online_variant"]
    opp, tier = final["opponent"], final["execution_tier"]
    variants = {"N": online, **{mode: normalize_variant(book_variant(online, book, mode)) for mode in ("D", "H")}}
    candidates = dict(variants)
    d1_brief = result["d1"] = {}
    dev_brief = result["development"] = {}
    evidence = {}

    def play(mode, rows, path, reuse_paths, summaries, confirmed=False):
        variant = candidates[mode]
        summary = run_pairs(variant, rows, path, opponent=opp, execution_tier=tier, reuse_paths=reuse_paths)
        records = games_of(summary, path)
        if summary["complete"] and summary["eligible"]:
            records, descriptor = validate_match_evidence(path / "games.jsonl", variant, rows, opp, tier)
            if confirmed:
                evidence[variant["variant_id"]] = descriptor
        summaries[mode] = dict(_brief(summary), metrics=game_metrics(records))
        return summary

    # 6.1.1 N/D/H on D1, reusing identical N and D games.
    d1_runs = {mode: play(mode, d1, output / "d1" / mode, reuse[mode], d1_brief) for mode in MODES}
    if not all(summary["complete"] for summary in d1_runs.values()):
        return _publish(output, final, result, "pending", "d1_mode_comparison_incomplete")

    # 6.1.2 Keep N and the highest eligible D1 book mode, or both on a tie.
    keep = ["N"]
    books_ok = rank(d1_runs, ["D", "H"])
    if books_ok:
        best = d1_runs[books_ok[0]]["score"]
        keep += [mode for mode in ("D", "H") if mode in books_ok and d1_runs[mode]["score"] == best]
    dev = {mode: play(mode, d0 + d1, output / "dev" / mode,
                     [output / "d1" / mode / "games.jsonl", *reuse[mode]], dev_brief, confirmed=True) for mode in keep}
    if not all(summary["complete"] for summary in dev.values()):
        return _publish(output, final, result, "pending", "development_confirmation_incomplete")
    order = rank(dev, ["N", "D", "H"])

    # 6.1.3 Only an H winner with a Step 5 source tie can trigger this recheck.
    if order and order[0] == "H" and other is not None:
        write_provider(other)
        candidates["H_second"] = normalize_variant(book_variant(online, other, "H"))
        other_d1 = output / "d1" / "H_second"
        trial = play("H_second", d1, other_d1, [], d1_brief)
        if not trial["complete"]:
            return _publish(output, final, result, "pending", "second_source_d1_incomplete")
        if trial["eligible"] and trial["score"] > d1_runs["H"]["score"]:
            trial = play("H_second", d0 + d1, output / "dev" / "H_second",
                         [other_d1 / "games.jsonl"], dev_brief, confirmed=True)
            if not trial["complete"]:
                return _publish(output, final, result, "pending", "second_source_confirmation_incomplete")
            dev["H_second"] = trial
            order = rank(dev, ["N", "D", "H", "H_second"])

    result["development_ranking"] = order
    result["paired_difference_vs_N"] = {mode: paired_difference(dev[mode], dev["N"])
                                       for mode in dev if mode != "N"}

    # 6.2 Measure all misses on identical fixed states.
    result["all_miss_overhead"] = all_miss_overhead(positions, book, variants,
                                                    overhead.get("repeats", 3), overhead.get("max_positions"))
    if not order:
        return _publish(output, final, result, "blocked", "no_eligible_development_variant")

    chosen = order[0]
    final_book = None
    control = None
    if chosen == "N":
        controls = [mode for mode in order if mode != "N"]
        if not controls:
            return _publish(output, final, result, "blocked", "no_eligible_full_development_book_control")
        control = candidates[controls[0]]
    else:
        final_book = other_descriptor if chosen == "H_second" else selection["selected_book"]
    variant = candidates[chosen]
    result.update(mode=chosen[0], selected=chosen)
    return _publish(output, final, result, "selected", mode=chosen[0], variant=variant,
                    book_file=final_book["book_file"] if final_book else None,
                    book_hash=final_book["hash"] if final_book else None, book_comparison_variant=control,
                    book_comparison_file=selection["book_file"] if control else None,
                    book_comparison_hash=selection["hash"] if control else None,
                    development_evidence=evidence, hashes={"online_config": online["config_hash"],
                                                           "final_config": variant["config_hash"]})


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
