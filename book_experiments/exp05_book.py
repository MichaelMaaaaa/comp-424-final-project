"""Step 5 (B): build B0/B1/B2 books, compare sources and sizes, select one book file.

Reads F's online configuration and D1 no-book trajectories (step 4) and A's
manifest (step 0). --smoke shrinks budgets and counts; its outputs are marked
execution_tier=smoke and are never a formal selection.
"""

import argparse
from collections import Counter
from pathlib import Path

from book_experiments.analysis import analysis_identity, analyze_candidates, screen_record, SCREEN_VERSION
from book_experiments.books import (book_entries, book_table, book_variant, check_label, load_current_book,
                                    make_book, save_book, subset_book)
from book_experiments.candidates import build_pool, select_candidates, stage_distribution
from book_experiments.collect import run_collection
from book_experiments.common import (code_identity, games_of, load_online, manifest_rows, opponent,
                                     rank, reference_games)
from book_experiments.coverage import game_metrics, replay_coverage
from book_experiments.representation import compare_representations
from search_experiments.common import SCHEMA_VERSION, atomic_json, load_json, restore_move
from search_experiments.runner import run_pairs

SCHEMES = ("B0", "B1", "B2")
DEFAULTS = {
    "collection": {"move_time": 1.6, "opponent_budget": 1.5, "max_depth": None,
                   "b0_selfplay_per_config": 4, "b1_selfplay_per_config": 2, "b1_reuses_b0_selfplay": True},
    "candidates": {"nc": 600},
    "analysis": {"budget": 10.0, "max_depth": None},
    "screening": {"min_depth": 3, "stable_depths": 2},
    "export": {"size_reference": 486, "sizes": [128, 256]},
    "representation": {"repeats": 3},
}


def section(config, name, smoke):
    merged = dict(DEFAULTS[name])
    merged.update(config.get(name, {}))
    if smoke:
        merged.update(config.get("smoke", {}).get(name, {}))
    return merged


def export_books(selected, records, screening, label, size, output):
    """Each scheme exports `size` entries: accepted initial states first, then
    pool frequency (descending), completed depth (descending), key (ascending)."""
    books, export = {}, {}
    for scheme in SCHEMES:
        rows = selected[scheme]
        accepted = [c for c in rows if screening[tuple(c["key"])]["accepted"]]
        depth = lambda c: records[tuple(c["key"])]["completed_depth"]
        initial = sorted((c for c in accepted if c["initial"]), key=lambda c: c["key"])
        rest = sorted((c for c in accepted if not c["initial"]), key=lambda c: (-c["frequency"], -depth(c), c["key"]))
        chosen = (initial + rest)[:size]
        entries = [(tuple(c["key"]), restore_move(records[tuple(c["key"])]["move"])) for c in chosen]
        details = {tuple(c["key"]): {"frequency": c["frequency"], "ply": c["ply"], "stage": c["stage"],
                                     "initial": c["initial"], "completed_depth": depth(c),
                                     "value": records[tuple(c["key"])]["value"],
                                     "screening": screening[tuple(c["key"])]["reason"]} for c in chosen}
        record = make_book(entries, f"{label}_{scheme.lower()}_k{size}", details=details)
        path = output / "books" / f"{scheme}_k{size}.json"
        provider = save_book(record, path)
        books[scheme] = record
        export[scheme] = {"candidates": len(rows), "accepted": len(accepted), "exported": len(chosen),
                          "initial_exported": sum(c["initial"] for c in chosen),
                          "stage_distribution": {"candidates": stage_distribution(rows),
                                                 "accepted": stage_distribution(accepted),
                                                 "exported": stage_distribution(chosen)},
                          "book_file": str(path), "hash": record["layout"]["hash"],
                          "version": record["layout"]["version"], **provider}
    return books, export


def play_book(online, record, rows, output, opp, tier):
    summary = run_pairs(book_variant(online, record, "D"), rows, output, opponent=opp, execution_tier=tier)
    return summary, game_metrics(games_of(summary, output))


def _brief(summary):
    keys = ("complete", "eligible", "W", "D", "L", "score", "technical_failures", "request_hash")
    return {key: summary.get(key) for key in keys}


def run(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tier = "smoke" if smoke else "formal"
    online = load_online(config, smoke)
    d0, d1 = manifest_rows(config, "D0", smoke), manifest_rows(config, "D1", smoke)
    references, reference_check = reference_games(config["d1_reference_games"], online)
    if not smoke and any(r.get("execution_tier") != "formal" for r in references):
        raise ValueError("a formal step 5 needs formal D1 reference games")
    # Smoke books get their own module names, so they can never replace a formal provider.
    label = check_label(config.get("smoke", {}).get("book_label", "smoke") if smoke
                        else config.get("book_label", "formal"))
    identity = code_identity()
    opp = opponent(config, smoke)
    settings = {name: section(config, name, smoke) for name in DEFAULTS}
    report = {"schema": SCHEMA_VERSION, "execution_tier": tier, "formal_selection": not smoke,
              "online_variant_id": online["variant_id"], "online_config_hash": online["config_hash"],
              "code_hash": identity["code_hash"], "settings": settings, "opponent": opp,
              "reference_games": str(config["d1_reference_games"]), "reference_check": reference_check}
    repeats = settings["representation"]["repeats"]

    # 5A baseline on the book currently embedded in the original submission.
    current, current_version = load_current_book()
    report["representation_current"] = compare_representations(
        current, output / "representation" / "current", repeats, f"current_{current_version}")

    # 5B.1 collection on D0, then 5B.2 candidate pools and the common Nc.
    games, collection_cost = run_collection(d0, online, settings["collection"], output / "collection",
                                            identity["code_hash"])
    report["collection"] = collection_cost
    if not collection_cost["complete"]:
        raise RuntimeError("collection is incomplete; rerun the same command to resume")
    pools = {"B0": build_pool(games, "B0"), "B1": build_pool(games, "B1")}
    selected, candidate_summary = select_candidates(pools, settings["candidates"]["nc"])
    report["candidates"] = candidate_summary
    atomic_json(output / "candidates.json", dict(candidate_summary, selected=selected))

    # 5B.3 one analysis per distinct key, shared by the three schemes.
    analysis = analysis_identity(online, settings["analysis"]["budget"], settings["analysis"]["max_depth"])
    keys = {tuple(c["key"]) for rows in selected.values() for c in rows}
    records = analyze_candidates(keys, online, analysis, output / "analysis")
    nc = candidate_summary["nc"]
    report["analysis"] = {"analysis_id": analysis["analysis_id"], "identity": analysis,
                          "distinct_keys": len(keys),
                          "logical_budget_seconds": {s: nc * analysis["budget"] for s in SCHEMES},
                          "actual_seconds_deduplicated": sum(records[k]["elapsed"] for k in keys),
                          "termination": dict(Counter(records[k]["termination"] for k in keys))}

    # 5C.1 the same screening for every scheme.
    screen = settings["screening"]
    screening = {k: screen_record(records[k], analysis["analysis_id"], screen["min_depth"],
                                  screen["stable_depths"]) for k in keys}
    atomic_json(output / "screening.json", {"version": SCREEN_VERSION, "settings": screen,
                "decisions": [dict(screening[k], key=list(k)) for k in sorted(keys)]})
    accepted = {s: sum(screening[tuple(c["key"])]["accepted"] for c in selected[s]) for s in SCHEMES}
    report["screening"] = {"version": SCREEN_VERSION, "accepted": accepted,
                           "reasons": {s: dict(Counter(screening[tuple(c["key"])]["reason"] for c in selected[s]))
                                       for s in SCHEMES}}

    # 5C.2 a common entry count K for the three books.
    size = min(settings["export"]["size_reference"], *accepted.values())
    report["K"] = size
    selection = {"schema": SCHEMA_VERSION, "execution_tier": tier, "formal_selection": not smoke,
                 "online_variant": online, "K": size, "selected_book": None, "mode_if_no_book": "N"}
    if size == 0:
        selection["status"] = "no_common_book"
        report["status"] = "no_common_book"
        atomic_json(output / "book_summary.json", report)
        atomic_json(output / "selected_book.json", selection)
        return selection
    books, export = export_books(selected, records, screening, label, size, output)
    report["export"] = export

    # 5C.3 sources: potential coverage on the D1 trajectories, then D-mode games on D1.
    report["source_coverage"] = {s: replay_coverage(book_table(books[s]), references) for s in SCHEMES}
    sources, source_metrics = {}, {}
    for scheme in SCHEMES:
        sources[scheme], source_metrics[scheme] = play_book(online, books[scheme], d1,
                                                            output / "games" / f"source_{scheme}", opp, tier)
    order = rank(sources, list(SCHEMES))
    report["source_games"] = {s: dict(_brief(sources[s]), metrics=source_metrics[s]) for s in SCHEMES}
    report["source_ranking"] = order
    if not order:
        report["status"] = "blocked_technical_failure"
        selection["status"] = "blocked_technical_failure"
        atomic_json(output / "book_summary.json", report)
        atomic_json(output / "selected_book.json", selection)
        return selection
    winner = order[0]
    second = order[1] if len(order) > 1 else None
    tie = second is not None and sources[winner]["score"] == sources[second]["score"]

    # 5C.4 nested sizes from the winner; a size without new coverage keeps the smaller one.
    sizes = sorted({min(n, size) for n in settings["export"]["sizes"]} | {size})
    candidates = {n: books[winner] if n == size else subset_book(books[winner], n, f"{label}_{winner.lower()}_k{n}")
                  for n in sizes}
    for n, record in candidates.items():
        if n != size:
            save_book(record, output / "books" / f"{winner}_k{n}.json")
    coverage = {n: replay_coverage(book_table(candidates[n]), references) for n in sizes}
    kept = []
    for n in sizes:
        if kept and set(coverage[n]["matched_turns"]) == set(coverage[kept[-1]]["matched_turns"]):
            continue
        kept.append(n)
    size_summaries = {size: sources[winner]} if size in kept else {}
    size_paths = {size: output / "games" / f"source_{winner}"}
    size_metrics = {size: source_metrics[winner]}
    for n in kept:
        if n not in size_summaries:
            size_paths[n] = output / "games" / f"size_{winner}_k{n}"
            size_summaries[n], size_metrics[n] = play_book(online, candidates[n], d1, size_paths[n], opp, tier)
    size_order = sorted((n for n in kept if size_summaries[n].get("eligible")),
                        key=lambda n: (-size_summaries[n]["score"], n))
    report["size_coverage"] = {str(n): coverage[n] for n in sizes}
    report["sizes_with_new_coverage"] = kept
    report["size_games"] = {str(n): dict(_brief(size_summaries[n]), metrics=size_metrics[n]) for n in kept}
    if not size_order:
        report["status"] = "blocked_technical_failure"
        selection["status"] = "blocked_technical_failure"
        atomic_json(output / "book_summary.json", report)
        atomic_json(output / "selected_book.json", selection)
        return selection
    k_star = size_order[0]
    chosen = candidates[k_star]
    chosen_file = output / "books" / f"{winner}_k{k_star}.json"

    # 6.1 step 3 may compare the runner-up source under H at k*, so export it now.
    second_file = None
    if tie:
        second_book = books[second] if k_star == size else subset_book(
            books[second], k_star, f"{label}_{second.lower()}_k{k_star}")
        second_file = output / "books" / f"{second}_k{k_star}.json"
        save_book(second_book, second_file)

    # 5A again on the selected book: same entries and moves, only the representation changes.
    report["representation_selected"] = compare_representations(
        book_entries(chosen), output / "representation" / "selected", repeats, chosen["label"])
    selection.update(status="selected", source=winner, second_source=second, source_tie=tie,
                     k_star=k_star, book_file=str(chosen_file), label=chosen["label"],
                     provider=chosen["provider"], hash=chosen["layout"]["hash"],
                     version=chosen["layout"]["version"], entries=chosen["layout"]["entries"],
                     variant_D=book_variant(online, chosen, "D"), d_games=str(size_paths[k_star] / "games.jsonl"),
                     second_source_book_file=None if second_file is None else str(second_file),
                     reference_games=str(config["d1_reference_games"]))
    report.update(status="selected", source=winner, second_source=second, source_tie=tie, k_star=k_star,
                  selected_book_file=str(chosen_file))
    atomic_json(output / "book_summary.json", report)
    atomic_json(output / "selected_book.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--online-config", help="override: F's fixed online configuration (step 4)")
    parser.add_argument("--manifest", help="override: A's manifest.json (step 0)")
    parser.add_argument("--references", help="override: F's D1 no-book games.jsonl (step 4)")
    parser.add_argument("--smoke", action="store_true", help="small self-test; never a formal selection")
    args = parser.parse_args()
    config = load_json(args.config)
    for name, value in (("online_config", args.online_config), ("manifest", args.manifest),
                        ("d1_reference_games", args.references)):
        if value:
            config[name] = value
    selection = run(config, args.output, args.smoke)
    print("status:", selection["status"], "K:", selection["K"], "source:", selection.get("source"),
          "k*:", selection.get("k_star"))


if __name__ == "__main__":
    main()
