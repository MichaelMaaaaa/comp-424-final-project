"""5A: compare book representations on the same position-action data.

Three controlled comparisons, each changing one property against the
baseline (D4 canonical entries, 15-byte compact keys, zlib):
  symmetry     expanded orientations vs D4   -> entries, size, query time
  key          tuple keys vs compact keys    -> key/query time, decoded memory
  compression  uncompressed vs zlib          -> init time, size, decoded memory
Every representation must return the same legal move for every covered
actual state, and the same miss for every uncovered one.
"""

import csv
import statistics
import time
import tracemalloc

from agents.book_policy import book_decode, book_lookup, book_query_key
from agents.game_state import apply_move, legal_moves, terminal
from agents.symmetry import (TRANSFORMS, canonicalize, decode_book_state, encode_book_state,
                             transform_state)
from book_experiments.books import encode_layout, render_provider
from search_experiments.common import atomic_json

BASELINE = "d4_compact_zlib"
LAYOUTS = {
    "d4_compact_zlib": {"symmetry": "d4", "key": "compact", "compression": "zlib"},
    "expanded_compact_zlib": {"symmetry": "expanded", "key": "compact", "compression": "zlib"},
    "d4_tuple_zlib": {"symmetry": "d4", "key": "tuple", "compression": "zlib"},
    "d4_compact_none": {"symmetry": "d4", "key": "compact", "compression": "none"},
}
COMPARISONS = [
    ("symmetry", "expanded_compact_zlib", ["entries", "data_bytes", "module_bytes", "query_us"]),
    ("key", "d4_tuple_zlib", ["key_us", "query_us", "decoded_bytes"]),
    ("compression", "d4_compact_none", ["init_ms", "data_bytes", "module_bytes", "decoded_bytes"]),
]


def query_states(entries):
    """All actual orientations of covered states, plus one uncovered child per entry."""
    covered, seen = [], set()
    for key, _ in entries:
        state = decode_book_state(key)
        for transform in TRANSFORMS:
            actual = transform_state(state, transform)
            raw = encode_book_state(actual)
            if raw not in seen:
                seen.add(raw)
                covered.append(actual)
    keys = {tuple(k) for k, _ in entries}
    misses = []
    for key, _ in entries:
        state = decode_book_state(key)
        for move in legal_moves(state):
            child = apply_move(state, move)
            if not terminal(child)[0] and canonicalize(child)[0] not in keys:
                misses.append(child)
                break
    return covered, misses


def _timed(function, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def compare_representations(entries, output, repeats=3, label="book"):
    """Measure all layouts; raise if any layout changes a returned move."""
    if not entries:
        return {"status": "empty_book", "label": label}
    covered, misses = query_states(entries)
    queries = covered + misses
    encoded = {name: encode_layout(entries, **layout) for name, layout in LAYOUTS.items()}
    tables, rows = {}, {}
    for name, (data, layout) in encoded.items():
        tracemalloc.start()
        table = book_decode(data, layout)
        decoded_bytes = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        tables[name] = (table, layout)
        module = render_provider({"label": "measure", "layout": layout, "data": list(data)})
        rows[name] = {"layout": name, **{k: layout[k] for k in ("symmetry", "key", "compression")},
                      "entries": layout["entries"], "data_bytes": sum(map(len, data)),
                      "module_bytes": len(module.encode()), "decoded_bytes": decoded_bytes,
                      "init_ms": None, "key_us": None, "query_us": None}
    reference = [book_lookup(*tables[BASELINE], state) for state in queries]
    for name, (table, layout) in tables.items():
        answers = [book_lookup(table, layout, state) for state in queries]
        if answers != reference:
            raise AssertionError(f"layout {name} changed a returned move")
    if any(move is None or move not in legal_moves(state) for state, move in zip(covered, reference)):
        raise AssertionError("a covered state returned no legal move")
    if any(move is not None for move in reference[len(covered):]):
        raise AssertionError("an uncovered state returned a move")
    # Alternate the layout order between repetitions so drift affects all equally.
    names = list(LAYOUTS)
    samples = {name: {"init_ms": [], "key_us": [], "query_us": []} for name in names}
    for repetition in range(repeats):
        for name in (names if repetition % 2 == 0 else list(reversed(names))):
            data, layout = encoded[name]
            table = tables[name][0]
            samples[name]["init_ms"].append(1000 * _timed(lambda: book_decode(data, layout), 1))
            samples[name]["key_us"].append(1e6 * _timed(
                lambda: [book_query_key(state, layout) for state in queries], 1) / len(queries))
            samples[name]["query_us"].append(1e6 * _timed(
                lambda: [book_lookup(table, layout, state) for state in queries], 1) / len(queries))
    for name in names:
        for metric, values in samples[name].items():
            rows[name][metric] = statistics.median(values)
    comparisons = []
    for prop, name, metrics in COMPARISONS:
        comparisons.append({"property": prop, "baseline": BASELINE, "alternative": name,
                            "metrics": {m: {"baseline": rows[BASELINE][m], "alternative": rows[name][m]}
                                        for m in metrics}})
    result = {"status": "measured", "label": label, "canonical_entries": len(entries),
              "covered_queries": len(covered), "miss_queries": len(misses), "repeats": repeats,
              "identical_moves": True, "layouts": rows, "comparisons": comparisons}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "representation.json", result)
    with (output / "representation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    return result
