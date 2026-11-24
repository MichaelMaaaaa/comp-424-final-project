"""Analyze canonical candidates with the existing search, without book lookup.

Moves use canonical coordinates; values use the state.player perspective.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import time

from constants import BOARD_SIZE, MAX_PLIES, NUM_SIZES
from agents import evaluation
from agents.iterative_deepening import search
from agents.symmetry import BOOK_KEY_SIZE, canonicalize, decode_book_state
from collect_positions import check_output_path, save_collection


# Update these labels when the rule or evaluation definitions change.
RULE_VERSION = "goblet-course-v1"
EVALUATION_VERSION = "e1-v1"


def current_versions():
    return {
        "rules": {
            "version": RULE_VERSION, "board_size": BOARD_SIZE, "num_sizes": NUM_SIZES,
            "allow_moves": True, "win_length": 4, "max_plies": MAX_PLIES,
        },
        "evaluation": {
            "version": EVALUATION_VERSION, "line_weight": evaluation.LINE_WEIGHT,
            "size_weights": evaluation.SIZE_WEIGHTS, "win_score": evaluation.WIN_SCORE,
        },
    }


def _metadata_equal(first, second):
    """Normalize JSON sequences while keeping booleans and numbers distinct."""
    try:
        return (json.dumps(first, sort_keys=True, allow_nan=False)
                == json.dumps(second, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return False


def _check_time_limit(time_limit):
    if (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
            or not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be a finite nonnegative number")


def _canonical_state(key):
    if not isinstance(key, (list, tuple)):
        raise ValueError("a candidate key must be a list or tuple of integers")
    state = decode_book_state(key)
    if canonicalize(state)[0] != tuple(key):
        raise ValueError("candidate key is not canonical")
    return state


def _move_record(move):
    if move is None:
        return None
    return {"src": move.src, "dest": move.dest, "size": move.size}


def analyze_position(key, time_limit=10.0):
    """Record all completed depths; the budget includes decoding and search."""
    started = time.perf_counter()
    _check_time_limit(time_limit)
    state = _canonical_state(key)
    versions = current_versions()
    result = search(state, started + time_limit)

    return {
        "key": tuple(key),
        "time_limit": time_limit,
        "versions": versions,
        "move": _move_record(result.move),
        "completed_depth": result.completed_depth,
        "value": result.value,
        "status": result.status,
        "nodes": result.nodes,
        "elapsed": time.perf_counter() - started,
        "iterations": [
            {"depth": depth, "move": _move_record(move), "value": value}
            for depth, move, value in result.iterations
        ],
    }


def _validate_collection(collection):
    """Check source links before spending time on any candidate."""
    required = {"format_version", "rules", "evaluation", "settings", "games", "positions"}
    if not isinstance(collection, dict) or not required.issubset(collection):
        raise ValueError("input is not a complete collection record")
    if type(collection["format_version"]) is not int or collection["format_version"] != 1:
        raise ValueError("unsupported collection format version")
    rules = {"allow_moves": True, "win_length": 4, "max_plies": MAX_PLIES}
    if not _metadata_equal(collection["rules"], rules):
        raise ValueError("collection rules do not match the official True/4 rules")
    if not isinstance(collection["evaluation"], dict) or not isinstance(collection["settings"], dict):
        raise ValueError("collection metadata must be dictionaries")
    if not isinstance(collection["games"], list) or not isinstance(collection["positions"], list):
        raise ValueError("games and positions must be lists")

    raw_states = {}
    game_ids = set()
    for game in collection["games"]:
        if not isinstance(game, dict):
            raise ValueError("invalid game record")
        game_id = game.get("game_id")
        if type(game_id) is not int or game_id < 0 or game_id in game_ids:
            raise ValueError("game IDs must be distinct nonnegative integers")
        game_ids.add(game_id)
        if not isinstance(game.get("turns"), list):
            raise ValueError("a game must contain a list of turns")
        for turn in game["turns"]:
            raw_key = turn.get("state") if isinstance(turn, dict) else None
            if not isinstance(raw_key, (list, tuple)) or len(raw_key) != BOOK_KEY_SIZE:
                raise ValueError("a recorded turn must contain a complete state key")
            ply = raw_key[-1]
            if type(ply) is not int or ply < 0 or (game_id, ply) in raw_states:
                raise ValueError("each game/ply must identify exactly one recorded state")
            raw_states[(game_id, ply)] = raw_key

    seen_keys = set()
    seen_occurrences = set()
    for position in collection["positions"]:
        if not isinstance(position, dict):
            raise ValueError("invalid candidate record")
        _canonical_state(position.get("key"))
        key = tuple(position["key"])
        if key in seen_keys:
            raise ValueError("collection contains duplicate canonical keys")
        seen_keys.add(key)
        count = position.get("count")
        occurrences = position.get("occurrences")
        if (type(count) is not int or count <= 0 or not isinstance(occurrences, list)
                or count != len(occurrences)):
            raise ValueError("candidate count must match its nonempty occurrence list")
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                raise ValueError("invalid occurrence record")
            game_id = occurrence.get("game_id")
            ply = occurrence.get("ply")
            transform = occurrence.get("transform")
            if (type(game_id) is not int or type(ply) is not int
                    or type(transform) is not int or not 0 <= transform < 8):
                raise ValueError("invalid occurrence identity or transform")
            identity = (game_id, ply)
            if identity not in raw_states or identity in seen_occurrences:
                raise ValueError("occurrence source is missing or repeated")
            seen_occurrences.add(identity)
            raw_state = decode_book_state(raw_states[identity])
            if canonicalize(raw_state) != (key, transform):
                raise ValueError("occurrence does not match its canonical candidate")


def analyze_positions(collection, time_limit=10.0, limit=None, progress=False):
    """Analyze candidates in opening/frequency order and retain their provenance."""
    _check_time_limit(time_limit)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    _validate_collection(collection)
    candidates = sorted(collection["positions"],
                        key=lambda item: (item["key"][-1], -item["count"], tuple(item["key"])))
    selected = candidates if limit is None else candidates[:limit]
    source = {name: copy.deepcopy(value) for name, value in collection.items()
              if name != "positions"}
    records = []

    for index, position in enumerate(selected, start=1):
        record = analyze_position(position["key"], time_limit)
        record["count"] = position["count"]
        record["occurrences"] = copy.deepcopy(position["occurrences"])
        records.append(record)
        if progress:
            print(f"Analyzed {index}/{len(selected)}: ply {record['key'][-1]}, "
                  f"depth {record['completed_depth']}, status {record['status']}", flush=True)

    return {
        "format_version": 1,
        "source_collection": source,
        "settings": {
            "time_limit": time_limit, "limit": limit,
            "total_candidates": len(candidates), "analyzed_candidates": len(records),
        },
        "positions": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="collected_positions.json")
    parser.add_argument("--output", default="analyzed_positions.json")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("input and output must be different files")

    try:
        check_output_path(args.output)
        with open(args.input, encoding="utf-8") as source:
            collection = json.load(source)
        analysis = analyze_positions(collection, args.time_limit, args.limit, progress=True)
        save_collection(analysis, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {len(analysis['positions'])} analyzed positions to {args.output}")


if __name__ == "__main__":
    main()
