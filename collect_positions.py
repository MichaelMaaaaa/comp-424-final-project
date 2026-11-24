"""Collect full self-play positions for offline position-book analysis."""

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import tempfile
import time

from constants import BOARD_SIZE, MAX_PLIES, NUM_SIZES
from helpers import Move
from agents.alpha_beta import SearchStats, alpha_beta
from agents.evaluation import LINE_WEIGHT, SIZE_WEIGHTS, WIN_SCORE, terminal_utility
from agents.game_state import StateTimeout, apply_move, initial_state, legal_moves, terminal
from agents.move_ordering import order_moves
from agents.symmetry import canonicalize, decode_book_state, encode_book_state
from agents.transposition_table import TranspositionTable


@dataclass
class CollectionSearchResult:
    move: Move | None = None
    completed_depth: int = 0
    value: int | None = None
    tied_moves: list = field(default_factory=list)
    nodes: int = 0
    elapsed: float = 0.0
    status: str = "fallback"


def _search_root_ties(state, depth, deadline, preferred_move, stats, tt):
    """Return the exact best value and all best moves at one positive depth."""
    stats.nodes += 1
    best_value = -float("inf")
    best_moves = []

    for move, child in order_moves(state, preferred_move, deadline):
        # Full child windows distinguish genuine ties from cutoff bounds.
        child_value, _, _ = alpha_beta(
            child, depth - 1, -float("inf"), float("inf"),
            deadline=deadline, stats=stats, tt=tt
        )
        value = -child_value
        if value > best_value:
            best_value = value
            best_moves = [move]
        elif value == best_value:
            best_moves.append(move)

    if time.perf_counter() >= deadline:
        raise StateTimeout("collection root search exceeded its deadline")
    return best_value, best_moves


def search_with_ties(state, deadline, rng):
    """Choose among the last complete depth's best moves; deadline is absolute."""
    started = time.perf_counter()
    encode_book_state(state)  # Reject rule variants incompatible with the book.
    finished, winner, _ = terminal(state)
    if finished:
        return CollectionSearchResult(
            value=terminal_utility(state, state.player, winner),
            elapsed=time.perf_counter() - started, status="terminal"
        )

    result = CollectionSearchResult(move=legal_moves(state)[0])
    stats = SearchStats()
    tt = TranspositionTable()
    preferred_move = None

    for depth in range(1, MAX_PLIES - state.ply + 1):
        if time.perf_counter() >= deadline:
            break
        try:
            value, moves = _search_root_ties(
                state, depth, deadline, preferred_move, stats, tt
            )
        except StateTimeout:
            break

        # Partial or late passes must not replace the last complete tie set.
        if time.perf_counter() >= deadline:
            break
        result.completed_depth = depth
        result.value = value
        result.tied_moves = moves
        result.status = "completed"
        preferred_move = moves[0]

    if result.tied_moves:
        result.move = rng.choice(result.tied_moves)
    result.nodes = stats.nodes
    result.elapsed = time.perf_counter() - started
    return result


def _check_settings(move_time, seed):
    if (isinstance(move_time, bool) or not isinstance(move_time, (int, float))
            or not math.isfinite(move_time) or move_time < 0):
        raise ValueError("move_time must be a finite nonnegative number")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _move_record(move):
    return {"src": move.src, "dest": move.dest, "size": move.size}


def play_game(state, move_time=1.6, seed=0):
    """Play from a full state until the official terminal condition is reached."""
    _check_settings(move_time, seed)
    first_state = encode_book_state(state)
    rng = random.Random(seed)
    turns = []

    while True:
        finished, winner, reason = terminal(state)
        if finished:
            break
        result = search_with_ties(state, time.perf_counter() + move_time, rng)
        if result.move not in legal_moves(state):
            raise RuntimeError("collection search returned an illegal move")
        turns.append({
            "state": encode_book_state(state),
            "move": _move_record(result.move),
            "completed_depth": result.completed_depth,
            "value": result.value,
            "tied_moves": [_move_record(move) for move in result.tied_moves],
            "status": result.status,
            "nodes": result.nodes,
            "elapsed": result.elapsed,
        })
        state = apply_move(state, result.move)

    return {
        "seed": seed,
        "initial_state": first_state,
        "final_state": encode_book_state(state),
        "turns": turns,
        "winner": winner,
        "reason": reason,
        "plies": state.ply,
    }


def collect_positions(configurations=None, games_per_config=1, move_time=1.6, seed=0):
    """Collect candidates; configurations contains ((row, col), removed_size)."""
    _check_settings(move_time, seed)
    if (isinstance(games_per_config, bool) or not isinstance(games_per_config, int)
            or games_per_config <= 0):
        raise ValueError("games_per_config must be a positive integer")
    if configurations is None:
        configurations = [((row, col), size)
                          for row in range(BOARD_SIZE)
                          for col in range(BOARD_SIZE)
                          for size in range(NUM_SIZES)]
    configurations = list(configurations)
    if not configurations:
        raise ValueError("at least one initial configuration is required")

    # Validate every start before running any games.
    starts = []
    for obstacle, removed_size in configurations:
        state = initial_state(obstacle, removed_size)
        encode_book_state(state)
        starts.append(state)

    games = []
    positions = {}
    for start, (_, removed_size) in zip(starts, configurations):
        for _ in range(games_per_config):
            game_id = len(games)
            game = play_game(start, move_time, seed + game_id)
            game.update(game_id=game_id, obstacle=start.obstacle,
                        removed_size=int(removed_size))
            games.append(game)

            for turn in game["turns"]:
                state = decode_book_state(turn["state"])
                key, transform = canonicalize(state)
                if key not in positions:
                    positions[key] = {"key": key, "count": 0, "occurrences": []}
                entry = positions[key]
                entry["count"] += 1
                entry["occurrences"].append({
                    "game_id": game_id, "ply": state.ply, "transform": transform
                })

    candidates = sorted(positions.values(),
                        key=lambda item: (item["key"][-1], -item["count"], item["key"]))
    return {
        "format_version": 1,
        "rules": {"allow_moves": True, "win_length": 4, "max_plies": MAX_PLIES},
        "evaluation": {"line_weight": LINE_WEIGHT, "size_weights": SIZE_WEIGHTS,
                       "win_score": WIN_SCORE},
        "settings": {
            "games_per_config": games_per_config, "move_time": move_time, "seed": seed,
            "configurations": [
                {"obstacle": start.obstacle, "removed_size": int(size)}
                for start, (_, size) in zip(starts, configurations)
            ],
        },
        "games": games,
        "positions": candidates,
    }


def check_output_path(path):
    """Check the destination before computation, without changing an existing file."""
    path = Path(path)
    if path.exists() and not path.is_file():
        raise ValueError("output path must name a file")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp"):
        pass


def save_collection(collection, path):
    """Replace the destination only after the complete JSON has been written."""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            json.dump(collection, output, indent=2)
            output.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-config", type=int, default=1)
    parser.add_argument("--move-time", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacle", type=int, nargs=2, metavar=("ROW", "COL"))
    parser.add_argument("--removed-size", type=int, choices=range(NUM_SIZES))
    parser.add_argument("--output", default="collected_positions.json")
    args = parser.parse_args()

    obstacles = [tuple(args.obstacle)] if args.obstacle is not None else [
        (row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
    ]
    sizes = [args.removed_size] if args.removed_size is not None else range(NUM_SIZES)
    configurations = [(obstacle, size) for obstacle in obstacles for size in sizes]
    try:
        check_output_path(args.output)
        collection = collect_positions(configurations, args.games_per_config,
                                       args.move_time, args.seed)
        save_collection(collection, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    occurrences = sum(position["count"] for position in collection["positions"])
    print(f"Saved {len(collection['games'])} games, {occurrences} occurrences, "
          f"{len(collection['positions'])} unique positions to {args.output}")


if __name__ == "__main__":
    main()
