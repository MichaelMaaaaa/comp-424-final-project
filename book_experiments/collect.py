"""5B.1 collection games on D0: self-play and games against greedy and the local AB.

Our side uses the online search and evaluation read from F's configuration,
no book, and a fixed per-move budget. At the last completed depth it chooses
uniformly among the exactly tied best root moves (full window per root move),
seeded per game; an unfinished depth never supplies a tie set.
"""

import importlib
from pathlib import Path
import random
import time

import numpy as np

from agents.game_state import apply_move, initial_state, is_legal_move, terminal, to_arrays
from agents.search_engine import SearchConfig, SearchEngine
from book_experiments.books import move_record
from search_experiments.common import (append_jsonl, atomic_json, digest, load_json, machine_info,
                                        read_jsonl, repair_jsonl_tail, resolve_evaluator, state_record)


def online_engine(online, evaluator=None):
    return SearchEngine(SearchConfig(**online["search"]),
                        evaluator=evaluator or resolve_evaluator(online["evaluation"]))


def game_seed(row, kind, index):
    return random.Random(f"{row['collection_seed']}/{kind}/{index}").randrange(2**32)


def collection_plan(d0_rows, settings):
    """B0: self-play per config. B1: fewer self-play games plus one greedy and one AB game.

    Greedy and AB games give our side opposite seats, alternating by manifest
    order, so both seats are used equally often over the D0 list.
    """
    b0, b1 = settings["b0_selfplay_per_config"], settings["b1_selfplay_per_config"]
    reuse = settings.get("b1_reuses_b0_selfplay", True)
    if reuse and b1 > b0:
        raise ValueError("B1 cannot reuse more self-play games than B0 has")
    games = []
    for index, row in enumerate(d0_rows):
        if row["split"] != "D0":
            raise ValueError("book candidates are collected on D0 only")
        for number in range(b0):
            pools = ["B0", "B1"] if reuse and number < b1 else ["B0"]
            games.append(dict(_game(row, "self", number), pools=pools))
        if not reuse:
            for number in range(b1):
                games.append(dict(_game(row, "self", b0 + number), pools=["B1"]))
        greedy_seat = 1 if index % 2 == 0 else 2
        games.append(dict(_game(row, "greedy", 0, greedy_seat), pools=["B1"]))
        games.append(dict(_game(row, "ab", 0, 3 - greedy_seat), pools=["B1"]))
    return games


def _game(row, kind, number, our_seat=None):
    return {"game_id": f"{kind}__{row['config_id']}__{number}", "config_id": row["config_id"],
            "split": row["split"], "obstacle": list(row["obstacle"]), "removed_size": row["removed_size"],
            "kind": kind, "our_seat": our_seat, "collection_seed": game_seed(row, kind, number)}


def make_opponent(kind, seed, budget):
    if kind == "greedy":
        from agents.greedy_agent import GreedyAgent
        agent = GreedyAgent()
        agent.rng = np.random.default_rng(seed)
        return agent
    if kind == "ab":
        module = importlib.import_module("baselines.ab_baseline_agent")
        module.TIME_LIMIT = budget
        return module.AlphaBetaBaseline()
    raise ValueError(f"unknown collection opponent {kind!r}")


def play_collection_game(game, online, settings, evaluator=None):
    """Play one game in process and record every pre-move state of both players."""
    rng = random.Random(game["collection_seed"])
    state = initial_state(tuple(game["obstacle"]), game["removed_size"])
    engine = online_engine(online, evaluator)
    opponent = None
    if game["kind"] != "self":
        opponent = make_opponent(game["kind"], game["collection_seed"], settings["opponent_budget"])
    turns, started = [], time.perf_counter()
    seconds = {"collector": 0.0, "opponent": 0.0}
    while True:
        finished, winner, reason = terminal(state)
        if finished:
            break
        actor = state.player
        before = state_record(state)
        ours = game["kind"] == "self" or actor == game["our_seat"]
        clock = time.perf_counter()
        if ours:
            result = engine.search(state, clock + settings["move_time"], settings.get("max_depth"),
                                   collect_ties=True, tie_rng=rng)
            move = result.move
            info = {"role": "collector", "completed_depth": result.completed_depth, "value": result.value,
                    "ties": len(result.optimal_moves), "status": result.status}
        else:
            board, reserves = to_arrays(state)
            move = opponent.step(board, reserves, actor, 3 - actor)
            info = {"role": game["kind"]}
        elapsed = time.perf_counter() - clock
        seconds["collector" if ours else "opponent"] += elapsed
        if move is None or not is_legal_move(state, move):
            return dict(game, status="failed", failure=f"illegal move by {info['role']} at ply {state.ply}",
                        turns=turns, elapsed=time.perf_counter() - started, seconds=seconds)
        turns.append(dict(info, state=before, ply=state.ply, player=actor, move=move_record(move),
                          elapsed=elapsed))
        state = apply_move(state, move)
    return dict(game, status="completed", winner=winner, reason=reason, plies=state.ply, turns=turns,
                elapsed=time.perf_counter() - started, seconds=seconds)


def run_collection(d0_rows, online, settings, output, identity, progress=True):
    """Play the plan once; a rerun with the same request skips completed games."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = collection_plan(d0_rows, settings)
    request = {"online_config_hash": online["config_hash"], "settings": settings, "plan": plan,
               "code": identity, "machine": machine_info()}
    request_hash = digest(request)
    plan_file = output / "plan.json"
    if plan_file.exists() and load_json(plan_file)["request_hash"] != request_hash:
        raise ValueError("collection request changed (code, config, plan or machine); use a new output directory")
    atomic_json(plan_file, {"request_hash": request_hash, "request": request})
    log = output / "games.jsonl"
    repair_jsonl_tail(log)
    records = [r for r in read_jsonl(log) if r.get("request_hash") == request_hash]
    done = {r["game_id"] for r in records if r["status"] == "completed"}
    evaluator = resolve_evaluator(online["evaluation"])
    for game in plan:
        if game["game_id"] in done:
            continue
        record = dict(play_collection_game(game, online, settings, evaluator), request_hash=request_hash)
        append_jsonl(log, record)
        records.append(record)
        if progress:
            print(f"collected {game['game_id']}: {record['status']}, {len(record['turns'])} turns, "
                  f"{record['elapsed']:.1f} s", flush=True)
    latest = {}
    for record in records:
        latest[record["game_id"]] = record
    games = [latest[g["game_id"]] for g in plan if g["game_id"] in latest]
    complete = all(g["status"] == "completed" for g in games) and len(games) == len(plan)
    cost = {"games": len(games), "complete": complete,
            "seconds_total": sum(g["elapsed"] for g in games),
            "seconds_collector": sum(g["seconds"]["collector"] for g in games),
            "seconds_opponent": sum(g["seconds"]["opponent"] for g in games),
            "by_pool": {pool: sum(g["elapsed"] for g in games if pool in g["pools"]) for pool in ("B0", "B1")}}
    atomic_json(output / "collection_summary.json", dict(cost, request_hash=request_hash))
    return games, cost
