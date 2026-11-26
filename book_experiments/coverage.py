"""Coverage replay (5C.3/5C.4) and per-game book metrics (5C.3/6.2).

Replay queries a book at every recorded candidate turn of fixed no-book
trajectories: potential coverage, not the hit rate of real games, because a
book move can change the rest of the game. Turns with a single legal move are
not counted, since the online agent returns them without querying.
"""

from collections import Counter, defaultdict
import statistics

from agents.game_state import legal_moves
from book_experiments.books import lookup
from book_experiments.candidates import STAGES
from search_experiments.common import obstacle_class, restore_state, stage

CLASSES = ("corner", "edge", "interior")


def candidate_turns(records):
    """Yield (game, turn, state, legal moves) for every queryable candidate turn."""
    for game in records:
        if game.get("status") != "adjudicated":
            continue
        for turn in game["turns"]:
            if turn.get("actor_role") != "candidate" or turn.get("move") is None:
                continue
            state = restore_state(turn["state"])
            moves = legal_moves(state)
            if len(moves) > 1:
                yield game, turn, state, moves


def replay_coverage(table, records):
    queried, matched = Counter(), Counter()
    hits = []
    for game, turn, state, moves in candidate_turns(records):
        group = (stage(turn["ply"]), obstacle_class(game["obstacle"]))
        queried[group] += 1
        move = lookup(table, state)
        if move is not None and move in moves:
            matched[group] += 1
            hits.append(f"{game['game_id']}@{turn['ply']}")
    total_q, total_m = sum(queried.values()), sum(matched.values())

    def split(index, names):
        return {name: {"queryable": sum(v for k, v in queried.items() if k[index] == name),
                       "matched": sum(v for k, v in matched.items() if k[index] == name)} for name in names}

    return {"queryable": total_q, "matched": total_m, "rate": total_m / total_q if total_q else None,
            "by_stage": split(0, STAGES), "by_obstacle_class": split(1, CLASSES), "matched_turns": sorted(hits)}


def game_metrics(records):
    """Queries, matches and adoption from telemetry; full-step time, init and memory from the referee."""
    games = [g for g in records if g.get("status") == "adjudicated"]
    queries = matches = direct = hint = searched = depth_zero = depth_na = 0
    games_with_hit = 0
    depths, step_times, inits, memory = [], [], [], []
    by_stage = defaultdict(Counter)
    for game in games:
        seat = game["candidate_seat"]
        hit_here = False
        for turn in game["turns"]:
            if turn.get("actor_role") != "candidate" or turn.get("move") is None:
                continue
            info = turn.get("search") or {}
            phase = stage(turn["ply"])
            by_stage[phase]["turns"] += 1
            if info.get("book_queried"):
                queries += 1
                by_stage[phase]["queries"] += 1
            if info.get("book_matched"):
                matches += 1
                hit_here = True
                by_stage[phase]["matches"] += 1
            action = info.get("book_action")
            if action == "direct":
                direct += 1
                by_stage[phase]["direct"] += 1
            elif action == "hint":
                hint += 1
                by_stage[phase]["hint"] += 1
            if info.get("status") in ("book", "single_move"):
                depth_na += 1
            elif info:
                searched += 1
                depth = info.get("completed_depth") or 0
                depth_zero += depth == 0
                depths.append(depth)
        games_with_hit += hit_here
        referee = game.get("referee") or {}
        step_times += [m["agent_time"] for m in referee.get("moves", [])
                       if m.get("player") == seat and "agent_time" in m]
        timing = (referee.get("time") or {}).get(str(seat)) or {}
        if timing.get("init") is not None:
            inits.append(timing["init"])
        rss = (referee.get("max_rss_mb") or {}).get(str(seat))
        if rss is not None:
            memory.append(rss)
    return {"games": len(games), "queries": queries, "matches": matches,
            "hit_rate": matches / queries if queries else None,
            "games_with_hit": games_with_hit, "games_with_hit_fraction": games_with_hit / len(games) if games else None,
            "direct_returns": direct, "hint_only": hint,
            "by_stage": {name: dict(by_stage[name]) for name in STAGES},
            "searched_turns": searched, "depth_not_applicable": depth_na, "depth_zero": depth_zero,
            "mean_completed_depth": statistics.mean(depths) if depths else None,
            "step_time_mean": statistics.mean(step_times) if step_times else None,
            "step_time_max": max(step_times) if step_times else None,
            "init_time_max": max(inits) if inits else None,
            "max_rss_mb": max(memory) if memory else None}
