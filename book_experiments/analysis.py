"""5B.3 offline analysis (once per key, shared by B0/B1/B2) and 5C.1 screening.

The analyser runs the same frozen search and evaluation as the online
configuration, never the book, with a per-candidate budget that includes
decoding the state. Records are reused only when search, evaluation, rules,
budget, code and key are identical.
"""

from pathlib import Path
import time

from constants import MAX_PLIES
from agents.game_state import apply_move, legal_moves, terminal
from agents.symmetry import canonicalize, decode_book_state
from book_experiments.books import move_record
from book_experiments.collect import online_engine
from search_experiments.common import (REPO_ROOT, append_jsonl, digest, file_hash, read_jsonl,
                                        repair_jsonl_tail, resolve_evaluator, restore_move)

ANALYZER_VERSION = "b-analysis-v1"
SCREEN_VERSION = "b-screen-v1"
RULES = {"allow_moves": True, "win_length": 4, "max_plies": MAX_PLIES}
ANALYSIS_FILES = ("agents/search_engine.py", "agents/evaluation.py", "agents/game_state.py",
                  "agents/symmetry.py", "agents/transposition_table.py", "constants.py", "helpers.py",
                  "agents/alpha_beta.py", "agents/move_ordering.py", "agents/iterative_deepening.py",
                  "search_experiments/common.py", "book_experiments/books.py",
                  "book_experiments/collect.py", "book_experiments/analysis.py")


def analysis_identity(online, budget, max_depth=None):
    identity = {"analyzer": ANALYZER_VERSION, "search": online["search"], "evaluation": online["evaluation"],
                "rules": RULES, "budget": budget, "max_depth": max_depth,
                "code": {name: file_hash(REPO_ROOT / name) for name in ANALYSIS_FILES}}
    return dict(identity, analysis_id=digest(identity))


def analyze_key(key, engine, budget, max_depth=None):
    """Search one canonical candidate; the budget starts before decoding."""
    started = time.perf_counter()
    key = tuple(key)
    state = decode_book_state(key)
    if canonicalize(state)[0] != key:
        raise ValueError("candidate key is not canonical")
    result = engine.search(state, started + budget, max_depth)
    elapsed = time.perf_counter() - started
    if result.status == "terminal":
        termination = "terminal"
    elif result.solved_remaining:
        termination = "solved_remaining"
    elif max_depth is not None and result.completed_depth >= max_depth:
        termination = "max_depth"
    else:
        termination = "deadline"
    return {"key": list(key), "ply": key[-1], "move": None if result.move is None else move_record(result.move),
            "value": result.value, "completed_depth": result.completed_depth, "status": result.status,
            "termination": termination, "elapsed": elapsed,
            "iterations": [{"depth": row["depth"], "move": move_record(row["move"]), "value": row["value"],
                            "elapsed": row["elapsed"]} for row in result.iterations]}


def analyze_candidates(keys, online, identity, output, progress=True):
    """Analyse each distinct key once; a rerun resumes from records with the same analysis_id."""
    log = Path(output) / "records.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl_tail(log)
    records = {tuple(r["key"]): r for r in read_jsonl(log) if r.get("analysis_id") == identity["analysis_id"]}
    evaluator = resolve_evaluator(online["evaluation"])
    pending = [tuple(k) for k in sorted({tuple(k) for k in keys}) if tuple(k) not in records]
    for index, key in enumerate(pending, start=1):
        record = analyze_key(key, online_engine(online, evaluator), identity["budget"], identity["max_depth"])
        record["analysis_id"] = identity["analysis_id"]
        append_jsonl(log, record)
        records[key] = record
        if progress and (index % 25 == 0 or index == len(pending)):
            print(f"analysed {index}/{len(pending)} new candidates (ply {key[-1]}, "
                  f"depth {record['completed_depth']})", flush=True)
    return records


def _decision(accepted, reason, reply=None):
    return {"accepted": accepted, "reason": reason, "winning_reply": None if reply is None else move_record(reply)}


def screen_record(record, analysis_id, min_depth=3, stable_depths=2):
    """Admission check; it reads the recorded analysis and never searches again."""
    if record.get("analysis_id") != analysis_id:
        return _decision(False, "version_mismatch")
    try:
        key = tuple(record["key"])
        state = decode_book_state(key)
        if canonicalize(state)[0] != key:
            raise ValueError("not canonical")
    except (KeyError, TypeError, ValueError):
        return _decision(False, "invalid_state")
    if terminal(state)[0]:
        return _decision(False, "terminal_state")
    moves = legal_moves(state)
    try:
        move = restore_move(record["move"])
    except (KeyError, TypeError, ValueError):
        move = None
    if move is None or move not in moves:
        return _decision(False, "illegal_move")
    rows = record.get("iterations")
    depth = record.get("completed_depth")
    if type(depth) is not int or depth < 0 or not isinstance(rows, list):
        return _decision(False, "invalid_analysis")
    if record.get("status") != ("completed" if depth else "fallback") or len(rows) != depth:
        return _decision(False, "invalid_analysis")
    if depth == 0:
        return _decision(False, "no_completed_depth")
    try:
        depths = [row["depth"] for row in rows]
        recommendations = [restore_move(row["move"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return _decision(False, "invalid_analysis")
    if (any(type(d) is not int or d != i for i, d in enumerate(depths, 1))
            or any(recommendation not in moves for recommendation in recommendations) or recommendations[-1] != move):
        return _decision(False, "invalid_analysis")
    child = apply_move(state, move)
    finished, winner, _ = terminal(child)
    # After a move, the next player's line wins first, so uncovering can lose at once.
    if finished and winner == state.player:
        return _decision(True, "immediate_win")
    if finished and winner == 3 - state.player:
        return _decision(False, "immediate_loss")
    if depth < min_depth:
        return _decision(False, "insufficient_depth")
    if len(set(recommendations[-stable_depths:])) != 1:
        return _decision(False, "unstable_move")
    if not finished:
        for reply in legal_moves(child):
            reply_finished, reply_winner, _ = terminal(apply_move(child, reply))
            if reply_finished and reply_winner == child.player:
                return _decision(False, "opponent_immediate_win", reply)
    return _decision(True, "stable_and_safe")
