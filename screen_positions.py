"""Screen analyzed recommendations using the design's book admission rules."""

import argparse
import copy
import json
from pathlib import Path

from constants import MAX_PLIES
from helpers import Move
from agents.game_state import apply_move, legal_moves, terminal
import analyze_positions as analysis
from collect_positions import check_output_path, save_collection


SCREENING_VERSION = "screen-v1"
MIN_COMPLETED_DEPTH = 3


def _read_move(data):
    if not isinstance(data, dict):
        raise ValueError("a move must contain src, dest and size")
    return Move(data["src"], data["dest"], data["size"])


def _versions_match(versions):
    return analysis._metadata_equal(versions, analysis.current_versions())


def _completed_moves(record, state, move, moves):
    """Validate the search history and return its normalized root moves."""
    depth = record.get("completed_depth")
    rows = record.get("iterations")
    if type(depth) is not int or not 0 <= depth <= MAX_PLIES - state.ply:
        raise ValueError("invalid completed depth")
    if not isinstance(rows, list):
        raise ValueError("iterations must be a list")
    if record.get("status") == "fallback":
        if depth != 0 or record["value"] is not None or rows:
            raise ValueError("inconsistent fallback record")
        return []
    if record.get("status") != "completed" or depth < 1 or len(rows) != depth:
        raise ValueError("inconsistent completed record")

    completed = []
    for expected_depth, row in enumerate(rows, start=1):
        if (not isinstance(row, dict) or type(row.get("depth")) is not int
                or row["depth"] != expected_depth or type(row.get("value")) is not int):
            raise ValueError("invalid completed iteration")
        action = _read_move(row.get("move"))
        if action not in moves:
            raise ValueError("iteration contains an illegal root move")
        completed.append(action)
    if (move != completed[-1] or type(record.get("value")) is not int
            or record["value"] != rows[-1]["value"]):
        raise ValueError("final summary does not match the last iteration")
    return completed


def _decision(accepted, reason, reply=None):
    reply_record = None
    if reply is not None:
        reply_record = {"src": reply.src, "dest": reply.dest, "size": reply.size}
    return {"accepted": accepted, "reason": reason, "winning_reply": reply_record}


def screen_position(record):
    """Check the recorded canonical move without replacing it or searching again."""
    if not isinstance(record, dict):
        raise ValueError("analysis record must be a dictionary")
    state = analysis._canonical_state(record.get("key"))
    if not _versions_match(record.get("versions")):
        return _decision(False, "version_mismatch")
    if terminal(state)[0]:
        return _decision(False, "terminal_state")

    moves = legal_moves(state)
    try:
        move = _read_move(record.get("move"))
    except (KeyError, TypeError, ValueError):
        return _decision(False, "illegal_move")
    if move not in moves:
        return _decision(False, "illegal_move")
    try:
        completed = _completed_moves(record, state, move, moves)
    except (KeyError, TypeError, ValueError):
        return _decision(False, "invalid_analysis")

    child = apply_move(state, move)
    finished, winner, _ = terminal(child)
    # The next player's line has priority, even when both players have lines.
    if finished and winner == state.player:
        return _decision(True, "immediate_win")
    if finished and winner == 3 - state.player:
        return _decision(False, "immediate_loss")
    if record["completed_depth"] < MIN_COMPLETED_DEPTH:
        return _decision(False, "insufficient_depth")
    if completed[-1] != completed[-2]:
        return _decision(False, "unstable_move")

    if not finished:
        for reply in legal_moves(child):
            reply_finished, reply_winner, _ = terminal(apply_move(child, reply))
            if reply_finished and reply_winner == child.player:
                return _decision(False, "opponent_immediate_win", reply)
    return _decision(True, "stable_and_safe")


def screen_positions(analyzed):
    """Preserve every analysis record and attach its admission decision."""
    if not isinstance(analyzed, dict):
        raise ValueError("input must be an analysis file")
    if type(analyzed.get("format_version")) is not int or analyzed["format_version"] != 1:
        raise ValueError("unsupported analysis format version")
    positions = analyzed.get("positions")
    settings = analyzed.get("settings")
    source = analyzed.get("source_collection")
    if (not isinstance(positions, list) or not isinstance(settings, dict)
            or not isinstance(source, dict)):
        raise ValueError("analysis file is missing its records or provenance")
    total = settings.get("total_candidates")
    count = settings.get("analyzed_candidates")
    if (type(total) is not int or total < len(positions)
            or type(count) is not int or count != len(positions)):
        raise ValueError("analysis counts do not match the recorded subset")

    collection = dict(source)
    collection["positions"] = positions
    analysis._validate_collection(collection)
    result = copy.deepcopy(analyzed)
    accepted = 0
    reasons = {}
    for record in result["positions"]:
        decision = screen_position(record)
        record["screening"] = decision
        accepted += int(decision["accepted"])
        reason = decision["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
    result["screening"] = {
        "version": SCREENING_VERSION,
        "versions": analysis.current_versions(),
        "min_completed_depth": MIN_COMPLETED_DEPTH,
        "screened": len(positions),
        "accepted": accepted,
        "rejected": len(positions) - accepted,
        "reasons": reasons,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="analyzed_positions.json")
    parser.add_argument("--output", default="screened_positions.json")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("input and output must be different files")
    try:
        check_output_path(args.output)
        with open(args.input, encoding="utf-8") as source:
            analyzed = json.load(source)
        result = screen_positions(analyzed)
        save_collection(result, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = result["screening"]
    print(f"Screened {summary['screened']} positions: {summary['accepted']} accepted, "
          f"{summary['rejected']} rejected. Saved to {args.output}")


if __name__ == "__main__":
    main()
