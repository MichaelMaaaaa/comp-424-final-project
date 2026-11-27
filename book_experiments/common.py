"""Inputs, identities and the unified selection rule shared by steps 5 and 6."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from agents.game_state import apply_move, apply_pass, initial_state, terminal
from book_experiments.books import DEFAULT_LAYOUT, book_entries, book_variant, make_book
from search_experiments.build import build_agent, build_opponent
from search_experiments.common import (REPO_ROOT, SCHEMA_VERSION, code_manifest, digest, file_hash,
                                       load_json, machine_info, normalize_variant, read_jsonl, restore_move,
                                       restore_state)
from search_experiments.development import load_development_rows, match_settings
from search_experiments.runner import TECHNICAL_REASONS, adjudication_fields, summarize_games


def code_identity():
    """A's code manifest plus B's experiment modules."""
    files = dict(code_manifest()["files"])
    for path in sorted((REPO_ROOT / "book_experiments").glob("*.py")):
        files[str(path.relative_to(REPO_ROOT))] = file_hash(path)
    return {"files": files, "code_hash": digest(files)}


def load_online(config, smoke):
    """Admit the exact no-book variant frozen by a successful Step 4."""
    tier, _, settings = match_settings(config, smoke)
    data = load_json(config["online_config"])
    expected_status = "smoke_only" if smoke else "selected"
    if (not isinstance(data, dict) or data.get("status") != expected_status or data.get("execution_tier") != tier
            or data.get("formal_selection") is not (not smoke) or not isinstance(data.get("variant"), dict)):
        raise ValueError("a successful same-tier Step 4 online configuration is required")
    if data.get("code_hash") != code_manifest()["code_hash"]:
        raise ValueError("Step 4 online configuration requires current code")
    manifest_hash = data.get("manifest_hash")
    if (not isinstance(manifest_hash, str) or not manifest_hash
            or (config.get("manifest") is not None and load_json(config["manifest"])["manifest_hash"] != manifest_hash)):
        raise ValueError("Step 4 online configuration requires the same frozen manifest")
    try:
        variant = normalize_variant(data["variant"])
    except (AttributeError, TypeError) as error:
        raise ValueError("invalid Step 4 online variant") from error
    if (digest(variant) != digest(data["variant"]) or variant["book"].get("mode") != "N"
            or variant["book_hash"] is not None):
        raise ValueError("Step 4 online configuration must contain an exact normalized N variant")
    if (variant["step_budget"] != settings.get("step_budget", 1.6)
            or (smoke and variant.get("max_depth") != settings["max_depth"])):
        raise ValueError("Step 4's frozen budget and depth must match the caller's execution settings")
    return variant


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def manifest_rows(config, split, smoke):
    if split not in ("D0", "D1"):
        raise ValueError("steps 5 and 6 use only the D0/D1 development configurations")
    limit = positive_integer(config.get("smoke", {}).get("max_configs"), "smoke max_configs") if smoke else None
    if limit is not None and limit >= 12:
        raise ValueError("smoke max_configs must select fewer than 12 configurations per split")
    _, d0, d1, _ = load_development_rows(config["manifest"], limit)
    return d0 if split == "D0" else d1


def opponent(config, smoke):
    return match_settings(config, smoke)[1]


def handoff_identity(config, online, tier, opp, manifest_hash):
    path = Path(config["online_config"]).resolve()
    return {"execution_tier": tier, "code_hash": code_manifest()["code_hash"],
            "book_code_hash": code_identity()["code_hash"], "manifest_hash": manifest_hash,
            "machine": machine_info(), "opponent": opp, "online_variant": online,
            "online_config_path": str(path), "online_config_hash": file_hash(path)}


def _check_identity(data, expected, label):
    if any(key not in data or digest(data[key]) != digest(value) for key, value in expected.items()):
        raise ValueError(f"{label} differs from the expected match evidence")


def _check_board_ending(state, winner, reason):
    if terminal(state) != (True, winner, reason):
        raise ValueError("recorded ending disagrees with the replayed final board")


def _check_adjudication(record, row, seat, budget):
    referee = record["referee"]
    _check_identity(referee, {"seed": row["setup_seed"], "obstacle": row["obstacle"],
                              "removed_size": row["removed_size"],
                              "rules": {"allow_moves": True, "win_length": 4}}, "referee")
    winner, reason = referee["winner"], referee["reason"]
    if type(winner) is not int or winner not in (0, 1, 2) or not reason:
        raise ValueError("missing referee adjudication")
    strikes = [referee["strikes"][str(player)] for player in (seat, 3 - seat)]
    if any(type(count) is not int or count < 0 for count in strikes) or (reason in TECHNICAL_REASONS and winner == 0):
        raise ValueError("invalid referee forfeit or strikes")
    maximum = record["candidate_max_step_s"]
    if maximum is not None and (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < 0):
        raise ValueError("invalid candidate step time")
    fields = adjudication_fields(winner, reason, seat, *strikes, maximum, budget)
    _check_identity(record, fields, "adjudication")
    moves, turns = referee["moves"], record["turns"]
    init_failure = reason in ("init_failed", "init_timeout")
    failed_turn = reason in TECHNICAL_REASONS and not init_failure
    if (referee["plies"] != len(moves) or len(turns) != len(moves) + int(failed_turn)
            or (init_failure and moves) or (not moves and reason not in TECHNICAL_REASONS)):
        raise ValueError("incomplete turn records in match evidence")
    state = initial_state(row["obstacle"], row["removed_size"])
    candidate_times = []
    for move, turn in zip(moves, turns):
        expected = {"ply": state.ply, "player": state.player}
        _check_identity(move, expected, "referee move")
        role = "candidate" if state.player == seat else "opponent"
        _check_identity(turn, dict(expected, move=move, actor_role=role), "turn")
        if restore_state(turn["state"]) != state or terminal(state)[0]:
            raise ValueError("turn state disagrees with the replayed trajectory")
        if move.get("pass") is True:
            state = apply_pass(state)
        else:
            state = apply_move(state, restore_move(move))
            if role == "candidate":
                candidate_times.append(move["agent_time"])
    if reason not in TECHNICAL_REASONS:
        _check_board_ending(state, winner, reason)
    rounded_maximum = None if maximum is None else round(maximum, 6)
    observed = max(candidate_times, default=None)
    if failed_turn:
        # SnapshotWorld saves the state where the actor forfeited, without adding a referee move.
        role = "candidate" if state.player == seat else "opponent"
        _check_identity(turns[-1], {"move": None, "failure_reason": reason, "ply": state.ply,
                                    "player": state.player, "actor_role": role}, "failed turn")
        if winner == state.player or restore_state(turns[-1]["state"]) != state or terminal(state)[0]:
            raise ValueError("failed turn disagrees with the referee adjudication")
    if reason == "too_many_strikes" and winner != seat:
        # The measured fourth strike has no played move, so only a lower bound survives in referee.moves.
        if rounded_maximum is None or (observed is not None and rounded_maximum < observed):
            raise ValueError("candidate timing omits the forfeited step")
    elif observed != rounded_maximum:
        raise ValueError("candidate timing differs from the referee record")


def validate_match_evidence(games_path, variant, rows, opponent, tier, *, require_eligible=True):
    """Verify exact runner evidence. Final reporting may retain ineligible or incomplete results."""
    games_path = Path(games_path).resolve()
    plan_path, summary_path = games_path.with_name("plan.json"), games_path.with_name("summary.json")
    try:
        return _validate_match_evidence(plan_path, games_path, summary_path, variant, rows, opponent, tier,
                                        require_eligible)
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"invalid match evidence at {games_path}: {error}") from error


def _validate_match_evidence(plan_path, games_path, summary_path, variant, rows, opponent, tier, require_eligible):
    if tier not in ("formal", "smoke") or not rows:
        raise ValueError("an execution tier and a nonempty frozen roster are required")
    if digest(normalize_variant(variant)) != digest(variant):
        raise ValueError("the expected match variant must be exactly normalized")
    budget = variant["step_budget"]
    if tier == "formal" and budget != 1.6:
        raise ValueError("formal match evidence requires a 1.6 second candidate budget")
    allowed_opponents = ("ab",) if require_eligible else ("ab", "random", "greedy")
    if opponent["kind"] not in allowed_opponents:
        raise ValueError("unexpected opponent for the requested match evidence")
    # The final schedule carries the same tier budgets for all three baselines.
    match_settings({"opponent": {"budget": opponent["budget"]},
                    "smoke": {"step_budget": budget, "opponent_budget": opponent["budget"]}}, smoke=tier == "smoke")
    manifest_hash = rows[0]["manifest_hash"]
    if not manifest_hash or any(row["manifest_hash"] != manifest_hash for row in rows):
        raise ValueError("match roster must belong to one frozen manifest")

    plan = load_json(plan_path)
    request = plan["request"]
    request_hash = digest(request)
    code, machine = code_manifest(), machine_info()
    _check_identity(plan, {"schema": SCHEMA_VERSION, "provenance": code, "request_hash": request_hash}, "plan")
    expected = {"variant": variant, "configurations": rows, "opponent": opponent, "manifest_hash": manifest_hash,
                "strict": True, "execution_tier": tier, "code_hash": code["code_hash"], "machine": machine}
    _check_identity(request, expected, "plan request")

    scheduled = {f"{variant['variant_id']}__{row['config_id']}__P{seat}": (row, seat)
                 for row in rows for seat in (1, 2)}
    if len(scheduled) != 2 * len(rows):
        raise ValueError("duplicate configuration in the expected roster")
    _check_identity(plan, {"games": [{"game_id": name, "config": row, "seat": seat}
                                     for name, (row, seat) in scheduled.items()]}, "game schedule")

    # Reconstruct the same artifacts as the runner without mutating its outputs.
    with TemporaryDirectory() as tmp:
        _, candidate = build_agent(variant, Path(tmp) / "candidate" / "student_agent.py")
        artifact = {"candidate_artifact_hash": candidate["artifact_hash"],
                    "candidate_source_hash": digest(candidate["source_files"])}
        _check_identity(request, artifact, "candidate artifact")
        opponent_info = {}
        for name, (row, seat) in scheduled.items():
            seed = int(row["opponent_rng_seed"]) ^ (seat * 0x9E3779B1)
            _, opponent_info[name] = build_opponent(opponent["kind"], Path(tmp) / "opponent" / "student_agent.py",
                                                    seed, opponent["budget"])

    # Admission is stricter than resume: a partial JSONL line is invalid evidence.
    records = [json.loads(line) for line in games_path.read_bytes().splitlines() if line.strip()]
    completed = {}
    variant_identity = {key: variant[key] for key in ("variant_id", "parent_variant", "config_hash", "book_hash")}
    context = dict(variant_identity, **artifact, schema=SCHEMA_VERSION, request_hash=request_hash,
                   manifest_hash=manifest_hash, code_hash=code["code_hash"], machine=machine,
                   strict=True, execution_tier=tier)
    for record in records:
        name = record["game_id"]
        if name not in scheduled:
            raise ValueError("unscheduled game in match evidence")
        row, seat = scheduled[name]
        expected_row = {key: row[key] for key in ("config_id", "split", "obstacle", "removed_size",
                                                  "setup_seed", "collection_seed")}
        baseline = opponent_info[name]
        _check_identity(record, dict(context, **expected_row, candidate_seat=seat, opponent=baseline,
                                     opponent_rng_seed=baseline["opponent_rng_seed"]), "game record")
        if not require_eligible:
            opponent_path = games_path.parent / "opponents" / name / "student_agent.py"
            if file_hash(opponent_path) != baseline["artifact_hash"]:
                raise ValueError("saved seeded opponent bytes differ from the scheduled build")
        if name in completed:
            raise ValueError("extra attempt after adjudication in match evidence")
        if record["status"] == "infrastructure_error":
            if (record.get("retryable") is not True or not record.get("error")
                    or any(key in record for key in ("referee", "score", "outcome"))):
                raise ValueError("invalid retryable infrastructure record")
            continue
        if record["status"] != "adjudicated":
            raise ValueError("invalid status in match evidence")
        _check_adjudication(record, row, seat, budget)
        completed[name] = record
    adjudicated = [completed[name] for name in scheduled if name in completed]
    summary = summarize_games(adjudicated, list(scheduled))
    summary.update(request_hash=request_hash, variant_id=variant["variant_id"], code_hash=code["code_hash"],
                   execution_tier=tier, formal_ranking_allowed=tier == "formal" and summary["eligible"])
    if digest(load_json(summary_path)) != digest(summary):
        raise ValueError("saved summary disagrees with the verified adjudications")
    if require_eligible and not summary["eligible"]:
        raise ValueError("complete eligible match evidence is required")
    descriptor = {"variant": variant, "request_hash": request_hash, "summary": summary}
    for name, path in (("plan", plan_path), ("games", games_path), ("summary", summary_path)):
        descriptor.update({f"{name}_path": str(path), f"{name}_hash": file_hash(path)})
    return adjudicated, descriptor


def reference_games(path, online, config, smoke=False):
    """Admit Step 4's exact D1 no-book trajectories for the caller's roster and tier."""
    tier, opp, _ = match_settings(config, smoke)
    if online != load_online(config, smoke):
        raise ValueError("D1 must use the exact Step 4 online variant")
    return validate_match_evidence(path, online, manifest_rows(config, "D1", smoke), opp, tier)


def validate_book_file(path):
    """Check the JSON entries, content version, count, provider and embedded bytes together."""
    try:
        book = load_json(path)
        layout = {key: book["layout"][key] for key in DEFAULT_LAYOUT}
        rebuilt = make_book(book_entries(book), book["label"], layout)
        if any(digest(book[key]) != digest(rebuilt[key]) for key in ("provider", "layout", "data")):
            raise ValueError("book entries, version or data hash disagree")
        return book
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"invalid exported book at {path}: {error}") from error


def selected_book_descriptor(path, online, source, K, k_star):
    """Return the actual exported book and the descriptor shared by Steps 5 and 6."""
    path = Path(path).resolve()
    if (source not in ("B0", "B1", "B2") or type(K) is not int or type(k_star) is not int
            or not 0 < k_star <= K):
        raise ValueError("invalid selected source or book size")
    book = validate_book_file(path)
    if book["layout"]["entries"] != k_star or not book["label"].endswith(f"_{source.lower()}_k{k_star}"):
        raise ValueError("selected source or book size differs from the export")
    descriptor = {key: book["layout"][key] for key in ("hash", "version", "entries")}
    descriptor.update(status="selected", K=K, k_star=k_star, source=source, book_file=str(path),
                      book_file_sha256=file_hash(path), label=book["label"], provider=book["provider"],
                      variant_D=normalize_variant(book_variant(online, book, "D")))
    return book, descriptor


def validate_empty_book(selection):
    """Only a consistent Step 5 empty decision permits the N-only handoff."""
    _check_identity(selection, {"status": "no_common_book", "K": 0, "selected_book": None,
                                "variant_D": None}, "Step 5 empty decision")
    selected_fields = ("source", "k_star", "book_file", "book_file_sha256", "hash", "version", "entries",
                       "label", "provider", "d_games", "second_source", "second_source_book_file", "second_source_book")
    if (any(selection.get(key) is not None for key in selected_fields)
            or selection.get("source_tie", False) is not False or selection.get("mode_if_no_book", "N") != "N"):
        raise ValueError("Step 5 no_common_book contains a contradictory book selection")


def validate_selected_exports(selection, online):
    """Read both Step 5 exports and compare their actual bytes to the frozen descriptors."""
    book, descriptor = selected_book_descriptor(
        selection["book_file"], online, selection["source"], selection["K"], selection["k_star"])
    _check_identity(selection, dict(descriptor, selected_book=descriptor), "Step 5 selected book")
    tie, other_file = selection.get("source_tie"), selection.get("second_source_book_file")
    if type(tie) is not bool or tie != bool(other_file):
        raise ValueError("Step 5 source tie and second-source book disagree")
    if not tie:
        _check_identity(selection, {"second_source_book": None}, "Step 5 untied source")
        return book, descriptor, None, None
    if selection["source"] == selection.get("second_source"):
        raise ValueError("Step 5 source tie requires a separate second-source book")
    other, other_descriptor = selected_book_descriptor(
        other_file, online, selection["second_source"], selection["K"], selection["k_star"])
    _check_identity(selection, {"second_source_book": other_descriptor}, "Step 5 second-source book")
    return book, descriptor, other, other_descriptor


def rank(summaries, order):
    """Unified rule: complete and eligible; score; then the earlier, simpler option."""
    valid = [name for name in order if summaries.get(name, {}).get("complete")
             and summaries[name].get("eligible")]
    return sorted(valid, key=lambda name: -summaries[name]["score"])


def games_of(summary, output):
    return [r for r in read_jsonl(Path(output) / "games.jsonl") if r.get("request_hash") == summary["request_hash"]]
