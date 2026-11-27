"""One paired-match implementation shared by A, F and B."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import traceback

from agents.game_state import GameState
from helpers import new_game
from runners import InProcessRunner, SubprocessRunner
from world import World
from search_experiments.build import build_agent, build_opponent, clear_generated_bytecode
from search_experiments.common import (SCHEMA_VERSION, append_jsonl, atomic_json, code_manifest,
                                        digest, file_hash, machine_info, normalize_variant, obstacle_class,
                                        read_jsonl, repair_jsonl_tail, stage, state_record)

TECHNICAL_REASONS = {"init_failed", "init_timeout", "crash", "hard_timeout", "too_many_strikes"}


def _config_id(obstacle, removed):
    return f"r{obstacle[0]}c{obstacle[1]}_{'SML'[removed]}"


def _validator_reservations():
    reserved = []
    for seed, split in ((0, "D0"), (1, "D1")):
        _, _, cell, removed = new_game(seed)
        reserved.append({"setup_seed": seed, "config_id": _config_id(cell, removed), "split": split})
    return reserved


def make_manifest(seed=20261125):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in range(4):
        for col in range(4):
            groups[obstacle_class((row, col))].append((row, col))
    # Lookup seeds, then explicitly verify their configurations; no autoplay seed arithmetic.
    matches = {}
    setup_seed = 0
    while len(matches) < 48:
        _, _, obstacle, removed = new_game(setup_seed)
        matches.setdefault((tuple(obstacle), int(removed)), setup_seed)
        setup_seed += 1
    reservations = _validator_reservations()
    reserved_splits = {r["config_id"]: r["split"] for r in reservations}
    rows = []
    for removed in range(3):
        assigned = {}
        for kind, (d0, d1) in {"corner": (1, 1), "edge": (2, 2), "interior": (1, 1)}.items():
            cells = []
            for cell in groups[kind]:
                split = reserved_splits.get(_config_id(cell, removed))
                if split is None:
                    cells.append(cell)
                else:
                    assigned[cell] = split
                    d0 -= split == "D0"
                    d1 -= split == "D1"
            rng.shuffle(cells)
            for index, cell in enumerate(cells):
                assigned[cell] = "D0" if index < d0 else "D1" if index < d0 + d1 else "T"
        for obstacle in sorted(assigned):
            config_id = _config_id(obstacle, removed)
            rows.append({"config_id": config_id, "obstacle": obstacle,
                         "obstacle_row": obstacle[0], "obstacle_col": obstacle[1],
                         "obstacle_class": obstacle_class(obstacle), "removed_size": removed,
                         "removed_name": "SML"[removed], "split": assigned[obstacle],
                         "setup_seed": matches[(obstacle, removed)],
                         "opponent_rng_seed": rng.randrange(2**32),
                         "collection_seed": rng.randrange(2**32)})
    manifest = {"schema": SCHEMA_VERSION, "manifest_seed": seed, "rows": rows,
                "reservation_policy": "default_validator_seeds_before_shuffle_v1",
                "validator_reservations": reservations}
    return dict(manifest, manifest_hash=digest(manifest))


def validate_manifest(manifest):
    rows = manifest["rows"]
    if len(rows) != 48 or len({(tuple(r["obstacle"]), r["removed_size"]) for r in rows}) != 48:
        raise ValueError("manifest must contain each of the 48 configurations exactly once")
    reservations = _validator_reservations()
    if (manifest.get("reservation_policy") != "default_validator_seeds_before_shuffle_v1"
            or manifest.get("validator_reservations") != reservations):
        raise ValueError("generate a manifest with the official validator reservations")
    if digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("manifest hash mismatch")
    for reservation in reservations:
        if not any(all(row[key] == value for key, value in reservation.items()) for row in rows):
            raise ValueError("validator setup seed must occupy its reserved development split")
    for split, multiplier in [("D0", 1), ("D1", 1), ("T", 2)]:
        for removed in range(3):
            counts = Counter(r["obstacle_class"] for r in rows if r["split"] == split and r["removed_size"] == removed)
            if counts != Counter(corner=multiplier, edge=2 * multiplier, interior=multiplier):
                raise ValueError("manifest split stratification is invalid")
    for row in rows:
        _, _, actual, removed = new_game(row["setup_seed"])
        if (tuple(actual) != tuple(row["obstacle"]) or removed != row["removed_size"]
                or tuple(actual) != (row["obstacle_row"], row["obstacle_col"])
                or row["obstacle_class"] != obstacle_class(actual)
                or row["config_id"] != _config_id(actual, removed) or row["removed_name"] != "SML"[removed]):
            raise ValueError(f"setup seed and configuration fields disagree for {row['config_id']}")
    # Attach the verified full roster identity to copies before callers select a subset.
    return [dict(row, manifest_hash=manifest["manifest_hash"]) for row in rows]


class SnapshotWorld(World):
    """Extend recording around the official referee; never replace adjudication."""
    def __init__(self, *args, **kwargs):
        self.turns = []
        self.step_times = {1: [], 2: []}
        super().__init__(*args, **kwargs)

    def adjudicate(self, player, result):
        if result.kind in ("ok", "error"):
            self.step_times[player].append(result.agent_time)
        return super().adjudicate(player, result)

    def step(self):
        if self.over:
            return
        old_ply, actor, old_count = self.ply, self.player, len(self.record.moves)
        before = GameState.from_observation(self.board, self.reserves, actor, old_ply)
        super().step()
        if len(self.record.moves) > old_count:
            self.turns.append({"state": state_record(before), "ply": old_ply, "player": actor,
                               "move": dict(self.record.moves[-1])})
        elif self.over and self.record.reason in TECHNICAL_REASONS:
            self.turns.append({"state": state_record(before), "ply": old_ply, "player": actor,
                               "move": None, "failure_reason": self.record.reason})


def _telemetry(log_path):
    records = {}
    if log_path is None or not Path(log_path).exists():
        return records
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if line.startswith("EXP_STEP "):
            try:
                entry = json.loads(line[len("EXP_STEP "):])
                records[entry["ply"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def adjudication_fields(winner, reason, seat, candidate_strikes, opponent_strikes, max_step, budget):
    """Project referee adjudication and measured candidate time into the paired result."""
    outcome = "D" if winner == 0 else "W" if winner == seat else "L"
    return {"outcome": outcome, "score": {"W": 1.0, "D": 0.5, "L": 0.0}[outcome],
            "technical_failure": (reason in TECHNICAL_REASONS and winner != seat) or candidate_strikes > 0,
            "opponent_technical_failure": (reason in TECHNICAL_REASONS and winner == seat) or opponent_strikes > 0,
            "candidate_step_budget": budget, "candidate_max_step_s": max_step,
            "candidate_budget_failure": max_step is not None and max_step > budget}


def _adjudicated_row(base, world, game, seat, budget):
    for turn in world.turns:
        turn["actor_role"] = "candidate" if turn["player"] == seat else "opponent"
        turn["near_cap"] = turn["ply"] >= 90
        if turn["player"] == seat:
            turn["search"] = None
    fields = adjudication_fields(game.winner, game.reason, seat, game.strikes[seat], game.strikes[3-seat],
                                 max(world.step_times[seat], default=None), budget)
    return dict(base, **fields, status="adjudicated", referee=game.to_dict(), turns=world.turns)


def _stop_after_failure(runners):
    interruption = None
    for participant in runners:
        try:
            participant.stop()
        except Exception:
            continue
        except BaseException as error:
            if interruption is None:
                interruption = error
    if interruption is not None:
        raise interruption


def summarize_games(records, planned_ids):
    completed = {r["game_id"]: r for r in records if r.get("status") == "adjudicated"}
    completed = {key: completed[key] for key in planned_ids if key in completed}
    counts = Counter(r["outcome"] for r in completed.values())
    pairs = defaultdict(dict)
    for record in completed.values():
        pairs[record["config_id"]][record["candidate_seat"]] = record["score"]
    pair_scores = {key: (values[1] + values[2]) / 2 for key, values in pairs.items() if set(values) == {1, 2}}
    complete = bool(planned_ids) and len(completed) == len(planned_ids)
    technical = sum(bool(r["technical_failure"]) for r in completed.values())
    opponent_technical = sum(bool(r.get("opponent_technical_failure")) for r in completed.values())
    budget_checks = [r.get("candidate_budget_failure") for r in completed.values()]
    return {"complete": complete, "planned": len(planned_ids), "adjudicated": len(completed),
            "W": counts["W"], "D": counts["D"], "L": counts["L"],
            "score": sum(r["score"] for r in completed.values()) / len(completed) if complete and completed else None,
            "technical_failures": technical,
            "opponent_technical_failures": opponent_technical,
            "candidate_budget_failures": budget_checks.count(True),
            "candidate_budget_unchecked": budget_checks.count(None),
            "eligible": complete and technical == 0 and opponent_technical == 0
                        and all(failed is False for failed in budget_checks),
            "paired_scores": pair_scores,
            "missing_game_ids": [key for key in planned_ids if key not in completed]}


def paired_difference(new_summary, old_summary, seed=20261125):
    new, old = new_summary["paired_scores"], old_summary["paired_scores"]
    if not new_summary["complete"] or not old_summary["complete"] or set(new) != set(old):
        return {"complete": False, "mean": None}
    differences = [new[key] - old[key] for key in sorted(new)]
    if not differences:
        return {"complete": False, "mean": None}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
    return {"complete": True, "mean": sum(differences) / len(differences),
            "improved": sum(v > 0 for v in differences), "unchanged": sum(v == 0 for v in differences),
            "worse": sum(v < 0 for v in differences), "bootstrap_config_ci95": [samples[49], samples[1949]],
            "sampling_unit": "configuration (both seats together)"}


def run_pairs(variant, configurations, output, *, opponent=None, strict=True, execution_tier="formal",
              allow_test=False, reuse_paths=(), expected_candidate_artifact_hash=None):
    """Resume exact requests only. Infrastructure failures remain pending for rerun."""
    if execution_tier not in ("formal", "smoke"):
        raise ValueError("execution_tier must be formal or smoke")
    if execution_tier == "formal" and not strict:
        raise ValueError("formal games require strict subprocess runners")
    if any(row["split"] == "T" for row in configurations) and not allow_test:
        raise ValueError("T is held out until F's final experiment")
    if not configurations:
        raise ValueError("configuration list is empty")
    if len({row["config_id"] for row in configurations}) != len(configurations):
        raise ValueError("duplicate config_id in scheduled configurations. Pass each configuration once.")
    manifest_hash = configurations[0].get("manifest_hash")
    if execution_tier == "formal" and (not isinstance(manifest_hash, str) or not manifest_hash.strip()):
        raise ValueError("formal games require rows from a validated frozen manifest")
    if any(row.get("manifest_hash") != manifest_hash for row in configurations):
        raise ValueError("all scheduled configurations must share one frozen manifest")
    for row in configurations:
        _, _, obstacle, removed = new_game(row["setup_seed"])
        if obstacle != tuple(row["obstacle"]) or removed != row["removed_size"]:
            raise ValueError(f"initial configuration differs from the manifest for {row['config_id']}. "
                             "Check setup_seed, obstacle and removed_size.")
    variant = normalize_variant(variant)
    opponent = dict(opponent or {"kind": "ab", "budget": 1.5})
    if execution_tier == "formal":
        if variant["step_budget"] != 1.6:
            raise ValueError("formal pairs require a candidate budget of 1.6 seconds")
        if opponent.get("kind") == "ab" and opponent.get("budget", 1.5) != 1.5:
            raise ValueError("formal pairs require an AB opponent budget of 1.5 seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = code_manifest()
    machine = machine_info()
    reuse_paths = [Path(path).resolve() for path in reuse_paths if Path(path).exists()]
    reusable = [record for path in reuse_paths for record in read_jsonl(path)
                if record.get("status") == "adjudicated"]
    request = {"variant": variant, "configurations": configurations, "opponent": opponent,
               "manifest_hash": manifest_hash,
               "strict": strict, "execution_tier": execution_tier,
               "code_hash": provenance["code_hash"], "machine": machine,
               "reuse_sources": {str(path): file_hash(path) for path in reuse_paths},
               "reuse_source_order": [str(path) for path in reuse_paths]}
    plan_file = output / "plan.json"
    agent_path = output / "candidate" / "student_agent.py"
    log_file = output / "games.jsonl"
    planned = [{"game_id": f"{variant['variant_id']}__{row['config_id']}__P{seat}",
                "config": row, "seat": seat} for row in configurations for seat in (1, 2)]
    # Check the generated artifact before replacing any saved candidate or plan.
    with TemporaryDirectory(dir=output) as staging:
        staged_agent, agent_meta = build_agent(variant, Path(staging) / agent_path.name)
        if (expected_candidate_artifact_hash is not None
                and agent_meta["artifact_hash"] != expected_candidate_artifact_hash):
            raise ValueError("built candidate artifact differs from the validated artifact")
        request["candidate_artifact_hash"] = agent_meta["artifact_hash"]
        request["candidate_source_hash"] = digest(agent_meta["source_files"])
        request_hash = digest(request)
        plan = {"schema": SCHEMA_VERSION, "request_hash": request_hash, "request": request,
                "provenance": provenance, "games": planned}
        if expected_candidate_artifact_hash is not None:
            if log_file.exists() and log_file.stat().st_size and not plan_file.exists():
                raise ValueError("saved game results require their frozen plan")
            if agent_path.exists() and file_hash(agent_path) != expected_candidate_artifact_hash:
                raise ValueError("saved candidate bytes differ from the validated artifact")
        if plan_file.exists():
            prior = json.loads(plan_file.read_text())
            if expected_candidate_artifact_hash is not None and digest(prior) != digest(plan):
                raise ValueError("saved match plan differs from the validated request or schedule")
            if prior["request_hash"] != request_hash:
                raise ValueError("resume configuration/code/machine/candidate artifact or source changed; use a new output directory")
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        staged_agent.replace(agent_path)
        clear_generated_bytecode(agent_path)
        staged_agent.with_suffix(".meta.json").replace(agent_path.with_suffix(".meta.json"))
    atomic_json(plan_file, plan)
    previous = [r for r in read_jsonl(log_file) if r.get("request_hash") == request_hash
                and r.get("manifest_hash") == manifest_hash
                and r.get("candidate_source_hash") == request["candidate_source_hash"]]
    repair_jsonl_tail(log_file)
    done = {r["game_id"] for r in previous if r.get("status") == "adjudicated"}
    for item in planned:
        game_id, row, seat = item["game_id"], item["config"], item["seat"]
        if game_id in done:
            continue
        seed = int(row["opponent_rng_seed"]) ^ (seat * 0x9E3779B1)
        opponent_path, opponent_meta = build_opponent(opponent["kind"],
            output / "opponents" / game_id / "student_agent.py", seed, opponent.get("budget", 1.5))
        runner_type = SubprocessRunner if strict else InProcessRunner
        runners = []
        for actual_seat in (1, 2):
            path = agent_path if actual_seat == seat else opponent_path
            arguments = {"log_dir": str(output / "logs" / game_id)} if strict else {}
            runners.append(runner_type(str(path.resolve()), **arguments))
        base = {"schema": SCHEMA_VERSION, "game_id": game_id, "request_hash": request_hash,
                "manifest_hash": manifest_hash,
                "variant_id": variant["variant_id"], "parent_variant": variant["parent_variant"],
                "code_hash": provenance["code_hash"], "config_hash": variant["config_hash"],
                "book_hash": variant["book_hash"], "config_id": row["config_id"], "split": row["split"],
                "obstacle": row["obstacle"], "removed_size": row["removed_size"], "candidate_seat": seat,
                "setup_seed": row["setup_seed"], "opponent_rng_seed": seed,
                "collection_seed": row["collection_seed"], "opponent": opponent_meta,
                "candidate_artifact_hash": agent_meta["artifact_hash"],
                "candidate_source_hash": request["candidate_source_hash"], "execution_tier": execution_tier,
                "strict": strict, "machine": machine,
                "attempt": 1 + sum(r.get("game_id") == game_id for r in previous)}
        equivalence = ("game_id", "config_hash", "code_hash", "book_hash", "config_id", "split",
                       "obstacle", "removed_size", "candidate_seat", "setup_seed", "opponent_rng_seed", "collection_seed",
                       "opponent", "candidate_artifact_hash", "candidate_source_hash", "execution_tier",
                       "strict", "machine", "manifest_hash")
        matching = next((record for record in reusable if all(
            digest(record.get(key)) == digest(base.get(key)) for key in equivalence)), None)
        if matching is not None:
            record = dict(matching, **base, reused_from_request_hash=matching["request_hash"])
            append_jsonl(log_file, record)
            previous.append(record)
            print(game_id, "reused", record["outcome"], flush=True)
            continue
        world = None
        try:
            referee = SnapshotWorld(row["setup_seed"], *runners, allow_moves=True, win_length=4)
            if tuple(referee.obstacle) != tuple(row["obstacle"]) or referee.removed != row["removed_size"]:
                raise ValueError("actual initial configuration differs from the manifest")
            world = referee
            game = world.play()
        except BaseException as error:
            try:
                # The referee may have finished before its own runner cleanup was interrupted.
                if world is not None and world.over and world.record.winner is not None:
                    record = _adjudicated_row(base, world, world.record, seat, variant["step_budget"])
                    record["post_play_error"] = traceback.format_exc()
                elif isinstance(error, Exception):
                    record = dict(base, status="infrastructure_error", error=traceback.format_exc(), retryable=True)
                else:
                    raise
                append_jsonl(log_file, record)
            finally:
                _stop_after_failure(runners)
            if not isinstance(error, Exception):
                raise
        else:
            record = _adjudicated_row(base, world, game, seat, variant["step_budget"])
            try:
                telemetry = _telemetry(game.log_paths[seat])
                for turn in world.turns:
                    if turn["player"] == seat:
                        turn["search"] = telemetry.get(turn["ply"])
            except BaseException as error:
                record["telemetry_error"] = traceback.format_exc()
                if not isinstance(error, Exception):
                    raise
            finally:
                append_jsonl(log_file, record)
        previous.append(record)
        print(game_id, record["status"], record.get("outcome", "pending"), flush=True)
    summary = summarize_games(previous, [item["game_id"] for item in planned])
    summary.update(request_hash=request_hash, variant_id=variant["variant_id"], execution_tier=execution_tier,
                   code_hash=provenance["code_hash"],
                   formal_ranking_allowed=execution_tier == "formal" and summary["eligible"])
    atomic_json(output / "summary.json", summary)
    return summary


def _assign_stratum(games, game_ids, rng, used_states, limit):
    """Match games to distinct states, moving earlier assignments when needed."""
    options, owners, selected = {}, {}, {}

    def assign(game_id, seen):
        for state_hash, game, turn in options[game_id]:
            if state_hash in seen or state_hash in used_states:
                continue
            seen.add(state_hash)
            owner = owners.get(state_hash)
            if owner is None or assign(owner, seen):
                owners[state_hash] = game_id
                selected[game_id] = state_hash, game, turn
                return True
        return False

    for game_id in game_ids:
        if len(selected) >= limit:
            break
        candidates = list(games[game_id])
        rng.shuffle(candidates)
        options[game_id] = [(digest(turn["state"]), game, turn) for game, turn in candidates]
        assign(game_id, set())
    return selected


def sample_positions(game_records, seed=424, per_group=2):
    """One state per distinct game within each class/size/stage stratum."""
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for game in game_records:
        if game.get("status") != "adjudicated" or game.get("split") not in ("D0", "D1"):
            continue
        for turn in game["turns"]:
            if turn["move"] is None or "failure_reason" in turn:
                continue
            key = (obstacle_class(game["obstacle"]), game["removed_size"], stage(turn["ply"]))
            groups[key][game["game_id"]].append((game, turn))
    positions, coverage, used_states = [], [], set()
    for kind in ("corner", "edge", "interior"):
        for removed in range(3):
            for phase in ("early", "middle", "late"):
                key = kind, removed, phase
                game_ids = sorted(groups[key])
                rng.shuffle(game_ids)
                assignments = _assign_stratum(groups[key], game_ids, rng, used_states, per_group)
                for game_id, (state_hash, game, turn) in assignments.items():
                    used_states.add(state_hash)
                    positions.append({"position_id": digest([game_id, turn["ply"], turn["state"]])[:16],
                        "state": turn["state"], "source_game": game_id, "source_variant": game["variant_id"],
                        "split": game["split"], "config_id": game["config_id"],
                        "removed_size": removed, "obstacle_class": kind, "stage": phase,
                        "near_cap": turn["ply"] >= 90, "source_move": turn["move"],
                        "actor_role": turn["actor_role"]})
                selected = len(assignments)
                coverage.append({"obstacle_class": kind, "removed_size": removed, "stage": phase,
                                 "target": per_group, "available_games": len(game_ids), "selected": selected,
                                 "missing": max(0, per_group - selected)})
    return {"schema": SCHEMA_VERSION, "sample_seed": seed, "target": 27 * per_group,
            "count": len(positions), "positions": positions, "coverage": coverage,
            "positions_hash": digest(positions)}
