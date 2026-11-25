"""Configuration, provenance, JSON schemas and atomic checkpoints."""

from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from agents.game_state import GameState, terminal
from agents.search_engine import SearchConfig, default_evaluator
from helpers import Move

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a407698955bb2ea9931c1264b2d0b7d0f8db479f"
SCHEMA_VERSION = "comp424-experiments-v1"


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Move):
        return {"src": value.src, "dest": value.dest, "size": value.size}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value):
    return json.dumps(value, default=json_default, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A killed writer may leave a final partial line. Earlier corruption is fatal.
            if index != len(lines) - 1:
                raise
    return records


def repair_jsonl_tail(path):
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        # Preserve a valid final record; remove only an incomplete tail.
        tail = data.rsplit(b"\n", 1)[-1]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            path.write_bytes(data[:len(data) - len(tail)])
        else:
            path.write_bytes(data + b"\n")


def code_manifest():
    paths = [p for p in (REPO_ROOT / "agents").glob("*.py")]
    paths += [p for p in (REPO_ROOT / "search_experiments").glob("*.py")]
    paths += [REPO_ROOT / name for name in ("helpers.py", "constants.py", "runners.py", "world.py",
                                           "allowed_imports.py", "validate_submission.py", "requirements.txt")]
    paths += [REPO_ROOT / "baselines" / "ab_baseline_agent.py"]
    files = {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in sorted(paths)}
    return {"source_commit": SOURCE_COMMIT, "files": files, "code_hash": digest(files)}


def machine_info():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def state_record(state):
    return {"board": state.board, "reserves": state.reserves, "player": state.player,
            "obstacle": state.obstacle, "ply": state.ply,
            "allow_moves": state.allow_moves, "win_length": state.win_length}


def restore_state(record):
    board = np.asarray(record["board"]).reshape(4, 4, 3)
    reserves = np.asarray(record["reserves"])
    state = GameState.from_observation(board, reserves, record["player"], record["ply"])
    if tuple(record["obstacle"]) != state.obstacle:
        raise ValueError("serialized obstacle disagrees with full board")
    return GameState(state.board, state.reserves, state.player, state.obstacle, state.ply,
                     record.get("allow_moves", True), record.get("win_length", 4))


def restore_move(record):
    if record is None or isinstance(record, Move):
        return record
    return Move(None if record["src"] is None else tuple(record["src"]), tuple(record["dest"]), record["size"])


def obstacle_class(obstacle):
    row, col = obstacle
    borders = int(row in (0, 3)) + int(col in (0, 3))
    return "corner" if borders == 2 else "edge" if borders == 1 else "interior"


def stage(ply):
    return "early" if ply < 10 else "middle" if ply < 30 else "late"


def normalize_variant(variant):
    result = dict(variant)
    if not isinstance(result.get("variant_id"), str) or not result["variant_id"].strip():
        raise ValueError("variant_id is required")
    result.setdefault("parent_variant", None)
    result["search"] = asdict(SearchConfig(**result.get("search", {})))
    result.setdefault("evaluation", {"evaluation_id": "E1"})
    result.setdefault("book", {"mode": "N"})
    result.setdefault("step_budget", 1.6)
    result.setdefault("telemetry", True)
    result["book_hash"] = result["book"].get("hash")
    result.pop("config_hash", None)
    result["config_hash"] = digest(result)
    return json.loads(canonical_json(result))


def resolve_evaluator(config):
    provider = importlib.import_module("agents.evaluation")
    factory = getattr(provider, "make_evaluator", default_evaluator)
    return factory(config)


def validate_positions(records, allow_test=False):
    if not records:
        raise ValueError("fixed position set is empty")
    for record in records:
        if record.get("split") == "T" and not allow_test:
            raise ValueError("held-out T positions must not enter development selection")
        if terminal(restore_state(record["state"]))[0]:
            raise ValueError("diagnostic positions must be nonterminal")
    return records


def run_command(command, log_path, cwd=REPO_ROOT):
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")
    return completed.stdout
