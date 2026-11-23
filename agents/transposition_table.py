"""Completed Alpha-Beta results and move hints in raw board coordinates."""

from dataclasses import dataclass
from helpers import Move

EXACT = "EXACT"
LOWER = "LOWER"
UPPER = "UPPER"


def encode_state(state):
    """Return a lossless immutable key, including covered pieces and ply."""
    return (state.board, state.reserves, state.player, state.obstacle,
            state.ply, state.allow_moves, state.win_length)


@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: str
    best_move: Move


class TranspositionTable:
    """Bound numeric entries by capacity; keep move hints across depths."""

    def __init__(self, max_entries=50000):
        if max_entries < 0:
            raise ValueError("max_entries must be nonnegative")
        self.max_entries = max_entries
        self.entries = {}
        self.best_moves = {}

    def get(self, key, depth):
        return self.entries.get((key, depth))

    def get_move(self, key):
        return self.best_moves.get(key)

    def store(self, key, depth, value, flag, best_move):
        entry_key = (key, depth)
        if entry_key not in self.entries and len(self.entries) >= self.max_entries:
            return
        self.entries[entry_key] = TTEntry(value, flag, best_move)
        if best_move is not None:
            self.best_moves[key] = best_move

    def clear(self):
        self.entries.clear()
        self.best_moves.clear()
