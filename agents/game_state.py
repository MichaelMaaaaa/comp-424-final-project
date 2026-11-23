"""Game state and rules for the COMP 424 Goblet Gobblers course variant.

Development module: inline this module in student_agent.py for final submission.
Use GameState.from_observation() at step() entry and apply_move() in the search.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import time

import numpy as np
import constants
from constants import (BOARD_SIZE, NUM_SIZES, EMPTY, PLAYER_1, PLAYER_2,
                       OBSTACLE, INITIAL_RESERVE, MAX_PLIES)
from helpers import LINES, Move


class StateTimeout(TimeoutError):
    """Move generation was interrupted; do not treat a partial list as complete."""


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise StateTimeout("game-state deadline exceeded")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _cell_index(cell):
    try:
        row, col = cell
    except (TypeError, ValueError):
        raise ValueError("a cell must be a (row, col) pair") from None
    row = _integer(row, "row")
    col = _integer(col, "col")
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("cell is outside the board")
    return row * BOARD_SIZE + col


@lru_cache(maxsize=64)
def _line_geometry(obstacle_index, win_length):
    lines = tuple(tuple(r * BOARD_SIZE + c for r, c in line)
                  for line in LINES[win_length]
                  if all(r * BOARD_SIZE + c != obstacle_index for r, c in line))
    degrees = [0] * (BOARD_SIZE * BOARD_SIZE)
    for line in lines:
        for index in line:
            degrees[index] += 1
    return lines, tuple(degrees)


@lru_cache(maxsize=1)
def _move_tables():
    cells = tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    placements = tuple(tuple(Move(None, cell, size) for cell in cells)
                       for size in range(NUM_SIZES))
    relocations = tuple(tuple(tuple(Move(src, dest, size) for dest in cells)
                             for src in cells) for size in range(NUM_SIZES))
    return placements, relocations


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable full state; covered pieces are retained.

    board[index][size], where index = row * BOARD_SIZE + col.
    reserves[player][size]; row 0 is retained and contains zeros.
    player is the NEXT player to act; ply counts COMPLETED turns.

    Use from_observation() for public input validation. The raw constructor is
    for already validated tuple data, as used by apply_move().
    """

    board: tuple
    reserves: tuple
    player: int
    obstacle: tuple
    ply: int
    allow_moves: bool = field(default_factory=lambda: bool(constants.ALLOW_MOVES))
    win_length: int = field(default_factory=lambda: int(constants.WIN_LENGTH))
    top_owners: tuple = field(init=False, repr=False, compare=False)
    top_sizes: tuple = field(init=False, repr=False, compare=False)
    active_lines: tuple = field(init=False, repr=False, compare=False)
    line_degrees: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        owners = []
        sizes = []
        for stack in self.board:
            if stack[0] == OBSTACLE:
                owners.append(OBSTACLE)
                sizes.append(-1)
                continue
            for size in range(NUM_SIZES - 1, -1, -1):
                if stack[size] != EMPTY:
                    owners.append(stack[size])
                    sizes.append(size)
                    break
            else:
                owners.append(EMPTY)
                sizes.append(-1)
        obstacle_index = self.obstacle[0] * BOARD_SIZE + self.obstacle[1]
        lines, degrees = _line_geometry(obstacle_index, self.win_length)
        object.__setattr__(self, "top_owners", tuple(owners))
        object.__setattr__(self, "top_sizes", tuple(sizes))
        object.__setattr__(self, "active_lines", lines)
        object.__setattr__(self, "line_degrees", degrees)

    @classmethod
    def from_observation(cls, board, reserves, player, ply):
        """Copy the framework arrays into a full immutable decision state.

        Supply ply explicitly. With the tournament rules, call number k (from
        zero) for an agent in seat player corresponds to 2*k + player - 1.
        Rule flags are read from constants, never assigned by this module.
        """
        board_array = np.asarray(board)
        reserve_array = np.asarray(reserves)
        if board_array.shape != (BOARD_SIZE, BOARD_SIZE, NUM_SIZES):
            raise ValueError("board has the wrong shape")
        if reserve_array.shape != (3, NUM_SIZES):
            raise ValueError("reserves has the wrong shape")
        if not np.issubdtype(board_array.dtype, np.integer):
            raise ValueError("board must contain integers")
        if not np.issubdtype(reserve_array.dtype, np.integer):
            raise ValueError("reserves must contain integers")
        player = _integer(player, "player")
        ply = _integer(ply, "ply")
        if player not in (PLAYER_1, PLAYER_2):
            raise ValueError("player must be 1 or 2")
        if ply < 0:
            raise ValueError("ply must be nonnegative")
        stacks = tuple(tuple(int(v) for v in stack)
                       for stack in board_array.reshape(-1, NUM_SIZES))
        obstacle_indices = []
        for index, stack in enumerate(stacks):
            if OBSTACLE in stack:
                if stack != (OBSTACLE,) * NUM_SIZES:
                    raise ValueError("the obstacle must fill every size slot")
                obstacle_indices.append(index)
            elif any(v not in (EMPTY, PLAYER_1, PLAYER_2) for v in stack):
                raise ValueError("invalid piece owner")
        if len(obstacle_indices) != 1:
            raise ValueError("board must contain exactly one obstacle")
        reserve_rows = tuple(tuple(int(v) for v in row) for row in reserve_array)
        if any(reserve_rows[0]):
            raise ValueError("unused reserve row 0 must contain zeros")
        if any(v < 0 or v > INITIAL_RESERVE[size]
               for row in reserve_rows[1:] for size, v in enumerate(row)):
            raise ValueError("invalid reserve count")
        return cls(stacks, reserve_rows, player,
                   divmod(obstacle_indices[0], BOARD_SIZE), ply)


def initial_state(obstacle=(0, 0), removed_size=0):
    """Build one of the 16*3 starting configurations without using randomness."""
    index = _cell_index(obstacle)
    removed_size = _integer(removed_size, "removed_size")
    if not 0 <= removed_size < NUM_SIZES:
        raise ValueError("invalid removed size")
    board = [(EMPTY,) * NUM_SIZES] * (BOARD_SIZE * BOARD_SIZE)
    board[index] = (OBSTACLE,) * NUM_SIZES
    counts = list(INITIAL_RESERVE)
    counts[removed_size] -= 1
    reserves = ((0,) * NUM_SIZES, tuple(counts), tuple(counts))
    return GameState(tuple(board), reserves, PLAYER_1, divmod(index, BOARD_SIZE), 0)


def to_arrays(state):
    """Return fresh int8 framework arrays. Neither array aliases the state."""
    board = np.array(state.board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE, NUM_SIZES)
    return board, np.array(state.reserves, dtype=np.int8)


def top_at(state, cell):
    index = _cell_index(cell)
    return state.top_owners[index], state.top_sizes[index]


def iter_legal_moves(state, deadline=None):
    """Yield all mechanically legal helpers.Move objects in framework order.

    Call terminal() before expanding a search node. As in the framework,
    this generator does not suppress moves just because a line already exists.
    A deadline is an absolute time.perf_counter() value, not a duration.
    """
    _check_deadline(deadline)
    placements, relocations = _move_tables()
    owners, sizes = state.top_owners, state.top_sizes
    count = BOARD_SIZE * BOARD_SIZE
    for size in range(NUM_SIZES):
        _check_deadline(deadline)
        if state.reserves[state.player][size] <= 0:
            continue
        for dest in range(count):
            _check_deadline(deadline)
            if owners[dest] != OBSTACLE and sizes[dest] < size:
                yield placements[size][dest]
    if not state.allow_moves:
        return
    for src in range(count):
        _check_deadline(deadline)
        if owners[src] != state.player:
            continue
        size = sizes[src]
        for dest in range(count):
            _check_deadline(deadline)
            if dest != src and owners[dest] != OBSTACLE and sizes[dest] < size:
                yield relocations[size][src][dest]


def legal_moves(state, deadline=None):
    """Return the complete legal move list, or raise StateTimeout."""
    return list(iter_legal_moves(state, deadline))


def has_legal_move(state, deadline=None):
    return next(iter_legal_moves(state, deadline), None) is not None


def is_legal_move(state, move):
    if not isinstance(move, Move):
        return False
    dest = move.dest[0] * BOARD_SIZE + move.dest[1]
    if state.top_owners[dest] == OBSTACLE or state.top_sizes[dest] >= move.size:
        return False
    if move.src is None:
        return state.reserves[state.player][move.size] > 0
    if not state.allow_moves or move.src == move.dest:
        return False
    src = move.src[0] * BOARD_SIZE + move.src[1]
    return state.top_owners[src] == state.player and state.top_sizes[src] == move.size


def apply_move(state, move):
    """Lift AND place a legal piece, then switch player and increment ply.

    The parent and its covered pieces are never modified. No terminal check
    runs between lifting and placement; terminal() is called on the successor.
    """
    if not is_legal_move(state, move):
        raise ValueError(f"illegal move for player {state.player}: {move!r}")
    board = list(state.board)
    reserves = state.reserves
    if move.src is None:
        counts = list(reserves[state.player])
        counts[move.size] -= 1
        rows = list(reserves)
        rows[state.player] = tuple(counts)
        reserves = tuple(rows)
    else:
        src = move.src[0] * BOARD_SIZE + move.src[1]
        stack = list(board[src])
        stack[move.size] = EMPTY
        board[src] = tuple(stack)
    dest = move.dest[0] * BOARD_SIZE + move.dest[1]
    stack = list(board[dest])
    stack[move.size] = state.player
    board[dest] = tuple(stack)
    return GameState(tuple(board), reserves, 3 - state.player, state.obstacle,
                     state.ply + 1, state.allow_moves, state.win_length)


def has_line(state, player):
    """Test only top pieces, ignoring complete lines through the obstacle."""
    return any(all(state.top_owners[index] == player for index in line)
               for line in state.active_lines)


def terminal(state, deadline=None):
    """Return (finished, winner, reason), matching helpers.check_endgame.

    After a complete move, the next player's line has priority. A line on
    ply MAX_PLIES wins before the draw cap is considered. Empty reserves and
    repeated positions alone do not end the tournament game.
    """
    _check_deadline(deadline)
    player = state.player
    opponent = 3 - player
    if has_line(state, player):
        return True, player, "revealed"
    if has_line(state, opponent):
        return True, opponent, "line"
    if state.ply >= MAX_PLIES:
        return True, 0, "ply_cap"
    if has_legal_move(state, deadline):
        return False, 0, ""
    if state.allow_moves:
        return True, opponent, "no_moves"
    other = GameState(state.board, state.reserves, opponent, state.obstacle,
                      state.ply, state.allow_moves, state.win_length)
    if not has_legal_move(other, deadline):
        return True, 0, "exhausted"
    return False, 0, ""


def must_pass(state, deadline=None):
    """Only for the framework's optional ALLOW_MOVES=False diagnostic mode."""
    return (not state.allow_moves and not terminal(state, deadline)[0]
            and not has_legal_move(state, deadline))


def apply_pass(state):
    """Advance a forced diagnostic-mode pass; passing is never voluntary."""
    if not must_pass(state):
        raise ValueError("passing is not legal in this state")
    return GameState(state.board, state.reserves, 3 - state.player, state.obstacle,
                     state.ply + 1, state.allow_moves, state.win_length)
