"""Student agent: edit ONLY this file."""
from dataclasses import dataclass, field
from functools import lru_cache
import time

import numpy as np
import constants
from constants import (BOARD_SIZE, NUM_SIZES, EMPTY, PLAYER_1, PLAYER_2,
                       OBSTACLE, INITIAL_RESERVE, MAX_PLIES)
from helpers import LINES, Move, get_valid_moves
from agents.agent import Agent
from store import register_agent


# Game state

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


# Evaluation

LINE_WEIGHT = 1000
SIZE_WEIGHTS = (1, 20, 100)
WIN_SCORE = 100000


def evaluate_e0(state, player):
    """Return nonterminal line control from player's perspective."""
    score = 0
    opponent = 3 - player

    for index, owner in enumerate(state.top_owners):
        if owner == player:
            score += LINE_WEIGHT * state.line_degrees[index]
        elif owner == opponent:
            score -= LINE_WEIGHT * state.line_degrees[index]

    return score


def evaluate(state, player):
    """Return E1 for a nonterminal state from player's perspective."""
    score = evaluate_e0(state, player)
    opponent = 3 - player

    for index, owner in enumerate(state.top_owners):
        if owner == player:
            score += SIZE_WEIGHTS[state.top_sizes[index]]
        elif owner == opponent:
            score -= SIZE_WEIGHTS[state.top_sizes[index]]

    return score


def terminal_utility(state, player, winner):
    """Score a finished state using the winner returned by terminal(state)."""
    if winner == 0:
        return 0
    if winner == player:
        return WIN_SCORE - state.ply
    return -WIN_SCORE + state.ply


# Move ordering

def move_code(move):
    """Return (source, destination, size), using -1 for reserve moves."""
    source = -1
    if move.src is not None:
        source = move.src[0] * BOARD_SIZE + move.src[1]
    destination = move.dest[0] * BOARD_SIZE + move.dest[1]
    return source, destination, move.size


def order_moves(state, preferred_move=None, deadline=None):
    """Return (move, child) pairs for a nonterminal state.

    A legal preferred move comes first; an illegal hint is ignored.
    """
    scored_moves = []
    preferred = None

    for move in legal_moves(state, deadline):
        child = apply_move(state, move)
        finished, winner, _ = terminal(child, deadline)

        if move == preferred_move:
            preferred = (move, child)
            continue

        # Use the perspective of the player choosing the move.
        if finished:
            score = terminal_utility(child, state.player, winner)
        else:
            score = evaluate(child, state.player)
        scored_moves.append((score, move, child))

    scored_moves.sort(key=lambda item: (-item[0], move_code(item[1])))

    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    for _, move, child in scored_moves:
        ordered.append((move, child))

    if deadline is not None and time.perf_counter() >= deadline:
        raise StateTimeout("move-ordering deadline exceeded")
    return ordered


# Transposition table

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


# Alpha beta

@dataclass
class SearchStats:
    """Count recursive calls across rounds, including interrupted work."""
    nodes: int = 0


def alpha_beta(state, depth, alpha=-float("inf"),
               beta=float("inf"), deadline=None, preferred_move=None, stats=None,
               tt=None):
    """Return (value, best_move, nodes) from state.player's perspective.

    A full window gives the exact value at this depth. The deadline is an
    absolute perf_counter time; StateTimeout propagates to the caller.
    The preferred move applies only at this node.
    The node count measures recursive search calls.
    Optional stats also counts calls interrupted by a timeout.
    Optional tt reuses values only at the same remaining depth.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if alpha >= beta:
        raise ValueError("alpha must be less than beta")

    if stats is not None:
        stats.nodes += 1

    # terminal() also checks the deadline at recursive entry.
    finished, winner, _ = terminal(state, deadline)
    if finished:
        return terminal_utility(state, state.player, winner), None, 1
    if depth == 0:
        return evaluate(state, state.player), None, 1

    alpha_start, beta_start = alpha, beta
    if tt is not None:
        key = encode_state(state)
        entry = tt.get(key, depth)
        if entry is not None and is_legal_move(state, entry.best_move):
            if entry.flag == EXACT:
                return entry.value, entry.best_move, 1
            if entry.flag == LOWER and entry.value >= beta:
                return entry.value, entry.best_move, 1
            if entry.flag == UPPER and entry.value <= alpha:
                return entry.value, entry.best_move, 1

        # A non-cutting bound does not tighten the search window.
        if preferred_move is None:
            preferred_move = tt.get_move(key)

    best_value = -float("inf")
    best_move = None
    nodes = 1

    for move, child in order_moves(state, preferred_move, deadline):
        child_value, _, child_nodes = alpha_beta(
            child, depth - 1, -beta, -alpha, deadline, stats=stats, tt=tt
        )
        value = -child_value
        nodes += child_nodes

        # An equal cutoff bound does not establish a genuine tie.
        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, best_value)
        if alpha >= beta:
            break

    if tt is not None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateTimeout("search deadline exceeded before caching")
        if best_value <= alpha_start:
            flag = UPPER
        elif best_value >= beta_start:
            flag = LOWER
        else:
            flag = EXACT
        tt.store(key, depth, best_value, flag, best_move)

    return best_value, best_move, nodes


# Iterative deepening

@dataclass
class SearchResult:
    """Root result; iterations stores completed (depth, move, value) tuples."""
    move: Move | None = None
    completed_depth: int = 0
    value: int | None = None
    nodes: int = 0
    elapsed: float = 0.0
    status: str = "fallback"
    iterations: list = field(default_factory=list)


def search(state, deadline):
    """Search the official course variant until the shared deadline.

    deadline is an absolute perf_counter time supplied by the caller.
    elapsed measures this call. Status is terminal, fallback, or completed;
    completed means at least one depth finished, not that the game is solved.
    """
    started = time.perf_counter()

    # Bounded preparation also handles an already-expired deadline.
    finished, winner, _ = terminal(state)
    if finished:
        return SearchResult(
            value=terminal_utility(state, state.player, winner),
            elapsed=time.perf_counter() - started, status="terminal"
        )

    result = SearchResult(move=next(iter_legal_moves(state)))
    stats = SearchStats()
    tt = TranspositionTable()
    preferred_move = None

    for depth in range(1, MAX_PLIES - state.ply + 1):
        if time.perf_counter() >= deadline:
            break

        try:
            value, move, _ = alpha_beta(
                state, depth, -float("inf"), float("inf"),
                deadline=deadline, preferred_move=preferred_move, stats=stats,
                tt=tt
            )
        except StateTimeout:
            break

        # Commit only a complete, on-time root search.
        if time.perf_counter() >= deadline:
            break
        result.move = move
        result.completed_depth = depth
        result.value = value
        result.status = "completed"
        result.iterations.append((depth, move, value))
        preferred_move = move

    result.nodes = stats.nodes
    result.elapsed = time.perf_counter() - started
    return result


@register_agent("student_agent")
class StudentAgent(Agent):
    """Iterative-deepening Alpha-Beta agent for the course rules."""

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.call_count = 0

    def step(self, board, reserves, player, opponent):
        started = time.perf_counter()
        deadline = started + 1.6

        # Each seat is called once every two plies.
        ply = 2 * self.call_count + player - 1
        self.call_count += 1

        moves = get_valid_moves(board, reserves, player)
        fallback = moves[0]
        if len(moves) == 1:
            return fallback

        state = GameState.from_observation(board, reserves, player, ply)
        if time.perf_counter() >= deadline:
            return fallback

        result = search(state, deadline)
        return result.move if result.move in moves else fallback
