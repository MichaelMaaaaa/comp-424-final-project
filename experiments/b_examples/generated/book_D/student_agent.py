from dataclasses import dataclass, field
from functools import lru_cache
import time
import numpy as np
import constants
from constants import BOARD_SIZE, NUM_SIZES, EMPTY, PLAYER_1, PLAYER_2, OBSTACLE, INITIAL_RESERVE, MAX_PLIES
from helpers import LINES, Move
from constants import BOARD_SIZE, NUM_SIZES
from helpers import Move
from dataclasses import dataclass
from dataclasses import asdict, dataclass, field, replace
import math
import sys
from constants import BOARD_SIZE, MAX_PLIES
from dataclasses import asdict
import json
from agents.agent import Agent
from helpers import get_valid_moves
import base64
import hashlib
import zlib
from store import register_agent

class StateTimeout(TimeoutError):
    """Move generation was interrupted; do not treat a partial list as complete."""

def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise StateTimeout('game-state deadline exceeded')

def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)

def _cell_index(cell):
    try:
        row, col = cell
    except (TypeError, ValueError):
        raise ValueError('a cell must be a (row, col) pair') from None
    row = _integer(row, 'row')
    col = _integer(col, 'col')
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError('cell is outside the board')
    return row * BOARD_SIZE + col

@lru_cache(maxsize=64)
def _line_geometry(obstacle_index, win_length):
    lines = tuple((tuple((r * BOARD_SIZE + c for r, c in line)) for line in LINES[win_length] if all((r * BOARD_SIZE + c != obstacle_index for r, c in line))))
    degrees = [0] * (BOARD_SIZE * BOARD_SIZE)
    for line in lines:
        for index in line:
            degrees[index] += 1
    return (lines, tuple(degrees))

@lru_cache(maxsize=1)
def _move_tables():
    cells = tuple(((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)))
    placements = tuple((tuple((Move(None, cell, size) for cell in cells)) for size in range(NUM_SIZES)))
    relocations = tuple((tuple((tuple((Move(src, dest, size) for dest in cells)) for src in cells)) for size in range(NUM_SIZES)))
    return (placements, relocations)

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
        object.__setattr__(self, 'top_owners', tuple(owners))
        object.__setattr__(self, 'top_sizes', tuple(sizes))
        object.__setattr__(self, 'active_lines', lines)
        object.__setattr__(self, 'line_degrees', degrees)

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
            raise ValueError('board has the wrong shape')
        if reserve_array.shape != (3, NUM_SIZES):
            raise ValueError('reserves has the wrong shape')
        if not np.issubdtype(board_array.dtype, np.integer):
            raise ValueError('board must contain integers')
        if not np.issubdtype(reserve_array.dtype, np.integer):
            raise ValueError('reserves must contain integers')
        player = _integer(player, 'player')
        ply = _integer(ply, 'ply')
        if player not in (PLAYER_1, PLAYER_2):
            raise ValueError('player must be 1 or 2')
        if ply < 0:
            raise ValueError('ply must be nonnegative')
        stacks = tuple((tuple((int(v) for v in stack)) for stack in board_array.reshape(-1, NUM_SIZES)))
        obstacle_indices = []
        for index, stack in enumerate(stacks):
            if OBSTACLE in stack:
                if stack != (OBSTACLE,) * NUM_SIZES:
                    raise ValueError('the obstacle must fill every size slot')
                obstacle_indices.append(index)
            elif any((v not in (EMPTY, PLAYER_1, PLAYER_2) for v in stack)):
                raise ValueError('invalid piece owner')
        if len(obstacle_indices) != 1:
            raise ValueError('board must contain exactly one obstacle')
        reserve_rows = tuple((tuple((int(v) for v in row)) for row in reserve_array))
        if any(reserve_rows[0]):
            raise ValueError('unused reserve row 0 must contain zeros')
        if any((v < 0 or v > INITIAL_RESERVE[size] for row in reserve_rows[1:] for size, v in enumerate(row))):
            raise ValueError('invalid reserve count')
        return cls(stacks, reserve_rows, player, divmod(obstacle_indices[0], BOARD_SIZE), ply)

def initial_state(obstacle=(0, 0), removed_size=0):
    """Build one of the 16*3 starting configurations without using randomness."""
    index = _cell_index(obstacle)
    removed_size = _integer(removed_size, 'removed_size')
    if not 0 <= removed_size < NUM_SIZES:
        raise ValueError('invalid removed size')
    board = [(EMPTY,) * NUM_SIZES] * (BOARD_SIZE * BOARD_SIZE)
    board[index] = (OBSTACLE,) * NUM_SIZES
    counts = list(INITIAL_RESERVE)
    counts[removed_size] -= 1
    reserves = ((0,) * NUM_SIZES, tuple(counts), tuple(counts))
    return GameState(tuple(board), reserves, PLAYER_1, divmod(index, BOARD_SIZE), 0)

def to_arrays(state):
    """Return fresh int8 framework arrays. Neither array aliases the state."""
    board = np.array(state.board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE, NUM_SIZES)
    return (board, np.array(state.reserves, dtype=np.int8))

def top_at(state, cell):
    index = _cell_index(cell)
    return (state.top_owners[index], state.top_sizes[index])

def iter_legal_moves(state, deadline=None):
    """Yield all mechanically legal helpers.Move objects in framework order.

    Call terminal() before expanding a search node. As in the framework,
    this generator does not suppress moves just because a line already exists.
    A deadline is an absolute time.perf_counter() value, not a duration.
    """
    _check_deadline(deadline)
    placements, relocations = _move_tables()
    owners, sizes = (state.top_owners, state.top_sizes)
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
            if dest != src and owners[dest] != OBSTACLE and (sizes[dest] < size):
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
        raise ValueError(f'illegal move for player {state.player}: {move!r}')
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
    return GameState(tuple(board), reserves, 3 - state.player, state.obstacle, state.ply + 1, state.allow_moves, state.win_length)

def has_line(state, player):
    """Test only top pieces, ignoring complete lines through the obstacle."""
    return any((all((state.top_owners[index] == player for index in line)) for line in state.active_lines))

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
        return (True, player, 'revealed')
    if has_line(state, opponent):
        return (True, opponent, 'line')
    if state.ply >= MAX_PLIES:
        return (True, 0, 'ply_cap')
    if has_legal_move(state, deadline):
        return (False, 0, '')
    if state.allow_moves:
        return (True, opponent, 'no_moves')
    other = GameState(state.board, state.reserves, opponent, state.obstacle, state.ply, state.allow_moves, state.win_length)
    if not has_legal_move(other, deadline):
        return (True, 0, 'exhausted')
    return (False, 0, '')

def must_pass(state, deadline=None):
    """Only for the framework's optional ALLOW_MOVES=False diagnostic mode."""
    return not state.allow_moves and (not terminal(state, deadline)[0]) and (not has_legal_move(state, deadline))

def apply_pass(state):
    """Advance a forced diagnostic-mode pass; passing is never voluntary."""
    if not must_pass(state):
        raise ValueError('passing is not legal in this state')
    return GameState(state.board, state.reserves, 3 - state.player, state.obstacle, state.ply + 1, state.allow_moves, state.win_length)
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
TRANSFORMS = (0, 1, 2, 3, 4, 5, 6, 7)
INVERSE_TRANSFORMS = (0, 3, 2, 1, 4, 5, 6, 7)
BOARD_SLOTS = BOARD_SIZE * BOARD_SIZE * NUM_SIZES
BOOK_KEY_SIZE = BOARD_SLOTS + 2 * NUM_SIZES + 2

def _check_transform_id(transform):
    if isinstance(transform, bool) or not isinstance(transform, int) or transform not in TRANSFORMS:
        raise ValueError('transform must be an integer from 0 to 7')

def encode_book_state(state):
    """Encode all board slots, both reserves, player, and ply in that order."""
    if not state.allow_moves or state.win_length != 4:
        raise ValueError('position-book keys require the official True/4 rules')
    values = []
    for stack in state.board:
        values.extend(stack)
    for player in (1, 2):
        values.extend(state.reserves[player])
    values.extend((state.player, state.ply))
    return tuple(values)

def decode_book_state(key):
    """Restore a full state with official rules, independently of global flags."""
    if len(key) != BOOK_KEY_SIZE:
        raise ValueError('position-book key has the wrong length')
    if any((isinstance(v, bool) or not isinstance(v, int) for v in key)):
        raise ValueError('position-book key must contain integers')
    key = tuple(key)
    stacks = [key[i:i + NUM_SIZES] for i in range(0, BOARD_SLOTS, NUM_SIZES)]
    board = [stacks[i:i + BOARD_SIZE] for i in range(0, len(stacks), BOARD_SIZE)]
    reserve_start = BOARD_SLOTS
    reserves = [(0,) * NUM_SIZES, key[reserve_start:reserve_start + NUM_SIZES], key[reserve_start + NUM_SIZES:reserve_start + 2 * NUM_SIZES]]
    state = GameState.from_observation(board, reserves, key[-2], key[-1])
    return GameState(state.board, state.reserves, state.player, state.obstacle, state.ply, allow_moves=True, win_length=4)

def transform_cell(cell, transform):
    """Map an original cell to its coordinates after the selected transform."""
    _check_transform_id(transform)
    row, col = cell
    if any((isinstance(v, bool) or not isinstance(v, int) for v in (row, col))):
        raise ValueError('cell coordinates must be integers')
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError('cell is outside the board')
    if transform >= 4:
        col = BOARD_SIZE - 1 - col
    for _ in range(transform % 4):
        row, col = (col, BOARD_SIZE - 1 - row)
    return (row, col)

def transform_state(state, transform):
    """Move whole stacks and the obstacle; preserve all other state fields."""
    board = [None] * (BOARD_SIZE * BOARD_SIZE)
    for index, stack in enumerate(state.board):
        row, col = transform_cell(divmod(index, BOARD_SIZE), transform)
        board[row * BOARD_SIZE + col] = stack
    obstacle = transform_cell(state.obstacle, transform)
    return GameState(tuple(board), state.reserves, state.player, obstacle, state.ply, state.allow_moves, state.win_length)

def transform_move(move, transform):
    """Transform both endpoints, keeping the reserve marker and piece size."""
    source = None
    if move.src is not None:
        source = transform_cell(move.src, transform)
    destination = transform_cell(move.dest, transform)
    return Move(source, destination, move.size)

def inverse_transform_move(move, transform):
    """Map a canonical action back using the query state's transform."""
    _check_transform_id(transform)
    return transform_move(move, INVERSE_TRANSFORMS[transform])

def canonicalize(state):
    """Return the smallest complete key and the first transform attaining it."""
    best_key = None
    best_transform = 0
    for transform in TRANSFORMS:
        key = encode_book_state(transform_state(state, transform))
        if best_key is None or key < best_key:
            best_key = key
            best_transform = transform
    return (best_key, best_transform)
EXACT = 'EXACT'
LOWER = 'LOWER'
UPPER = 'UPPER'

def encode_state(state):
    """Return a lossless immutable key, including covered pieces and ply."""
    return (state.board, state.reserves, state.player, state.obstacle, state.ply, state.allow_moves, state.win_length)

@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: str
    best_move: Move

class TranspositionTable:
    """Bound numeric entries by capacity; keep move hints across depths."""

    def __init__(self, max_entries=50000):
        if max_entries < 0:
            raise ValueError('max_entries must be nonnegative')
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

@dataclass(frozen=True)
class SearchConfig:
    alpha_beta: bool = True
    ordering: bool = True
    tt: bool = True
    preferred_first: bool = False
    reuse_results: bool = False
    key_mode: str = 'tuple'
    symmetry: bool = False
    tt_capacity: int = 50000

    def __post_init__(self):
        for name in ('alpha_beta', 'ordering', 'tt', 'preferred_first', 'reuse_results', 'symmetry'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        if self.key_mode not in ('tuple', 'compact'):
            raise ValueError('key_mode must be tuple or compact')
        if isinstance(self.tt_capacity, bool) or not isinstance(self.tt_capacity, int) or self.tt_capacity < 0:
            raise ValueError('tt_capacity must be a nonnegative integer')
        if not self.ordering and (self.preferred_first or self.reuse_results):
            raise ValueError('ordering optimizations require ordering')
        if not self.tt and (self.symmetry or self.key_mode != 'tuple'):
            raise ValueError('key optimizations require TT')

def component_configs():
    return {'C0': SearchConfig(alpha_beta=False, ordering=False, tt=False), 'C1': SearchConfig(ordering=False, tt=False), 'C2': SearchConfig(tt=False), 'C3': SearchConfig()}

def default_evaluator(config=None):
    """Compatibility adapter; F may supply make_evaluator(config) instead.

    Only the original E0/E1 are implemented here. Parameter experiments belong
    to evaluation.py; unknown parameters fail rather than silently using E1.
    """
    config = dict(config or {'evaluation_id': 'E1'})
    name = config.get('evaluation_id', 'E1')
    strength = config.get('lambda', 0 if name == 'E0' else 1)
    sizes = tuple(config.get('q', (1, 20, 100)))
    if name == 'E0' and strength == 0:
        return evaluate_e0
    if name == 'E1' and strength == 1 and (sizes == (1, 20, 100)):
        return evaluate
    raise ValueError("custom evaluation requires F's make_evaluator(config) provider")

def action_code(move):
    src = -1 if move.src is None else move.src[0] * BOARD_SIZE + move.src[1]
    return (src, move.dest[0] * BOARD_SIZE + move.dest[1], move.size)

def _pack_search_key(state, board=None):
    if not 0 <= state.ply <= MAX_PLIES:
        raise ValueError('compact TT keys require an official-range ply')
    packed = 0
    for stack in state.board if board is None else board:
        for owner in stack:
            packed = packed << 2 | owner
    for row in state.reserves[1:]:
        for count in row:
            packed = packed << 2 | count
    packed = packed << 2 | state.player
    packed = packed << 7 | state.ply
    packed = packed << 1 | int(state.allow_moves)
    packed = packed << 1 | state.win_length - 3
    return packed.to_bytes(15, 'big')

@lru_cache(maxsize=16)
def obstacle_stabilizers(obstacle):
    """Use B's transform API; only transforms fixing THIS obstacle are allowed."""
    result = []
    for transform in TRANSFORMS:
        if transform_cell(obstacle, transform) != obstacle:
            continue
        permutation = [0] * (BOARD_SIZE * BOARD_SIZE)
        for index in range(BOARD_SIZE * BOARD_SIZE):
            row, col = transform_cell(divmod(index, BOARD_SIZE), transform)
            permutation[row * BOARD_SIZE + col] = index
        result.append((transform, tuple(permutation)))
    return tuple(result)

def search_key(state, config):
    """Return (complete key, original-to-key transform); no book canonicalizer."""
    if not config.symmetry:
        return (_pack_search_key(state) if config.key_mode == 'compact' else encode_state(state), 0)
    best = chosen = None
    for transform, permutation in obstacle_stabilizers(state.obstacle):
        board = tuple((state.board[index] for index in permutation))
        if config.key_mode == 'compact':
            key = _pack_search_key(state, board)
        else:
            key = (board, state.reserves, state.player, state.obstacle, state.ply, state.allow_moves, state.win_length)
        if best is None or key < best:
            best, chosen = (key, transform)
    return (best, chosen)

@dataclass
class SearchCounters:
    nodes: int = 0
    successors: int = 0
    terminal_calls: int = 0
    evaluation_calls: int = 0
    cutoffs: int = 0
    first_child_cutoffs: int = 0
    preferred_cutoffs: int = 0
    sorted_children: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_exact_returns: int = 0
    tt_bound_returns: int = 0
    tt_hint_hits: int = 0
    tt_stores: int = 0
    tt_capacity_rejections: int = 0
    key_seconds: float = 0.0
    tt_seconds: float = 0.0
    terminal_returns: int = 0
    depth_returns: int = 0
    interrupted_iterations: int = 0
    successor_seconds: float = 0.0
    terminal_seconds: float = 0.0
    evaluation_seconds: float = 0.0

@dataclass
class EngineResult:
    move: object = None
    value: object = None
    completed_depth: int = 0
    elapsed: float = 0.0
    status: str = 'fallback'
    iterations: list = field(default_factory=list)
    optimal_moves: tuple = ()
    statistics: dict = field(default_factory=dict)
    solved_remaining: bool = False

class SearchEngine:

    def __init__(self, config=None, evaluator=None, utility=None, diagnostics=False):
        self.config = config if isinstance(config, SearchConfig) else SearchConfig(**config or {})
        self.evaluator = evaluator or evaluate
        self.utility = utility or terminal_utility
        self.diagnostics = diagnostics
        self.counters = SearchCounters()
        self.tt = TranspositionTable(self.config.tt_capacity)
        self._root_scores = []

    def _check(self, deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateTimeout('search deadline exceeded')

    def _terminal(self, state, deadline):
        if self.diagnostics:
            self.counters.terminal_calls += 1
            tick = time.perf_counter()
        result = terminal(state, deadline)
        if self.diagnostics:
            self.counters.terminal_seconds += time.perf_counter() - tick
        return result

    def _evaluate(self, state):
        if self.diagnostics:
            self.counters.evaluation_calls += 1
            tick = time.perf_counter()
        result = self.evaluator(state, state.player)
        if self.diagnostics:
            self.counters.evaluation_seconds += time.perf_counter() - tick
        return result

    def _child(self, state, move):
        if self.diagnostics:
            self.counters.successors += 1
            tick = time.perf_counter()
        result = apply_move(state, move)
        if self.diagnostics:
            self.counters.successor_seconds += time.perf_counter() - tick
        return result

    def _children(self, state, preferred, deadline):
        config = self.config
        if not config.ordering:
            for move in iter_legal_moves(state, deadline):
                yield (move, self._child(state, move), None)
            return
        early = preferred is not None and config.preferred_first and is_legal_move(state, preferred)
        if early:
            child = self._child(state, preferred)
            outcome = self._terminal(child, deadline)
            yield (preferred, child, (outcome, None) if config.reuse_results else None)
        scored = []
        delayed = None
        for move in iter_legal_moves(state, deadline):
            self._check(deadline)
            if early and move == preferred:
                continue
            child = self._child(state, move)
            outcome = self._terminal(child, deadline)
            if move == preferred:
                delayed = (move, child, (outcome, None) if config.reuse_results else None)
                continue
            value = self.utility(child, child.player, outcome[1]) if outcome[0] else self._evaluate(child)
            prepared = (outcome, value if not outcome[0] else None) if config.reuse_results else None
            scored.append((-value, action_code(move), move, child, prepared))
            if self.diagnostics:
                self.counters.sorted_children += 1
        scored.sort(key=lambda item: (-item[0], item[1]))
        self._check(deadline)
        if delayed is not None:
            yield delayed
        for _, _, move, child, prepared in scored:
            self._check(deadline)
            yield (move, child, prepared)

    def _search(self, state, depth, alpha, beta, deadline, preferred=None, prepared=None, root=False, collect_ties=False):
        self._check(deadline)
        if self.diagnostics:
            self.counters.nodes += 1
        outcome = prepared[0] if prepared is not None else self._terminal(state, deadline)
        if outcome[0]:
            if self.diagnostics:
                self.counters.terminal_returns += 1
            return (self.utility(state, state.player, outcome[1]), None)
        if depth == 0:
            if self.diagnostics:
                self.counters.depth_returns += 1
            return (prepared[1] if prepared is not None and prepared[1] is not None else self._evaluate(state), None)
        if not state.allow_moves and must_pass(state, deadline):
            value, _ = self._search(apply_pass(state), depth - 1, -beta, -alpha, deadline)
            return (-value, None)
        alpha0, beta0 = (alpha, beta)
        key, transform = (None, 0)
        if self.config.tt:
            tick = time.perf_counter() if self.diagnostics else 0
            key, transform = search_key(state, self.config)
            if self.diagnostics:
                self.counters.key_seconds += time.perf_counter() - tick
                self.counters.tt_probes += 1
            tick = time.perf_counter() if self.diagnostics else 0
            entry = self.tt.get(key, depth)
            hint = self.tt.get_move(key)
            if self.diagnostics:
                self.counters.tt_seconds += time.perf_counter() - tick
            move = None if entry is None else entry.best_move
            if move is not None and transform:
                move = inverse_transform_move(move, transform)
            if entry is not None and is_legal_move(state, move):
                if self.diagnostics:
                    self.counters.tt_hits += 1
                if not (root and collect_ties):
                    if entry.flag == EXACT:
                        if self.diagnostics:
                            self.counters.tt_exact_returns += 1
                        return (entry.value, move)
                    if entry.flag == LOWER and entry.value >= beta or (entry.flag == UPPER and entry.value <= alpha):
                        if self.diagnostics:
                            self.counters.tt_bound_returns += 1
                        return (entry.value, move)
            if self.config.ordering and preferred is None and (hint is not None):
                if transform:
                    hint = inverse_transform_move(hint, transform)
                if is_legal_move(state, hint):
                    preferred = hint
                    if self.diagnostics:
                        self.counters.tt_hint_hits += 1
        if not self.config.ordering:
            preferred = None
        best, best_move = (-math.inf, None)
        for index, (move, child, reuse) in enumerate(self._children(state, preferred, deadline)):
            full = not self.config.alpha_beta or (root and collect_ties)
            lo, hi = (-math.inf, math.inf) if full else (-beta, -alpha)
            value, _ = self._search(child, depth - 1, lo, hi, deadline, prepared=reuse)
            value = -value
            if root and collect_ties:
                self._root_scores.append((move, value))
            if value > best:
                best, best_move = (value, move)
            if self.config.alpha_beta and (not (root and collect_ties)):
                alpha = max(alpha, best)
                if alpha >= beta:
                    if self.diagnostics:
                        self.counters.cutoffs += 1
                        self.counters.first_child_cutoffs += int(index == 0)
                        self.counters.preferred_cutoffs += int(index == 0 and move == preferred)
                    break
        self._check(deadline)
        if self.config.tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta0 else EXACT
            canonical_move = transform_move(best_move, transform) if transform else best_move
            at_capacity = len(self.tt.entries) >= self.config.tt_capacity and (key, depth) not in self.tt.entries
            tick = time.perf_counter() if self.diagnostics else 0
            self.tt.store(key, depth, best, flag, canonical_move)
            if self.diagnostics:
                self.counters.tt_seconds += time.perf_counter() - tick
                self.counters.tt_capacity_rejections += int(at_capacity)
                self.counters.tt_stores += int(not at_capacity)
        return (best, best_move)

    def statistics(self):
        if not self.diagnostics:
            return {'enabled': False}
        seen = set()

        def retained_bytes(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            result = sys.getsizeof(value)
            if isinstance(value, dict):
                result += sum((retained_bytes(k) + retained_bytes(v) for k, v in value.items()))
            elif isinstance(value, (tuple, list)):
                result += sum((retained_bytes(v) for v in value))
            elif hasattr(value, '__dict__'):
                result += retained_bytes(vars(value))
            return result
        return dict(enabled=True, **asdict(self.counters), tt_entries=len(self.tt.entries), approximate_tt_bytes=retained_bytes((self.tt.entries, self.tt.best_moves)))

    def fixed_depth(self, state, depth, deadline=None, preferred=None, collect_ties=False, clear_tt=True):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        started = time.perf_counter()
        if clear_tt:
            self.tt.clear()
            self.counters = SearchCounters()
        result = EngineResult()
        self._root_scores = []
        ties = collect_ties or not self.config.alpha_beta
        try:
            value, move = self._search(state, depth, -math.inf, math.inf, deadline, preferred=preferred, root=True, collect_ties=ties)
            self._check(deadline)
        except StateTimeout:
            if self.diagnostics:
                self.counters.interrupted_iterations += 1
            result.status = 'incomplete'
        else:
            result.value, result.move = (value, move)
            result.completed_depth = depth
            result.status = 'completed'
            if ties:
                result.optimal_moves = tuple((m for m, v in self._root_scores if v == value))
            result.solved_remaining = depth >= MAX_PLIES - state.ply
        result.elapsed = time.perf_counter() - started
        result.statistics = self.statistics()
        return result

    def search(self, state, deadline, max_depth=None, preferred=None, collect_ties=False, tie_rng=None):
        """Iterative deepening; commit only complete, on-time root iterations."""
        started = time.perf_counter()
        self.tt.clear()
        self.counters = SearchCounters()
        outcome = terminal(state)
        if outcome[0]:
            return EngineResult(value=self.utility(state, state.player, outcome[1]), status='terminal', elapsed=time.perf_counter() - started, statistics=self.statistics())
        fallback = next(iter_legal_moves(state), None)
        result = EngineResult(move=fallback)
        limit = MAX_PLIES - state.ply if max_depth is None else min(max_depth, MAX_PLIES - state.ply)
        for depth in range(1, limit + 1):
            if time.perf_counter() >= deadline:
                break
            iteration = self.fixed_depth(state, depth, deadline, preferred, collect_ties, clear_tt=False)
            if iteration.status != 'completed':
                break
            result.move, result.value = (iteration.move, iteration.value)
            result.completed_depth, result.status = (depth, 'completed')
            result.optimal_moves = iteration.optimal_moves
            result.iterations.append({'depth': depth, 'move': iteration.move, 'value': iteration.value, 'elapsed': time.perf_counter() - started})
            preferred = iteration.move if self.config.ordering else None
        if tie_rng is not None and result.optimal_moves:
            result.move = tie_rng.choice(result.optimal_moves)
        result.elapsed = time.perf_counter() - started
        result.statistics = self.statistics()
        result.solved_remaining = result.completed_depth >= MAX_PLIES - state.ply
        return result

class ExperimentAgent(Agent):

    def __init__(self, variant=None, evaluation_factory=None, book_factory=None):
        super().__init__()
        self.variant = dict(variant or {'variant_id': 'S0_E1_N', 'search': {}, 'evaluation': {'evaluation_id': 'E1'}, 'book': {'mode': 'N'}})
        self.name = self.variant['variant_id']
        budget = self.variant.get('step_budget', 1.6)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or (not math.isfinite(budget)) or (budget < 0):
            raise ValueError('step_budget must be finite and nonnegative')
        if budget > 1.6:
            raise ValueError('online experiment budget must not exceed 1.6 seconds')
        self.budget = budget
        self.call_count = 0
        self.last_info = None
        factory = evaluation_factory or default_evaluator
        self.engine = SearchEngine(SearchConfig(**self.variant.get('search', {})), evaluator=factory(self.variant.get('evaluation')), diagnostics=False)
        book_config = self.variant.get('book', {'mode': 'N'})
        self.book_hook = book_factory(book_config, self.variant) if book_factory is not None else None
        if book_config.get('mode', 'N') != 'N' and self.book_hook is None:
            raise ValueError("a non-N book mode requires B's book_factory(config, variant)")

    def _report(self, info, started):
        info['processing_before_report_s'] = time.perf_counter() - started
        self.last_info = info
        if self.variant.get('telemetry', False):
            print('EXP_STEP ' + json.dumps(info, separators=(',', ':')), flush=True)

    def step(self, board, reserves, player, opponent):
        started = time.perf_counter()
        deadline = started + max(0.0, self.budget - min(0.002, self.budget / 10))
        ply = 2 * self.call_count + player - 1
        self.call_count += 1
        moves = get_valid_moves(board, reserves, player)
        fallback = moves[0]
        info = {'variant_id': self.name, 'ply': ply, 'completed_depth': None, 'status': 'single_move', 'book_queried': False, 'book_matched': False, 'book_action': 'none'}
        if len(moves) == 1:
            self._report(info, started)
            return fallback
        state = GameState.from_observation(board, reserves, player, ply)
        preferred = None
        if self.book_hook is not None and time.perf_counter() < deadline:
            info['book_queried'] = True
            try:
                decision = self.book_hook(state, tuple(moves), deadline)
            except StateTimeout:
                decision = {}
            info.update(book_matched=bool(decision.get('matched', False)), book_version=decision.get('version'), book_metadata=decision.get('metadata', {}))
            direct = decision.get('direct_move')
            if direct is not None and is_legal_move(state, direct) and (time.perf_counter() < deadline):
                info.update(status='book', book_action='direct')
                self._report(info, started)
                return direct
            hint = decision.get('preferred_move')
            if hint is not None and is_legal_move(state, hint):
                preferred = hint
                info['book_action'] = 'hint'
        result = self.engine.search(state, deadline, self.variant.get('max_depth'), preferred)
        info.update(status=result.status, completed_depth=result.completed_depth, value=result.value, search_elapsed_s=result.elapsed, iterations=[{'depth': r['depth'], 'value': r['value'], 'elapsed': r['elapsed']} for r in result.iterations], solved_remaining=result.solved_remaining)
        self._report(info, started)
        return result.move if result.move in moves else fallback
BOOK_MODES = ('N', 'D', 'H')
BOOK_LAYOUT_CHOICES = {'symmetry': ('d4', 'expanded'), 'key': ('compact', 'tuple'), 'compression': ('zlib', 'none')}
BOOK_COMPACT_BYTES = 15
BOOK_MOVE_BYTES = 2
BOOK_MOVE_CODES = (BOARD_SIZE * BOARD_SIZE + 1) * BOARD_SIZE * BOARD_SIZE * NUM_SIZES

def book_pack_key(key):
    """Pack a full key: 2 bits per board slot, reserve and player; 7 bits for ply."""
    packed = 0
    for value in key[:-1]:
        packed = packed << 2 | value
    return (packed << 7 | key[-1]).to_bytes(BOOK_COMPACT_BYTES, 'big')

def book_unpack_key(data):
    packed = int.from_bytes(data, 'big')
    ply = packed & 127
    packed >>= 7
    values = []
    for _ in range(BOOK_KEY_SIZE - 1):
        values.append(packed & 3)
        packed >>= 2
    return tuple(reversed(values)) + (ply,)

def book_check_key(key):
    """Reject keys that the 2-bit/7-bit packing cannot represent exactly."""
    key = tuple(key)
    if len(key) != BOOK_KEY_SIZE or any((isinstance(v, bool) or not isinstance(v, int) for v in key)):
        raise ValueError('book key must contain 56 integers')
    if any((not 0 <= v <= 3 for v in key[:-1])) or not 0 <= key[-1] < 128:
        raise ValueError('book key value outside the packed range')
    return key

def book_pack_move(move):
    """Pack a move as (source + 1) * 48 + destination * 3 + size; source -1 is a placement."""
    source = -1 if move.src is None else move.src[0] * BOARD_SIZE + move.src[1]
    destination = move.dest[0] * BOARD_SIZE + move.dest[1]
    return (source + 1) * BOARD_SIZE * BOARD_SIZE * NUM_SIZES + destination * NUM_SIZES + move.size

def book_unpack_move(code):
    if not 0 <= code < BOOK_MOVE_CODES:
        raise ValueError('book move code out of range')
    source, rest = divmod(code, BOARD_SIZE * BOARD_SIZE * NUM_SIZES)
    destination, size = divmod(rest, NUM_SIZES)
    src = None if source == 0 else divmod(source - 1, BOARD_SIZE)
    return Move(src, divmod(destination, BOARD_SIZE), size)

def book_check_layout(layout):
    for name, choices in BOOK_LAYOUT_CHOICES.items():
        if layout.get(name) not in choices:
            raise ValueError(f'unsupported book layout {name}={layout.get(name)!r}')
    entries = layout.get('entries')
    if isinstance(entries, bool) or not isinstance(entries, int) or entries < 0:
        raise ValueError('book layout needs a nonnegative entry count')
    return layout

def book_data_hash(data):
    return hashlib.sha256(''.join(data).encode('ascii')).hexdigest()

def book_decode(data, layout):
    """Decode embedded chunks into {lookup key: move code}; done once at agent init."""
    book_check_layout(layout)
    raw = base64.b64decode(''.join(data).encode('ascii'), validate=True)
    if layout['compression'] == 'zlib':
        raw = zlib.decompress(raw)
    key_bytes = BOOK_COMPACT_BYTES if layout['key'] == 'compact' else BOOK_KEY_SIZE
    width = key_bytes + BOOK_MOVE_BYTES
    if len(raw) != width * layout['entries']:
        raise ValueError('book length does not match its entry count')
    table = {}
    for start in range(0, len(raw), width):
        key = raw[start:start + key_bytes]
        if layout['key'] == 'tuple':
            key = tuple(key)
        table[key] = int.from_bytes(raw[start + key_bytes:start + width], 'big')
    if len(table) != layout['entries']:
        raise ValueError('book contains duplicate keys')
    return table

def book_query_key(state, layout):
    """Return (lookup key, transform) of a state under the stored layout."""
    if layout['symmetry'] == 'd4':
        key, transform = canonicalize(state)
    else:
        key, transform = (encode_book_state(state), 0)
    return (book_pack_key(key) if layout['key'] == 'compact' else key, transform)

def book_lookup(table, layout, state):
    """Return the stored move in the state's own coordinates, or None on a miss."""
    key, transform = book_query_key(state, layout)
    code = table.get(key)
    if code is None:
        return None
    move = book_unpack_move(code)
    return inverse_transform_move(move, transform) if transform else move

def book_build_hook(data, layout, book_config, variant):
    """Return A's hook for mode D or H; mode N returns None, so nothing is queried."""
    mode = book_config.get('mode', 'N')
    if mode not in BOOK_MODES:
        raise ValueError(f'book mode must be one of {BOOK_MODES}')
    book_check_layout(layout)
    if book_data_hash(data) != layout.get('hash'):
        raise ValueError('embedded book data does not match its recorded hash')
    for name in ('hash', 'version', 'entries', 'symmetry', 'key', 'compression'):
        expected = book_config.get(name)
        if expected is not None and expected != layout.get(name):
            raise ValueError(f'book {name} mismatch: configured {expected!r}, embedded {layout.get(name)!r}')
    if mode == 'N':
        return None
    table = book_decode(data, layout)
    version = layout.get('version')

    def hook(state, legal_moves, deadline):
        started = time.perf_counter()
        if started >= deadline:
            raise StateTimeout('no time left for the book query')
        move = book_lookup(table, layout, state)
        elapsed = time.perf_counter() - started
        legal = move is not None and move in legal_moves
        decision = {'matched': legal, 'version': version, 'direct_move': None, 'preferred_move': None, 'metadata': {'query_seconds': elapsed, 'mode': mode, 'key_found': move is not None, 'invalid_entry': move is not None and (not legal)}}
        if legal:
            decision['direct_move' if mode == 'D' else 'preferred_move'] = move
        return decision
    return hook
BOOK_LAYOUT = {'canonical_entries': 3, 'compression': 'zlib', 'entries': 3, 'hash': 'd2f3434ae9c076237c52fa60dd94192902df0d86bc95aa258629848d04953c77', 'key': 'compact', 'symmetry': 'd4', 'version': '2f7398b19e9d71386b5ace0103fb1c01682fa4d0e92e2e38b9f775b103395bff'}
BOOK_DATA = ('eNpjYIAD9u/3GhhEGJAE/n5DE/hfAxQAAJu6BxU=',)

def make_book_hook(book_config, variant):
    return book_build_hook(BOOK_DATA, BOOK_LAYOUT, book_config, variant)
EXPERIMENT_VARIANT = {'book': {'entries': 3, 'hash': 'd2f3434ae9c076237c52fa60dd94192902df0d86bc95aa258629848d04953c77', 'mode': 'D', 'provider': 'agents.books.smoke_b0_k3', 'version': '2f7398b19e9d71386b5ace0103fb1c01682fa4d0e92e2e38b9f775b103395bff'}, 'book_hash': 'd2f3434ae9c076237c52fa60dd94192902df0d86bc95aa258629848d04953c77', 'config_hash': '59dd556f969676b8cfa398111221b1a061b835f269cda2ecf068d3632f8f2b3e', 'evaluation': {'evaluation_id': 'E1'}, 'max_depth': 2, 'parent_variant': 'S0_E1_N', 'search': {'alpha_beta': True, 'key_mode': 'tuple', 'ordering': True, 'preferred_first': False, 'reuse_results': False, 'symmetry': False, 'tt': True, 'tt_capacity': 50000}, 'step_budget': 0.012, 'telemetry': True, 'variant_id': 'S0_E1_N__smoke_b0_k3_D'}

@register_agent('student_agent')
class StudentAgent(ExperimentAgent):

    def __init__(self):
        super().__init__(EXPERIMENT_VARIANT, default_evaluator, make_book_hook)
