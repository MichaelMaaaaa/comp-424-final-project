"""Complete position-book keys and square symmetries for the course rules."""

from constants import BOARD_SIZE, NUM_SIZES
from helpers import Move
from agents.game_state import GameState


# 0-3: clockwise rotations; 4-7: left-right reflection, then rotation.
TRANSFORMS = (0, 1, 2, 3, 4, 5, 6, 7)
INVERSE_TRANSFORMS = (0, 3, 2, 1, 4, 5, 6, 7)
BOARD_SLOTS = BOARD_SIZE * BOARD_SIZE * NUM_SIZES
BOOK_KEY_SIZE = BOARD_SLOTS + 2 * NUM_SIZES + 2


def _check_transform_id(transform):
    if (isinstance(transform, bool) or not isinstance(transform, int)
            or transform not in TRANSFORMS):
        raise ValueError("transform must be an integer from 0 to 7")


def encode_book_state(state):
    """Encode all board slots, both reserves, player, and ply in that order."""
    if not state.allow_moves or state.win_length != 4:
        raise ValueError("position-book keys require the official True/4 rules")

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
        raise ValueError("position-book key has the wrong length")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in key):
        raise ValueError("position-book key must contain integers")

    key = tuple(key)
    stacks = [key[i:i + NUM_SIZES] for i in range(0, BOARD_SLOTS, NUM_SIZES)]
    board = [stacks[i:i + BOARD_SIZE] for i in range(0, len(stacks), BOARD_SIZE)]
    reserve_start = BOARD_SLOTS
    reserves = [(0,) * NUM_SIZES,
                key[reserve_start:reserve_start + NUM_SIZES],
                key[reserve_start + NUM_SIZES:reserve_start + 2 * NUM_SIZES]]
    state = GameState.from_observation(board, reserves, key[-2], key[-1])
    return GameState(state.board, state.reserves, state.player, state.obstacle,
                     state.ply, allow_moves=True, win_length=4)


def transform_cell(cell, transform):
    """Map an original cell to its coordinates after the selected transform."""
    _check_transform_id(transform)
    row, col = cell
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (row, col)):
        raise ValueError("cell coordinates must be integers")
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("cell is outside the board")

    if transform >= 4:
        col = BOARD_SIZE - 1 - col
    for _ in range(transform % 4):
        row, col = col, BOARD_SIZE - 1 - row
    return row, col


def transform_state(state, transform):
    """Move whole stacks and the obstacle; preserve all other state fields."""
    board = [None] * (BOARD_SIZE * BOARD_SIZE)
    for index, stack in enumerate(state.board):
        row, col = transform_cell(divmod(index, BOARD_SIZE), transform)
        board[row * BOARD_SIZE + col] = stack
    obstacle = transform_cell(state.obstacle, transform)
    return GameState(tuple(board), state.reserves, state.player, obstacle,
                     state.ply, state.allow_moves, state.win_length)


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
    return best_key, best_transform
