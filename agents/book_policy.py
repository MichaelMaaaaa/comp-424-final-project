"""Online position book: decoding, lookup and the N/D/H book hook.

Generated provider modules (agents/books/*.py) hold the book data as module
constants and define make_book_hook(book_config, variant), which calls
book_build_hook() below. The offline tools in book_experiments use the same
packing and lookup functions, so the stored and queried encodings agree.

Layouts (5A): symmetry "d4" stores one canonical entry per symmetry class and
maps the move back with the inverse of the query's transform; "expanded"
stores every orientation under its raw key. Keys are "compact" (15 bytes) or
"tuple" (56 integers). Data is "zlib" compressed or stored as is ("none").
"""

import base64
import hashlib
import time
import zlib

from constants import BOARD_SIZE, NUM_SIZES
from helpers import Move
from agents.game_state import StateTimeout
from agents.symmetry import BOOK_KEY_SIZE, canonicalize, encode_book_state, inverse_transform_move

BOOK_MODES = ("N", "D", "H")
BOOK_LAYOUT_CHOICES = {"symmetry": ("d4", "expanded"), "key": ("compact", "tuple"),
                       "compression": ("zlib", "none")}
BOOK_COMPACT_BYTES = 15
BOOK_MOVE_BYTES = 2
BOOK_MOVE_CODES = (BOARD_SIZE * BOARD_SIZE + 1) * BOARD_SIZE * BOARD_SIZE * NUM_SIZES


def book_pack_key(key):
    """Pack a full key: 2 bits per board slot, reserve and player; 7 bits for ply."""
    packed = 0
    for value in key[:-1]:
        packed = (packed << 2) | value
    return ((packed << 7) | key[-1]).to_bytes(BOOK_COMPACT_BYTES, "big")


def book_unpack_key(data):
    packed = int.from_bytes(data, "big")
    ply = packed & 0x7F
    packed >>= 7
    values = []
    for _ in range(BOOK_KEY_SIZE - 1):
        values.append(packed & 3)
        packed >>= 2
    return tuple(reversed(values)) + (ply,)


def book_check_key(key):
    """Reject keys that the 2-bit/7-bit packing cannot represent exactly."""
    key = tuple(key)
    if len(key) != BOOK_KEY_SIZE or any(isinstance(v, bool) or not isinstance(v, int) for v in key):
        raise ValueError("book key must contain 56 integers")
    if any(not 0 <= v <= 3 for v in key[:-1]) or not 0 <= key[-1] < 128:
        raise ValueError("book key value outside the packed range")
    return key


def book_pack_move(move):
    """Pack a move as (source + 1) * 48 + destination * 3 + size; source -1 is a placement."""
    source = -1 if move.src is None else move.src[0] * BOARD_SIZE + move.src[1]
    destination = move.dest[0] * BOARD_SIZE + move.dest[1]
    return (source + 1) * BOARD_SIZE * BOARD_SIZE * NUM_SIZES + destination * NUM_SIZES + move.size


def book_unpack_move(code):
    if not 0 <= code < BOOK_MOVE_CODES:
        raise ValueError("book move code out of range")
    source, rest = divmod(code, BOARD_SIZE * BOARD_SIZE * NUM_SIZES)
    destination, size = divmod(rest, NUM_SIZES)
    src = None if source == 0 else divmod(source - 1, BOARD_SIZE)
    return Move(src, divmod(destination, BOARD_SIZE), size)


def book_check_layout(layout):
    for name, choices in BOOK_LAYOUT_CHOICES.items():
        if layout.get(name) not in choices:
            raise ValueError(f"unsupported book layout {name}={layout.get(name)!r}")
    entries = layout.get("entries")
    if isinstance(entries, bool) or not isinstance(entries, int) or entries < 0:
        raise ValueError("book layout needs a nonnegative entry count")
    return layout


def book_data_hash(data):
    return hashlib.sha256("".join(data).encode("ascii")).hexdigest()


def book_decode(data, layout):
    """Decode embedded chunks into {lookup key: move code}; done once at agent init."""
    book_check_layout(layout)
    raw = base64.b64decode("".join(data).encode("ascii"), validate=True)
    if layout["compression"] == "zlib":
        raw = zlib.decompress(raw)
    key_bytes = BOOK_COMPACT_BYTES if layout["key"] == "compact" else BOOK_KEY_SIZE
    width = key_bytes + BOOK_MOVE_BYTES
    if len(raw) != width * layout["entries"]:
        raise ValueError("book length does not match its entry count")
    table = {}
    for start in range(0, len(raw), width):
        key = raw[start:start + key_bytes]
        if layout["key"] == "tuple":
            key = tuple(key)
        table[key] = int.from_bytes(raw[start + key_bytes:start + width], "big")
    if len(table) != layout["entries"]:
        raise ValueError("book contains duplicate keys")
    return table


def book_query_key(state, layout):
    """Return (lookup key, transform) of a state under the stored layout."""
    if layout["symmetry"] == "d4":
        key, transform = canonicalize(state)
    else:
        key, transform = encode_book_state(state), 0
    return (book_pack_key(key) if layout["key"] == "compact" else key), transform


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
    mode = book_config.get("mode", "N")
    if mode not in BOOK_MODES:
        raise ValueError(f"book mode must be one of {BOOK_MODES}")
    book_check_layout(layout)
    if book_data_hash(data) != layout.get("hash"):
        raise ValueError("embedded book data does not match its recorded hash")
    for name in ("hash", "version", "entries", "symmetry", "key", "compression"):
        expected = book_config.get(name)
        if expected is not None and expected != layout.get(name):
            raise ValueError(f"book {name} mismatch: configured {expected!r}, embedded {layout.get(name)!r}")
    if mode == "N":
        return None
    table = book_decode(data, layout)
    version = layout.get("version")

    def hook(state, legal_moves, deadline):
        started = time.perf_counter()
        if started >= deadline:
            raise StateTimeout("no time left for the book query")
        move = book_lookup(table, layout, state)
        elapsed = time.perf_counter() - started
        legal = move is not None and move in legal_moves
        decision = {"matched": legal, "version": version, "direct_move": None, "preferred_move": None,
                    "metadata": {"query_seconds": elapsed, "mode": mode, "key_found": move is not None,
                                 "invalid_entry": move is not None and not legal}}
        if legal:
            decision["direct_move" if mode == "D" else "preferred_move"] = move
        return decision

    return hook
