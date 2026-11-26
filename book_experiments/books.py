"""Book files, layouts and generated provider modules shared by steps 5 and 6.

A book file (JSON) keeps the canonical entries in export order together with
the exact embedded data, so the provider module can be re-created byte for
byte on another machine. Provider modules are written to agents/books/, which
A's builder inlines when a variant names them in book.provider.
"""

import ast
import base64
from pathlib import Path
import re
import zlib

from agents.book_policy import (BOOK_LAYOUT_CHOICES, book_check_key, book_data_hash, book_decode,
                                book_lookup, book_pack_key, book_pack_move, book_unpack_key,
                                book_unpack_move)
from agents.symmetry import (TRANSFORMS, canonicalize, decode_book_state, encode_book_state,
                             inverse_transform_move, transform_state)
from search_experiments.common import REPO_ROOT, atomic_json, digest, load_json, restore_move

BOOK_PACKAGE = "agents.books"
CHUNK = 76
DEFAULT_LAYOUT = {"symmetry": "d4", "key": "compact", "compression": "zlib"}
PROVIDER_TEMPLATE = '''"""Generated position book {label}; do not edit by hand."""

from agents.book_policy import book_build_hook

BOOK_LAYOUT = {layout}
BOOK_DATA = (
{chunks}
)


def make_book_hook(book_config, variant):
    return book_build_hook(BOOK_DATA, BOOK_LAYOUT, book_config, variant)
'''


def move_record(move):
    return {"src": None if move.src is None else list(move.src), "dest": list(move.dest), "size": move.size}


def content_version(entries):
    """Identity of the canonical (key -> move) pairs, independent of the layout."""
    return digest(sorted([list(key), book_pack_move(move)] for key, move in entries))


def expand_entries(entries):
    """Every orientation under its raw key.

    Each actual state is canonicalized again with the same function used
    online, and its move is the inverse transform of the canonical move. A
    self-symmetric state therefore gets exactly the move a D4 lookup returns.
    """
    expanded = {}
    for key, move in entries:
        state = decode_book_state(key)
        for transform in TRANSFORMS:
            actual = transform_state(state, transform)
            canonical, chosen = canonicalize(actual)
            if canonical != tuple(key):
                raise ValueError("canonical key changed under a symmetry")
            actual_move = inverse_transform_move(move, chosen) if chosen else move
            if expanded.setdefault(encode_book_state(actual), actual_move) != actual_move:
                raise ValueError("one actual state received two different moves")
    return sorted(expanded.items())


def encode_layout(entries, symmetry="d4", key="compact", compression="zlib"):
    """Return (data chunks, layout) for canonical entries in one representation."""
    layout = {"symmetry": symmetry, "key": key, "compression": compression}
    for name, choices in BOOK_LAYOUT_CHOICES.items():
        if layout[name] not in choices:
            raise ValueError(f"unsupported {name}: {layout[name]!r}")
    entries = [(book_check_key(k), m) for k, m in entries]
    if len({k for k, _ in entries}) != len(entries):
        raise ValueError("duplicate canonical keys")
    for k, _ in entries:
        if canonicalize(decode_book_state(k))[0] != k:
            raise ValueError("book entry key is not canonical")
    stored = expand_entries(entries) if symmetry == "expanded" else sorted(entries)
    blob = bytearray()
    for stored_key, move in stored:
        blob += book_pack_key(stored_key) if key == "compact" else bytes(stored_key)
        blob += book_pack_move(move).to_bytes(2, "big")
    raw = zlib.compress(bytes(blob), 9) if compression == "zlib" else bytes(blob)
    text = base64.b64encode(raw).decode("ascii")
    data = tuple(text[i:i + CHUNK] for i in range(0, len(text), CHUNK)) or ("",)
    layout.update(entries=len(stored), canonical_entries=len(entries),
                  hash=book_data_hash(data), version=content_version(entries))
    return data, layout


def check_label(label):
    if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", label):
        raise ValueError(f"book label must be a lower-case identifier: {label!r}")
    return label


def make_book(entries, label, layout=None, details=None):
    """Book record for canonical entries in export order; details maps key -> provenance."""
    check_label(label)
    data, layout = encode_layout(entries, **(layout or DEFAULT_LAYOUT))
    details = details or {}
    return {"label": label, "provider": f"{BOOK_PACKAGE}.{label}", "layout": layout, "data": list(data),
            "entries": [dict(details.get(tuple(k), {}), key=list(k), move=move_record(m)) for k, m in entries]}


def book_entries(record):
    return [(tuple(row["key"]), restore_move(row["move"])) for row in record["entries"]]


def book_table(record):
    """In-process lookup table built from the recorded data, never from an imported module."""
    return book_decode(tuple(record["data"]), record["layout"]), record["layout"]


def lookup(table, state):
    return book_lookup(table[0], table[1], state)


def render_provider(record):
    chunks = "\n".join(f"    {chunk!r}," for chunk in record["data"])
    layout = repr(dict(sorted(record["layout"].items())))
    return PROVIDER_TEMPLATE.format(label=record["label"], layout=layout, chunks=chunks)


def write_provider(record, root=REPO_ROOT):
    """Write agents/books/<label>.py after checking the data against its hash."""
    check_label(record["label"])
    if book_data_hash(tuple(record["data"])) != record["layout"]["hash"]:
        raise ValueError("book data does not match its recorded hash")
    folder = Path(root) / "agents" / "books"
    folder.mkdir(parents=True, exist_ok=True)
    init = folder / "__init__.py"
    if not init.exists():
        init.write_text('"""Generated position books; local files, not committed."""\n', encoding="utf-8")
    path = folder / f"{record['label']}.py"
    text = render_provider(record)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return {"provider": record["provider"], "path": str(path), "module_bytes": len(text.encode())}


def save_book(record, path, root=REPO_ROOT):
    atomic_json(path, record)
    return write_provider(record, root)


def load_book(path, root=REPO_ROOT):
    record = load_json(path)
    write_provider(record, root)
    return record


def subset_book(record, size, label):
    """Prefix of the export order, re-encoded as its own book."""
    entries = book_entries(record)[:size]
    details = {tuple(row["key"]): {k: v for k, v in row.items() if k not in ("key", "move")}
               for row in record["entries"][:size]}
    layout = {name: record["layout"][name] for name in DEFAULT_LAYOUT}
    return make_book(entries, label, layout, details)


def book_variant(online, record, mode):
    """The online variant with this book in mode D or H; same inputs, same variant."""
    if mode not in ("D", "H"):
        raise ValueError("book variants use mode D or H; N is the online variant itself")
    variant = {key: value for key, value in online.items() if key not in ("config_hash", "book_hash")}
    variant["variant_id"] = f"{online['variant_id']}__{record['label']}_{mode}"
    variant["parent_variant"] = online["variant_id"]
    variant["book"] = {"mode": mode, "provider": record["provider"], "hash": record["layout"]["hash"],
                       "version": record["layout"]["version"], "entries": record["layout"]["entries"]}
    variant["telemetry"] = True
    return variant


def load_current_book(path=REPO_ROOT / "agents" / "student_agent.py"):
    """Entries of the book embedded in the original submission (15-byte key + 2-byte move, zlib)."""
    values = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("BOOK_DATA", "BOOK_VERSION")):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    blob = values.get("BOOK_DATA", "")
    blob = "".join(blob) if isinstance(blob, tuple) else blob
    raw = zlib.decompress(base64.b64decode(blob)) if blob else b""
    if len(raw) % 17:
        raise ValueError("current book has a truncated entry")
    entries = [(book_unpack_key(raw[i:i + 15]), book_unpack_move(int.from_bytes(raw[i + 15:i + 17], "big")))
               for i in range(0, len(raw), 17)]
    return entries, values.get("BOOK_VERSION")
