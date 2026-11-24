"""Export screened book entries into agents/student_agent.py.

Accepted entries are sorted by decreasing occurrence count, then decreasing
completed depth, then increasing canonical key, and retained while the
submission stays within its size budget. The embedded book is then decoded
and checked against agents/symmetry.py from every symmetric orientation.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib

from helpers import Move
from agents import student_agent as agent
from agents.game_state import legal_moves
from agents.symmetry import (canonicalize, decode_book_state, inverse_transform_move,
                             transform_state)
import screen_positions as screening


SUBMISSION = Path("agents/student_agent.py")
SIZE_LIMIT = 2_000_000
SIZE_MARGIN = 50_000
LINE_WIDTH = 76
BLOCK = re.compile(r"# BOOK-DATA-BEGIN\n.*?# BOOK-DATA-END\n", re.DOTALL)


def accepted_entries(screened):
    """Return (key, move, count, depth) for accepted, version-matching records."""
    summary = screened.get("screening")
    if not isinstance(summary, dict) or summary.get("version") != screening.SCREENING_VERSION:
        raise ValueError("input is not a screening file of the current version")
    if not screening._versions_match(summary.get("versions")):
        raise ValueError("screening versions do not match the current rules and evaluation")

    entries = []
    for record in screened["positions"]:
        if not record.get("screening", {}).get("accepted"):
            continue
        key = tuple(record["key"])
        move = screening._read_move(record["move"])
        state = decode_book_state(key)
        if canonicalize(state)[0] != key or move not in legal_moves(state):
            raise ValueError(f"accepted entry is not canonical or not legal: {key}")
        entries.append((key, move, record["count"], record["completed_depth"]))
    entries.sort(key=lambda item: (-item[2], -item[3], item[0]))
    return entries


def pack_entries(entries):
    data = b"".join(agent.book_pack_key(key) + agent.book_pack_move(move).to_bytes(2, "big")
                    for key, move, _, _ in entries)
    return base64.b64encode(zlib.compress(data, 9)).decode("ascii")


def render_block(blob, version):
    lines = ["# BOOK-DATA-BEGIN", f'BOOK_VERSION = "{version}"']
    if blob:
        lines.append("BOOK_DATA = (")
        for start in range(0, len(blob), LINE_WIDTH):
            lines.append(f'    "{blob[start:start + LINE_WIDTH]}"')
        lines.append(")")
    else:
        lines.append('BOOK_DATA = ""')
    lines.append("# BOOK-DATA-END")
    return "\n".join(lines) + "\n"


def build_source(source, entries):
    blob = pack_entries(entries)
    version = hashlib.sha256(blob.encode("ascii")).hexdigest()[:12] if blob else "empty"
    return BLOCK.sub(lambda _: render_block(blob, version), source, count=1), version


def fit_entries(source, entries, budget):
    """Return the largest prefix of entries whose submission fits the budget."""
    low, high = 0, len(entries)
    while low < high:
        middle = (low + high + 1) // 2
        text, _ = build_source(source, entries[:middle])
        if len(text.encode("utf-8")) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def verify(entries, blob_source):
    """Decode the written book and check every entry from all eight orientations."""
    namespace = {}
    match = BLOCK.search(blob_source)
    exec(compile(match.group(0), "<book>", "exec"), namespace)
    book = agent.book_decode(namespace["BOOK_DATA"])
    if len(book) != len(entries):
        raise AssertionError(f"decoded {len(book)} entries, expected {len(entries)}")

    for key, move, _, _ in entries:
        canonical_state = decode_book_state(key)
        for transform in range(8):
            state = transform_state(canonical_state, transform)
            reference_key, reference_transform = canonicalize(state)
            if agent.book_canonicalize(state) != (reference_key, reference_transform):
                raise AssertionError("agent canonical key differs from agents/symmetry.py")
            expected = inverse_transform_move(move, reference_transform)
            found = agent.book_lookup(book, state, legal_moves(state))
            if found != expected:
                raise AssertionError(f"lookup mismatch: {found} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="screened_positions.json")
    parser.add_argument("--submission", default=str(SUBMISSION))
    parser.add_argument("--max-entries", type=int)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        entries = accepted_entries(json.load(handle))
    if args.max_entries is not None:
        entries = entries[:args.max_entries]

    path = Path(args.submission)
    source = path.read_text(encoding="utf-8")
    if not BLOCK.search(source):
        parser.error("submission has no BOOK-DATA-BEGIN/END block")

    kept = fit_entries(source, entries, SIZE_LIMIT - SIZE_MARGIN)
    entries = entries[:kept]
    text, version = build_source(source, entries)
    verify(entries, text)
    path.write_text(text, encoding="utf-8")
    print(f"Embedded {len(entries)} entries (book {version}); "
          f"{path} is {len(text.encode('utf-8'))} bytes")


if __name__ == "__main__":
    main()
