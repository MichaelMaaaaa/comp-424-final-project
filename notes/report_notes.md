# Report notes: position book and baseline experiments

Draft material for the report (Agent design §2, Quantitative analysis §3,
Strengths and limitations §4). Numbers marked TODO are filled in once the
runs finish. Keep measured results and predictions separate.

## 1. Position book: lookup and export (Agent design)

**Purpose.** The book replaces online search in positions that were analysed
offline with a longer budget (10 s instead of 1.6 s). A hit returns in well
under 0.2 s; a miss falls through to iterative-deepening alpha-beta with the
same deadline, so the book can only save time, never cost it.

**Canonical keys.** A position and its rotations/reflections have the same
game value, so the book stores one entry per symmetry class. For each of the
8 transforms of the square (dihedral group D4: 4 rotations, 4 reflections),
the board stacks and obstacle are moved and the full state is encoded as a
56-value tuple: 48 board slots (16 cells × 3 sizes, covered pieces included),
both reserves (2 × 3), player to move, and ply. The lexicographically smallest
encoding is the canonical key; ties go to the first transform in a fixed
order. Reserves, sizes, player and ply are unaffected by the transform.
This matches agents/symmetry.py exactly, which the offline pipeline used.

**Why ply is in the key.** The terminal utility uses absolute ply (earlier
wins score higher) and the 100-ply cap makes the same board at different
plies genuinely different positions, so keys match only at the same ply.

**Storage.** Each slot, reserve count and player fits in 2 bits and ply in
7 bits: 55 × 2 + 7 = 117 bits → 15 bytes per key. A move is packed as
(source + 1) × 48 + destination × 3 + size (source = −1 for a reserve
placement), which is below 816 → 2 bytes. Entries (17 bytes) are
concatenated, zlib-compressed, base64-encoded and embedded in
student_agent.py, since the submission must be one file under 2 MB with no
file access. The book is decoded once per process at agent construction.

**Lookup.** At each step: canonicalize the current state (8 encodings),
look up the packed key, map the stored canonical move back with the inverse
of the query's transform, and return it only if it is in the current legal
move list. Any miss or invalid entry falls through to search.

**Export (export_book.py).** Takes entries accepted by screening, sorts them
by decreasing occurrence count, then decreasing completed depth, then
increasing key (as in the design), and keeps the longest prefix whose
submission file stays under 1.95 MB (binary search on the prefix length).
It then decodes the written book and checks every entry from all 8
orientations against agents/symmetry.py: same canonical key and transform,
and the lookup returns the expected legal move.

**Book size.** 48 self-play games gave 4550 unique canonical positions;
the first 600 (plies 0-14) were analysed, 486 accepted and embedded.

## 2. Experiment setup (Quantitative analysis)

- Harness: `simulator.py --autoplay --strict`, i.e. each agent in its own
  process under the tournament time and memory limits. Games are played in
  pairs with the same start configuration and seats swapped, so first-move
  advantage cancels within each pair.
- Opponents: `random_agent`, `greedy_agent` (one-ply: wins if possible,
  avoids immediate losses), and our local alpha-beta baseline
  (baselines/ab_baseline_agent.py, below).
- Hardware: Intel Core i5-3230M (2 cores, 2.6 GHz), Python 3.11.18,
  NumPy 1.26.4. Timed agents reach shallower depths on slower machines, so
  results from different machines should be reported separately. Some runs
  shared the CPU with the offline book pipeline (TODO: mark which).
- Summaries: `python summarize_results.py experiments/*.jsonl` reports
  W/D/L, score (draw = 0.5), per-seat results, results by obstacle class
  (corner / edge / center) and removed size, step times, strikes, forfeits
  and peak memory. Search depth is not in the game records.

**Local alpha-beta baseline.** The course's "simple alpha-beta baseline with
heuristic evaluation" is not published, so we wrote a stand-in that shares
no code with our agent: minimax with alpha-beta pruning, iterative deepening
to at most 3 plies within 1.5 s, framework move order, no transposition
table. Evaluation counts open lines: a line containing top pieces of only one
player scores 1/10/100/1000 for 1/2/3/4 pieces; plus a small size term
(1/2/4). Using the framework's NumPy helpers it reaches depth 2 in about
0.65 s at the opening; depth 3 takes about 18 s, so it effectively plays at
depth 2. Sanity check: it beat greedy_agent 6/0/0 (3/3 per seat).

## 3. Results so far (measured)

Agent: alpha-beta from commit cc417fd, empty book.

| Opponent | Games | W/D/L | As P1 | As P2 | Mean plies | Max step | Strikes / forfeits |
|---|---|---|---|---|---|---|---|
| random_agent | 10 | 10/0/0 | 5/0/0 | 5/0/0 | 11.7 | 1.60 s | 0 / 0 |
| greedy_agent | 10 | 10/0/0 | 5/0/0 | 5/0/0 | 25.9 | 1.60 s | 0 / 0 |
| ab_baseline (local) | 20 | 4/16/0 | 2/8/0 | 2/8/0 | 86.5 | 1.61 s | 0 / 0 |

Peak memory 37–38 MB (limit 500 MB). Every step used the full 1.60 s
budget, leaving 0.4 s of the 2 s limit as margin. The ab_baseline run
shared the CPU with the offline book collection.

**Main finding: draws against alpha-beta.** Against the local baseline the
agent never lost but drew 16 of 20 games, all at the 100-ply cap (4 wins
ended by a line). By obstacle class: corner 0/8/0, edge 1/1/0, center
3/7/0; by removed size: S 2/6/0, M 0/4/0, L 2/6/0 (small samples).
Beating the course baseline is required for the 60% performance level, so
converting draws into wins is the priority. Hypothesis, not yet checked:
the evaluation (line degree + piece size) has no term for threats such as
three own top pieces in an open line, so neither side forces progress.

**Why the draws happen (replay of the 16 drawn games).** Both players use
up their reserves by ply 19-34 (two games: ply 97). From then on every move
is a board move (672 each, against 128 placements each), and the players
oscillate: one position recurs 11-19 times per game, typically a large
piece moved back and forth between two squares by each side. The rules have
no repetition draw, so these games run to the 100-ply cap. Neither agent
detects repetition, and with deterministic tie-breaking both keep choosing
the same moves. Candidate fixes: score a return to an earlier position of
the same game as a draw (0) in search, so the side that is ahead avoids it;
evaluation terms for three-in-a-line threats and uncoverable large pieces.

### Book vs no book (controlled)

Same opponent, same 10 seeds (428984-428993, both seats), --strict.

| Agent | W/D/L | As P1 | As P2 | Mean plies | Book hits |
|---|---|---|---|---|---|
| no book (cc417fd) | 4/16/0 | 2/8/0 | 2/8/0 | 86.5 | - |
| book, 486 entries (8bd710c) | 3/17/0 | 1/9/0 | 2/8/0 | 88.7 | 18 of 887 turns |

Book hits by ply: ply 0: 10/10, ply 1: 4/10, ply 2: 2/10, ply 3: 2/10,
ply 4 onward: 0. The difference of one win is within noise for 20 games.
Measured conclusion: the book is used only in the first two moves per side
and does not change results against this opponent; it does not address the
draws. Book build: 600 positions (plies 0-14) analysed at 10 s, 486
accepted, 114 rejected (all "unstable_move"); submission 31.9 KB.

## 4. Limitations (Strengths and limitations)

- **Coverage.** Book positions come only from our own self-play, one game
  per start configuration. Opponents that deviate early leave the book after
  one or two moves, so most hits are expected in the opening. Predicted,
  not yet measured; the hit rate per ply will settle it.
- **Quality.** Entries are screened (completed depth ≥ 3, same move at the
  last two depths, no immediate loss or immediate winning reply), which is a
  quality filter, not a proof of optimal play.
- **Staleness.** The book reflects the evaluation and rules at build time;
  any change to evaluation weights requires rebuilding it.
- **Baseline uncertainty.** Our stand-in baseline may be weaker or stronger
  than the course's; results against it are an estimate of the 60% level,
  not a guarantee.
