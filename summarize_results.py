"""Summarize simulator --json game records in the team's shared results format.

Usage:
    python summarize_results.py experiments/*.jsonl
    python summarize_results.py experiments/a.jsonl --agent student_agent --csv out.csv

For each input file, the tracked agent's results are reported overall, by
seat, by obstacle class (corner / edge / center) and by removed piece size.
Score counts a win as 1 and a draw as 0.5. Step times cover the tracked
agent's moves only. Search depth is not in the game records and must be
measured separately.
"""

import argparse
import csv
import json
from pathlib import Path


SIZE_NAMES = ("S", "M", "L")


def obstacle_class(cell):
    row, col = cell
    edge_row, edge_col = row in (0, 3), col in (0, 3)
    if edge_row and edge_col:
        return "corner"
    if edge_row or edge_col:
        return "edge"
    return "center"


def tracked_seat(record, agent):
    """Return the seat (1 or 2) the tracked agent played, or None."""
    seats = [int(seat) for seat, info in record["players"].items()
             if Path(info["agent"]).stem == agent or info["agent"] == agent]
    return seats[0] if len(seats) == 1 else None


def opponent_name(record, seat):
    return Path(record["players"][str(3 - seat)]["agent"]).stem


class Tally:
    def __init__(self):
        self.wins = self.draws = self.losses = 0

    def add(self, outcome):
        if outcome == "W":
            self.wins += 1
        elif outcome == "D":
            self.draws += 1
        else:
            self.losses += 1

    @property
    def games(self):
        return self.wins + self.draws + self.losses

    def text(self):
        if not self.games:
            return "-"
        score = (self.wins + 0.5 * self.draws) / self.games
        return f"{self.wins}/{self.draws}/{self.losses} ({score:.0%})"


def summarize(path, agent):
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    overall, by_seat = Tally(), {1: Tally(), 2: Tally()}
    by_obstacle, by_removed = {}, {}
    opponents, plies, times, reasons = set(), [], [], {}
    strikes = forfeits = 0
    memory = []

    for record in records:
        seat = tracked_seat(record, agent)
        if seat is None:
            continue
        opponents.add(opponent_name(record, seat))
        winner = record["winner"]
        outcome = "D" if winner == 0 else ("W" if winner == seat else "L")
        overall.add(outcome)
        by_seat[seat].add(outcome)
        by_obstacle.setdefault(obstacle_class(record["obstacle"]), Tally()).add(outcome)
        by_removed.setdefault(SIZE_NAMES[record["removed_size"]], Tally()).add(outcome)
        plies.append(record["plies"])
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
        strikes += record["strikes"].get(str(seat), 0)
        if outcome == "L" and record["reason"] in ("hard_timeout", "crash", "too_many_strikes"):
            forfeits += 1
        rss = record.get("max_rss_mb", {}).get(str(seat))
        if rss is not None:
            memory.append(rss)
        times.extend(move["agent_time"] for move in record["moves"]
                     if move.get("player") == seat and "agent_time" in move)

    return {
        "file": Path(path).name,
        "opponent": ", ".join(sorted(opponents)) or "-",
        "games": overall.games,
        "W/D/L (score)": overall.text(),
        "as P1": by_seat[1].text(),
        "as P2": by_seat[2].text(),
        **{f"obstacle {name}": by_obstacle[name].text()
           for name in ("corner", "edge", "center") if name in by_obstacle},
        **{f"removed {name}": by_removed[name].text()
           for name in SIZE_NAMES if name in by_removed},
        "plies mean": f"{sum(plies) / len(plies):.1f}" if plies else "-",
        "step mean (s)": f"{sum(times) / len(times):.2f}" if times else "-",
        "step max (s)": f"{max(times):.2f}" if times else "-",
        "strikes": strikes,
        "forfeits": forfeits,
        "peak MB": f"{max(memory):.0f}" if memory else "-",
        "end reasons": ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())),
    }


def print_table(rows):
    """Print one column per file, one row per measure."""
    names = []
    for row in rows:
        names.extend(key for key in row if key not in names)
    width = max(len(name) for name in names)
    for name in names:
        cells = " | ".join(str(row.get(name, "-")) for row in rows)
        print(f"{name:<{width}} | {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--agent", default="student_agent",
                        help="registry name or file stem of the agent to track")
    parser.add_argument("--csv", help="also write the summary to this CSV file")
    args = parser.parse_args()

    rows = [summarize(path, args.agent) for path in args.files]
    print_table(rows)
    if args.csv:
        fields = []
        for row in rows:
            fields.extend(key for key in row if key not in fields)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="-")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
