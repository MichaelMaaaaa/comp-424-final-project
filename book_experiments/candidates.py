"""5B.1-5B.2 candidate pools and the common candidate count Nc.

A candidate is a complete non-terminal pre-move state, merged by its D4
canonical key. Frequency counts its occurrences in the whole pool. Every pool
contains the nine initial canonical states (3 obstacle classes x 3 removed
sizes), which count toward Nc and are screened like any other candidate.
"""

from collections import Counter

from agents.game_state import initial_state, terminal
from agents.symmetry import canonicalize
from search_experiments.common import restore_state, stage

STAGES = ("early", "middle", "late")


def initial_keys():
    keys = {canonicalize(initial_state(obstacle, removed))[0]
            for obstacle in ((0, 0), (0, 1), (1, 1)) for removed in range(3)}
    if len(keys) != 9:
        raise AssertionError("expected nine initial symmetry classes")
    return sorted(keys)


def build_pool(games, pool):
    """Merge all recorded pre-move states of the games assigned to this pool."""
    counts = Counter()
    for game in games:
        if pool not in game["pools"] or game["status"] != "completed":
            continue
        for turn in game["turns"]:
            state = restore_state(turn["state"])
            if terminal(state)[0]:
                raise ValueError("a recorded pre-move state is terminal")
            counts[canonicalize(state)[0]] += 1
    initial = set(initial_keys())
    for key in initial:
        counts.setdefault(key, 0)
    return [{"key": list(key), "ply": key[-1], "frequency": counts[key], "stage": stage(key[-1]),
             "initial": key in initial} for key in sorted(counts)]


def by_ply(pool):
    """B0/B1 order: ply ascending, then frequency descending, then key ascending."""
    return sorted(pool, key=lambda c: (c["ply"], -c["frequency"], c["key"]))


def common_count(requested, pools):
    """Analyse the same number of candidates in every scheme."""
    available = min(len(pool) for pool in pools)
    nc = min(requested, available)
    if nc < 9:
        raise ValueError("Nc must leave room for the nine initial states")
    return nc


def stage_quotas(nc):
    base, extra = divmod(nc, 3)
    return {name: base + (1 if index < extra else 0) for index, name in enumerate(STAGES)}


def select_by_stage(pool, nc):
    """B2: equal early/middle/late quotas; within a stage frequency, then ply, then key.

    The nine initial states take early places first. If the early quota is
    smaller than nine, the initial states are kept and the rest is filled in
    early -> middle -> late rotation. A stage short of its quota is topped up
    from the other stages' unselected candidates in the same rotation.
    """
    order = lambda c: (-c["frequency"], c["ply"], c["key"])
    initial = sorted((c for c in pool if c["initial"]), key=lambda c: c["key"])
    queues = {name: sorted((c for c in pool if c["stage"] == name and not c["initial"]), key=order)
              for name in STAGES}
    quotas = stage_quotas(nc)
    selected = list(initial[:nc])
    if quotas["early"] >= len(initial):
        for name in STAGES:
            want = quotas[name] - (len(initial) if name == "early" else 0)
            selected.extend(queues[name][:want])
            queues[name] = queues[name][want:]
    while len(selected) < nc and any(queues.values()):
        for name in STAGES:
            if queues[name] and len(selected) < nc:
                selected.append(queues[name].pop(0))
    return selected


def stage_distribution(candidates):
    counts = Counter(c["stage"] for c in candidates)
    return {name: counts.get(name, 0) for name in STAGES}


def select_candidates(pools, requested):
    """Return {scheme: selected candidates} for B0, B1 and B2, plus Nc and pool sizes."""
    nc = common_count(requested, [pools["B0"], pools["B1"]])
    selected = {"B0": by_ply(pools["B0"])[:nc], "B1": by_ply(pools["B1"])[:nc],
                "B2": select_by_stage(pools["B1"], nc)}
    for scheme, rows in selected.items():
        if len(rows) != nc or len({tuple(r["key"]) for r in rows}) != nc:
            raise AssertionError(f"{scheme} did not select {nc} distinct candidates")
    summary = {"nc_requested": requested, "nc": nc,
               "pool_sizes": {"B0": len(pools["B0"]), "B1": len(pools["B1"]), "B2": len(pools["B1"])},
               "stage_distribution": {scheme: stage_distribution(rows) for scheme, rows in selected.items()},
               "pool_stage_distribution": {name: stage_distribution(pools[name]) for name in ("B0", "B1")}}
    return selected, summary
