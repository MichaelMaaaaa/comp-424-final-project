"""Read-only Step 8 reducers. All repetitions are within one saved trajectory."""

from collections import Counter, defaultdict
from statistics import mean

from search_experiments.common import canonical_json, obstacle_class, stage
from search_experiments.runner import paired_difference, summarize_games


def result_summary(candidate, opponent, config_ids, records, infrastructure_attempts):
    planned = [f"{candidate}__{config}__P{seat}" for config in config_ids for seat in (1, 2)]
    if len(set(planned)) != len(planned):
        raise ValueError("duplicate final configuration")
    seen = set()
    scheduled = set(planned)
    for row in records:
        name = f"{candidate}__{row['config_id']}__P{row['candidate_seat']}"
        if row["game_id"] != name or name not in scheduled or name in seen or row["status"] != "adjudicated":
            raise ValueError("unexpected or duplicate adjudication")
        seen.add(name)
    summary = summarize_games(records, planned)
    return dict(summary, candidate=candidate, opponent=opponent,
                status="pending" if not summary["complete"] else "complete" if summary["eligible"] else "blocked",
                observed_mean_score=mean(r["score"] for r in records) if records else None,
                infrastructure_attempts=infrastructure_attempts)


def contrasts(roles, ab_results, config_ids, seed):
    rows = []
    for old_role, new_role in (("V0", "V1"), ("V1", "V2"), ("V2", "V3")):
        old, new = roles[old_role], roles.get(new_role)
        row = dict(old_role=old_role, new_role=new_role, old_candidate=old, new_candidate=new, alias=old == new,
                   opponent="ab", status="omitted" if new is None else "pending", complete=False,
                   old_eligible=ab_results[old]["eligible"], new_eligible=None, n_configs=0,
                   planned_configs=len(config_ids), per_config={}, mean=None, bootstrap_config_ci95=None,
                   bootstrap_seed=seed, resamples=2000, sampling_unit="configuration (both seats together)")
        if new is not None:
            before, after = ab_results[old], ab_results[new]
            row["new_eligible"] = after["eligible"]
            common = set(before["paired_scores"]) & set(after["paired_scores"])
            row["per_config"] = {c: after["paired_scores"][c] - before["paired_scores"][c]
                                 for c in config_ids if c in common}
            row["n_configs"] = len(row["per_config"])
            if before["complete"] and after["complete"] and common == set(config_ids):
                row.update(paired_difference(after, before, seed), status="complete")
        rows.append(row)
    return rows


def _field_status(info, field):
    if info is None:
        return "missing_payload"
    if field not in info:
        return "missing_field"
    return "null" if info[field] is None else "observed"


def turn_rows(game, opponent):
    positions, actions = set(), set()
    rows = []
    played = iter(game["referee"]["moves"])
    for turn in game["turns"]:
        state, player = turn["state"], turn["player"]
        position = canonical_json({key: value for key, value in state.items() if key != "ply"})
        move = next(played) if turn["move"] is not None else None
        action = None
        if move is None:
            kind = "failed"
        elif move.get("pass") is True:
            kind = "pass"
        else:
            kind = "placement" if move["src"] is None else "relocation"
            action = (player, None if move["src"] is None else tuple(move["src"]), tuple(move["dest"]), move["size"])
        role = turn["actor_role"]
        candidate_call = role == "candidate" and kind != "pass"
        info = turn.get("search")
        raw = info or {}
        depth_status = _field_status(info, "completed_depth")
        if not candidate_call or raw.get("status") in ("single_move", "book") or raw.get("stop_reason") in (
                "single_move", "book_direct", "deadline_before_search"):
            depth_status = "not_applicable"
        elif raw.get("status") == "fallback" and raw.get("stop_reason") is None:
            depth_status = "unknown_search_entry"
        reserves = [sum(state["reserves"][seat]) for seat in (1, 2)]
        row = dict(candidate=game["variant_id"], opponent=opponent, game_id=game["game_id"],
                   config_id=game["config_id"], candidate_seat=game["candidate_seat"], split=game["split"],
                   ply=turn["ply"], player=player, actor_role=role, stage=stage(turn["ply"]),
                   obstacle_class=obstacle_class(game["obstacle"]), removed_size=game["removed_size"],
                   near_cap=turn["ply"] >= 90, reserves_p1=reserves[0], reserves_p2=reserves[1],
                   reserves_actor=reserves[player-1], reserves_total=sum(reserves),
                   action_type=kind, failure_reason=turn.get("failure_reason"), repeated_position=position in positions,
                   repeated_action=action in actions if action is not None else None,
                   src=(move or {}).get("src"), dest=(move or {}).get("dest"), size=(move or {}).get("size"),
                   flags=(move or {}).get("flags"), step_s=(move or {}).get("agent_time"),
                   candidate_call=candidate_call, telemetry_observed=info is not None if candidate_call else None,
                   raw_status=raw.get("status"), raw_completed_depth=raw.get("completed_depth"),
                   completed_depth=raw.get("completed_depth") if depth_status == "observed" else None,
                   depth_status=depth_status, stop_reason=raw.get("stop_reason"),
                   interrupted_depth=raw.get("interrupted_depth"), book_metadata=raw.get("book_metadata"))
        for field in ("book_queried", "book_matched", "book_action", "stop_reason", "interrupted_depth"):
            row[field] = raw.get(field)
            row[field + "_status"] = _field_status(info, field) if candidate_call else "not_applicable"
        rows.append(row)
        positions.add(position)
        if action is not None:
            actions.add(action)
    return rows


def candidate_start_resources(ref, seat):
    """Read candidate initialization time and RSS, accounting for startup order."""
    init = (ref.get("time", {}).get(str(seat)) or {}).get("init")
    rss = (ref.get("max_rss_mb") or {}).get(str(seat))
    if ref["reason"] in ("init_failed", "init_timeout"):
        failed_seat = 3 - ref["winner"]
        if seat > failed_seat:
            init, rss = None, None
        elif seat == failed_seat and init == 0:
            init = None
    return init, rss


def game_row(game, opponent, turns):
    ref, seat = game["referee"], game["candidate_seat"]
    init, rss = candidate_start_resources(ref, seat)
    counts = Counter(t["action_type"] for t in turns)
    return {**{k: game.get(k) for k in ("game_id", "config_id", "split", "removed_size", "candidate_seat",
                                         "score", "outcome", "technical_failure", "opponent_technical_failure",
                                         "candidate_budget_failure", "candidate_max_step_s")},
            "candidate": game["variant_id"], "opponent": opponent, "obstacle_class": obstacle_class(game["obstacle"]),
            "reason": ref["reason"], "winner": ref["winner"], "plies": ref["plies"], "init_s": init, "rss_mb": rss,
            "draw_100_ply": (ref["winner"], ref["reason"], ref["plies"]) == (0, "ply_cap", 100),
            "snapshots": len(turns), "repeated_positions": sum(t["repeated_position"] for t in turns),
            "applied_actions": counts["placement"] + counts["relocation"],
            "repeated_actions": sum(t["repeated_action"] is True for t in turns),
            **{k: counts[k] for k in ("placement", "relocation", "pass", "failed")}}


def live_metrics(game_ids, turns):
    calls = [t for t in turns if t["candidate_call"]]
    observed = [t for t in calls if type(t["book_queried"]) is bool]
    queried = [t for t in observed if t["book_queried"]]
    known = [t for t in queried if type(t["book_matched"]) is bool]
    hits = sum(t["book_matched"] for t in known)
    adopted = [t["book_action"] for t in calls if t["book_action"] in ("none", "direct", "hint")]
    by_game = defaultdict(list)
    for turn in calls:
        by_game[turn["game_id"]].append(turn)
    hit_games = known_games = 0
    for game_id in game_ids:
        group = by_game[game_id]
        hit = any(t["book_queried"] is True and t["book_matched"] is True for t in group)
        if hit or all(t["book_queried"] is False or
                      (t["book_queried"] is True and type(t["book_matched"]) is bool) for t in group):
            known_games += 1
            hit_games += hit
    no_calls_in_observed_games = bool(game_ids) and not calls
    return dict(games=len(game_ids), candidate_calls=len(calls),
                telemetry_observed=sum(t["telemetry_observed"] for t in calls),
                query_observed=len(observed), query_missing=len(calls)-len(observed),
                queries=len(queried) if observed or no_calls_in_observed_games else None, match_observed_queries=len(known),
                match_missing_queries=len(queried)-len(known),
                matches=hits if known or (observed and not queried) or no_calls_in_observed_games else None,
                hit_rate=hits/len(known) if known else None, adoption_observed=len(adopted),
                direct_returns=adopted.count("direct") if adopted or no_calls_in_observed_games else None,
                hints=adopted.count("hint") if adopted or no_calls_in_observed_games else None,
                games_with_known_hit_status=known_games, games_with_hit=hit_games if known_games else None,
                games_with_hit_fraction=hit_games/known_games if known_games else None)


def _observations(rows, field, prefix):
    values = [r[field] for r in rows if r[field] is not None]
    return {prefix + "_observed": len(values), prefix + "_mean": mean(values) if values else None,
            prefix + "_max": max(values) if values else None}


def turn_metrics(turns):
    candidate = [t for t in turns if t["candidate_call"]]
    return dict(turns=len(turns), games=len({t["game_id"] for t in turns}),
                repeated_positions=sum(t["repeated_position"] for t in turns),
                repeated_actions=sum(t["repeated_action"] is True for t in turns),
                applied_actions=sum(t["repeated_action"] is not None for t in turns),
                depth_zero=(sum(t["completed_depth"] == 0 for t in candidate)
                            if any(t["completed_depth"] is not None for t in candidate) else None),
                depth_status=dict(Counter(t["depth_status"] for t in candidate)),
                stop_reasons=dict(Counter(t["stop_reason"] for t in candidate if t["stop_reason"] is not None)),
                stop_reason_observed=sum(t["stop_reason"] is not None for t in candidate),
                **_observations(candidate, "completed_depth", "depth"), **_observations(candidate, "step_s", "step_s"))


def game_metrics(games):
    counts = Counter(g["outcome"] for g in games)
    return dict(games=len(games), W=counts["W"], D=counts["D"], L=counts["L"],
                observed_mean_score=mean(g["score"] for g in games) if games else None,
                **{key: sum(g[key] for g in games) for key in
                   ("draw_100_ply", "repeated_positions", "snapshots", "repeated_actions", "applied_actions",
                    "placement", "relocation", "pass", "failed", "technical_failure", "opponent_technical_failure")},
                **_observations(games, "init_s", "init_s"), **_observations(games, "rss_mb", "rss_mb"),
                **_observations(games, "candidate_max_step_s", "candidate_max_step_s"))


def strata(rows, fields, reducer):
    """Count distinct games and observations within a marginal or joint stratum."""
    output = []
    for field in fields:
        groups = defaultdict(list)
        dimensions = (field,) if isinstance(field, str) else field
        for row in rows:
            groups[tuple(row[key] for key in dimensions)].append(row)
        for values, group in groups.items():
            keys = dict(zip(dimensions, values))
            output.append(dict(dimension="/".join(dimensions), value=values[0] if len(values) == 1 else keys,
                               **keys, **reducer(group)))
    return output
