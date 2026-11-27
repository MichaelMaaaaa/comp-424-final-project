"""Static evaluation and terminal utility for Goblet Gobblers."""

from collections.abc import Mapping

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


def evaluation_parameters(config=None):
    """Validate evaluation settings and return (lambda, q) with an immutable q."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("evaluation config must be a mapping")
    if not config:
        return 1, SIZE_WEIGHTS
    if config.keys() - {"evaluation_id", "lambda", "q"}:
        raise ValueError("evaluation config supports only evaluation_id, lambda and q")

    name = config.get("evaluation_id")
    if "evaluation_id" in config and (not isinstance(name, str) or not name.strip()):
        raise ValueError("evaluation_id must be a nonempty string")
    if name not in ("E0", "E1") and not {"lambda", "q"} <= config.keys():
        raise ValueError("custom evaluation requires both lambda and q")

    strength = config.get("lambda", 0 if name == "E0" else 1)
    sizes = config.get("q", SIZE_WEIGHTS)
    if isinstance(strength, bool) or not isinstance(strength, int) or not 0 <= strength <= 10:
        raise ValueError("lambda must be an integer from 0 to 10")
    if not isinstance(sizes, (list, tuple)) or len(sizes) != 3:
        raise ValueError("q must be a list or tuple of three integers")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= 100
           for weight in sizes) or sizes[2] != 100:
        raise ValueError("q weights must be integers from 1 to 100 with q[2] equal to 100")
    sizes = tuple(sizes)

    if name == "E0" and strength != 0:
        raise ValueError("E0 evaluation requires lambda=0")
    if name == "E1" and (strength != 1 or sizes != SIZE_WEIGHTS):
        raise ValueError("E1 evaluation requires lambda=1 and q=(1, 20, 100)")
    return strength, sizes


def make_evaluator(config=None):
    """Return a nonterminal evaluator with fixed line and size weights."""
    strength, sizes = evaluation_parameters(config)
    if strength == 0:
        return evaluate_e0
    if strength == 1 and sizes == SIZE_WEIGHTS:
        return evaluate
    weights = tuple(strength * size for size in sizes)

    def configured_evaluate(state, player):
        score = evaluate_e0(state, player)
        opponent = 3 - player
        for owner, size in zip(state.top_owners, state.top_sizes):
            if owner == player:
                score += weights[size]
            elif owner == opponent:
                score -= weights[size]
        return score

    return configured_evaluate
