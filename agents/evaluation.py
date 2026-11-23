"""Static evaluation and terminal utility for Goblet Gobblers."""

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
