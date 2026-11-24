"""A simple alpha-beta baseline for local testing (not a submission).

Stands in for the course's "simple alpha-beta baseline with heuristic
evaluation", whose exact definition has not been published. It uses only the
framework helpers and shares no code with student_agent.py, so results against
it are an independent check.

Search: iterative-deepening minimax with alpha-beta pruning, up to MAX_DEPTH
plies, stopping at TIME_LIMIT seconds. No move ordering beyond the framework's
move order, no transposition table.

Evaluation (open lines): a line holding top pieces of only one player scores
LINE_SCORES[n] for that player, where n is the number of its pieces in the
line; mixed lines and lines through the obstacle score 0. Top-piece sizes add
a small material term.

Usage:
    python simulator.py --player_1 student_agent \
        --player_2 baselines/ab_baseline_agent.py --autoplay --strict
"""
import time

from agents.agent import Agent
from constants import BOARD_SIZE
from helpers import LINES, check_endgame, copy_state, execute_move, get_valid_moves, top_view
from store import register_agent


MAX_DEPTH = 3
TIME_LIMIT = 1.5
WIN = 1_000_000
LINE_SCORES = (0, 1, 10, 100, 1000)
SIZE_SCORES = (1, 2, 4)


class _Timeout(Exception):
    pass


def evaluate(board, player):
    owners, sizes = top_view(board)
    own, sz = owners.tolist(), sizes.tolist()
    opponent = 3 - player
    score = 0
    for line in LINES[4]:
        mine = theirs = 0
        for r, c in line:
            if own[r][c] == player:
                mine += 1
            elif own[r][c] == opponent:
                theirs += 1
        if mine and not theirs:
            score += LINE_SCORES[mine]
        elif theirs and not mine:
            score -= LINE_SCORES[theirs]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if own[r][c] == player:
                score += SIZE_SCORES[sz[r][c]]
            elif own[r][c] == opponent:
                score -= SIZE_SCORES[sz[r][c]]
    return score


@register_agent("ab_baseline")
class AlphaBetaBaseline(Agent):

    def __init__(self):
        super().__init__()
        self.name = "AlphaBetaBaseline"
        self.calls = 0

    def step(self, board, reserves, player, opponent):
        self.deadline = time.perf_counter() + TIME_LIMIT
        self.root_player = player
        ply = 2 * self.calls + player - 1
        self.calls += 1

        moves = get_valid_moves(board, reserves, player)
        best = moves[0]
        for depth in range(1, MAX_DEPTH + 1):
            try:
                _, move = self._search(board, reserves, player, ply, depth,
                                       -float("inf"), float("inf"))
            except _Timeout:
                break
            if move is not None:
                best = move
        return best

    def _search(self, board, reserves, to_move, ply, depth, alpha, beta):
        """Minimax value from root_player's perspective, with alpha-beta."""
        if time.perf_counter() >= self.deadline:
            raise _Timeout
        over, winner, _ = check_endgame(board, reserves, to_move, ply)
        if over:
            if winner == 0:
                return 0, None
            sign = 1 if winner == self.root_player else -1
            return sign * (WIN - ply), None
        if depth == 0:
            return evaluate(board, self.root_player), None

        maximizing = to_move == self.root_player
        best_value = -float("inf") if maximizing else float("inf")
        best_move = None
        for move in get_valid_moves(board, reserves, to_move):
            b, r = copy_state(board, reserves)
            execute_move(b, r, to_move, move)
            value, _ = self._search(b, r, 3 - to_move, ply + 1, depth - 1, alpha, beta)
            if maximizing and value > best_value:
                best_value, best_move = value, move
                alpha = max(alpha, value)
            elif not maximizing and value < best_value:
                best_value, best_move = value, move
                beta = min(beta, value)
            if alpha >= beta:
                break
        return best_value, best_move
