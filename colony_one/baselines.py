"""Reference opponents: :class:`RandomAgent` and :class:`GreedyAgent`.

Two floors to measure your own agent against without needing a trained model.
Both are NumPy-only and follow the same ``select(board, team, actions_left,
round_number)`` protocol the play scripts and the validator expect.

The random *ONNX* baseline is a different thing entirely -- a shipped file,
``play/random.onnx``, built by ``tests/random_onnx.py``.
"""

from __future__ import annotations

import numpy as np

from colony_one import encoding
from colony_one.action import Action, Position
from colony_one.board import Board
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team


class RandomAgent:
    """Picks a uniformly random legal action.

    The floor. Any agent worth submitting beats this decisively.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, board: Board, team: str, actions_left: int,
               round_number: int) -> Action | None:
        actions = legal_actions_for_team(board, team)
        if not actions:
            return None
        return actions[int(self.rng.integers(len(actions)))]


_PIECE_VALUE = {
    PieceType.KERNEL: 100,
    PieceType.ICE: 50,
    PieceType.DAEMON: 45,
    PieceType.TRACE: 30,
    PieceType.EXPLOIT: 25,
    PieceType.PING: 15,
}
"""Rough material values for :class:`GreedyAgent`.

Hand-set, not tuned -- this is a sparring partner, not an evaluation function.
"""


class GreedyAgent:
    """Takes the most valuable capture available, otherwise moves randomly.

    A slightly stronger floor than :class:`RandomAgent`: an agent that cannot beat
    it has not learned to protect its pieces.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, board: Board, team: str, actions_left: int,
               round_number: int) -> Action | None:
        actions = legal_actions_for_team(board, team)
        if not actions:
            return None

        # Urgency 1.0 (observation plane 14) means the fire falls at the end of
        # this round, so a Kernel in it has to move now.
        if encoding.firewall_urgency(round_number) == 1.0:
            actions = self._kernel_escapes(board, actions) or actions

        best_value = 0.0
        best: list[Action] = []
        for action in actions:
            record = board.apply_action(action)
            captured = record.captured_piece
            value = _PIECE_VALUE.get(captured.piece_type, 0.0) if captured else 0.0
            board.undo_action(record)

            if value > best_value:
                best_value, best = value, [action]
            elif value == best_value:
                best.append(action)

        pool = best if best_value > 0 else actions
        return pool[int(self.rng.integers(len(pool)))]

    @staticmethod
    def _kernel_escapes(board: Board, actions: list[Action]) -> list[Action]:
        def _burns_next(board: Board, pos: Position) -> bool:
            row, col = pos
            if board.is_locked(row, col):
                return False
            return any(board.is_locked(row + dr, col + dc)
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1))

        """The moves that carry our Kernel off a burning square, if any."""
        return [
            action
            for action in actions
            if action.kind == "move"
            and board.piece_at(*action.from_pos).piece_type is PieceType.KERNEL
            and _burns_next(board, action.from_pos)
            and not _burns_next(board, action.to_pos)
        ]
