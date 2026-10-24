"""Board/tensor encoding and action indexing for Colony: One.

**Canonical view.** The board is always encoded from the perspective of the
team to move. When Magenta is to move the board is rotated 180 degrees -- the
same point reflection that places Cyan and Magenta opposite each other in
:meth:`~colony_one.board.Board.reset`. The network therefore never has to
learn "Magenta" versus "Cyan", only "own" versus "enemy", which halves the
effective state space.

Because the rotation makes both teams look alike, plane 15 states explicitly
whose turn it is. Without it, two positions with *different* values could
produce identical tensors -- see :data:`NUM_PLANES` below.
"""

from __future__ import annotations

import numpy as np

from colony_one import constants
from colony_one.action import Action
from colony_one.board import Board
from colony_one.pieces import PieceType

N = constants.BOARD_SIZE  # 8

NUM_PIECE_TYPES = len(PieceType)  # 6
"""One occupancy plane per piece type per team.

The plane order is :class:`~colony_one.pieces.PieceType` declaration
order, via :meth:`~colony_one.pieces.PieceType.encode_index`. Do not
reorder the enum.
"""

# Plane budget: 6 own + 6 enemy occupancy, locked, actions_left, firewall
# urgency, side-to-move.
NUM_PLANES = NUM_PIECE_TYPES * 2 + 4  # 16
"""Number of observation planes.

Plane 15 (side-to-move) is what distinguishes this encoding from a purely
canonical one. It is required because several rules are asymmetric: Magenta
takes every tiebreak, and the firewall only shrinks after Magenta's turn, so
an otherwise identical position has a different value depending on whose turn
it is. A canonical-only encoding cannot express that difference and no value
head could learn it.
"""

# Plane indices, named so downstream code never hardcodes offsets.
LOCKED_PLANE = NUM_PIECE_TYPES * 2            # 12
ACTIONS_LEFT_PLANE = NUM_PIECE_TYPES * 2 + 1  # 13
URGENCY_PLANE = NUM_PIECE_TYPES * 2 + 2       # 14
SIDE_TO_MOVE_PLANE = NUM_PIECE_TYPES * 2 + 3  # 15

NUM_SQUARES = N * N  # 64
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 4096
"""Flat action space: ``(from_square, to_square)``.

**There is no ``kind`` axis.** Target geometry and occupancy uniquely
determine whether a given ``(from, to)`` pair is a move or a shot -- no piece
can ever play both to the same square -- so a separate axis would double the
policy head while every second entry stayed permanently unreachable. See
:class:`~colony_one.action.Action` for the invariant, and
:func:`resolve_action` for recovering the kind.

Very sparse -- the real branching factor is roughly 12 to 30 -- but a fixed,
framework-neutral size that any policy head can emit.
"""


def _opponent(team: str) -> str:
    return constants.MAGENTA if team == constants.CYAN else constants.CYAN


def _rotate(row: int, col: int) -> tuple[int, int]:
    """180-degree point reflection about the board centre."""
    return (N - 1 - row, N - 1 - col)


def canonical_pos(pos: tuple[int, int], team: str) -> tuple[int, int]:
    """Convert a real board position into ``team``'s canonical view."""
    return pos if team == constants.CYAN else _rotate(*pos)


def real_pos(pos: tuple[int, int], team: str) -> tuple[int, int]:
    """Inverse of :func:`canonical_pos`. The rotation is its own inverse."""
    return canonical_pos(pos, team)


def firewall_urgency(round_number: int) -> float:
    """How close the next firewall shrink is, normalised to ``[0, 1]``.

    ``1.0`` means a shrink fires at the end of the current round, ``0.0``
    means that no shrink remains.

    The divisor is fixed at the first shrink interval
    (``SHRINK_ROUNDS[0]`` == 8 rounds) rather than the actual gap to the
    previous shrink. After a normal shrink this resets deep -- e.g. round 9,
    right after the round-8 shrink, reads ``1/8`` and climbs back to ``1.0``
    over the full 8-round gap to round 16. The final 24 -> 26 pair is only
    two rounds apart, so the same fixed divisor gives a much shallower reset:
    round 25 reads ``0.875`` instead of ``1/8`` before climbing back to
    ``1.0`` at round 26. This is engine behaviour, not a bug to fix here.
    """
    next_shrink = None
    for s in constants.SHRINK_ROUNDS:
        if s >= round_number:
            next_shrink = s
            break
    if next_shrink is None:
        return 0.0
    gap = constants.SHRINK_ROUNDS[0]
    remaining = next_shrink - round_number
    return float(max(0.0, min(1.0, 1.0 - remaining / gap)))


def encode_board(
    board: Board, team: str, actions_left: int, round_number: int
) -> np.ndarray:
    """Encode the position from ``team``'s perspective.

    Args:
        board: Current board state.
        team: The team to move; determines the canonical rotation and which
            occupancy planes count as "own".
        actions_left: Actions remaining in the current turn (1 or 2).
        round_number: Current round, 1-based.

    Returns:
        ``(16, 8, 8)`` float32 array with all values in ``[0, 1]``.

        =======  ==================================================
        Plane    Meaning
        =======  ==================================================
        0-5      Own pieces, in ``PieceType`` order
        6-11     Enemy pieces, same order
        12       Firewall-locked squares
        13       ``actions_left / MAX_ACTION_PER_TURN`` (0.5 or 1.0)
        14       Firewall urgency
        15       1.0 if Magenta is to move, else 0.0
        =======  ==================================================
    """
    planes = np.zeros((NUM_PLANES, N, N), dtype=np.float32)

    for row in range(N):
        for col in range(N):
            crow, ccol = canonical_pos((row, col), team)
            piece = board.grid[row][col]
            if piece is not None:
                type_idx = piece.piece_type.encode_index()
                offset = 0 if piece.team == team else NUM_PIECE_TYPES
                planes[offset + type_idx, crow, ccol] = 1.0
            if board.is_locked(row, col):
                planes[LOCKED_PLANE, crow, ccol] = 1.0

    planes[ACTIONS_LEFT_PLANE, :, :] = actions_left / constants.MAX_ACTION_PER_TURN
    planes[URGENCY_PLANE, :, :] = firewall_urgency(round_number)
    planes[SIDE_TO_MOVE_PLANE, :, :] = 1.0 if team == constants.MAGENTA else 0.0
    return planes


def action_index(action: Action, team: str) -> int:
    """Convert a real :class:`Action` into a canonical action index.

    ``action.kind`` is deliberately **not** encoded: it is implied by the
    ``(from, to)`` pair together with the board (see :data:`NUM_ACTIONS`), so
    a move and a shot can never collide on one index.

    The index is only meaningful together with the team it was produced for,
    since the coordinates are canonicalised.
    """
    fr, fc = canonical_pos(action.from_pos, team)
    tr, tc = canonical_pos(action.to_pos, team)
    from_idx = fr * N + fc
    to_idx = tr * N + tc
    return from_idx * NUM_SQUARES + to_idx


def index_to_positions(idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Invert :func:`action_index`, returning **canonical** coordinates.

    Returns ``(from_pos, to_pos)``. The kind is not recoverable from an index
    alone -- use :func:`resolve_action`, which reads it off the board.
    """
    to_idx = idx % NUM_SQUARES
    from_idx = idx // NUM_SQUARES
    from_pos: tuple[int, int] = (from_idx // N, from_idx % N)
    to_pos: tuple[int, int] = (to_idx // N, to_idx % N)
    return from_pos, to_pos


def index_to_squares(idx: int, team: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Invert :func:`action_index` into **real** board coordinates.

    Like :func:`index_to_positions` but undoes the canonical rotation, so the
    result indexes ``board.grid`` directly.
    """
    from_pos, to_pos = index_to_positions(idx)
    return real_pos(from_pos, team), real_pos(to_pos, team)


def resolve_action(idx: int, team: str, board: Board) -> Action | None:
    """Turn an action index into a real :class:`Action` using the board.

    Since :data:`NUM_ACTIONS` has no kind axis, the kind is recovered by
    looking the ``(from, to)`` pair up among the legal actions for ``team``.
    That lookup is exact: the pair identifies at most one legal action.

    Returns:
        The matching :class:`Action`, or ``None`` if the index is not legal
        in this position.
    """
    from colony_one.rules import legal_actions

    from_pos, to_pos = index_to_squares(idx, team)
    piece = board.piece_at(*from_pos)
    if piece is None or piece.team != team:
        return None
    for action in legal_actions(board, from_pos):
        if action.to_pos == to_pos:
            return action
    return None


def decode_action(idx: int, team: str, board: Board) -> Action:
    """Invert :func:`action_index`, returning a real :class:`Action`.

    Requires the ``board`` the index was produced for: the kind is implied by
    the position, not stored in the index.

    Raises:
        ValueError: The index is not a legal action for ``team`` here.
    """
    action = resolve_action(idx, team, board)
    if action is None:
        from_pos, to_pos = index_to_squares(idx, team)
        raise ValueError(
            f"action index {idx} ({from_pos} -> {to_pos}) is not legal for {team}"
        )
    return action


def legal_action_mask(board: Board, team: str) -> np.ndarray:
    """Boolean mask of shape ``(4096,)``, ``True`` at every legal index.

    Convenient for masking a policy head's logits before softmax::

        logits[~mask] = -np.inf
    """
    from colony_one.rules import legal_actions_for_team

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal_actions_for_team(board, team):
        mask[action_index(action, team)] = True
    return mask
