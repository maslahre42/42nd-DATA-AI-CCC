"""Movement rules for Colony: One.

Computes every legal action for a given piece. An action is either a **move**
(``kind="move"``, the piece relocates to the target square) or a **shot**
(``kind="shoot"``, the piece stays put and only the target piece is removed).

Target geometry and occupancy uniquely determine which kind of action applies,
so there is never more than one action for a given ``(from_pos, to_pos)``
pair. That invariant is what makes ``(from, to)`` a sufficient identifier in
move notation and in the UI.

These are the game's *full* legality rules. There is no check or checkmate
concept and therefore no self-check filter: a Kernel may legally move next to
an attacked square, and may capture the enemy Kernel directly (subject to the
bodyguard rule in :meth:`~colony_one.board.Board.apply_action`).

    PING    extended range, captures diagonally by moving
    DAEMON  slides orthogonally, shoots adjacent
    EXPLOIT jumps, ignores everything in between
    TRACE   steps one square orthogonally, shoots diagonally at range
    ICE     short range in all 8 directions, captures on contact
    KERNEL  one square in any direction
"""

from __future__ import annotations

from colony_one import constants
from colony_one.action import Action, ActionKind, Position
from colony_one.board import Board
from colony_one.pieces import Piece, PieceType

# Direction vectors as (row_delta, col_delta).
_DIAGONALS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ORTHOGONALS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ALL8: list[tuple[int, int]] = _ORTHOGONALS + _DIAGONALS
_EXPLOIT_JUMPS: list[tuple[int, int]] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


def legal_actions(board: Board, pos: Position) -> list[Action]:
    """Every legal action for the piece on ``pos``.

    Returns an empty list if the square is empty or firewall-locked.
    """
    piece = board.piece_at(*pos)
    if piece is None:
        return []
    # A piece on a locked square cannot act. Defensive only: the firewall
    # destroys whatever stands on a square as it locks it.
    if board.is_locked(*pos):
        return []

    dispatch = {
        PieceType.PING: _ping_actions,
        PieceType.DAEMON: _daemon_actions,
        PieceType.TRACE: _trace_actions,
        PieceType.ICE: _ice_actions,
        PieceType.EXPLOIT: _exploit_actions,
        PieceType.KERNEL: _kernel_actions,
    }
    return dispatch[piece.piece_type](board, pos, piece)


def legal_actions_for_team(board: Board, team: str) -> list[Action]:
    """Every legal action for every piece of ``team``.

    Ordering is row-major by source square, then per-piece generator order.
    It is deterministic and must stay so: search tie-breaking and the action
    ordering seen by agents both depend on it.
    """
    actions: list[Action] = []
    for row in range(constants.BOARD_SIZE):
        for col in range(constants.BOARD_SIZE):
            piece = board.grid[row][col]
            if piece is None or piece.team != team:
                continue
            actions.extend(legal_actions(board, (row, col)))
    return actions


# ---------------------------------------------------------------------------
# Composable movement helpers
# ---------------------------------------------------------------------------
def _sliding_actions(
    board: Board,
    pos: Position,
    piece: Piece,
    directions: list[tuple[int, int]],
    max_range: int,
    *,
    contact_capture: bool,
) -> list[Action]:
    """Slide through empty squares, stopping at the first piece.

    Friendly pieces always block. An enemy piece is captured by moving onto
    it only when ``contact_capture`` is set or the distance exceeds 1 -- the
    Daemon uses the latter, so an adjacent enemy blocks it instead of being
    captured.
    """
    actions: list[Action] = []
    row, col = pos
    for dr, dc in directions:
        r, c = row + dr, col + dc
        distance = 1
        while distance <= max_range and board.in_bounds(r, c):
            if board.is_locked(r, c):
                break  # the firewall blocks like a wall
            target = board.grid[r][c]
            if target is None:
                actions.append(Action(pos, (r, c), "move"))
            elif target.team != piece.team and (contact_capture or distance > 1):
                actions.append(Action(pos, (r, c), "move"))
                break
            else:
                break  # friendly piece, or an unreachable contact-range enemy
            r += dr
            c += dc
            distance += 1
    return actions


def _capture_slide(
    board: Board,
    pos: Position,
    piece: Piece,
    directions: list[tuple[int, int]],
    max_range: int,
    kind: ActionKind,
    min_range: int = 1,
) -> list[Action]:
    """Slide through empty squares without acting, stopping at the first piece.

    Never acts *through* a piece: the ray breaks at the first occupied square,
    whichever team it belongs to. An enemy standing closer than ``min_range``
    is not a valid target and still blocks -- that is the Trace's minimum
    shooting distance of 2.
    """
    actions: list[Action] = []
    row, col = pos
    for dr, dc in directions:
        r, c = row + dr, col + dc
        distance = 1
        while distance <= max_range and board.in_bounds(r, c):
            if board.is_locked(r, c):
                break
            target = board.grid[r][c]
            if target is not None:
                if target.team != piece.team and distance >= min_range:
                    actions.append(Action(pos, (r, c), kind))
                break
            r += dr
            c += dc
            distance += 1
    return actions


def _adjacent_move(
    board: Board, pos: Position, piece: Piece, directions: list[tuple[int, int]]
) -> list[Action]:
    """Distance 1, empty target squares only."""
    actions: list[Action] = []
    row, col = pos
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if (
            board.in_bounds(r, c)
            and not board.is_locked(r, c)
            and board.grid[r][c] is None
        ):
            actions.append(Action(pos, (r, c), "move"))
    return actions


def _adjacent_shoot(
    board: Board, pos: Position, piece: Piece, directions: list[tuple[int, int]]
) -> list[Action]:
    """Distance 1, enemy-occupied target squares only."""
    actions: list[Action] = []
    row, col = pos
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if not board.in_bounds(r, c):
            continue
        if board.is_locked(r, c):
            continue
        target = board.grid[r][c]
        if target is not None and target.team != piece.team:
            actions.append(Action(pos, (r, c), "shoot"))
    return actions


def _empty_slide(
    board: Board,
    pos: Position,
    piece: Piece,
    directions: list[tuple[int, int]],
    max_range: int,
) -> list[Action]:
    """Slide onto empty squares only; stops at any piece, never captures."""
    actions: list[Action] = []
    row, col = pos
    for dr, dc in directions:
        r, c = row + dr, col + dc
        distance = 1
        while (
            distance <= max_range
            and board.in_bounds(r, c)
            and not board.is_locked(r, c)
            and board.grid[r][c] is None
        ):
            actions.append(Action(pos, (r, c), "move"))
            r += dr
            c += dc
            distance += 1
    return actions


# ---------------------------------------------------------------------------
# Per-piece movement patterns
# ---------------------------------------------------------------------------
def _daemon_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """Slide up to the Daemon's range orthogonally, or shoot a neighbour.

    The distance-1 asymmetry is deliberate and easy to get wrong: an adjacent
    enemy **blocks** the slide and cannot be captured by moving onto it. It
    must be shot instead. At distance 2 or 3 the enemy is captured by moving.
    """
    return _sliding_actions(
        board, pos, piece, _ORTHOGONALS, PieceType.DAEMON.move_range,
        contact_capture=False,
    ) + _adjacent_shoot(board, pos, piece, _ALL8)


def _trace_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """Step one square orthogonally, or shoot diagonally at range.

    Two restrictions that are easy to miss: the step is **orthogonal only**
    (not all 8 directions), and the shot has a **minimum range of 2** -- an
    enemy on an adjacent diagonal is too close to shoot, and blocks the line
    behind it. The Trace never captures by moving.
    """
    return _adjacent_move(board, pos, piece, _ORTHOGONALS) + _capture_slide(
        board,
        pos,
        piece,
        _DIAGONALS,
        PieceType.TRACE.shoot_range,
        "shoot",
        min_range=2,
    )


def _ice_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """Slide up to the Ice's range in all 8 directions, capturing on contact."""
    return _sliding_actions(
        board, pos, piece, _ALL8, PieceType.ICE.move_range, contact_capture=True
    )


def _ping_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """Slide orthogonally onto empty squares, capture diagonally by moving."""
    return _empty_slide(
        board, pos, piece, _ORTHOGONALS, PieceType.PING.move_range
    ) + _capture_slide(
        board, pos, piece, _DIAGONALS, PieceType.PING.capture_range, "move"
    )


def _exploit_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """Exploit jumps. Ignores everything in between."""
    actions: list[Action] = []
    row, col = pos
    for dr, dc in _EXPLOIT_JUMPS:
        r, c = row + dr, col + dc
        if not board.in_bounds(r, c):
            continue
        if board.is_locked(r, c):
            continue
        target = board.grid[r][c]
        if target is None or target.team != piece.team:
            actions.append(Action(pos, (r, c), "move"))
    return actions


def _kernel_actions(board: Board, pos: Position, piece: Piece) -> list[Action]:
    """One square in any direction."""
    actions: list[Action] = []
    row, col = pos
    for dr, dc in _ALL8:
        r, c = row + dr, col + dc
        if not board.in_bounds(r, c):
            continue
        if board.is_locked(r, c):
            continue
        target = board.grid[r][c]
        if target is None or target.team != piece.team:
            actions.append(Action(pos, (r, c), "move"))
    return actions
