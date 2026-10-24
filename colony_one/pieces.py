"""Piece types and the Piece runtime object for Colony: One.

All knowledge about a piece type -- its identity, abbreviation, and movement
ranges -- lives on the :class:`PieceType` enum in one place, so no second
table exists that could drift out of sync with it. :class:`Piece` is the
runtime state of one piece standing on the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Teams are plain strings. Spelled out here rather than imported from
# constants so that this module has no internal imports, which lets
# constants.py describe the starting position in terms of piece names.
CYAN: str = "CYAN"
MAGENTA: str = "MAGENTA"


class PieceType(str, Enum):
    """The six piece types, each carrying its own movement metadata.

    Subclasses :class:`str`, so ``PieceType.PING == "PING"`` holds and a
    member works anywhere a plain string piece name did before.

    Attributes:
        abbr: Two-letter code used in move notation (e.g. ``"PI"``).
        move_range: How far the piece slides when moving. ``0`` for pieces
            that do not slide -- the Exploit jumps, the Kernel steps once.
        capture_range: How far the piece captures *by moving* along its
            capture directions. Only the Ping uses this.
        shoot_range: How far the piece shoots without moving. Only the Trace
            uses this; the Daemon shoots at range 1, handled directly.
        sacrifice_rank: Bodyguard order -- the lowest-ranked adjacent guard
            is given up first to protect a Kernel.
    """

    def __new__(
        cls,
        value: str,
        abbr: str,
        move_range: int,
        capture_range: int,
        shoot_range: int,
        sacrifice_rank: int,
    ) -> PieceType:
        """Build a member that is also the piece's canonical string value."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.abbr = abbr
        obj.move_range = move_range
        obj.capture_range = capture_range
        obj.shoot_range = shoot_range
        obj.sacrifice_rank = sacrifice_rank
        return obj

    # (value, abbr, move_range, capture_range, shoot_range, sacrifice_rank)
    # Declaration order is the observation plane order -- do not reorder.
    PING: PieceType = "PING", "PI", 2, 2, 0, 0
    DAEMON: PieceType = "DAEMON", "DE", 3, 0, 0, 3
    EXPLOIT: PieceType = "EXPLOIT", "EX", 0, 0, 0, 1
    TRACE: PieceType = "TRACE", "TR", 1, 0, 3, 2
    ICE: PieceType = "ICE", "IC", 2, 0, 0, 4
    KERNEL: PieceType = "KERNEL", "KE", 0, 0, 0, 5

    def encode_index(self) -> int:
        """The plane offset used for this piece type in board encoding."""
        return _ENCODE_INDEX[self]


_ENCODE_INDEX: dict[PieceType, int] = {
    piece: i for i, piece in enumerate(PieceType)
}


@dataclass
class Piece:
    """A single piece on the board.

    Attributes:
        piece_type: The :class:`PieceType` (e.g. ``PieceType.PING``).
        team: Either ``CYAN`` or ``MAGENTA``.
        has_moved: ``True`` once the piece has moved at least once.

            No rule reads this flag -- it is kept because
            :class:`~colony_one.board.ActionRecord` saves and restores it,
            so that undo is an exact inverse.
    """

    piece_type: PieceType
    team: str
    has_moved: bool = False

    @property
    def abbr(self) -> str:
        """Two-letter code used in move notation (e.g. ``"PI"``)."""
        return self.piece_type.abbr

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return f"Piece({self.team[:1]}:{self.abbr})"
