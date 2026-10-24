"""Rule constants for Colony: One.

This module contains *only* constants that affect gameplay. Rendering values
(window size, colours, fonts, animation timings) live in the main application
and are deliberately absent here, so that this package stays free of any
graphics dependency.

Piece identities and their movement metadata are **not** here: they live on
:class:`~colony_one.pieces.PieceType`, which is the single source of truth
for anything piece-specific. This module holds the rest of the game
configuration: teams, board, firewall timing, and the starting position.

Every value below is authoritative: the tournament referee uses these exact
numbers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
CYAN: str = "CYAN"
"""Player 1. Starts bottom-left and always moves first."""

MAGENTA: str = "MAGENTA"
"""Player 2. Starts top-right, and takes every tiebreak the rules contain."""

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
BOARD_SIZE: int = 8
"""Edge length of the (square) board."""

MAX_ACTION_PER_TURN: int = 2
"""Actions a team plays per turn. One ``step()`` is one *action*, not a turn."""

# ------------------------------------------------------------------
# Firewall
# ------------------------------------------------------------------
SHRINK_ROUNDS: tuple[int, ...] = (8, 16, 24, 26)
"""Rounds at the end of which the outermost open ring is locked.

The board contracts 8x8 -> 6x6 -> 4x4 -> 2x2 -> fully locked. Any piece
standing on a newly locked square is destroyed immediately, Kernels included.

A **round** is one turn for each team. The shrink listed for round ``R`` fires
at the *end* of ``R``, once Magenta (who moves second) has finished its turn.
So the whole of round 8 is played on an 8x8 board, and the board is 6x6 from
round 9 onwards.

Ring 3 -- the last one -- locks at the end of round 26, destroying everything
still standing, both Kernels included. That is what guarantees termination:
there is no ply cap anywhere in the rules, and none is needed.
"""

# ---------------------------------------------------------------------------
# Starting position
# ---------------------------------------------------------------------------
MAGENTA_SETUP: dict[tuple[int, int], str] = {
    (0, 7): "KERNEL",
    (1, 6): "ICE",
    (0, 5): "DAEMON",
    (2, 7): "TRACE",
    (1, 4): "EXPLOIT",
    (3, 6): "PING",
}
"""Magenta's starting squares. Cyan mirrors these through the board centre.

The values are :class:`~colony_one.pieces.PieceType` *values*, so
``PieceType(name)`` recovers the enum member for each one.
"""
