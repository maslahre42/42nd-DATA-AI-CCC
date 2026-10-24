"""Core action types for Colony: One.

These are the shared vocabulary of the whole package, so they live at the
protocol level rather than inside any one rule module. ``rules``, ``encoding``
and ``env`` all speak in terms of :class:`Action`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Position = tuple[int, int]
"""A board square as ``(row, col)``, both 0-indexed, row 0 at the top."""

ActionKind = Literal["move", "shoot"]
"""Which of the two kinds of play an action is."""


@dataclass(frozen=True)
class Action:
    """A single action a piece plays.

    A piece "moves" or "shoots"; both are plays, so they share one type.

    ``kind="move"`` relocates the piece to ``to_pos``, capturing any enemy
    piece standing there. ``kind="shoot"`` removes the piece on ``to_pos``
    while the shooter stays on ``from_pos``.

    Target geometry and occupancy uniquely determine which kind applies, so
    there is never more than one action for a given ``(from_pos, to_pos)``
    pair. That invariant is what makes ``(from, to)`` a sufficient identifier
    in move notation and in the UI.
    """

    from_pos: Position
    to_pos: Position
    kind: ActionKind
