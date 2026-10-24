"""Plain-text presentation of the board and status line.

All display concerns live here so the core environment stays free of layout
and formatting details.

Legend:
    ``cK``  a Cyan piece (lowercase team letter + first letter of the abbr)
    ``mK``  a Magenta piece
    ``##``  a firewall-locked square
    ``::``  a square on the ring that the firewall will lock at the end of
            the current round -- shown only during that round
    ``..``  an empty square
"""

from __future__ import annotations

from colony_one import constants
from colony_one.board import Board


def _next_shrink_round(round_number: int) -> int | None:
    """The upcoming round at whose end the firewall shrinks, or ``None``."""
    for s in constants.SHRINK_ROUNDS:
        if s >= round_number:
            return s
    return None


def render(
    board: Board,
    *,
    current_team: str,
    actions_left: int,
    round_number: int,
    action: int,
    terminated: bool,
    winner: str | None,
) -> str:
    """A plain-text board rendering, row 0 (Magenta's home) at the top."""
    lines = []
    # Warn about the fire line only during the round in which it will burn.
    next_shrink = _next_shrink_round(round_number)
    warn = next_shrink is not None and next_shrink == round_number
    next_warning_ring = board.next_open_ring() if warn else -1

    for row in range(constants.BOARD_SIZE):
        cells = []
        for col in range(constants.BOARD_SIZE):
            if board.locked[row][col]:
                cells.append("##")
                continue

            piece = board.grid[row][col]
            if piece is not None:
                prefix = "c" if piece.team == constants.CYAN else "m"
                cells.append(prefix + piece.abbr[0])
            elif next_warning_ring >= 0 and board.is_on_ring(row, col, next_warning_ring):
                cells.append("::")
            else:
                cells.append("..")
        rank = constants.BOARD_SIZE - row
        lines.append(f"{rank} " + " ".join(cells))

    files = "  ".join(chr(ord("A") + c) for c in range(constants.BOARD_SIZE))
    lines.append("  " + files)

    status = (
        f"{current_team} to move, {actions_left} action(s) left, "
        f"round {round_number}, action {action}"
    )
    if terminated:
        status = f"game over -- winner: {winner}"
    lines.append(status)
    return "\n".join(lines)
