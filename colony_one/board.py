"""Board state, starting position, and the firewall for Colony: One.

Coordinate convention:
    ``board.grid[row][col]`` with ``row = 0`` at the **top** (Magenta's home)
    and ``row = 7`` at the **bottom** (Cyan's home). ``col = 0`` is the
    leftmost column.

    Display notation (``A1``-``H8``) maps ``col`` to a letter and shows
    ``BOARD_SIZE - row`` as the number, so row 7 is rank "1". That is
    presentation only -- every API here speaks ``(row, col)``.

Starting position:
    Each team owns exactly one of each piece type. Magenta is clustered in the
    top-right corner, Cyan is its 180-degree point reflection in the
    bottom-left. See :data:`constants.MAGENTA_SETUP`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from colony_one import constants
from colony_one.pieces import Piece, PieceType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from colony_one.action import Action


@dataclass
class ActionRecord:
    """Everything needed to undo an action and to know what it actually did.

    Produced by :meth:`Board.apply_action`, consumed by
    :meth:`Board.undo_action`.

    Attributes:
        from_pos: Where the acting piece stood.
        to_pos: The square the action was aimed at.
        moved_piece: The acting piece.
        had_moved_before: ``moved_piece.has_moved`` prior to the action.
        actual_target_pos: Where a piece actually died, or ``None`` if
            nothing died. This differs from ``to_pos`` when the Kernel
            bodyguard rule redirects the attack.
        captured_piece: The piece that died, or ``None``.
        moved: Whether the acting piece relocated. ``False`` for shots and
            for bodyguard-redirected attacks.
    """

    from_pos: tuple[int, int]
    to_pos: tuple[int, int]
    moved_piece: Piece
    had_moved_before: bool
    actual_target_pos: tuple[int, int] | None
    captured_piece: Piece | None
    moved: bool


@dataclass
class FirewallRecord:
    """Which ring was locked and which pieces it destroyed.

    Produced by :meth:`Board.apply_firewall_shrink`, consumed by
    :meth:`Board.undo_firewall_shrink`.
    """

    ring: int
    killed: list[tuple[int, int, Piece]]


class Board:
    """The 8x8 board: piece placement plus firewall lock state."""

    def __init__(self) -> None:
        """Create a board in the standard starting position."""
        self.grid: list[list[Piece | None]] = [
            [None for _ in range(constants.BOARD_SIZE)]
            for _ in range(constants.BOARD_SIZE)
        ]
        # Firewall state: True means locked (and therefore uninhabitable).
        self.locked: list[list[bool]] = [
            [False for _ in range(constants.BOARD_SIZE)]
            for _ in range(constants.BOARD_SIZE)
        ]
        self.reset()

    def reset(self) -> None:
        """Restore the starting position and clear all firewall locks."""
        last = constants.BOARD_SIZE - 1

        for row in range(constants.BOARD_SIZE):
            for col in range(constants.BOARD_SIZE):
                self.grid[row][col] = None
                self.locked[row][col] = False

        # Magenta top-right, Cyan as the point reflection bottom-left.
        for (row, col), piece_name in constants.MAGENTA_SETUP.items():
            piece_type = PieceType(piece_name)
            self.grid[row][col] = Piece(piece_type, constants.MAGENTA)
            self.grid[last - row][last - col] = Piece(piece_type, constants.CYAN)

    @classmethod
    def from_observation(cls, observation, team: str = constants.CYAN) -> Board:
        """Reconstruct a board from a ``(16, 8, 8)`` observation.

        Inverse of :func:`~colony_one.encoding.encode_board` for the
        piece-placement and locked planes. The observation is *team*
        perspective (planes 0-5 are the given team's pieces, 6-11 the
        opponent's, and for Magenta the board is 180-rotated), so ``team`` is
        required to place pieces back in absolute board coordinates.

        Useful for loading a position from a saved training batch, or for
        setting up a test position without hand-placing pieces.

        Note:
            The observation does not encode ``has_moved``, so reconstructed
            pieces always start with ``has_moved=False``. It does not encode
            ``actions_left`` or ``round_number`` in recoverable form either;
            those are scalars on the env, not the board.
        """
        import numpy as np

        from colony_one import encoding, pieces

        obs = np.asarray(observation, dtype=np.float32)
        expected = (encoding.NUM_PLANES, constants.BOARD_SIZE, constants.BOARD_SIZE)
        if obs.ndim != 3 or obs.shape != expected:
            raise ValueError(f"observation must be shape {expected}, got {obs.shape}")

        n = constants.BOARD_SIZE
        board = object.__new__(cls)
        board.grid = [[None for _ in range(n)] for _ in range(n)]
        board.locked = [[False for _ in range(n)] for _ in range(n)]

        enemy = constants.MAGENTA if team == constants.CYAN else constants.CYAN
        for row in range(n):
            for col in range(n):
                crow, ccol = encoding.canonical_pos((row, col), team)
                # Own planes (0..NUM_PIECE_TYPES-1), opponent planes next.
                for type_idx, piece_type in enumerate(pieces.PieceType):
                    if obs[type_idx, crow, ccol] > 0.5:
                        board.grid[row][col] = pieces.Piece(piece_type, team)
                        break
                    if obs[encoding.NUM_PIECE_TYPES + type_idx, crow, ccol] > 0.5:
                        board.grid[row][col] = pieces.Piece(piece_type, enemy)
                        break
                board.locked[row][col] = bool(
                    obs[encoding.LOCKED_PLANE, crow, ccol] > 0.5
                )

        return board

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        """Whether ``(row, col)`` lies on the board."""
        return 0 <= row < constants.BOARD_SIZE and 0 <= col < constants.BOARD_SIZE

    def piece_at(self, row: int, col: int) -> Piece | None:
        """The piece at ``(row, col)``, or ``None`` if empty or off-board."""
        if not self.in_bounds(row, col):
            return None
        return self.grid[row][col]

    def only_kernels_remain(self) -> bool:
        """Whether no piece other than a Kernel is left on the board.

        With only the two Kernels left neither side can force anything --
        there is no check or checkmate in this game, so a Kernel may freely
        step next to the enemy Kernel or capture it. Two Kernels can evade
        each other indefinitely, so without a special rule the game could
        never end. :meth:`kernel_endgame_winner` is that rule -- it names a
        winner immediately rather than letting the position stall.
        """
        for row in self.grid:
            for piece in row:
                if piece is not None and piece.piece_type is not PieceType.KERNEL:
                    return False
        return True

    def kernel_endgame_winner(self, acting_team: str) -> str | None:
        """Who wins the bare-Kernels endgame, decided by distance to centre.

        If a Kernel is in reach to the other Kernel, it loses,
        because it could be hit in the next round anyway. This is, fun fact,
        the only way for CYAN to win a both-Kernels-on-Firewall situation.

        Else, the one standing on the **smaller
        ring** -- closer to the centre of the board -- wins, because the
        firewall would eventually resolve the position that way anyway. Equal
        rings go to **Magenta**.

        The ring is the Chebyshev distance from the board centre, floored to
        the nearest integer. With an even board there is no single centre
        square, so the centre lies at ``3.5, 3.5`` and the four core squares
        all sit at ring 0 after flooring.

        Returns:
            ``CYAN``, ``MAGENTA``, or ``None`` if the position is not a bare-
            Kernels endgame at all.
        """
        if not self.only_kernels_remain():
            return None

        cyan_kernel_pos = None
        magenta_kernel_pos = None
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                piece = self.grid[row][col]
                if piece and piece.piece_type is PieceType.KERNEL:
                    if piece.team == constants.CYAN:
                        cyan_kernel_pos = (row, col)
                    else:
                        magenta_kernel_pos = (row, col)

        # A Kernel already gone means the game should have ended earlier;
        # Magenta takes it, consistent with every other tiebreak.
        if not cyan_kernel_pos or not magenta_kernel_pos:
            return constants.MAGENTA

        cyan_magenta_distance = max(abs(cyan_kernel_pos[0] - magenta_kernel_pos[0]),
                                    abs(cyan_kernel_pos[1] - magenta_kernel_pos[1]))

        if cyan_magenta_distance <= 2:
            if acting_team == constants.MAGENTA:
                return constants.CYAN
            else:
                return constants.MAGENTA

        board_size = len(self.grid)
        centre = (board_size - 1) / 2.0

        cyan_ring = int(
            max(abs(cyan_kernel_pos[0] - centre), abs(cyan_kernel_pos[1] - centre))
        )
        magenta_ring = int(
            max(abs(magenta_kernel_pos[0] - centre), abs(magenta_kernel_pos[1] - centre))
        )

        # Closer to the centre wins; ties go to Magenta.
        if cyan_ring < magenta_ring:
            return constants.CYAN
        return constants.MAGENTA

    def _neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """The up-to-8 on-board squares adjacent to ``pos``."""
        row, col = pos
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self.in_bounds(r, c):
                    result.append((r, c))
        return result

    def _pick_guard(
            self, kernel_pos: tuple[int, int], team: str
    ) -> tuple[int, int] | None:
        """The most expendable friendly non-Kernel piece adjacent to a Kernel.

        Returns ``None`` when the Kernel has no such neighbour, in which case
        it can be captured normally. Selection follows each piece type's
        :attr:`~colony_one.pieces.PieceType.sacrifice_rank` and is fully
        deterministic.
        """
        guards = [
            npos
            for npos in self._neighbors(kernel_pos)
            if (guard := self.grid[npos[0]][npos[1]]) is not None
               and guard.team == team
               and guard.piece_type is not PieceType.KERNEL
        ]
        if not guards:
            return None

        def guard_priority(pos: tuple[int, int]) -> int:
            piece = self.grid[pos[0]][pos[1]]
            assert piece is not None
            return piece.piece_type.sacrifice_rank

        return min(guards, key=guard_priority)

    # ------------------------------------------------------------------
    # Undoable action execution
    # ------------------------------------------------------------------
    def apply_action(self, action: Action) -> ActionRecord:
        """Execute an action and return a record that can undo it.

        Three cases, checked in this order:

        1. **Bodyguard redirect.** If the target square holds a Kernel that
           has at least one friendly non-Kernel neighbour, that guard dies
           instead. The Kernel survives and the attacker stays on
           ``from_pos`` -- even for a ``"move"`` action. This applies to
           shots as well as moves.
        2. **Shoot.** The piece on ``to_pos`` is removed; the shooter does
           not move.
        3. **Move.** The acting piece relocates to ``to_pos``, capturing
           whatever stood there.

        Because of case 1, ``to_pos`` is *not* reliably where the acting
        piece ends up. Use ``record.moved`` and ``record.actual_target_pos``.
        """
        fr, fc = action.from_pos
        tr, tc = action.to_pos
        piece = self.grid[fr][fc]
        assert piece is not None, "apply_action requires a piece on from_pos"
        had_moved_before = piece.has_moved

        target = self.grid[tr][tc]
        guard_pos: tuple[int, int] | None = None
        if target is not None and target.piece_type is PieceType.KERNEL:
            guard_pos = self._pick_guard(action.to_pos, target.team)

        if guard_pos is not None:
            gr, gc = guard_pos
            captured = self.grid[gr][gc]
            self.grid[gr][gc] = None
            return ActionRecord(
                from_pos=action.from_pos,
                to_pos=action.to_pos,
                moved_piece=piece,
                had_moved_before=had_moved_before,
                actual_target_pos=guard_pos,
                captured_piece=captured,
                moved=False,
            )

        captured = target
        if action.kind == "shoot":
            self.grid[tr][tc] = None
            return ActionRecord(
                from_pos=action.from_pos,
                to_pos=action.to_pos,
                moved_piece=piece,
                had_moved_before=had_moved_before,
                actual_target_pos=action.to_pos if captured is not None else None,
                captured_piece=captured,
                moved=False,
            )

        self.grid[tr][tc] = piece
        self.grid[fr][fc] = None
        piece.has_moved = True
        return ActionRecord(
            from_pos=action.from_pos,
            to_pos=action.to_pos,
            moved_piece=piece,
            had_moved_before=had_moved_before,
            actual_target_pos=action.to_pos if captured is not None else None,
            captured_piece=captured,
            moved=True,
        )

    def undo_action(self, record: ActionRecord) -> None:
        """Reverse an action previously applied by :meth:`apply_action`."""
        if record.moved:
            fr, fc = record.from_pos
            tr, tc = record.to_pos
            self.grid[fr][fc] = record.moved_piece
            self.grid[tr][tc] = record.captured_piece
        elif record.actual_target_pos is not None:
            ar, ac = record.actual_target_pos
            self.grid[ar][ac] = record.captured_piece

        record.moved_piece.has_moved = record.had_moved_before

    # ------------------------------------------------------------------
    # Firewall
    # ------------------------------------------------------------------
    def is_locked(self, row: int, col: int) -> bool:
        """Whether ``(row, col)`` is locked by the firewall.

        Off-board coordinates count as locked, which lets movement
        generators treat the board edge and the firewall identically.
        """
        if not self.in_bounds(row, col):
            return True
        return self.locked[row][col]

    @staticmethod
    def locked_cells_for_round(round_number: int) -> list[list[bool]]:
        """Reconstruct the lock matrix in force *during* ``round_number``.

        Useful for replaying a game from an action list without stepping the
        firewall. Note the strict ``<``: the shrink listed for round ``R``
        fires at the *end* of ``R``, so the whole of round ``R`` is still
        played on the larger board and ring ``R`` only counts as locked from
        round ``R + 1``.
        """
        n = constants.BOARD_SIZE
        locked: list[list[bool]] = [[False] * n for _ in range(n)]
        rings_locked = sum(1 for r in constants.SHRINK_ROUNDS if r < round_number)
        for ring in range(rings_locked):
            for c in range(ring, n - ring):
                locked[ring][c] = True
                locked[n - 1 - ring][c] = True
            for r in range(ring + 1, n - 1 - ring):
                locked[r][ring] = True
                locked[r][n - 1 - ring] = True
        return locked

    def next_open_ring(self) -> int:
        """Index of the outermost ring that is still open, or ``-1`` if none.

        Ring 0 is the outer edge, ring 1 the next one in, and so on.
        """
        n = constants.BOARD_SIZE
        for ring in range(n // 2):
            if not self.locked[ring][ring]:
                return ring
        return -1

    def is_on_ring(self, row: int, col: int, ring: int) -> bool:
        """Whether ``(row, col)`` lies on ring ``ring``."""
        n = constants.BOARD_SIZE
        return row == ring or row == n - 1 - ring or col == ring or col == n - 1 - ring

    @staticmethod
    def ring_cells(ring: int) -> list[tuple[int, int]]:
        """All squares of ``ring``, outermost ring being 0.

        Top and bottom edges are interleaved column by column, then the left
        and right columns.
        """
        n = constants.BOARD_SIZE
        cells: list[tuple[int, int]] = []
        for c in range(ring, n - ring):
            cells.append((ring, c))  # top edge
            cells.append((n - 1 - ring, c))  # bottom edge
        for r in range(ring + 1, n - 1 - ring):
            cells.append((r, ring))  # left edge
            cells.append((r, n - 1 - ring))  # right edge
        return cells

    def lock_outer_ring(self) -> list[tuple[int, int, Piece]]:
        """Lock the outermost open ring and destroy everything on it.

        Returns the destroyed pieces as ``(row, col, piece)`` in
        :meth:`ring_cells` order, or an empty list if the board is already
        fully locked.
        """
        record = self.apply_firewall_shrink()
        return record.killed if record is not None else []

    def apply_firewall_shrink(self) -> FirewallRecord | None:
        """Like :meth:`lock_outer_ring` but returns an undoable record.

        Returns ``None`` when every ring is already locked.
        """
        ring = self.next_open_ring()
        if ring < 0:
            return None

        killed: list[tuple[int, int, Piece]] = []
        for row, col in self.ring_cells(ring):
            self.locked[row][col] = True
            piece = self.grid[row][col]
            if piece is not None:
                self.grid[row][col] = None
                killed.append((row, col, piece))

        return FirewallRecord(ring=ring, killed=killed)

    def undo_firewall_shrink(self, record: FirewallRecord) -> None:
        """Reverse a shrink: unlock the ring and restore destroyed pieces."""
        for row, col in self.ring_cells(record.ring):
            self.locked[row][col] = False
        for row, col, piece in record.killed:
            self.grid[row][col] = piece
