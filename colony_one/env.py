"""The Colony: One environment.

A ROUND consists of 2 TURNS, one turn per team.
A TURN consists of 2 ACTIONS by the same team.

    ROUND 1: Cyan's turn (2 actions), then Magenta's turn (2 actions)
    ROUND 2: Cyan's turn (2 actions), then Magenta's turn (2 actions)
    ...

**Each call to** :meth:`ColonyOneEnv.step` **applies ONE ACTION**, not one
turn. After an action:

- ``actions_left`` decreases by 1
- if ``actions_left > 0``: ``current_team`` stays the same
- if ``actions_left == 0``: ``current_team`` switches, and ``round_number``
  increments once Magenta's turn has ended

Always read ``env.current_team`` and ``env.round_number`` after each
``step()``. Never assume alternation.

There is no action cap and no draw by exhaustion: the firewall locks the final
ring at the end of round 26, which destroys both Kernels and ends the game.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from colony_one import constants, renderer
from colony_one.action import Action
from colony_one.board import Board
from colony_one.encoding import NUM_ACTIONS, action_index, encode_board
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team

StepResult = tuple[np.ndarray, float, bool, dict[str, Any]]
"""What :meth:`ColonyOneEnv.step` returns: ``(obs, reward, terminated, info)``."""


class IllegalActionError(ValueError):
    """Raised when :meth:`ColonyOneEnv.step` is given an illegal action."""


class ColonyOneEnv:
    """A single game of Colony: One.

    Args:
        seed: Seed for the stochastic helpers (:meth:`sample_legal_action` and
            :meth:`sample_legal_action_index`). The game itself is fully
            deterministic.
        scene: Scene UUID selecting the starting board. Defaults to the
            reserved default UUID (standard opening); a custom UUID is loaded
            via ``Board.create_from_uuid``. Never ``None``.

    Attributes:
        board: The underlying :class:`~colony_one.board.Board`.
        current_team: Team to move, ``"CYAN"`` or ``"MAGENTA"``.
        actions_left: Actions remaining in the current turn (1 or 2).
        round_number: Current round, 1-based. A round ends once both teams
            have taken their turn.
        action: Current action number, 1-based.
        terminated: Whether the game has ended.
        winner: Winning team, or ``None`` while the game is running. Once
            ``terminated`` it always names a team -- there are no draws.
    """

    def __init__(
            self,
            seed: int | None = None,
            scene: np.ndarray | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._scene = scene
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Start a new game and return the initial observation."""
        if self._scene is None:
            self._set_scene_from_default_file()

        self.board = Board.from_observation(self._scene)
        self.current_team: str = constants.CYAN  # Cyan always moves first
        self.actions_left: int = constants.MAX_ACTION_PER_TURN
        self.round_number: int = 1
        self.action: int = 1
        self.terminated: bool = False
        self.winner: str | None = None
        return self.observation()

    def clone(self) -> ColonyOneEnv:
        """A deep, fully independent copy. Intended for tree search."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Observations and legal actions
    # ------------------------------------------------------------------
    def observation(self) -> np.ndarray:
        """The ``(16, 8, 8)`` float32 observation for the team to move."""
        return encode_board(
            self.board, self.current_team, self.actions_left, self.round_number
        )

    def legal_actions(self) -> list[Action]:
        """Legal actions as :class:`Action` objects, in engine order."""
        if self.terminated:
            return []
        return legal_actions_for_team(self.board, self.current_team)

    def legal_action_indices(self) -> np.ndarray:
        """Legal actions as canonical indices, shape ``(n,)`` int32."""
        return np.array(
            [action_index(a, self.current_team) for a in self.legal_actions()],
            dtype=np.int32,
        )

    def legal_mask(self) -> np.ndarray:
        """Boolean mask of shape ``(4096,)``, ``True`` at legal indices."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for action in self.legal_actions():
            mask[action_index(action, self.current_team)] = True
        return mask

    def sample_legal_action(self) -> Action:
        """A uniformly random legal action, using the env's seed."""
        actions = self.legal_actions()
        if not actions:
            raise ValueError("no legal actions available")
        return self._rng.choice(actions)

    def sample_legal_action_index(self) -> int:
        """A uniformly random legal action index, using the env's seed."""
        indices = self.legal_action_indices()
        if len(indices) == 0:
            raise ValueError("no legal actions available")
        return int(self._rng.choice(indices))

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def step(self, action: int | Action) -> StepResult:
        """Apply one action (one of the two that make up a turn).

        Args:
            action: A canonical action index, or an :class:`Action`.

        Returns:
            ``(observation, reward, terminated, info)``.

            ``reward`` is from the perspective of the team that just acted:
            ``+1.0`` if that team won, ``-1.0`` if it lost, ``0.0`` otherwise.

            ``observation`` is for the *new* team to move, which after the
            first action of a turn is still the same team.

        Raises:
            IllegalActionError: The action is not currently legal.
            RuntimeError: The game has already ended.
        """
        if self.terminated:
            raise RuntimeError("game is over; call reset() to start a new one")

        acting_team = self.current_team
        action = self._coerce_action(action)
        self.actions_left -= 1
        self.action += 1
        record = self.board.apply_action(action)

        # Capturing a Kernel wins immediately: no turn change, no firewall
        # check, no endgame check.
        captured = record.captured_piece
        if captured is not None and captured.piece_type is PieceType.KERNEL:
            return self._finish(winner=acting_team, acting_team=acting_team)

        if self.actions_left > 0:
            # Same team acts again; the turn is not over.
            return self._continue()

        return self._end_turn(acting_team)

    def _end_turn(self, acting_team: str) -> StepResult:
        """Hand over the turn, resolve the bare-Kernels endgame and firewall."""
        active_round_number = self.round_number
        # The round advances once Magenta -- who moves second -- is done.
        if self.current_team == constants.MAGENTA:
            self.round_number += 1
        self.current_team = (
            constants.MAGENTA if self.current_team == constants.CYAN else constants.CYAN
        )
        self.actions_left = constants.MAX_ACTION_PER_TURN

        # Bare-Kernels endgame.
        if self.board.only_kernels_remain():
            endgame_winner = self.board.kernel_endgame_winner(acting_team=acting_team)
            if endgame_winner is None:
                # Only reachable if only_kernels_remain() and
                # kernel_endgame_winner() disagree, which would be a bug.
                raise RuntimeError("bare-Kernels endgame did not name a winner")
            return self._finish(winner=endgame_winner, acting_team=acting_team)

        # The firewall shrinks at the end of a full round, i.e. only after
        # Magenta's turn, and is tested against the round that just finished.
        shrink_due = active_round_number in constants.SHRINK_ROUNDS
        if acting_team == constants.MAGENTA and shrink_due:
            killed = self.board.lock_outer_ring()
            killed_kernels = [p for _, _, p in killed if p.piece_type is PieceType.KERNEL]

            if killed_kernels:
                if len(killed_kernels) == 2:
                    # Both Kernels burn at once -- Magenta takes the tiebreak.
                    return self._finish(
                        winner=constants.MAGENTA, acting_team=acting_team
                    )
                piece = killed_kernels[0]
                winner = (
                    constants.MAGENTA if piece.team == constants.CYAN else constants.CYAN
                )
                return self._finish(winner=winner, acting_team=acting_team)

        return self._continue()

    def _coerce_action(self, action: int | Action) -> Action:
        """Validate an action and return it as an :class:`Action`."""
        legal = self.legal_actions()
        if not legal:
            raise IllegalActionError("no legal actions available")

        if isinstance(action, Action):
            if action not in legal:
                raise IllegalActionError(
                    f"{action!r} is not legal for {self.current_team}"
                )
            return action

        idx = int(action)
        if not 0 <= idx < NUM_ACTIONS:
            raise IllegalActionError(
                f"action index {idx} out of range [0, {NUM_ACTIONS})"
            )
        for candidate in legal:
            if action_index(candidate, self.current_team) == idx:
                return candidate
        raise IllegalActionError(
            f"action index {idx} is not legal for {self.current_team}"
        )

    def _continue(self) -> StepResult:
        """Package a non-terminal transition."""
        return self.observation(), 0.0, False, self._info(self.current_team)

    def _finish(self, winner: str, acting_team: str) -> StepResult:
        """Package a terminal transition, scoring from ``acting_team``'s view."""
        self.terminated = True
        self.winner = winner
        reward = 1.0 if winner == acting_team else -1.0
        return self.observation(), reward, True, self._info(acting_team)

    def _info(self, acting_team: str) -> dict[str, Any]:
        return {
            "acting_team": acting_team,
            "current_team": self.current_team,
            "actions_left": self.actions_left,
            "round_number": self.round_number,
            "action": self.action,
            "winner": self.winner,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def format_action(self, action: Action) -> str:
        """Format an action in chess-like notation (e.g. ``A8 -> B7``)."""
        fr, fc = action.from_pos
        tr, tc = action.to_pos
        from_coord = f"{chr(ord('A') + fc)}{constants.BOARD_SIZE - fr}"
        to_coord = f"{chr(ord('A') + tc)}{constants.BOARD_SIZE - tr}"
        return f"{from_coord} -> {to_coord}"

    def result_for(self, team: str) -> float:
        """Final result from ``team``'s perspective: ``+1`` or ``-1``.

        Raises:
            RuntimeError: The game is still in progress.
        """
        if not self.terminated:
            raise RuntimeError("game is still in progress")
        return 1.0 if self.winner == team else -1.0

    def render(self) -> str:
        """A plain-text board rendering (see :func:`renderer.render`)."""
        return renderer.render(
            self.board,
            current_team=self.current_team,
            actions_left=self.actions_left,
            round_number=self.round_number,
            action=self.action,
            terminated=self.terminated,
            winner=self.winner,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        if self.terminated:
            return f"<ColonyOneEnv terminated winner={self.winner}>"
        return (
            f"<ColonyOneEnv {self.current_team} to move, "
            f"round={self.round_number} action={self.action}>"
        )

    def _set_scene_from_default_file(self) -> None:
        scene_path = Path(__file__).with_name("scene.json")
        with scene_path.open(encoding="utf-8") as scene_file:
            self._scene = np.asarray(json.load(scene_file), dtype=np.float32)
