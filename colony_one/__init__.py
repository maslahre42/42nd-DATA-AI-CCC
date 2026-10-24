"""Colony: One -- standalone game environment for tournament participants.

Quick start::

    from colony_one import ColonyOneEnv

    env = ColonyOneEnv(seed=0)
    obs = env.reset()                       # (16, 8, 8) float32

    while not env.terminated:
        actions = env.legal_actions()
        obs, reward, terminated, info = env.step(actions[0])

Note that :meth:`~colony_one.env.ColonyOneEnv.step` applies **one action**,
not a full turn. A turn is two actions, so the side to move usually does
not change between consecutive steps -- check ``env.current_team``.

See ``docs/submission.md`` for the ONNX model contract used at the tournament.
"""

from __future__ import annotations

from colony_one import constants, encoding, mcts
from colony_one.action import Action, ActionKind, Position
from colony_one.board import ActionRecord, Board, FirewallRecord
from colony_one.encoding import (
    NUM_ACTIONS,
    NUM_PLANES,
    action_index,
    decode_action,
    encode_board,
    index_to_positions,
    legal_action_mask,
)
from colony_one.env import ColonyOneEnv, IllegalActionError
from colony_one.mcts import MCTSConfig
from colony_one.pieces import Piece, PieceType
from colony_one.rules import legal_actions, legal_actions_for_team

__version__ = "1.0.0"

__all__ = [
    "Action",
    "ActionKind",
    "ActionRecord",
    "Board",
    "ColonyOneEnv",
    "FirewallRecord",
    "IllegalActionError",
    "MCTSConfig",
    "NUM_ACTIONS",
    "NUM_PLANES",
    "Piece",
    "PieceType",
    "Position",
    "action_index",
    "constants",
    "decode_action",
    "encode_board",
    "encoding",
    "index_to_positions",
    "legal_action_mask",
    "legal_actions",
    "legal_actions_for_team",
    "mcts",
    "__version__",
]
