"""PUCT Monte-Carlo tree search (AlphaZero style), framework-neutral.

This is the search the tournament referee runs. It is published so you can
reproduce the referee's behaviour exactly on your own machine before
submitting.

**You do not have to use it.** Your model only has to satisfy the contract in
``docs/submission.md``: one input, two outputs. The referee wraps whatever you
submit in this search. Train however you like -- AlphaZero, PPO, DQN, or by
hand -- and see ``docs/submission.md`` for how each maps onto the two outputs.

The evaluator is a plug-in
--------------------------
Everything below depends on a single callable:

.. code-block:: python

    def evaluate(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''(B, 16, 8, 8) float32 -> ((B, 4096) logits, (B,) values)'''

No ONNX, no framework at all is imported here -- just NumPy. Pass
:class:`~colony_one.onnx_runner.OnnxEvaluator` (what the referee uses), a
thin wrapper around your own network, or a stub for testing.

Batched evaluation
------------------
Search runs in two phases. A descent (:func:`_descend`) walks to an unexpanded
leaf, encodes the position there and *stops* without calling the evaluator,
recording the path it took. Once ``MCTSConfig.batch_size`` leaves are
collected they go through the evaluator in a single call, after which priors
are set and values are backed up along the recorded paths.

So that concurrent descents in one batch do not all take the same path, each
traversed edge carries a *virtual loss* (``MCTSConfig.virtual_loss``) that is
released during backup. With ``batch_size == 1`` the behaviour is identical to
a purely sequential search.

The referee does not play deterministically
-------------------------------------------
A pure PUCT search is a function of the position, so two fixed models would
play one identical game however many times you asked for -- a 100-game match
would measure a single line of play, repeated. An **opening warmup** prevents
that. Over its window (:attr:`MCTSConfig.warmup_actions`, 8 actions = the
first two rounds by default) two things are in force:

* **Root sampling.** The root action is drawn by visit count at
  :func:`warmup_temperature` -- four actions each at 1.0, 0.5.
* **Root Dirichlet noise** (:func:`_set_priors`) perturbs the root priors.

Past the window both switch off together and the referee plays greedily, at
the full strength of the model, for every position that actually decides the
game. It is on by default, so a local :func:`search_action` and a tournament
game are the same code path with the same settings.

Both draw from the ``rng`` you pass in, so a match stays reproducible when you
want it to be: same seeds, same games.

Neither is a lever a submission can pull. The noise is mixed into the priors
*after* they are normalised, so its share is fixed at ``dirichlet_eps`` no
matter how extreme the logits were, and the temperature samples *visit
counts*, which the simulation budget bounds. A model cannot scale its way out
of the opening randomisation by shouting.

Turn structure
--------------
The search carries the same turn logic as the environment: one turn is up to
``MAX_ACTION_PER_TURN`` (2) actions by the same team, the value perspective
flips only at a real turn change, and at a turn change (:func:`_turn_end`) the
bare-Kernels endgame and the firewall shrink are resolved exactly as
:meth:`~colony_one.env.ColonyOneEnv._end_turn` resolves them -- including
the one-turn delay on the endgame and the rule that the shrink only fires
after Magenta's turn.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from colony_one import constants, encoding
from colony_one.action import Action
from colony_one.board import Board
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team


class Evaluator(Protocol):
    """What the search needs from a model.

    Any object with this ``__call__`` works -- including a plain function.
    """

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions.

        Args:
            planes: ``(B, 16, 8, 8)`` float32, canonical view (see
                :mod:`colony_one.encoding`).

        Returns:
            ``(logits, values)`` where ``logits`` is ``(B, 4096)`` float32 of
            **raw, unnormalised** action scores and ``values`` is ``(B,)``
            float32 in ``[-1, 1]``, from the perspective of the team to move.
        """


EvaluatorFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _opponent(team: str) -> str:
    return constants.MAGENTA if team == constants.CYAN else constants.CYAN


class Node:
    """A node in the search tree.

    Statistics live on the *edges*: ``P``/``N``/``W`` are keyed by action
    index. ``VL`` counts unresolved descents through an edge (virtual loss),
    and ``pending`` marks a node already queued for evaluation this round.

    A node carries no hidden game state: beyond the board, the rules need only
    ``team``/``actions_left``/``round_number`` -- exactly what the observation
    encodes.
    """

    __slots__ = (
        "team", "actions_left", "round_number",
        "P", "N", "W", "VL", "children", "expanded", "pending", "leaf_value",
    )

    def __init__(self, team: str, actions_left: int, round_number: int) -> None:
        self.team = team
        self.actions_left = actions_left
        self.round_number = round_number
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.VL: dict[int, int] = {}
        self.children: dict[int, Node] = {}
        self.expanded = False
        self.pending = False
        self.leaf_value = 0.0


@dataclass
class MCTSConfig:
    """Search parameters.

    The tournament referee's exact settings are published in
    ``docs/submission.md``; the defaults here are the same ones.
    """

    simulations: int = 300
    c_puct: float = 1.5

    # Root exploration noise, mixed into the priors *after* they are
    # normalised (see :func:`_set_priors`). It runs for the duration of the
    # opening warmup and not one action longer: its job is to vary the
    # opening, not to dilute the positions that actually decide the game.
    add_root_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.15

    # The opening warmup: one temperature per block of ``warmup_step_actions``
    # actions, then greedy play for the rest of the game. See
    # :func:`warmup_temperature`. A block is one full round (2 actions per
    # team), so both teams always search at the same temperature.
    #
    # This one window governs the root noise as well, so a self-play loop that
    # wants exploration for longer widens the warmup rather than juggling two
    # separate schedules.
    warmup_temperatures: tuple[float, ...] = (1.0, 0.5)
    warmup_step_actions: int = 4

    @property
    def warmup_actions(self) -> int:
        """Length of the warmup in actions; ``0`` disables it entirely."""
        return len(self.warmup_temperatures) * max(0, self.warmup_step_actions)

    # Leaves collected before one evaluator call. 1 == exactly sequential
    # search. On CPU the optimum is usually 8-32; it is worth measuring.
    batch_size: int = 8

    # Temporary penalty per open edge so that concurrent descents diverge.
    # In Q units, so sensible values are in [0.5, 3.0].
    virtual_loss: float = 1.0

    # Adaptive simulation budget: instead of searching every root position
    # with a fixed ``simulations``, scale with the number of legal actions at
    # the root. ``simulations`` stays the upper bound. Off by default.
    adaptive_simulations: bool = False
    sim_multiplier: float = 15.0
    sim_floor: int = 150


# ---------------------------------------------------------- opening warmup
def move_number(team: str, actions_left: int, round_number: int) -> int:
    """The 1-based action number of the position being searched.

    Counts every action of the game, both teams together, so it lines up with
    :attr:`ColonyOneEnv.action <colony_one.env.ColonyOneEnv.action>` as read
    *before* the step that plays it: round 1 is actions 1-4 (Cyan, Cyan,
    Magenta, Magenta), round 2 is actions 5-8, and so on.

    The search never receives this counter -- it is reconstructed from the
    three scalars the observation already encodes, so nothing extra has to be
    threaded through a caller.
    """
    per_turn = constants.MAX_ACTION_PER_TURN
    completed_rounds = (round_number - 1) * 2 * per_turn
    team_offset = 0 if team == constants.CYAN else per_turn
    return completed_rounds + team_offset + (per_turn - actions_left) + 1


def warmup_temperature(
    move: int,
    temperatures: Sequence[float] = (1.0, 0.5),
    step_actions: int = 4,
) -> float:
    """The referee's root-sampling temperature on action number ``move``.

    A step function, not a smooth decay: ``temperatures[0]`` for the first
    ``step_actions`` actions, ``temperatures[1]`` for the next block, and so
    on, then ``0`` -- greedy play -- for the rest of the game. At the defaults
    that is four actions each at 1.0 and 0.5, greedy from action 9 on.

    A block is deliberately one full round. Both of a team's actions and both
    of the opponent's fall in the same block, so the two sides are randomised
    equally; a per-action decay would hand whichever team moves first a
    systematically hotter opening.

    The warmup exists because the referee is otherwise a pure function of the
    position: two fixed models would play the same game every time, and a
    100-game match would measure a single line of play repeated 100 times.
    Root noise (:func:`_set_priors`) runs over exactly the same window, and
    together they make repeated matches an actual sample.
    """
    if move < 1 or step_actions < 1:
        return 0.0
    block = (move - 1) // step_actions
    if block >= len(temperatures):
        return 0.0
    return float(temperatures[block])


@dataclass
class _Pending:
    """A collected leaf awaiting its evaluation."""

    node: Node
    planes: np.ndarray
    legal: list[Action]
    sign: int
    path: list[tuple[Node, int, int]] = field(default_factory=list)


# --------------------------------------------------------------- expansion
def _set_priors(
    node: Node,
    legal: list[Action],
    logits: np.ndarray,
    value: float,
    dirichlet: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Turn a logit vector into edge priors.

    Only *legal* indices are read, and the softmax is taken over those alone.
    Illegal actions therefore receive no probability mass at all, whatever the
    model emitted for them -- masking is structural, not a post-hoc filter. A
    model that never learns to suppress illegal actions is not penalised, and
    one that does is not rewarded twice.

    **Noise is mixed in after normalisation, and that ordering is load-bearing.**
    Added to the raw logits instead, Dirichlet noise would be worth whatever
    the model's own scale made it worth: a model emitting logits of +-1000
    would drown it completely and keep a one-hot prior, and the exploration
    the referee mandates in the opening would apply to everyone except the
    models most able to exploit its absence. Mixing into the normalised
    distribution instead makes the noise's share exactly ``dirichlet_eps``
    whatever the logits were, so no scaling trick can buy a model its way out.
    """
    node.expanded = True
    node.pending = False

    if not legal:
        node.leaf_value = value
        return

    indices = [encoding.action_index(a, node.team) for a in legal]
    legal_logits = np.asarray(logits, dtype=np.float64)[indices]
    # Subtract the max before exponentiating: standard guard against
    # overflow for models that emit large logits.
    shifted = legal_logits - np.max(legal_logits)
    exps = np.exp(shifted)
    probs = exps / exps.sum()

    if dirichlet is not None:
        alpha, eps = dirichlet
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.dirichlet([alpha] * len(indices))
        probs = (1.0 - eps) * probs + eps * noise

    for idx, p in zip(indices, probs, strict=True):
        node.P[idx] = float(p)
        node.N[idx] = 0
        node.W[idx] = 0.0


def _expand_root(
    node: Node,
    board: Board,
    evaluate: EvaluatorFn,
    dirichlet: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Evaluate the root on its own (once per search, not batchable)."""
    legal = legal_actions_for_team(board, node.team)
    planes = encoding.encode_board(
        board, node.team, node.actions_left, node.round_number
    )
    logits_batch, values = evaluate(planes[None, ...])
    value = float(values[0])
    _set_priors(node, legal, logits_batch[0], value, dirichlet, rng)
    return value


# --------------------------------------------------------------- selection
def _select_action(node: Node, c_puct: float, virtual_loss: float = 0.0) -> int | None:
    """PUCT: ``argmax_a Q(a) + c_puct * P(a) * sqrt(sum N) / (1 + N(a))``.

    Open descents (``node.VL``) count as visits with a losing value, so that
    concurrent descents take different paths.
    """
    if not node.P:
        return None

    total = sum(node.N.values()) + sum(node.VL.values())
    # ``total + 1`` rather than ``total``: with sqrt(0) == 0 the U term would
    # be 0 for every action on a node's very first visit, so selection would
    # ignore the policy entirely and the first action in dict order would win.
    # This way the highest prior wins the first visit.
    sqrt_total = math.sqrt(total + 1)

    best_score = -float("inf")
    best_action: int | None = None
    for a, p in node.P.items():
        vl = node.VL.get(a, 0)
        n = node.N[a] + vl
        q = (node.W[a] - virtual_loss * vl) / n if n > 0 else 0.0
        u = c_puct * p * sqrt_total / (1 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


# ------------------------------------------------------------------ descent
def _descend(
    node: Node,
    board: Board,
    sign: int,
    c_puct: float,
    virtual_loss: float,
    path: list[tuple[Node, int, int]],
) -> tuple[float | None, int, _Pending | None]:
    """Descend to a leaf without calling the evaluator.

    ``sign`` is the perspective relative to the root: ``+1`` the root team's
    view, ``-1`` the opponent's. It flips only at a real turn change
    (:func:`_turn_end`).

    ``path`` is filled with ``(node, action, sign)`` per traversed edge.

    Returns:
        Exactly one of:

        * ``(value, sign_of_value, None)`` -- the result is already known
          (terminal, or a cached ``leaf_value``) and can be backed up directly.
        * ``(None, sign, pending)`` -- a leaf awaiting evaluation.

        ``(None, sign, None)`` means: abort, this leaf is already queued this
        round.
    """
    if not node.expanded:
        if node.pending:
            return None, sign, None  # already queued this round
        # Both board-dependent steps happen here, while the board still holds
        # the leaf position; it may be undone afterwards.
        legal = legal_actions_for_team(board, node.team)
        planes = encoding.encode_board(
            board, node.team, node.actions_left, node.round_number
        )
        node.pending = True
        return None, sign, _Pending(node, planes, legal, sign, list(path))

    action_index = _select_action(node, c_puct, virtual_loss)
    if action_index is None:
        return node.leaf_value, sign, None

    path.append((node, action_index, sign))

    # The index carries only (from, to); the kind is implied by the position,
    # so it is resolved against the board we are standing on.
    action = encoding.decode_action(action_index, node.team, board)
    record = board.apply_action(action)
    try:
        if (
            record.captured_piece is not None
            and record.captured_piece.piece_type is PieceType.KERNEL
        ):
            # Capturing a Kernel ends the game at once: a win for node.team.
            return 1.0, sign, None

        # Kernel bodyguard (see Board._pick_guard): an attack on a guarded
        # Kernel kills a guard instead, so the state reached by this action is
        # not the state its coordinates suggest. The child is therefore not
        # cached under ``action_index`` -- guarding against a future rules
        # change reintroducing non-determinism here. Harmless today: it costs
        # one extra node in a rare case.
        bodyguard_redirect = (
            record.actual_target_pos is not None
            and record.actual_target_pos != action.to_pos
        )

        next_actions_left = node.actions_left - 1
        if next_actions_left > 0:
            # Same turn continues -> perspective unchanged.
            child = _child(
                node, action_index, node.team, next_actions_left,
                node.round_number, cache=not bodyguard_redirect,
            )
            return _descend(child, board, sign, c_puct, virtual_loss, path)

        return _turn_end(
            node, board, action_index, sign, c_puct, virtual_loss, path,
            cache_child=not bodyguard_redirect,
        )
    finally:
        board.undo_action(record)


def _child(
    node: Node,
    action_index: int,
    team: str,
    actions_left: int,
    round_number: int,
    cache: bool = True,
) -> Node:
    """The child reached by ``action_index``, reused from ``node`` if cached.

    ``cache=False`` builds a throwaway child for actions whose resulting state
    is not the one their coordinates suggest (see the bodyguard redirect).
    """
    if cache:
        child = node.children.get(action_index)
        if child is not None:
            return child

    child = Node(team, actions_left, round_number)
    if cache:
        node.children[action_index] = child
    return child


def _turn_end(
    node: Node,
    board: Board,
    action_index: int,
    sign: int,
    c_puct: float,
    virtual_loss: float,
    path: list[tuple[Node, int, int]],
    cache_child: bool = True,
) -> tuple[float | None, int, _Pending | None]:
    """Handle the end of a turn, mirroring ``ColonyOneEnv._end_turn``.

    In order: resolve the bare-Kernels endgame, then -- only after Magenta's
    turn, and only in a shrink round -- apply the firewall. Values are returned
    in ``node.team``'s perspective, which is what ``sign`` denotes.
    """
    # Bare-Kernels endgame: a pure function of the board and the team that just
    # acted, so the search reads it straight off the position.
    if board.only_kernels_remain():
        winner = board.kernel_endgame_winner(acting_team=node.team)
        return (1.0 if winner == node.team else -1.0), sign, None

    firewall_record = None
    if node.team == constants.MAGENTA and node.round_number in constants.SHRINK_ROUNDS:
        firewall_record = board.apply_firewall_shrink()

    try:
        if firewall_record is not None:
            killed_kernels = [
                p for _, _, p in firewall_record.killed
                if p.piece_type is PieceType.KERNEL
            ]
            if killed_kernels:
                if len(killed_kernels) == 2:
                    # Both Kernels burn at once -- Magenta takes the tiebreak.
                    winner = constants.MAGENTA
                else:
                    winner = _opponent(killed_kernels[0].team)
                return (1.0 if winner == node.team else -1.0), sign, None

        # A round completes when Magenta -- who moves second -- has finished.
        next_round = (
            node.round_number + 1 if node.team == constants.MAGENTA
            else node.round_number
        )
        child = _child(
            node, action_index, _opponent(node.team),
            constants.MAX_ACTION_PER_TURN, next_round, cache=cache_child,
        )
        # Real turn change -> perspective flips.
        return _descend(child, board, -sign, c_puct, virtual_loss, path)
    finally:
        if firewall_record is not None:
            board.undo_firewall_shrink(firewall_record)


# ------------------------------------------------------- virtual loss/backup
def _apply_virtual_loss(path: list[tuple[Node, int, int]]) -> None:
    for node, action, _ in path:
        node.VL[action] = node.VL.get(action, 0) + 1


def _backup(
    path: list[tuple[Node, int, int]],
    value: float,
    value_sign: int,
    release_virtual_loss: bool,
) -> None:
    """Propagate ``value`` (in perspective ``value_sign``) along ``path``.

    Each path entry records its own perspective, so the conversion is just the
    product of the two signs: same perspective adds ``+value``, flipped adds
    ``-value``.
    """
    for node, action, node_sign in path:
        node.N[action] = node.N.get(action, 0) + 1
        node.W[action] = node.W.get(action, 0.0) + value * node_sign * value_sign
        if release_virtual_loss:
            remaining = node.VL.get(action, 0) - 1
            if remaining > 0:
                node.VL[action] = remaining
            else:
                node.VL.pop(action, None)


# ---------------------------------------------------------------------- run
def run(
    board: Board,
    team: str,
    actions_left: int,
    round_number: int,
    evaluate: EvaluatorFn,
    mcts_config: MCTSConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Node:
    """Run simulations from the root and return the root node.

    ``board`` is mutated during the search and restored by undo, so it is
    unchanged on return. Visit counts end up in ``root.N``.

    Args:
        board: Position to search from.
        team: Team to move.
        actions_left: Actions remaining in the current turn (1 or 2).
        round_number: Current round, 1-based.
        evaluate: See :class:`Evaluator`.
        mcts_config: Search parameters; defaults to :class:`MCTSConfig`.
        rng: Generator for the root Dirichlet noise. Only used while the
            warmup is running and ``add_root_noise`` is set -- which it is by
            default; pass one to make a search reproducible.

    Returns:
        The root :class:`Node`.
    """
    mcts_config = mcts_config or MCTSConfig()
    root = Node(team, actions_left, round_number)
    # The noise shares the warmup's window: past it the referee wants the
    # strongest play the model can give, not a slice of a random prior.
    in_warmup = (
        move_number(team, actions_left, round_number) <= mcts_config.warmup_actions
    )
    dirichlet = (
        (mcts_config.dirichlet_alpha, mcts_config.dirichlet_eps)
        if mcts_config.add_root_noise and in_warmup else None
    )
    _expand_root(root, board, evaluate, dirichlet=dirichlet, rng=rng)

    batch_size = max(1, mcts_config.batch_size)
    virtual_loss = mcts_config.virtual_loss
    if mcts_config.adaptive_simulations:
        remaining = adaptive_simulation_budget(
            len(root.P), actions_left, mcts_config.sim_multiplier,
            mcts_config.sim_floor, mcts_config.simulations,
        )
    else:
        remaining = mcts_config.simulations

    while remaining > 0:
        pending: list[_Pending] = []
        done_this_round = 0

        try:
            for _ in range(min(batch_size, remaining)):
                path: list[tuple[Node, int, int]] = []
                value, value_sign, leaf = _descend(
                    root, board, +1, mcts_config.c_puct, virtual_loss, path
                )

                if leaf is not None:
                    _apply_virtual_loss(leaf.path)
                    pending.append(leaf)
                    done_this_round += 1
                elif value is not None:
                    # Terminal or known leaf_value: no evaluator call needed.
                    _backup(path, value, value_sign, release_virtual_loss=False)
                    done_this_round += 1
                else:
                    # Leaf already queued -> end the round here, otherwise the
                    # same position would be expanded twice.
                    break

            if pending:
                planes = np.stack([p.planes for p in pending])
                logits_batch, values = evaluate(planes)
                for leaf, logits, value in zip(
                    pending, logits_batch, values, strict=True
                ):
                    value = float(value)
                    _set_priors(leaf.node, leaf.legal, logits, value)
                    _backup(leaf.path, value, leaf.sign, release_virtual_loss=True)
        finally:
            # Safety net: no node stays marked "queued" if evaluation raised.
            for leaf in pending:
                leaf.node.pending = False

        if done_this_round == 0:
            break  # root has no actions -- avoid an infinite loop
        remaining -= done_this_round

    return root


def adaptive_simulation_budget(
    branching_factor: int, actions_left: int, multiplier: float, floor: int, cap: int
) -> int:
    """Scale the simulation budget with branching factor and ``actions_left``.

    The ``actions_left`` factor corrects an asymmetry between the two actions
    of a turn. At ``actions_left == 1`` (the last action) a single simulation
    already reaches a real turn change, so the value at the new leaf is
    immediately meaningful. At ``actions_left == 2`` the same simulation lands
    in an *unfinished* position -- the same player still owes a second action
    -- and a root child must be visited several times before its own second
    action is searched at all. Without this factor, first actions would be
    systematically under-searched at equal branching.

    ``cap`` stays the budget for the most branching positions: adaptivity only
    moves *below* that maximum, never above. ``floor`` keeps very narrow
    endgames from being fobbed off with an absurdly small search.
    """
    if branching_factor <= 0:
        return floor
    scaled = round(multiplier * branching_factor * actions_left)
    return max(floor, min(cap, scaled))


def visit_distribution(root: Node) -> tuple[list[int], list[int]]:
    """``(action_indices, visit_counts)``, sorted by index."""
    items = sorted(root.N.items())
    return [a for a, _ in items], [n for _, n in items]


def policy_target(root: Node) -> np.ndarray:
    """The full visit-count policy target, of size :data:`encoding.NUM_ACTIONS`.

    This is the AlphaZero policy training target: search visits are a
    strictly better policy than the raw network output that produced them.
    """
    target = np.zeros(encoding.NUM_ACTIONS, dtype=np.float32)
    indices, counts = visit_distribution(root)
    total = sum(counts)
    if total == 0:
        return target
    for idx, n in zip(indices, counts, strict=True):
        target[idx] = n / total
    return target


def select_action_index(
    root: Node, temperature: float = 0.0, rng: np.random.Generator | None = None
) -> int:
    """Pick a root action by visit-count temperature.

    ``temperature <= 0`` picks greedily (most visits) -- what the referee does
    once the opening warmup is over. Otherwise actions are sampled
    proportional to ``visits ** (1 / temperature)``, which is what varies the
    referee's openings and what self-play uses throughout.

    Sampling reads *visit counts*, not the model's logits, so its spread is
    bounded by the simulation budget however extreme the raw policy was.

    Raises:
        ValueError: The root has no legal actions.
    """
    indices, counts = visit_distribution(root)
    if not indices:
        raise ValueError("root node has no legal actions")

    counts_arr = np.asarray(counts, dtype=np.float64)
    if temperature <= 1e-3:
        return indices[int(np.argmax(counts_arr))]

    weights = counts_arr ** (1.0 / temperature)
    total = weights.sum()
    if total <= 0:
        probs = np.full(len(indices), 1.0 / len(indices))
    else:
        probs = weights / total
    generator = rng if rng is not None else np.random.default_rng()
    return indices[int(generator.choice(len(indices), p=probs))]


def search_action(
    board: Board,
    team: str,
    actions_left: int,
    round_number: int,
    evaluate: EvaluatorFn,
    mcts_config: MCTSConfig | None = None,
    temperature: float | None = None,
    rng: np.random.Generator | None = None,
) -> Action | None:
    """Search and return the chosen action directly.

    This is the referee's entry point, and calling it the way the referee does
    -- config and temperature left at their defaults -- reproduces tournament
    play exactly. There is no separate "local" mode to get wrong.

    Args:
        temperature: Root sampling temperature. ``None`` (the default) follows
            :func:`warmup_temperature`, which is what the referee does. Pass a
            float to override the schedule -- ``0.0`` for greedy play from the
            first action, a constant above zero for self-play.
        rng: Source for the root noise and the root sampling, both of which
            draw from it. Pass one to make a game reproducible; leave it
            ``None`` and every call is independently random.

    Returns ``None`` if ``team`` has no legal action in this position.
    """
    if not legal_actions_for_team(board, team):
        return None
    mcts_config = mcts_config or MCTSConfig()
    if temperature is None:
        temperature = warmup_temperature(
            move_number(team, actions_left, round_number),
            mcts_config.warmup_temperatures,
            mcts_config.warmup_step_actions,
        )
    root = run(
        board, team, actions_left, round_number, evaluate, mcts_config, rng=rng,
    )
    action_index = select_action_index(root, temperature, rng=rng)
    return encoding.decode_action(action_index, team, board)
