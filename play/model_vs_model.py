#!/usr/bin/env python3
"""Play two ONNX models under referee MCTS.

    python play/model_vs_model.py --model-a mine.onnx --model-b theirs.onnx --games 50
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

CYAN = constants.CYAN
MAGENTA = constants.MAGENTA


class SearchAgent:
    """An ONNX model driven by referee MCTS."""

    def __init__(self, model_path: Path, simulations: int, seed: int = 0) -> None:
        self.evaluator = OnnxEvaluator(model_path)
        self.config = mcts.MCTSConfig(simulations=simulations, batch_size=8)
        self.rng = np.random.default_rng(seed)
        self.name = model_path.name

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, board, team, actions_left, round_number):
        return mcts.search_action(
            board, team, actions_left, round_number,
            self.evaluator, self.config, rng=self.rng,
        )


def play_game(cyan_agent, magenta_agent, render: bool = False
              ) -> tuple[str | None, int]:
    """Play one game and return (winner, actions)."""
    env = ColonyOneEnv()

    if render:
        print(env.render())
        print()

    while not env.terminated:
        agent = cyan_agent if env.current_team == CYAN else magenta_agent
        action = agent.select(
            env.board, env.current_team, env.actions_left, env.round_number
        )
        if action is None:
            raise RuntimeError("agent returned no action in a playable position")

        team, kind = env.current_team, action.kind
        number = env.action
        env.step(action)

        if render:
            formatted = env.format_action(action)
            print(f"action {number}: {team} {kind} {formatted}")
            print(env.render())
            print()

    return env.winner, env.action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-a", type=Path, required=True,
                        help="the first ONNX model")
    parser.add_argument("--model-b", type=Path, required=True,
                        help="the second ONNX model")
    parser.add_argument("--games", type=int, default=1,
                        help="number of games to play (default: 1)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--render", action="store_true",
                        help="print the board after every action")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for per-game reseeding "
                             "(default: derived from the current time)")
    args = parser.parse_args()

    if args.render and args.games > 1:
        print("note: --render with multiple games is very verbose\n")

    base_seed = args.seed if args.seed is not None else secrets.randbits(32)
    print(f"base seed: {base_seed}")

    agent_a = SearchAgent(args.model_a, args.simulations, seed=base_seed)
    agent_b = SearchAgent(args.model_b, args.simulations, seed=base_seed + 1)
    opponent_name = agent_b.name

    print(f"{agent_a.name} (A)  vs  {opponent_name} (B)")
    print(f"{args.games} game(s), sides swapped every other game\n")

    results: Counter[str] = Counter()
    total_actions = 0
    start = time.perf_counter()

    for game in range(args.games):
        # Swap sides so a first-move advantage cannot skew the score.
        swap = game % 2 == 1
        agent_a.reseed(base_seed + game * 2)
        agent_b.reseed(base_seed + game * 2 + 1)
        cyan_agent, magenta_agent = (agent_b, agent_a) if swap else (agent_a, agent_b)

        winner, actions = play_game(cyan_agent, magenta_agent, render=args.render)
        total_actions += actions

        a_won = (winner == MAGENTA) if swap else (winner == CYAN)
        results["a" if a_won else "b"] += 1

        if args.games > 1 and (game + 1) % max(1, args.games // 10) == 0:
            print(f"  {game + 1}/{args.games} games...")

    elapsed = time.perf_counter() - start
    a_wins = results["a"]
    b_wins = results["b"]

    print(f"\n{'=' * 60}")
    print(f"{agent_a.name:<28} {a_wins:>4} wins")
    print(f"{opponent_name:<28} {b_wins:>4} wins")
    print(f"{'=' * 60}")

    print(f"{agent_a.name} win rate: {a_wins / args.games:.1%}")
    print(f"average game length: {total_actions / args.games:.1f} actions")
    print(f"total time: {elapsed:.1f}s ({elapsed / args.games:.2f}s per game)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
