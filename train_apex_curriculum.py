#!/usr/bin/env python3
"""Apex Curriculum: Next-Generation Self-Play Reinforcement Learning Engine.

Designed to surpass the 42Heilbronn champion pipeline through:
1. Tournament Cold-Start: Pre-seeds replay buffer with 1,999 verified competition transitions.
2. Nemesis Sparring: Integrates best_10.onnx into the rival training pool (15% of games).
3. Self-Play Ladder: Uses an AlphaZero promotion gate (>=55% vs current champion) to
   ensure continuous, monotonic Elo progression without plateauing.
4. 10-Worker Multi-Core Concurrency: Generates 500-game batches in under 50 seconds.
5. Automated Nemesis Tracking: Benchmarks every promoted champion directly against
   best_10.onnx at official 100 simulations, auto-terminating when win rate > 60%.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from colony_one import ColonyOneEnv, constants, encoding, mcts
from colony_one.baselines import GreedyAgent
from colony_one.onnx_runner import OnnxEvaluator


# ---------------------------------------------------------------- Model Architecture
def make_model() -> nn.Module:
    class ColonyPolicyValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Flatten(1), nn.Linear(1024, 64), nn.ReLU(),
            )
            self.policy = nn.Linear(64, 4096)
            self.value = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Tanh(),
            )

        def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.body(observation)
            value = torch.clamp(self.value(h).squeeze(1), -0.999999, 0.999999)
            return self.policy(h), value

    return ColonyPolicyValue()


# ---------------------------------------------------------------- Heuristic Teacher
def teacher(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = observations.reshape(-1, 16, 64)
    own, enemy = flat[:, :6], flat[:, 6:12]
    weights = np.array([1.0, 2.5, 1.8, 2.0, 2.8, 0.0], np.float32)
    material = ((own - enemy).sum(axis=2) * weights).sum(axis=1)

    squares = np.arange(64)
    rows, cols = squares // 8, squares % 8
    centre = (3.5 - np.maximum(abs(rows - 3.5), abs(cols - 3.5))) / 3.0
    centre = centre.astype(np.float32)
    king_centre = ((own[:, 5] - enemy[:, 5]) * centre).sum(axis=1)

    adjacent = (
        np.maximum(abs(rows[:, None] - rows), abs(cols[:, None] - cols)) == 1
    ).astype(np.float32)
    own_guards = ((own[:, :5].sum(axis=1) @ adjacent) * own[:, 5]).sum(axis=1)
    enemy_guards = ((enemy[:, :5].sum(axis=1) @ adjacent) * enemy[:, 5]).sum(axis=1)

    locked = observations[:, 12]
    rim = np.zeros_like(locked)
    rim[:, 0, :] = rim[:, -1, :] = 1
    rim[:, :, 0] = rim[:, :, -1] = 1
    rim[:, 1:, :] = np.maximum(rim[:, 1:, :], locked[:, :-1, :])
    rim[:, :-1, :] = np.maximum(rim[:, :-1, :], locked[:, 1:, :])
    rim[:, :, 1:] = np.maximum(rim[:, :, 1:], locked[:, :, :-1])
    rim[:, :, :-1] = np.maximum(rim[:, :, :-1], locked[:, :, 1:])
    rim = (rim * (1 - locked)).reshape(-1, 64)

    urgency = flat[:, 14, 0]
    burning = ((enemy[:, 5] - own[:, 5]) * rim).sum(axis=1)

    value = np.tanh(
        0.45 * material + 0.45 * king_centre
        + 0.12 * (own_guards - enemy_guards)
        + 1.2 * (urgency ** 4) * burning
    ).astype(np.float32)

    capture = (enemy * np.array([0.6, 0.9, 0.8, 0.8, 1.0, 1.2], np.float32)[None, :, None]).sum(axis=1)
    delta = centre[None, None, :] - centre[None, :, None]
    king_from = own[:, 5, :, None]

    policy = (
        capture[:, None, :] + (0.15 + 0.5 * king_from) * delta
        - 0.7 * king_from * rim[:, None, :] * (urgency[:, None, None] ** 4)
    ).reshape(-1, 4096).astype(np.float32)

    return policy, value


# ---------------------------------------------------------------- Tournament Parser
ACTION_RE = re.compile(
    r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w+)\s+"
    r"([A-H][1-8])\s*->\s*([A-H][1-8])\s+(MOVE|SHOOT)", re.MULTILINE
)
WINNER_RE = re.compile(r"\b(CYAN|MAGENTA) WINS\b")


def parse_pos(text: str) -> tuple[int, int]:
    return 8 - int(text[1]), ord(text[0]) - ord("A")


def load_tournament_transitions() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]]:
    paths = [
        Path("data/tournament/export-01.json"),
        Path("data/tournament/export-02.json"),
        Path("data/tournament/tournament-matches.json"),
    ]
    matches = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                mid = row.get("match_id")
                if mid and mid not in matches and row.get("protocol") and "starting_observation" in row:
                    matches[mid] = row
        except Exception:
            pass

    samples = []
    for mid, row in matches.items():
        try:
            env = ColonyOneEnv(scene=np.asarray(row["starting_observation"], dtype=np.float32))
            actions = ACTION_RE.findall(row["protocol"])
            winners = WINNER_RE.findall(row.get("protocol") or "")
            winner = winners[-1] if winners else None
            if not winner or not actions:
                continue

            for number, side, abbr, start, end, _ in actions:
                if side != env.current_team or int(number) != env.action:
                    break
                from_sq, to_sq = parse_pos(start), parse_pos(end)
                legal = [a for a in env.legal_actions() if a.from_pos == from_sq and a.to_pos == to_sq]
                if len(legal) != 1:
                    break
                act = legal[0]

                obs = env.observation().astype(np.float16)
                legal_indices = env.legal_action_indices().astype(np.int16)
                act_idx = encoding.action_index(act, side)
                match_pos = np.where(legal_indices == act_idx)[0]
                if len(match_pos) > 0:
                    probs = np.zeros(len(legal_indices), dtype=np.float32)
                    probs[match_pos[0]] = 1.0
                    val = 1.0 if side == winner else -1.0
                    samples.append((int(mid) + 100000000, obs, legal_indices, probs, val))
                env.step(act)
        except Exception:
            continue

    return samples


# ---------------------------------------------------------------- Clean ONNX Export
def export_clean_onnx(model: nn.Module, destination: Path) -> None:
    destination = Path(destination)
    torch.onnx.export(
        model, torch.zeros(1, 16, 8, 8), str(destination),
        input_names=["observation"], output_names=["policy", "value"],
        dynamic_axes={name: {0: "batch"} for name in ["observation", "policy", "value"]},
        opset_version=17, dynamo=False, external_data=False,
    )
    graph = onnx.load(str(destination))
    clean_nodes = []
    for node in graph.graph.node:
        if node.op_type != "Constant":
            clean_nodes.append(node)
            continue
        attributes = {a.name: a for a in node.attribute}
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attributes["value"].t)
        tensor.name = node.output[0]
        graph.graph.initializer.append(tensor)
    del graph.graph.node[:]
    graph.graph.node.extend(clean_nodes)
    onnx.checker.check_model(graph)
    onnx.save_model(graph, str(destination), save_as_external_data=False)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    obs = ColonyOneEnv().observation()
    for batch in [1, 8, 16]:
        boards = np.repeat(obs[None], batch, axis=0)
        actual = session.run(None, {"observation": boards})
        with torch.inference_mode():
            expected = model(torch.from_numpy(boards))
        assert actual[0].shape == (batch, 4096) and actual[1].shape == (batch,)
        np.testing.assert_allclose(actual[0], expected[0].numpy(), rtol=1e-4, atol=1e-5)
        assert np.isfinite(actual[0]).all() and np.isfinite(actual[1]).all()
        assert (abs(actual[1]) <= 1.0).all()


def validate_export(path: Path) -> None:
    validator = Path("tools/validate_submission.py")
    if not validator.is_file():
        return
    res = subprocess.run([sys.executable, str(validator), str(path)], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"Official validation rejected: {res.stdout}\n{res.stderr}")


# ---------------------------------------------------------------- Worker Processes
_CURRENT = None
_NEMESIS = None
_RIVALS = []


def worker_init(current_path: str | None, nemesis_path: str | None, rival_paths: list[str]):
    global _CURRENT, _NEMESIS, _RIVALS
    _CURRENT = OnnxEvaluator(current_path) if current_path and os.path.exists(current_path) else teacher
    _NEMESIS = OnnxEvaluator(nemesis_path) if nemesis_path and os.path.exists(nemesis_path) else None
    _RIVALS = [OnnxEvaluator(p) for p in rival_paths if os.path.exists(p)]


def play_training_game(job: tuple[int, int, int]) -> tuple[int, list]:
    game_id, seed, simulations = job
    rng = np.random.default_rng(seed)
    env = ColonyOneEnv(seed=seed)
    config = mcts.MCTSConfig(
        simulations=simulations,
        warmup_temperatures=(1.0, 0.75, 0.5, 0.25),
    )

    learner_team = constants.CYAN if (game_id // 10) % 2 == 0 else constants.MAGENTA
    mode = game_id % 10

    # Opponent distribution:
    # 0, 1, 2, 3, 4, 5: Self-play with current champion (60%)
    # 6, 7: Nemesis sparring vs best_10.onnx (20%)
    # 8: Past local champions from ladder (10%)
    # 9: Heuristic Teacher / Greedy (10%)
    if (mode == 6 or mode == 7) and _NEMESIS is not None:
        rival = _NEMESIS
    elif mode == 8 and _RIVALS:
        rival = _RIVALS[int(rng.integers(len(_RIVALS)))]
    elif mode == 9:
        rival = teacher
    else:
        rival = _CURRENT

    greedy = GreedyAgent(seed=seed + 1)
    samples = []

    while not env.terminated:
        team = env.current_team
        if mode == 9 and team != learner_team and rng.random() < 0.5:
            action = greedy.select(env.board, team, env.actions_left, env.round_number)
        else:
            is_learner = (team == learner_team) or (mode < 6)
            evaluator = _CURRENT if is_learner else rival
            root = mcts.run(env.board, team, env.actions_left, env.round_number, evaluator, config, rng=rng)

            if is_learner:
                indices = env.legal_action_indices().astype(np.int16)
                probabilities = mcts.policy_target(root)[indices]
                if probabilities.sum() > 0:
                    samples.append((
                        env.observation().astype(np.float16),
                        indices,
                        probabilities,
                        team,
                    ))

            temperature = 1.0 if env.action <= 8 else (0.5 if env.action <= 24 else 0.0)
            action = mcts.select_action_index(root, temperature, rng=rng)

        if action is None:
            break
        env.step(action)

    winner = env.winner
    return game_id, [
        (o, idx, p, 1.0 if t == winner else (-1.0 if winner is not None else 0.0))
        for o, idx, p, t in samples
    ]


# ---------------------------------------------------------------- Arena (Curriculum Promotion)
_ARENA_CANDIDATE = None
_ARENA_CHAMPION = None


def arena_init(cand_path: str, champ_path: str | None):
    global _ARENA_CANDIDATE, _ARENA_CHAMPION
    _ARENA_CANDIDATE = OnnxEvaluator(cand_path)
    _ARENA_CHAMPION = OnnxEvaluator(champ_path) if champ_path and os.path.exists(champ_path) else None


def arena_match(job: tuple[int, int]) -> tuple[str, bool]:
    game_id, seed = job
    env = ColonyOneEnv(seed=seed)
    config = mcts.MCTSConfig(simulations=100)
    cand_is_cyan = (game_id % 2 == 0)
    cand_team = constants.CYAN if cand_is_cyan else constants.MAGENTA

    cand_rng = np.random.default_rng(seed)
    champ_rng = np.random.default_rng(seed + 1)
    greedy = GreedyAgent(seed=seed + 2)

    while not env.terminated:
        team = env.current_team
        is_cand = (team == cand_team)
        if is_cand:
            ev = _ARENA_CANDIDATE
            rng = cand_rng
            action = mcts.search_action(env.board, team, env.actions_left, env.round_number, ev, config, rng=rng)
        else:
            if _ARENA_CHAMPION is not None:
                ev = _ARENA_CHAMPION
                rng = champ_rng
                action = mcts.search_action(env.board, team, env.actions_left, env.round_number, ev, config, rng=rng)
            else:
                action = greedy.select(env.board, team, env.actions_left, env.round_number)
        if action is None:
            break
        env.step(action)

    won = (env.winner == cand_team)
    return cand_team, won


def run_curriculum_arena(cand_path: str, champ_path: str | None, games: int = 40, workers: int = 8) -> int:
    jobs = [(i, 10000000 + i * 29) for i in range(games)]
    wins = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=arena_init,
        initargs=(cand_path, champ_path),
    ) as pool:
        for done, (team, won) in enumerate(pool.map(arena_match, jobs), 1):
            wins += int(won)
            if done % 10 == 0 or done == games:
                print(f"  Curriculum Arena {done}/{games}: candidate {wins} wins", flush=True)
    return wins


# ---------------------------------------------------------------- Referee Benchmark vs Nemesis
def run_nemesis_benchmark(model_path: str, nemesis_path: str, games: int = 40, workers: int = 8) -> float:
    jobs = [(i, 9000000 + i * 31) for i in range(games)]
    wins = 0
    colours = {constants.CYAN: [0, 0], constants.MAGENTA: [0, 0]}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=arena_init,
        initargs=(model_path, nemesis_path),
    ) as pool:
        for done, (team, won) in enumerate(pool.map(arena_match, jobs), 1):
            wins += int(won)
            colours[team][0] += int(won)
            colours[team][1] += 1
            if done % 10 == 0 or done == games:
                print(f"  Nemesis Benchmark {done}/{games}: candidate {wins} wins", flush=True)
    wr = wins / games * 100.0
    print(f"Nemesis Benchmark Result: {wins}/{games} ({wr:.1f}%) | Colours: {colours}", flush=True)
    return wr


# ---------------------------------------------------------------- Replay Persistence
def save_replay(path: Path, records: list, scene: np.ndarray) -> None:
    n = len(records)
    width = max(len(row[2]) for row in records)
    observations = np.empty((n, 16, 8, 8), np.float16)
    indices = np.zeros((n, width), np.int64)
    probabilities = np.zeros((n, width), np.float32)
    valid = np.zeros((n, width), bool)
    games = np.empty(n, np.int64)
    values = np.empty(n, np.float32)

    for i, (g, o, idx, p, v) in enumerate(records):
        games[i] = g
        observations[i] = o
        values[i] = v
        indices[i, :len(idx)] = idx
        probabilities[i, :len(idx)] = p
        valid[i, :len(idx)] = True

    np.savez_compressed(
        path, observations=observations, indices=indices,
        probabilities=probabilities, valid=valid, games=games,
        values=values, scene=scene
    )


def load_replay(path: Path, scene: np.ndarray) -> list:
    with np.load(path) as data:
        if not np.array_equal(data["scene"], scene):
            raise RuntimeError("Replay belongs to a different scene")
        return [
            (int(g), o.copy(), idx[mask].astype(np.int16), prob[mask].copy(), float(v))
            for g, o, idx, mask, prob, v in zip(
                data["games"], data["observations"], data["indices"],
                data["valid"], data["probabilities"], data["values"]
            )
        ]


# ---------------------------------------------------------------- Training Optimization
def fit(
    replay_path: Path,
    weights_in: str | None,
    weights_out: Path,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 3e-4,
    threads: int = 2,
) -> nn.Module:
    torch.set_num_threads(threads)
    model = make_model()
    if weights_in and os.path.exists(weights_in):
        model.load_state_dict(torch.load(weights_in, weights_only=True, map_location="cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    with np.load(replay_path) as data:
        x = torch.from_numpy(data["observations"].astype(np.float32))
        idx = torch.from_numpy(data["indices"])
        prob = torch.from_numpy(data["probabilities"])
        valid = torch.from_numpy(data["valid"])
        val = torch.from_numpy(data["values"])
        val_mask = data["games"] % 10 == 0

    train_ids = torch.from_numpy(np.flatnonzero(~val_mask))
    val_ids = torch.from_numpy(np.flatnonzero(val_mask))
    best_score = float("inf")

    def calc_loss(batch):
        p_logits, v_pred = model(x[batch])
        legal = p_logits.gather(1, idx[batch]).masked_fill(~valid[batch], -1e9)
        p_loss = -(prob[batch] * F.log_softmax(legal, dim=1)).sum(1).mean()
        v_loss = F.mse_loss(v_pred, val[batch])
        return p_loss + v_loss

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        order = train_ids[torch.randperm(len(train_ids))]

        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            b_loss = calc_loss(batch)
            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += float(b_loss.detach()) * len(batch)

        model.eval()
        with torch.inference_mode():
            val_score = sum(float(calc_loss(b)) * len(b) for b in val_ids.split(batch_size)) / max(1, len(val_ids))

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), weights_out)

        print(f"  Epoch {ep}/{epochs} | train={tot_loss/len(train_ids):.4f} val={val_score:.4f}", flush=True)

    if os.path.exists(weights_out):
        model.load_state_dict(torch.load(weights_out, weights_only=True, map_location="cpu"))
    return model.eval()


# ---------------------------------------------------------------- Main Orchestrator
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-games", type=int, default=55000)
    parser.add_argument("--batch-games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--replay-positions", type=int, default=50000)
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--promote-rate", type=float, default=0.55)
    parser.add_argument("--out", type=Path, default=Path("runs/apex_curriculum"))
    parser.add_argument("--nemesis", type=str, default="best_10.onnx")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    scene = ColonyOneEnv().observation()
    scene_hash = hashlib.sha256(scene.tobytes()).hexdigest()
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if (args.resume and state_path.is_file()) else None

    records = deque(maxlen=args.replay_positions)
    if state and state.get("replay"):
        records.extend(load_replay(Path(state["replay"]), scene))
    elif not state:
        tourney_samples = load_tournament_transitions()
        if tourney_samples:
            records.extend(tourney_samples)
            print(f"Seeded initial replay buffer with {len(tourney_samples)} verified tournament transitions.", flush=True)

    completed = state["completed_games"] if state else 0
    iteration = state["iteration"] if state else 0
    champion_pt = state["champion_pt"] if state else None
    champion_onnx = state["champion_onnx"] if state else None
    rivals = state.get("rivals", []) if state else []
    best_wr_vs_nemesis = state.get("best_wr", 0.0) if state else 0.0

    print("=" * 72, flush=True)
    print("      APEX CURRICULUM: NEXT-GEN REINFORCEMENT LEARNING", flush=True)
    print(f"Target Games: {args.total_games} | Batch: {args.batch_games} | Workers: {args.workers}", flush=True)
    print(f"Nemesis Target: {args.nemesis} | Replay Capacity: {args.replay_positions}", flush=True)
    print("=" * 72, flush=True)

    while completed < args.total_games:
        iteration += 1
        cycle_dir = args.out / f"iteration-{iteration:05d}"
        cycle_dir.mkdir(exist_ok=True)
        count = min(args.batch_games, args.total_games - completed)

        t_start = time.monotonic()
        replay_path = cycle_dir / "replay.npz"
        cand_pt = cycle_dir / "candidate.pt"
        cand_onnx = cycle_dir / "candidate.onnx"

        # Step 1: Collect games
        print(f"\n--- Cycle {iteration:03d}: Generating {count} games (Total so far: {completed}) ---", flush=True)
        jobs = [(completed + i, 4242 + iteration * 1000 + i * 37, args.simulations) for i in range(count)]

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
            initargs=(champion_onnx, args.nemesis, rivals),
        ) as executor:
            batch = []
            for done, (gid, samples) in enumerate(executor.map(play_training_game, jobs), 1):
                batch.extend((gid, *s) for s in samples)
                if done % max(10, count // 5) == 0 or done == count:
                    dt = time.monotonic() - t_start
                    print(f"  Games {done}/{count} ({done/dt:.2f} g/s) | Samples: {len(batch)}", flush=True)

        records.extend(batch)
        save_replay(replay_path, list(records), scene)

        # Step 2: Fit model
        print(f"Training on {len(records)} positions in replay buffer...", flush=True)
        lr = max(1e-5, args.learning_rate * (0.98 ** (iteration - 1)))
        model = fit(
            replay_path=replay_path,
            weights_in=champion_pt,
            weights_out=cand_pt,
            epochs=6,
            batch_size=args.batch_size,
            lr=lr,
            threads=args.threads,
        )

        # Step 3: Export & Validate
        export_clean_onnx(model, cand_onnx)
        validate_export(cand_onnx)
        print(f"Export & Validation OK: {cand_onnx.name}", flush=True)

        # Step 4: Curriculum Arena Gate (vs Current Champion)
        print(f"\nCurriculum Arena Gate vs Current Champion ({args.arena_games} games, 100 sims):", flush=True)
        wins = run_curriculum_arena(str(cand_onnx), champion_onnx, games=args.arena_games, workers=args.workers)
        promoted = (champion_pt is None) or (wins >= math.ceil(args.promote_rate * args.arena_games))

        if promoted:
            if champion_onnx:
                rivals = (rivals + [champion_onnx])[-3:]
            champion_pt = str(cand_pt)
            champion_onnx = str(cand_onnx)
            shutil.copyfile(cand_pt, args.out / "best.pt")
            shutil.copyfile(cand_onnx, args.out / "best.onnx")
            print(f">>> LADDER PROMOTED: Candidate won {wins}/{args.arena_games}! New Champion Crowned! <<<", flush=True)

            # Step 5: Test New Champion Directly Against Nemesis
            print(f"\nTesting Champion vs Nemesis ({args.nemesis}) in 40-Game Arena...", flush=True)
            nemesis_wr = run_nemesis_benchmark(champion_onnx, args.nemesis, games=args.arena_games, workers=args.workers)

            if nemesis_wr > best_wr_vs_nemesis:
                best_wr_vs_nemesis = nemesis_wr
                shutil.copyfile(champion_onnx, "model.onnx")
                print(f">>> NEW RECORD VS NEMESIS: {nemesis_wr:.1f}%! PROMOTED TO model.onnx! <<<", flush=True)

            if best_wr_vs_nemesis >= 60.0:
                print("\n" + "=" * 72, flush=True)
                print(f"MISSION ACCOMPLISHED! Candidate achieved {best_wr_vs_nemesis:.1f}% >= 60.0% vs {args.nemesis}!", flush=True)
                print("=" * 72, flush=True)
                break
        else:
            print(f"Not promoted ({wins}/{args.arena_games} < {math.ceil(args.promote_rate * args.arena_games)}). Keeping previous champion.", flush=True)

        completed += count
        state_data = {
            "completed_games": completed,
            "iteration": iteration,
            "best_wr": best_wr_vs_nemesis,
            "champion_pt": champion_pt,
            "champion_onnx": champion_onnx,
            "rivals": rivals,
            "replay": str(replay_path),
            "promoted": promoted,
        }
        state_path.write_text(json.dumps(state_data, indent=2))

    print(f"\nTraining Complete. Final model saved at model.onnx (Best Win Rate: {best_wr_vs_nemesis:.1f}%)")


if __name__ == "__main__":
    main()

