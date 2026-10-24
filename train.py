"""Training and ONNX Export Script for Colony: One (Level 3).

Collects demonstration games from GreedyAgent and RandomAgent,
trains ColonyNet using policy (cross-entropy) and value (MSE) loss,
and exports the trained network to a compliant `model.onnx`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from colony_one import constants
from colony_one.baselines import GreedyAgent, RandomAgent
from colony_one.encoding import NUM_ACTIONS, action_index
from colony_one.env import ColonyOneEnv
from model import ColonyNet


class GameTrajectoryDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray, values: np.ndarray):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).long()
        self.values = torch.from_numpy(values).float()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx], self.values[idx]


def collect_dataset(num_games: int = 350, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate self-play and sparring games between Greedy and Random agents."""
    print(f"Collecting {num_games} training games...")
    rng = np.random.default_rng(seed)

    all_obs = []
    all_actions = []
    all_values = []

    for game_idx in range(num_games):
        game_seed = int(rng.integers(0, 1000000))
        env = ColonyOneEnv(seed=game_seed)

        # Mix opponent pairings:
        # 60% Greedy vs Greedy, 20% Greedy vs Random, 20% Random vs Greedy
        p = rng.random()
        if p < 0.6:
            cyan_agent = GreedyAgent(seed=int(rng.integers(0, 1000000)))
            magenta_agent = GreedyAgent(seed=int(rng.integers(0, 1000000)))
        elif p < 0.8:
            cyan_agent = GreedyAgent(seed=int(rng.integers(0, 1000000)))
            magenta_agent = RandomAgent(seed=int(rng.integers(0, 1000000)))
        else:
            cyan_agent = RandomAgent(seed=int(rng.integers(0, 1000000)))
            magenta_agent = GreedyAgent(seed=int(rng.integers(0, 1000000)))

        history = []  # tuples of (obs, canonical_action_idx, moving_team)

        obs = env.reset()
        while not env.terminated:
            team = env.current_team
            agent = cyan_agent if team == constants.CYAN else magenta_agent
            action = agent.select(env.board, team, env.actions_left, env.round_number)
            if action is None:
                break

            act_idx = action_index(action, team)
            history.append((obs, act_idx, team))
            obs, reward, terminated, info = env.step(action)

        winner = env.winner  # constants.CYAN or constants.MAGENTA
        if winner is not None:
            for state_obs, chosen_act, acting_team in history:
                target_value = 1.0 if acting_team == winner else -1.0
                all_obs.append(state_obs)
                all_actions.append(chosen_act)
                all_values.append(target_value)

        if (game_idx + 1) % 50 == 0 or game_idx == num_games - 1:
            print(f"  Processed {game_idx + 1}/{num_games} games ({len(all_obs)} transitions)")

    return (
        np.array(all_obs, dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_values, dtype=np.float32),
    )


def train_model(
    model: nn.Module,
    dataset: GameTrajectoryDataset,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Train ColonyNet on collected trajectories."""
    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    print(f"\nTraining on {device.upper()} for {epochs} epochs (dataset size: {len(dataset)})...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_actions = 0
        total_samples = 0

        for obs, act, val in loader:
            obs = obs.to(device)
            act = act.to(device)
            val = val.to(device)

            optimizer.zero_grad()
            p_logits, v_pred = model(obs)

            p_loss = policy_criterion(p_logits, act)
            v_loss = value_criterion(v_pred, val)
            loss = p_loss + 0.5 * v_loss

            loss.backward()
            optimizer.step()

            batch_len = obs.size(0)
            total_loss += loss.item() * batch_len
            total_p_loss += p_loss.item() * batch_len
            total_v_loss += v_loss.item() * batch_len

            preds = p_logits.argmax(dim=-1)
            correct_actions += (preds == act).sum().item()
            total_samples += batch_len

        scheduler.step()
        avg_loss = total_loss / total_samples
        avg_p_loss = total_p_loss / total_samples
        avg_v_loss = total_v_loss / total_samples
        acc = correct_actions / total_samples * 100.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Loss: {avg_loss:.4f} (P: {avg_p_loss:.4f}, V: {avg_v_loss:.4f}) | "
            f"Policy Acc: {acc:.2f}%"
        )


def export_to_onnx(model: nn.Module, output_path: str = "model.onnx") -> None:
    """Export the trained model to ONNX adhering strictly to tournament specifications."""
    print(f"\nExporting model to {output_path}...")
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 16, 8, 8, dtype=torch.float32)

    # Export using TorchScript exporter (dynamo=False avoids onnxscript dependency)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["policy", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        dynamo=False,
    )

    # Post-process: Tournament disallows 'Constant' operator nodes;
    # Convert any Constant nodes into graph initializers so validator passes.
    onnx_model = onnx.load(output_path)
    clean_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    t = attr.t
                    t.name = node.output[0]
                    onnx_model.graph.initializer.append(t)
        else:
            clean_nodes.append(node)

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(clean_nodes)
    onnx.save(onnx_model, output_path)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Export completed: {output_path} ({file_size_mb:.2f} MB)")
    if file_size_mb > 5.0:
        print("[WARNING] File size exceeds the 5 MB tournament limit!")
    else:
        print("[OK] File size is well within the 5 MB limit.")


def main():
    parser = argparse.ArgumentParser(description="Train Colony: One policy-value network")
    parser.add_argument("--games", type=int, default=350, help="Number of games to collect")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    # Step 1: Collect Data
    obs, acts, vals = collect_dataset(num_games=args.games)
    dataset = GameTrajectoryDataset(obs, acts, vals)

    # Step 2: Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColonyNet()

    # Step 3: Train
    train_model(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Step 4: Export to ONNX
    export_to_onnx(model, output_path=args.output)

    # Step 5: Automatically run validate_submission.py
    print("\nRunning submission validator:")
    val_script = Path("tools") / "validate_submission.py"
    if val_script.exists():
        subprocess.run([sys.executable, str(val_script), args.output])


if __name__ == "__main__":
    main()
