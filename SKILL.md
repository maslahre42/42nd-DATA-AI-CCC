---
name: competitive-rl-alphazero
description: >-
  Comprehensive guide and engineering runbook for competitive Game AI,
  AlphaZero self-play reinforcement learning, MCTS integration, loss formulation,
  and avoiding catastrophic pitfalls in game contests.
---

# Competitive Reinforcement Learning & AlphaZero Engineering Guide

A battle-tested engineering handbook and diagnostic runbook for training superhuman neural network agents in turn-based strategy games (Cloudflight Coding Contest, Kaggle Game AI, NeurIPS competitions).

---

## 1. The 5 Fatal Pitfalls of Game AI (And How to Prevent Them)

### Pitfall 1: One-Hot Cross-Entropy on Sparse Action Spaces (Logit Collapse)
* **The Symptom:** Model win rate drops to 0.0%. Inspecting the policy logits reveals extreme negative values (e.g., `-35.0` to `-48.0`) on normal moves, while one arbitrary move has `-12.0`. In MCTS, softmax over legal actions turns this into a 95% prior on the `-12.0` move and 0.00% on everything else, blinding the search tree.
* **The Cause:** Using standard PyTorch `nn.CrossEntropyLoss(policy_logits, target_action)` over a large flat action space (e.g., $N=4096$). In any given board state, only 20–40 actions are legal. Standard cross-entropy pushes the 1 target action UP and pushes all other 4,095 actions DOWN towards $-\infty$. Over thousands of gradient steps, unvisited legal actions get crushed.
* **The Correct Solution:** **Gathered Legal-Softmax Loss**. Only compare legal actions against each other:
```python
# Gather only the legal action logits for each sample
legal_logits = policy_out.gather(1, legal_indices).masked_fill(~valid_mask, -1e9)
log_probs = F.log_softmax(legal_logits, dim=1)
# Target is the soft MCTS visit distribution, NOT a 1-hot index!
policy_loss = -(target_probabilities * log_probs).sum(dim=1).mean()
```

---

### Pitfall 2: The "Toddler vs. Usain Bolt" Trap (Premature Hard-Gating)
* **The Symptom:** Training stalls at 0%–12% win rate. Checkpoints never promote, and the replay buffer fills with repeated losses.
* **The Cause:** Gating early candidates directly against a 50,000-iteration grandmaster model. A Cycle 1 network cannot beat a grandmaster. If you require beating the grandmaster to promote, the "champion" weights never update. The training loop gets stuck playing with weak weights, generating bad data, and reinforcing losses.
* **The Correct Solution:** **The AlphaZero Elo Ladder**:
  1. **Local Curriculum Gate:** Candidate must beat the *current local champion* with $\ge 55\%$ (e.g., 22/40 wins) in an arena to become the new champion. This guarantees monotonic skill climbing.
  2. **Adversarial Nemesis Sparring in the Training Mix:** Allocate 15%–25% of training games directly against the grandmaster target (`best_model.onnx`). The learner steadily absorbs counterplay against the target as its Elo rises.
  3. **Nemesis Tracking:** Benchmark every promoted local champion against the grandmaster target. Terminate when target win rate $> 60\%$.

---

### Pitfall 3: Canned Opening Scripts
* **The Symptom:** Agent plays 1–3 brilliant moves, then suddenly makes a catastrophic blunder (e.g., sacrificing a queen/daemon for nothing or abandoning the king).
* **The Cause:** Hardcoding fixed opening lines (e.g., "Moves 1–8: play line X"). Turn-based games branch immediately: if the opponent plays an unexpected response on Move 2, the canned move on Move 3 is played on an entirely different board geometry, often walking straight into an enemy capture.
* **The Correct Solution:** **Tournament Replay Buffer Seeding (Cold-Start)**.
  Instead of hardcoding moves, ingest 1,000–2,000 positions from human tournament archives directly into the initial replay buffer. The network learns natural opening priors while remaining fully responsive to opponent deviations.

---

### Pitfall 4: The GPU Fallacy (Amdahl's Law in Game AI)
* **The Symptom:** Believing the training pipeline is slow because "it is not running on GPU."
* **The Reality:** In board game RL with MCTS:
  - **95% of wall-clock time** is CPU-bound game logic (`Board.apply_action()`, collision checks, MCTS tree traversal) and batch-1 inference. Passing batch-1 tensors across the PCIe bus to GPU VRAM is actually *slower* than running on local CPU cache.
  - **5% of wall-clock time** is neural network backpropagation (e.g., 2 seconds per cycle on a 338k parameter network).
  - Accelerating 5% of your code with a GPU leaves 95% untouched (Amdahl's Law).
* **The High-Performance Strategy:** Scale CPU worker processes to match logical CPU cores (e.g., 8–10 workers using `ProcessPoolExecutor(mp_context='spawn')`). This generates 8–12 games/second on modern multicore CPUs.

---

### Pitfall 5: Root Valuation vs. Leaf Valuation in MCTS
* **The Symptom:** MCTS search chooses seemingly random moves or ignores obvious one-move captures.
* **The Cause:** An evaluator function inside MCTS evaluates `board` from the enclosing scope (the root board) instead of the observation planes generated at each leaf node during tree expansion.
* **The Correct Solution:** Always evaluate the batched observation tensor passed into the evaluator:
```python
def evaluator(planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Evaluate batched leaf states, NOT root board!
    logits, values = onnx_session.run(None, {"observation": planes_batch})
    return logits, values
```

---

## 2. Complete AlphaZero Architecture Blueprint

```
                      +-----------------------------+
                      |   Current Local Champion    |
                      +--------------+--------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Cycle Generation (500 Games)       |
                  |  - 60% Candidate Self-Play          |
                  |  - 20% Adversarial Nemesis Sparring  |
                  |  - 10% Past Accepted Champions       |
                  |  - 10% Heuristic Teacher / Baseline  |
                  +------------------+-------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Rolling Replay Buffer              |
                  |   (50,000 Positions, Float16)        |
                  +------------------+-------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   PyTorch Optimization (6 Epochs)    |
                  |   - Gathered Legal-Softmax Loss      |
                  |   - MSE Value Loss                   |
                  |   - AdamW (lr=3e-4, Cosine Decay)    |
                  +------------------+-------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Export & Validation (ONNX Opset 17)|
                  |   - Constant folding into inits      |
                  +------------------+-------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Balanced Arena Gate (40 Games)     |
                  |   Candidate vs. Local Champion       |
                  +------------------+-------------------+
                         /                   \
        Win Rate >= 55% /                     \ Win Rate < 55%
                       v                       v
         +--------------------------+    +-----------------------+
         | Promoted: New Champion   |    | Retain Previous Best  |
         | - Update champion.pt     |    +-----------------------+
         | - Benchmark vs Nemesis   |
         | - If > 60% vs Nemesis:   |
         |   SAVE model.onnx & EXIT |
         +--------------------------+
```

---

## 3. Proven Hyperparameter Reference Table

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| `replay_positions` | 50,000 | Prevents catastrophic forgetting while discarding stale data |
| `batch_games` | 400 – 500 | Balance between stable policy updates and fast cycle feedback |
| `simulations` (training) | 50 – 70 | Fast game generation with sufficient tactical lookahead |
| `simulations` (arena) | 100 | Official tournament search depth |
| `batch_size` | 128 | Smooth gradient updates for AdamW |
| `learning_rate` | $3 \times 10^{-4} \to 1 \times 10^{-5}$ | Cosine annealing schedule |
| `weight_decay` | $1 \times 10^{-4}$ | Prevents weight explosion |
| `grad_clip_norm` | 1.0 | Prevents exploding gradients on high-value endgames |
| `warmup_temperatures` | `(1.0, 0.75, 0.5, 0.25)` | Opening exploration in actions 1–4 |
| `move_temperatures` | Actions 1–8: 1.0; 9–24: 0.5; 25+: 0.0 | Exploitation transition |
| `promote_rate` | 0.55 (22/40 wins) | Strict barrier against noise promotions |

---

## 4. Reusable Code Template: `fit()` Function

```python
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
        # Game-level holdout validation (no transition leakage)
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

    if os.path.exists(weights_out):
        model.load_state_dict(torch.load(weights_out, weights_only=True, map_location="cpu"))
    return model.eval()
```

---

## 5. Tournament Compliance Checklist

Before uploading any candidate `.onnx` to a live tournament server:
- [ ] **Opset Version:** Exactly 17.
- [ ] **Graph Initializers:** No bare `Constant` operator nodes in the graph (convert to graph initializers).
- [ ] **Value Clamping:** Strict clamp `[-0.999999, 0.999999]` to avoid float32 rounding overshoot at $\pm 1.0$.
- [ ] **File Size:** Under the contest limit (e.g., $< 5$ MiB; standard ColonyNet is ~1.29 MiB).
- [ ] **Dynamic Batching:** Verified on batch sizes 1, 8, and 16.
- [ ] **Output Consistency:** Max difference between PyTorch and ONNX Runtime outputs $\le 10^{-4}$.
- [ ] **Execution Provider:** Runs cleanly on `CPUExecutionProvider` within latency limits ($< 40$ ms).

