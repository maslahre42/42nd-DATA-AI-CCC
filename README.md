# Colony: One — AlphaZero Reinforcement Learning & Competitive Game AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Opset 17](https://img.shields.io/badge/ONNX-Opset%2017-005ced.svg)](https://onnx.ai/)
[![Validator](https://img.shields.io/badge/Submission-ACCEPTED-brightgreen.svg)](tools/validate_submission.py)
[![Winner Certificate](https://img.shields.io/badge/CCC%20Winner-Certificate-gold.svg)](docs/certificates/ccc-winner-certificate-42nd-data-ai-ccc-5873.pdf)
[![License: Non-Commercial Event](https://img.shields.io/badge/License-CCC%20Event-lightgrey.svg)](LICENSE)

An end-to-end AlphaZero self-play reinforcement learning engine and neural network model developed for **Colony: One** (Level 4, all 6 piece types) in the **42nd Cloudflight Coding Contest (CCC) — AI / Data Track**.

The trained model (`model.onnx`) pairs a 338,593-parameter dual-headed convolutional network with Monte Carlo Tree Search (MCTS, 100 simulations per move) to compete autonomously against international grandmaster models.

---

## 🏆 Key Achievements & Benchmarks

* **Winner Certificate:** Awarded the official 42nd Cloudflight Coding Contest Winner Certificate ([docs/certificates/ccc-winner-certificate-42nd-data-ai-ccc-5873.pdf](docs/certificates/ccc-winner-certificate-42nd-data-ai-ccc-5873.pdf)).
* **Tournament Compliance:** Passes the official validator (`tools/validate_submission.py`) with zero warnings: strictly clamped values in `[-0.999999, 0.999999]`, dynamic batching (1, 8, 16), zero `Constant` graph nodes, and 1.29 MiB payload (limit: 5.0 MiB).
* **Competitive Performance:** Defeated `best_10.onnx` (Germany #1 / Global 24th championship model) with a **55.0% win rate** (22/40 wins: 60.0% as Cyan, 50.0% as Magenta) under official referee MCTS evaluation (100 simulations/move).
* **High-Throughput Parallel Engine:** Multi-process CPU pipeline generating **8–11 games/second** across 10 workers, training a full 500-game AlphaZero cycle every 50 seconds.
* **Engineering Handbook:** Comprehensive runbook included in [`SKILL.md`](SKILL.md) covering loss formulations, MCTS tuning, and anti-pitfall guidelines for competitive RL.

---

## 🧠 Architectural Highlights

### 1. The Model Architecture (`model.py`)
A compact, highly expressive convolutional architecture designed to run within strict referee latency budgets (< 40 ms per batch):

```
Observation (batch, 16, 8, 8)
         │
         ├──> Conv2d(16, 16, 3x3, padding=1) + ReLU
         ├──> Conv2d(16, 16, 3x3, padding=1) + ReLU
         ├──> Flatten(1) -> Linear(1024, 64) + ReLU   <── [Shared Bottleneck]
         │
         ├──> Policy Head: Linear(64, 4096)           <── Raw Move Logits
         │
         └──> Value Head:  Linear(64, 32) + ReLU
                           Linear(32, 1) + Tanh       <── Side-to-Move [-1, 1]
```

### 2. Gathered Legal-Softmax Loss
Standard `CrossEntropyLoss` over a sparse 4,096-action space causes catastrophic logit collapse: unvisited legal moves get pushed to $-40$, destroying MCTS priors. 

Our engine isolates and normalizes **legal action logits only**:
```python
# Gather and mask only legal actions
legal_logits = policy_out.gather(1, legal_indices).masked_fill(~valid_mask, -1e9)
log_probs = F.log_softmax(legal_logits, dim=1)

# Supervise against normalized MCTS visit count distribution
policy_loss = -(target_probabilities * log_probs).sum(dim=1).mean()
value_loss = F.mse_loss(value_prediction, game_outcome)
loss = policy_loss + value_loss
```

### 3. The AlphaZero Elo Curriculum Ladder
To prevent premature gating collapse, the engine implements a two-tier progression:
1. **Local Elo Ladder:** A candidate must defeat the *current local champion* with $\ge 55\%$ in a 40-game arena to earn promotion.
2. **Nemesis Sparring Pool:** 20% of all self-play games are played directly against the grandmaster target (`best_10.onnx`).
3. **Tournament Cold-Start:** The 50,000-position replay buffer is pre-seeded with 1,999 verified transitions from real competition games.

---

## 📁 Repository Structure

```
.
├── colony_one/               # Official game engine, rules, board representation, MCTS
│   ├── action.py             # Action representations and coordinate encoding
│   ├── board.py              # 8x8 Board state and piece mechanics
│   ├── constants.py          # Teams, piece attributes, starting layouts
│   ├── encoding.py           # 16-plane board tensor encoding (64x64 action space)
│   ├── mcts.py               # Official tournament Monte Carlo Tree Search
│   ├── pieces.py             # Ping, Daemon, Exploit, Trace, Ice, Kernel definitions
│   └── rules.py              # Legal move generation, shooting, bodyguards, firewall
│
├── data/                     # Competition datasets & match protocols
│   └── tournament/           # 53 verified human tournament match replays
│
├── docs/                     # Challenge documentation & certificates
│   ├── certificates/         # Official contest winner certificate (PDF)
│   ├── observation.md        # Tensor specifications and plane semantics
│   ├── rules.md              # Complete Colony: One game rules
│   └── submission.md         # Referee MCTS parameters & submission contracts
│
├── play/                     # Evaluation & Sparring Harness
│   ├── model_vs_model.py     # Official referee arena match runner
│   └── random.onnx           # Baseline evaluation model
│
├── tools/                    # Verification Utilities
│   └── validate_submission.py # Official tournament ONNX submission validator
│
├── model.py                  # PyTorch ColonyNet architecture definition
├── export.py                 # ONNX opset 17 export & initializer folding script
├── model.onnx                # Validated champion model (promoted submission)
├── train_apex_curriculum.py  # Production AlphaZero self-play curriculum training engine
├── SKILL.md                  # Comprehensive Competitive RL & AlphaZero Handbook
├── pyproject.toml            # Package configuration
├── requirements.txt          # Python dependencies
└── LICENSE                   # Event contest license
```

---

## ⚡ Quickstart

### 1. Installation

Requires **Python 3.11+ (64-bit)**:

```bash
# Clone the repository
git clone https://github.com/<your-username>/colony-one-alphazero.git
cd colony-one-alphazero

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies and local package
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Validate the Submission Model

Verify that `model.onnx` passes all official tournament constraints:

```bash
python tools/validate_submission.py model.onnx
```

Expected output:
```
1/3  Tournament acceptance contract: [ok]
2/3  Check resource limits:           [ok] (peak RSS ~4 MiB, CPU 0.02 s)
3/3  Level 3/4 match limits:          [ok]
====================================================================
ACCEPTED -- this model satisfies the submission contract.
====================================================================
```

### 3. Run Head-to-Head Model Evaluation

Pit `model.onnx` against any baseline or rival model under official referee MCTS (100 simulations per move, alternating sides):

```bash
# Benchmark against random baseline (40 games)
python play/model_vs_model.py --model-a model.onnx --model-b play/random.onnx --games 40 --simulations 100

# Benchmark against the German championship model
python play/model_vs_model.py --model-a model.onnx --model-b best_10.onnx --games 40 --simulations 100
```

### 4. Train with Apex Curriculum Reinforcement Learning

Launch the multi-process AlphaZero training engine:

```bash
python train_apex_curriculum.py \
  --total-games 55000 \
  --batch-games 500 \
  --simulations 100 \
  --workers 10 \
  --arena-games 40 \
  --promote-rate 0.55 \
  --out runs/apex_curriculum_prod
```

To resume an existing training run:
```bash
python train_apex_curriculum.py --resume --out runs/apex_curriculum_prod
```

---

## 📖 Competitive RL Engineering Handbook (`SKILL.md`)

For a deep dive into reinforcement learning for competitive turn-based games, consult [`SKILL.md`](SKILL.md). It covers:
* The mathematics behind legal-action logit masking vs standard cross-entropy.
* Amdahl's Law in board game AI: why 95% of time is CPU-bound game tree traversal and how to optimize worker concurrency.
* Replay buffer management, fixed game-id holdout validation, and temperature annealing schedules.
* A pre-submission checklist for game AI competitions.

---

## 📜 License & Acknowledgments

* **Game Engine & Environment:** Copyright © **Paul Puntschart (2026)**. Supplied for the 42nd Cloudflight Coding Contest (CCC). Retained under the original contest license in [`LICENSE`](LICENSE).
* **Reference Opponent:** `best_10.onnx` by team **42Heilbronn_GTA6** (Eray Çakıray & Tobias Kern, 1st place Germany, 24th global).
* **Apex Engine & Implementation:** Independent reinforcement learning curriculum, gathered loss formulation, and optimization by the repository author.
