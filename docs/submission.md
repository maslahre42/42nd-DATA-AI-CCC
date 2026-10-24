# Submission and Referee

## Submission contract

Submit one self-contained `model.onnx` file.

```
input   observation  (batch, 16, 8, 8)  float32
output  policy       (batch, 4096)      float32, raw logits
output  value        (batch,)           float32, in [-1, 1]
```

Requirements:

- Dynamic batch axis on input and both outputs.
- Exactly one float32 input and two float32 outputs, in the order above.
- Policy output is logits, not probabilities.
- Value is from the side-to-move perspective and is finite in `[-1, 1]`.
- Maximum file size: 5 MiB.
- One file only: no external tensor data.
- Standard `ai.onnx` opset 13-18 only. No custom or ONNX-ML domains.
- No control flow, nested graphs, sparse tensors, sequence/optional values, or
  dynamic-allocation operators.

Validate the exact acceptance contract locally:

```bash
python tools/validate_submission.py model.onnx
```

`ACCEPTED` means the file meets the server's structural and isolated CPU
inference checks, and one Level 3 match played on your machine fits the
server's match limits.

### Limits

| | CPU time | Total time | Memory |
|---|---|---|---|
| Check (every upload) | 2 s | 3 s | 384 MiB |
| Level 3 match | 15 s | 20 s | 512 MiB |

`tools/validate_submission.py` measures both on your machine and stops a little
earlier (1.8 s and 13.5 s of CPU time, 320 MiB and 425 MiB resident) so a
model it accepts has some headroom. When it rejects a model it prints the
same message the server shows, followed by what it measured. Matches differ
in length and machines differ in speed, so keep clear of the limits rather
than just under them.

## Export traps

- Export all three batch axes as dynamic. The search evaluates up to 8 positions per call
  and the validator also checks batch 16.
- Keep weights embedded in `model.onnx`; disable exporter external-data mode.
- Return value shape `(batch,)`, not `(batch, 1)`.
- Export inference mode so dropout and batch normalization are frozen.

The server validates models in a tightly resource-limited process and runs a
fixed, deterministic dry-run corpus of board positions at batch sizes 1 and
16 -- not just one all-ones tensor, but real, legal, sparse boards from
seeded rollouts plus a few numeric extremes, so a model that is cheap on one
contrived input and expensive on a real position cannot pass by accident.
This prevents an upload from consuming the API worker's normal memory budget
during validation. It is a format and resource check, not a claim that
native ML runtimes are a security sandbox.

### Supported ONNX subset

The accepted operator set is an **allow-list, not a deny-list**: an operator
that is not explicitly listed below is rejected, including any operator this
validator has never heard of. The list is intentionally broad enough for
ordinary convolutional, residual, MLP, normalization, and attention-style
policy/value networks (attention decomposes into `MatMul`/`Softmax`/
`Transpose` at the supported opset range), but it excludes graph features
that can hide unbounded work or make a graph's output size a function of
*input values* rather than input shape.

- Use standard `ai.onnx` opset **13 through 18**. No other domain is
  accepted -- not even `ai.onnx.ml`.
- Do not use local functions, training graphs, sparse tensors, nested graphs,
  sequence/optional values, or external tensor data.
- Allowed operators: elementwise arithmetic and comparisons (`Add`, `Sub`,
  `Mul`, `Div`, `Pow`, `Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Min`, `Max`,
  `Sum`, `Mean`, `Clip`, `Equal`, `Greater`, `GreaterOrEqual`, `Less`,
  `LessOrEqual`, `And`, `Or`, `Not`, `Xor`, `Where`); activations (`Relu`,
  `LeakyRelu`, `PRelu`, `Elu`, `Selu`, `Sigmoid`, `Tanh`, `Softplus`,
  `Softsign`, `HardSigmoid`, `Softmax`, `LogSoftmax`); linear algebra and
  convolution (`Conv`, `Gemm`, `MatMul`); normalization in inference mode
  (`BatchNormalization`, `InstanceNormalization`, `LayerNormalization`);
  pooling (`AveragePool`, `MaxPool`, `GlobalAveragePool`,
  `GlobalMaxPool`); structural reshaping (`Identity`, `Cast`, `Flatten`,
  `Transpose`, `Concat`, `Reshape`, `Squeeze`, `Unsqueeze`, `Split`);
  reductions (`ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`,
  `ReduceProd`); and constrained indexing (`Gather`, `Slice`).
- For `Reshape`, `Squeeze`, `Unsqueeze`, `Split`, `Slice`, and `Gather`, every
  shape- or index-like input (the new shape, the axes, the slice bounds, the
  gather indices) must be **baked into the graph as a constant** at export
  time -- an initializer, or the output of a `Constant` node -- not computed
  from the observation at runtime. A shape argument that varies with the
  input is exactly how a tiny file can allocate an arbitrarily large buffer
  on a real board while looking cheap on a synthetic one; this validator
  checks the graph for that pattern directly rather than trusting a single
  dry-run input to expose it.
- `Cast` may only target `float32`, `float16`, `int32`, `int64`, or `bool`.
- `BatchNormalization` must run in inference mode: a single output, and
  `training_mode` absent or `0`. Export in eval mode (`model.eval()` in
  PyTorch, `training=False` in Keras).
- Everything else standard ONNX defines -- including `If`, `Loop`, `Scan`,
  every `Sequence`/`Optional` operator, `Pad`, `OneHot`, `ScatterND`,
  `ScatterElements`, `Expand`, `Tile`, `Resize`, `Range`, `ConstantOfShape`,
  `NonZero`, `Compress`, `Unique`, and the `Random*`/`Multinomial`/
  `Bernoulli` family -- is rejected. None of these are needed for a
  conventional policy/value network, and several of them are exactly how a
  tiny file can hide a large, input-dependent allocation.
- Models have structural and decoded-weight budgets in addition to the 5 MiB
  upload limit. Keep the graph conventional: no generated large constants or
  thousands of trivial nodes.

The validation process has a 192 MiB virtual-address-space cap and a short CPU
deadline. This is deliberately below the pod's 512 MiB budget so only one
validation process runs at a time without crowding out the API worker. A model
can be valid but still be rejected if it cannot initialize and run within these
published limits.

You still train in PyTorch, JAX, TensorFlow, or whatever you like. ONNX is
just the export format.

### Two export traps

Every framework's exporter can fall into these, and both are **silent** — the
export succeeds and the file looks fine. Together they account for most
rejected submissions.

**External weights.** Exporters may split large tensors out into a sidecar
file next to the `.onnx` — commonly `model.onnx.data`. You then submit the
`.onnx` alone, as instructed, and the referee receives a graph with no weights
in it. Look for a second file appearing next to your export; if your exporter
has a switch for this (PyTorch calls it `external_data`, and 2.9+ turns it on
by default), turn it off.

**A frozen batch axis.** Exporters trace your network on one example input and
often bake that input's batch size into the graph, pinning it to batch 1. The
search evaluates up to 8 positions per call and the validator checks batch sizes 1 and 16,
so such a model is rejected.
Whatever your framework names the option — PyTorch's `dynamic_axes`, TF's
`None` batch dimension, JAX's polymorphic shapes — mark axis 0 of all three
tensors as dynamic.

Both are caught by `tools/validate_submission.py`, so the cost of hitting one
is a rerun rather than a failed submission. Run it every time you export, not
just before you upload.

---

## Why the referee runs the search

Your model is a **position evaluator**. The referee wraps it in a Monte-Carlo
tree search (MCTS) and that search chooses the moves.

### What MCTS actually is

It is much simpler than its reputation. Given a position, the search:

1. asks your model "how good is this position, and which moves look
   promising?";
2. plays the most promising move in its head and asks again;
3. repeats a few hundred times, building a tree of what it explored;
4. plays the move it spent the most visits on.

That is the whole idea: **your model judges positions, the search does the
lookahead.** A model that evaluates well but cannot plan gets planning for
free.

The exact algorithm is published in `colony_one/mcts.py`. You can run it
locally with your own model to see precisely what the referee will do.

### Why this arrangement

Two reasons, and it is worth being honest about which is which.

**It is the fairness mechanism, not the security mechanism.** The security
comes entirely from ONNX. But the shared search means every submission gets
the same lookahead: the tournament measures *evaluation quality*, not who
wrote the fastest tree search in C++. It also means a plain supervised or
reinforcement-learned evaluator competes on even terms with a full AlphaZero
implementation.

**It keeps the submission simple.** You ship one file with a fixed signature.
No environment, no dependencies, no build.

### The referee's settings

```python
from colony_one import mcts

config = mcts.MCTSConfig()
action = mcts.search_action(
    env.board, env.current_team, env.actions_left, env.round_number,
    evaluator, config,
)
```

`evaluator` accepts `(B, 16, 8, 8)` float32 observations and returns policy
logits `(B, 4096)` plus values `(B,)`. `MCTSConfig()` is the referee
configuration, except for the number of simulations, which the server sets per
level (60 in Level 3, 100 in the Level 4 tournament; 300 is only the local
default):

```text
simulations       60 in Level 3, 100 in the Level 4 tournament
c_puct            1.5
batch_size         8
virtual_loss       1.0
dirichlet_alpha    0.3
dirichlet_eps      0.15
```

Root Dirichlet noise is enabled and root actions are sampled during the first
eight actions: temperature `1.0` for actions 1-4, `0.5` for actions 5-8, then
greedy selection. Call the search once per action, not once per two-action
turn.

## Output mappings

- Policy-value / AlphaZero: export policy logits and value directly.
- Actor-critic: export actor logits and a value constrained to `[-1, 1]`.
- Q-learning: Q-values may be policy logits; derive a bounded value inside the
  exported graph.
- Any method is valid if it produces the required tensors.

The referee masks legal actions itself. Mask them during training as well.

- [ ] one `.onnx` file, under 5 MiB
- [ ] input `(batch, 16, 8, 8)` float32, batch axis **dynamic**
- [ ] weights **inside** the file — no `.onnx.data` sidecar
- [ ] exactly two outputs, in order: policy `(batch, 4096)`, value `(batch,)`
- [ ] value has shape `(batch,)`, not `(batch, 1)`
- [ ] policy is **raw logits**
- [ ] value is in `[-1, 1]`, from the **team to move's** perspective
- [ ] `ai.onnx` opset 13-18 only; only the allow-listed operators, with
      constant shape/index arguments where required (see above)
- [ ] exported in inference mode, so dropout and batch-norm are frozen
      (`model.eval()` in PyTorch, `training=False` in Keras)
- [ ] `tools/validate_submission.py` prints `ACCEPTED`
- [ ] beats `GreedyAgent` more often than not
