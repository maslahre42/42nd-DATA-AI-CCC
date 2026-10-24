"""Portable, OS-independent submission acceptance policy for Colony: One.

This module holds the *policy* -- the exact contract a submitted ONNX model
must satisfy -- shared by two different enforcement wrappers:

* ``submission_validation.py``, the Linux server validator, which runs this
  policy inside a ``resource``-limited, ``fcntl``-locked child process. That
  is the authoritative, security-critical path.
* ``submission_check.py``, the cross-platform participant tool, which runs
  the *same* policy without POSIX-only sandboxing, so participants on
  Windows and macOS can check a model before submitting it.

To keep that split real rather than aspirational, this module must never
import ``fcntl``, ``resource``, or any other POSIX-only module, directly or
transitively -- doing so would make it (and everything that imports it)
fail to import on Windows. Only the two wrappers above may do that, and
only inside the functions that actually need it.

ONNX is untrusted input. Every check in here is designed to run in a
process that has not yet decided the file is safe: it inspects a bounded
number of structural properties before it hands the graph to ONNX Runtime,
and even the dry run below is deliberately against a small, fixed, and
*adversarial* corpus of inputs -- not just the single all-ones tensor a much
earlier version of this validator used, which a value-dependent-allocation
model can pass trivially by staying cheap on that one input and expensive on
every real board it will actually see in a match.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from colony_one import encoding

MAX_MODEL_BYTES = 5 * 1024 * 1024

# These caps are deliberately structural and cheap to inspect. The file-size
# cap alone does not bound protobuf object counts or decoded tensor sizes.
MAX_NODES = 1_024
MAX_INITIALIZERS = 512
MAX_TENSORS = 1_024
MAX_ATTRIBUTES = 4_096
MAX_TENSOR_RANK = 8
MAX_TENSOR_BYTES = 16 * 1024 * 1024
MAX_TOTAL_TENSOR_BYTES = 32 * 1024 * 1024
MIN_STANDARD_OPSET = 13
MAX_STANDARD_OPSET = 18

# The wall-clock budget for the whole dry run (parse + session creation +
# every batch below). This is a plain integer, not a POSIX rlimit, so it is
# safe to share between both wrappers: the Linux wrapper additionally
# enforces a CPU-time rlimit inside this window, and the portable wrapper
# relies on this alone.
VALIDATION_TIMEOUT_SECONDS = 3
CHECK_CPU_SECONDS = 2
CHECK_ADDRESS_SPACE_MIB = 384
LEVEL3_SIMULATIONS = 60
MATCH_CPU_SECONDS = 15
MATCH_TIMEOUT_SECONDS = 20
MATCH_ADDRESS_SPACE_MIB = 512

TOO_SLOW_TO_CHECK = (
    "Your model is too slow to check: the check must finish within "
    f"{CHECK_CPU_SECONDS} seconds of CPU time and {VALIDATION_TIMEOUT_SECONDS} seconds "
    "in total. Use a smaller or simpler network."
)
TOO_MUCH_MEMORY_TO_CHECK = (
    "Your model needs more memory than allowed while it is checked "
    f"({CHECK_ADDRESS_SPACE_MIB} MiB). Use a smaller network."
)
TOO_SLOW_TO_PLAY = (
    "Your model is too slow to play a match: a Level 3 match must finish within "
    f"{MATCH_CPU_SECONDS} seconds of CPU time and {MATCH_TIMEOUT_SECONDS} seconds "
    "in total. Use a smaller or simpler network."
)
TOO_MUCH_MEMORY_TO_PLAY = (
    "Your model needs more memory than allowed while it is played "
    f"({MATCH_ADDRESS_SPACE_MIB} MiB). Use a smaller network."
)
RUNTIME_CRASHED = (
    "The ONNX runtime crashed while running your model. Re-export it "
    '(see docs/submission.md, "Export traps") and run tools/validate_submission.py.'
)
MEMORY_ERROR_MARKERS = (
    "failed to map segment",
    "Failed to serialize proto",
    "Failed to allocate memory",
    "bad_alloc",
    "MemoryError",
)
VERDICT_PREFIXES = ("REJECTED: ", "INTERNAL_ERROR: ", "SANDBOX_UNAVAILABLE: ")

ALLOWED_DOMAINS = frozenset({"", "ai.onnx"})
"""Operator domains a submission may use.

``""`` and ``"ai.onnx"`` are both spellings of the standard operator set.
Anything else -- including ``ai.onnx.ml`` -- means a custom or non-core
operator, which is either native code or outside the reviewed operator
allow-list below, either of which this validator does not run.
"""

ALLOWED_OPERATORS = frozenset({
    # Elementwise arithmetic and comparisons. Every one of these has an
    # output shape derivable from static input shapes alone -- none of them
    # can turn a runtime *value* into a buffer size.
    "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Abs", "Sqrt", "Exp", "Log",
    "Min", "Max", "Sum", "Mean", "Clip",
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "And", "Or", "Not", "Xor", "Where",
    # Activations.
    "Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Sigmoid", "Tanh",
    "Softplus", "Softsign", "HardSigmoid", "Softmax", "LogSoftmax",
    # Linear algebra / convolution. Weight shapes are graph attributes or
    # initializer shapes, never runtime values.
    "Conv", "Gemm", "MatMul",
    # Normalization (inference-mode only; see _check_batch_normalization).
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    # Pooling.
    "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
    # Shape-preserving structural operators.
    "Identity", "Cast", "Flatten", "Transpose", "Concat",
    # Shape transforms whose shape-like inputs must be constant -- see
    # CONSTANT_INPUT_POSITIONS. Without that constraint these are exactly
    # the value-dependent-allocation operators the tournament threat model
    # exists to exclude.
    "Reshape", "Squeeze", "Unsqueeze", "Split",
    # Reductions.
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    # Constrained indexing -- see CONSTANT_INPUT_POSITIONS.
    "Gather", "Slice",
})
"""The complete set of operators a submission may use.

This is an allow-list, not a deny-list: an operator not named here is
rejected, including any future ONNX operator this validator has never heard
of. Extending it requires deciding, for the new operator, whether any of its
inputs are shape-like (see ``CONSTANT_INPUT_POSITIONS``) and adding a test
that a non-constant use of that input is still rejected.

Notably absent, and deliberately so: ``Pad``, ``OneHot``, ``ScatterND``,
``ScatterElements``, ``Expand``, ``Tile``, ``Resize``, ``Range``,
``ConstantOfShape``, ``NonZero``, ``Compress``, ``Unique``, the
``Random*``/``Multinomial``/``Bernoulli`` family, control flow (``If``,
``Loop``, ``Scan``), and every ``Sequence``/``Optional`` operator. Several
of these (particularly ``Pad`` with a non-constant ``pads`` input) can make
a graph's output size a function of the *input values* rather than its
shape, so a validator that only tests one board position -- however
carefully chosen -- can never rule them out by dry-running alone.
"""

CONSTANT_INPUT_POSITIONS: dict[str, tuple[int, ...]] = {
    "Reshape": (1,),
    "Squeeze": (1,),
    "Unsqueeze": (1,),
    "Slice": (1, 2, 3, 4),
    "Split": (1,),
    "Gather": (1,),
}
"""Zero-based input positions that must be constant for the given operator.

An operator's shape-like input (a target shape, axis list, or index tensor)
becomes a value-dependent allocation the moment it can vary at runtime. All
of these operators are legitimate and common in policy/value networks when
their shape-like arguments are fixed at export time -- so instead of
banning them outright, the graph is required to have baked those arguments
in as constants, which the checker below verifies from the model itself
rather than trusting the exporter.
"""

ALLOWED_CAST_TYPES = frozenset({"FLOAT", "FLOAT16", "INT32", "INT64", "BOOL"})
"""Target element types :func:`_check_cast` permits for a ``Cast`` node.

A plain float32 policy/value network never needs to cast into an exotic or
sub-byte numeric type; restricting the target keeps ``Cast`` from becoming a
back door into representations the rest of this policy does not reason
about.
"""


class ModelRejected(ValueError):
    """A submitted model failed the tournament acceptance contract."""


def _reject(message: str) -> None:
    raise ModelRejected(message)


def _checked_product(values: list[int] | tuple[int, ...], label: str) -> int:
    result = 1
    for value in values:
        if value < 0:
            _reject(f"{label} has a negative dimension")
        if value and result > MAX_TOTAL_TENSOR_BYTES // value:
            _reject(f"{label} is too large")
        result *= value
    return result


def _tensor_element_size(tensor: Any) -> int:
    # ONNX permits several compact encodings in protobuf. Count decoded
    # bytes, rather than trusting the serialized representation.
    from onnx import TensorProto

    sizes = {
        "BOOL": 1,
        "FLOAT8E4M3FN": 1,
        "FLOAT8E4M3FNUZ": 1,
        "FLOAT8E5M2": 1,
        "FLOAT8E5M2FNUZ": 1,
        "INT8": 1,
        "UINT8": 1,
        "BFLOAT16": 2,
        "FLOAT16": 2,
        "INT16": 2,
        "UINT16": 2,
        "FLOAT": 4,
        "INT32": 4,
        "UINT32": 4,
        "FLOAT64": 8,
        "INT64": 8,
        "UINT64": 8,
        "COMPLEX64": 8,
        "COMPLEX128": 16,
    }
    try:
        return sizes[TensorProto.DataType.Name(tensor.data_type)]
    except KeyError:
        _reject(f"Tensor '{tensor.name or '<unnamed>'}' uses an unsupported data type")


def _check_tensor(tensor: Any, state: dict[str, int], label: str) -> None:
    from onnx import TensorProto

    state["tensors"] += 1
    if state["tensors"] > MAX_TENSORS:
        _reject(f"Model has more than {MAX_TENSORS} tensors")
    if len(tensor.dims) > MAX_TENSOR_RANK:
        _reject(f"{label} has rank over {MAX_TENSOR_RANK}")
    if any(dimension <= 0 for dimension in tensor.dims):
        _reject(f"{label} has a non-positive dimension")
    if (
            tensor.data_location == TensorProto.EXTERNAL
            or tensor.external_data
    ):
        _reject(f"Tensor '{tensor.name or '<unnamed>'}' uses external weight data")
    elements = _checked_product(list(tensor.dims), label)
    decoded_bytes = elements * _tensor_element_size(tensor)
    if decoded_bytes > MAX_TENSOR_BYTES:
        _reject(
            f"{label} exceeds the {MAX_TENSOR_BYTES // (1024 * 1024)} MiB tensor limit"
        )
    state["tensor_bytes"] += decoded_bytes
    if state["tensor_bytes"] > MAX_TOTAL_TENSOR_BYTES:
        _reject(
            "Decoded model tensors exceed the "
            f"{MAX_TOTAL_TENSOR_BYTES // (1024 * 1024)} MiB aggregate limit"
        )


def _check_value_info(
        value_info: Any,
        *,
        label: str,
        expected_shape: tuple[int, ...],
        batch_name: str | None = None,
) -> str:
    from onnx import TensorProto

    if not value_info.type.HasField("tensor_type"):
        _reject(f"{label} must be a tensor")
    tensor = value_info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        _reject(f"{label} must use float32")
    shape = tensor.shape
    if len(shape.dim) != len(expected_shape) + 1:
        _reject(f"{label} must have rank {len(expected_shape) + 1}")

    batch = shape.dim[0]
    if batch.HasField("dim_value") or not batch.HasField("dim_param"):
        _reject(f"{label} batch dimension must be a named dynamic dimension")
    if batch_name is not None and batch.dim_param != batch_name:
        _reject(f"{label} batch dimension must match the input batch dimension")

    for position, expected in enumerate(expected_shape, start=1):
        dim = shape.dim[position]
        if not dim.HasField("dim_value") or dim.dim_value != expected:
            expected_str = ", ".join(map(str, expected_shape))
            _reject(f"{label} shape must be [batch, {expected_str}]")
    return batch.dim_param


def _check_sparse_tensor(sparse: Any, state: dict[str, int], label: str) -> None:
    # Sparse initializers are unnecessary for the published neural-network
    # format and complicate both decoded-size accounting and external-data
    # policy.
    del sparse, state
    _reject(f"{label} uses a sparse tensor, which is not allowed")


def _constant_names(graph: Any) -> set[str]:
    """Names of tensors whose value is fixed at export time.

    Covers graph initializers and the outputs of ``Constant`` nodes. This is
    intentionally a single, shallow pass -- it does not fold arbitrary
    ``Identity``/``Cast`` chains -- so a shape-like input must be *directly*
    one of these to count as constant. That is a stricter requirement than
    strictly necessary for every legitimate export, but a false rejection
    here is a quick, clear export fix; a false acceptance is exactly the gap
    this policy exists to close.
    """
    names = {initializer.name for initializer in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            names.add(node.output[0])
    return names


def _check_constant_inputs(node: Any, constants: set[str]) -> None:
    positions = CONSTANT_INPUT_POSITIONS.get(node.op_type)
    if not positions:
        return
    for position in positions:
        if position >= len(node.input):
            continue
        name = node.input[position]
        if not name:
            # An omitted optional input is serialized as an empty string.
            continue
        if name not in constants:
            _reject(
                f"Operator '{node.op_type}' input {position} must be a "
                "constant baked into the graph at export time, not a value "
                "computed at runtime"
            )


def _check_cast(node: Any) -> None:
    from onnx import AttributeProto, TensorProto

    if node.op_type != "Cast":
        return
    for attribute in node.attribute:
        if attribute.name == "to" and attribute.type == AttributeProto.INT:
            type_name = TensorProto.DataType.Name(attribute.i)
            if type_name not in ALLOWED_CAST_TYPES:
                _reject(f"Cast to '{type_name}' is not allowed in submissions")


def _check_batch_normalization(node: Any) -> None:
    from onnx import AttributeProto

    if node.op_type != "BatchNormalization":
        return
    if len(node.output) > 1:
        _reject(
            "BatchNormalization must run in inference mode (a single "
            "output) -- export the model in eval mode"
        )
    for attribute in node.attribute:
        if (
                attribute.name == "training_mode"
                and attribute.type == AttributeProto.INT
                and attribute.i
        ):
            _reject(
                "BatchNormalization training_mode is not allowed in "
                "submissions -- export the model in eval mode"
            )


def _visit_graph(graph: Any, state: dict[str, int], depth: int = 0) -> None:
    from onnx import AttributeProto

    if depth > 0:
        _reject("Nested ONNX graphs are not allowed")
    if len(graph.initializer) > MAX_INITIALIZERS:
        _reject(f"Graph has more than {MAX_INITIALIZERS} initializers")
    for initializer in graph.initializer:
        label = f"Initializer '{initializer.name or '<unnamed>'}'"
        _check_tensor(initializer, state, label)
    for sparse in graph.sparse_initializer:
        _check_sparse_tensor(sparse, state, "Graph")

    constants = _constant_names(graph)

    for node in graph.node:
        state["nodes"] += 1
        if state["nodes"] > MAX_NODES:
            _reject(f"Model has more than {MAX_NODES} nodes")
        domain = node.domain or ""
        if domain not in ALLOWED_DOMAINS:
            _reject(f"Custom operator domain is not allowed: {domain or '<default>'}")
        if node.op_type not in ALLOWED_OPERATORS:
            _reject(f"Operator '{node.op_type}' is not allowed in submissions")
        _check_constant_inputs(node, constants)
        _check_cast(node)
        _check_batch_normalization(node)

        state["attributes"] += len(node.attribute)
        if state["attributes"] > MAX_ATTRIBUTES:
            _reject(f"Model has more than {MAX_ATTRIBUTES} attributes")
        for attribute in node.attribute:
            if attribute.type == AttributeProto.TENSOR:
                _check_tensor(attribute.t, state, f"Attribute '{attribute.name}'")
            elif attribute.type == AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    _check_tensor(tensor, state, f"Attribute '{attribute.name}'")
            elif attribute.type == AttributeProto.SPARSE_TENSOR:
                _check_sparse_tensor(attribute.sparse_tensor, state, "Attribute")
            elif attribute.type == AttributeProto.SPARSE_TENSORS:
                for sparse in attribute.sparse_tensors:
                    _check_sparse_tensor(sparse, state, "Attribute")
            elif attribute.type == AttributeProto.GRAPH:
                _visit_graph(attribute.g, state, depth + 1)
            elif attribute.type == AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    _visit_graph(nested, state, depth + 1)


def check_domains_and_opset(model: Any) -> None:
    """Reject unsupported domains, opset versions, or non-inference graphs."""
    if model.training_info:
        _reject("Training graphs are not allowed")
    if model.functions:
        _reject("Local ONNX functions are not allowed")

    imports: dict[str, int] = {}
    for item in model.opset_import:
        domain = item.domain or ""
        if domain in imports:
            _reject(f"Duplicate opset import for domain '{domain or 'ai.onnx'}'")
        imports[domain] = item.version
    if set(imports) - ALLOWED_DOMAINS:
        foreign = ", ".join(sorted(set(imports) - ALLOWED_DOMAINS))
        _reject(f"Custom operator domain is not allowed: {foreign}")
    if not imports:
        _reject("Model has no opset import")
    for domain, version in imports.items():
        if not MIN_STANDARD_OPSET <= version <= MAX_STANDARD_OPSET:
            _reject(
                f"Opset {version} for domain '{domain or 'ai.onnx'}' is unsupported; "
                f"use opset {MIN_STANDARD_OPSET}-{MAX_STANDARD_OPSET}"
            )


def check_io_contract(model: Any) -> str:
    """Enforce the one-input/two-output shape contract. Returns the batch name."""
    from onnx import TensorProto

    if len(model.graph.input) != 1:
        _reject("Model must have exactly one graph input")
    if len(model.graph.output) != 2:
        _reject("Model must have exactly two graph outputs (policy, then value)")
    input_name = model.graph.input[0].name
    if any(initializer.name == input_name for initializer in model.graph.initializer):
        _reject("Model input must not be an initializer")

    batch_name = _check_value_info(
        model.graph.input[0],
        label="input",
        expected_shape=(encoding.NUM_PLANES, 8, 8),
    )
    _check_value_info(
        model.graph.output[0],
        label="policy output",
        expected_shape=(encoding.NUM_ACTIONS,),
        batch_name=batch_name,
    )
    _check_value_info(
        model.graph.output[1],
        label="value output",
        expected_shape=(),
        batch_name=batch_name,
    )

    # Referencing TensorProto here makes the caller reject malformed graph
    # input type declarations before ORT sees them.
    if model.graph.input[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        _reject("input must use float32")
    return batch_name


def check_graph_structure(model: Any) -> None:
    """Enforce structural caps and the operator allow-list on the whole graph."""
    state = {"nodes": 0, "attributes": 0, "tensors": 0, "tensor_bytes": 0}
    _visit_graph(model.graph, state)


def validate_submission_contract(model: Any) -> None:
    """The full authoritative acceptance policy for a parsed ONNX model."""
    check_domains_and_opset(model)
    check_io_contract(model)
    check_graph_structure(model)


# ---------------------------------------------------------------------------
# Adversarial dry-run corpus
# ---------------------------------------------------------------------------
def _rollout_observations(seed: int, max_plies: int) -> list[np.ndarray]:
    """Real, legal observations from a seeded random self-play rollout.

    Returns the initial observation plus one after every ply actually
    played (stopping early if the game ends), so with ``max_plies >= 3`` the
    first four entries alone already cover both ``actions_left`` values
    (2 and 1) and both ``side_to_move`` values (CYAN and MAGENTA) on real
    board states -- unlike a synthetic all-ones or all-zeros tensor, these
    are boards that can actually occur in a match.
    """
    from colony_one.baselines import RandomAgent
    from colony_one.env import ColonyOneEnv

    env = ColonyOneEnv(seed=seed)
    agent = RandomAgent(seed=seed)
    observations = [env.observation()]
    for _ in range(max_plies):
        if env.terminated:
            break
        action = agent.select(
            env.board, env.current_team, env.actions_left, env.round_number
        )
        if action is None:
            break
        env.step(action)
        observations.append(env.observation())
    return observations


def _plane_isolation_cases() -> list[np.ndarray]:
    """One synthetic case per plane: that plane saturated, everything else zero.

    Not a reachable game state, but a cheap way to probe every plane's
    numeric extreme independently, including planes (like the empty-square
    count implied by the piece planes) that a value-dependent-allocation
    model might key on.
    """
    cases = []
    for plane in range(encoding.NUM_PLANES):
        planes = np.zeros((encoding.NUM_PLANES, 8, 8), dtype=np.float32)
        planes[plane] = 1.0
        cases.append(planes)
    return cases


def _validation_corpus() -> list[np.ndarray]:
    """The fixed, deterministic set of single-position dry-run inputs.

    All-zeros and all-ones are kept as blunt numeric extremes, but the bulk
    of this corpus is real, legal, *sparse* board states reached by seeded
    random self-play -- the class of input the shipped baseline validator
    never exercised, and exactly the class a value-dependent-allocation
    model (see SECURITY_REVIEW_ai-colony-one.md P1-2) is cheap on for an
    all-ones dry run and expensive on in a real match.
    """
    cases: list[np.ndarray] = [
        np.zeros((encoding.NUM_PLANES, 8, 8), dtype=np.float32),
        np.ones((encoding.NUM_PLANES, 8, 8), dtype=np.float32),
    ]
    # Early game: covers both actions_left values and both side-to-move
    # values on real, legal boards.
    cases.extend(_rollout_observations(seed=0, max_plies=3))
    # A sparse mid-game position.
    cases.extend(_rollout_observations(seed=1, max_plies=30)[-1:])
    # A sparse, likely near-endgame position (more captures, more locked
    # squares from firewall shrinks).
    cases.extend(_rollout_observations(seed=2, max_plies=90)[-1:])
    cases.extend(_plane_isolation_cases())
    return cases


def _stack_batch(cases: list[np.ndarray], batch_size: int) -> np.ndarray:
    return np.stack([cases[i % len(cases)] for i in range(batch_size)])


def _check_batch_outputs(policy: Any, value: Any, batch: int) -> None:
    if not isinstance(policy, np.ndarray) or policy.dtype != np.float32:
        _reject("policy output must produce float32 at runtime")
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        _reject("value output must produce float32 at runtime")
    if policy.shape != (batch, encoding.NUM_ACTIONS):
        _reject(
            f"policy output is {policy.shape} at batch {batch}, expected "
            f"({batch}, {encoding.NUM_ACTIONS})"
        )
    if value.shape != (batch,):
        _reject(f"value output is {value.shape} at batch {batch}, expected ({batch},)")
    if not np.isfinite(policy).all() or not np.isfinite(value).all():
        _reject("Model outputs must be finite")
    if np.any(value < -1.0) or np.any(value > 1.0):
        _reject("value output must stay within [-1, 1]")


def child_environment() -> dict[str, str]:
    """A minimal, credential-free environment for the dry-run child process.

    Shared by both wrappers: neither should let an upload inherit cloud
    credentials, proxy settings, Python path overrides, or multi-threaded
    BLAS settings from the parent process.
    """
    import tempfile

    return {
        "HOME": tempfile.gettempdir(),
        "LANG": "C",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }


def inspect_and_dry_run_model(file_path: Path, max_size: int = MAX_MODEL_BYTES) -> float:
    """Parse, inspect, load, and dry-run a model. Returns the dry-run seconds.

    This is the entire acceptance policy in one call: it does not apply any
    resource limits itself, so the caller (the Linux child in
    ``submission_validation.py``, or the best-effort child in
    ``submission_check.py``) is responsible for isolation. Both wrappers
    call this exact function so that "accepted locally" and "accepted by the
    server" mean the same policy decision.
    """
    import onnx
    import onnxruntime as ort

    data = file_path.read_bytes()
    if len(data) > max_size:
        _reject(f"Model exceeds the {max_size}-byte size limit")
    if not data:
        _reject("Uploaded model is empty")
    try:
        # Check the bytes directly: onnx.checker.check_model(ModelProto) internally
        # calls model.SerializeToString(), which would hold data + proto tree +
        # re-serialized copy simultaneously.  Passing bytes skips that extra
        # allocation and keeps peak virtual address space within RLIMIT_AS.
        onnx.checker.check_model(data)
        model = onnx.load_from_string(data)
    except ModelRejected:
        raise
    except Exception as exc:
        _reject(f"Invalid ONNX model: {str(exc)[:300]}")
    validate_submission_contract(model)

    start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            str(file_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        reason = plain_runtime_error(str(exc))[:300]
        _reject(f"Model dry run failed to create a CPU session: {reason}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 2:
        _reject("Runtime input/output contract mismatch")

    corpus = _validation_corpus()
    batches = (
        np.expand_dims(corpus[0], 0),  # batch 1, all-zeros
        np.expand_dims(corpus[1], 0),  # batch 1, all-ones
        _stack_batch(corpus, 16),  # batch 16, heterogeneous real + extreme boards
    )
    for batch_planes in batches:
        batch = batch_planes.shape[0]
        try:
            policy, value = session.run(
                [outputs[0].name, outputs[1].name],
                {inputs[0].name: batch_planes},
            )
        except Exception as exc:
            reason = plain_runtime_error(str(exc))[:300]
            _reject(f"Model dry run failed at batch {batch}: {reason}")
        _check_batch_outputs(policy, value, batch)

    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration < 0:
        _reject("Model dry run reported an invalid duration")
    return duration


def read_limited(handle: Any, limit: int = 4_096) -> str:
    """Read at most ``limit`` bytes from a seekable handle, decoded leniently."""
    handle.seek(0)
    return handle.read(limit).decode("utf-8", "replace").strip()


_TERMINAL_COLOURS = re.compile(r"\x1b\[[0-9;]*m")
_RUNTIME_STATUS = re.compile(r"\[ONNXRuntimeError\] : \d+ : \w+ : ")
_RUNTIME_LOAD_PATH = re.compile(r"Load model from \S+ failed:\s*")
_RUNTIME_NODE = re.compile(
    r"Non-zero status code returned while running (\w+) node\. Name:'([^']*)' "
    r"Status Message: "
)
_RUNTIME_INTERNALS = re.compile(
    r"/onnxruntime_src/\S+:\d+ (?:[^()]*?\([^()]*\)(?: const)? )?(?:[^.]*? was false\. )?"
)


def one_line(text: str) -> str:
    return " ".join(text.split())


def plain_runtime_error(text: str) -> str:
    """ONNX Runtime error text without status codes, source paths or C++ signatures."""
    text = _RUNTIME_STATUS.sub("", text)
    text = _RUNTIME_LOAD_PATH.sub("", text)
    text = _RUNTIME_NODE.sub(
        lambda m: f"{m[1]} node '{m[2]}' failed: " if m[2] else f"{m[1]} node failed: ",
        text,
    )
    return one_line(_RUNTIME_INTERNALS.sub("", text))


def read_verdict(handle: Any, limit: int = 8_192) -> str:
    """A child's last verdict line from its stderr tail, else that tail as one line."""
    handle.seek(0, os.SEEK_END)
    handle.seek(max(0, handle.tell() - limit))
    text = _TERMINAL_COLOURS.sub("", handle.read().decode("utf-8", "replace"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(VERDICT_PREFIXES):
            return line
    return " ".join(lines)
