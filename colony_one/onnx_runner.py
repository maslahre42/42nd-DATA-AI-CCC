"""Loading and running submitted ONNX models.

Why ONNX and not ``.pt``
------------------------
A PyTorch ``.pt`` file is a *pickle*. Loading one executes arbitrary Python
from inside the file -- ``torch.load`` on an untrusted checkpoint is equivalent
to running an untrusted script. No sandboxing of the model helps, because the
code runs before any model exists.

ONNX is a protobuf: a description of a computation graph, plus weights. It is
data, not code. Loading one cannot execute anything; the runtime walks the
graph and runs a fixed set of numeric kernels.

That is the whole reason the tournament accepts ONNX only. It is not an
inconvenience invented to annoy participants -- it is what makes it possible
to run strangers' models on one machine at all.

What this module adds on top
----------------------------
ONNX being data does not make every ONNX file safe to run, so the loader also:

* **caps the file size** -- a multi-gigabyte model is a denial of service;
* **rejects custom operators** -- only the standard ``ai.onnx`` operator
  allow-list published in :mod:`colony_one.submission_policy`, since a
  custom op is a native library, i.e. code again;
* **pins the CPU execution provider** -- no GPU, no surprise backends;
* **disables external weight files** -- a model must be one self-contained
  file, not a loader that reaches into the filesystem;
* **verifies the I/O signature** against the published contract, so a
  mismatch is a clear error at submission time rather than a mysterious
  failure mid-tournament.

The contract
------------
One input, two outputs::

    input   "observation"  (batch, 16, 8, 8)  float32
    output  "policy"       (batch, 4096)      float32   raw logits
    output  "value"        (batch,)           float32   in [-1, 1]

The batch dimension **must be dynamic** -- the referee's search evaluates
positions in batches (``MCTSConfig.batch_size``, 16 by default), so a model
exported with a fixed batch of 1 will fail. See ``docs/submission.md``.

Names are matched by *position*, not by string, so you may call your outputs
whatever you like.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from colony_one import encoding
from colony_one.submission_policy import ALLOWED_DOMAINS, MAX_MODEL_BYTES

__all__ = [
    "MAX_MODEL_BYTES",
    "ALLOWED_DOMAINS",
    "InvalidModelError",
    "OnnxEvaluator",
    "check_model_file",
    "check_operators",
]

# MAX_MODEL_BYTES and ALLOWED_DOMAINS are re-exported from
# colony_one.submission_policy -- the single source of truth for both,
# shared with the authoritative upload validator -- rather than duplicated
# here. A previous version of this module defined its own copies, and its
# own ALLOWED_DOMAINS included "ai.onnx.ml", which the upload validator has
# never accepted: a model using ai.onnx.ml operators would pass this
# evaluator-level check but be rejected at upload, or vice versa if this
# check ever ran first. Sharing the constant makes that drift structurally
# impossible instead of merely documented.

EXPECTED_OBS_SHAPE = (encoding.NUM_PLANES, 8, 8)
EXPECTED_POLICY_SIZE = encoding.NUM_ACTIONS


class InvalidModelError(ValueError):
    """A model does not satisfy the submission contract.

    The message is written to be shown directly to a participant.
    """


def _fail(message: str) -> InvalidModelError:
    return InvalidModelError(message)


def check_model_file(path: str | os.PathLike[str]) -> int:
    """Check the file itself, before anything parses it.

    Returns:
        The file size in bytes.

    Raises:
        InvalidModelError: Missing, empty, not a file, or over the size cap.
    """
    path = Path(path)
    if not path.exists():
        raise _fail(f"model file not found: {path}")
    if not path.is_file():
        raise _fail(f"not a file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise _fail(f"model file is empty: {path}")
    if size > MAX_MODEL_BYTES:
        mib = 1024 * 1024
        raise _fail(
            f"model is {size / mib:.1f} MiB, over the {MAX_MODEL_BYTES // mib} MiB "
            "limit. Reduce the network size, or export weights as float16."
        )
    return size


def check_operators(path: str | os.PathLike[str]) -> list[str]:
    """Reject custom operators and anything outside the operator allow-list.

    Only the standard operator set is allowed; a custom op means a native
    library the referee would have to load. This is a lighter check than
    the authoritative upload validator (:mod:`colony_one.submission_policy`)
    -- no structural caps, no dry run -- but it enforces the same domain and
    operator allow-list, so a model loaded directly through
    :class:`OnnxEvaluator` (bypassing the upload endpoint, e.g. from a local
    script) cannot use anything the upload path would have rejected.

    Returns:
        The operator domains the model uses.

    Raises:
        InvalidModelError: The file will not parse, uses a foreign domain,
            or uses an operator outside the allow-list.
    """
    from colony_one.submission_policy import ALLOWED_OPERATORS

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise _fail(
            "the 'onnx' package is required to validate a model: pip install onnx"
        ) from exc

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise _fail(f"file is not a valid ONNX model: {exc}") from exc

    domains = {opset.domain for opset in model.opset_import}
    foreign = sorted(d for d in domains if d not in ALLOWED_DOMAINS)
    if foreign:
        raise _fail(
            f"model uses non-standard operator domain(s): {', '.join(foreign)}. "
            "Only standard ONNX operators are accepted -- custom operators are "
            "native code and cannot be run on the tournament server."
        )

    disallowed = sorted({
        node.op_type for node in model.graph.node
        if node.op_type not in ALLOWED_OPERATORS
    })
    if disallowed:
        raise _fail(
            f"model uses operator(s) not on the submission allow-list: "
            f"{', '.join(disallowed)}. See docs/submission.md for the "
            "supported operator set."
        )

    # A model must be self-contained. External data is a pointer to another
    # file on disk, which both breaks a single-file submission and lets a
    # model reference paths outside its own directory.
    for initializer in model.graph.initializer:
        if initializer.HasField("data_location") and initializer.data_location != 0:
            raise _fail(
                f"tensor '{initializer.name}' stores its weights in an external "
                "file. Export with all weights inside the .onnx file -- turn "
                "off your exporter's external-data option."
            )

    return sorted(domains)


class OnnxEvaluator:
    """Runs a submitted model as an :class:`~colony_one.mcts.Evaluator`.

    This is exactly what the tournament referee uses, so a model that works
    here works there.

    Example::

        from colony_one import mcts
        from colony_one.onnx_runner import OnnxEvaluator

        evaluator = OnnxEvaluator("model.onnx")
        action = mcts.search_action(
            board, team, actions_left, round_number, evaluator
        )

    Args:
        path: Path to the ``.onnx`` file.
        validate: Run the file-size and operator checks on load. Leave on;
            it is off only in tests that construct known-good models.
        num_threads: Intra-op threads for the runtime. The referee pins this
            so that every submission gets the same compute.

    Raises:
        InvalidModelError: The model fails any contract check.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        validate: bool = True,
        num_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise _fail(
                "the 'onnxruntime' package is required to run a model: "
                "pip install onnxruntime"
            ) from exc

        self.path = Path(path)
        if validate:
            check_model_file(self.path)
            check_operators(self.path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        # Log errors only: a submission must not be able to flood the
        # referee's output.
        options.log_severity_level = 3

        try:
            self._session = ort.InferenceSession(
                str(self.path),
                sess_options=options,
                providers=["CPUExecutionProvider"],  # CPU only, always
            )
        except Exception as exc:
            raise _fail(f"onnxruntime could not load the model: {exc}") from exc

        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        if len(outputs) < 2:
            raise _fail(
                f"model has {len(outputs)} output(s); exactly 2 are required "
                "(policy logits, then value). See docs/submission.md."
            )
        self._policy_name = outputs[0].name
        self._value_name = outputs[1].name

        if validate:
            self._check_signature()

    # ------------------------------------------------------------------
    def _check_signature(self) -> None:
        """Verify the declared I/O shapes against the published contract.

        Symbolic dimensions (strings) are left alone -- that is how a dynamic
        batch axis appears, and it is required. Only concrete mismatches fail.
        """
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise _fail(
                f"model takes {len(inputs)} inputs; exactly 1 is required "
                f"(the (batch, {encoding.NUM_PLANES}, 8, 8) observation)."
            )

        shape = inputs[0].shape
        if len(shape) != 4:
            raise _fail(
                f"input has {len(shape)} dimensions {shape}; 4 are required: "
                f"(batch, {encoding.NUM_PLANES}, 8, 8)."
            )
        for axis, expected in zip((1, 2, 3), EXPECTED_OBS_SHAPE, strict=True):
            actual = shape[axis]
            if isinstance(actual, int) and actual != expected:
                raise _fail(
                    f"input dimension {axis} is {actual}, expected {expected}. "
                    f"The observation is (batch, {encoding.NUM_PLANES}, 8, 8). "
                    "A 15-plane model is from an older revision of the spec."
                )
        if isinstance(shape[0], int) and shape[0] != 1:
            raise _fail(
                f"input batch dimension is fixed at {shape[0]}. It must be "
                "dynamic -- the referee evaluates positions in batches. Mark "
                "axis 0 of every input and output as dynamic when you export."
            )

        policy_shape = self._session.get_outputs()[0].shape
        last = policy_shape[-1] if policy_shape else None
        if isinstance(last, int) and last != EXPECTED_POLICY_SIZE:
            raise _fail(
                f"policy output has size {last}, expected {EXPECTED_POLICY_SIZE}. "
                f"The policy head must emit one logit per action index."
            )

    # ------------------------------------------------------------------
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions.

        Args:
            planes: ``(B, 16, 8, 8)``, or a single ``(16, 8, 8)`` position.

        Returns:
            ``(logits, values)`` with shapes ``(B, 4096)`` and ``(B,)``, both
            float32. Non-finite values are replaced with 0, so a model that
            emits NaN plays badly rather than crashing the referee.

        Raises:
            InvalidModelError: The model's actual output shapes are wrong.
        """
        planes = np.asarray(planes, dtype=np.float32)
        if planes.ndim == 3:
            planes = planes[None, ...]
        if planes.shape[1:] != EXPECTED_OBS_SHAPE:
            raise _fail(
                f"expected observations of shape (batch, {EXPECTED_OBS_SHAPE[0]}, "
                f"8, 8), got {planes.shape}"
            )
        batch = planes.shape[0]

        policy, value = self._session.run(
            [self._policy_name, self._value_name], {self._input_name: planes}
        )

        policy = np.asarray(policy, dtype=np.float32).reshape(batch, -1)
        if policy.shape[1] != EXPECTED_POLICY_SIZE:
            raise _fail(
                f"policy output has {policy.shape[1]} entries per position, "
                f"expected {EXPECTED_POLICY_SIZE}"
            )

        value = np.asarray(value, dtype=np.float32).reshape(batch)

        # A NaN or inf would poison the search's arithmetic. Neutralise it
        # here: the offending model simply expresses no opinion.
        policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
        value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        # The contract says [-1, 1]; clip rather than trust.
        value = np.clip(value, -1.0, 1.0)
        return policy, value

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return f"<OnnxEvaluator {self.path.name}>"
