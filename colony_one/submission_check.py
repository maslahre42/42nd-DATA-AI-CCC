"""Cross-platform submission acceptance check.

Runs the exact acceptance policy in :mod:`colony_one.submission_policy` in
an isolated child process, on Windows, macOS, and Linux alike.

 A model this check accepts is checked against the same acceptance
*policy* the server uses, but the server additionally enforces hard resource
ceilings so passing here is not an unconditional guarantee of server acceptance.
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colony_one.submission_policy import (  # noqa: E402
    LEVEL3_SIMULATIONS,
    MAX_MODEL_BYTES,
    MEMORY_ERROR_MARKERS,
    RUNTIME_CRASHED,
    TOO_MUCH_MEMORY_TO_CHECK,
    TOO_SLOW_TO_CHECK,
    VALIDATION_TIMEOUT_SECONDS,
    ModelRejected,
    child_environment,
    inspect_and_dry_run_model,
    one_line,
    plain_runtime_error,
    read_limited,
    read_verdict,
)

__all__ = [
    "MAX_MODEL_BYTES",
    "ModelRejected",
    "validate_model_bytes_portable",
    "validate_model_file_portable",
]


def _reject(message: str) -> None:
    raise ModelRejected(message)


def _child_dry_run(file_path: Path) -> float:
    return inspect_and_dry_run_model(file_path)


def _child_main(path_arg: str) -> int:
    try:
        duration = _child_dry_run(Path(path_arg))
    except ModelRejected as exc:
        print(f"REJECTED: {one_line(str(exc))[:500]}", file=sys.stderr)
        return 1
    except BaseException as exc:  # noqa: BLE001 - never leak a traceback
        reason = plain_runtime_error(str(exc))[:300]
        print(f"INTERNAL_ERROR: {type(exc).__name__}: {reason}", file=sys.stderr)
        return 1
    print(json.dumps({"duration_seconds": duration}, separators=(",", ":")))
    return 0


def play_level3_match(file_path: Path) -> int:
    """Play the server's Level 3 match (model with MCTS as CYAN against GreedyAgent)."""
    import numpy as np

    from colony_one import constants, mcts
    from colony_one.baselines import GreedyAgent
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator

    env = ColonyOneEnv()
    evaluator = OnnxEvaluator(file_path)
    config = mcts.MCTSConfig(simulations=LEVEL3_SIMULATIONS)
    opponent = GreedyAgent(seed=0)
    rng = np.random.default_rng(0)
    actions = 0
    while not env.terminated:
        if env.current_team == constants.CYAN:
            action = mcts.search_action(
                env.board, env.current_team, env.actions_left, env.round_number,
                evaluator, config, rng=rng,
            )
        else:
            action = opponent.select(
                env.board, env.current_team, env.actions_left, env.round_number
            )
        if action is None:
            break
        env.step(action)
        actions += 1
    return actions


def _match_child_main(path_arg: str) -> int:
    try:
        actions = play_level3_match(Path(path_arg))
    except BaseException as exc:  # noqa: BLE001 - never leak a traceback
        reason = plain_runtime_error(str(exc))[:500]
        print(f"REJECTED: Your model failed while playing: {reason}", file=sys.stderr)
        return 1
    print(json.dumps({"actions": actions}, separators=(",", ":")))
    return 0


def _terminate(process: subprocess.Popen) -> None:
    """Best-effort termination of the dry-run child.

    On POSIX this kills the child's own process group (``start_new_session``
    below), matching the Linux server validator. On Windows there is no
    portable process-group kill in the standard library: ``Popen.kill()``
    terminates only the immediate child. This validator's own dry run never
    spawns a subprocess of its own, so that is sufficient for the built-in
    check; it is called out here rather than silently assumed.
    """
    if sys.platform != "win32":
        import os
        import signal

        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with contextlib.suppress(OSError):
            process.kill()


def _run_child(file_path: Path) -> float:
    command = [
        sys.executable, "-I", str(Path(__file__).resolve()), "--child", str(file_path),
    ]
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=child_environment(),
            close_fds=True,
            **popen_kwargs,
        )
        try:
            returncode = process.wait(timeout=VALIDATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            process.wait()
            raise ModelRejected(TOO_SLOW_TO_CHECK) from exc

        if returncode:
            detail = read_verdict(stderr)
            if any(marker in detail for marker in MEMORY_ERROR_MARKERS):
                _reject(TOO_MUCH_MEMORY_TO_CHECK)
            if sys.platform != "win32" and returncode < 0:
                import signal

                killed = -returncode == signal.SIGKILL
                _reject(TOO_MUCH_MEMORY_TO_CHECK if killed else RUNTIME_CRASHED)
            if detail.startswith("REJECTED: "):
                _reject(detail.removeprefix("REJECTED: "))
            reason = detail.removeprefix("INTERNAL_ERROR: ")
            _reject(
                f"The model check failed unexpectedly (exit code {returncode}): {reason}"
            )
        try:
            result = json.loads(read_limited(stdout))
            duration = float(result["duration_seconds"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelRejected("Model dry run did not report its duration") from exc
        if not math.isfinite(duration) or duration < 0:
            _reject("Model dry run reported an invalid duration")
        return duration


def validate_model_bytes_portable(data: bytes, max_size: int = MAX_MODEL_BYTES) -> float:
    """Validate ONNX bytes in an isolated child process, on any platform.

    Enforces the same acceptance policy as the Linux server validator, in
    an isolated subprocess with a wall-clock timeout, but without the
    server's mandatory POSIX resource limits (address space, CPU time, file
    descriptors). See the module docstring.
    """
    if len(data) > max_size:
        _reject(f"Model exceeds the {max_size}-byte size limit")
    if not data:
        _reject("Uploaded model is empty")
    try:
        with tempfile.TemporaryDirectory(prefix="onnx-validate-") as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(data)
            return _run_child(path)
    except ModelRejected:
        raise
    except OSError as exc:
        raise ModelRejected(f"Unable to run model dry run: {str(exc)[:300]}") from exc


def validate_model_file_portable(path: str | Path) -> float:
    """Validate a local regular model file without an unbounded parent read."""
    path = Path(path)
    try:
        if not path.exists():
            _reject(f"Model file not found: {path}")
        stat = path.stat()
        if not path.is_file():
            _reject(f"Model path is not a regular file: {path}")
        if stat.st_size > MAX_MODEL_BYTES:
            _reject(f"Model exceeds the {MAX_MODEL_BYTES}-byte size limit")
        if stat.st_size == 0:
            _reject("Uploaded model is empty")
        data = path.read_bytes()
    except ModelRejected:
        raise
    except OSError as exc:
        raise ModelRejected(f"Unable to read model file: {str(exc)[:300]}") from exc
    return validate_model_bytes_portable(data)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--match-child":
        raise SystemExit(_match_child_main(sys.argv[2]))
    raise SystemExit("submission_check.py is not a command-line tool")
