#!/usr/bin/env python3
"""Validate a Colony: One tournament submission.

Run this before you submit. It performs the same acceptance *policy* as the
tournament server -- file size, ONNX graph, operator allow-list, input
contract, and an isolated CPU dry run against a fixed adversarial corpus of
board positions. This runs on Windows, macOS, and Linux alike.

    python tools/validate_submission.py model.onnx

Exit code 0 means accepted; 1 means rejected, with the reason printed.

Passing certifies *format*, not *strength*. To measure strength, let two
models play each other with ``play/model_vs_model.py``.

Resource limits
---------------
A second dry run uses psutil monitoring as a best-effort approximation of
the server's hard POSIX rlimits.  The thresholds are intentionally slightly
stricter than the server's to compensate for measurement differences (RSS
vs virtual address space, polled vs instant CPU time).

A model rejected here will also be rejected by the server.  A model
accepted here may still be rejected if it exceeds virtual address space or
CPU time in ways psutil polling cannot observe; the server enforces those
limits exactly via ``RLIMIT_AS`` and ``RLIMIT_CPU``.

A third run plays one Level 3 match as the server does (referee MCTS with
60 simulations against ``GreedyAgent``) under the match limits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from colony_one.submission_check import (  # noqa: E402
    ModelRejected,
    validate_model_file_portable,
)
from colony_one.submission_policy import (  # noqa: E402
    MATCH_CPU_SECONDS,
    MATCH_TIMEOUT_SECONDS,
    MEMORY_ERROR_MARKERS,
    RUNTIME_CRASHED,
    TOO_MUCH_MEMORY_TO_CHECK,
    TOO_MUCH_MEMORY_TO_PLAY,
    TOO_SLOW_TO_CHECK,
    TOO_SLOW_TO_PLAY,
    VALIDATION_TIMEOUT_SECONDS,
    child_environment,
    read_limited,
    read_verdict,
)

OK = "  [ok]"
FAIL = "  [FAIL]"
INFO = "       "

# Approximate resource caps -- intentionally stricter than the server's hard
# POSIX rlimits. RSS is always <= virtual address space, so a lower RSS cap
# compensates for the difference in what psutil measures vs RLIMIT_AS.
_APPROX_MAX_RSS_BYTES = 320 * 1024 * 1024   # server RLIMIT_AS  = 384 MiB
_APPROX_MAX_CPU_SECONDS = 1.8               # server RLIMIT_CPU = 2 s
_APPROX_MAX_OPEN_FILES = 28                 # server RLIMIT_NOFILE = 32
_APPROX_MAX_MATCH_RSS_BYTES = 425 * 1024 * 1024   # server RLIMIT_AS  = 512 MiB
_APPROX_MAX_MATCH_CPU_SECONDS = 13.5              # server RLIMIT_CPU = 15 s
_MONITOR_POLL_SECONDS = 0.05


class CheckFailed(Exception):
    """A validation step failed; the message is shown to the participant."""


def step(title: str) -> None:
    print(f"\n{title}")


def ok(message: str) -> None:
    print(f"{OK} {message}")


def info(message: str) -> None:
    print(f"{INFO} {message}")


# ---------------------------------------------------------------------------
def check_contract(path: Path) -> float:
    """Run the same acceptance *policy* the tournament server enforces.

    This runs cross-platform, in an isolated child process with a
    wall-clock timeout, but it does not enforce the Linux server's
    mandatory address-space and CPU-time resource limits -- those require
    POSIX and are applied only by the server itself. A model rejected here
    will also be rejected by the server; the server can additionally reject
    a model that is accepted here but exceeds its resource ceilings.
    """
    step("1/3  Tournament acceptance contract")
    dry_run_seconds = validate_model_file_portable(path)
    ok(
        "accepted by the same acceptance policy used by the tournament server "
        f"({dry_run_seconds * 1000:.1f} ms dry run)"
    )
    info(
        "the Linux server additionally enforces hard memory/CPU limits "
        "this check cannot reproduce on every platform"
    )
    return dry_run_seconds


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Limits:
    cpu_seconds: float
    rss_bytes: int
    too_slow: str
    too_much_memory: str


_CHECK_LIMITS = _Limits(
    _APPROX_MAX_CPU_SECONDS, _APPROX_MAX_RSS_BYTES,
    TOO_SLOW_TO_CHECK, TOO_MUCH_MEMORY_TO_CHECK,
)
_MATCH_LIMITS = _Limits(
    _APPROX_MAX_MATCH_CPU_SECONDS, _APPROX_MAX_MATCH_RSS_BYTES,
    TOO_SLOW_TO_PLAY, TOO_MUCH_MEMORY_TO_PLAY,
)


def _monitor_child(
    process: subprocess.Popen,
    result: dict,
    stop_event: threading.Event,
    limits: _Limits,
) -> None:
    """Poll *process* resource usage; kill it and set result['exceeded'] if over limit."""
    try:
        ps = psutil.Process(process.pid)
        while not stop_event.is_set():
            try:
                with ps.oneshot():
                    rss = ps.memory_info().rss
                    cpu = ps.cpu_times()
                    cpu_s = cpu.user + cpu.system
                    try:
                        fds = ps.num_fds()
                    except AttributeError:
                        fds = 0  # Windows has no direct equivalent
                result["peak_rss"] = max(result.get("peak_rss", 0), rss)
                result["peak_cpu"] = max(result.get("peak_cpu", 0.0), cpu_s)
                result["peak_fds"] = max(result.get("peak_fds", 0), fds)
                if rss > limits.rss_bytes:
                    result["exceeded"] = (
                        f"{limits.too_much_memory} (measured here: "
                        f"{rss // (1024 * 1024)} MiB resident; "
                        f"this tool stops at {limits.rss_bytes // (1024 * 1024)} MiB)"
                    )
                    with contextlib.suppress(OSError):
                        process.kill()
                    return
                if cpu_s > limits.cpu_seconds:
                    result["exceeded"] = (
                        f"{limits.too_slow} (measured here: {cpu_s:.2f} s of CPU time; "
                        f"this tool stops at {limits.cpu_seconds} s)"
                    )
                    with contextlib.suppress(OSError):
                        process.kill()
                    return
                if fds and fds > _APPROX_MAX_OPEN_FILES:
                    result["exceeded"] = (
                        f"open file descriptors {fds} "
                        f"exceed the {_APPROX_MAX_OPEN_FILES} limit"
                    )
                    with contextlib.suppress(OSError):
                        process.kill()
                    return
            except psutil.NoSuchProcess:
                return
            stop_event.wait(_MONITOR_POLL_SECONDS)
    except Exception:  # noqa: BLE001
        pass


def _terminate_child(process: subprocess.Popen) -> None:
    if sys.platform != "win32":
        import os
        import signal
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with contextlib.suppress(OSError):
            process.kill()


def _run_monitored(
    mode: str, path: Path, timeout: int, limits: _Limits
) -> tuple[dict, str]:
    """Run ``submission_check.py mode path`` under psutil; returns (usage, stdout)."""
    import colony_one.submission_check as _sc
    check_module = Path(_sc.__file__).resolve()
    command = [sys.executable, "-I", str(check_module), mode, str(path.resolve())]
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    resource_result: dict = {}
    stop_event = threading.Event()

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

        monitor = threading.Thread(
            target=_monitor_child,
            args=(process, resource_result, stop_event, limits),
            daemon=True,
        )
        monitor.start()

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_child(process)
            process.wait()
            raise ModelRejected(limits.too_slow) from None
        finally:
            stop_event.set()
            monitor.join(timeout=1.0)

        # Check resource limits before interpreting the exit code: the monitor
        # may have killed the child (giving a non-zero exit) after detecting a
        # limit breach.
        if "exceeded" in resource_result:
            raise ModelRejected(resource_result["exceeded"])

        if returncode:
            detail = read_verdict(stderr)
            if any(marker in detail for marker in MEMORY_ERROR_MARKERS):
                raise ModelRejected(limits.too_much_memory)
            if sys.platform != "win32" and returncode < 0:
                import signal
                killed = -returncode == signal.SIGKILL
                raise ModelRejected(limits.too_much_memory if killed else RUNTIME_CRASHED)
            if detail.startswith("REJECTED: "):
                raise ModelRejected(detail.removeprefix("REJECTED: "))
            raise CheckFailed(
                f"Resource-check run failed (exit {returncode}): "
                + (detail[:200] if detail else "(no output)")
            )
        return resource_result, read_limited(stdout)


def _usage(resource_result: dict) -> str:
    peak_mib = resource_result.get("peak_rss", 0) // (1024 * 1024)
    peak_cpu = resource_result.get("peak_cpu", 0.0)
    parts = [f"peak RSS {peak_mib} MiB", f"CPU {peak_cpu:.2f} s"]
    if resource_result.get("peak_fds", 0):
        parts.append(f"open files {resource_result['peak_fds']}")
    return ", ".join(parts)


def check_resources(path: Path) -> None:
    """Re-run the dry run under psutil monitoring to approximate server limits."""
    step("2/3  Check resource limits")
    resource_result, _ = _run_monitored(
        "--child", path, VALIDATION_TIMEOUT_SECONDS, _CHECK_LIMITS
    )
    ok("within approximate resource limits")
    info("resource usage: " + _usage(resource_result))
    info(
        "the Linux server measures virtual address space and CPU time via "
        "hard POSIX rlimits; psutil RSS and polled CPU are approximations"
    )


def check_level3_match(path: Path) -> None:
    """Play one Level 3 match as the server does, under its time and memory limits."""
    step("3/3  Level 3 match limits")
    resource_result, stdout = _run_monitored(
        "--match-child", path, MATCH_TIMEOUT_SECONDS, _MATCH_LIMITS
    )
    try:
        actions = int(json.loads(stdout)["actions"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckFailed("Level 3 match did not report its result") from exc
    ok(f"one Level 3 match fits the server's limits ({actions} actions vs GreedyAgent)")
    info("resource usage: " + _usage(resource_result))
    info(
        f"the server allows {MATCH_CPU_SECONDS} s of CPU time per match; "
        "matches vary in length, so leave headroom"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Colony: One ONNX submission.",
        epilog="Exit code 0 = accepted, 1 = rejected.",
    )
    parser.add_argument("model", type=Path, help="path to your .onnx file")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("Colony: One -- submission validator")
    print(f"model: {args.model}")
    print("=" * 68)

    try:
        check_contract(args.model)
        check_resources(args.model)
        check_level3_match(args.model)
    except ModelRejected as exc:
        print(f"\n{FAIL} {exc}")
        print("\n" + "=" * 68)
        print("REJECTED -- see docs/submission.md")
        print("=" * 68)
        return 1
    except Exception as exc:
        print(f"\n{FAIL} unexpected error: {type(exc).__name__}: {exc}")
        print("\n" + "=" * 68)
        print("REJECTED -- if you believe this is a validator bug, contact the")
        print("            organisers.")
        print("=" * 68)
        return 1

    print("\n" + "=" * 68)
    print("ACCEPTED -- this model satisfies the submission contract.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
