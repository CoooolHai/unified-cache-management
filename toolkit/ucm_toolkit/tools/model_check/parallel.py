"""Local worker launcher for model-check tensor parallelism.

The controller only owns subprocess lifecycle.  It does not import torch or
vLLM and therefore cannot accidentally construct a service or load weights.
Workers receive a fixed rank contract and communicate through torch's
distributed process group once their platform entrypoint starts.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from typing import Mapping

from .topology import TensorParallelTopology


def _resolve_worker_module(module: str) -> str:
    """Resolve ``python -m`` entrypoints to an importable worker module.

    A module executed with ``python -m package.module`` observes
    ``__name__ == "__main__"``.  Reusing that value for child workers would
    launch ``python -m __main__``, which has no importable module spec.
    """

    if module != "__main__":
        return module
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    resolved = getattr(main_spec, "name", None)
    if not resolved or resolved == "__main__":
        raise RuntimeError(
            "cannot resolve the model-check worker module from __main__; "
            "launch the checker with python -m package.module"
        )
    return str(resolved)


def launch_workers(
    module: str,
    topology: TensorParallelTopology,
    base_env: Mapping[str, str] | None = None,
    *,
    timeout_seconds: float = 300.0,
) -> int:
    """Launch all TP ranks, propagate the first failure, and reap reliably."""

    worker_module = _resolve_worker_module(module)
    env_base = dict(os.environ if base_env is None else base_env)
    # Every rank must hash the exact same synthetic prompt.  The legacy
    # single-rank modules used time_ns^pid at import time, so establish one
    # controller-owned salt before starting any worker.
    env_base.setdefault("UCM_MODEL_CHECK_REQUEST_TOKEN_SALT", str(secrets.randbits(63)))
    processes: list[subprocess.Popen[bytes]] = []
    started = time.monotonic()
    try:
        for rank in range(topology.size):
            env = dict(env_base)
            env.update(topology.rank(rank).environment())
            # The parent has already installed the complete physical mask.
            # Keeping it unchanged makes logical rank==CUDA/NPU visible index.
            process = subprocess.Popen(
                [sys.executable, "-m", worker_module], env=env
            )
            processes.append(process)

        while processes:
            if time.monotonic() - started > timeout_seconds:
                for process in processes:
                    _terminate(process)
                return 124
            finished = []
            for process in processes:
                code = process.poll()
                if code is not None:
                    finished.append((process, code))
            if finished:
                first_code = next((code for _, code in finished if code), 0)
                if first_code:
                    for process in processes:
                        if process.poll() is None:
                            _terminate(process)
                    return int(first_code)
                processes = [p for p in processes if p.poll() is None]
            time.sleep(0.05)
        return 0
    except BaseException:
        for process in processes:
            if process.poll() is None:
                _terminate(process)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                _terminate(process)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Terminate a worker and escalate to kill if it ignores SIGTERM."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
