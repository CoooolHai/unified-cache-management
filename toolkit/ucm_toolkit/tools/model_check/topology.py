"""Pure-Python topology helpers for the model-check launcher.

This module deliberately has no torch/vLLM imports.  The command-line process
uses it to validate and normalize a single-node tensor-parallel topology, while
workers use the resulting environment variables to initialize vLLM's process
groups.  Device ids in ``devices`` are *physical* ids; workers always use the
corresponding logical id after the visibility mask is installed.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass


def parse_devices(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated physical device list."""

    if value is None or not value.strip():
        return (0,)
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("devices must not contain an empty item")
        try:
            device = int(item)
        except ValueError as exc:
            raise ValueError(f"device id must be an integer, got {item!r}") from exc
        if device < 0:
            raise ValueError(f"device id must be non-negative, got {device}")
        result.append(device)
    if len(set(result)) != len(result):
        raise ValueError(f"devices must be unique, got {result!r}")
    return tuple(result)


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return an ephemeral TCP port selected by the OS."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class TensorParallelTopology:
    """Validated single-node TP topology.

    ``physical_devices`` are the user-facing ids.  ``logical_device`` is the
    index visible inside a worker after ``CUDA_VISIBLE_DEVICES`` or
    ``ASCEND_RT_VISIBLE_DEVICES`` is set to the physical list.
    """

    size: int
    physical_devices: tuple[int, ...]
    master_addr: str = "127.0.0.1"
    master_port: int = 0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if len(self.physical_devices) != self.size:
            raise ValueError(
                "the number of physical devices must equal tensor_parallel_size: "
                f"tp={self.size}, devices={self.physical_devices!r}"
            )
        if self.master_port < 0 or self.master_port > 65535:
            raise ValueError("master_port must be between 0 and 65535")

    @property
    def visible_devices(self) -> str:
        return ",".join(str(device) for device in self.physical_devices)

    @property
    def rendezvous_port(self) -> int:
        """Return the already-resolved rendezvous port.

        ``make_topology`` resolves an ephemeral port once.  It is important
        that calling ``rank()`` for multiple workers never selects a different
        port for each worker.
        """

        return self.master_port

    def rank(self, rank: int) -> "RankMapping":
        if rank < 0 or rank >= self.size:
            raise ValueError(f"rank {rank} is outside [0, {self.size})")
        return RankMapping(
            rank=rank,
            world_size=self.size,
            physical_device=self.physical_devices[rank],
            logical_device=rank,
            master_addr=self.master_addr,
            master_port=self.rendezvous_port,
        )


@dataclass(frozen=True)
class RankMapping:
    rank: int
    world_size: int
    physical_device: int
    logical_device: int
    master_addr: str
    master_port: int

    def environment(self) -> dict[str, str]:
        return {
            "UCM_MODEL_CHECK_RANK": str(self.rank),
            "UCM_MODEL_CHECK_WORLD_SIZE": str(self.world_size),
            "UCM_MODEL_CHECK_LOCAL_RANK": str(self.logical_device),
            "UCM_MODEL_CHECK_MASTER_ADDR": self.master_addr,
            "UCM_MODEL_CHECK_MASTER_PORT": str(self.master_port),
            "UCM_MODEL_CHECK_PHYSICAL_DEVICE": str(self.physical_device),
            # Also expose the conventional torch.distributed contract for
            # platform plugins that inspect standard launcher variables.
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.logical_device),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }


def make_topology(
    tensor_parallel_size: int,
    devices: str | None,
    *,
    master_addr: str = "127.0.0.1",
    master_port: int = 0,
) -> TensorParallelTopology:
    """Build and validate the launcher topology."""

    physical_devices = parse_devices(devices)
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if len(physical_devices) != tensor_parallel_size:
        raise ValueError(
            "the number of physical devices must equal tensor_parallel_size: "
            f"tp={tensor_parallel_size}, devices={physical_devices!r}"
        )
    resolved_port = master_port or pick_free_port(master_addr)
    return TensorParallelTopology(
        size=tensor_parallel_size,
        physical_devices=physical_devices,
        master_addr=master_addr,
        master_port=resolved_port,
    )
