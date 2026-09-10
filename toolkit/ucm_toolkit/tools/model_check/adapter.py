"""Toolkit adapter for the UCM model compatibility checker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

from ...errors import ToolkitError
from ...registry import ToolAdapter
from ...runner import run_command
from .config import (
    ADDITIONAL_CONFIG_ENV,
    BLOCK_SIZE_ENV,
    CONNECTOR_MODULE_PATH_ENV,
    DEVICE_ENV,
    DEVICES_ENV,
    DTYPE_ENV,
    KV_CACHE_DTYPE_ENV,
    LEGACY_CONNECTOR_MODULE,
    MODEL_ENV,
    STORAGE_BACKENDS_ENV,
    STORE_PIPELINE_ENV,
    TOKENS_ENV,
    TENSOR_PARALLEL_SIZE_ENV,
    PLATFORM_ENV,
    MASTER_ADDR_ENV,
    MASTER_PORT_ENV,
    USE_LAYERWISE_ENV,
)
from .topology import make_topology


def _json_object(value: str) -> dict[str, object]:
    """Parse one command-line JSON object."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return decoded


def _detect_platform() -> str:
    """Detect the installed serving stack without importing torch or vLLM."""
    if importlib.util.find_spec("vllm_ascend") is not None:
        return "ascend"
    if importlib.util.find_spec("vllm") is not None:
        try:
            from importlib.metadata import version as _pkg_version

            if "+cpu" in _pkg_version("vllm"):
                return "cpu"
        except Exception:
            pass
        return "cuda"
    raise ToolkitError(
        "model-check cannot detect an installed vLLM or vLLM-Ascend stack"
    )


class ModelCheckTool(ToolAdapter):
    """Launch the CUDA or Ascend checker in an isolated child process."""

    name = "model-check"
    aliases = ("model_check",)
    description = (
        "Check a model's vLLM KV-cache layout and UCM dump/load compatibility "
        "without loading checkpoint weights."
    )
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register model-check configuration arguments."""
        parser.add_argument(
            "--model",
            help="model directory or Hugging Face model identifier",
        )
        parser.add_argument("--tokens", type=int, help="synthetic request length")
        parser.add_argument(
            "--block-size", type=int, help="vLLM KV-cache block size in tokens"
        )
        parser.add_argument(
            "--layerwise",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="enable or disable UCM layerwise mode",
        )
        parser.add_argument(
            "--additional-config",
            type=_json_object,
            help="vLLM additional_config as a JSON object",
        )
        parser.add_argument("--store-pipeline", help="UCM store pipeline")
        parser.add_argument(
            "--storage-backends",
            help="colon-separated UCM storage backend paths",
        )
        parser.add_argument(
            "--device-id",
            default=None,
            help="physical accelerator id exposed to the checker process",
        )
        parser.add_argument(
            "--devices",
            help="comma-separated physical accelerator ids for TP workers",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=None,
            help="number of tensor-parallel workers (single-node TP)",
        )
        parser.add_argument(
            "--platform",
            choices=("auto", "cuda", "ascend", "cpu"),
            default=None,
            help="serving platform; auto detects the installed stack",
        )
        parser.add_argument("--master-addr", default=None)
        parser.add_argument("--master-port", type=int, default=None)
        parser.add_argument("--dtype", help="vLLM model dtype")
        parser.add_argument("--kv-cache-dtype", help="vLLM KV-cache dtype")
        parser.add_argument(
            "--connector-module-path",
            default=None,
            help="module containing the UCMConnector facade",
        )

    def _build_run_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run model-check",
            description=self.description,
        )
        self.add_run_args(parser)
        return parser

    def run(self, tool_args: list[str]) -> int:
        """Run the selected checker without importing its runtime dependencies."""
        try:
            args = self._build_run_parser().parse_args(tool_args)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

        env = os.environ.copy()
        requested_platform = args.platform or env.get(PLATFORM_ENV, "auto")
        if requested_platform not in ("auto", "cuda", "ascend", "cpu"):
            raise ToolkitError(
                "invalid model-check platform: "
                f"{requested_platform!r}; expected auto/cuda/ascend/cpu"
            )
        platform = (
            requested_platform
            if requested_platform != "auto"
            else _detect_platform()
        )
        try:
            tp_size = args.tensor_parallel_size
            if tp_size is None:
                tp_size = int(env.get(TENSOR_PARALLEL_SIZE_ENV, "1"))
            device_id = args.device_id or env.get(DEVICE_ENV, "0")
            devices = args.devices or env.get(DEVICES_ENV) or device_id
            requested_port = (
                args.master_port
                if args.master_port is not None
                else int(env.get(MASTER_PORT_ENV, "0"))
            )
            # TP1 never initializes a process group in the controller.  Keep
            # its legacy path socket-free; TP>1 resolves one ephemeral port.
            if tp_size == 1 and requested_port == 0:
                requested_port = 29500
            topology = make_topology(
                tp_size,
                devices,
                master_addr=args.master_addr
                or env.get(MASTER_ADDR_ENV, "127.0.0.1"),
                master_port=requested_port,
            )
        except ValueError as exc:
            raise ToolkitError(f"invalid model-check TP topology: {exc}") from exc
        string_options = (
            ("model", MODEL_ENV),
            ("tokens", TOKENS_ENV),
            ("block_size", BLOCK_SIZE_ENV),
            ("store_pipeline", STORE_PIPELINE_ENV),
            ("storage_backends", STORAGE_BACKENDS_ENV),
            ("dtype", DTYPE_ENV),
            ("kv_cache_dtype", KV_CACHE_DTYPE_ENV),
        )
        for option, env_name in string_options:
            value = getattr(args, option)
            if value is not None:
                env[env_name] = str(value)
        if args.additional_config is not None:
            env[ADDITIONAL_CONFIG_ENV] = json.dumps(args.additional_config)
        if args.layerwise is not None:
            env[USE_LAYERWISE_ENV] = str(args.layerwise).lower()
        env[DEVICE_ENV] = device_id
        connector_module_path = args.connector_module_path or env.get(
            CONNECTOR_MODULE_PATH_ENV, LEGACY_CONNECTOR_MODULE
        )
        env[CONNECTOR_MODULE_PATH_ENV] = connector_module_path
        if tp_size != 1 or args.devices is not None:
            env[DEVICES_ENV] = topology.visible_devices
            env[TENSOR_PARALLEL_SIZE_ENV] = str(tp_size)
        if args.platform is not None:
            env[PLATFORM_ENV] = args.platform
        if tp_size != 1 or args.master_addr is not None:
            env[MASTER_ADDR_ENV] = topology.master_addr
        if tp_size != 1 or args.master_port is not None:
            env[MASTER_PORT_ENV] = str(topology.rendezvous_port)
        if platform == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = topology.visible_devices
        elif platform == "ascend":
            env["ASCEND_RT_VISIBLE_DEVICES"] = topology.visible_devices
        # cpu: no device-visibility variable needed
        module = f"{__package__}.{platform}"
        return run_command([sys.executable, "-m", module], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Report whether at least one supported serving stack is importable."""
        common = ("torch", "vllm", "ucm")
        missing_common = [
            name for name in common if importlib.util.find_spec(name) is None
        ]
        cuda_ok = not missing_common
        ascend_ok = cuda_ok and importlib.util.find_spec("vllm_ascend") is not None

        common_status = "OK" if cuda_ok else f"MISSING ({', '.join(missing_common)})"
        print(f"{self.name}: cuda {common_status}")
        ascend_status = "OK" if ascend_ok else "MISSING (vllm_ascend or common stack)"
        print(f"{self.name}: ascend {ascend_status}")
        return 0 if cuda_ok or ascend_ok else 1
