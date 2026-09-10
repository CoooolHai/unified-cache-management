"""Toolkit-level tests for the model compatibility checker."""

from __future__ import annotations

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry  # noqa: E402
from ucm_toolkit.cli import main  # noqa: E402
from ucm_toolkit.tools.model_check.adapter import ModelCheckTool  # noqa: E402
from ucm_toolkit.tools.model_check.config import load_config  # noqa: E402
from ucm_toolkit.tools.model_check.parallel import launch_workers  # noqa: E402
from ucm_toolkit.tools.model_check.topology import (  # noqa: E402
    make_topology,
    parse_devices,
)


class ModelCheckToolkitTest(unittest.TestCase):
    """Verify registration and subprocess dispatch without importing vLLM."""

    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_model_check_is_registered(self):
        registry.init_builtin_tools()

        tool = registry.get("model-check")

        self.assertEqual(tool.name, "model-check")
        self.assertIn("model_check", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_list_shows_model_check(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["list"])

        self.assertEqual(result, 0)
        self.assertIn("model-check", output.getvalue())

    def test_help_does_not_import_runtime_modules(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["run", "model-check", "--help"])

        self.assertEqual(result, 0)
        self.assertIn("--model", output.getvalue())
        self.assertIn("--device-id", output.getvalue())
        self.assertIn("--tokens", output.getvalue())
        self.assertIn("--additional-config", output.getvalue())
        self.assertIn("--connector-module-path", output.getvalue())
        self.assertNotIn("ucm_toolkit.tools.model_check.cuda", sys.modules)
        self.assertNotIn("ucm_toolkit.tools.model_check.ascend", sys.modules)

    def test_help_exposes_tp_topology_options(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["run", "model-check", "--help"])
        self.assertEqual(result, 0)
        self.assertIn("--tensor-parallel-size", output.getvalue())
        self.assertIn("--devices", output.getvalue())
        self.assertIn("--platform", output.getvalue())

    def test_topology_maps_physical_devices_to_logical_ranks(self):
        topology = make_topology(3, "7,2,5", master_port=23456)
        self.assertEqual(topology.visible_devices, "7,2,5")
        mappings = [topology.rank(rank) for rank in range(3)]
        self.assertEqual([item.physical_device for item in mappings], [7, 2, 5])
        self.assertEqual([item.logical_device for item in mappings], [0, 1, 2])
        self.assertEqual({item.master_port for item in mappings}, {23456})
        self.assertEqual(mappings[1].environment()["UCM_MODEL_CHECK_RANK"], "1")

    def test_topology_rejects_tp_device_count_mismatch(self):
        with self.assertRaises(ValueError):
            make_topology(2, "0")
        with self.assertRaises(ValueError):
            parse_devices("0,0")

    def test_worker_launcher_exports_one_shared_distributed_contract(self):
        class FinishedProcess:
            def poll(self):
                return 0

        topology = make_topology(2, "2,5", master_port=23456)
        with (
            patch(
                "ucm_toolkit.tools.model_check.parallel.subprocess.Popen",
                side_effect=[FinishedProcess(), FinishedProcess()],
            ) as popen,
            patch("ucm_toolkit.tools.model_check.parallel.time.sleep"),
        ):
            result = launch_workers("example.module", topology, {})

        self.assertEqual(result, 0)
        worker_envs = [call.kwargs["env"] for call in popen.call_args_list]
        self.assertEqual([env["RANK"] for env in worker_envs], ["0", "1"])
        self.assertEqual({env["WORLD_SIZE"] for env in worker_envs}, {"2"})
        self.assertEqual({env["MASTER_PORT"] for env in worker_envs}, {"23456"})
        self.assertEqual(
            {env["UCM_MODEL_CHECK_REQUEST_TOKEN_SALT"] for env in worker_envs},
            {worker_envs[0]["UCM_MODEL_CHECK_REQUEST_TOKEN_SALT"]},
        )

    def test_worker_launcher_resolves_python_m_main_module(self):
        class FinishedProcess:
            def poll(self):
                return 0

        topology = make_topology(1, "0", master_port=23456)
        main_module = types.SimpleNamespace(
            __spec__=types.SimpleNamespace(
                name="ucm_toolkit.tools.model_check.cuda"
            )
        )
        with (
            patch.dict(sys.modules, {"__main__": main_module}),
            patch(
                "ucm_toolkit.tools.model_check.parallel.subprocess.Popen",
                return_value=FinishedProcess(),
            ) as popen,
            patch("ucm_toolkit.tools.model_check.parallel.time.sleep"),
        ):
            result = launch_workers("__main__", topology, {})

        self.assertEqual(result, 0)
        self.assertEqual(
            popen.call_args.args[0],
            [
                sys.executable,
                "-m",
                "ucm_toolkit.tools.model_check.cuda",
            ],
        )

    def test_worker_launcher_terminates_peers_after_rank_failure(self):
        class FakeProcess:
            def __init__(self, returncode):
                self.returncode = returncode
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

        failed = FakeProcess(7)
        waiting = FakeProcess(None)
        topology = make_topology(2, "0,1", master_port=23456)
        with (
            patch(
                "ucm_toolkit.tools.model_check.parallel.subprocess.Popen",
                side_effect=[failed, waiting],
            ),
            patch("ucm_toolkit.tools.model_check.parallel.time.sleep"),
        ):
            result = launch_workers("example.module", topology, {})

        self.assertEqual(result, 7)
        self.assertTrue(waiting.terminated)

    def test_tp_cli_exports_physical_device_list(self):
        tool = ModelCheckTool()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: object() if name == "vllm" else None,
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=0
            ) as run,
            patch(
                "ucm_toolkit.tools.model_check.adapter.make_topology",
                return_value=make_topology(2, "2,5", master_port=23456),
            ),
        ):
            result = tool.run(
                [
                    "--platform",
                    "cuda",
                    "--tensor-parallel-size",
                    "2",
                    "--devices",
                    "2,5",
                ]
            )
        self.assertEqual(result, 0)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["UCM_MODEL_CHECK_DEVICES"], "2,5")
        self.assertEqual(env["UCM_MODEL_CHECK_TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")

    def test_cli_defaults_preserve_device_and_connector_environment(self):
        tool = ModelCheckTool()
        inherited = {
            "UCM_MODEL_CHECK_DEVICE_ID": "6",
            "UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH": (
                "ucm.integration.vllm.v2.ucm_connector"
            ),
        }
        with (
            patch.dict(os.environ, inherited, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: object() if name == "vllm" else None,
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=0
            ) as run,
        ):
            result = tool.run([])

        self.assertEqual(result, 0)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["UCM_MODEL_CHECK_DEVICE_ID"], "6")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(
            env["UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH"],
            "ucm.integration.vllm.v2.ucm_connector",
        )

    def test_child_configuration_reads_all_overrides(self):
        values = {
            "UCM_MODEL_CHECK_MODEL": "org/model",
            "UCM_MODEL_CHECK_TOKENS": "2048",
            "UCM_MODEL_CHECK_BLOCK_SIZE": "32",
            "UCM_MODEL_CHECK_USE_LAYERWISE": "false",
            "UCM_MODEL_CHECK_ADDITIONAL_CONFIG": '{"feature": true}',
            "UCM_MODEL_CHECK_STORE_PIPELINE": "Cache|Fake",
            "UCM_MODEL_CHECK_STORAGE_BACKENDS": "/cache/0:/cache/1",
            "UCM_MODEL_CHECK_DEVICE_ID": "5",
            "UCM_MODEL_CHECK_DEVICES": "5,7",
            "UCM_MODEL_CHECK_TENSOR_PARALLEL_SIZE": "2",
            "UCM_MODEL_CHECK_PLATFORM": "cuda",
            "UCM_MODEL_CHECK_MASTER_ADDR": "127.0.0.1",
            "UCM_MODEL_CHECK_MASTER_PORT": "23456",
            "UCM_MODEL_CHECK_DTYPE": "float16",
            "UCM_MODEL_CHECK_KV_CACHE_DTYPE": "auto",
            "UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH": (
                "ucm.integration.vllm.v2.ucm_connector"
            ),
        }
        with patch.dict(os.environ, values, clear=True):
            config = load_config()

        self.assertEqual(config.model, "org/model")
        self.assertEqual(config.tokens, 2048)
        self.assertEqual(config.block_size, 32)
        self.assertFalse(config.use_layerwise)
        self.assertEqual(config.additional_config, {"feature": True})
        self.assertEqual(config.store_pipeline, "Cache|Fake")
        self.assertEqual(config.storage_backends, "/cache/0:/cache/1")
        self.assertEqual(config.visible_devices, "5")
        self.assertEqual(config.devices, "5,7")
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.platform, "cuda")
        self.assertEqual(config.master_port, 23456)
        self.assertEqual(config.dtype, "float16")
        self.assertEqual(config.kv_cache_dtype, "auto")
        self.assertEqual(
            config.connector_module_path,
            "ucm.integration.vllm.v2.ucm_connector",
        )

    def test_cuda_runs_as_child_module(self):
        tool = ModelCheckTool()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: object() if name == "vllm" else None,
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=7
            ) as run,
        ):
            result = tool.run(
                [
                    "--model",
                    "/models/example",
                    "--tokens",
                    "8192",
                    "--block-size",
                    "128",
                    "--no-layerwise",
                    "--additional-config",
                    '{"enable_sparse_sfa_c8": true}',
                    "--store-pipeline",
                    "Cache|Posix",
                    "--storage-backends",
                    "/data/0:/data/1",
                    "--device-id",
                    "7",
                    "--dtype",
                    "bfloat16",
                    "--kv-cache-dtype",
                    "fp8",
                    "--connector-module-path",
                    "ucm.integration.vllm.v2.ucm_connector",
                ]
            )

        self.assertEqual(result, 7)
        run.assert_called_once_with(
            [sys.executable, "-m", "ucm_toolkit.tools.model_check.cuda"],
            env={
                "UCM_MODEL_CHECK_MODEL": "/models/example",
                "UCM_MODEL_CHECK_TOKENS": "8192",
                "UCM_MODEL_CHECK_BLOCK_SIZE": "128",
                "UCM_MODEL_CHECK_USE_LAYERWISE": "false",
                "UCM_MODEL_CHECK_ADDITIONAL_CONFIG": ('{"enable_sparse_sfa_c8": true}'),
                "UCM_MODEL_CHECK_STORE_PIPELINE": "Cache|Posix",
                "UCM_MODEL_CHECK_STORAGE_BACKENDS": "/data/0:/data/1",
                "UCM_MODEL_CHECK_DEVICE_ID": "7",
                "UCM_MODEL_CHECK_DTYPE": "bfloat16",
                "UCM_MODEL_CHECK_KV_CACHE_DTYPE": "fp8",
                "UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH": (
                    "ucm.integration.vllm.v2.ucm_connector"
                ),
                "CUDA_VISIBLE_DEVICES": "7",
            },
        )

    def test_ascend_runs_as_child_module(self):
        tool = ModelCheckTool()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: (object() if name == "vllm_ascend" else None),
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=8
            ) as run,
        ):
            result = tool.run(["--model", "org/model", "--device-id", "3"])

        self.assertEqual(result, 8)
        run.assert_called_once_with(
            [sys.executable, "-m", "ucm_toolkit.tools.model_check.ascend"],
            env={
                "UCM_MODEL_CHECK_MODEL": "org/model",
                "UCM_MODEL_CHECK_DEVICE_ID": "3",
                "UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH": (
                    "ucm.integration.vllm.ucm_connector"
                ),
                "ASCEND_RT_VISIBLE_DEVICES": "3",
            },
        )


if __name__ == "__main__":
    unittest.main()
