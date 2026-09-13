"""Small dependency-free regression tests for the vLLM KV-cache adapters."""

import ast
import math
import types
import unittest
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[3]
HLA_PATH = ROOT / "ucm/integration/vllm/hla_connector.py"
HMA_PATH = ROOT / "ucm/integration/vllm/hma_connector.py"
UCM_PATH = ROOT / "ucm/integration/vllm/ucm_connector.py"


def _load_raw_tensor_layers():
    tree = ast.parse(HLA_PATH.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if getattr(n, "name", None) == "_raw_tensor_layers"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object}
    exec(compile(module, str(HLA_PATH), "exec"), namespace)
    return namespace["_raw_tensor_layers"]


def _load_hma_methods(names):
    tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == "UCMFAWAConnector")
    nodes = [n for n in cls.body if getattr(n, "name", None) in names]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    class GroupMeta:
        def __init__(self, group_id, token_block_size, tail_blocks, tail_tokens):
            self.group_id = group_id
            self.token_block_size = token_block_size
            self.tail_blocks = tail_blocks
            self.tail_tokens = tail_tokens

    namespace = {
        "np": __import__("numpy"),
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "KVCacheGroupMeta": GroupMeta,
        "extract_layer_index": lambda name: int(name.split(".")[1]),
        "logger": types.SimpleNamespace(
            info_once=lambda *_args: None, info=lambda *_args: None
        ),
    }
    exec(compile(module, str(HMA_PATH), "exec"), namespace)
    for name in names:
        value = namespace[name]
        if isinstance(value, staticmethod):
            namespace[name] = value.__func__
    return namespace


def _init_fake_connector(methods, *, ratios, features, resolver):
    spec_full = types.SimpleNamespace(
        block_size=32,
        sliding_window=None,
        tokens_per_state=1,
        page_size_bytes=640,
        prefix_cacheable=True,
    )
    spec_swa = types.SimpleNamespace(
        block_size=32,
        sliding_window=128,
        tokens_per_state=1,
        page_size_bytes=640,
        prefix_cacheable=True,
    )
    spec_transient = types.SimpleNamespace(
        block_size=8,
        sliding_window=None,
        tokens_per_state=1,
        page_size_bytes=128,
        prefix_cacheable=False,
    )
    groups = [
        types.SimpleNamespace(kv_cache_spec=spec_full, layer_names=("l0",)),
        types.SimpleNamespace(kv_cache_spec=spec_swa, layer_names=("swa_cache",)),
        types.SimpleNamespace(kv_cache_spec=spec_transient, layer_names=("ring",)),
    ]
    hf_config = types.SimpleNamespace(compress_ratios=ratios, **features)
    config = types.SimpleNamespace(
        kv_cache_groups=groups,
        model_config=types.SimpleNamespace(hf_config=hf_config),
        speculative_config=None,
    )
    self_obj = types.SimpleNamespace(
        _kv_cache_config=config,
        _vllm_config=config,
        is_ascend_layout=False,
        ascend_base_block_size=None,
        hash_block_size=256,
        transient_group_ids=set(),
        fa_group_ids=[],
        window_group_ids=[],
        group_metas={},
        file_size={},
        max_token_block_size=0,
    )
    if "_derive_v41_file_sizes" in methods:
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
    self_obj.can_handle_ascend_kv_cache_config = lambda _config: False
    methods_ns = {"resolve_kv_cache_block_sizes": resolver}
    return self_obj, groups, methods_ns


class UCMV41CompatibilityTest(unittest.TestCase):
    def test_raw_tensor_layers_supports_new_and_legacy_api(self):
        accessor = _load_raw_tensor_layers()
        self.assertEqual(
            accessor(type("T", (), {"layers": ("a", "b")})()), ("a", "b")
        )
        self.assertEqual(
            accessor(type("T", (), {"shared_by": ("legacy",)})()), ("legacy",)
        )
        self.assertEqual(accessor(type("T", (), {})()), ())

    def test_route_checks_fawa_before_hla(self):
        connector = UCM_PATH.read_text(encoding="utf-8")
        self.assertLess(
            connector.index("is_fawa = UCMFAWAConnector.can_handle_kv_cache_config"),
            connector.index(
                "UCMHybridLinearAttentionConnector.supports_kv_cache_layout"
            ),
        )

    def test_standardized_layout_uses_state_dimension(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            shape = (2, 1, 32, 4)
            dtype = "bf16"

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return 4096

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (128, 128, 4, 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        layout = namespace["KVCacheGroupLayout"](
            {"model.layers.0.swa": Tensor()},
            expected_block_size=32,
            standardized_layout=True,
            expected_tokens_per_state=1,
        )
        self.assertEqual(layout.block_strides.tolist(), [128])
        self.assertEqual(layout.segment_tensor_size_list(32, 32), [128])

    def test_ratio_two_standardized_segment_scales_physical_states(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            shape = (2, 1, 16, 4)
            dtype = "bf16"

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return 4096

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (64, 64, 4, 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        layout = namespace["KVCacheGroupLayout"](
            {"model.layers.0.mla": Tensor()},
            expected_block_size=32,
            standardized_layout=True,
            expected_tokens_per_state=2,
        )
        self.assertEqual(layout.segment_tensor_size_list(32, 32), [64])

    def test_transient_slice_keeps_original_group_index(self):
        methods = _load_hma_methods({"_slice_group_block_ids"})
        self_obj = types.SimpleNamespace(
            transient_group_ids={2},
            window_group_ids=[1],
            group_metas={2: types.SimpleNamespace(token_block_size=8)},
        )
        self_obj._slice_group_block_ids = types.MethodType(
            methods["_slice_group_block_ids"], self_obj
        )
        self.assertEqual(
            self_obj._slice_group_block_ids(
                2, [10, 11], __import__("numpy").array([31]), False
            ),
            [],
        )
        dispatch = [[], [], [], [], []]
        dispatch[2] = self_obj._slice_group_block_ids(
            2, [10, 11], __import__("numpy").array([31]), False
        )
        self.assertEqual(len(dispatch), 5)
        self.assertEqual(dispatch[2], [])

    def test_dynamic_size_scales_physical_group_block(self):
        methods = _load_hma_methods({"_specs_for_group", "_derive_v41_file_sizes"})
        spec = types.SimpleNamespace(block_size=64, page_size_bytes=640)
        group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=("l0",))
        self_obj = types.SimpleNamespace(
            transient_group_ids=set(),
            fa_group_ids=[0],
            window_group_ids=[],
            group_metas={
                0: types.SimpleNamespace(tail_tokens=128, tail_blocks=2)
            },
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
        self.assertEqual(
            self_obj._derive_v41_file_sizes([group]), {"FA": 4096, "WA": 0}
        )

    def test_v41_model_features_select_resolved_hash_without_ratio_two(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        resolver = lambda _cache_config, _vllm_config: (64, 32)
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            resolver
        )
        self_obj, groups, _ = _init_fake_connector(
            methods,
            ratios=[0, 1, 1],
            features={"kv_source_layer_ids": (1,), "index_source_layer_ids": (1,)},
            resolver=resolver,
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj._init_group_metas()
        self.assertTrue(self_obj.is_v41_layout)
        self.assertEqual(self_obj.hash_block_size, 32)
        self.assertEqual(sorted(self_obj.transient_group_ids), [2])
        self.assertEqual(self_obj.group_metas[2].tail_blocks, 0)

    def test_invalid_resolver_result_fails_closed(self):
        methods = _load_hma_methods({"_init_group_metas"})
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            lambda _cache_config, _vllm_config: [32, 32]
        )
        self_obj, _groups, _ = _init_fake_connector(
            methods,
            ratios=[0, 1],
            features={"kv_source_layer_ids": (1,)},
            resolver=lambda *_args: [32, 32],
        )
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        with self.assertRaisesRegex(RuntimeError, "Unable to resolve"):
            self_obj._init_group_metas()

    def test_v40_topology_keeps_legacy_hash_and_file_sizes(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        self_obj, groups, _ = _init_fake_connector(
            methods, ratios=[0, 4, 128], features={}, resolver=None
        )
        groups[2].kv_cache_spec.prefix_cacheable = True
        for group in groups:
            group.kv_cache_spec.block_size = 256
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj._init_group_metas()
        self.assertFalse(self_obj.is_v41_layout)
        self.assertEqual(self_obj.hash_block_size, 256)
        expected_fa = ((37376 + 8448) * 21 + 1168 * 20 + 4095) // 4096 * 4096
        self.assertEqual(self_obj.file_size["FA"], expected_fa)

    def test_real_v41_flash_topology_keeps_nine_groups(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        groups = []
        swa_layers = []
        for group_id, layer_count in enumerate((6, 6, 6, 6, 6, 5, 5)):
            names = tuple(
                f"model.layers.{group_id}_{i}.swa_cache"
                for i in range(layer_count)
            )
            swa_layers.extend(names)
            spec = types.SimpleNamespace(
                block_size=64,
                sliding_window=128,
                tokens_per_state=1,
                page_size_bytes=19008,
                prefix_cacheable=True,
            )
            groups.append(
                types.SimpleNamespace(kv_cache_spec=spec, layer_names=names)
            )
        csa_names = tuple(f"model.layers.source_{i}" for i in range(4))
        index_names = tuple(f"model.layers.index_{i}" for i in range(4))
        all_fa_names = csa_names + index_names
        inner_specs = {
            name: types.SimpleNamespace(
                block_size=64,
                tokens_per_state=2,
                page_size_bytes=1000,
                prefix_cacheable=True,
            )
            for name in all_fa_names
        }
        fa_spec = types.SimpleNamespace(
            block_size=64,
            kv_cache_specs=inner_specs,
            prefix_cacheable=True,
        )
        groups.append(
            types.SimpleNamespace(kv_cache_spec=fa_spec, layer_names=all_fa_names)
        )
        ring_spec = types.SimpleNamespace(
            block_size=16, tokens_per_state=1, prefix_cacheable=False
        )
        groups.append(
            types.SimpleNamespace(kv_cache_spec=ring_spec, layer_names=("ring",))
        )
        hf_config = types.SimpleNamespace(
            compress_ratios=[0] * 40,
            kv_source_layer_ids=(2, 8, 14, 20),
            index_source_layer_ids=(2, 8, 14, 20),
        )
        config = types.SimpleNamespace(
            kv_cache_groups=groups,
            model_config=types.SimpleNamespace(hf_config=hf_config),
            speculative_config=None,
        )
        self_obj = types.SimpleNamespace(
            _kv_cache_config=config,
            _vllm_config=config,
            is_ascend_layout=False,
            ascend_base_block_size=None,
            hash_block_size=256,
            transient_group_ids=set(),
            fa_group_ids=[],
            window_group_ids=[],
            group_metas={},
            file_size={},
            max_token_block_size=0,
        )
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            lambda *_args: (64, 64)
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj.can_handle_ascend_kv_cache_config = lambda _config: False
        self_obj._init_group_metas()
        self.assertEqual(len(self_obj.group_metas), 9)
        self.assertEqual(self_obj.transient_group_ids, {8})
        self.assertEqual(self_obj.fa_group_ids, [7])
        self.assertEqual(self_obj.window_group_ids, list(range(7)))
        self.assertEqual(self_obj.hash_block_size, 64)
        self.assertEqual(self_obj.group_metas[8].token_block_size, 16)
        self.assertEqual(self_obj.group_metas[8].tail_blocks, 0)
        expected_fa = (8 * 1000 + 4095) // 4096 * 4096
        expected_wa = (40 * 19008 * 2 + 4095) // 4096 * 4096
        self.assertEqual(
            self_obj.file_size, {"FA": expected_fa, "WA": expected_wa}
        )


if __name__ == "__main__":
    unittest.main()
