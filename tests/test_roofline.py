"""Unit tests for RooflineAnalyzer and AnalysisResults."""

import pytest

from roofline.core.analyzer import RooflineAnalyzer, AnalysisResults, _infer_source
from roofline.core.layer_info import LayerInfo, LayerStats
from roofline.hardware.hw_spec import HWSpec
from roofline.hardware.hw_database import HW_DB, lookup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h100():
    return HW_DB["h100_sxm"]


@pytest.fixture
def a100():
    return HW_DB["a100_80gb"]


@pytest.fixture
def analyzer():
    return RooflineAnalyzer()


def _dummy_layer(layer_type="Linear", in_f=512, out_f=256):
    return LayerInfo(
        name=f"{layer_type}_0",
        layer_type=layer_type,
        input_shapes=[(1, in_f)],
        output_shapes=[(1, out_f)],
        num_params=in_f * out_f,
        dtype="float32",
        attrs={"in_features": in_f, "out_features": out_f, "bias": False},
    )


# ---------------------------------------------------------------------------
# HW_DB
# ---------------------------------------------------------------------------

class TestHWDatabase:
    def test_h100_exists(self):
        assert "h100_sxm" in HW_DB

    def test_a100_exists(self):
        assert "a100_80gb" in HW_DB

    def test_apple_m4_pro_exists(self):
        assert "m4_pro" in HW_DB

    def test_all_entries_have_peak_flops(self):
        for key, spec in HW_DB.items():
            assert spec.peak_flops, f"{key} has empty peak_flops"

    def test_all_entries_have_mem_bw(self):
        for key, spec in HW_DB.items():
            assert spec.peak_mem_bw > 0, f"{key} has zero peak_mem_bw"

    def test_apple_unified_memory(self):
        for key in ("m2", "m3_pro", "m4_max", "m5_ultra"):
            assert HW_DB[key].unified_memory is True

    def test_alias_lookup_h100(self):
        spec = lookup("h100 sxm")
        assert spec is not None
        assert "H100" in spec.name

    def test_alias_lookup_rtx4090(self):
        spec = lookup("rtx 4090")
        assert spec is not None

    def test_alias_lookup_raspberry_pi5(self):
        spec = lookup("pi5")
        assert spec is not None
        assert spec.name == "Raspberry Pi 5"

    def test_alias_lookup_raspberry_pi4(self):
        spec = lookup("pi4")
        assert spec is not None
        assert spec.name == "Raspberry Pi 4"

    def test_alias_lookup_arduino_nicla(self):
        spec = lookup("nicla")
        assert spec is not None
        assert spec.name == "Arduino Nicla Vision"

    def test_lookup_unknown_returns_none(self):
        assert lookup("nonexistent_gpu_xyz") is None


# ---------------------------------------------------------------------------
# HWSpec ridge point + attainable performance
# ---------------------------------------------------------------------------

class TestHWSpec:
    def test_ridge_point_h100(self, h100):
        # H100 FP16: 989e12 / 3.35e12 ≈ 295
        ridge = h100.ridge_point("float16")
        assert 280 < ridge < 310

    def test_attainable_memory_bound(self, h100):
        # AI = 1 → well below ridge → memory bound
        ai = 1.0
        perf = h100.attainable_performance(ai, "float16")
        expected = ai * h100.peak_mem_bw  # = 3.35e12
        assert abs(perf - expected) < 1e9

    def test_attainable_compute_bound(self, h100):
        # AI = 10000 → compute bound
        ai = 10_000.0
        perf = h100.attainable_performance(ai, "float16")
        assert abs(perf - h100.peak_flops["float16"]) < 1e9

    def test_attainable_at_ridge_point(self, h100):
        ridge = h100.ridge_point("float16")
        perf_mem = ridge * h100.peak_mem_bw
        perf_comp = h100.peak_flops["float16"]
        # At ridge both ceilings are equal
        assert abs(perf_mem - perf_comp) / perf_comp < 0.01


# ---------------------------------------------------------------------------
# RooflineAnalyzer with a tiny PyTorch model
# ---------------------------------------------------------------------------

class TestRooflineAnalyzer:
    def test_analyze_pytorch_mlp(self, analyzer, h100):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        results = analyzer.analyze(
            model=model,
            input_shapes=[(1, 256)],
            hw=h100,
            dtype="float32",
        )

        assert isinstance(results, AnalysisResults)
        assert len(results.layers) > 0
        assert results.total_flops > 0
        assert results.hw.name == h100.name

    def test_every_layer_has_bottleneck(self, analyzer, h100):
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        model = nn.Linear(128, 64)
        results = analyzer.analyze(
            model=model, input_shapes=[(1, 128)], hw=h100, dtype="float32"
        )
        for ls in results.layers:
            assert ls.bottleneck in ("compute", "memory")

    def test_theoretical_time_positive(self, analyzer, h100):
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        model = nn.Linear(1024, 1024)
        results = analyzer.analyze(
            model=model, input_shapes=[(1, 1024)], hw=h100, dtype="float16"
        )
        assert results.theoretical_time_ms > 0

    def test_training_mode_more_bytes(self, analyzer, h100):
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        model = nn.Linear(256, 256)
        r_inf = analyzer.analyze(model=model, input_shapes=[(1, 256)], hw=h100, mode="inference")
        r_tr = analyzer.analyze(model=model, input_shapes=[(1, 256)], hw=h100, mode="training")
        assert r_tr.total_bytes > r_inf.total_bytes

    def test_memory_vs_compute_bound_classification(self, analyzer):
        """A tiny batch=1 linear should be memory-bound on any GPU."""
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        hw = HW_DB["h100_sxm"]
        model = nn.Linear(4096, 4096)  # AI ≈ 1 at batch=1 → memory bound
        results = analyzer.analyze(
            model=model, input_shapes=[(1, 4096)], hw=hw, dtype="float16"
        )
        linear_layers = [l for l in results.layers if l.layer.layer_type == "Linear"]
        if linear_layers:
            assert linear_layers[0].bottleneck == "memory"

    def test_larger_batch_increases_ai(self, analyzer, h100):
        """Larger batch moves AI rightward (less memory-bound)."""
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        model = nn.Linear(4096, 4096)
        r1 = analyzer.analyze(model=model, input_shapes=[(1, 4096)], hw=h100, dtype="float16")
        r128 = analyzer.analyze(model=model, input_shapes=[(128, 4096)], hw=h100, dtype="float16")

        ai_1 = max((l.arithmetic_intensity for l in r1.layers if l.layer.layer_type == "Linear"), default=0)
        ai_128 = max((l.arithmetic_intensity for l in r128.layers if l.layer.layer_type == "Linear"), default=0)
        assert ai_128 > ai_1


# ---------------------------------------------------------------------------
# AnalysisResults helpers
# ---------------------------------------------------------------------------

class TestAnalysisResults:
    def _make_results(self, hw):
        layers = []
        for i, (lt, flops, w, a, bottleneck) in enumerate([
            ("Linear", 100_000_000, 1_000_000, 500_000, "memory"),
            ("Conv2d", 500_000_000, 200_000, 800_000, "compute"),
            ("BatchNorm", 1_000_000, 50_000, 100_000, "memory"),
        ]):
            li = _dummy_layer(lt)
            ls = LayerStats(
                layer=li,
                flops=flops,
                weight_bytes=w,
                activation_bytes=a,
                ridge_point=hw.ridge_point("float16"),
                attainable_perf=hw.attainable_performance(flops / (w + a), "float16"),
                bottleneck=bottleneck,
                theoretical_time_ms=flops / hw.peak_flops.get("float16", 1e12) * 1000,
                hw_name=hw.name,
                dtype_used="float16",
            )
            layers.append(ls)
        return AnalysisResults(layers=layers, hw=hw, dtype="float16", mode="inference")

    def test_total_flops(self):
        results = self._make_results(HW_DB["h100_sxm"])
        assert results.total_flops == 601_000_000

    def test_memory_bound_layers(self):
        results = self._make_results(HW_DB["h100_sxm"])
        assert len(results.memory_bound_layers()) == 2

    def test_compute_bound_layers(self):
        results = self._make_results(HW_DB["h100_sxm"])
        assert len(results.compute_bound_layers()) == 1

    def test_to_dataframe_shape(self):
        pytest.importorskip("pandas")
        results = self._make_results(HW_DB["h100_sxm"])
        df = results.to_dataframe()
        assert len(df) == 3
        assert "arithmetic_intensity" in df.columns
        assert "bottleneck" in df.columns


# ---------------------------------------------------------------------------
# Source auto-detection
# ---------------------------------------------------------------------------

class TestInferSource:
    def test_pytorch_module(self):
        try:
            import torch.nn as nn
            model = nn.Linear(10, 5)
            assert _infer_source(model) == "pytorch"
        except ImportError:
            pytest.skip("torch not installed")

    def test_onnx_file(self):
        assert _infer_source("model.onnx") == "onnx"

    def test_pt_file(self):
        assert _infer_source("model.pt") == "pytorch"

    def test_h5_file(self):
        assert _infer_source("model.h5") == "tensorflow"

    def test_zip_file(self):
        assert _infer_source("model.zip") == "zip"

    def test_hf_string(self):
        assert _infer_source("bert-base-uncased") == "huggingface"
