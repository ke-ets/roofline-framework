"""Unit tests for FlopCounter."""

import pytest
from roofline.core.layer_info import LayerInfo
from roofline.core.flop_counter import FlopCounter


@pytest.fixture
def fc():
    return FlopCounter()


def _make_layer(layer_type, input_shapes=None, output_shapes=None, attrs=None, num_params=0):
    return LayerInfo(
        name="test",
        layer_type=layer_type,
        input_shapes=input_shapes or [],
        output_shapes=output_shapes or [],
        num_params=num_params,
        dtype="float32",
        attrs=attrs or {},
    )


# ---------------------------------------------------------------------------
# Linear / Gemm
# ---------------------------------------------------------------------------

class TestLinear:
    def test_linear_from_attrs(self, fc):
        layer = _make_layer(
            "Linear",
            attrs={"in_features": 512, "out_features": 256, "bias": False},
        )
        assert fc.count(layer) == 2 * 512 * 256

    def test_linear_with_bias(self, fc):
        layer = _make_layer(
            "Linear",
            attrs={"in_features": 512, "out_features": 256, "bias": True},
        )
        expected = 2 * 512 * 256 + 256
        assert fc.count(layer) == expected

    def test_linear_batch(self, fc):
        layer = _make_layer(
            "Linear",
            input_shapes=[(4, 512)],
            attrs={"in_features": 512, "out_features": 256, "bias": False},
        )
        assert fc.count(layer) == 2 * 4 * 512 * 256

    def test_gemm_from_shapes(self, fc):
        # Gemm: (1, 128) × (128, 64) = 2 * 1 * 128 * 64
        layer = _make_layer(
            "Gemm",
            input_shapes=[(1, 128), (128, 64)],
        )
        assert fc.count(layer) == 2 * 1 * 128 * 64

    def test_matmul_batched(self, fc):
        # (2, 8, 64) × (2, 64, 32) → 2 * 2*8 * 64 * 32
        layer = _make_layer(
            "MatMul",
            input_shapes=[(2, 8, 64), (2, 64, 32)],
        )
        expected = 2 * (2 * 8) * 64 * 32
        assert fc.count(layer) == expected


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------

class TestConv:
    def test_conv2d_basic(self, fc):
        # Conv2d: 2 * Cin * Cout * Kh * Kw * Ho * Wo (batch=1)
        layer = _make_layer(
            "Conv2d",
            input_shapes=[(1, 3, 224, 224)],
            output_shapes=[(1, 64, 112, 112)],
            attrs={"in_channels": 3, "out_channels": 64, "kernel_size": (7, 7), "groups": 1, "bias": False},
        )
        # Spatial output = 1 * 112 * 112
        expected = 2 * 3 * 64 * 7 * 7 * 112 * 112
        assert fc.count(layer) == expected

    def test_conv2d_depthwise(self, fc):
        # Depthwise: groups == in_channels
        layer = _make_layer(
            "Conv2d",
            input_shapes=[(1, 32, 56, 56)],
            output_shapes=[(1, 32, 56, 56)],
            attrs={"in_channels": 32, "out_channels": 32, "kernel_size": (3, 3), "groups": 32, "bias": False},
        )
        # (32 // 32) * 32 * 9 * 56 * 56
        expected = 2 * 1 * 32 * 9 * 56 * 56
        assert fc.count(layer) == expected

    def test_conv1d(self, fc):
        layer = _make_layer(
            "Conv1d",
            input_shapes=[(1, 16, 128)],
            output_shapes=[(1, 32, 128)],
            attrs={"in_channels": 16, "out_channels": 32, "kernel_size": (3,), "groups": 1},
        )
        expected = 2 * 16 * 32 * 3 * 128
        assert fc.count(layer) == expected


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    def test_mha_basic(self, fc):
        layer = _make_layer(
            "MultiHeadAttention",
            input_shapes=[(1, 128, 768)],
            attrs={"embed_dim": 768, "num_heads": 12},
        )
        flops = fc.count(layer)
        # Should be large (> 100M for these dims)
        assert flops > 100_000_000
        # QKV projections alone: 3 * 2 * 1 * 128 * 768^2 = 3 * 2 * 128 * 589824
        qkv = 3 * 2 * 1 * 128 * 768 * 768
        assert flops >= qkv


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNorm:
    def test_layernorm(self, fc):
        layer = _make_layer(
            "LayerNorm",
            input_shapes=[(1, 128, 768)],
        )
        expected = 2 * 1 * 128 * 768
        assert fc.count(layer) == expected

    def test_batchnorm(self, fc):
        layer = _make_layer(
            "BatchNorm",
            input_shapes=[(32, 64, 56, 56)],
        )
        expected = 2 * 32 * 64 * 56 * 56
        assert fc.count(layer) == expected


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_embedding_zero_flops(self, fc):
        layer = _make_layer("Embedding", attrs={"num_embeddings": 50000, "embedding_dim": 768})
        assert fc.count(layer) == 0


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestActivations:
    @pytest.mark.parametrize("layer_type,multiplier", [
        ("ReLU", 1),
        ("GELU", 8),
        ("SiLU", 4),
        ("Sigmoid", 4),
    ])
    def test_activation_flops(self, fc, layer_type, multiplier):
        layer = _make_layer(layer_type, input_shapes=[(1, 128, 768)])
        expected = multiplier * 1 * 128 * 768
        assert fc.count(layer) == expected


# ---------------------------------------------------------------------------
# Recurrent
# ---------------------------------------------------------------------------

class TestRecurrent:
    def test_lstm_basic(self, fc):
        layer = _make_layer(
            "LSTM",
            input_shapes=[(1, 64, 256)],   # batch=1, seq=64, input=256
            attrs={"input_size": 256, "hidden_size": 512, "num_layers": 1, "bidirectional": False},
        )
        flops = fc.count(layer)
        # 4 gates × (2*256*512 + 2*512*512 + 512) per step × 64 steps
        per_step = 4 * (2 * 256 * 512 + 2 * 512 * 512 + 512)
        expected = 1 * 64 * per_step
        assert flops == expected

    def test_gru_bidirectional(self, fc):
        layer = _make_layer(
            "GRU",
            input_shapes=[(1, 32, 128)],
            attrs={"input_size": 128, "hidden_size": 256, "num_layers": 1, "bidirectional": True},
        )
        flops = fc.count(layer)
        per_step = 3 * (2 * 128 * 256 + 2 * 256 * 256 + 256)
        expected = 1 * 32 * per_step * 2  # bidirectional
        assert flops == expected


# ---------------------------------------------------------------------------
# Unknown / zero-flops ops
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_unknown_returns_zero(self, fc):
        layer = _make_layer("NonExistentOp")
        assert fc.count(layer) == 0

    def test_reshape_zero(self, fc):
        layer = _make_layer("Reshape", input_shapes=[(1, 512)])
        assert fc.count(layer) == 0

    def test_dropout_zero(self, fc):
        layer = _make_layer("Dropout", input_shapes=[(1, 512)])
        assert fc.count(layer) == 0


# ---------------------------------------------------------------------------
# Custom handler registration
# ---------------------------------------------------------------------------

class TestCustomHandler:
    def test_register_custom(self, fc):
        @fc.register("MyOp")
        def count_my_op(layer):
            return layer.attrs.get("n", 0) * 5

        layer = _make_layer("MyOp", attrs={"n": 100})
        assert fc.count(layer) == 500
