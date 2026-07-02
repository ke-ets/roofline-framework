"""Unit tests for MemoryEstimator."""

import pytest
from roofline.core.layer_info import LayerInfo
from roofline.core.memory_estimator import MemoryEstimator


@pytest.fixture
def me():
    return MemoryEstimator()


def _make_layer(layer_type, input_shapes=None, output_shapes=None, attrs=None, num_params=0, dtype="float32"):
    return LayerInfo(
        name="test",
        layer_type=layer_type,
        input_shapes=input_shapes or [],
        output_shapes=output_shapes or [],
        num_params=num_params,
        dtype=dtype,
        attrs=attrs or {},
    )


# ---------------------------------------------------------------------------
# Weight bytes
# ---------------------------------------------------------------------------

class TestWeightBytes:
    def test_linear_weight_bytes_from_num_params(self, me):
        # 512*256 weights + 256 bias = 131328 params × 4 bytes = 525312
        layer = _make_layer("Linear", num_params=512 * 256 + 256)
        w, a, g, o = me.estimate(layer, mode="inference", dtype="float32")
        assert w == (512 * 256 + 256) * 4

    def test_linear_weight_bytes_fp16(self, me):
        layer = _make_layer("Linear", num_params=1000)
        w, _, _, _ = me.estimate(layer, mode="inference", dtype="float16")
        assert w == 1000 * 2

    def test_linear_weight_bytes_int8(self, me):
        layer = _make_layer("Linear", num_params=1000)
        w, _, _, _ = me.estimate(layer, dtype="int8")
        assert w == 1000 * 1

    def test_linear_weight_from_attrs(self, me):
        layer = _make_layer(
            "Linear",
            num_params=0,
            attrs={"in_features": 128, "out_features": 64, "bias": True},
        )
        w, _, _, _ = me.estimate(layer, dtype="float32")
        assert w == (128 * 64 + 64) * 4

    def test_conv2d_weight_from_attrs(self, me):
        # 64 * 3 * 3 * 3 params (no bias)
        layer = _make_layer(
            "Conv2d",
            num_params=0,
            attrs={"in_channels": 3, "out_channels": 64, "kernel_size": (3, 3), "groups": 1, "bias": False},
        )
        w, _, _, _ = me.estimate(layer, dtype="float32")
        assert w == 64 * 3 * 3 * 3 * 4

    def test_embedding_weight_from_attrs(self, me):
        layer = _make_layer(
            "Embedding",
            num_params=0,
            attrs={"num_embeddings": 1000, "embedding_dim": 128},
        )
        w, _, _, _ = me.estimate(layer, dtype="float32")
        assert w == 1000 * 128 * 4


# ---------------------------------------------------------------------------
# Activation bytes
# ---------------------------------------------------------------------------

class TestActivationBytes:
    def test_activation_bytes_input_output(self, me):
        # Input: (1, 512) → 512 elems; Output: (1, 256) → 256 elems
        layer = _make_layer(
            "Linear",
            input_shapes=[(1, 512)],
            output_shapes=[(1, 256)],
        )
        _, a, _, _ = me.estimate(layer, dtype="float32")
        assert a == (512 + 256) * 4

    def test_activation_bytes_fp16(self, me):
        layer = _make_layer(
            "Linear",
            input_shapes=[(1, 512)],
            output_shapes=[(1, 256)],
        )
        _, a, _, _ = me.estimate(layer, dtype="float16")
        assert a == (512 + 256) * 2

    def test_conv2d_activation_bytes(self, me):
        # Input: (1, 3, 224, 224), Output: (1, 64, 112, 112)
        layer = _make_layer(
            "Conv2d",
            input_shapes=[(1, 3, 224, 224)],
            output_shapes=[(1, 64, 112, 112)],
        )
        _, a, _, _ = me.estimate(layer, dtype="float32")
        in_elems = 3 * 224 * 224
        out_elems = 64 * 112 * 112
        assert a == (in_elems + out_elems) * 4


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

class TestTrainingMode:
    def test_training_adds_grad_bytes(self, me):
        layer = _make_layer("Linear", num_params=1000)
        w_inf, _, g_inf, o_inf = me.estimate(layer, mode="inference")
        w_tr, _, g_tr, o_tr = me.estimate(layer, mode="training")

        assert g_inf == 0
        assert o_inf == 0
        assert g_tr == w_tr          # grads same size as weights
        assert o_tr == 1000 * 8     # Adam: 2 × fp32 per param

    def test_training_total_greater_than_inference(self, me):
        layer = _make_layer(
            "Linear",
            input_shapes=[(1, 512)],
            output_shapes=[(1, 256)],
            num_params=512 * 256,
        )
        w, a, g, o = me.estimate(layer, mode="inference")
        total_inf = w + a + g + o

        w2, a2, g2, o2 = me.estimate(layer, mode="training")
        total_tr = w2 + a2 + g2 + o2

        assert total_tr > total_inf


# ---------------------------------------------------------------------------
# dtype_bytes helper
# ---------------------------------------------------------------------------

class TestDtypeBytes:
    def test_known_dtypes(self):
        from roofline.core.layer_info import dtype_bytes
        assert dtype_bytes("float32") == 4
        assert dtype_bytes("float16") == 2
        assert dtype_bytes("bfloat16") == 2
        assert dtype_bytes("int8") == 1
        assert dtype_bytes("int4") == 0.5

    def test_unknown_dtype_raises(self):
        from roofline.core.layer_info import dtype_bytes
        with pytest.raises(ValueError):
            dtype_bytes("float128")
