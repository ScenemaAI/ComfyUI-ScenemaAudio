# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for utility functions (non-GPU)."""

import sys
import os
import types

# Mock comfy before importing nodes
comfy_mock = types.ModuleType("comfy")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.get_torch_device = lambda: "cpu"
comfy_mock.model_management = comfy_mm
sys.modules["comfy"] = comfy_mock
sys.modules["comfy.model_management"] = comfy_mm

# Add package root
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import torch
from nodes.utils import (
    Int8Linear,
    FPS,
    MAX_REF_SECONDS,
    HF_REPO,
    TRANSFORMER_BF16,
    TRANSFORMER_INT8,
    PIPELINE_CKPT,
    VAE_ENCODER_CKPT,
)


class TestInt8Linear:
    def test_forward_shape(self):
        out_features, in_features = 64, 32
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.randn(out_features)
        layer = Int8Linear(weight_int8, scale)

        x = torch.randn(2, in_features)
        y = layer(x)
        assert y.shape == (2, out_features)

    def test_forward_with_bias(self):
        out_features, in_features = 64, 32
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.randn(out_features)
        bias = torch.randn(out_features)
        layer = Int8Linear(weight_int8, scale, bias)

        x = torch.randn(2, in_features)
        y = layer(x)
        assert y.shape == (2, out_features)

    def test_no_bias_is_none(self):
        weight_int8 = torch.randint(-128, 127, (8, 4), dtype=torch.int8)
        scale = torch.randn(8)
        layer = Int8Linear(weight_int8, scale)
        assert layer.bias is None

    def test_dtype_matches_input(self):
        weight_int8 = torch.randint(-128, 127, (8, 4), dtype=torch.int8)
        scale = torch.randn(8)
        layer = Int8Linear(weight_int8, scale)

        x_f32 = torch.randn(1, 4, dtype=torch.float32)
        y = layer(x_f32)
        assert y.dtype == torch.float32

        x_f16 = torch.randn(1, 4, dtype=torch.float16)
        y = layer(x_f16)
        assert y.dtype == torch.float16
