# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio model loader node for ComfyUI.

Loads the 3.3B audio-only transformer checkpoint with INT8 or bf16
precision. Keeps model on CPU until needed for sampling, allowing
ComfyUI to manage VRAM across nodes.
"""

import logging

import comfy.model_management
import torch

from .utils import (
    TRANSFORMER_BF16,
    TRANSFORMER_INT8,
    download_model,
    load_transformer,
)

logger = logging.getLogger(__name__)


class ScenemaAudioModelLoader:
    """Loads the Scenema Audio transformer model.

    Downloads the checkpoint from HuggingFace on first use and caches it.
    Supports INT8 (4.9 GB) and bf16 (9.8 GB) precision.
    Model stays on CPU until sampling to leave VRAM free for text encoding.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["int8", "bf16"], {"default": "int8"}),
            },
        }

    def load(self, precision):
        filename = TRANSFORMER_INT8 if precision == "int8" else TRANSFORMER_BF16
        logger.info("Downloading/loading %s...", filename)
        path = download_model(filename)

        mdl_wrapper, config = load_transformer(path)

        # Keep on CPU — moved to GPU on demand during sampling
        device = comfy.model_management.get_torch_device()

        return ({"model": mdl_wrapper, "config": config, "device": device},)
