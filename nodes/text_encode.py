# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding node for ComfyUI.

Delegates entirely to the LTX pipeline's PromptEncoder — the same code
path production Scenema Audio uses. Custom shortcuts (audio-only
extraction, HF device_map Gemma) diverged subtly enough to flatten
emotional dynamics; this matches production byte-for-byte.
"""

import gc
import logging
import os

import comfy.model_management
import torch
from huggingface_hub import snapshot_download
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import OffloadMode

from .utils import download_model, PIPELINE_CKPT

logger = logging.getLogger(__name__)

DEFAULT_GEMMA = "google/gemma-3-12b-it"

# Cached pipeline (kept alive across calls). Rebuilding is expensive.
_PIPELINE: DistilledPipeline | None = None
_PIPELINE_KEY: str | None = None


def _resolve_gemma_path(gemma_path):
    """Resolve Gemma path to a local directory."""
    if os.path.isdir(gemma_path):
        return gemma_path
    return snapshot_download(gemma_path)


def _get_pipeline(gemma_path, pipeline_path):
    """Get or build the DistilledPipeline. Cached across calls."""
    global _PIPELINE, _PIPELINE_KEY
    key = f"{gemma_path}::{pipeline_path}"
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    if _PIPELINE is not None:
        logger.info("Rebuilding pipeline (paths changed)")
        del _PIPELINE
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Building DistilledPipeline (Gemma + emb_proc)...")
    _PIPELINE = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=OffloadMode.CPU,
    )
    _PIPELINE_KEY = key
    logger.info("Pipeline ready")
    return _PIPELINE


def _encode_via_pipeline(compiled_prompt, gemma_path, pipeline_path):
    """Encode a prompt via the LTX pipeline's PromptEncoder.

    Matches production Scenema Audio's CPU-streaming code path exactly:
    the pipeline internally streams Gemma layer-by-layer, then runs the
    full process_hidden_states on the embeddings processor. This is
    critical for expressive output — subtle differences in the encoding
    path flatten action-tag responsiveness (laughs, whispers, etc.).
    """
    pipeline = _get_pipeline(gemma_path, pipeline_path)
    logger.info("Encoding prompt via pipeline.prompt_encoder...")
    with torch.inference_mode():
        (emb,) = pipeline.prompt_encoder([compiled_prompt])
        vc = emb.video_encoding
        ac = emb.audio_encoding
    gc.collect()
    torch.cuda.empty_cache()
    return vc, ac


class ScenemaAudioTextEncode:
    """Encodes text prompts via the LTX pipeline's Gemma-driven encoder.

    Runs the same code path as production Scenema Audio (`pipeline.prompt_encoder`).
    Gemma is streamed from CPU RAM on demand — no explicit quantization
    choice is exposed to users; the pipeline manages memory automatically.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "encode"
    RETURN_TYPES = ("SA_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "compiled_prompt": ("STRING", {"forceInput": True}),
                "model": ("SA_MODEL",),
            },
            "optional": {
                "gemma_path": ("STRING", {"default": "auto"}),
            },
        }

    def encode(self, compiled_prompt, model, gemma_path="auto"):
        if gemma_path == "auto":
            gemma_path = DEFAULT_GEMMA

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_CKPT)

        vc, ac = _encode_via_pipeline(compiled_prompt, gemma_local, pipeline_path)

        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
