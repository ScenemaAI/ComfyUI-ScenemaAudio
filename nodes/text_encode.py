# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding.

Ports production Scenema Audio's three-path Gemma encoding strategy:
1. NF4 quantized on GPU (~8GB, ~0.1s/encode) — auto-selected for 12-39GB cards
2. bf16 GPU-resident (~24GB, ~1-2s/encode) — auto-selected for 40GB+ cards
3. bf16 CPU streaming (~7s/encode) — fallback for <12GB cards

All paths cache the text encoder and embeddings processor across calls,
so subsequent encodes reuse the same models without rebuild/teardown.
This matches production/scenema-audio/src/audio_core/engine.py.
"""

import gc
import logging
import os

import comfy.model_management
import torch
from huggingface_hub import HfFolder, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import OffloadMode
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

from .utils import download_model, PIPELINE_CKPT

logger = logging.getLogger(__name__)

DEFAULT_GEMMA = "google/gemma-3-12b-it"

# VRAM thresholds matching production:
# >= 40GB: bf16 Gemma fully on GPU (fastest, best quality)
# 12-39GB: NF4 quantized Gemma on GPU (~8GB, minor quality tradeoff)
# <12GB: streaming from CPU RAM (slow but works everywhere)
HIGH_VRAM_THRESHOLD_GB = 40
NF4_MIN_VRAM_GB = 12

# Module-level cache — persists across node invocations for the lifetime
# of the ComfyUI process. Rebuilding these is the difference between ~5s
# and ~50s per generation.
_PIPELINE: DistilledPipeline | None = None
_PIPELINE_KEY: str | None = None
_RESIDENT_TEXT_ENCODER = None
_NF4_GEMMA_MODEL = None
_CACHED_EMB_PROC = None
_CACHED_TOKENIZER: LTXVGemmaTokenizer | None = None
_ENCODE_MODE: str | None = None  # "nf4" | "bf16_gpu" | "streaming"


_HF_TOKEN_MSG = (
    "Gemma 3 12B is a gated model and requires a HuggingFace token.\n"
    "\n"
    "1. Visit https://huggingface.co/google/gemma-3-12b-it and click "
    "'Agree and access repository'.\n"
    "2. Create a token at https://huggingface.co/settings/tokens "
    "(any scope with read access works).\n"
    "3. Provide the token in one of these ways before launching ComfyUI:\n"
    "     huggingface-cli login\n"
    "   OR set the environment variable:\n"
    "     export HF_TOKEN=hf_...\n"
    "\n"
    "Once done, restart ComfyUI and retry."
)


def _check_hf_token():
    """Fail loudly if no HF token is available before we try to download Gemma."""
    has_env = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    has_cli = bool(HfFolder.get_token())
    if not (has_env or has_cli):
        raise RuntimeError(_HF_TOKEN_MSG)


def _resolve_gemma_path(gemma_path):
    """Resolve Gemma path to a local directory (auto-downloads if needed).

    Fails loudly with actionable instructions if the user hasn't set up
    HuggingFace credentials — Gemma 3 12B is gated and can't be
    downloaded anonymously.
    """
    if os.path.isdir(gemma_path):
        return gemma_path

    _check_hf_token()
    try:
        return snapshot_download(gemma_path)
    except GatedRepoError as e:
        raise RuntimeError(
            f"HuggingFace rejected the token for {gemma_path}. "
            f"Make sure you've accepted the license at "
            f"https://huggingface.co/{gemma_path}\n\nOriginal error: {e}"
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HuggingFace repo {gemma_path} not found or you lack access. "
            f"Check the path and your token permissions.\n\nOriginal error: {e}"
        ) from e


def _select_encode_mode(vram_gb: float) -> str:
    """Match production's Gemma loading strategy for the current VRAM tier."""
    if vram_gb >= HIGH_VRAM_THRESHOLD_GB:
        return "bf16_gpu"
    if vram_gb >= NF4_MIN_VRAM_GB:
        return "nf4"
    return "streaming"


def _build_pipeline(gemma_path, pipeline_path):
    """Instantiate DistilledPipeline once and cache it."""
    global _PIPELINE, _PIPELINE_KEY
    key = f"{gemma_path}::{pipeline_path}"
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    if _PIPELINE is not None:
        logger.info("Rebuilding pipeline (paths changed)")
        _teardown_cache()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    offload = OffloadMode.NONE if vram_gb >= HIGH_VRAM_THRESHOLD_GB else OffloadMode.CPU

    logger.info("Building DistilledPipeline (this happens once)...")
    _PIPELINE = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=offload,
    )
    _PIPELINE_KEY = key
    return _PIPELINE


def _build_nf4_gemma(gemma_path):
    """Load Gemma 3 12B with BitsAndBytes NF4 quantization (~8GB on GPU)."""
    logger.info("Loading Gemma NF4 on GPU (one-time)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path,
        quantization_config=quant_config,
        device_map="cuda",
        dtype=torch.bfloat16,
    ).eval()
    vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
    logger.info("Gemma NF4 resident: %.1fGB VRAM", vram_used_gb)
    return model


def _ensure_cache(gemma_path, pipeline_path):
    """Set up the encode-mode-appropriate caches. Idempotent per process."""
    global _RESIDENT_TEXT_ENCODER, _NF4_GEMMA_MODEL, _CACHED_EMB_PROC
    global _CACHED_TOKENIZER, _ENCODE_MODE

    pipeline = _build_pipeline(gemma_path, pipeline_path)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    mode = _select_encode_mode(vram_gb)

    if _ENCODE_MODE == mode:
        return mode

    _teardown_cache(keep_pipeline=True)
    _ENCODE_MODE = mode
    pe = pipeline.prompt_encoder

    if mode == "nf4":
        _NF4_GEMMA_MODEL = _build_nf4_gemma(gemma_path)
        _CACHED_EMB_PROC = pe._embeddings_processor_builder.build(
            device="cuda", dtype=torch.bfloat16,
        ).eval()
        _CACHED_TOKENIZER = LTXVGemmaTokenizer(gemma_path)
        logger.info("NF4 encode path ready (Gemma + emb_proc GPU-resident)")
    elif mode == "bf16_gpu":
        logger.info("Loading bf16 Gemma text encoder on GPU (one-time)...")
        _RESIDENT_TEXT_ENCODER = pe._text_encoder_builder.build(
            device=torch.device("cuda"), dtype=torch.bfloat16,
        ).eval()
        _CACHED_EMB_PROC = pe._embeddings_processor_builder.build(
            device="cuda", dtype=torch.bfloat16,
        ).eval()
        vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info("bf16 Gemma resident: %.1fGB VRAM", vram_used_gb)
    else:
        logger.info("Streaming encode path (Gemma streams from CPU per call)")

    return mode


def _teardown_cache(keep_pipeline=False):
    """Free cached models (used when switching modes or on unload)."""
    global _PIPELINE, _PIPELINE_KEY, _RESIDENT_TEXT_ENCODER
    global _NF4_GEMMA_MODEL, _CACHED_EMB_PROC, _CACHED_TOKENIZER, _ENCODE_MODE

    if _RESIDENT_TEXT_ENCODER is not None:
        try:
            _RESIDENT_TEXT_ENCODER.teardown()
        except AttributeError:
            pass
        _RESIDENT_TEXT_ENCODER = None
    _NF4_GEMMA_MODEL = None
    _CACHED_EMB_PROC = None
    _CACHED_TOKENIZER = None
    _ENCODE_MODE = None
    if not keep_pipeline:
        _PIPELINE = None
        _PIPELINE_KEY = None
    gc.collect()
    torch.cuda.empty_cache()


def _encode_via_pipeline(compiled_prompt, gemma_path, pipeline_path):
    """Encode via production's three-path strategy.

    Reuses cached text encoder + emb_proc across calls, matching
    production/scenema-audio/src/audio_core/engine.py.encode_text.
    """
    mode = _ensure_cache(gemma_path, pipeline_path)
    pipeline = _PIPELINE

    with torch.inference_mode():
        if mode == "nf4":
            tp = _CACHED_TOKENIZER.tokenize_with_weights(compiled_prompt)["gemma"]
            ids = torch.tensor([[t[0] for t in tp]], device="cuda")
            mask = torch.tensor([[w[1] for w in tp]], device="cuda")
            out = _NF4_GEMMA_MODEL.model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
            )
            hs = out.hidden_states
            am = mask
            emb = _CACHED_EMB_PROC.process_hidden_states(hs, am)
            del out, ids
        elif mode == "bf16_gpu":
            hs, am = _RESIDENT_TEXT_ENCODER.encode(compiled_prompt)
            emb = _CACHED_EMB_PROC.process_hidden_states(hs, am)
        else:
            (emb,) = pipeline.prompt_encoder([compiled_prompt])

        vc = emb.video_encoding
        ac = emb.audio_encoding

    return vc, ac


class ScenemaAudioTextEncode:
    """Encodes text prompts via the LTX pipeline's Gemma-driven encoder.

    Uses the same three-path strategy as production Scenema Audio, chosen
    automatically based on available VRAM.
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
