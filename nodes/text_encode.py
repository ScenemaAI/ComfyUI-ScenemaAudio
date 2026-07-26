# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding node for ComfyUI.

Encodes compiled text prompts via Gemma 3 12B. Supports multiple
quantization strategies for different VRAM sizes:
- nf4: Pre-quantized NF4 on GPU (~8GB, needs 12GB+ card) — recommended
- bf16: Full precision, all on GPU (needs 40GB+) — best quality
- cpu: bf16 with CPU/GPU split via device_map (fallback for <12GB cards)

Gemma and the embeddings processor are cached across calls (module-level
singletons). This eliminates the 5-10s reload cost on repeat generations.
"""

import gc
import logging
import os

import comfy.model_management
import torch
from huggingface_hub import snapshot_download
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import OffloadMode
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

from .utils import download_model, PIPELINE_AUDIO_CKPT

logger = logging.getLogger(__name__)

DEFAULT_GEMMA_NF4 = "unsloth/gemma-3-12b-it-bnb-4bit"
DEFAULT_GEMMA_BF16 = "google/gemma-3-12b-it"

# VRAM threshold above which we keep Gemma resident on GPU across calls.
# Below this, Gemma is loaded per-call so the transformer has room to sample.
HIGH_VRAM_GEMMA_CACHE_GB = 24

# Module-level singletons — persist across node invocations for the lifetime
# of the ComfyUI process. Eliminates model reload cost on repeat generations.
_GEMMA_CACHE: dict[str, Gemma3ForConditionalGeneration] = {}
_EMB_PROC_CACHE: dict[str, tuple] = {}


def _auto_quantize(vram_gb: float) -> str:
    """Auto-select the quantization mode for the current GPU.

    40GB+ → bf16 (best quality, GPU-resident)
    12-39GB → nf4 (recommended; fast, small VRAM footprint)
    <12GB → cpu (bf16 streaming from CPU; slow but works on 8GB)
    """
    if vram_gb >= 40:
        return "bf16"
    if vram_gb >= 12:
        return "nf4"
    return "cpu"


def _get_default_gemma(quantize):
    """Get the default Gemma model path for a quantization mode."""
    if quantize == "nf4":
        return DEFAULT_GEMMA_NF4
    return DEFAULT_GEMMA_BF16


def _resolve_gemma_path(gemma_path):
    """Resolve Gemma path to a local directory."""
    if os.path.isdir(gemma_path):
        return gemma_path
    return snapshot_download(gemma_path)


def _strip_video_components(emb_proc):
    """Remove video-only components from the embeddings processor.

    Deletes video_aggregate_embed (1.5GB) and video_connector (3.5GB),
    keeping only audio_aggregate_embed (~0.7GB) and audio_connector (~0.5GB).
    Reduces emb_proc from 6.2GB to ~1.3GB so it fits on GPU.
    """
    if hasattr(emb_proc, 'feature_extractor'):
        fe = emb_proc.feature_extractor
        if hasattr(fe, 'video_aggregate_embed'):
            del fe.video_aggregate_embed
            fe.video_aggregate_embed = None
    if hasattr(emb_proc, 'video_connector'):
        del emb_proc.video_connector
        emb_proc.video_connector = None
    gc.collect()


def _audio_only_embeddings(emb_proc, hidden_states, attention_mask):
    """Run only the audio path of the embeddings processor.

    Bypasses the video feature extractor and video connector entirely.
    Returns (vc, ac) where vc is a dummy zero tensor for sampler compatibility.
    """
    from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask, _to_binary_mask

    fe = emb_proc.feature_extractor

    encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states

    from ltx_core.text_encoders.gemma.feature_extractor import norm_and_concat_per_token_rms, _rescale_norm
    normed = norm_and_concat_per_token_rms(encoded, attention_mask)
    normed = normed.to(encoded.dtype)

    a_dim = fe.audio_aggregate_embed.out_features
    audio_feats = fe.audio_aggregate_embed(_rescale_norm(normed, a_dim, fe.embedding_dim))
    del encoded, normed

    additive_mask = convert_to_additive_mask(attention_mask, audio_feats.dtype)
    audio_encoded, _ = emb_proc.audio_connector(audio_feats, additive_mask)
    del audio_feats, additive_mask

    vc = torch.zeros(1, audio_encoded.shape[1], 4096, device=audio_encoded.device, dtype=audio_encoded.dtype)
    ac = audio_encoded

    return vc, ac


def _build_embeddings_processor(gemma_path, pipeline_path):
    """Build the embeddings processor and tokenizer on CPU (uncached)."""
    tokenizer_path = gemma_path
    if not os.path.exists(os.path.join(gemma_path, "tokenizer.model")):
        tokenizer_path = _resolve_gemma_path(DEFAULT_GEMMA_BF16)

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=tokenizer_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=OffloadMode.CPU,
    )
    pe = pipeline.prompt_encoder
    emb_proc = pe._embeddings_processor_builder.build(
        device="cpu", dtype=torch.bfloat16
    ).eval()
    tokenizer = LTXVGemmaTokenizer(tokenizer_path)

    del pipeline, pe
    gc.collect()

    return emb_proc, tokenizer


def get_or_load_emb_proc(gemma_path, pipeline_path):
    """Get cached emb_proc + tokenizer, or build and cache.

    Emb_proc is stored on CPU. Callers move to GPU during use and back
    to CPU after — keeps the cache VRAM-neutral between generations.
    Stripping video components happens once at cache time.
    """
    if pipeline_path in _EMB_PROC_CACHE:
        return _EMB_PROC_CACHE[pipeline_path]

    logger.info("Building embeddings processor (first call, cached hereafter)...")
    emb_proc, tokenizer = _build_embeddings_processor(gemma_path, pipeline_path)
    _strip_video_components(emb_proc)
    _EMB_PROC_CACHE[pipeline_path] = (emb_proc, tokenizer)
    return emb_proc, tokenizer


def get_or_load_gemma(gemma_path, quantize, load_kwargs):
    """Get cached Gemma if VRAM allows co-residency with the transformer.

    24GB+ cards keep Gemma resident permanently (no reload cost on repeat
    generations). 12-23GB cards load per-call so the transformer has room
    to sample. Switching quantize modes evicts the old cached model.

    Returns:
        (gemma_model, was_cached) tuple. Callers use was_cached to decide
        whether to free the model after use.
    """
    key = f"{gemma_path}::{quantize}"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # CPU-streaming mode never caches — the whole point is to not occupy VRAM
    should_cache = vram_gb >= HIGH_VRAM_GEMMA_CACHE_GB and quantize != "cpu"

    if should_cache and key in _GEMMA_CACHE:
        return _GEMMA_CACHE[key], True

    # Evict any different cached Gemma before loading a new one
    if should_cache and _GEMMA_CACHE:
        logger.info("Evicting cached Gemma (quantize mode changed)")
        _GEMMA_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Loading Gemma (%s)...", quantize)
    gemma = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, **load_kwargs
    ).eval()

    if should_cache:
        _GEMMA_CACHE[key] = gemma
        logger.info("Gemma cached for lifetime of process")

    return gemma, should_cache


def _free_vram():
    """Free ComfyUI-managed models and empty the CUDA cache."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def _build_gemma_load_kwargs(quantize, vram_gb):
    """Build the from_pretrained kwargs for a given quantize mode."""
    if quantize == "nf4":
        # Pre-quantized NF4 must load entirely on GPU (bnb can't CPU-split)
        return {
            "device_map": "auto",
            "max_memory": {0: f"{int(vram_gb - 2)}GiB", "cpu": "32GiB"},
            "dtype": torch.bfloat16,
        }
    if quantize == "cpu":
        # bf16 split across CPU/GPU — cap at 6GB to co-exist with transformer
        return {
            "device_map": "auto",
            "max_memory": {0: "6GiB", "cpu": "32GiB"},
            "dtype": torch.bfloat16,
        }
    # bf16 all on GPU (40GB+ cards)
    return {"device_map": "cuda", "dtype": torch.bfloat16}


def _encode_gemma(compiled_prompt, gemma_path, pipeline_path, load_kwargs, quantize):
    """Encode a prompt via Gemma + audio-only embeddings processor.

    Uses cached Gemma and emb_proc when available (see get_or_load_*).
    Emb_proc moves CPU→GPU→CPU per call to keep VRAM footprint minimal.
    """
    emb_proc, tokenizer = get_or_load_emb_proc(gemma_path, pipeline_path)
    gemma_model, was_cached = get_or_load_gemma(gemma_path, quantize, load_kwargs)

    with torch.inference_mode():
        tp = tokenizer.tokenize_with_weights(compiled_prompt)["gemma"]
        ids = torch.tensor([[t[0] for t in tp]], device="cuda")
        mask = torch.tensor([[w[1] for w in tp]], device="cuda")
        out = gemma_model.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True,
        )
        hs_cpu = tuple(h.cpu() for h in out.hidden_states)
        mask_cpu = mask.cpu()
        del out, ids, mask

        if not was_cached:
            del gemma_model
            gc.collect()
            torch.cuda.empty_cache()

        emb_proc.cuda()
        hs_gpu = tuple(h.cuda() for h in hs_cpu)
        mask_gpu = mask_cpu.cuda()
        del hs_cpu, mask_cpu

        vc, ac = _audio_only_embeddings(emb_proc, hs_gpu, mask_gpu)
        del hs_gpu, mask_gpu

    emb_proc.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    return vc, ac


class ScenemaAudioTextEncode:
    """Encodes text prompts via Gemma 3 12B for the audio diffusion model.

    Quantization modes:
    - nf4: Pre-quantized NF4 on GPU. Fast. Needs 12GB+ VRAM. Recommended.
    - bf16: Full precision on GPU. Best quality. Needs 40GB+ VRAM.
    - cpu: bf16 split across CPU/GPU. Slow but works on 8GB cards.
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
                "quantize": (["auto", "nf4", "bf16", "cpu"], {"default": "auto"}),
            },
        }

    def encode(self, compiled_prompt, model, gemma_path="auto", quantize="auto"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if quantize == "auto":
            quantize = _auto_quantize(vram_gb)
            logger.info("VRAM %.0fGB: auto-selected quantize=%s", vram_gb, quantize)

        if gemma_path == "auto":
            gemma_path = _get_default_gemma(quantize)

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_AUDIO_CKPT)

        logger.info("Encoding prompt with Gemma (%s, %s)...", quantize, gemma_path)

        load_kwargs = _build_gemma_load_kwargs(quantize, vram_gb)
        vc, ac = _encode_gemma(compiled_prompt, gemma_local, pipeline_path, load_kwargs, quantize)

        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
