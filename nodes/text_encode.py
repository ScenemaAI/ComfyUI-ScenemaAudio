# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding node for ComfyUI.

Encodes compiled text prompts via Gemma 3 12B. Supports multiple
quantization strategies for different VRAM sizes:
- nf4: Pre-quantized NF4 on GPU (~8GB, needs 12GB+ card)
- cpu: bf16 with CPU/GPU split via device_map (works on 8GB+)
- bf16: Full precision, all on GPU (needs 40GB+)
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

    # Run feature extractor's audio path only
    encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states

    # norm_and_concat_per_token_rms (inline from feature_extractor)
    from ltx_core.text_encoders.gemma.feature_extractor import norm_and_concat_per_token_rms, _rescale_norm
    normed = norm_and_concat_per_token_rms(encoded, attention_mask)
    normed = normed.to(encoded.dtype)

    a_dim = fe.audio_aggregate_embed.out_features
    audio_feats = fe.audio_aggregate_embed(_rescale_norm(normed, a_dim, fe.embedding_dim))
    del encoded, normed

    # Run audio connector only
    additive_mask = convert_to_additive_mask(attention_mask, audio_feats.dtype)
    audio_encoded, _ = emb_proc.audio_connector(audio_feats, additive_mask)
    del audio_feats, additive_mask

    # Dummy video encoding (zeros) — sampler needs it but audio-only model ignores it
    vc = torch.zeros(1, audio_encoded.shape[1], 4096, device=audio_encoded.device, dtype=audio_encoded.dtype)
    ac = audio_encoded

    return vc, ac


def _free_vram():
    """Free all GPU memory before loading a large model."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def _build_embeddings_processor(gemma_path, pipeline_path):
    """Build the embeddings processor and tokenizer on CPU."""
    # DistilledPipeline needs a gemma_root with tokenizer.model.
    # Pre-quantized repos may not have it, resolve to full gemma for tokenizer.
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


def _encode_gemma(compiled_prompt, gemma_path, pipeline_path, load_kwargs):
    """Core encoding: load Gemma, encode, run embeddings processor on CPU.

    This is the shared implementation for all quantization modes.
    The only difference is how Gemma is loaded (load_kwargs).
    """
    _free_vram()

    # Step 1: Build embeddings processor on CPU
    emb_proc, tokenizer = _build_embeddings_processor(gemma_path, pipeline_path)

    # Step 2: Load Gemma
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, **load_kwargs
    ).eval()

    _vram = torch.cuda.memory_allocated() / 1e9
    _peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM after Gemma load: %.2fGB (peak %.2fGB)", _vram, _peak)

    # Step 3: Encode text via Gemma
    with torch.inference_mode():
        tp = tokenizer.tokenize_with_weights(compiled_prompt)["gemma"]
        ids = torch.tensor([[t[0] for t in tp]], device="cuda")
        mask = torch.tensor([[w[1] for w in tp]], device="cuda")
        out = gemma_model.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True,
        )
        # Move hidden states to CPU for embeddings processor
        hs_cpu = tuple(h.cpu() for h in out.hidden_states)
        mask_cpu = mask.cpu()

        # Free Gemma
        del gemma_model, out, ids, mask
        gc.collect()
        torch.cuda.empty_cache()

        # Run audio-only embeddings on GPU.
        # Strip video components (5GB) and run only audio path (~1.3GB) on GPU.
        _strip_video_components(emb_proc)
        emb_proc = emb_proc.cuda()
        hs_gpu = tuple(h.cuda() for h in hs_cpu)
        mask_gpu = mask_cpu.cuda()
        del hs_cpu, mask_cpu

        vc, ac = _audio_only_embeddings(emb_proc, hs_gpu, mask_gpu)
        del hs_gpu, mask_gpu

    del emb_proc, tokenizer, emb
    gc.collect()
    torch.cuda.empty_cache()

    return vc, ac


def _get_default_gemma(quantize):
    """Get the default Gemma model path for a quantization mode."""
    if quantize == "nf4":
        return DEFAULT_GEMMA_NF4
    return DEFAULT_GEMMA_BF16


class ScenemaAudioTextEncode:
    """Encodes text prompts via Gemma 3 12B for the audio diffusion model.

    Quantization modes:
    - nf4: Pre-quantized NF4 on GPU. Fast. Needs 12GB+ VRAM.
    - cpu: bf16 split across CPU/GPU. Slower but works on 8GB cards.
    - bf16: Full precision on GPU. Best quality. Needs 40GB+ VRAM.
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
                "quantize": (["auto", "nf4", "cpu", "bf16"], {"default": "auto"}),
            },
        }

    def encode(self, compiled_prompt, model, gemma_path="auto", quantize="auto"):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Auto-detect best mode based on VRAM
        if quantize == "auto":
            if vram_gb >= 40:
                quantize = "bf16"
            else:
                quantize = "cpu"
            logger.info("VRAM %.0fGB: auto-selected quantize=%s", vram_gb, quantize)

        # Resolve default gemma path based on mode
        if gemma_path == "auto":
            gemma_path = _get_default_gemma(quantize)

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_AUDIO_CKPT)

        logger.info("Encoding prompt with Gemma (%s, %s)...", quantize, gemma_path)

        # NF4 pre-quantized must load entirely on GPU (bnb can't split CPU/GPU).
        # bf16/cpu mode caps at 6GB to match transformer peak (~5.5GB).
        gemma_gpu_cap = "6GiB"

        if quantize == "nf4":
            # Pre-quantized NF4 — needs full GPU (~8GB), can't split
            load_kwargs = {
                "device_map": "auto",
                "max_memory": {0: f"{int(vram_gb - 2)}GiB", "cpu": "32GiB"},
                "dtype": torch.bfloat16,
            }
            vc, ac = _encode_gemma(compiled_prompt, gemma_local, pipeline_path, load_kwargs)

        elif quantize == "cpu":
            # bf16 split across CPU and GPU — fits 8GB cards
            load_kwargs = {
                "device_map": "auto",
                "max_memory": {0: gemma_gpu_cap, "cpu": "32GiB"},
                "dtype": torch.bfloat16,
            }
            vc, ac = _encode_gemma(compiled_prompt, gemma_local, pipeline_path, load_kwargs)

        else:
            # bf16 all on GPU
            load_kwargs = {
                "device_map": "cuda",
                "dtype": torch.bfloat16,
            }
            vc, ac = _encode_gemma(compiled_prompt, gemma_local, pipeline_path, load_kwargs)

        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
