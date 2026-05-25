# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding node for ComfyUI.

Encodes compiled text prompts via Gemma 3 12B. Supports NF4 quantization
for low-VRAM cards and bf16 for high-VRAM cards.
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

from .utils import download_model, PIPELINE_CKPT

logger = logging.getLogger(__name__)

DEFAULT_GEMMA = "unsloth/gemma-3-12b-it-bnb-4bit"


def _resolve_gemma_path(gemma_path):
    """Resolve Gemma path to a local directory."""
    if os.path.isdir(gemma_path):
        return gemma_path
    return snapshot_download(gemma_path)


def _is_pre_quantized(gemma_path):
    """Check if the Gemma model is already pre-quantized."""
    return "bnb-4bit" in gemma_path or "bnb_4bit" in gemma_path


def _free_vram():
    """Free all GPU memory before loading a large model."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def _build_embeddings_processor(gemma_path, pipeline_path):
    """Build the embeddings processor and tokenizer on CPU.

    Extracts just the text projection weights from the pipeline checkpoint
    without loading the full model to GPU. Returns the processor and
    tokenizer, then frees the pipeline.
    """
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=OffloadMode.CPU,
    )
    pe = pipeline.prompt_encoder
    emb_proc = pe._embeddings_processor_builder.build(
        device="cpu", dtype=torch.bfloat16
    ).eval()
    tokenizer = LTXVGemmaTokenizer(gemma_path)

    del pipeline, pe
    gc.collect()

    return emb_proc, tokenizer


def _encode_nf4(compiled_prompt, gemma_path, pipeline_path):
    """Encode prompt using NF4-quantized Gemma (~8 GB VRAM).

    Loads the embeddings processor on CPU first, then loads Gemma to GPU,
    runs inference, and frees everything. This ensures the pipeline
    checkpoint and Gemma are never on GPU simultaneously.
    """
    _free_vram()

    # Step 1: Build embeddings processor on CPU (no GPU needed)
    emb_proc, tokenizer = _build_embeddings_processor(gemma_path, pipeline_path)

    # Step 2: Load Gemma to GPU with memory limit
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    max_gpu_mem = f"{int(vram_gb - 2)}GiB"

    load_kwargs = {
        "device_map": "auto",
        "max_memory": {0: max_gpu_mem, "cpu": "32GiB"},
        "dtype": torch.bfloat16,
    }
    if not _is_pre_quantized(gemma_path):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, **load_kwargs
    ).eval()

    _vram = torch.cuda.memory_allocated() / 1e9
    _peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM after Gemma load: %.2fGB (peak %.2fGB)", _vram, _peak)

    # Step 3: Encode — run Gemma on GPU, embeddings processor on CPU
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

        # Free Gemma from GPU before running embeddings processor
        del gemma_model, out, ids, mask
        gc.collect()
        torch.cuda.empty_cache()

        _vram = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after Gemma freed: %.2fGB", _vram)

        # Run embeddings processor on GPU (now free)
        emb_proc = emb_proc.cuda()
        _vram = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after emb_proc to GPU: %.2fGB", _vram)
        hs_gpu = tuple(h.cuda() for h in hs_cpu)
        mask_gpu = mask_cpu.cuda()
        del hs_cpu, mask_cpu

        emb = emb_proc.process_hidden_states(hs_gpu, mask_gpu)
        vc = emb.video_encoding
        ac = emb.audio_encoding

    # Step 4: Free everything
    del emb_proc, tokenizer, emb, hs_gpu, mask_gpu
    gc.collect()
    torch.cuda.empty_cache()

    return vc, ac


def _encode_bf16(compiled_prompt, gemma_path, pipeline_path):
    """Encode prompt using bf16 Gemma via pipeline prompt encoder."""
    _free_vram()

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=OffloadMode.CPU,
    )

    with torch.inference_mode():
        (emb,) = pipeline.prompt_encoder([compiled_prompt])
        vc = emb.video_encoding
        ac = emb.audio_encoding

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return vc, ac


class ScenemaAudioTextEncode:
    """Encodes text prompts via Gemma 3 12B for the audio diffusion model.

    Supports two quantization modes:
    - nf4: BitsAndBytes 4-bit (~8 GB VRAM, fast)
    - bf16: Full precision via pipeline prompt encoder
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
                "gemma_path": ("STRING", {"default": DEFAULT_GEMMA}),
                "quantize": (["nf4", "bf16"], {"default": "nf4"}),
            },
        }

    def encode(self, compiled_prompt, model, gemma_path=DEFAULT_GEMMA, quantize="nf4"):
        logger.info("Encoding prompt with Gemma (%s)...", quantize)
        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_CKPT)

        if quantize == "nf4":
            vc, ac = _encode_nf4(compiled_prompt, gemma_local, pipeline_path)
        else:
            vc, ac = _encode_bf16(compiled_prompt, gemma_local, pipeline_path)

        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
