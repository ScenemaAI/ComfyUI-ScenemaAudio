# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio text encoding node for ComfyUI.

Encodes compiled text prompts via Gemma 3 12B. Supports NF4 quantization
for low-VRAM cards and bf16 for high-VRAM cards.
"""

import gc
import logging

import torch
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import OffloadMode
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

from .utils import download_model, PIPELINE_CKPT

logger = logging.getLogger(__name__)

DEFAULT_GEMMA = "google/gemma-3-12b-it"


def _encode_nf4(compiled_prompt, gemma_path, pipeline_path):
    """Encode prompt using NF4-quantized Gemma (~8 GB VRAM)."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path,
        quantization_config=quant_config,
        device_map="cuda",
        dtype=torch.bfloat16,
    ).eval()

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=pipeline_path,
        gemma_root=gemma_path,
        spatial_upsampler_path=None,
        loras=[],
        offload_mode=OffloadMode.CPU,
    )
    pe = pipeline.prompt_encoder
    emb_proc = pe._embeddings_processor_builder.build(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    tokenizer = LTXVGemmaTokenizer(gemma_path)

    with torch.inference_mode():
        tp = tokenizer.tokenize_with_weights(compiled_prompt)["gemma"]
        ids = torch.tensor([[t[0] for t in tp]], device="cuda")
        mask = torch.tensor([[w[1] for w in tp]], device="cuda")
        out = gemma_model.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True,
        )
        emb = emb_proc.process_hidden_states(out.hidden_states, mask)
        vc = emb.video_encoding
        ac = emb.audio_encoding

    del gemma_model, emb_proc, tokenizer, pipeline, out, emb
    gc.collect()
    torch.cuda.empty_cache()
    return vc, ac


def _encode_bf16(compiled_prompt, gemma_path, pipeline_path):
    """Encode prompt using bf16 Gemma via pipeline prompt encoder."""
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
        pipeline_path = download_model(PIPELINE_CKPT)

        if quantize == "nf4":
            vc, ac = _encode_nf4(compiled_prompt, gemma_path, pipeline_path)
        else:
            vc, ac = _encode_bf16(compiled_prompt, gemma_path, pipeline_path)

        logger.info("Text encoding complete")
        return ({"video_context": vc, "audio_context": ac},)
