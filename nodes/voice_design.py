# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Voice Design node for ComfyUI.

Generates a 15-second voice preview from a voice description.
Output can be previewed directly or fed into VAE Encode as a
reference for voice cloning in Extended Generate.
"""

import logging
import os
import sys

import torch
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_pipelines.distilled import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop

from .sampler import (
    _build_pixel_shape, _build_video_state, _build_audio_state,
)
from .text_encode import _encode_nf4, _encode_bf16, _resolve_gemma_path
from .utils import download_model, PIPELINE_CKPT

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.compiler import compile_prompt

logger = logging.getLogger(__name__)

VOICE_DESIGN_DURATION_S = 15.0
VOICE_DESIGN_TEXT = "The old lighthouse had stood on the cliff for over a century, its beam cutting through the fog like a blade of light."


class ScenemaAudioVoiceDesign:
    """Generates a 15-second voice preview.

    Quick way to audition a voice description before committing to
    a full generation. Output audio can be:
    - Previewed directly via Preview Audio
    - Fed into VAE Encode → Extended Generate ref_latent for voice cloning
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "design"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SA_MODEL",),
                "vae": ("SA_VAE",),
                "voice": ("STRING", {
                    "multiline": True,
                    "default": "Female, mid 30s. Warm alto. Clear British accent. Confident and articulate.",
                }),
                "gender": (["male", "female"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "preview_text": ("STRING", {
                    "multiline": True,
                    "default": VOICE_DESIGN_TEXT,
                }),
                "scene": ("STRING", {"default": ""}),
                "gemma_path": ("STRING", {"default": "google/gemma-3-12b-it"}),
                "quantize": (["nf4", "bf16"], {"default": "nf4"}),
            },
        }

    @torch.inference_mode()
    def design(self, model, vae, voice, gender, seed,
               preview_text=VOICE_DESIGN_TEXT, scene="",
               gemma_path="google/gemma-3-12b-it", quantize="nf4"):

        # Build XML and compile
        attrs = f'voice="{voice}" gender="{gender}"'
        if scene and scene.strip():
            attrs += f' scene="{scene.strip()}"'
        xml = f"<speak {attrs}>{preview_text.strip()}</speak>"
        result = compile_prompt(xml)

        # Encode text
        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_CKPT)
        if quantize == "nf4":
            vc, ac = _encode_nf4(result.prompt, gemma_local, pipeline_path)
        else:
            vc, ac = _encode_bf16(result.prompt, gemma_local, pipeline_path)

        # Sample single chunk at fixed 15s
        mdl_wrapper = model["model"]
        device = model["device"]

        pixel_shape = _build_pixel_shape(VOICE_DESIGN_DURATION_S)
        gen = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)

        video_state = _build_video_state(pixel_shape, vc, noiser, device)
        audio_state, audio_tools = _build_audio_state(pixel_shape, ac, noiser, device)

        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        stepper = EulerDiffusionStep()
        wrapped = BatchSplitAdapter(mdl_wrapper, max_batch_size=1)

        logger.info("Voice design: generating 15s preview...")
        _, audio_state_out = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=wrapped,
            denoiser=SimpleDenoiser(vc, ac),
        )

        audio_state_out = audio_tools.clear_conditioning(audio_state_out)
        audio_state_out = audio_tools.unpatchify(audio_state_out)

        # Decode
        decoder = vae["decoder"]
        audio_obj = decoder(audio_state_out.latent.cuda())
        waveform = audio_obj.waveform.cpu()
        sr = audio_obj.sampling_rate
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        logger.info("Voice design complete: %.1fs", waveform.shape[-1] / sr)
        return ({"waveform": waveform, "sample_rate": sr},)
