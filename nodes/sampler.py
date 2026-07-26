# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio sampler node for ComfyUI.

Runs 8-step distilled diffusion on the audio-only transformer with
optional A2V reference conditioning for voice identity.
"""

import logging
from dataclasses import replace as dc_replace

import torch
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.tools import AudioLatentTools, LatentState, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DISTILLED_SIGMAS
from ltx_pipelines.utils.blocks import ModalitySpec, _build_state
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop

from .utils import FPS

logger = logging.getLogger(__name__)


def _build_pixel_shape(duration_s):
    """Compute the LTX pixel shape from target duration."""
    num_frames = ((int(duration_s * FPS) + 7) // 8) * 8 + 1
    return VideoPixelShape(batch=1, frames=num_frames, width=64, height=64, fps=FPS)


def _build_video_state(pixel_shape, vc, noiser, device):
    """Build the video latent state (dummy, required by the joint model)."""
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools = VideoLatentTools(
        VideoLatentPatchifier(patch_size=1), v_shape, fps=FPS
    )
    return _build_state(
        ModalitySpec(context=vc, conditionings=[]),
        video_tools, noiser, torch.bfloat16, device,
    )


def _build_audio_state(pixel_shape, ac, noiser, device):
    """Build the audio latent state for denoising."""
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    audio_state = _build_state(
        ModalitySpec(context=ac),
        audio_tools, noiser, torch.bfloat16, device,
    )
    return audio_state, audio_tools


def _apply_a2v_reference(audio_state, ac, ref_latent, seed, device):
    """Prepend A2V reference latent to audio state for voice conditioning.

    Returns the modified audio state with reference frames prepended
    and the number of reference frames added.
    """
    ref = ref_latent.to(device=device, dtype=torch.bfloat16)
    ref_frames = ref.shape[2]
    total_t = ref_frames + audio_state.latent.shape[1]

    ref_patchified = ref.permute(0, 2, 1, 3).reshape(1, ref_frames, -1)
    combined_latent = torch.cat([ref_patchified, audio_state.latent], dim=1)

    ref_mask = torch.zeros(
        1, ref_frames, 1, device=device, dtype=audio_state.denoise_mask.dtype
    )
    combined_mask = torch.cat([ref_mask, audio_state.denoise_mask], dim=1)
    combined_clean = torch.cat(
        [ref_patchified, torch.zeros_like(audio_state.clean_latent)], dim=1
    )

    combined_a_shape = AudioLatentShape(
        batch=1, channels=8, frames=total_t, mel_bins=16
    )
    combined_audio_tools = AudioLatentTools(
        AudioPatchifier(patch_size=1), combined_a_shape
    )
    gen = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=gen)
    tmp_state = _build_state(
        ModalitySpec(context=ac),
        combined_audio_tools, noiser, torch.bfloat16, device,
    )
    combined_positions = tmp_state.positions
    del tmp_state

    combined_state = LatentState(
        latent=combined_latent,
        denoise_mask=combined_mask,
        positions=combined_positions,
        clean_latent=combined_clean,
        attention_mask=None,
    )
    return combined_state, ref_frames


def _strip_reference_frames(audio_state_out, ref_frames):
    """Remove prepended reference frames from the denoised output."""
    return dc_replace(
        audio_state_out,
        latent=audio_state_out.latent[:, ref_frames:],
        denoise_mask=audio_state_out.denoise_mask[:, ref_frames:],
        positions=audio_state_out.positions[:, :, ref_frames:],
        clean_latent=(
            audio_state_out.clean_latent[:, ref_frames:]
            if audio_state_out.clean_latent is not None
            else None
        ),
    )


class ScenemaAudioSampler:
    """Runs the audio diffusion sampling loop.

    8-step distilled Euler denoising with optional A2V reference
    latent for voice conditioning.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "sample"
    RETURN_TYPES = ("SA_LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SA_MODEL",),
                "conditioning": ("SA_CONDITIONING",),
                "duration_s": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ref_latent": ("SA_LATENT",),
            },
        }

    @torch.inference_mode()
    def sample(self, model, conditioning, duration_s, seed, ref_latent=None):
        mdl_wrapper = model["model"]
        device = model["device"]
        mdl_wrapper.to(device)
        vc = conditioning["video_context"]
        ac = conditioning["audio_context"]

        pixel_shape = _build_pixel_shape(duration_s)

        gen = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)

        video_state = _build_video_state(pixel_shape, vc, noiser, device)
        audio_state, audio_tools = _build_audio_state(pixel_shape, ac, noiser, device)

        ref_frames = 0
        if ref_latent is not None:
            audio_state, ref_frames = _apply_a2v_reference(
                audio_state, ac, ref_latent, seed, device
            )

        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        stepper = EulerDiffusionStep()
        wrapped = BatchSplitAdapter(mdl_wrapper, max_batch_size=1)

        logger.info("Denoising %.1fs (%d frames)...", duration_s, pixel_shape.frames)
        _, audio_state_out = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=wrapped,
            denoiser=SimpleDenoiser(vc, ac),
        )

        if ref_frames > 0 and audio_state_out is not None:
            audio_state_out = _strip_reference_frames(audio_state_out, ref_frames)

        audio_state_out = audio_tools.clear_conditioning(audio_state_out)
        audio_state_out = audio_tools.unpatchify(audio_state_out)

        if torch.isnan(audio_state_out.latent).any():
            logger.warning("NaN detected in denoised latent")

        # Transformer stays on GPU. ComfyUI will evict via unload_all_models
        # if another workflow needs the VRAM. Manually offloading here just
        # forces a re-shuttle on the next sample call.
        torch.cuda.empty_cache()

        logger.info("Sampling complete")
        return (audio_state_out.latent,)
