# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Extended Generate node for ComfyUI.

Handles long-form audio generation with automatic chunking,
A2V voice conditioning between chunks, and concatenation.
"""

import logging
import os
import sys

import numpy as np
import torch
import torchaudio
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio
from ltx_pipelines.distilled import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop

from .sampler import (
    _build_pixel_shape, _build_video_state, _build_audio_state,
    _apply_a2v_reference, _strip_reference_frames,
)
from .text_encode import _encode_nf4, _encode_bf16, _resolve_gemma_path
from .utils import FPS, download_model, PIPELINE_CKPT

# Ensure audio_core is importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.chunker import plan_chunks, estimate_duration, ChunkSpec
from audio_core.compiler import compile_prompt

logger = logging.getLogger(__name__)

REF_TAIL_SECONDS = 3.0


def _encode_text(model_data, compiled_prompt, gemma_path, quantize):
    """Encode a single chunk's prompt via Gemma."""
    gemma_local = _resolve_gemma_path(gemma_path)
    pipeline_path = download_model(PIPELINE_CKPT)

    if quantize == "nf4":
        return _encode_nf4(compiled_prompt, gemma_local, pipeline_path)
    else:
        return _encode_bf16(compiled_prompt, gemma_local, pipeline_path)


def _sample_chunk(model_data, vc, ac, duration_s, seed, ref_latent=None):
    """Run diffusion sampling for a single chunk."""
    mdl_wrapper = model_data["model"]
    device = model_data["device"]

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

    return audio_state_out.latent


def _decode_latent(vae_data, latent):
    """Decode audio latent to waveform."""
    decoder = vae_data["decoder"]
    audio_obj = decoder(latent.cuda())
    waveform = audio_obj.waveform.cpu()
    sr = audio_obj.sampling_rate
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sr


def _encode_reference(vae_data, waveform, sr, max_seconds=REF_TAIL_SECONDS):
    """Encode tail of waveform as A2V reference for next chunk."""
    encoder = vae_data["encoder"]
    vae_sr = vae_data["sample_rate"]

    tail_samples = int(max_seconds * sr)
    wav = waveform[0, :, -tail_samples:]

    if sr != vae_sr:
        wav = torchaudio.functional.resample(wav.float(), sr, vae_sr)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    encoder_was_cpu = str(next(encoder.parameters()).device) == "cpu"
    if encoder_was_cpu:
        encoder.cuda()

    audio_obj = Audio(waveform=wav.unsqueeze(0).cuda(), sampling_rate=vae_sr)
    latent = encode_audio(audio_obj, encoder)

    if encoder_was_cpu:
        encoder.cpu()

    return latent


class ScenemaAudioExtendedGenerate:
    """Generates audio of any length with automatic chunking.

    For short text (under 15s), runs a single generation pass.
    For longer text, automatically splits at sentence boundaries using
    Kokoro duration estimation, generates each chunk with A2V voice
    conditioning from the previous chunk, and concatenates the results.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SA_MODEL",),
                "vae": ("SA_VAE",),
                "compiled_prompt": ("STRING", {"forceInput": True}),
                "speech_text": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "pace": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                }),
                "gemma_path": ("STRING", {"default": "google/gemma-3-12b-it"}),
                "quantize": (["nf4", "bf16"], {"default": "nf4"}),
                "ref_latent": ("SA_LATENT",),
                "xml_prompt": ("STRING", {"default": ""}),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, compiled_prompt, speech_text, seed,
                 pace=1.5, gemma_path="google/gemma-3-12b-it", quantize="nf4",
                 ref_latent=None, xml_prompt=""):

        chunks = self._plan(xml_prompt, compiled_prompt, speech_text, seed, pace)

        logger.info("Generating %d chunk(s)...", len(chunks))

        waveforms = []
        current_ref = ref_latent
        sr = None

        for i, chunk in enumerate(chunks):
            logger.info(
                "Chunk %d/%d (%.1fs, seed=%d): %s",
                i + 1, len(chunks), chunk.duration_s, chunk.seed,
                chunk.expected_text[:60],
            )

            vc, ac = _encode_text(model, chunk.compiled_prompt, gemma_path, quantize)
            latent = _sample_chunk(model, vc, ac, chunk.duration_s, chunk.seed, current_ref)
            waveform, sr = _decode_latent(vae, latent)
            waveforms.append(waveform)

            if i < len(chunks) - 1:
                current_ref = _encode_reference(vae, waveform, sr)

        combined = torch.cat([w.squeeze(0) for w in waveforms], dim=-1).unsqueeze(0)
        total_duration = combined.shape[-1] / sr

        logger.info("Extended generate complete: %.1fs from %d chunk(s)",
                     total_duration, len(chunks))

        return ({"waveform": combined, "sample_rate": sr},)

    def _plan(self, xml_prompt, compiled_prompt, speech_text, seed, pace):
        """Plan chunks from xml_prompt or fall back to single chunk."""
        if xml_prompt and xml_prompt.strip():
            return plan_chunks(xml_prompt.strip(), base_seed=seed, pace=pace)

        duration = estimate_duration(speech_text, multiplier=pace)
        return [ChunkSpec(
            compiled_prompt=compiled_prompt,
            duration_s=duration,
            seed=seed,
            expected_text=speech_text,
        )]
