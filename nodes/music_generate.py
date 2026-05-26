# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Music Generate node for ComfyUI.

Generates music, SFX, and ambient audio of any length.
Uses A2V latent chaining for continuity across chunks.
No Kokoro, no SeedVC, no Whisper validation.
"""

import logging
import math
import os
import sys

import torch
import torchaudio
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio

from .sampler import (
    _build_pixel_shape, _build_video_state, _build_audio_state,
    _apply_a2v_reference, _strip_reference_frames,
)
from .text_encode import _encode_gemma, _resolve_gemma_path, _get_default_gemma
from .utils import FPS, download_model, PIPELINE_AUDIO_CKPT

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_pipelines.distilled import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop

logger = logging.getLogger(__name__)

MAX_CHUNK_DURATION = 15.0
REF_TAIL_SECONDS = 3.0


def _compile_music_prompt(description, scene):
    """Compile a music/SFX prompt in scene mode."""
    parts = []
    if scene and scene.strip():
        parts.append(f"{scene.strip()}.")
    parts.append(description.strip())
    return " ".join(parts)


def _sample(model_data, vc, ac, duration_s, seed, ref_latent=None):
    """Run diffusion sampling for a single chunk."""
    mdl_wrapper = model_data["model"]
    device = model_data["device"]
    mdl_wrapper.to(device)

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

    mdl_wrapper.to("cpu")
    torch.cuda.empty_cache()

    return audio_state_out.latent


def _decode(vae_data, latent):
    """Decode audio latent to waveform."""
    decoder = vae_data["decoder"]
    audio_obj = decoder(latent.cuda())
    waveform = audio_obj.waveform.cpu()
    sr = audio_obj.sampling_rate
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sr


def _encode_tail(vae_data, waveform, sr):
    """Encode tail of waveform as A2V reference for next chunk."""
    encoder = vae_data["encoder"]
    vae_sr = vae_data["sample_rate"]

    tail_samples = int(REF_TAIL_SECONDS * sr)
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


class ScenemaAudioMusicGenerate:
    """Generates music, SFX, and ambient audio of any length.

    For durations under 15s, runs a single generation pass.
    For longer durations, splits into chunks with A2V latent
    chaining for continuity (same key, tempo, instrumentation).
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
                "description": ("STRING", {
                    "multiline": True,
                    "default": "Acoustic folk guitar, light cheerful strumming, warm and sunny",
                }),
                "duration_s": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 300.0, "step": 1.0,
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "scene": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
                "gemma_path": ("STRING", {"default": "auto"}),
                "quantize": (["auto", "nf4", "cpu", "bf16"], {"default": "auto"}),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, description, duration_s, seed,
                 scene="", gemma_path="auto", quantize="auto"):

        prompt = _compile_music_prompt(description, scene)
        num_chunks = max(1, math.ceil(duration_s / MAX_CHUNK_DURATION))
        chunk_duration = min(duration_s, MAX_CHUNK_DURATION)

        logger.info("Music generate: %.1fs total, %d chunk(s) of %.1fs",
                     duration_s, num_chunks, chunk_duration)

        # Auto-detect quantization
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if quantize == "auto":
            if vram_gb >= 40:
                quantize = "bf16"
            elif vram_gb >= 12:
                quantize = "nf4"
            else:
                quantize = "cpu"

        if gemma_path == "auto":
            gemma_path = _get_default_gemma(quantize)

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_AUDIO_CKPT)

        if quantize == "nf4":
            load_kwargs = {"device_map": "auto", "max_memory": {0: f"{int(vram_gb - 2)}GiB", "cpu": "32GiB"}, "dtype": torch.bfloat16}
        elif quantize == "cpu":
            load_kwargs = {"device_map": "auto", "max_memory": {0: f"{int(vram_gb - 2)}GiB", "cpu": "32GiB"}, "dtype": torch.bfloat16}
        else:
            load_kwargs = {"device_map": "cuda", "dtype": torch.bfloat16}

        vc, ac = _encode_gemma(prompt, gemma_local, pipeline_path, load_kwargs)

        waveforms = []
        ref_latent = None
        sr = None

        for i in range(num_chunks):
            # Last chunk may be shorter
            this_duration = min(chunk_duration, duration_s - i * chunk_duration)
            this_seed = seed + i * 1000

            logger.info("Chunk %d/%d (%.1fs, seed=%d)", i + 1, num_chunks,
                         this_duration, this_seed)

            latent = _sample(model, vc, ac, this_duration, this_seed, ref_latent)
            waveform, sr = _decode(vae, latent)
            waveforms.append(waveform)

            if i < num_chunks - 1:
                ref_latent = _encode_tail(vae, waveform, sr)

        combined = torch.cat([w.squeeze(0) for w in waveforms], dim=-1).unsqueeze(0)
        total = combined.shape[-1] / sr
        logger.info("Music generate complete: %.1fs", total)

        return ({"waveform": combined, "sample_rate": sr},)
