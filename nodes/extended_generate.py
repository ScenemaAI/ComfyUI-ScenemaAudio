# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Extended Generate node for ComfyUI.

All-in-one node for short and long-form audio generation.
Handles chunking, A2V voice conditioning, Whisper validation,
and concatenation internally.
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
from audio_core.whisper_aligner import validate_text

from .seedvc import convert_voice

logger = logging.getLogger(__name__)

REF_TAIL_SECONDS = 3.0
MAX_RETRIES = 3
RETRY_DURATION_FACTOR = 1.3
MIN_WORD_MATCH_RATIO = 0.90


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


def _waveform_to_numpy(waveform, sr):
    """Convert ComfyUI waveform tensor to numpy for validation."""
    wav = waveform.squeeze(0).numpy()  # (channels, samples)
    if wav.ndim == 2:
        wav = wav.T  # (samples, channels)
    return wav


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


def _generate_chunk_with_validation(
    model, vae, chunk, gemma_path, quantize, ref_latent, validate, min_match_ratio,
):
    """Generate a single chunk with optional Whisper validation and retry."""
    vc, ac = _encode_text(model, chunk.compiled_prompt, gemma_path, quantize)

    duration = chunk.duration_s
    seed = chunk.seed
    best_waveform = None
    best_sr = None
    best_ratio = -1.0

    attempts = 1 if not validate else MAX_RETRIES + 1

    for attempt in range(attempts):
        latent = _sample_chunk(model, vc, ac, duration, seed, ref_latent)
        waveform, sr = _decode_latent(vae, latent)

        if not validate:
            return waveform, sr

        wav_np = _waveform_to_numpy(waveform, sr)
        passed, transcribed, ratio = validate_text(
            wav_np, sr, chunk.expected_text,
            language=chunk.language,
            min_word_ratio=min_match_ratio,
        )

        if ratio > best_ratio:
            best_waveform = waveform
            best_sr = sr
            best_ratio = ratio

        if passed:
            logger.info("  Validated: %.0f%% word match", ratio * 100)
            return waveform, sr

        if attempt < MAX_RETRIES:
            duration = min(duration * RETRY_DURATION_FACTOR, 20.0)
            seed += 1
            logger.info(
                "  Retry %d: %.0f%% match, extending to %.1fs, seed=%d",
                attempt + 1, ratio * 100, duration, seed,
            )

    logger.warning("  Best %.0f%% match after %d retries, accepting", best_ratio * 100, MAX_RETRIES)
    return best_waveform, best_sr


class ScenemaAudioExtendedGenerate:
    """Generates audio of any length with automatic chunking.

    For short text (under 15s), runs a single generation pass.
    For longer text, automatically splits at sentence boundaries using
    Kokoro duration estimation, generates each chunk with A2V voice
    conditioning from the previous chunk, and concatenates the results.

    Includes optional Whisper validation with retry for quality control.
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
                "xml_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "validate": ("BOOLEAN", {"default": False}),
                "min_match_ratio": ("FLOAT", {
                    "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "skip_vc": ("BOOLEAN", {"default": False}),
                "vc_steps": ("INT", {
                    "default": 25, "min": 5, "max": 50, "step": 5,
                }),
                "vc_cfg_rate": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, compiled_prompt, speech_text, seed,
                 pace=1.5, gemma_path="google/gemma-3-12b-it", quantize="nf4",
                 ref_latent=None, xml_prompt="", validate=False, min_match_ratio=0.90,
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):

        chunks = self._plan(xml_prompt, compiled_prompt, speech_text, seed, pace)

        logger.info("Generating %d chunk(s) (validate=%s, skip_vc=%s)...",
                     len(chunks), validate, skip_vc)

        waveforms = []
        current_ref = ref_latent
        sr = None

        for i, chunk in enumerate(chunks):
            logger.info(
                "Chunk %d/%d (%.1fs, seed=%d): %s",
                i + 1, len(chunks), chunk.duration_s, chunk.seed,
                chunk.expected_text[:60],
            )

            waveform, sr = _generate_chunk_with_validation(
                model, vae, chunk, gemma_path, quantize,
                current_ref, validate, min_match_ratio,
            )
            waveforms.append(waveform)

            if i < len(chunks) - 1:
                current_ref = _encode_reference(vae, waveform, sr)

        combined = torch.cat([w.squeeze(0) for w in waveforms], dim=-1).unsqueeze(0)
        combined_audio = {"waveform": combined, "sample_rate": sr}

        # SeedVC voice conversion for multi-chunk consistency or voice cloning
        needs_vc = ref_latent is not None or len(chunks) > 1
        if not skip_vc and needs_vc:
            combined_audio = self._apply_vc(
                combined_audio, waveforms, sr, ref_latent, vae,
                vc_steps, vc_cfg_rate,
            )

        total_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
        logger.info("Extended generate complete: %.1fs from %d chunk(s)",
                     total_duration, len(chunks))

        return (combined_audio,)

    def _apply_vc(self, combined_audio, chunk_waveforms, sr, ref_latent, vae,
                  vc_steps, vc_cfg_rate):
        """Apply SeedVC for voice consistency.

        If reference audio provided via ref_latent: convert against reference.
        If no reference: convert all against chunk 0 (first chunk sets identity).
        Same logic as production processor._apply_seedvc.
        """
        # Build reference audio for SeedVC
        # If ref_latent was provided, we don't have the raw reference audio here.
        # Use chunk 0 as the voice identity anchor (same as production without ref).
        chunk0_audio = {
            "waveform": chunk_waveforms[0],
            "sample_rate": sr,
        }

        logger.info("Applying SeedVC (%d steps, cfg_rate=%.2f)...", vc_steps, vc_cfg_rate)
        result = convert_voice(combined_audio, chunk0_audio, vc_steps, vc_cfg_rate)

        # Resample back to original SR if SeedVC changed it
        if result["sample_rate"] != sr:
            result_wav = torchaudio.functional.resample(
                result["waveform"].float(), result["sample_rate"], sr
            )
            result = {"waveform": result_wav, "sample_rate": sr}

        return result

    def _plan(self, xml_prompt, compiled_prompt, speech_text, seed, pace):
        """Plan chunks from xml_prompt or fall back to single chunk.

        When xml_prompt is provided and contains enough text, uses the
        full chunker with Kokoro duration estimation and action tag mapping.
        Otherwise falls back to a single chunk from compiled_prompt.
        """
        if xml_prompt and xml_prompt.strip():
            try:
                chunks = plan_chunks(xml_prompt.strip(), base_seed=seed, pace=pace)
                if chunks:
                    return chunks
            except Exception as e:
                logger.warning("Chunking failed, falling back to single chunk: %s", e)

        duration = estimate_duration(speech_text, multiplier=pace)
        return [ChunkSpec(
            compiled_prompt=compiled_prompt,
            duration_s=duration,
            seed=seed,
            expected_text=speech_text,
        )]
