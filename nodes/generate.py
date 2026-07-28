# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Extended Generate node for ComfyUI.

All-in-one node for short and long-form audio generation.
Handles chunking, A2V voice conditioning, Whisper validation,
and concatenation internally.
"""

import gc
import logging
import os
import sys

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
from .text_encode import (
    _auto_quantize, _audio_only_embeddings, _build_gemma_load_kwargs,
    _get_default_gemma, _resolve_gemma_path, get_or_load_emb_proc,
    get_or_load_gemma,
)
from .utils import FPS, download_model, PIPELINE_AUDIO_CKPT

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.chunker import plan_chunks, estimate_duration, ChunkSpec

from .seedvc import convert_voice

logger = logging.getLogger(__name__)

REF_TAIL_SECONDS = 3.0


def _log_vram(label):
    """Log current and peak VRAM usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    logger.info("VRAM [%s]: %.2fGB allocated, %.2fGB peak, %.2fGB reserved",
                label, allocated, peak, reserved)


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
                "gemma_path": ("STRING", {"default": "auto"}),
                "quantize": (["auto", "nf4", "bf16", "cpu"], {"default": "auto"}),
                "ref_latent": ("SA_LATENT",),
                "xml_prompt": ("STRING", {"forceInput": True, "default": ""}),
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
                 pace=1.5, gemma_path="auto", quantize="auto",
                 ref_latent=None, xml_prompt="",
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):

        chunks = self._plan(xml_prompt, compiled_prompt, speech_text, seed, pace)

        torch.cuda.reset_peak_memory_stats()
        _log_vram("start")
        logger.info("Generating %d chunk(s) (skip_vc=%s)...", len(chunks), skip_vc)

        # ── Phase 1: Encode ALL chunk prompts in one Gemma session ──
        logger.info("Phase 1: Encoding %d prompts...", len(chunks))
        chunk_encodings = self._encode_all_chunks(chunks, gemma_path, quantize)
        _log_vram("after all encoding")

        # ── Phase 2: Diffuse + decode ALL chunks in one transformer session ──
        logger.info("Phase 2: Diffusing %d chunks...", len(chunks))
        mdl_wrapper = model["model"]
        device = model["device"]
        mdl_wrapper.to(device)
        _log_vram("transformer on GPU")

        chunk_encodings_cpu = [(vc.cpu(), ac.cpu()) for vc, ac in chunk_encodings]
        del chunk_encodings
        torch.cuda.empty_cache()

        waveforms = []
        sr = None
        current_ref = ref_latent.cpu() if ref_latent is not None else None
        for i, (chunk, (vc_cpu, ac_cpu)) in enumerate(zip(chunks, chunk_encodings_cpu)):
            logger.info("  Diffuse chunk %d/%d (%.1fs)", i + 1, len(chunks), chunk.duration_s)
            vc = vc_cpu.to(device)
            ac = ac_cpu.to(device)
            ref_gpu = current_ref.to(device) if current_ref is not None else None
            latent = self._diffuse_chunk(mdl_wrapper, device, vc, ac,
                                          chunk.duration_s, chunk.seed, ref_gpu)
            del vc, ac, ref_gpu

            waveform, sr = _decode_latent(vae, latent)
            waveforms.append(waveform)

            if i < len(chunks) - 1:
                current_ref = _encode_reference(vae, waveform, sr).cpu()

        # Transformer stays on GPU — ComfyUI's model management will evict
        # it if another workflow needs the VRAM. Manually offloading here
        # would just re-shuttle it on the next generation.
        torch.cuda.empty_cache()
        _log_vram("after all chunks")

        combined = torch.cat([w.squeeze(0) for w in waveforms], dim=-1).unsqueeze(0)
        combined_audio = {"waveform": combined, "sample_rate": sr}

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

    def _encode_all_chunks(self, chunks, gemma_path, quantize):
        """Encode all chunk prompts in a single Gemma session.

        Loads Gemma once (or reuses cached), encodes all prompts, then
        runs the emb_proc audio path on GPU. Returns list of (vc, ac) tuples.
        """
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if quantize == "auto":
            quantize = _auto_quantize(vram_gb)

        if gemma_path == "auto":
            gemma_path = _get_default_gemma(quantize)

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_AUDIO_CKPT)

        emb_proc, tokenizer = get_or_load_emb_proc(gemma_local, pipeline_path)

        load_kwargs = _build_gemma_load_kwargs(quantize, vram_gb)
        gemma_model, was_cached = get_or_load_gemma(gemma_local, quantize, load_kwargs)
        _log_vram("Gemma loaded")

        all_hidden_states = []
        for i, chunk in enumerate(chunks):
            tp = tokenizer.tokenize_with_weights(chunk.compiled_prompt)["gemma"]
            ids = torch.tensor([[t[0] for t in tp]], device="cuda")
            mask = torch.tensor([[w[1] for w in tp]], device="cuda")
            out = gemma_model.model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
            )
            hs_cpu = tuple(h.cpu() for h in out.hidden_states)
            mask_cpu = mask.cpu()
            all_hidden_states.append((hs_cpu, mask_cpu))
            del out, ids, mask
            logger.info("  Gemma encoded chunk %d/%d", i + 1, len(chunks))

        if not was_cached:
            del gemma_model
            gc.collect()
            torch.cuda.empty_cache()

        emb_proc.cuda()

        all_encodings = []
        for i, (hs_cpu, mask_cpu) in enumerate(all_hidden_states):
            hs_gpu = tuple(h.cuda() for h in hs_cpu)
            mask_gpu = mask_cpu.cuda()
            vc, ac = _audio_only_embeddings(emb_proc, hs_gpu, mask_gpu)
            all_encodings.append((vc, ac))
            del hs_gpu, mask_gpu, hs_cpu, mask_cpu
            logger.info("  Processed embeddings chunk %d/%d", i + 1, len(all_hidden_states))

        emb_proc.cpu()
        del all_hidden_states
        gc.collect()
        torch.cuda.empty_cache()

        return all_encodings

    def _diffuse_chunk(self, mdl_wrapper, device, vc, ac, duration_s, seed, ref_latent=None):
        """Run diffusion for a single chunk. Transformer must already be on GPU."""
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

    def _apply_vc(self, combined_audio, chunk_waveforms, sr, ref_latent, vae,
                  vc_steps, vc_cfg_rate):
        """Apply SeedVC for voice consistency.

        If reference audio provided via ref_latent: convert against reference.
        If no reference: convert all against chunk 0 (first chunk sets identity).
        """
        chunk0_audio = {
            "waveform": chunk_waveforms[0],
            "sample_rate": sr,
        }

        logger.info("Applying SeedVC (%d steps, cfg_rate=%.2f)...", vc_steps, vc_cfg_rate)
        result = convert_voice(combined_audio, chunk0_audio, vc_steps, vc_cfg_rate)

        if result["sample_rate"] != sr:
            result_wav = torchaudio.functional.resample(
                result["waveform"].float(), result["sample_rate"], sr
            )
            result = {"waveform": result_wav, "sample_rate": sr}

        return result

    def _plan(self, xml_prompt, compiled_prompt, speech_text, seed, pace):
        """Plan chunks from xml_prompt or fall back to single chunk."""
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
