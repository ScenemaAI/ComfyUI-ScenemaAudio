# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio Generate — the main speech generation node.

Takes voice description, gender, speech text, and scene directly. Builds
the XML the LTX model expects internally, plans chunks via Kokoro, and
diffuses each chunk with A2V voice chaining. Auto-cleans background bleed
when the chosen scene implies studio-clean intent.
"""

import gc
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
from .text_encode import _encode_via_pipeline, _resolve_gemma_path, DEFAULT_GEMMA
from .utils import FPS, download_model, PIPELINE_CKPT

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.chunker import plan_chunks, estimate_duration, ChunkSpec
from audio_core.compiler import compile_prompt

from .seedvc import convert_voice

logger = logging.getLogger(__name__)

REF_TAIL_SECONDS = 3.0
MAX_RETRIES = 3
RETRY_DURATION_FACTOR = 1.3

# Curated scene presets. First entry is a sentinel that forces the user
# to make an explicit choice — we raise an error if it's not overridden.
SCENE_SENTINEL = "Choose a scene..."
SCENE_PRESETS = [
    SCENE_SENTINEL,
    "Absolute silence",
    "Quiet indoor room",
    "Reverberant hall",
    "Broadcast studio",
    "Outdoor, open air",
    "Café or restaurant",
    "Windy outdoors",
    "Rainy outdoors",
]

# Scenes that carry the "no ambient / studio-clean" intent. Extended Generate
# auto-strips background bleed on these unless strip_background_sfx overrides.
CLEAN_SPEECH_SCENES = {"Absolute silence", "Broadcast studio"}

# 12 tested languages (from the announcement + Pro blog posts on scenema.ai)
LANGUAGE_OPTIONS = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Japanese",
    "Korean",
    "Chinese",
    "Hindi",
    "Arabic",
    "Swahili",
]

LANGUAGE_CODES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
    "Chinese": "zh", "Hindi": "hi", "Arabic": "ar", "Swahili": "sw",
}


def _derive_shot(scene):
    """Auto-derive the film 'shot' attribute from scene semantics.

    Clean scenes (silence, studio) get 'closeup' — dry, dialogue-focused.
    All other scenes get 'wide' — ambient bleeds in around the speech.
    """
    return "closeup" if scene in CLEAN_SPEECH_SCENES else "wide"


import re as _re


def _xml_escape(text):
    """Escape &, <, > for XML text content. Attribute values need extra quoting."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _speech_body_parts(speech_text):
    """Parse [bracketed] cues in speech text and yield interleaved
    text and <action> XML fragments in document order.

    Example: "Hello. [He laughs bitterly] Goodbye." becomes:
      ['  Hello.', '  <action>He laughs bitterly</action>', '  Goodbye.']
    """
    parts = _re.split(r'\[([^\]]+)\]', speech_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                yield f"  {_xml_escape(text)}"
        else:
            cue = part.strip()
            if cue:
                yield f"  <action>{_xml_escape(cue)}</action>"


def _build_xml(voice_description, gender, speech_text, scene, custom_scene,
               action_tags, language):
    """Construct the <speak> XML the compiler expects from form fields.

    action_tags: multiline field, one cue per line, prepended before the first
        sentence to set the opening delivery.
    speech_text: may contain inline [bracketed cues] that become <action>
        tags at the exact position they appear — use this for mid-speech
        cues like [He laughs], [She whispers], [His voice cracks], etc.
    """
    scene_text = custom_scene.strip() if custom_scene and custom_scene.strip() else scene
    lang_code = LANGUAGE_CODES.get(language, "en")
    shot = _derive_shot(scene)

    voice_attr = _xml_escape(voice_description).replace('"', '&quot;')
    attrs = f'voice="{voice_attr}" gender="{gender}"'
    if scene_text:
        attrs += f' scene="{_xml_escape(scene_text)}"'
    if lang_code != "en":
        attrs += f' language="{lang_code}"'
    if shot != "closeup":
        attrs += f' shot="{shot}"'

    body_parts = []
    if action_tags and action_tags.strip():
        for line in action_tags.strip().split("\n"):
            line = line.strip()
            if line:
                body_parts.append(f"  <action>{_xml_escape(line)}</action>")
    body_parts.extend(_speech_body_parts(speech_text))
    body = "\n".join(body_parts)

    return f"<speak {attrs}>\n{body}\n</speak>"


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


class ScenemaAudioGenerate:
    """Generate expressive speech from a voice description + text.

    Handles long text automatically by splitting at sentence boundaries
    (Kokoro-timed), chaining chunks with A2V voice conditioning for
    consistency, then polishing with SeedVC. Auto-strips background bleed
    for clean scenes. Optional Whisper validation retries chunks that
    dropped words.
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
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "Male, late 60s. Deep, gravelly. Slow and deliberate. The weight of the cosmos in every word.",
                    "tooltip": "Describe the voice: age, gender presentation, timbre, accent, delivery style.",
                }),
                "gender": (["male", "female"], {
                    "tooltip": "Grammatical gender used for pronouns in the compiled prompt (he/she).",
                }),
                "speech_text": ("STRING", {
                    "multiline": True,
                    "default": "Look again at that dot. That's here. That's home. That's us.",
                    "tooltip": "The text to speak. Use [bracketed cues] inline for mid-speech performance direction: [He laughs], [She whispers], [His voice cracks]. Long text is auto-split at sentence boundaries.",
                }),
                "scene": (SCENE_PRESETS, {
                    "tooltip": "Acoustic environment. Injected into the prompt so the model imagines the space. 'Absolute silence' and 'Broadcast studio' auto-strip background bleed after generation.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_scene": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Freeform scene description. When non-empty, overrides the scene dropdown (e.g. 'Empty subway platform, distant train').",
                }),
                "action_tags": ("STRING", {
                    "multiline": True,
                    "default": "He pauses, weighing his words\nHe leans forward",
                    "tooltip": "Delivery cues. One per line. Each becomes a stage direction the model performs (e.g. 'He whispers', 'She laughs bitterly').",
                }),
                "language": (LANGUAGE_OPTIONS, {
                    "default": "English",
                    "tooltip": "Target language for the speech text. Write the text in that language.",
                }),
                "pace": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on Kokoro's duration estimate. Higher = slower speech (more time per word). 1.5 is validated as a safe default.",
                }),
                "strip_background_sfx": (["auto", "yes", "no"], {
                    "default": "auto",
                    "tooltip": "auto = strip only for clean scenes (silence, studio). yes = always strip. no = never strip.",
                }),
                "gemma_path": ("STRING", {"default": "auto"}),
                "ref_latent": ("SA_LATENT", {
                    "tooltip": "Optional voice reference for zero-shot cloning. Connect from VAE Encode fed by LoadAudio.",
                }),
                "validate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whisper-check each chunk and retry with more time if words were dropped. Adds ~1s per chunk.",
                }),
                "min_match_ratio": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "skip_vc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip SeedVC voice-consistency polish across chunks. Faster, slightly less consistent voice.",
                }),
                "vc_steps": ("INT", {"default": 25, "min": 5, "max": 50, "step": 5}),
                "vc_cfg_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    @torch.inference_mode()
    def generate(self, model, vae, voice_description, gender, speech_text, scene, seed,
                 custom_scene="", action_tags="", language="English", pace=1.5,
                 strip_background_sfx="auto",
                 gemma_path="auto", ref_latent=None,
                 validate=False, min_match_ratio=0.9,
                 skip_vc=False, vc_steps=25, vc_cfg_rate=0.5):

        if scene == SCENE_SENTINEL:
            raise ValueError(
                "scene: required field. Pick one of the acoustic presets, "
                "or provide a custom_scene string for a freeform description."
            )

        xml_prompt = _build_xml(voice_description, gender, speech_text,
                                 scene, custom_scene, action_tags, language)
        logger.info("XML prompt:\n%s", xml_prompt)
        compiled = compile_prompt(xml_prompt)
        logger.info("Compiled prompt: %s", compiled.prompt)
        chunks = self._plan(xml_prompt, compiled.prompt, compiled.speech_text, seed, pace)
        for i, c in enumerate(chunks):
            logger.info("Chunk %d prompt: %s", i + 1, c.compiled_prompt)

        torch.cuda.reset_peak_memory_stats()
        _log_vram("start")
        logger.info("Generating %d chunk(s) (validate=%s, skip_vc=%s)...",
                     len(chunks), validate, skip_vc)

        # Offload transformer to CPU before Phase 1: Gemma needs the VRAM.
        mdl_wrapper = model["model"]
        device = model["device"]
        mdl_wrapper.to("cpu")
        torch.cuda.empty_cache()
        _log_vram("transformer offloaded pre-Phase 1")

        # ── Phase 1: Encode ALL chunk prompts in one Gemma session ──
        logger.info("Phase 1: Encoding %d prompts...", len(chunks))
        chunk_encodings = self._encode_all_chunks(chunks, gemma_path)
        _log_vram("after all encoding")

        # ── Phase 2: Diffuse + decode ALL chunks in one transformer session ──
        logger.info("Phase 2: Diffusing %d chunks...", len(chunks))
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

            if validate:
                waveform = self._diffuse_with_validation(
                    mdl_wrapper, device, vae, vc, ac, chunk, ref_gpu, min_match_ratio
                )
            else:
                latent = self._diffuse_chunk(mdl_wrapper, device, vc, ac,
                                              chunk.duration_s, chunk.seed, ref_gpu)
                waveform, sr = _decode_latent(vae, latent)

            if validate:
                sr = waveform["sample_rate"]
                waveform = waveform["waveform"]

            del vc, ac, ref_gpu
            waveforms.append(waveform)

            if i < len(chunks) - 1:
                current_ref = _encode_reference(vae, waveform, sr).cpu()

        mdl_wrapper.to("cpu")
        torch.cuda.empty_cache()
        _log_vram("after all chunks")

        combined = torch.cat([w.squeeze(0) for w in waveforms], dim=-1).unsqueeze(0)
        combined_audio = {"waveform": combined, "sample_rate": sr}

        should_strip = self._should_strip_background(scene, strip_background_sfx)

        # SeedVC only runs when a reference audio was provided. For
        # description-only generation, A2V tail-chaining + the voice
        # description keep chunks consistent, and skipping SeedVC preserves
        # laughs, whispers, and other expressive vocals that its speech
        # re-synthesis would otherwise flatten.
        needs_vc = ref_latent is not None
        if not skip_vc and needs_vc:
            combined_audio = self._apply_vc(
                combined_audio, waveforms, sr, ref_latent, vae,
                vc_steps, vc_cfg_rate,
            )
        elif ref_latent is None:
            logger.info("No reference audio — skipping SeedVC (voice description handles consistency)")

        if should_strip:
            combined_audio = self._strip_background(combined_audio)

        total_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
        logger.info("Extended generate complete: %.1fs from %d chunk(s)",
                     total_duration, len(chunks))

        return (combined_audio,)

    def _should_strip_background(self, scene, override):
        """Decide whether to run MelBandRoFormer post-processing.

        auto: strip if scene implies studio-clean intent (Absolute silence,
              Broadcast studio). Ambient scenes (outdoor, café, hall, etc.)
              are left alone since users chose them for the ambient bleed.
        yes: always strip.
        no: never strip.
        """
        if override == "yes":
            return True
        if override == "no":
            return False
        return scene in CLEAN_SPEECH_SCENES

    def _strip_background(self, audio):
        """Run MelBandRoFormer to isolate speech from any ambient bleed."""
        from .vocal_separator import _run_separator
        logger.info("Stripping background SFX (scene = studio-clean intent)...")
        vocals_t, _, sr = _run_separator(audio)
        return {"waveform": vocals_t, "sample_rate": sr}

    def _diffuse_with_validation(self, mdl_wrapper, device, vae, vc, ac, chunk,
                                  ref_gpu, min_match_ratio):
        """Diffuse a chunk with Whisper validation and retry on word-match failure."""
        from audio_core.whisper_aligner import validate_text

        duration = chunk.duration_s
        seed = chunk.seed
        best_waveform = None
        best_sr = None
        best_ratio = -1.0

        for attempt in range(MAX_RETRIES + 1):
            latent = self._diffuse_chunk(mdl_wrapper, device, vc, ac, duration, seed, ref_gpu)
            waveform, sr = _decode_latent(vae, latent)

            wav_np = waveform.squeeze(0).numpy()
            if wav_np.ndim == 2:
                wav_np = wav_np.T
            passed, transcribed, ratio = validate_text(
                wav_np, sr, chunk.expected_text,
                language=chunk.language, min_word_ratio=min_match_ratio,
            )

            if ratio > best_ratio:
                best_waveform = waveform
                best_sr = sr
                best_ratio = ratio

            if passed:
                logger.info("  Validated: %.0f%% word match", ratio * 100)
                return {"waveform": best_waveform, "sample_rate": best_sr}

            if attempt < MAX_RETRIES:
                duration = min(duration * RETRY_DURATION_FACTOR, 20.0)
                seed += 1
                logger.info("  Retry %d: %.0f%% match, extending to %.1fs, seed=%d",
                             attempt + 1, ratio * 100, duration, seed)

        logger.warning("  Best %.0f%% match after %d retries, accepting",
                        best_ratio * 100, MAX_RETRIES)
        return {"waveform": best_waveform, "sample_rate": best_sr}

    def _encode_all_chunks(self, chunks, gemma_path):
        """Encode all chunk prompts via the LTX pipeline's PromptEncoder.

        Matches production Scenema Audio exactly. The pipeline internally
        streams Gemma layer-by-layer for encoding, then runs full
        process_hidden_states through the embeddings processor.
        """
        if gemma_path == "auto":
            gemma_path = DEFAULT_GEMMA

        gemma_local = _resolve_gemma_path(gemma_path)
        pipeline_path = download_model(PIPELINE_CKPT)

        all_encodings = []
        for i, chunk in enumerate(chunks):
            logger.info("  Encoding chunk %d/%d", i + 1, len(chunks))
            vc, ac = _encode_via_pipeline(chunk.compiled_prompt, gemma_local, pipeline_path)
            all_encodings.append((vc, ac))

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
        """Apply SeedVC for voice consistency across chunks."""
        chunk0_audio = {"waveform": chunk_waveforms[0], "sample_rate": sr}

        logger.info("Applying SeedVC (%d steps, cfg_rate=%.2f)...", vc_steps, vc_cfg_rate)
        result = convert_voice(combined_audio, chunk0_audio, vc_steps, vc_cfg_rate)

        if result["sample_rate"] != sr:
            result_wav = torchaudio.functional.resample(
                result["waveform"].float(), result["sample_rate"], sr
            )
            result = {"waveform": result_wav, "sample_rate": sr}

        return result

    def _plan(self, xml_prompt, compiled_prompt, speech_text, seed, pace):
        """Plan chunks from the XML — falls back to single chunk on failure."""
        try:
            chunks = plan_chunks(xml_prompt, base_seed=seed, pace=pace)
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
