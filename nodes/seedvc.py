# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio voice conversion node for ComfyUI.

Converts voice identity of generated speech to match a reference speaker
while preserving prosody, rhythm, and emotion. Uses the Seed-VC model.

Note: SeedVC requires its repository cloned locally with model checkpoints.
The SEEDVC_PATH environment variable must point to the cloned repo.
SeedVC's internal imports (app_vc, modules.bigvgan) are loaded dynamically
because they require the repo on sys.path and cwd set to the repo root.
This is a constraint of SeedVC's architecture, not a design choice.
"""

import inspect
import logging
import os
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

SEEDVC_SR = 22050
DEFAULT_STEPS = 25
DEFAULT_CFG_RATE = 0.5


def _get_seedvc_path():
    """Resolve the SeedVC repository path."""
    seedvc_path = os.environ.get("SEEDVC_PATH", "")
    if not seedvc_path or not os.path.isdir(seedvc_path):
        raise ImportError(
            "SeedVC not found. Set SEEDVC_PATH environment variable "
            "to the path of the cloned seed-vc repository. "
            "See: https://github.com/Plachtaa/seed-vc"
        )
    return seedvc_path


def _load_seedvc(seedvc_path):
    """Load SeedVC models to GPU.

    SeedVC requires cwd set to its repo root and its modules on sys.path.
    This is a constraint of SeedVC's internal imports, not our choice.
    """
    original_cwd = os.getcwd()
    os.chdir(seedvc_path)

    if "gradio" not in sys.modules:
        sys.modules["gradio"] = types.ModuleType("gradio")

    if seedvc_path not in sys.path:
        sys.path.insert(0, seedvc_path)

    os.environ.setdefault(
        "HF_HUB_CACHE",
        str(Path(seedvc_path) / "checkpoints" / "hf_cache"),
    )

    # Patch BigVGAN for huggingface_hub compat
    import modules.bigvgan.bigvgan as bigvgan_mod
    orig_from_pretrained = bigvgan_mod.BigVGAN._from_pretrained

    @classmethod
    def patched_from_pretrained(cls, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return orig_from_pretrained.__func__(cls, **kwargs)

    bigvgan_mod.BigVGAN._from_pretrained = patched_from_pretrained

    import app_vc
    app_vc.device = torch.device("cuda")

    args = Namespace(checkpoint=None, config=None, fp16=True, gpu=0)
    (
        app_vc.model,
        app_vc.semantic_fn,
        app_vc.vocoder_fn,
        app_vc.campplus_model,
        app_vc.to_mel,
        app_vc.mel_fn_args,
    ) = app_vc.load_models(args)

    app_vc.max_context_window = app_vc.sr // app_vc.hop_length * 30
    app_vc.overlap_wave_len = app_vc.overlap_frame_len * app_vc.hop_length

    os.chdir(original_cwd)
    return app_vc


def _run_conversion(app_vc, source_path, target_path, steps, cfg_rate):
    """Run SeedVC voice conversion and return audio samples."""
    vc_kwargs = {
        "source": source_path,
        "target": target_path,
        "diffusion_steps": steps,
        "length_adjust": 1.0,
        "inference_cfg_rate": cfg_rate,
    }
    sig = inspect.signature(app_vc.voice_conversion)
    if "n_quantizers" in sig.parameters:
        vc_kwargs["n_quantizers"] = 3

    audio_tuple = None
    for result in app_vc.voice_conversion(**vc_kwargs):
        if isinstance(result, tuple) and len(result) == 2:
            _, audio_tuple = result

    if audio_tuple is None:
        raise RuntimeError("SeedVC produced no output")

    sample_rate, samples = audio_tuple
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    peak = np.abs(samples).max()
    if peak > 1.0:
        samples = samples / peak

    return samples, sample_rate


def _cleanup_seedvc(app_vc):
    """Free SeedVC models from GPU."""
    for attr in ["model", "semantic_fn", "vocoder_fn", "campplus_model", "to_mel"]:
        if hasattr(app_vc, attr):
            delattr(app_vc, attr)
    torch.cuda.empty_cache()


def _audio_to_temp_wav(audio_data, target_sr):
    """Write ComfyUI AUDIO to a temp WAV file at target sample rate."""
    wav = audio_data["waveform"][0]
    sr = audio_data["sample_rate"]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.float(), sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav.squeeze().numpy(), target_sr)
    return tmp.name


class ScenemaAudioSeedVC:
    """Voice conversion using SeedVC.

    Converts the voice identity of source audio to match a reference
    speaker while preserving the source's delivery, emotion, and pacing.
    Requires SEEDVC_PATH environment variable pointing to the cloned
    seed-vc repository.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "convert"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("AUDIO",),
                "reference": ("AUDIO",),
            },
            "optional": {
                "steps": ("INT", {
                    "default": DEFAULT_STEPS, "min": 5, "max": 50, "step": 5,
                }),
                "cfg_rate": ("FLOAT", {
                    "default": DEFAULT_CFG_RATE, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
            },
        }

    @torch.inference_mode()
    def convert(self, source, reference, steps=DEFAULT_STEPS, cfg_rate=DEFAULT_CFG_RATE):
        seedvc_path = _get_seedvc_path()

        src_path = _audio_to_temp_wav(source, SEEDVC_SR)
        ref_path = _audio_to_temp_wav(reference, SEEDVC_SR)

        try:
            logger.info("Running voice conversion (%d steps, cfg_rate=%.2f)...", steps, cfg_rate)
            app_vc = _load_seedvc(seedvc_path)
            samples, out_sr = _run_conversion(app_vc, src_path, ref_path, steps, cfg_rate)
            _cleanup_seedvc(app_vc)
        finally:
            os.unlink(src_path)
            os.unlink(ref_path)

        out_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
        logger.info("Voice conversion complete: %.1fs", len(samples) / out_sr)
        return ({"waveform": out_tensor, "sample_rate": out_sr},)
