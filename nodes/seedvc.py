# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio voice conversion node for ComfyUI.

Converts voice identity of generated speech to match a reference speaker
while preserving prosody, rhythm, and emotion. Uses vendored Seed-VC code
with model weights auto-downloaded from HuggingFace.
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

# Path to vendored SeedVC code
_VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "seedvc")

# Singleton to avoid reloading models on every call
_app_vc = None
_seedvc_loaded = False


def _ensure_seedvc_on_path():
    """Add vendored SeedVC to sys.path."""
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)
    if "gradio" not in sys.modules:
        sys.modules["gradio"] = types.ModuleType("gradio")


def _load_seedvc():
    """Load SeedVC models to GPU using vendored code."""
    global _app_vc, _seedvc_loaded

    if _seedvc_loaded:
        return _app_vc

    _ensure_seedvc_on_path()

    # SeedVC downloads checkpoints relative to cwd
    original_cwd = os.getcwd()
    os.makedirs(os.path.join(_VENDOR_PATH, "checkpoints"), exist_ok=True)
    os.chdir(_VENDOR_PATH)

    os.environ.setdefault(
        "HF_HUB_CACHE",
        os.path.join(_VENDOR_PATH, "checkpoints", "hf_cache"),
    )

    # NOTE: these imports MUST stay inside _load_seedvc, after
    # _ensure_seedvc_on_path() (adds vendor/seedvc to sys.path) and
    # os.chdir(_VENDOR_PATH) above. The vendored SeedVC modules load
    # config files relative to cwd at import time — moving these to the
    # top of the file breaks path resolution.
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

    _app_vc = app_vc
    _seedvc_loaded = True
    logger.info("SeedVC loaded: sr=%d", app_vc.sr)
    return app_vc


def _unload_seedvc():
    """Free SeedVC models from GPU."""
    global _app_vc, _seedvc_loaded

    if not _seedvc_loaded:
        return

    for attr in ["model", "semantic_fn", "vocoder_fn", "campplus_model", "to_mel"]:
        if hasattr(_app_vc, attr):
            delattr(_app_vc, attr)

    _app_vc = None
    _seedvc_loaded = False
    torch.cuda.empty_cache()
    logger.info("SeedVC unloaded")


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


def convert_voice(source_audio, reference_audio, steps=DEFAULT_STEPS, cfg_rate=DEFAULT_CFG_RATE):
    """Convert voice identity. Used by both the standalone node and Extended Generate.

    Args:
        source_audio: ComfyUI AUDIO dict (source speech)
        reference_audio: ComfyUI AUDIO dict (target voice identity)
        steps: SeedVC diffusion steps
        cfg_rate: Classifier-free guidance rate

    Returns:
        ComfyUI AUDIO dict with converted voice
    """
    app_vc = _load_seedvc()

    src_path = _audio_to_temp_wav(source_audio, SEEDVC_SR)
    ref_path = _audio_to_temp_wav(reference_audio, SEEDVC_SR)

    try:
        samples, out_sr = _run_conversion(app_vc, src_path, ref_path, steps, cfg_rate)
    finally:
        os.unlink(src_path)
        os.unlink(ref_path)
        # SeedVC holds ~3.5GB on GPU. Unload so a subsequent run (which starts
        # with Voice Design loading a 5.5GB transformer) doesn't OOM on 8GB cards.
        _unload_seedvc()

    out_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    return {"waveform": out_tensor, "sample_rate": out_sr}


class ScenemaAudioVoiceClone:
    """Voice conversion using SeedVC.

    Converts the voice identity of source audio to match a reference
    speaker while preserving the source's delivery, emotion, and pacing.
    Model weights are auto-downloaded from HuggingFace on first use.
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
        logger.info("Running voice conversion (%d steps, cfg_rate=%.2f)...", steps, cfg_rate)
        result = convert_voice(source, reference, steps, cfg_rate)
        logger.info("Voice conversion complete: %.1fs",
                     result["waveform"].shape[-1] / result["sample_rate"])
        return (result,)
