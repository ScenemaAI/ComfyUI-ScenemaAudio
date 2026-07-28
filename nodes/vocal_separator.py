# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""MelBandRoFormer utility for stripping background from generated speech.

No standalone ComfyUI node — the logic is invoked automatically by
Extended Generate when the chosen scene implies studio-clean intent
(Absolute silence, Broadcast studio) or when the user overrides via
strip_background_sfx. See extended_generate.py for the decision logic.
"""

import logging
import os
import sys

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# MelBandRoFormer architecture is vendored in vendor/mel_band_roformer/.
# The parent __init__.py adds vendor/ to sys.path so this import works.
try:
    from mel_band_roformer.mel_band_roformer import MelBandRoformer
except ImportError:
    MelBandRoformer = None

logger = logging.getLogger(__name__)

MELBAND_HF_REPO = "Kijai/MelBandRoFormer_comfy"
MELBAND_FILENAME = "MelBandRoformer_fp16.safetensors"
MELBAND_SR = 44100
CHUNK_SIZE = 352800  # ~8 seconds at 44100Hz
OVERLAP_FACTOR = 2

MODEL_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": MELBAND_SR,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}


def _load_melband_model():
    """Download and load MelBandRoFormer model."""
    if MelBandRoformer is None:
        raise ImportError(
            "MelBandRoFormer architecture failed to import from vendor/. "
            "Reinstall the ComfyUI-ScenemaAudio package."
        )

    model_path = hf_hub_download(repo_id=MELBAND_HF_REPO, filename=MELBAND_FILENAME)
    model = MelBandRoformer(**MODEL_CONFIG)
    sd = load_file(model_path)
    model.load_state_dict(sd)
    del sd
    return model.cuda().eval().float()


def _chunked_inference(model, audio_np):
    """Run inference in overlapping chunks with fade windows."""
    total_samples = audio_np.shape[1]
    chunk_size = CHUNK_SIZE
    overlap = chunk_size // OVERLAP_FACTOR
    step = chunk_size - overlap

    fade_in = np.linspace(0, 1, overlap, dtype=np.float32)
    fade_out = np.linspace(1, 0, overlap, dtype=np.float32)

    result = np.zeros_like(audio_np)
    weight = np.zeros(total_samples, dtype=np.float32)

    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_size, total_samples)
        chunk = audio_np[:, pos:end]

        if chunk.shape[1] < chunk_size:
            pad_width = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)))

        chunk_t = torch.from_numpy(chunk.copy()).unsqueeze(0).cuda().float()
        out = model(chunk_t)
        out_np = out.squeeze(0).cpu().float().numpy()[:, :end - pos]

        chunk_len = end - pos
        w = np.ones(chunk_len, dtype=np.float32)
        if pos > 0:
            fade_len = min(overlap, chunk_len)
            w[:fade_len] *= fade_in[:fade_len]
        if end < total_samples:
            fade_len = min(overlap, chunk_len)
            w[-fade_len:] *= fade_out[:fade_len]

        result[:, pos:end] += out_np * w[np.newaxis, :]
        weight[pos:end] += w

        pos += step

    weight = np.maximum(weight, 1e-8)
    result /= weight[np.newaxis, :]
    return result


def _run_separator(audio):
    """Run MelBandRoFormer separation and return (vocals, background) waveforms."""
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    wav = waveform[0]

    if sr != MELBAND_SR:
        wav = torchaudio.functional.resample(wav.float(), sr, MELBAND_SR)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    audio_np = wav.numpy()
    logger.info("Separating vocals (%.1fs audio)...", audio_np.shape[1] / MELBAND_SR)

    model = _load_melband_model()
    vocals_np = _chunked_inference(model, audio_np)
    del model
    torch.cuda.empty_cache()

    background_np = audio_np - vocals_np

    vocals_t = torch.from_numpy(vocals_np).unsqueeze(0)
    background_t = torch.from_numpy(background_np).unsqueeze(0)
    if sr != MELBAND_SR:
        vocals_t = torchaudio.functional.resample(vocals_t.float(), MELBAND_SR, sr)
        background_t = torchaudio.functional.resample(background_t.float(), MELBAND_SR, sr)

    return vocals_t, background_t, sr


