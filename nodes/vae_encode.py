# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio VAE encode node for ComfyUI.

Encodes reference audio to latent for A2V voice conditioning.
"""

import logging

import torch
import torchaudio
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio

from .utils import MAX_REF_SECONDS

logger = logging.getLogger(__name__)

# Hard cap on reference audio length. Longer clips don't improve voice
# cloning quality and eat GPU memory during encoding.
MAX_REF_CAP_SECONDS = 20.0


class ScenemaAudioVAEEncode:
    """Encodes reference audio to latent for A2V voice conditioning.

    Takes a standard ComfyUI AUDIO input (e.g. from LoadAudio) and
    encodes it via the Audio VAE encoder. Audio longer than 20 seconds
    is trimmed — additional length doesn't improve voice cloning and
    wastes VRAM.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "encode"
    RETURN_TYPES = ("SA_LATENT",)
    RETURN_NAMES = ("ref_latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("SA_VAE",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "max_seconds": ("FLOAT", {
                    "default": MAX_REF_CAP_SECONDS,
                    "min": 1.0,
                    "max": MAX_REF_CAP_SECONDS,
                    "step": 0.5,
                    "tooltip": "How many seconds of the reference audio to encode. Hard-capped at 20s.",
                }),
            },
        }

    @torch.inference_mode()
    def encode(self, vae, audio, max_seconds=MAX_REF_CAP_SECONDS):
        # Enforce the cap even if a caller somehow passes a larger value.
        max_seconds = min(max_seconds, MAX_REF_CAP_SECONDS)
        encoder = vae["encoder"]
        vae_sr = vae["sample_rate"]

        waveform = audio["waveform"]  # (batch, channels, samples)
        sr = audio["sample_rate"]

        # Take first item in batch
        wav = waveform[0]  # (channels, samples)

        # Resample if needed
        if sr != vae_sr:
            wav = torchaudio.functional.resample(wav.float(), sr, vae_sr)

        # Truncate to max_seconds
        max_samples = int(max_seconds * vae_sr)
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Encode via Audio VAE
        device = next(encoder.parameters()).device
        if str(device) == "cpu":
            encoder = encoder.cuda()

        audio_obj = Audio(waveform=wav.unsqueeze(0).cuda(), sampling_rate=vae_sr)
        latent = encode_audio(audio_obj, encoder)

        if str(device) == "cpu":
            encoder = encoder.cpu()

        logger.info("Reference encoded: %s", latent.shape)
        return (latent,)
