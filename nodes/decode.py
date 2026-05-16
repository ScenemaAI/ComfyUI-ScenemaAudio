# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio decode node for ComfyUI.

Decodes audio latents to waveform via the Audio VAE decoder + vocoder.
Outputs standard ComfyUI AUDIO type.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class ScenemaAudioDecode:
    """Decodes audio latents to waveform.

    Uses the pipeline's audio decoder (VAE + vocoder) to convert
    latent representation back to an audio waveform. Outputs the
    standard ComfyUI AUDIO type compatible with all audio nodes.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "decode"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("SA_VAE",),
                "latent": ("SA_LATENT",),
            },
        }

    @torch.inference_mode()
    def decode(self, vae, latent):
        pipeline = vae["pipeline"]

        logger.info("Decoding audio latent %s...", latent.shape)
        audio_obj = pipeline.audio_decoder(latent)

        # Extract waveform and format as ComfyUI AUDIO type
        waveform = audio_obj.waveform  # (B, C, samples) or (C, samples)
        sr = audio_obj.sampling_rate

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # (1, C, samples)

        # ComfyUI AUDIO format: {"waveform": (batch, channels, samples), "sample_rate": int}
        logger.info("Decoded: %d samples at %d Hz (%.1fs)",
                     waveform.shape[-1], sr, waveform.shape[-1] / sr)

        return ({"waveform": waveform.cpu(), "sample_rate": sr},)
