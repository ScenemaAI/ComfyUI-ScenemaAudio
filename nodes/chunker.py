# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio chunker node for ComfyUI.

Splits long text into duration-based chunks using Kokoro TTS
for phoneme-level timing estimation. Each chunk stays under
the 15-second generation limit.
"""

import logging
import os
import sys

import torch
import torchaudio

# Ensure audio_core is importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.chunker import plan_chunks, ChunkSpec

logger = logging.getLogger(__name__)


class ScenemaAudioChunker:
    """Splits long text into generation chunks.

    Uses Kokoro TTS (82M params, CPU) for phoneme-level duration
    estimation. Each chunk stays under the 15-second model limit.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "chunk"
    RETURN_TYPES = ("SA_CHUNK_LIST",)
    RETURN_NAMES = ("chunks",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xml_prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "pace": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                }),
            },
        }

    def chunk(self, xml_prompt, seed, pace=1.5):
        logger.info("Planning chunks (pace=%.1f)...", pace)
        chunks = plan_chunks(xml_prompt, base_seed=seed, pace=pace)
        logger.info("Planned %d chunks", len(chunks))
        return (chunks,)


class ScenemaAudioConcatenate:
    """Concatenates multiple audio clips into one."""

    CATEGORY = "Scenema Audio"
    FUNCTION = "concatenate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("SA_AUDIO_LIST",),
            },
        }

    def concatenate(self, audio_list):
        if not audio_list:
            raise ValueError("No audio clips to concatenate")

        sr = audio_list[0]["sample_rate"]
        waveforms = []
        for item in audio_list:
            wav = item["waveform"]
            if item["sample_rate"] != sr:
                wav = torchaudio.functional.resample(wav.float(), item["sample_rate"], sr)
            waveforms.append(wav.squeeze(0))

        combined = torch.cat(waveforms, dim=-1).unsqueeze(0)
        logger.info("Concatenated %d clips: %.1fs total",
                     len(audio_list), combined.shape[-1] / sr)
        return ({"waveform": combined, "sample_rate": sr},)
