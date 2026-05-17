# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio VAE loader node for ComfyUI.

Loads the Audio VAE decoder (from pipeline checkpoint) and encoder
(from standalone checkpoint) for audio latent encoding/decoding.
"""

import logging

import torch
from ltx_pipelines.distilled import DistilledPipeline

from .utils import (
    PIPELINE_CKPT,
    VAE_ENCODER_CKPT,
    download_model,
    load_vae_encoder,
)

logger = logging.getLogger(__name__)


class ScenemaAudioVAELoader:
    """Loads the Scenema Audio VAE (encoder + decoder).

    The decoder comes from the pipeline checkpoint (6.7 GB) and the
    encoder from the standalone VAE encoder checkpoint (42.7 MB).
    Both are downloaded from HuggingFace on first use.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("SA_VAE",)
    RETURN_NAMES = ("vae",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    def load(self):
        logger.info("Downloading/loading VAE checkpoints...")
        pipeline_path = download_model(PIPELINE_CKPT)
        encoder_path = download_model(VAE_ENCODER_CKPT)

        # Build pipeline for the audio decoder
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=pipeline_path,
            gemma_root=None,
            spatial_upsampler_path=None,
            loras=[],
        )

        # Load VAE encoder separately
        config_from_pipeline = pipeline.stage._config
        encoder, vae_sr = load_vae_encoder(config_from_pipeline, encoder_path)

        return ({
            "pipeline": pipeline,
            "encoder": encoder,
            "sample_rate": vae_sr,
        },)
