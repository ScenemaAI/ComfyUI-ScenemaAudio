# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Shared utilities for Scenema Audio ComfyUI nodes.

INT8 linear layer, audio-only transformer patch, HuggingFace download helpers.
"""

import gc
import json
import logging

import torch
from huggingface_hub import hf_hub_download
from ltx_core.batch_split import BatchedPerturbationConfig
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio
from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, rms_norm
from safetensors import safe_open
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

HF_REPO = "ScenemaAI/scenema-audio"

TRANSFORMER_BF16 = "scenema-audio-transformer.safetensors"
TRANSFORMER_INT8 = "scenema-audio-transformer-int8.safetensors"
PIPELINE_CKPT = "scenema-audio-pipeline.safetensors"
PIPELINE_AUDIO_CKPT = "scenema-audio-pipeline-audio.safetensors"
VAE_ENCODER_CKPT = "scenema-audio-vae-encoder.safetensors"

FPS = 24
MAX_REF_SECONDS = 5


# ── INT8 Linear ────────────────────────────────────────────────────────


class Int8Linear(torch.nn.Module):
    """Linear layer with INT8 weights, dequantized to input dtype during forward."""

    def __init__(self, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_parameter("bias", torch.nn.Parameter(bias))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


# ── Audio-Only Forward Patch ───────────────────────────────────────────


def audio_only_forward(self, video, audio, perturbations=None):
    """Monkey-patched forward for audio-only transformer blocks.

    Skips all video computation and only runs audio self-attention,
    cross-attention, and feedforward.
    """
    if video is None and audio is None:
        raise ValueError("Need at least one modality")
    batch_size = (video or audio).x.shape[0]
    if perturbations is None:
        perturbations = BatchedPerturbationConfig.empty(batch_size)
    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None
    run_ax = audio is not None and audio.enabled and ax.numel() > 0
    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
        )
        norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        ax = (
            ax
            + self.audio_attn1(
                norm_ax, pe=audio.positional_embeddings, mask=audio.self_attention_mask
            )
            * agate_msa
        )
        del agate_msa, norm_ax
        ax = ax + self._apply_text_cross_attention(
            ax,
            audio.context,
            self.audio_attn2,
            self.audio_scale_shift_table,
            getattr(self, "audio_prompt_scale_shift_table", None),
            audio.timesteps,
            audio.prompt_timestep,
            audio.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        ashift_ff, ascale_ff, agate_ff = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
        )
        norm_ax_ff = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_ff) + ashift_ff
        del ashift_ff, ascale_ff
        ax = ax + self.audio_ff(norm_ax_ff) * agate_ff
        del agate_ff, norm_ax_ff
    if video is not None:
        object.__setattr__(video, "x", vx)
    if audio is not None:
        object.__setattr__(audio, "x", ax)
    return video, audio


# ── Meta Tensor Materialization ────────────────────────────────────────


def materialize_meta_tensors(module, device="cpu"):
    """Replace meta tensors with zeros on the specified device."""
    for name, param in list(module.named_parameters()):
        if param.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._parameters[parts[-1]] = torch.nn.Parameter(
                torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
            )
    for name, buf in list(module.named_buffers()):
        if buf.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = torch.zeros(
                buf.shape, dtype=torch.bfloat16, device=device
            )


# ── HuggingFace Download ──────────────────────────────────────────────


def download_model(filename, cache_dir=None):
    """Download a model file from HuggingFace if not already cached."""
    return hf_hub_download(repo_id=HF_REPO, filename=filename, cache_dir=cache_dir)


# ── Model Loading ─────────────────────────────────────────────────────


def _detect_int8_keys(state_dict):
    """Detect INT8 checkpoint format and return key mappings."""
    int8_map = {
        k.replace(".weight.int8", ""): k for k in state_dict if k.endswith(".weight.int8")
    }
    scale_map = {
        k.replace(".weight.scale", ""): k for k in state_dict if k.endswith(".weight.scale")
    }
    return int8_map, scale_map


def _load_int8_weights(mdl_wrapper, state_dict, int8_map, scale_map):
    """Load INT8 quantized weights into the model."""
    regular_sd = {
        k: v for k, v in state_dict.items()
        if not k.endswith(".int8") and not k.endswith(".scale")
    }
    mdl_wrapper.load_state_dict(regular_sd, strict=False, assign=True)

    n_replaced = 0
    for name in int8_map:
        w_int8 = state_dict[int8_map[name]]
        w_scale = state_dict[scale_map[name]]
        parts = name.split(".")
        parent = mdl_wrapper
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        bias_key = name + ".bias"
        bias = state_dict.get(bias_key)
        if bias is None and hasattr(old, "bias") and old.bias is not None:
            bias = old.bias.data
        setattr(parent, parts[-1], Int8Linear(w_int8, w_scale, bias))
        n_replaced += 1

    logger.info("INT8: replaced %d Linear layers", n_replaced)


def _nuke_video_paths(mdl):
    """Replace video computation paths with Identity modules."""
    for block in mdl.transformer_blocks:
        block.attn1 = torch.nn.Identity()
        block.attn2 = torch.nn.Identity()
        block.ff = torch.nn.Identity()
        block.audio_to_video_attn = torch.nn.Identity()
    gc.collect()


def load_transformer(checkpoint_path):
    """Load the audio-only transformer from a safetensors checkpoint.

    Handles both bf16 and INT8 quantized checkpoints. Applies the
    audio-only forward patch and nukes video computation paths.

    Returns:
        Tuple of (X0Model wrapper, config dict).
    """
    with safe_open(checkpoint_path, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    with torch.device("meta"):
        mdl = LTXModelConfigurator.from_config(config)

    sd = load_file(checkpoint_path, device="cpu")
    int8_map, scale_map = _detect_int8_keys(sd)
    is_int8 = len(int8_map) > 0

    mdl_wrapper = X0Model(mdl)

    if is_int8:
        _load_int8_weights(mdl_wrapper, sd, int8_map, scale_map)
    else:
        mdl_wrapper.load_state_dict(sd, strict=False, assign=True)

    del sd
    gc.collect()

    _nuke_video_paths(mdl)
    materialize_meta_tensors(mdl_wrapper)

    cross_pe = max(
        mdl.positional_embedding_max_pos[0],
        mdl.audio_positional_embedding_max_pos[0],
    )
    mdl._init_preprocessors(cross_pe)

    BasicAVTransformerBlock.forward = audio_only_forward

    return mdl_wrapper.eval(), config


def load_vae_encoder(config, checkpoint_path):
    """Load the Audio VAE encoder from a standalone checkpoint.

    Returns:
        Tuple of (encoder module, sample_rate).
    """
    avae_cfg = config["audio_vae"]
    preproc = avae_cfg["preprocessing"]
    vae_sr = preproc["audio"]["sampling_rate"]

    with torch.device("meta"):
        encoder = AudioEncoderConfigurator().from_config(avae_cfg)

    sd = load_file(checkpoint_path, device="cpu")
    encoder.load_state_dict(sd, strict=False, assign=True)

    pcs = encoder.per_channel_statistics
    if "per_channel_statistics.std-of-means" in sd:
        pcs._buffers["std-of-means"] = sd["per_channel_statistics.std-of-means"]
        pcs._buffers["mean-of-means"] = sd["per_channel_statistics.mean-of-means"]
    del sd

    dd = avae_cfg["model"]["params"]["ddconfig"]
    encoder.mel_bins = dd["mel_bins"]
    encoder.mid.attn_1 = torch.nn.Identity()

    materialize_meta_tensors(encoder, device="cpu")

    return encoder.eval().to(torch.bfloat16), vae_sr


def extract_wav(audio_obj):
    """Extract numpy waveform from an LTX Audio object."""
    w = audio_obj.waveform.cpu().float().numpy()
    if w.ndim == 3:
        w = w.squeeze(0)
    if w.ndim == 2:
        w = w.T
    return w, audio_obj.sampling_rate
