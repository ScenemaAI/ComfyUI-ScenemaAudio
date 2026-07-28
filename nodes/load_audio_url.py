# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio URL loader — download a reference audio clip from a URL.

Alternative to ComfyUI's LoadAudio node when the reference lives on the
web (mp3/wav on a CDN, a demo clip URL, etc.) instead of the local
input folder. Downloads to a temp file, decodes via torchaudio, and
returns a standard ComfyUI AUDIO dict.
"""

import logging
import os
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
import torchaudio

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_S = 30
USER_AGENT = "ComfyUI-ScenemaAudio/1.0"


class ScenemaAudioLoadAudioURL:
    """Downloads an audio file from a URL and returns it as AUDIO.

    Feed the output into VAE Encode → Generate.ref_latent to clone the
    voice from an online reference. Accepts mp3, wav, flac, ogg, m4a —
    anything torchaudio can decode. The reference is later capped at
    20s inside VAE Encode.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "load"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Direct URL to an audio file (mp3, wav, flac, m4a, ogg).",
                }),
            },
        }

    def load(self, url):
        url = url.strip()
        if not url:
            raise ValueError("url: required field. Paste a direct link to an audio file.")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"url: unsupported scheme '{parsed.scheme}'. Use http or https.")

        suffix = os.path.splitext(parsed.path)[1] or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            logger.info("Downloading reference audio from %s...", url)
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
                with open(tmp_path, "wb") as f:
                    while chunk := resp.read(65536):
                        f.write(chunk)

            waveform, sr = torchaudio.load(tmp_path)
            logger.info("Loaded %.1fs of audio at %dHz", waveform.shape[-1] / sr, sr)

            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            return ({"waveform": waveform, "sample_rate": sr},)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
