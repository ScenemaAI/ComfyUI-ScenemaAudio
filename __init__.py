# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""ComfyUI custom nodes for Scenema Audio.

Expressive text-to-speech with zero-shot voice cloning via
LTX 2.3 audio-only diffusion.
"""

import os
import sys

# Add vendored packages to sys.path before any node imports.
# ltx_core, ltx_pipelines, seedvc, and mel_band_roformer are vendored
# to avoid external git dependencies and ensure one-click install.
_vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

# Point subprocess calls to "ffmpeg" at the imageio-ffmpeg bundled binary,
# so users don't need a system-wide ffmpeg install. imageio-ffmpeg is a
# pip-installable Python package that ships an ffmpeg binary — declared
# in requirements.txt.
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    if _ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    # pydub caches AudioSegment.converter at import time via shutil.which("ffmpeg").
    # Set it explicitly so it uses the bundled binary even if imported before us.
    try:
        from pydub import AudioSegment
        AudioSegment.converter = _ffmpeg_exe
        AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()  # ffprobe not bundled; ffmpeg-only fallback
    except ImportError:
        pass
except (ImportError, RuntimeError):
    pass

try:
    from .nodes.model_loader import ScenemaAudioModelLoader
    from .nodes.vae_loader import ScenemaAudioVAELoader
    from .nodes.text_encode import ScenemaAudioTextEncode
    from .nodes.sampler import ScenemaAudioSampler
    from .nodes.decode import ScenemaAudioDecode
    from .nodes.vae_encode import ScenemaAudioVAEEncode
    from .nodes.load_audio_url import ScenemaAudioLoadAudioURL
    from .nodes.seedvc import ScenemaAudioVoiceClone
    from .nodes.chunker import ScenemaAudioChunker, ScenemaAudioConcatenate
    from .nodes.generate import ScenemaAudioGenerate

    NODE_CLASS_MAPPINGS = {
        "ScenemaAudioModelLoader": ScenemaAudioModelLoader,
        "ScenemaAudioVAELoader": ScenemaAudioVAELoader,
        "ScenemaAudioTextEncode": ScenemaAudioTextEncode,
        "ScenemaAudioSampler": ScenemaAudioSampler,
        "ScenemaAudioDecode": ScenemaAudioDecode,
        "ScenemaAudioVAEEncode": ScenemaAudioVAEEncode,
        "ScenemaAudioLoadAudioURL": ScenemaAudioLoadAudioURL,
        "ScenemaAudioVoiceClone": ScenemaAudioVoiceClone,
        "ScenemaAudioChunker": ScenemaAudioChunker,
        "ScenemaAudioConcatenate": ScenemaAudioConcatenate,
        "ScenemaAudioGenerate": ScenemaAudioGenerate,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ScenemaAudioModelLoader": "Scenema Audio Model Loader",
        "ScenemaAudioVAELoader": "Scenema Audio VAE Loader",
        "ScenemaAudioTextEncode": "Scenema Audio Text Encode",
        "ScenemaAudioSampler": "Scenema Audio Sampler",
        "ScenemaAudioDecode": "Scenema Audio Decode",
        "ScenemaAudioVAEEncode": "Scenema Audio VAE Encode",
        "ScenemaAudioLoadAudioURL": "Scenema Audio Load Audio from URL",
        "ScenemaAudioVoiceClone": "Scenema Audio Voice Clone",
        "ScenemaAudioChunker": "Scenema Audio Chunker",
        "ScenemaAudioConcatenate": "Scenema Audio Concatenate",
        "ScenemaAudioGenerate": "Scenema Audio Generate",
    }

    # JS extensions live in web/ — powers the preset dropdown's auto-fill.
    WEB_DIRECTORY = "./web"

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

except ImportError:
    # Running outside ComfyUI (e.g. pytest)
    pass
