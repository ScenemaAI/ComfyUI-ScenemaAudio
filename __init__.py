# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""ComfyUI custom nodes for Scenema Audio.

Expressive text-to-speech with zero-shot voice cloning via
LTX 2.3 audio-only diffusion.
"""

import os
import shutil
import sys

# Add vendored packages to sys.path before any node imports.
# ltx_core, ltx_pipelines, seedvc, and mel_band_roformer are vendored
# to avoid external git dependencies and ensure one-click install.
_vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

# Point subprocess calls to "ffmpeg" at the imageio-ffmpeg bundled binary,
# so users don't need a system-wide ffmpeg install. imageio-ffmpeg is a
# pip-installable Python package that ships an ffmpeg binary, declared
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
        AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
except (ImportError, RuntimeError):
    pass

try:
    from .nodes.model_loader import ScenemaAudioModelLoader
    from .nodes.vae_loader import ScenemaAudioVAELoader
    from .nodes.vae_encode import ScenemaAudioVAEEncode
    from .nodes.load_audio_url import ScenemaAudioLoadAudioURL
    from .nodes.seedvc import ScenemaAudioVoiceClone
    from .nodes.generate import ScenemaAudioGenerate

    NODE_CLASS_MAPPINGS = {
        "ScenemaAudioModelLoader": ScenemaAudioModelLoader,
        "ScenemaAudioVAELoader": ScenemaAudioVAELoader,
        "ScenemaAudioVAEEncode": ScenemaAudioVAEEncode,
        "ScenemaAudioLoadAudioURL": ScenemaAudioLoadAudioURL,
        "ScenemaAudioVoiceClone": ScenemaAudioVoiceClone,
        "ScenemaAudioGenerate": ScenemaAudioGenerate,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ScenemaAudioModelLoader": "Scenema Audio Model Loader",
        "ScenemaAudioVAELoader": "Scenema Audio VAE Loader",
        "ScenemaAudioVAEEncode": "Scenema Audio VAE Encode",
        "ScenemaAudioLoadAudioURL": "Scenema Audio Load Audio from URL",
        "ScenemaAudioVoiceClone": "Scenema Audio Voice Clone",
        "ScenemaAudioGenerate": "Scenema Audio Generate",
    }

    # JS extensions live in web/. Powers the preset dropdown's auto-fill.
    WEB_DIRECTORY = "./web"

    # Copy shipped workflow files into ComfyUI's user workflows directory so
    # they appear in the Workflows sidebar without the user having to drag
    # JSON files onto the canvas. Only copies files that don't already exist,
    # so user edits are preserved across restarts.
    def _install_workflows():
        try:
            import folder_paths
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(pkg_dir, "workflows")
            if not os.path.isdir(src_dir):
                return
            user_dir = getattr(folder_paths, "user_directory", None) \
                or folder_paths.get_user_directory()
            dest_dir = os.path.join(user_dir, "default", "workflows", "Scenema Audio")
            os.makedirs(dest_dir, exist_ok=True)
            for fn in os.listdir(src_dir):
                if not fn.endswith(".json"):
                    continue
                dest = os.path.join(dest_dir, fn)
                if not os.path.exists(dest):
                    shutil.copy2(os.path.join(src_dir, fn), dest)
        except Exception:
            pass

    _install_workflows()

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

except ImportError:
    # Running outside ComfyUI (e.g. pytest)
    pass
