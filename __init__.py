# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""ComfyUI custom nodes for Scenema Audio.

Expressive text-to-speech with zero-shot voice cloning via
LTX 2.3 audio-only diffusion.
"""

try:
    from .nodes.prompt_compiler import ScenemaAudioPromptCompiler
    from .nodes.model_loader import ScenemaAudioModelLoader
    from .nodes.vae_loader import ScenemaAudioVAELoader
    from .nodes.text_encode import ScenemaAudioTextEncode
    from .nodes.sampler import ScenemaAudioSampler
    from .nodes.decode import ScenemaAudioDecode
    from .nodes.vae_encode import ScenemaAudioVAEEncode

    NODE_CLASS_MAPPINGS = {
        "ScenemaAudioPromptCompiler": ScenemaAudioPromptCompiler,
        "ScenemaAudioModelLoader": ScenemaAudioModelLoader,
        "ScenemaAudioVAELoader": ScenemaAudioVAELoader,
        "ScenemaAudioTextEncode": ScenemaAudioTextEncode,
        "ScenemaAudioSampler": ScenemaAudioSampler,
        "ScenemaAudioDecode": ScenemaAudioDecode,
        "ScenemaAudioVAEEncode": ScenemaAudioVAEEncode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ScenemaAudioPromptCompiler": "Scenema Audio Prompt Compiler",
        "ScenemaAudioModelLoader": "Scenema Audio Model Loader",
        "ScenemaAudioVAELoader": "Scenema Audio VAE Loader",
        "ScenemaAudioTextEncode": "Scenema Audio Text Encode",
        "ScenemaAudioSampler": "Scenema Audio Sampler",
        "ScenemaAudioDecode": "Scenema Audio Decode",
        "ScenemaAudioVAEEncode": "Scenema Audio VAE Encode",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

except ImportError:
    # Running outside ComfyUI (e.g. pytest). Node registration not available.
    pass
