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
    from .nodes.vocal_separator import ScenemaAudioVocalSeparator
    from .nodes.seedvc import ScenemaAudioVoiceClone
    from .nodes.chunker import ScenemaAudioChunker, ScenemaAudioConcatenate
    from .nodes.extended_generate import ScenemaAudioExtendedGenerate
    from .nodes.music_generate import ScenemaAudioMusicGenerate
    from .nodes.voice_design import ScenemaAudioVoiceDesign

    NODE_CLASS_MAPPINGS = {
        "ScenemaAudioPromptCompiler": ScenemaAudioPromptCompiler,
        "ScenemaAudioModelLoader": ScenemaAudioModelLoader,
        "ScenemaAudioVAELoader": ScenemaAudioVAELoader,
        "ScenemaAudioTextEncode": ScenemaAudioTextEncode,
        "ScenemaAudioSampler": ScenemaAudioSampler,
        "ScenemaAudioDecode": ScenemaAudioDecode,
        "ScenemaAudioVAEEncode": ScenemaAudioVAEEncode,
        "ScenemaAudioVocalSeparator": ScenemaAudioVocalSeparator,
        "ScenemaAudioVoiceClone": ScenemaAudioVoiceClone,
        "ScenemaAudioChunker": ScenemaAudioChunker,
        "ScenemaAudioConcatenate": ScenemaAudioConcatenate,
        "ScenemaAudioExtendedGenerate": ScenemaAudioExtendedGenerate,
        "ScenemaAudioMusicGenerate": ScenemaAudioMusicGenerate,
        "ScenemaAudioVoiceDesign": ScenemaAudioVoiceDesign,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ScenemaAudioPromptCompiler": "Scenema Audio Prompt Compiler",
        "ScenemaAudioModelLoader": "Scenema Audio Model Loader",
        "ScenemaAudioVAELoader": "Scenema Audio VAE Loader",
        "ScenemaAudioTextEncode": "Scenema Audio Text Encode",
        "ScenemaAudioSampler": "Scenema Audio Sampler",
        "ScenemaAudioDecode": "Scenema Audio Decode",
        "ScenemaAudioVAEEncode": "Scenema Audio VAE Encode",
        "ScenemaAudioVocalSeparator": "Scenema Audio Vocal Separator",
        "ScenemaAudioVoiceClone": "Scenema Audio Voice Clone",
        "ScenemaAudioChunker": "Scenema Audio Chunker",
        "ScenemaAudioConcatenate": "Scenema Audio Concatenate",
        "ScenemaAudioExtendedGenerate": "Scenema Audio Extended Generate",
        "ScenemaAudioMusicGenerate": "Scenema Audio Music Generate",
        "ScenemaAudioVoiceDesign": "Scenema Audio Voice Design",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

except ImportError:
    # Running outside ComfyUI (e.g. pytest)
    pass
