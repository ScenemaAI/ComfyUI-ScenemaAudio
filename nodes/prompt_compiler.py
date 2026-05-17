# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio prompt compiler node for ComfyUI.

Compiles voice description, speech text, and action tags into the
flat text prompt that the LTX 2.3 audio model expects.
"""

import os
import sys

# Ensure audio_core is importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from audio_core.compiler import compile_prompt, compile_chunk_prompt


class ScenemaAudioPromptCompiler:
    """Compiles voice + speech + actions into the prompt format for Scenema Audio.

    Accepts either individual fields (voice, speech text, action tags) or
    raw XML for power users. Outputs the compiled flat text prompt that
    Gemma encodes for the diffusion model.
    """

    CATEGORY = "Scenema Audio"
    FUNCTION = "compile"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("compiled_prompt", "speech_text", "xml_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": ("STRING", {
                    "multiline": True,
                    "default": "Male, mid 30s. Warm baritone. Clear American accent.",
                }),
                "gender": (["male", "female"],),
                "speech_text": ("STRING", {
                    "multiline": True,
                    "default": "The old lighthouse had stood on the cliff for over a century.",
                }),
            },
            "optional": {
                "scene": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
                "action_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "shot": (["closeup", "wide", "scene"], {"default": "closeup"}),
                "language": ("STRING", {"default": "en"}),
                "xml_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
        }

    def compile(self, voice, gender, speech_text, scene="", action_tags="",
                shot="closeup", language="en", xml_prompt=""):
        # If raw XML is provided, use it directly
        if xml_prompt and xml_prompt.strip():
            result = compile_prompt(xml_prompt.strip())
            return (result.prompt, result.speech_text, xml_prompt.strip())

        # Build XML from individual fields
        attrs = f'voice="{voice}" gender="{gender}"'
        if scene and scene.strip():
            attrs += f' scene="{scene.strip()}"'
        if language and language != "en":
            attrs += f' language="{language}"'
        if shot and shot != "closeup":
            attrs += f' shot="{shot}"'

        # Build body with action tags interleaved
        body_parts = []
        if action_tags and action_tags.strip():
            for line in action_tags.strip().split("\n"):
                line = line.strip()
                if line:
                    body_parts.append(f"  <action>{line}</action>")

        body_parts.append(f"  {speech_text.strip()}")
        body = "\n".join(body_parts)

        xml = f"<speak {attrs}>\n{body}\n</speak>"
        result = compile_prompt(xml)
        return (result.prompt, result.speech_text, xml)
