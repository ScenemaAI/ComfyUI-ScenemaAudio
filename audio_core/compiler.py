# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""XML prompt compiler for Scenema Audio.

Compiles a <speak> XML prompt into the video-style flat text prompt
that the LTX 2.3 audio model expects.

Supports three block types inside <speak>:
  <action>  — delivery/performance cues (how the person speaks/acts)
  <sound>   — audio events that should be heard (SFX, ambient sounds)
  Text      — the actual speech content

And three shot modes via the shot attribute:
  closeup (default) — speech-focused, no SFX, clean audio
  wide              — environment + speech, SFX prominent
  scene             — raw scene description, maximum SFX

Example (closeup mode):
  Input:
    <speak voice="Deep male voice" scene="A dimly lit room" gender="male">
      <action>He takes a slow breath</action>
      Many years later, as he faced the firing squad...
    </speak>

  Output:
    Close-up in a dimly lit room. He takes a slow breath.
    "Many years later, as he faced the firing squad..."
    Deep male voice.

Example (scene mode with SFX):
  Input:
    <speak voice="Tense male whisper" scene="Dark room, heavy rain"
           gender="male" shot="scene">
      <sound>A phone rings twice then stops</sound>
      <action>He picks up the receiver and speaks in a low whisper</action>
      Its done. The package is at the location.
      <sound>Thunder rumbles in the distance</sound>
      <action>He continues urgently</action>
      You have thirty minutes.
    </speak>

  Output:
    Dark room, heavy rain. A phone rings twice then stops.
    He picks up the receiver and speaks in a low whisper:
    "Its done. The package is at the location."
    Thunder rumbles in the distance. He continues urgently:
    "You have thirty minutes."
    Tense male whisper. Dark room, heavy rain.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

DEFAULT_SCENE = "a person speaking to camera"


@dataclass
class CompiledPrompt:
    prompt: str
    speech_text: str
    voice: str
    scene: str | None
    language: str
    gender: str
    shot: str


@dataclass
class TextBlock:
    text: str


@dataclass
class ActionBlock:
    text: str


@dataclass
class SoundBlock:
    text: str


Block = TextBlock | ActionBlock | SoundBlock


def _extract_blocks(root: ET.Element) -> list[Block]:
    """Walk <speak> children in document order, extract text, action, and sound blocks."""
    blocks: list[Block] = []

    if root.text and root.text.strip():
        blocks.append(TextBlock(text=root.text.strip()))

    for child in root:
        if child.tag == "action" and child.text and child.text.strip():
            blocks.append(ActionBlock(text=child.text.strip()))
        elif child.tag == "sound" and child.text and child.text.strip():
            blocks.append(SoundBlock(text=child.text.strip()))
        if child.tail and child.tail.strip():
            blocks.append(TextBlock(text=child.tail.strip()))

    return blocks


def _ensure_trailing_punctuation(text: str) -> str:
    """Ensure text ends with sentence-ending punctuation."""
    if text and text[-1] not in ".!?\"'":
        return text + "."
    return text


SHOT_PREFIXES = {
    "closeup": "Close-up in",
    "wide": "Wide shot of",
    "scene": "",
}


def _compile_blocks(
    blocks: list[Block],
    voice: str,
    scene: str | None,
    gender: str = "male",
    shot: str = "closeup",
) -> str:
    """Compile blocks into the video-style prompt string."""
    parts: list[str] = []
    is_scene_mode = shot in ("scene", "wide")
    pronoun = "She" if gender == "female" else "He"

    scene_text = scene if scene else DEFAULT_SCENE
    prefix = SHOT_PREFIXES.get(shot, SHOT_PREFIXES["closeup"])
    if prefix:
        parts.append(f"{prefix} {scene_text}.")
    else:
        parts.append(f"{scene_text}.")

    first_speech = True
    for block in blocks:
        if isinstance(block, SoundBlock):
            # Sound events compile as standalone sentences
            parts.append(_ensure_trailing_punctuation(block.text))
        elif isinstance(block, ActionBlock):
            if is_scene_mode:
                # In scene/wide mode, action flows into speech with connector
                # Don't add punctuation — the colon before the quote handles it
                parts.append(block.text + ":")
            else:
                # In closeup mode, action is a standalone sentence
                parts.append(_ensure_trailing_punctuation(block.text))
        elif isinstance(block, TextBlock):
            clean_text = _ensure_trailing_punctuation(block.text)
            if (
                is_scene_mode
                and first_speech
                and not any(isinstance(b, ActionBlock) for b in blocks)
            ):
                # No action before first speech in scene mode — add pronoun
                parts.append(f'{pronoun} speaks: "{clean_text}"')
            else:
                parts.append(f'"{clean_text}"')
            first_speech = False

    parts.append(_ensure_trailing_punctuation(voice))

    # In scene/wide mode, repeat scene as SFX reinforcement at the end
    if is_scene_mode and scene:
        parts.append(_ensure_trailing_punctuation(scene))

    return " ".join(parts)


def _extract_speech_only(blocks: list[Block]) -> str:
    """Extract only speech text (no actions or sounds) for duration estimation."""
    texts = [b.text for b in blocks if isinstance(b, TextBlock)]
    return " ".join(texts)


def compile_prompt(xml_string: str) -> CompiledPrompt:
    """Compile a <speak> XML prompt into a video-style text prompt.

    Args:
        xml_string: Valid <speak> XML string (must pass validate_prompt first)

    Returns:
        CompiledPrompt with the compiled prompt and extracted metadata
    """
    root = ET.fromstring(xml_string)

    voice = root.get("voice", "").strip()
    scene = root.get("scene")
    if scene:
        scene = scene.strip()
    language = root.get("language", "en").strip()
    gender = root.get("gender", "male").strip()
    shot = root.get("shot", "closeup").strip()

    blocks = _extract_blocks(root)
    prompt = _compile_blocks(blocks, voice, scene, gender, shot)
    speech_text = _extract_speech_only(blocks)

    return CompiledPrompt(
        prompt=prompt,
        speech_text=speech_text,
        voice=voice,
        scene=scene,
        language=language,
        gender=gender,
        shot=shot,
    )


def extract_sentence_actions(xml_string: str) -> dict[int, list[str]]:
    """Map sentence indices to their preceding action blocks.

    Walks the XML blocks in order, tracking the most recent action(s).
    When a text block is encountered, its sentences inherit the pending actions.
    Only the first sentence of each text block gets the actions (the action
    precedes the text block in the XML).

    Returns:
        Dict mapping sentence index (0-based across all speech text) to a list
        of action strings that precede that sentence.
    """
    root = ET.fromstring(xml_string)
    blocks = _extract_blocks(root)

    sentence_actions: dict[int, list[str]] = {}
    pending_actions: list[str] = []
    sentence_idx = 0

    for block in blocks:
        if isinstance(block, ActionBlock):
            pending_actions.append(block.text)
        elif isinstance(block, TextBlock):
            # Split this text block into sentences to count them
            text = block.text.strip()
            sentences = []
            current = ""
            for char in text:
                current += char
                if char in ".!?":
                    s = current.strip()
                    if s:
                        sentences.append(s)
                    current = ""
            if current.strip():
                sentences.append(current.strip())

            if pending_actions and sentences:
                sentence_actions[sentence_idx] = pending_actions.copy()
                pending_actions.clear()

            sentence_idx += len(sentences)

    return sentence_actions


def extract_speech_text(xml_string: str) -> str:
    """Extract only the speech text from XML, ignoring actions and sounds.

    Useful for duration estimation (Kokoro) without compiling the full prompt.
    """
    root = ET.fromstring(xml_string)
    blocks = _extract_blocks(root)
    return _extract_speech_only(blocks)


def compile_chunk_prompt(
    speech_text: str,
    voice: str,
    scene: str | None = None,
    actions_before: list[str] | None = None,
    actions_after: list[str] | None = None,
    gender: str = "male",
    shot: str = "closeup",
) -> str:
    """Compile a single chunk's prompt from pre-split text.

    Used by the chunker to build per-chunk prompts after text splitting.

    Args:
        speech_text: The chunk's speech text portion.
        voice: Voice description string.
        scene: Scene description string (optional).
        actions_before: Action blocks to prepend before speech.
        actions_after: Action blocks to append after speech.

    Returns:
        Compiled video-style prompt string.
    """
    blocks: list[Block] = []

    if actions_before:
        for a in actions_before:
            blocks.append(ActionBlock(text=a))

    blocks.append(TextBlock(text=speech_text))

    if actions_after:
        for a in actions_after:
            blocks.append(ActionBlock(text=a))

    return _compile_blocks(blocks, voice, scene, gender, shot)
