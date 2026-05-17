# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""XML prompt validation for Scenema Audio.

Validates the <speak> XML format:
  <speak voice="..." scene="..." language="...">
    <action>delivery/stage direction</action>
    Speech text here.
    <action>more direction</action>
    More speech text.
  </speak>

Only <speak> root with <action> children allowed. All content is freeform.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

ALLOWED_CHILD_TAGS = {"action", "sound"}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    voice: str | None = None
    scene: str | None = None
    language: str | None = None


def validate_prompt(xml_string: str) -> ValidationResult:
    """Validate a Scenema Audio XML prompt.

    Checks for valid XML structure, required <speak> root element,
    required voice attribute, and only <action> child elements.

    Args:
        xml_string: Raw XML string to validate.

    Returns:
        ValidationResult with parsed attributes if valid,
        or a list of errors if invalid.
    """
    errors: list[str] = []

    if not xml_string or not xml_string.strip():
        return ValidationResult(valid=False, errors=["Prompt is empty"])

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        return ValidationResult(valid=False, errors=[f"Invalid XML: {e}"])

    if root.tag != "speak":
        errors.append(f"Root element must be <speak>, got <{root.tag}>")
        return ValidationResult(valid=False, errors=errors)

    voice = root.get("voice")
    if not voice or not voice.strip():
        errors.append("Missing required 'voice' attribute on <speak>")

    gender = root.get("gender")
    if not gender or gender.strip() not in ("male", "female"):
        errors.append(
            "Missing or invalid 'gender' attribute on <speak>. Must be 'male' or 'female'"
        )

    scene = root.get("scene")
    language = root.get("language", "en")

    allowed_attrs = {"voice", "scene", "language", "gender", "shot"}
    for attr in root.attrib:
        if attr not in allowed_attrs:
            errors.append(f"Unknown attribute '{attr}' on <speak>")

    for child in root:
        if child.tag not in ALLOWED_CHILD_TAGS:
            errors.append(
                f"Unsupported tag <{child.tag}>. Only <action> and <sound> are allowed inside <speak>"
            )
        if len(list(child)) > 0:
            errors.append(f"<{child.tag}> must contain only text, no nested elements")

    has_text = False
    if root.text and root.text.strip():
        has_text = True
    for child in root:
        if child.tail and child.tail.strip():
            has_text = True
            break

    if not has_text:
        errors.append("Prompt must contain at least one speech text node")

    if errors:
        return ValidationResult(valid=False, errors=errors)

    return ValidationResult(
        valid=True,
        voice=voice.strip() if voice else None,
        scene=scene.strip() if scene else None,
        language=language.strip() if language else None,
    )
