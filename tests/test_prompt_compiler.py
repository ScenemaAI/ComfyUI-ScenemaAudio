# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for the Scenema Audio prompt compiler.

Tests the core compiler logic (audio_core.compiler) which is pure Python
with no GPU or ComfyUI dependencies.
"""

from compiler import (
    compile_prompt,
    compile_chunk_prompt,
    extract_speech_text,
    extract_sentence_actions,
)


class TestCompilePrompt:
    def test_basic_closeup(self):
        xml = '<speak voice="Deep male voice" gender="male">Hello world.</speak>'
        result = compile_prompt(xml)
        assert "Hello world." in result.prompt
        assert "Deep male voice" in result.prompt
        assert result.speech_text == "Hello world."
        assert result.gender == "male"
        assert result.shot == "closeup"

    def test_with_scene(self):
        xml = '<speak voice="Female voice" gender="female" scene="A quiet room">Testing.</speak>'
        result = compile_prompt(xml)
        assert "quiet room" in result.prompt
        assert result.scene == "A quiet room"

    def test_with_action(self):
        xml = '<speak voice="Male voice" gender="male"><action>He whispers</action>Secret.</speak>'
        result = compile_prompt(xml)
        assert "He whispers" in result.prompt
        assert "Secret." in result.prompt

    def test_scene_mode(self):
        xml = '<speak voice="Tense whisper" gender="male" shot="scene" scene="Dark room"><sound>Thunder</sound>Run.</speak>'
        result = compile_prompt(xml)
        assert "Thunder" in result.prompt
        assert result.shot == "scene"
        assert result.prompt.count("Dark room") >= 2

    def test_wide_mode(self):
        xml = '<speak voice="Female voice" gender="female" shot="wide" scene="Beach">Hello.</speak>'
        result = compile_prompt(xml)
        assert "Wide shot" in result.prompt

    def test_language_extraction(self):
        xml = '<speak voice="French woman" gender="female" language="fr">Bonjour.</speak>'
        result = compile_prompt(xml)
        assert result.language == "fr"

    def test_default_language(self):
        xml = '<speak voice="Male voice" gender="male">Hello.</speak>'
        result = compile_prompt(xml)
        assert result.language == "en"

    def test_speech_text_extraction(self):
        xml = '<speak voice="Voice" gender="male"><action>Direction</action>Speech one. <action>More</action>Speech two.</speak>'
        result = compile_prompt(xml)
        assert "Speech one." in result.speech_text
        assert "Speech two." in result.speech_text
        assert "Direction" not in result.speech_text

    def test_multiple_actions(self):
        xml = """<speak voice="Male" gender="male">
            <action>Calm</action>First line.
            <action>Angry</action>Second line.
        </speak>"""
        result = compile_prompt(xml)
        assert "Calm" in result.prompt
        assert "Angry" in result.prompt
        assert "First line." in result.prompt
        assert "Second line." in result.prompt


class TestCompileChunkPrompt:
    def test_basic_chunk(self):
        result = compile_chunk_prompt(
            speech_text="Hello world.",
            voice="Deep male voice",
        )
        assert "Hello world." in result
        assert "Deep male voice" in result

    def test_chunk_with_actions(self):
        result = compile_chunk_prompt(
            speech_text="Hello world.",
            voice="Voice",
            actions_before=["He whispers"],
        )
        assert "He whispers" in result
        assert "Hello world." in result

    def test_chunk_with_scene(self):
        result = compile_chunk_prompt(
            speech_text="Hello.",
            voice="Voice",
            scene="A dark room",
        )
        assert "dark room" in result


class TestExtractSpeechText:
    def test_basic(self):
        xml = '<speak voice="Voice" gender="male">Hello world.</speak>'
        assert extract_speech_text(xml) == "Hello world."

    def test_with_actions(self):
        xml = '<speak voice="Voice" gender="male"><action>He whispers</action>Secret.</speak>'
        result = extract_speech_text(xml)
        assert result == "Secret."
        assert "whispers" not in result


class TestExtractSentenceActions:
    def test_no_actions(self):
        xml = '<speak voice="Voice" gender="male">Hello world.</speak>'
        result = extract_sentence_actions(xml)
        assert result == {}

    def test_single_action(self):
        xml = '<speak voice="Voice" gender="male"><action>He whispers</action>Secret.</speak>'
        result = extract_sentence_actions(xml)
        assert 0 in result
        assert "He whispers" in result[0]

    def test_multiple_actions(self):
        xml = '<speak voice="Voice" gender="male"><action>Calm</action>First. <action>Angry</action>Second.</speak>'
        result = extract_sentence_actions(xml)
        assert 0 in result
        assert "Calm" in result[0]
