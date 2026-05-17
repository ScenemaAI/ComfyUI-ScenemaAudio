# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for the Scenema Audio prompt compiler.

Tests the core compiler logic (audio_core/compiler.py) which is pure Python
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

    def test_default_scene_applied(self):
        xml = '<speak voice="Voice" gender="male">Hello.</speak>'
        result = compile_prompt(xml)
        assert "a person speaking to camera" in result.prompt

    def test_female_pronoun(self):
        xml = '<speak voice="Female" gender="female" shot="scene" scene="Office">Hello.</speak>'
        result = compile_prompt(xml)
        assert "She speaks" in result.prompt


class TestSoundTags:
    def test_sound_in_scene_mode(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Dock"><sound>Heavy rain</sound>Get inside.</speak>'
        result = compile_prompt(xml)
        assert "Heavy rain" in result.prompt
        assert "Get inside." in result.prompt

    def test_sound_in_closeup_mode(self):
        xml = '<speak voice="Voice" gender="male"><sound>Thunder</sound>Hello.</speak>'
        result = compile_prompt(xml)
        assert "Thunder" in result.prompt

    def test_sound_not_in_speech_text(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Storm"><sound>Lightning crack</sound>Run!</speak>'
        result = compile_prompt(xml)
        assert "Lightning crack" not in result.speech_text
        assert "Run!" in result.speech_text

    def test_multiple_sounds(self):
        xml = """<speak voice="Voice" gender="male" shot="scene" scene="Bank">
            <sound>Alarm bells ringing</sound>
            <action>She shouts</action>
            Nobody move!
            <sound>Police sirens outside</sound>
            Stay down!
        </speak>"""
        result = compile_prompt(xml)
        assert "Alarm bells" in result.prompt
        assert "Police sirens" in result.prompt
        assert "Nobody move!" in result.speech_text
        assert "Stay down!" in result.speech_text

    def test_sound_with_action_interleaved(self):
        xml = """<speak voice="Male whisper" gender="male" shot="scene" scene="Dark alley">
            <sound>Footsteps approaching</sound>
            <action>He freezes</action>
            Who is there?
        </speak>"""
        result = compile_prompt(xml)
        assert "Footsteps approaching" in result.prompt
        assert "He freezes" in result.prompt
        assert "Who is there?" in result.speech_text


class TestShotModes:
    def test_closeup_prefix(self):
        xml = '<speak voice="Voice" gender="male" shot="closeup" scene="Office">Hi.</speak>'
        result = compile_prompt(xml)
        assert "Close-up in" in result.prompt

    def test_wide_prefix(self):
        xml = '<speak voice="Voice" gender="male" shot="wide" scene="Beach">Hi.</speak>'
        result = compile_prompt(xml)
        assert "Wide shot of" in result.prompt

    def test_scene_no_prefix(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Dark room">Hi.</speak>'
        result = compile_prompt(xml)
        assert "Close-up" not in result.prompt
        assert "Wide shot" not in result.prompt
        assert "Dark room." in result.prompt

    def test_scene_mode_repeats_scene_at_end(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Rainy street">Hello.</speak>'
        result = compile_prompt(xml)
        occurrences = result.prompt.count("Rainy street")
        assert occurrences >= 2

    def test_closeup_no_scene_repeat(self):
        xml = '<speak voice="Voice" gender="male" shot="closeup" scene="Office">Hello.</speak>'
        result = compile_prompt(xml)
        occurrences = result.prompt.count("Office")
        assert occurrences == 1

    def test_action_colon_in_scene_mode(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Room"><action>He sighs</action>Fine.</speak>'
        result = compile_prompt(xml)
        assert "He sighs:" in result.prompt

    def test_action_period_in_closeup_mode(self):
        xml = '<speak voice="Voice" gender="male" shot="closeup"><action>He sighs</action>Fine.</speak>'
        result = compile_prompt(xml)
        assert "He sighs." in result.prompt


class TestCompileChunkPrompt:
    def test_basic_chunk(self):
        result = compile_chunk_prompt(
            speech_text="Hello world.",
            voice="Deep male voice",
        )
        assert "Hello world." in result
        assert "Deep male voice" in result

    def test_chunk_with_actions_before(self):
        result = compile_chunk_prompt(
            speech_text="Hello world.",
            voice="Voice",
            actions_before=["He whispers"],
        )
        assert "He whispers" in result
        assert "Hello world." in result

    def test_chunk_with_actions_after(self):
        result = compile_chunk_prompt(
            speech_text="Hello.",
            voice="Voice",
            actions_after=["He pauses"],
        )
        assert "He pauses" in result

    def test_chunk_with_scene(self):
        result = compile_chunk_prompt(
            speech_text="Hello.",
            voice="Voice",
            scene="A dark room",
        )
        assert "dark room" in result

    def test_chunk_scene_mode(self):
        result = compile_chunk_prompt(
            speech_text="Run!",
            voice="Voice",
            scene="Burning building",
            shot="scene",
        )
        assert result.count("Burning building") >= 2


class TestExtractSpeechText:
    def test_basic(self):
        xml = '<speak voice="Voice" gender="male">Hello world.</speak>'
        assert extract_speech_text(xml) == "Hello world."

    def test_with_actions(self):
        xml = '<speak voice="Voice" gender="male"><action>He whispers</action>Secret.</speak>'
        result = extract_speech_text(xml)
        assert result == "Secret."
        assert "whispers" not in result

    def test_with_sounds(self):
        xml = '<speak voice="Voice" gender="male" shot="scene" scene="Storm"><sound>Thunder</sound>Run!</speak>'
        result = extract_speech_text(xml)
        assert result == "Run!"
        assert "Thunder" not in result

    def test_multiple_text_blocks(self):
        xml = '<speak voice="Voice" gender="male"><action>Calm</action>First. <action>Angry</action>Second.</speak>'
        result = extract_speech_text(xml)
        assert "First." in result
        assert "Second." in result


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

    def test_multiple_actions_different_sentences(self):
        xml = '<speak voice="Voice" gender="male"><action>Calm</action>First. <action>Angry</action>Second.</speak>'
        result = extract_sentence_actions(xml)
        assert 0 in result
        assert "Calm" in result[0]

    def test_action_not_applied_to_sound(self):
        xml = '<speak voice="Voice" gender="male"><sound>Thunder</sound><action>He runs</action>Go!</speak>'
        result = extract_sentence_actions(xml)
        assert 0 in result
        assert "He runs" in result[0]
