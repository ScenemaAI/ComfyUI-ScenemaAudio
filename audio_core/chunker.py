# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Text chunking and duration estimation for Scenema Audio.

Splits long text into chunks at sentence boundaries using Kokoro TTS
phoneme-level timing as the source of truth for duration. No word counting.

Algorithm:
  1. Split text into sentences
  2. Estimate each sentence's duration via Kokoro (one call per sentence)
  3. Greedily merge: accumulate sentence durations, start a new chunk
     when running_sum * LTX_MULTIPLIER exceeds MAX_CHUNK_DURATION_S
"""

import logging
import random
from dataclasses import dataclass

from .compiler import compile_chunk_prompt, compile_prompt, extract_sentence_actions
from .validator import validate_prompt

logger = logging.getLogger(__name__)

FALLBACK_WORDS_PER_SEC = 2.2  # Test-environment-only fallback when Kokoro is mocked
ACTION_DURATION_S = 1.5  # Extra time per action block
MAX_CHUNK_DURATION_S = (
    15.0  # Safe generation limit — model trained on 20s but repeats beyond ~15s
)
LTX_MULTIPLIER = 1.5  # LTX speaks slower than Kokoro; overshoot for trimming

# Kokoro singleton (loaded once, reused)
_kokoro_pipeline = None
_kokoro_available: bool | None = None


def _get_kokoro():
    """Get or initialize the Kokoro TTS pipeline for duration estimation.

    Kokoro is 82M params, runs on CPU. Loaded once and cached.
    Falls back to word-count heuristic only in test environments.
    """
    global _kokoro_pipeline, _kokoro_available

    if _kokoro_available is False:
        return None

    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    try:
        from kokoro import KPipeline

        pipe = KPipeline(lang_code="a")
        # Verify it's a real Kokoro pipeline (not a mock in tests)
        if not hasattr(pipe, "__module__") or "kokoro" not in str(
            getattr(pipe, "__module__", "")
        ):
            raise TypeError("Kokoro pipeline is not genuine (test mock)")
        _kokoro_pipeline = pipe
        _kokoro_available = True
        logger.info("Kokoro TTS loaded for duration estimation")
        return _kokoro_pipeline
    except TypeError:
        # Test environment with mocks, fall back silently
        _kokoro_available = False
        return None
    except (ImportError, Exception) as e:
        _kokoro_available = False
        logger.error("Kokoro is required but not available: %s", e)
        raise RuntimeError(
            f"Kokoro TTS is a required dependency for duration estimation. "
            f"Install it with: pip install kokoro. Error: {e}"
        ) from e


def _kokoro_duration(text: str) -> float | None:
    """Estimate speech duration using Kokoro TTS phoneme-level timing.

    Args:
        text: Speech text to estimate duration for

    Returns:
        Duration in seconds, or None if Kokoro unavailable
    """
    pipe = _get_kokoro()
    if pipe is None:
        return None

    try:
        total_frames = 0
        for result in pipe(text, voice="af_heart"):
            if hasattr(result, "audio") and result.audio is not None:
                total_frames += len(result.audio)

        # Kokoro outputs at 24000Hz
        duration = total_frames / 24000.0
        return duration
    except Exception as e:
        logger.warning("Kokoro estimation failed: %s", e)
        return None


@dataclass
class ChunkSpec:
    compiled_prompt: str
    duration_s: float
    seed: int
    expected_text: str
    language: str = "en"


def _split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences at .!? boundaries."""
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?":
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _estimate_sentence_durations(sentences: list[str]) -> list[float]:
    """Estimate Kokoro duration for each sentence individually.

    One Kokoro call per sentence. Returns raw Kokoro durations (before
    LTX multiplier). Falls back to word-count heuristic per sentence
    only in test environments where Kokoro is mocked.
    """
    durations = []
    for sent in sentences:
        dur = _kokoro_duration(sent)
        if dur is None:
            # Test environment fallback only
            dur = len(sent.split()) / FALLBACK_WORDS_PER_SEC + 0.3
        durations.append(dur)
    return durations


def split_text_by_duration(
    text: str,
    multiplier: float = LTX_MULTIPLIER,
    max_duration: float = MAX_CHUNK_DURATION_S,
) -> list[tuple[str, float]]:
    """Split text into chunks using Kokoro duration estimation.

    Kokoro is the source of truth for duration. No word counting.

    Algorithm:
      1. Split text into sentences
      2. Estimate each sentence's duration via Kokoro (one call per sentence)
      3. Greedily merge: accumulate durations, start a new chunk when
         running_sum * multiplier would exceed max_duration

    Duration is additive across sentences because Kokoro estimates are
    phoneme-level with no cross-sentence dependencies.

    Args:
        text: Full speech text.
        multiplier: LTX speaks slower than Kokoro; applied to estimates.
        max_duration: Max audio duration per chunk (model training limit).

    Returns:
        List of (chunk_text, estimated_ltx_duration) tuples.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    # Split long sentences at commas if they exceed max_duration on their own
    expanded = []
    for sent in sentences:
        dur = _estimate_sentence_durations([sent])[0]
        if dur * multiplier > max_duration and "," in sent:
            # Split at commas and re-estimate
            clauses = [c.strip() for c in sent.split(",") if c.strip()]
            clause_durs = _estimate_sentence_durations(clauses)
            sub_texts: list[str] = []
            sub_dur = 0.0
            for clause, cdur in zip(clauses, clause_durs):
                if sub_texts and (sub_dur + cdur) * multiplier > max_duration:
                    expanded.append(", ".join(sub_texts))
                    sub_texts = []
                    sub_dur = 0.0
                sub_texts.append(clause)
                sub_dur += cdur
            if sub_texts:
                expanded.append(", ".join(sub_texts))
        else:
            expanded.append(sent)

    durations = _estimate_sentence_durations(expanded)

    chunks: list[tuple[str, float]] = []
    current_texts: list[str] = []
    current_dur = 0.0

    for sent, dur in zip(expanded, durations):
        if current_texts and (current_dur + dur) * multiplier > max_duration:
            chunk_text = " ".join(current_texts)
            chunks.append((chunk_text, min(current_dur * multiplier, max_duration)))
            current_texts = []
            current_dur = 0.0

        current_texts.append(sent)
        current_dur += dur

    if current_texts:
        chunk_text = " ".join(current_texts)
        chunks.append((chunk_text, min(current_dur * multiplier, max_duration)))

    return chunks


def estimate_duration(
    text: str,
    num_actions: int = 0,
    multiplier: float = LTX_MULTIPLIER,
) -> float:
    """Estimate audio duration for a single chunk of text.

    Used for single-chunk prompts that don't need splitting.

    Args:
        text: Speech text (no actions)
        num_actions: Number of action blocks (adds time for breaths/pauses)
        multiplier: Duration multiplier (LTX speaks slower than Kokoro)
    """
    kokoro_dur = _kokoro_duration(text)

    if kokoro_dur is not None:
        base_duration = kokoro_dur
        logger.debug("Kokoro estimate: %.1fs for '%s'", kokoro_dur, text[:40])
    else:
        words = len(text.split())
        base_duration = words / FALLBACK_WORDS_PER_SEC + 0.5

    action_time = num_actions * ACTION_DURATION_S
    duration = (base_duration + action_time) * multiplier
    return min(duration, MAX_CHUNK_DURATION_S)


def plan_chunks(
    xml_string: str,
    base_seed: int = -1,
    pace: float = LTX_MULTIPLIER,
) -> list[ChunkSpec]:
    """Plan generation chunks from an XML prompt.

    Validates XML, extracts text, splits into duration-based chunks
    using Kokoro, and builds per-chunk compiled prompts.

    Args:
        xml_string: Valid <speak> XML string
        base_seed: Base seed (-1 for random, otherwise sequential per chunk)
        pace: Duration multiplier (default 1.5). Higher = slower speech.
    """
    result = validate_prompt(xml_string)
    if not result.valid:
        raise ValueError(f"Invalid prompt: {'; '.join(result.errors)}")

    compiled = compile_prompt(xml_string)

    if base_seed == -1:
        base_seed = random.randint(0, 999999)

    # Check if entire text fits in a single chunk (uncapped duration for this check)
    kokoro_dur = _kokoro_duration(compiled.speech_text)
    if kokoro_dur is not None:
        total_dur = kokoro_dur * pace
    else:
        words = len(compiled.speech_text.split())
        total_dur = (words / FALLBACK_WORDS_PER_SEC + 0.5) * pace

    if total_dur <= MAX_CHUNK_DURATION_S:
        return [
            ChunkSpec(
                compiled_prompt=compiled.prompt,
                duration_s=min(total_dur, MAX_CHUNK_DURATION_S),
                seed=base_seed,
                expected_text=compiled.speech_text,
                language=compiled.language,
            )
        ]

    # Extract action-to-sentence mapping before splitting
    sentence_action_map = extract_sentence_actions(xml_string)

    # Split by Kokoro-estimated duration
    text_chunks = split_text_by_duration(compiled.speech_text, multiplier=pace)

    # Track which global sentence index each chunk starts at
    global_sentence_idx = 0

    specs: list[ChunkSpec] = []
    for i, (chunk_text, chunk_dur) in enumerate(text_chunks):
        # Find actions that belong to this chunk's first sentence
        actions_before = sentence_action_map.get(global_sentence_idx)

        chunk_prompt = compile_chunk_prompt(
            speech_text=chunk_text,
            voice=compiled.voice,
            scene=compiled.scene,
            actions_before=actions_before,
            gender=compiled.gender,
            shot=compiled.shot,
        )
        specs.append(
            ChunkSpec(
                compiled_prompt=chunk_prompt,
                duration_s=chunk_dur,
                seed=base_seed + i * 1000,
                expected_text=chunk_text,
                language=compiled.language,
            )
        )

        # Count sentences in this chunk to advance global index
        chunk_sentences = _split_into_sentences(chunk_text)
        global_sentence_idx += len(chunk_sentences)

    logger.info(
        "Planned %d chunks (%.1fs total estimated)",
        len(specs),
        sum(s.duration_s for s in specs),
    )
    return specs
