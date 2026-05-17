# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Whisper alignment for audio validation in Scenema Audio.

Uses faster-whisper (CTranslate2) on GPU to transcribe generated audio
and validate that the expected text was spoken. Whisper-small is 244M
params (~1GB VRAM, float16). Runs after denoise when VRAM is free.
"""

import logging
import re
import unicodedata

import librosa
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_whisper():
    """Get or initialize the whisper-small model."""
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    logger.info("Loading whisper-small for alignment validation (GPU, float16)...")
    _whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
    logger.info("whisper-small loaded (GPU)")
    return _whisper_model


def transcribe(audio_np: np.ndarray, sr: int, language: str = "en") -> str:
    """Transcribe audio and return the text."""
    model = _get_whisper()

    if audio_np.ndim == 2:
        audio_mono = audio_np.mean(axis=1).astype(np.float32)
    else:
        audio_mono = audio_np.astype(np.float32)

    if sr != 16000:
        audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)

    try:
        segments, _ = model.transcribe(
            audio_mono,
            language=language,
            word_timestamps=False,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except (ValueError, TypeError):
        logger.debug("Whisper transcribe returned unexpected type (test env?)")
        text = ""

    return text


def validate_text(
    audio_np: np.ndarray,
    sr: int,
    expected_text: str,
    language: str = "en",
    min_word_ratio: float = 0.6,
) -> tuple[bool, str, float]:
    """Validate that generated audio contains the expected text."""
    transcribed = transcribe(audio_np, sr, language)

    def normalize(t):
        t = unicodedata.normalize("NFD", t)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return set(t.split())

    expected_words = normalize(expected_text)
    transcribed_words = normalize(transcribed)

    if not expected_words:
        return True, transcribed, 1.0

    matched = expected_words & transcribed_words
    ratio = len(matched) / len(expected_words)

    passed = ratio >= min_word_ratio
    if not passed:
        logger.warning(
            "Validation failed: %.0f%% word match (need %.0f%%). "
            "Expected: %s... Got: %s...",
            ratio * 100,
            min_word_ratio * 100,
            expected_text[:60],
            transcribed[:60],
        )

    return passed, transcribed, ratio
