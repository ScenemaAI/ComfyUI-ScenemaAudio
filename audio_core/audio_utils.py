# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Audio utility functions for Scenema Audio.

Silence trimming, volume normalization, wav I/O, format conversion.
"""

import logging
import math

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def trim_silence(
    audio_np: np.ndarray,
    sr: int,
    max_silence: float = 0.5,
    threshold_db: float = -40,
) -> np.ndarray:
    """Trim silence exceeding max_silence from start and end of audio.

    Keeps up to max_silence seconds of silence at boundaries.

    Args:
        audio_np: Audio samples, shape (samples,) or (samples, channels).
        sr: Sample rate in Hz.
        max_silence: Maximum silence to keep at head/tail in seconds.
        threshold_db: Amplitude threshold below which audio is considered silence.

    Returns:
        Trimmed audio array with the same number of dimensions as input.
    """
    threshold = 10 ** (threshold_db / 20.0)
    max_silent_samples = int(max_silence * sr)
    window = int(0.02 * sr)  # 20ms analysis window

    if audio_np.ndim == 2:
        mono = audio_np.mean(axis=1)
    else:
        mono = audio_np

    if len(mono) < window:
        return audio_np

    energy = np.array(
        [
            np.abs(mono[i : i + window]).max()
            for i in range(0, len(mono) - window, window)
        ]
    )

    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        return audio_np

    first_voiced = max(0, voiced[0] * window - max_silent_samples)
    last_voiced = min(len(audio_np), (voiced[-1] + 1) * window + max_silent_samples)

    return audio_np[first_voiced:last_voiced]


def normalize_volume(
    audio_np: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """Normalize audio volume to target LUFS (approximate via RMS).

    Uses a simplified RMS-based LUFS approximation suitable for
    per-chunk normalization before concatenation.

    Args:
        audio_np: Audio samples, shape (samples,) or (samples, channels).
        sr: Sample rate in Hz.
        target_lufs: Target loudness in LUFS (default -23, EBU R128).

    Returns:
        Volume-normalized audio array, soft-clipped to prevent distortion.
    """
    if audio_np.ndim == 2:
        mono = audio_np.mean(axis=1)
    else:
        mono = audio_np

    rms = np.sqrt(np.mean(mono**2))
    if rms < 1e-8:
        return audio_np

    current_lufs = 20 * math.log10(rms) - 0.691
    gain_db = target_lufs - current_lufs
    gain = 10 ** (gain_db / 20.0)
    gain = max(0.1, min(gain, 10.0))

    result = audio_np * gain

    peak = np.abs(result).max()
    if peak > 0.99:
        result = result * (0.99 / peak)

    return result


def extract_wav(audio_obj) -> tuple[np.ndarray, int]:
    """Extract numpy waveform from an LTX Audio object.

    Handles shapes: (B,C,samples) -> (samples,C), (C,samples) -> (samples,C).

    Args:
        audio_obj: LTX pipeline Audio object with .waveform and .sampling_rate.

    Returns:
        Tuple of (waveform as float32 numpy, sample_rate).
    """
    w = audio_obj.waveform.cpu().float().numpy()
    if w.ndim == 3:
        w = w.squeeze(0)
    if w.ndim == 2:
        w = w.T
    return w, audio_obj.sampling_rate


def save_wav(audio_np: np.ndarray, sr: int, path: str) -> None:
    """Save audio to WAV file.

    Args:
        audio_np: Audio samples, shape (samples,) or (samples, channels).
        sr: Sample rate in Hz.
        path: Output file path.
    """
    sf.write(path, audio_np, sr)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load audio from WAV file.

    Args:
        path: Input file path.

    Returns:
        Tuple of (audio samples as float64 numpy, sample_rate).
    """
    data, sr = sf.read(path)
    return data, sr


def to_mono(audio_np: np.ndarray) -> np.ndarray:
    """Convert stereo to mono by averaging channels.

    Args:
        audio_np: Audio samples, shape (samples, 2) for stereo or (samples,) for mono.

    Returns:
        Mono audio array, shape (samples,).
    """
    if audio_np.ndim == 2 and audio_np.shape[1] == 2:
        return audio_np.mean(axis=1)
    return audio_np


def shorten_long_silence(
    audio_np: np.ndarray,
    sr: int,
    max_duration: float = 1.0,
    target_duration: float = 0.3,
    threshold_db: float = -35,
) -> np.ndarray:
    """Shorten silence regions longer than max_duration to target_duration.

    Unlike silenceremove which deletes silence entirely, this preserves
    a natural pause of target_duration seconds. Prevents chunk boundary
    artifacts while keeping the audio flow natural.

    Args:
        audio_np: Audio samples, shape (samples,) or (samples, channels).
        sr: Sample rate in Hz.
        max_duration: Silence longer than this is shortened.
        target_duration: Silence is shortened to this duration.
        threshold_db: Amplitude threshold below which audio is silence.

    Returns:
        Audio with long silence regions shortened.
    """
    threshold = 10 ** (threshold_db / 20.0)
    window = int(0.02 * sr)  # 20ms analysis window
    max_samples = int(max_duration * sr)
    target_samples = int(target_duration * sr)

    if audio_np.ndim == 2:
        mono = audio_np.mean(axis=1)
    else:
        mono = audio_np

    if len(mono) < window:
        return audio_np

    # Find silent regions
    energy = np.array(
        [
            np.abs(mono[i : i + window]).max()
            for i in range(0, len(mono) - window, window)
        ]
    )
    is_silent = energy < threshold

    # Build list of (start_sample, end_sample) for silence regions
    silence_regions = []
    in_silence = False
    start = 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i * window
            in_silence = True
        elif not silent and in_silence:
            end = i * window
            if end - start > max_samples:
                silence_regions.append((start, end))
            in_silence = False
    if in_silence:
        end = len(mono)
        if end - start > max_samples:
            silence_regions.append((start, end))

    if not silence_regions:
        return audio_np

    # Build output by keeping non-silence and shortening long silence
    parts = []
    prev_end = 0
    for s_start, s_end in silence_regions:
        # Keep audio before this silence
        parts.append(audio_np[prev_end:s_start])
        # Add shortened silence (target_duration worth)
        parts.append(audio_np[s_start : s_start + target_samples])
        prev_end = s_end

    # Keep remaining audio after last silence
    parts.append(audio_np[prev_end:])

    result = np.concatenate(parts, axis=0)
    shortened = (len(audio_np) - len(result)) / sr
    if shortened > 0:
        logger.info(
            "Shortened %d silence regions, removed %.1fs",
            len(silence_regions),
            shortened,
        )
    return result


def ensure_stereo(audio_np: np.ndarray) -> np.ndarray:
    """Convert mono to stereo by duplicating the channel.

    Args:
        audio_np: Audio samples, shape (samples,) for mono or (samples, 2) for stereo.

    Returns:
        Stereo audio array, shape (samples, 2).
    """
    if audio_np.ndim == 1:
        return np.stack([audio_np, audio_np], axis=-1)
    return audio_np
