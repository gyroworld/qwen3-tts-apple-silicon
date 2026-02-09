"""Tests for qwen3_tts.apple_audio (Apple-native conversion). Apple Silicon macOS only."""

import os
import platform
import sys
import tempfile
import wave

import pytest

from qwen3_tts.apple_audio import convert_to_wav

DARWIN = sys.platform == "darwin"
ARM64 = platform.machine() == "arm64"
APPLE_SILICON = DARWIN and ARM64


def _avfoundation_available():
    """True if AVFoundation is loaded (conversion supported)."""
    import qwen3_tts.apple_audio as m
    return m._AVFoundation is not None and m._Foundation is not None


def _make_wav(path: str, sample_rate: int, channels: int = 1, num_frames: int = 100) -> None:
    """Write a minimal 16-bit PCM WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * num_frames)


@pytest.mark.skipif(not APPLE_SILICON, reason="Apple Silicon required")
def test_convert_to_wav_missing_file():
    """Missing input path returns None."""
    assert convert_to_wav("/nonexistent/path.wav") is None


@pytest.mark.skipif(not APPLE_SILICON, reason="Apple Silicon required")
def test_convert_to_wav_already_correct(temp_dir):
    """WAV already 24 kHz mono is returned unchanged."""
    wav_path = os.path.join(temp_dir, "ok.wav")
    _make_wav(wav_path, 24000, 1)
    result = convert_to_wav(wav_path, sample_rate=24000)
    assert result == wav_path


@pytest.mark.skipif(not APPLE_SILICON or not _avfoundation_available(), reason="Apple Silicon with AVFoundation required for conversion")
def test_convert_to_wav_wrong_format_returns_new_path(temp_dir):
    """WAV with wrong sample rate is converted; returns path to new temp WAV."""
    wav_path = os.path.join(temp_dir, "44k.wav")
    _make_wav(wav_path, 44100, 1)
    result = convert_to_wav(wav_path, sample_rate=24000)
    assert result is not None
    assert result != wav_path
    assert os.path.exists(result)
    assert result.endswith(".wav")
    with wave.open(result, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000
        assert wf.getsampwidth() == 2
    if result != wav_path:
        try:
            os.remove(result)
        except OSError:
            pass


@pytest.mark.skipif(not APPLE_SILICON or not _avfoundation_available(), reason="Apple Silicon with AVFoundation required for conversion")
def test_convert_to_wav_custom_sample_rate(temp_dir):
    """Custom sample_rate is respected when input needs conversion."""
    wav_path = os.path.join(temp_dir, "16k.wav")
    _make_wav(wav_path, 16000, 1)
    result = convert_to_wav(wav_path, sample_rate=8000)
    assert result is not None
    assert os.path.exists(result)
    with wave.open(result, "rb") as wf:
        assert wf.getframerate() == 8000
    if result != wav_path and os.path.exists(result):
        try:
            os.remove(result)
        except OSError:
            pass
