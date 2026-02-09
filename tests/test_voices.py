"""Tests for qwen3_tts.voices (get_saved_voices with temp VOICES_DIR)."""

import os

import pytest


def test_get_saved_voices_empty(voices_dir):
    import qwen3_tts.voices as voices_module
    orig = voices_module.VOICES_DIR
    try:
        voices_module.VOICES_DIR = voices_dir
        from qwen3_tts.voices import get_saved_voices
        assert get_saved_voices() == []
    finally:
        voices_module.VOICES_DIR = orig


def test_get_saved_voices_one_file(voices_dir):
    import qwen3_tts.voices as voices_module
    orig = voices_module.VOICES_DIR
    try:
        voices_module.VOICES_DIR = voices_dir
        with open(os.path.join(voices_dir, "alice.wav"), "w") as f:
            f.write("x")
        from qwen3_tts.voices import get_saved_voices
        assert get_saved_voices() == ["alice"]
    finally:
        voices_module.VOICES_DIR = orig


def test_get_saved_voices_ignores_non_wav(voices_dir):
    import qwen3_tts.voices as voices_module
    orig = voices_module.VOICES_DIR
    try:
        voices_module.VOICES_DIR = voices_dir
        with open(os.path.join(voices_dir, "alice.wav"), "w") as f:
            f.write("x")
        with open(os.path.join(voices_dir, "readme.txt"), "w") as f:
            f.write("x")
        from qwen3_tts.voices import get_saved_voices
        result = get_saved_voices()
        assert "alice" in result
        assert result == ["alice"]
    finally:
        voices_module.VOICES_DIR = orig


def test_get_saved_voices_sorted(voices_dir):
    import qwen3_tts.voices as voices_module
    orig = voices_module.VOICES_DIR
    try:
        voices_module.VOICES_DIR = voices_dir
        for name in ("zoe.wav", "alice.wav", "bob.wav"):
            with open(os.path.join(voices_dir, name), "w") as f:
                f.write("x")
        from qwen3_tts.voices import get_saved_voices
        assert get_saved_voices() == ["alice", "bob", "zoe"]
    finally:
        voices_module.VOICES_DIR = orig
