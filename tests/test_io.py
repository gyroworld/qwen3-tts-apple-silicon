"""Tests for qwen3_tts.io path and save helpers (no model load)."""

import os

import pytest

from qwen3_tts.io import clean_path, get_smart_path


def test_clean_path_plain():
    assert clean_path("/path/to/file") == "/path/to/file"
    assert clean_path("file.txt") == "file.txt"


def test_clean_path_quoted_single():
    assert clean_path("'/path/with spaces'") == "/path/with spaces"


def test_clean_path_quoted_double():
    assert clean_path('"/path/with spaces"') == "/path/with spaces"


def test_clean_path_escaped_spaces():
    assert clean_path("/path/with\\ spaces") == "/path/with spaces"


def test_clean_path_strips():
    assert clean_path("  /path  ") == "/path"


def test_get_smart_path_missing(temp_dir):
    # Pass a custom models dir via monkeypatch or use a non-existent folder
    import qwen3_tts.io as io_module
    orig = io_module.MODELS_DIR
    try:
        io_module.MODELS_DIR = temp_dir
        assert get_smart_path("nonexistent") is None
    finally:
        io_module.MODELS_DIR = orig


def test_get_smart_path_direct_folder(temp_dir):
    """When folder exists with no snapshots/, returns folder path."""
    import qwen3_tts.io as io_module
    orig = io_module.MODELS_DIR
    try:
        io_module.MODELS_DIR = temp_dir
        folder = os.path.join(temp_dir, "my_model")
        os.makedirs(folder, exist_ok=True)
        assert get_smart_path("my_model") == folder
    finally:
        io_module.MODELS_DIR = orig


def test_get_smart_path_snapshots_layout(temp_dir):
    """When folder has snapshots/<hash>/, returns snapshot path."""
    import qwen3_tts.io as io_module
    orig = io_module.MODELS_DIR
    try:
        io_module.MODELS_DIR = temp_dir
        folder = os.path.join(temp_dir, "my_model")
        snapshots = os.path.join(folder, "snapshots", "abc123hash")
        os.makedirs(snapshots, exist_ok=True)
        result = get_smart_path("my_model")
        assert result is not None
        assert "snapshots" in result
        assert result.endswith("abc123hash") or "abc123hash" in result
    finally:
        io_module.MODELS_DIR = orig
