"""Shared pytest fixtures (temp dirs, path overrides)."""

import os
import tempfile
import shutil

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory; yield path; cleanup after test."""
    d = tempfile.mkdtemp(prefix="qwen3_tts_test_")
    try:
        yield d
    finally:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def voices_dir(temp_dir):
    """A temp directory suitable as VOICES_DIR (e.g. for get_saved_voices tests)."""
    return temp_dir
