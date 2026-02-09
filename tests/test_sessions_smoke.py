"""Smoke tests: session and main menu imports (no MLX required for collection)."""

import pytest


def test_sessions_importable():
    """Sessions can be imported without loading MLX (lazy generate_audio)."""
    from qwen3_tts.sessions import run_custom_session, run_design_session, run_clone_manager
    assert callable(run_custom_session)
    assert callable(run_design_session)
    assert callable(run_clone_manager)
