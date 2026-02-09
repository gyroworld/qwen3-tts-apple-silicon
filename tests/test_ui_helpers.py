"""Tests for qwen3_tts.ui pure helpers (no TTY)."""

import pytest

from qwen3_tts.ui import normalize_whitespace, _rich_to_ansi


def test_normalize_whitespace_none():
    assert normalize_whitespace(None) == ""


def test_normalize_whitespace_empty():
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""


def test_normalize_whitespace_strips():
    assert normalize_whitespace("  hello  ") == "hello"
    assert normalize_whitespace("\thello\n") == "hello"


def test_rich_to_ansi_returns_non_empty():
    out = _rich_to_ansi("[bold]test[/bold]")
    assert isinstance(out, str)
    assert len(out) > 0


def test_rich_to_ansi_plain_text():
    out = _rich_to_ansi("plain")
    assert "plain" in out or len(out) >= 5
