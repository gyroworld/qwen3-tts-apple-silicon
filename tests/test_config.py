"""Tests for qwen3_tts.config."""

import os

import pytest

from qwen3_tts import config


def test_path_constants_are_absolute():
    assert os.path.isabs(config.BASE_OUTPUT_DIR)
    assert os.path.isabs(config.MODELS_DIR)
    assert os.path.isabs(config.VOICES_DIR)
    assert config.BASE_OUTPUT_DIR.endswith("outputs")
    assert config.MODELS_DIR.endswith("models")
    assert config.VOICES_DIR.endswith("voices")


def test_models_has_expected_keys():
    assert set(config.MODELS.keys()) == {"1", "2", "3"}


def test_each_model_has_required_fields():
    required = {"name", "repo_id", "folder", "mode", "output_subfolder", "description", "icon"}
    for key, info in config.MODELS.items():
        assert info.keys() >= required, f"Model {key} missing fields"


def test_speaker_map_structure():
    assert "English" in config.SPEAKER_MAP
    assert "Chinese" in config.SPEAKER_MAP
    assert isinstance(config.SPEAKER_MAP["English"], list)
    assert len(config.SPEAKER_MAP["English"]) >= 1


def test_emotion_presets_keys():
    assert set(config.EMOTION_PRESETS.keys()) == {"1", "2", "3", "4", "5", "6"}
    for k, v in config.EMOTION_PRESETS.items():
        assert isinstance(v, tuple) and len(v) == 2


def test_speed_presets_keys():
    assert set(config.SPEED_PRESETS.keys()) == {"1", "2", "3"}
    for k, v in config.SPEED_PRESETS.items():
        assert isinstance(v, tuple) and len(v) == 2
        assert v[1] > 0


def test_numeric_constants():
    assert config.SAMPLE_RATE == 24000
    assert config.MAX_TEXT_LENGTH == 10000
    assert config.FILENAME_MAX_LEN == 20
    assert config.AUTO_PLAY in (True, False)
