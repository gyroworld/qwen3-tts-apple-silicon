"""Configuration and model/speaker presets. No app dependencies (only os)."""

import os

BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")

AUTO_PLAY = True
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20
MAX_TEXT_LENGTH = 10000

MODELS = {
    "1": {
        "name": "Custom Voice",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "mode": "custom",
        "output_subfolder": "CustomVoice",
        "description": "Preset speakers with emotion & speed control",
        "icon": "\U0001f399",
    },
    "2": {
        "name": "Voice Design",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "mode": "design",
        "output_subfolder": "VoiceDesign",
        "description": "Design a voice from a text description",
        "icon": "\U0001f3a8",
    },
    "3": {
        "name": "Voice Cloning",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mode": "clone_manager",
        "output_subfolder": "Clones",
        "description": "Clone any voice from a reference audio sample",
        "icon": "\U0001f9ec",
    },
}

SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}

EMOTION_PRESETS = {
    "1": ("Normal", "Normal tone"),
    "2": ("Sad", "Sad and crying, speaking slowly"),
    "3": ("Excited", "Excited and happy, speaking very fast"),
    "4": ("Angry", "Angry and shouting"),
    "5": ("Whisper", "Whispering quietly"),
    "6": ("Custom", None),
}

SPEED_PRESETS = {
    "1": ("Normal", 1.0),
    "2": ("Fast", 1.3),
    "3": ("Slow", 0.8),
}
