# Qwen3-TTS for Apple Silicon

Run **Qwen3-TTS** text-to-speech entirely on your Mac. No cloud services, no API keys — everything stays local on Apple Silicon.

Built on [MLX](https://github.com/ml-explore/mlx), Apple's framework for efficient machine-learning inference on M-series chips.

---

## Features

- **Custom Voice** — 9 built-in speakers across English, Chinese, Japanese, and Korean with emotion and speed control
- **Voice Design** — Create new voices from a text description (_"a warm, elderly British gentleman"_)
- **Voice Cloning** — Clone any voice from a short audio sample
- **Voice Manager** — Enroll, update, and delete saved voices
- **Auto-Transcription** — Optional macOS Speech framework integration for automatic reference transcripts
- **100% Offline** — Runs entirely on-device using quantised MLX models

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **Hardware** | Apple Silicon Mac (M1 / M2 / M3 / M4) |
| **OS** | macOS 13+ |
| **Python** | 3.10+ |
| **RAM** | ~4–6 GB free for 1.7B models |
| **ffmpeg** | Required for non-WAV audio conversion |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/kapi2800/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

brew install ffmpeg   # needed for audio format conversion
```

### 2. Download models

Download the models you need from HuggingFace and place them in a `models/` directory at the project root.

| Model | Use Case | Link |
|-------|----------|------|
| **CustomVoice** | Preset speakers + emotion/speed | [Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| **VoiceDesign** | Design voices from descriptions | [Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |
| **Base** | Voice cloning from audio | [Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |

Your directory should look like this:

```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
└── Qwen3-TTS-12Hz-1.7B-Base-8bit/
```

### 3. Run

```bash
source .venv/bin/activate
python app.py
```

The main menu shows available modes with a green dot for models that are detected:

```
 1  🎙️  Custom Voice    Preset speakers with emotion & speed control       ●
 2  🎨  Voice Design    Design a voice from a text description             ●
 3  🧬  Voice Cloning   Clone any voice from a reference audio sample      ●
 q      Exit
```

Press a key to select — no Enter needed.

---

## Usage Guide

### Custom Voice

Select a speaker, pick an emotion preset (or write your own), choose a speed, then start typing text to generate speech.

**Speakers:** Ryan, Aiden, Ethan, Chelsie, Serena, Vivian (EN) · Uncle_Fu, Dylan, Eric (ZH) · Ono_Anna (JA) · Sohee (KO)

**Emotions:** Normal · Sad · Excited · Angry · Whisper · Custom

**Speeds:** Normal (1.0x) · Fast (1.3x) · Slow (0.8x)

### Voice Design

Describe the voice you want and the model will synthesise it. Works best with specific descriptions:

> _"An excited young woman speaking quickly with an American accent"_

### Voice Cloning

The cloning manager lets you:

| Option | Description |
|--------|-------------|
| **Saved Voices** | Pick from previously enrolled voices |
| **Enroll New** | Add a voice from an audio sample + transcript |
| **Quick Clone** | One-shot clone without saving |
| **Delete / Update** | Manage your voice library |

For best results, use a clean 5–10 second audio clip with a matching transcript.

---

## Tips

- **Long text** — Drag a `.txt` file into the terminal instead of typing
- **Navigation** — Press `q`, `b`, or `Esc` at any prompt to go back
- **Auto-play** — Generated audio plays automatically via `afplay`
- **Transcription** — On macOS, the app can use Apple's built-in Speech framework to auto-transcribe reference audio for cloning
- **Output** — All generated files are saved to `outputs/` organised by mode

---

## Why MLX?

MLX models are specifically optimised for Apple Silicon. Compared to standard PyTorch inference:

| Metric | PyTorch | MLX |
|--------|---------|-----|
| **RAM** | 10+ GB | 2–3 GB |
| **CPU Temp** | 80–90°C | 40–50°C |

_Tested on M4 MacBook Air (fanless) with 1.7B 8-bit models._

MLX runs natively on the Apple Neural Engine and GPU — better performance with less heat and battery drain.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mlx_audio not found` | Activate the venv: `source .venv/bin/activate` |
| Model not found | Verify folder names in `models/` match exactly |
| Audio won't play | Check macOS sound settings; ensure `afplay` works |
| ffmpeg errors | Install with `brew install ffmpeg` |

---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Original model by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) — MLX framework for audio models
- [MLX Community](https://huggingface.co/mlx-community) — Pre-converted MLX models

---

**If this project helped you, please give it a star!**
