# Qwen3-TTS for Apple Silicon

Run **Qwen3-TTS** text-to-speech entirely on your Mac. No cloud services, no API keys — everything stays local on Apple Silicon.

---

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4) only — Intel Macs are not supported
- macOS 13+
- Python 3.10+
- ~4–6 GB free RAM for 1.7B models
- Optional: ffmpeg for non-WAV audio conversion; on macOS, built-in Apple conversion is used if ffmpeg is not installed

---

## Install and run

The app and its MLX stack require Apple Silicon; the steps below assume you are on an M-series Mac.

```bash
git clone https://github.com/gyroworld/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: ffmpeg for conversion (on macOS, built-in conversion is used if ffmpeg is not installed)
brew install ffmpeg

python app.py
```

Models are **downloaded automatically** the first time you choose each mode (Custom Voice, Voice Design, Voice Cloning). No manual download needed.

Reference: [CustomVoice](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) · [VoiceDesign](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) · [Base](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) (folder names: `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`, `Qwen3-TTS-12Hz-1.7B-Base-8bit`).

---

## Features

- **Custom Voice** — 9 built-in speakers (EN, ZH, JA, KO) with emotion and speed control
- **Voice Design** — Create voices from a text description
- **Voice Cloning** — Clone any voice from a short audio sample
- **Voice Manager** — Enroll, update, and delete saved voices
- **Auto-Transcription** — Optional macOS Speech framework for reference transcripts
- **100% Offline** — Runs on-device with quantised MLX models

---

## Usage

Run `python app.py` (from project root with venv active), choose a mode with **1**, **2**, or **3** — a green dot means the model is already present. Follow the prompts.

- Drag a `.txt` file into the terminal for long text instead of typing
- Press `q`, `b`, or `Esc` to go back at any prompt
- Generated files are saved to `outputs/` by mode

---

## Tips

- **Long text** — Drag a `.txt` file into the terminal instead of typing
- **Navigation** — Press `q`, `b`, or `Esc` at any prompt to go back
- **Auto-play** — Generated audio plays automatically via `afplay`
- **Transcription** — On macOS, the app can use Apple's Speech framework to auto-transcribe reference audio for cloning
- **Output** — All generated files are saved to `outputs/` organised by mode

---

## Why MLX?

MLX is optimised for Apple Silicon. Compared to standard PyTorch inference:

- **RAM** — ~2–3 GB (MLX) vs 10+ GB (PyTorch)
- **CPU temp** — ~40–50°C (MLX) vs 80–90°C (PyTorch)

_Tested on M4 MacBook Air (fanless) with 1.7B 8-bit models._ MLX runs natively on the Apple Neural Engine and GPU for better performance with less heat and battery drain.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App exits immediately / "runs only on Apple Silicon" | Use an Apple Silicon Mac (M1/M2/M3/M4); Intel Macs and non-macOS are not supported |
| `mlx_audio not found` | Activate the venv: `source .venv/bin/activate` |
| Model not found | Verify folder names in `models/` match exactly |
| Audio won't play | Check macOS sound settings; ensure `afplay` works |
| Conversion fails | The app uses ffmpeg when installed, or (on macOS) built-in Apple conversion. Try `brew install ffmpeg`, or ensure your audio format is supported |

---

## Testing

With the venv active, install with dev dependencies and run tests (no MLX or models required):

```bash
pip install -e ".[dev]"
pytest tests/
```

Or: `python -m pytest tests/`

---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Original model by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) — MLX framework for audio models
- [MLX Community](https://huggingface.co/mlx-community) — Pre-converted MLX models

---

If this project helped you, please give it a star!
