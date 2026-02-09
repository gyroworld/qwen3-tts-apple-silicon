"""Model path/download/load, audio save/convert, temp dirs. Depends on config and ui."""

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime

from huggingface_hub import snapshot_download
from rich.panel import Panel

from qwen3_tts import config
from qwen3_tts.ui import console, clear_screen, safe_line_input, normalize_whitespace

if sys.platform == "darwin":
    from qwen3_tts import apple_audio
else:
    apple_audio = None

BASE_OUTPUT_DIR = config.BASE_OUTPUT_DIR
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
MODELS_DIR = config.MODELS_DIR
SAMPLE_RATE = config.SAMPLE_RATE
FILENAME_MAX_LEN = config.FILENAME_MAX_LEN
AUTO_PLAY = config.AUTO_PLAY


def clean_path(user_input):
    """Sanitise a file path (handles drag-and-drop quoting / escaping)."""
    path = user_input.strip()
    if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
        path = path[1:-1]
    return path.replace("\\ ", " ")


def get_smart_path(folder_name):
    """Resolve the real model path, handling HuggingFace snapshot layouts."""
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full_path):
        return None
    snapshots_dir = os.path.join(full_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith('.')]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])
    return full_path


def ensure_model(info):
    """Check for local model; download from HuggingFace if missing."""
    path = get_smart_path(info["folder"])
    if path:
        return path

    repo_id = info["repo_id"]
    target_dir = os.path.join(MODELS_DIR, info["folder"])

    console.print()
    console.print(Panel(
        f"[bold]{info['name']}[/bold] model not found locally.\n\n"
            f"  Repo:  [cyan]{repo_id}[/cyan]\n"
            f"  Dest:  [muted]models/{info['folder']}/[/muted]\n\n"
            "[muted]Downloading from Hugging Face...[/muted]",
        title="[bold yellow]Downloading Model[/bold yellow]",
        border_style="yellow",
    ))

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
        )
    except KeyboardInterrupt:
        console.print("\n  [warning]Download cancelled.[/warning]")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        return None
    except Exception as e:
        console.print(f"  [error]Download failed:[/error] {e}")
        return None

    path = get_smart_path(info["folder"])
    if path:
        console.print(f"  [success]Download complete.[/success]")
    return path


def load_model_with_progress(model_path, model_name):
    """Load a TTS model with a Rich spinner, suppressing noisy library output."""
    from rich.panel import Panel
    noisy_loggers = ["transformers", "mlx_audio", "mlx", "tokenizers"]
    _logging = logging
    old_levels = {}
    for name in noisy_loggers:
        logger = _logging.getLogger(name)
        old_levels[name] = logger.level
        logger.setLevel(logging.CRITICAL)
    try:
        with console.status(
            f"[bold cyan]Loading {model_name}...[/bold cyan]", spinner="dots"
        ):
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                    from mlx_audio.tts.utils import load_model
                    model = load_model(model_path)
                console.print(f"  [success]\u2713 {model_name} loaded[/success]")
                return model
            except (OSError, RuntimeError, ValueError) as e:
                console.print(f"  [error]Failed to load model:[/error] {e}")
                return None
            except Exception as e:
                console.print(f"  [error]Unexpected model error:[/error] {e}")
                return None
    finally:
        for name, level in old_levels.items():
            _logging.getLogger(name).setLevel(level)


def make_temp_dir():
    return tempfile.mkdtemp(prefix="qwen3_tts_")


def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_audio_file(temp_folder, subfolder, text_snippet):
    """Move generated audio to the output directory and optionally play it."""
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    snippet = (text_snippet or "").strip()
    clean_text = (
        re.sub(r"[^\w\s-]", "", snippet)[:FILENAME_MAX_LEN]
        .strip()
        .replace(" ", "_")
        or "audio"
    )
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)
    if os.path.exists(final_path):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(final_path):
            final_path = os.path.join(save_path, f"{base}_{counter}{ext}")
            counter += 1
    source_file = os.path.join(temp_folder, "audio_000.wav")

    if os.path.exists(source_file):
        try:
            shutil.move(source_file, final_path)
        except (OSError, shutil.Error) as e:
            console.print(f"  [error]Save failed:[/error] {e}")
            cleanup_temp_dir(temp_folder)
            return
        rel_path = f"outputs/{subfolder}/{os.path.basename(final_path)}"
        console.print(f"  [success]\u2713 Saved:[/success] {rel_path}")

        if AUTO_PLAY:
            console.print("  [muted]\u266b Playing audio...[/muted]")
            try:
                subprocess.run(
                    ["afplay", final_path],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

        time.sleep(1)
        clear_screen()

    cleanup_temp_dir(temp_folder)


def get_text_input():
    """Prompt for text input; supports .txt drag-and-drop."""
    try:
        console.print()
        raw = safe_line_input(
            "[accent]Enter text[/accent] [muted](drag .txt file, or [bold]q[/bold] to go back)[/muted]: "
        )
        if raw is None:
            return None
        text = normalize_whitespace(raw)
        if not text or text.lower() in ("exit", "quit", "q"):
            return None
        if len(text) > MAX_TEXT_LENGTH:
            console.print(
                f"  [warning]Text too long ({len(text):,} chars). Max is {MAX_TEXT_LENGTH:,}.[/warning]"
            )
            return None

        clean_p = clean_path(raw)
        if os.path.exists(clean_p) and clean_p.endswith(".txt"):
            console.print(f"  [info]Reading:[/info] {os.path.basename(clean_p)}")
            try:
                with open(clean_p, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if len(content) > MAX_TEXT_LENGTH:
                    console.print(
                        f"  [warning]File too long ({len(content):,} chars). "
                        f"Max is {MAX_TEXT_LENGTH:,}.[/warning]"
                    )
                    return None
                console.print(f"  [muted]({len(content):,} characters loaded)[/muted]")
                return content if content else None
            except IOError as e:
                console.print(f"  [error]File read error:[/error] {e}")
                return None
        return text
    except KeyboardInterrupt:
        return None


def _ffmpeg_available():
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_audio_if_needed(input_path):
    """Convert an audio file to 24 kHz mono WAV if necessary.

    Uses ffmpeg when installed; on macOS falls back to built-in Apple conversion
    (afconvert) when ffmpeg is missing or conversion fails.
    """
    if not os.path.exists(input_path):
        console.print(f"  [error]File not found:[/error] {input_path}")
        return None

    _, ext = os.path.splitext(input_path)

    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, "rb") as f:
                if f.getnchannels() == 1 and f.getframerate() == SAMPLE_RATE:
                    return input_path
        except wave.Error:
            pass

    console.print(f"  [info]Converting {ext} \u2192 WAV...[/info]")

    if _ffmpeg_available():
        temp_file = tempfile.NamedTemporaryFile(
            prefix="qwen3_convert_",
            suffix=".wav",
            delete=False,
        )
        temp_wav = temp_file.name
        temp_file.close()
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", input_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", temp_wav,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            console.print("  [success]\u2713 Conversion complete[/success]")
            return temp_wav
        except (subprocess.CalledProcessError, FileNotFoundError):
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if sys.platform != "darwin" or apple_audio is None:
                console.print("  [error]Could not convert audio. Is ffmpeg installed?[/error]")
                return None
            # Fall through to Apple conversion on macOS

    if sys.platform == "darwin" and apple_audio is not None:
        result = apple_audio.convert_to_wav(input_path, sample_rate=SAMPLE_RATE)
        if result is not None:
            console.print("  [success]\u2713 Conversion complete[/success]")
            return result

    console.print(
        "  [error]Could not convert audio. On macOS, conversion uses ffmpeg or built-in afconvert.[/error]"
    )
    return None
