#!/usr/bin/env python3
"""
Qwen3-TTS Manager — A beautiful Rich-based CLI for text-to-speech on Apple Silicon.
Run: python app.py
"""

import os
import sys
import shutil
import time
import wave
import gc
import re
import subprocess
import warnings
import tempfile
from datetime import datetime
import contextlib

from io import StringIO
from huggingface_hub import snapshot_download

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress transformers / tokenizers warnings that bypass Python's warnings module
# (e.g. Mistral regex, unregistered model_type). These are harmless for TTS.
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme
from rich import box

# ─── Theme ────────────────────────────────────────────────────

custom_theme = Theme({
    "info": "cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "muted": "grey62",
    "accent": "bold cyan",
})

console = Console(theme=custom_theme, file=sys.stdout)

# ─── MLX Audio Import ─────────────────────────────────────────

try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    console.print(Panel(
        "[error]'mlx_audio' library not found.[/error]\n\n"
        "Run: [accent]source .venv/bin/activate[/accent]",
        title="[bold red]Import Error[/bold red]",
        border_style="red",
    ))
    sys.exit(1)

# ─── Optional: Apple Speech (macOS built-in transcription) ─────

APPLE_SPEECH_AVAILABLE = False
if sys.platform == "darwin":
    try:
        import Foundation
        from Speech import (
            SFSpeechRecognizer,
            SFSpeechURLRecognitionRequest,
        )
        APPLE_SPEECH_AVAILABLE = True
    except Exception:
        pass

# ─── Configuration ────────────────────────────────────────────

BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")

AUTO_PLAY = True
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20
MAX_TEXT_LENGTH = 10000

# ─── Pro Models Only (1.7B) ──────────────────────────────────

MODELS = {
    "1": {
        "name": "Custom Voice",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "mode": "custom",
        "output_subfolder": "CustomVoice",
        "description": "Preset speakers with emotion & speed control",
        "icon": "\U0001f399",          # 🎙
    },
    "2": {
        "name": "Voice Design",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "mode": "design",
        "output_subfolder": "VoiceDesign",
        "description": "Design a voice from a text description",
        "icon": "\U0001f3a8",          # 🎨
    },
    "3": {
        "name": "Voice Cloning",
        "repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mode": "clone_manager",
        "output_subfolder": "Clones",
        "description": "Clone any voice from a reference audio sample",
        "icon": "\U0001f9ec",          # 🧬
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


# ─── Utility Functions ────────────────────────────────────────

def _rich_to_ansi(markup):
    """Convert Rich markup to an ANSI string for prompt_toolkit prompts."""
    buf = StringIO()
    c = Console(file=buf, theme=custom_theme, force_terminal=True)
    c.print(markup, end="", highlight=False)
    return buf.getvalue()


def safe_line_input(prompt_markup):
    """Read a full line of input with backspace protection for the prompt.

    Uses prompt_toolkit for robust input handling.  Backspace can never erase
    the prompt text.  Supports normal typing, backspace, Ctrl-C, Ctrl-U
    (clear typed text), Ctrl-D (go back), and Enter.
    Returns the entered string, or ``None`` on Ctrl-D / EOF.
    """
    if not sys.stdin.isatty():
        try:
            return Prompt.ask(prompt_markup)
        except (EOFError, KeyboardInterrupt):
            return None

    try:
        return pt_prompt(ANSI(_rich_to_ansi(prompt_markup)))
    except EOFError:
        return None
    except KeyboardInterrupt:
        raise


def instant_menu_choice(prompt_markup, valid_keys):
    """Show a styled prompt and return a key immediately on press.

    Uses prompt_toolkit with key bindings so that the user's selection is
    returned as soon as a valid key is pressed — no Enter needed.

    Returns the selected key (lowercase) if it is in *valid_keys*.
    Returns ``None`` when Escape is pressed (go back).
    """
    # Non-TTY fallback: use Rich Prompt with Enter-based input.
    if not sys.stdin.isatty():
        valid = {str(k).lower() for k in valid_keys}
        while True:
            try:
                response = Prompt.ask(prompt_markup)
            except (EOFError, KeyboardInterrupt):
                return None
            if response is None:
                return None
            lower = response.strip().lower()
            if lower in valid:
                return lower
            console.print("  [warning]Invalid selection.[/warning]")

    # Build key bindings for instant single-key selection.
    kb = KeyBindings()

    for _key in valid_keys:
        @kb.add(_key)
        def _handle(event, k=_key):
            event.app.exit(result=k)
        if _key.isalpha():
            @kb.add(_key.upper())
            def _handle_upper(event, k=_key):
                event.app.exit(result=k)

    @kb.add("escape")
    def _escape(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(result="__interrupt__")

    # prompt_toolkit handles both prompt display and input atomically.
    result = pt_prompt(ANSI(_rich_to_ansi(prompt_markup)), key_bindings=kb)

    if result == "__interrupt__":
        console.print()
        raise KeyboardInterrupt

    if result is not None:
        # prompt_toolkit adds a newline; move back up to keep selection on same line
        sys.stdout.write("\x1b[1A\r\x1b[2K")
        console.print(f"{prompt_markup}[bold cyan]{result}[/bold cyan]")

    return result


def clear_screen():
    """Clear the terminal screen."""
    console.clear()


def clean_memory():
    gc.collect()


def make_temp_dir():
    return tempfile.mkdtemp(prefix="qwen3_tts_")


def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def confirm_overwrite(label):
    try:
        response = Prompt.ask(
            f"[warning]{label} already exists. Overwrite?[/warning] [muted](y/n)[/muted]",
            choices=["y", "n"],
            default="n",
            show_choices=False,
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() == "y"


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


def clean_path(user_input):
    """Sanitise a file path (handles drag-and-drop quoting / escaping)."""
    path = user_input.strip()
    if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
        path = path[1:-1]
    return path.replace("\\ ", " ")


def get_saved_voices():
    """Return a sorted list of enrolled voice names."""
    if not os.path.exists(VOICES_DIR):
        return []
    voices = [f.replace(".wav", "") for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    return sorted(voices)


# ─── Apple Speech transcription (macOS built-in) ───────────────

def transcribe_wav_with_apple_speech(wav_path: str) -> str | None:
    """Transcribe a WAV file using macOS built-in Speech framework.

    Returns the transcribed text, or None if unavailable, denied, or failed.
    """
    if not APPLE_SPEECH_AVAILABLE or not os.path.exists(wav_path):
        return None

    # SFSpeechRecognizerAuthorizationStatusAuthorized = 3
    AUTHORIZED = 3
    auth_result = [None]

    def auth_callback(status):
        auth_result[0] = status

    SFSpeechRecognizer.requestAuthorization_(auth_callback)
    run_loop = Foundation.NSRunLoop.currentRunLoop()
    deadline = time.time() + 60
    while auth_result[0] is None and time.time() < deadline:
        run_loop.runUntilDate_(
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
    if auth_result[0] != AUTHORIZED:
        return None

    locale = Foundation.NSLocale.localeWithLocaleIdentifier_("en-US")
    recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
    if recognizer is None or not recognizer.isAvailable():
        return None

    url = Foundation.NSURL.fileURLWithPath_(wav_path)
    request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    if request is None:
        return None

    transcript_result = [None]
    transcript_error = [None]

    def result_handler(result, error):
        if error is not None:
            transcript_error[0] = error
            return
        if result is not None and result.isFinal():
            trans = result.bestTranscription()
            if trans is not None:
                transcript_result[0] = trans.formattedString()

    task = recognizer.recognitionTaskWithRequest_resultHandler_(request, result_handler)
    if task is None:
        return None

    deadline = time.time() + 60
    while transcript_result[0] is None and transcript_error[0] is None and time.time() < deadline:
        run_loop.runUntilDate_(
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

    text = transcript_result[0]
    return (text.strip() or None) if text else None


def _offer_apple_transcribe(wav_path: str, prompt_msg: str) -> str | None:
    """If Apple Speech is available, ask user and transcribe WAV. Returns transcript or None."""
    if not APPLE_SPEECH_AVAILABLE:
        return None
    choice = instant_menu_choice(
        f"  [accent]{prompt_msg}[/accent] [muted](y/n)[/muted]: ",
        {"y", "n"},
        )
    if choice != "y":
        return None
    console.print("  [info]Transcribing with Apple...[/info]")
    result = transcribe_wav_with_apple_speech(wav_path)
    if result:
        console.print(f"  [success]\u2713 Transcript:[/success] [muted]{result[:80]}{'...' if len(result) > 80 else ''}[/muted]")
    else:
        console.print("  [warning]Transcription unavailable or denied.[/warning]")
    return result


# ─── Rich-enhanced helpers ────────────────────────────────────

def normalize_whitespace(s):
    """Return string with leading/trailing whitespace removed; None and empty become empty string."""
    if s is None:
        return ""
    return s.strip()


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


def load_model_with_progress(model_path, model_name):
    """Load a TTS model with a Rich spinner, suppressing noisy library output."""
    with console.status(
        f"[bold cyan]Loading {model_name}...[/bold cyan]", spinner="dots"
    ):
        try:
            # Suppress noisy library output at the logging level instead of
            # redirecting stdout/stderr (which breaks the Rich spinner).
            noisy_loggers = ["transformers", "mlx_audio", "mlx", "tokenizers"]
            old_levels = {}
            for name in noisy_loggers:
                logger = _logging.getLogger(name)
                old_levels[name] = logger.level
                logger.setLevel(_logging.CRITICAL)
            try:
                # Redirect stderr only to suppress bare print() calls from
                # libraries.  stdout stays untouched so the Rich spinner renders.
                with open(os.devnull, "w") as devnull, \
                     contextlib.redirect_stderr(devnull):
                    model = load_model(model_path)
            finally:
                for name, level in old_levels.items():
                    _logging.getLogger(name).setLevel(level)
            console.print(f"  [success]\u2713 {model_name} loaded[/success]")
            return model
        except (OSError, RuntimeError, ValueError) as e:
            console.print(f"  [error]Failed to load model:[/error] {e}")
            return None
        except Exception as e:
            console.print(f"  [error]Unexpected model error:[/error] {e}")
            return None


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


def convert_audio_if_needed(input_path):
    """Convert an audio file to 24 kHz mono WAV if necessary."""
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

    temp_file = tempfile.NamedTemporaryFile(
        prefix="qwen3_convert_",
        suffix=".wav",
        delete=False,
    )
    temp_wav = temp_file.name
    temp_file.close()
    console.print(f"  [info]Converting {ext} \u2192 WAV...[/info]")

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", input_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", temp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        console.print("  [success]\u2713 Conversion complete[/success]")
        return temp_wav
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("  [error]Could not convert audio. Is ffmpeg installed?[/error]")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return None


# ─── Banner ───────────────────────────────────────────────────

def print_banner():
    """Display a compact app banner."""
    title = Text()
    title.append("Q", style="bold magenta")
    title.append("W", style="bold cyan")
    title.append("E", style="bold green")
    title.append("N", style="bold yellow")
    title.append("3", style="bold red")
    title.append("-", style="bold white")
    title.append("T", style="bold magenta")
    title.append("T", style="bold cyan")
    title.append("S", style="bold green")
    title.append("  ", style="")
    title.append("Apple Silicon \u00b7 MLX \u00b7 Pro", style="grey62")
    console.print(Rule(title, style="bright_cyan"))


# ─── Session: Custom Voice ────────────────────────────────────

def run_custom_session(model_key):
    info = MODELS[model_key]
    model_path = ensure_model(info)
    if not model_path:
        return

    model = load_model_with_progress(model_path, info["name"])
    if not model:
        return

    clear_screen()

    # ── Speaker selection ─────────────────────────────────────
    console.print()
    console.print(Rule("[bold magenta]Custom Voice Setup[/bold magenta]", style="magenta"))
    console.print()

    all_speakers = [n for names in SPEAKER_MAP.values() for n in names]
    sections = []
    idx = 1
    for lang, names in SPEAKER_MAP.items():
        entries = "  ".join(
            f"[bold cyan]{idx + i}[/bold cyan] [cyan]{n}[/cyan]"
            for i, n in enumerate(names)
        )
        sections.append(f"  [bold yellow]{lang}[/bold yellow]\n  {entries}")
        idx += len(names)
    console.print(Panel(
        "\n\n".join(sections),
        title="[bold cyan]Available Speakers[/bold cyan]",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    ))
    console.print()

    while True:
        speaker_raw = Prompt.ask(
            "[accent]Select speaker[/accent] [muted](number or name, [bold]b[/bold] to go back)[/muted]",
        )
        choice = speaker_raw.strip()
        if choice.lower() in ("b", "back", "q", "quit", "exit"):
            clean_memory()
            return
        if not choice:
            console.print("  [warning]Please enter a speaker number or name.[/warning]")
            continue
        # Try numeric selection first
        try:
            num = int(choice)
            if 1 <= num <= len(all_speakers):
                speaker = all_speakers[num - 1]
                console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
                break
            else:
                console.print(f"  [warning]Enter a number between 1 and {len(all_speakers)}.[/warning]")
                continue
        except ValueError:
            pass
        # Fall back to name matching
        if choice in all_speakers:
            speaker = choice
            console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
            break
        console.print("  [warning]Unknown speaker \u2014 try again.[/warning]")
        continue

    # ── Emotion selection ─────────────────────────────────────
    console.print()
    emo_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    emo_table.add_column("Key", style="bold cyan", justify="right")
    emo_table.add_column("Emotion", style="white")
    emo_table.add_column("Description", style="grey62")
    for k, (name, desc) in EMOTION_PRESETS.items():
        emo_table.add_row(k, name, desc or "[italic]you describe it[/italic]")
    emo_table.add_row("[muted]b[/muted]", "[muted]Back[/muted]", "[muted]Return to main menu[/muted]")
    console.print(
        Panel(emo_table, title="[bold]Emotion[/bold]", border_style="magenta", expand=False)
    )

    emotion_choice = instant_menu_choice(
        "[accent]Select emotion[/accent]: ",
        set(EMOTION_PRESETS.keys()) | {"b"},
    )
    if emotion_choice is None or emotion_choice == "b":
        clean_memory()
        return
    if emotion_choice == "6":
        base_instruct = normalize_whitespace(
            Prompt.ask("  [accent]Describe the emotion[/accent]")
        ) or "Normal tone"
    else:
        _, base_instruct = EMOTION_PRESETS[emotion_choice]
    console.print(f"  [success]\u2713 Emotion:[/success] {base_instruct}")

    # ── Speed selection ───────────────────────────────────────
    console.print()
    spd_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    spd_table.add_column("Key", style="bold cyan", justify="right")
    spd_table.add_column("Speed", style="white")
    spd_table.add_column("Value", style="grey62")
    for k, (name, val) in SPEED_PRESETS.items():
        spd_table.add_row(k, name, f"{val}x")
    spd_table.add_row("[muted]b[/muted]", "[muted]Back[/muted]", "[muted]Return to main menu[/muted]")
    console.print(
        Panel(spd_table, title="[bold]Speed[/bold]", border_style="magenta", expand=False)
    )

    speed_choice = instant_menu_choice(
        "[accent]Select speed[/accent]: ",
        set(SPEED_PRESETS.keys()) | {"b"},
    )
    if speed_choice is None or speed_choice == "b":
        clean_memory()
        return
    _, speed = SPEED_PRESETS[speed_choice]
    console.print(f"  [success]\u2713 Speed:[/success] {speed}x")

    # ── Generation loop ───────────────────────────────────────
    console.print()
    console.print(Rule("[bold green]Ready to Generate[/bold green]", style="green"))

    while True:
        text = get_text_input()
        if text is None:
            break
        if not text:
            console.print("  [warning]No text entered. Type something or press q to go back.[/warning]")
            continue
        temp_dir = make_temp_dir()
        try:
            generate_audio(
                model=model,
                text=text,
                voice=speaker.lower(),
                instruct=base_instruct,
                speed=speed,
                output_path=temp_dir,
            )
            console.print()
            save_audio_file(temp_dir, info["output_subfolder"], text)
        except Exception as e:
            console.print(f"  [error]Generation error:[/error] {e}")
        finally:
            cleanup_temp_dir(temp_dir)

    clean_memory()


# ─── Session: Voice Design ────────────────────────────────────

def run_design_session(model_key):
    info = MODELS[model_key]
    model_path = ensure_model(info)
    if not model_path:
        return

    model = load_model_with_progress(model_path, info["name"])
    if not model:
        return

    clear_screen()
    console.print()
    console.print(Rule("[bold magenta]Voice Design[/bold magenta]", style="magenta"))
    console.print()

    console.print(Panel(
        "[grey62]Describe the voice you want to create. Be specific about:[/grey62]\n\n"
        "  [cyan]\u2022[/cyan] Gender, age, tone\n"
        "  [cyan]\u2022[/cyan] Speaking style and pace\n"
        "  [cyan]\u2022[/cyan] Accent or language characteristics\n\n"
        "[grey62]Example:[/grey62] [italic]\"A warm, elderly British gentleman speaking slowly and thoughtfully\"[/italic]",
        title="[bold]Tips[/bold]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    instruct_raw = Prompt.ask("[accent]Describe the voice[/accent] [muted](or [bold]b[/bold] to go back)[/muted]")
    instruct = instruct_raw.strip()
    if instruct.lower() in ("b", "back", "q", "quit", "exit"):
        clean_memory()
        return
    if not instruct:
        console.print("  [warning]No description provided.[/warning]")
        return
    console.print(f"  [success]\u2713 Voice description set[/success]")

    # ── Generation loop ───────────────────────────────────────
    console.print()
    console.print(Rule("[bold green]Ready to Generate[/bold green]", style="green"))

    while True:
        text = get_text_input()
        if text is None:
            break
        if not text:
            console.print("  [warning]No text entered. Type something or press q to go back.[/warning]")
            continue
        temp_dir = make_temp_dir()
        try:
            generate_audio(
                model=model,
                text=text,
                instruct=instruct,
                output_path=temp_dir,
            )
            console.print()
            save_audio_file(temp_dir, info["output_subfolder"], text)
        except Exception as e:
            console.print(f"  [error]Generation error:[/error] {e}")
        finally:
            cleanup_temp_dir(temp_dir)

    clean_memory()


# ─── Session: Voice Cloning ───────────────────────────────────

def enroll_new_voice():
    """Enroll a new voice for later cloning."""
    clear_screen()
    console.print()
    console.print(Rule("[bold magenta]Enroll New Voice[/bold magenta]", style="magenta"))
    console.print()

    name = Prompt.ask("[accent]Voice name[/accent] [muted](e.g. Boss, Mom)[/muted]")
    if not name.strip():
        console.print("  [warning]No name provided.[/warning]")
        return

    safe_name = re.sub(r"[^\w\s-]", "", name.strip()).strip().replace(" ", "_") or "voice"

    console.print()
    ref_input = Prompt.ask("[accent]Drag & drop reference audio file[/accent]")
    raw_path = clean_path(ref_input)

    if len(raw_path) > 300 or "\n" in raw_path:
        console.print("  [error]Input too long or invalid.[/error]")
        return

    clean_wav_path = convert_audio_if_needed(raw_path)
    if not clean_wav_path:
        return

    console.print()
    console.print(Panel(
        "[white]For best cloning quality, type [bold]exactly[/bold] what the person says in the audio.\n"
        "You can also drag & drop a .txt file containing the transcript.[/white]",
        border_style="yellow",
        expand=False,
    ))
    ref_text_input = Prompt.ask("[accent]Transcript[/accent]")
    clean_ref = clean_path(ref_text_input.strip())
    if os.path.exists(clean_ref) and clean_ref.endswith(".txt"):
        try:
            with open(clean_ref, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
            console.print(f"  [info]Read transcript from:[/info] {os.path.basename(clean_ref)}")
            console.print(f"  [muted]({len(ref_text):,} characters loaded)[/muted]")
        except IOError as e:
            console.print(f"  [error]File read error:[/error] {e}")
            return
    else:
        ref_text = ref_text_input.strip()
        if not ref_text and APPLE_SPEECH_AVAILABLE:
            console.print()
            apple_result = _offer_apple_transcribe(
                clean_wav_path,
                "Transcribe with Apple?",
            )
            if apple_result:
                ref_text = apple_result

    os.makedirs(VOICES_DIR, exist_ok=True)
    target_wav = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    target_txt = os.path.join(VOICES_DIR, f"{safe_name}.txt")
    if os.path.exists(target_wav) or os.path.exists(target_txt):
        if not confirm_overwrite(f"Voice '{safe_name}'"):
            if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
                os.remove(clean_wav_path)
            console.print("  [muted]Cancelled.[/muted]")
            return

    try:
        shutil.copy(clean_wav_path, target_wav)
        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(ref_text)
    except OSError as e:
        console.print(f"  [error]Could not save voice files:[/error] {e}")
        if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
            os.remove(clean_wav_path)
        return

    if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
        os.remove(clean_wav_path)

    console.print(f"\n  [success]\u2713 Voice '{safe_name}' enrolled successfully![/success]")
    time.sleep(1)
    clear_screen()


def _pick_saved_voice(action_label):
    """Show a list of saved voices and let the user pick one. Returns the name or None."""
    saved = get_saved_voices()
    if not saved:
        console.print(
            "  [warning]No saved voices found. Enroll a voice first.[/warning]"
        )
        return None

    console.print()
    voice_table = Table(
        title="Saved Voices",
        box=box.ROUNDED,
        border_style="green",
        title_style="bold green",
    )
    voice_table.add_column("#", style="bold cyan", justify="right")
    voice_table.add_column("Name", style="white")
    voice_table.add_column("Transcript", style="grey62", justify="center")
    for i, v in enumerate(saved):
        has_txt = (
            "[green]\u2713[/green]"
            if os.path.exists(os.path.join(VOICES_DIR, f"{v}.txt"))
            else "[grey62]\u2014[/grey62]"
        )
        voice_table.add_row(str(i + 1), v, has_txt)
    console.print(voice_table)
    console.print()

    try:
        response = Prompt.ask(
            f"[accent]{action_label}[/accent] [muted](or [bold]b[/bold] to go back)[/muted]"
        )
        if response.strip().lower() in ("b", "back", "q", "quit", "exit"):
            return None
        idx = int(response) - 1
        if idx < 0 or idx >= len(saved):
            console.print("  [error]Invalid selection.[/error]")
            return None
        return saved[idx]
    except (ValueError, IndexError):
        console.print("  [error]Invalid selection.[/error]")
        return None


def delete_voice():
    """Delete a previously enrolled voice."""
    clear_screen()
    console.print()
    console.print(Rule("[bold red]Delete Voice[/bold red]", style="red"))

    name = _pick_saved_voice("Pick voice to delete")
    if name is None:
        return

    confirm = instant_menu_choice(
        f"  [warning]Delete '{name}'?[/warning] [muted](y/n)[/muted]: ",
        {"y", "n"},
    )
    if confirm is None or confirm != "y":
        console.print("  [muted]Cancelled.[/muted]")
        return

    wav_path = os.path.join(VOICES_DIR, f"{name}.wav")
    txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
    if os.path.exists(wav_path):
        os.remove(wav_path)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    console.print(f"  [success]\u2713 Voice '{name}' deleted.[/success]")
    time.sleep(1)
    clear_screen()


def update_voice():
    """Re-enroll a saved voice with a new audio sample and transcript."""
    clear_screen()
    console.print()
    console.print(Rule("[bold yellow]Update Voice[/bold yellow]", style="yellow"))

    name = _pick_saved_voice("Pick voice to update")
    if name is None:
        return

    console.print(f"\n  [info]Re-enrolling '[bold]{name}[/bold]'...[/info]\n")

    ref_input = Prompt.ask("[accent]Drag & drop new reference audio file[/accent]")
    raw_path = clean_path(ref_input)

    if len(raw_path) > 300 or "\n" in raw_path:
        console.print("  [error]Input too long or invalid.[/error]")
        return

    clean_wav_path = convert_audio_if_needed(raw_path)
    if not clean_wav_path:
        return

    console.print()
    console.print(Panel(
        "[white]For best cloning quality, type [bold]exactly[/bold] what the person says in the audio.\n"
        "You can also drag & drop a .txt file containing the transcript.[/white]",
        border_style="yellow",
        expand=False,
    ))
    ref_text_input = Prompt.ask("[accent]Transcript[/accent]")
    clean_ref = clean_path(ref_text_input.strip())
    if os.path.exists(clean_ref) and clean_ref.endswith(".txt"):
        try:
            with open(clean_ref, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
            console.print(f"  [info]Read transcript from:[/info] {os.path.basename(clean_ref)}")
        except IOError as e:
            console.print(f"  [error]File read error:[/error] {e}")
            return
    else:
        ref_text = ref_text_input.strip()
        if not ref_text and APPLE_SPEECH_AVAILABLE:
            console.print()
            apple_result = _offer_apple_transcribe(
                clean_wav_path,
                "Transcribe with Apple?",
            )
            if apple_result:
                ref_text = apple_result

    target_wav = os.path.join(VOICES_DIR, f"{name}.wav")
    target_txt = os.path.join(VOICES_DIR, f"{name}.txt")

    try:
        shutil.copy(clean_wav_path, target_wav)
        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(ref_text)
    except OSError as e:
        console.print(f"  [error]Could not update voice files:[/error] {e}")
        if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
            os.remove(clean_wav_path)
        return

    if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
        os.remove(clean_wav_path)

    console.print(f"\n  [success]\u2713 Voice '{name}' updated successfully![/success]")
    time.sleep(1)
    clear_screen()


def run_clone_manager(model_key):
    clear_screen()
    console.print()
    console.print(Rule("[bold magenta]Voice Cloning[/bold magenta]", style="magenta"))
    console.print()

    # Sub-menu
    menu_table = Table(box=box.ROUNDED, border_style="cyan", show_header=False, padding=(0, 2))
    menu_table.add_column("Key", style="bold cyan", justify="right")
    menu_table.add_column("Option", style="bold white")
    menu_table.add_column("Description", style="grey62")
    menu_table.add_row("1", "Saved Voices", "Pick from previously enrolled voices")
    menu_table.add_row("2", "Enroll New Voice", "Add a new voice from an audio sample")
    menu_table.add_row("3", "Quick Clone", "One-shot clone from any audio file")
    menu_table.add_row("4", "Delete Voice", "Remove a saved voice")
    menu_table.add_row("5", "Update Voice", "Re-enroll a voice with new audio")
    menu_table.add_row("b", "Back", "Return to main menu")
    console.print(menu_table)
    console.print()

    sub_choice = instant_menu_choice(
        "[accent]Select option[/accent]: ",
        {"1", "2", "3", "4", "5", "b"},
    )

    if sub_choice is None or sub_choice == "b":
        return
    if sub_choice == "2":
        enroll_new_voice()
        return
    if sub_choice == "4":
        delete_voice()
        return
    if sub_choice == "5":
        update_voice()
        return

    # Load the base model
    info = MODELS[model_key]
    model_path = ensure_model(info)
    if not model_path:
        return

    model = load_model_with_progress(model_path, "Base Model")
    if not model:
        return

    clear_screen()

    ref_audio, ref_text = None, None
    temp_ref_audio = False

    # ── Option 1: Saved Voices ────────────────────────────────
    if sub_choice == "1":
        saved = get_saved_voices()
        if not saved:
            console.print(
                "  [warning]No saved voices found. Enroll a voice first (option 2).[/warning]"
            )
            clean_memory()
            return

        console.print()
        voice_table = Table(
            title="Saved Voices",
            box=box.ROUNDED,
            border_style="green",
            title_style="bold green",
        )
        voice_table.add_column("#", style="bold cyan", justify="right")
        voice_table.add_column("Name", style="white")
        voice_table.add_column("Transcript", style="grey62", justify="center")
        for i, v in enumerate(saved):
            has_txt = (
                "[green]\u2713[/green]"
                if os.path.exists(os.path.join(VOICES_DIR, f"{v}.txt"))
                else "[grey62]\u2014[/grey62]"
            )
            voice_table.add_row(str(i + 1), v, has_txt)
        console.print(voice_table)
        console.print()

        # Single keypress (no Enter) when 9 or fewer voices; otherwise Prompt for 10+
        if len(saved) <= 9:
            valid_keys = {str(i + 1) for i in range(len(saved))} | {"b", "q"}
            response = instant_menu_choice(
                "[accent]Pick voice number[/accent] [muted](or [bold]b[/bold] to go back)[/muted]: ",
                valid_keys,
            )
            if response is None or response in ("b", "q"):
                clean_memory()
                return
            idx = int(response) - 1
        else:
            try:
                response = Prompt.ask(
                    "[accent]Pick voice number[/accent] [muted](or [bold]b[/bold] to go back)[/muted]"
                )
                if response.strip().lower() in ("b", "back", "q", "quit", "exit"):
                    clean_memory()
                    return
                idx = int(response) - 1
                if idx < 0 or idx >= len(saved):
                    console.print("  [error]Invalid selection.[/error]")
                    return
            except ValueError:
                console.print("  [error]Invalid selection.[/error]")
                return
        name = saved[idx]
        ref_audio = os.path.join(VOICES_DIR, f"{name}.wav")
        if not os.path.exists(ref_audio):
            console.print("  [error]Selected voice file is missing.[/error]")
            clean_memory()
            return
        txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip() or "."
        else:
            ref_text = "."
        console.print(f"  [success]\u2713 Loaded voice:[/success] {name}")
        clone_subfolder = os.path.join("Clones", name)

    # ── Option 3: Quick Clone ─────────────────────────────────
    elif sub_choice == "3":
        console.print()
        ref_input = Prompt.ask("[accent]Drag & drop reference audio[/accent]")
        raw_path = clean_path(ref_input.strip())
        ref_audio = convert_audio_if_needed(raw_path)
        if not ref_audio:
            return
        temp_ref_audio = ref_audio != raw_path
        console.print()
        console.print(Panel(
            "[white]For best cloning quality, type [bold]exactly[/bold] what the person says in the audio.\n"
            "You can also drag & drop a .txt file containing the transcript.[/white]",
            border_style="yellow",
            expand=False,
        ))
        ref_text_input = Prompt.ask(
            "[accent]Transcript[/accent] [muted](optional \u2014 drag .txt or press Enter to skip)[/muted]",
            default="",
        )
        clean_ref = clean_path(ref_text_input.strip())
        if clean_ref and os.path.exists(clean_ref) and clean_ref.endswith(".txt"):
            try:
                with open(clean_ref, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
                console.print(f"  [info]Read transcript from:[/info] {os.path.basename(clean_ref)}")
            except IOError as e:
                console.print(f"  [error]File read error:[/error] {e}")
                ref_text = "."
        else:
            ref_text = ref_text_input.strip() or "."
        if ref_text == "." and APPLE_SPEECH_AVAILABLE:
            console.print()
            apple_result = _offer_apple_transcribe(
                ref_audio,
                "Transcribe with Apple's built-in speech recognition?",
            )
            if apple_result:
                ref_text = apple_result
        clone_subfolder = os.path.join("Clones", "QuickClones")
    else:
        return

    # ── Generation loop ───────────────────────────────────────
    # When we have no transcript (placeholder "."), offer Apple Speech if available.
    if ref_text == "." and APPLE_SPEECH_AVAILABLE:
        console.print()
        apple_result = _offer_apple_transcribe(
            ref_audio,
            "Transcribe this voice with Apple?",
        )
        if apple_result:
            ref_text = apple_result
    # Never pass None/empty ref_text: mlx_audio would then run STT (Whisper), which
    # downloads ~1.6GB and can fail with missing multilingual.tiktoken in the package.
    ref_text = ref_text or "."
    console.print()
    voice_label = os.path.basename(str(ref_audio))
    console.print(Rule(f"[bold green]Cloning from: {voice_label}[/bold green]", style="green"))

    while True:
        text = get_text_input()
        if text is None:
            break
        if not text:
            console.print("  [warning]No text entered. Type something or press q to go back.[/warning]")
            continue
        temp_dir = make_temp_dir()
        try:
            generate_audio(
                model=model,
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_path=temp_dir,
            )
            console.print()
            save_audio_file(temp_dir, clone_subfolder, text)
        except Exception as e:
            console.print(f"  [error]Generation error:[/error] {e}")
        finally:
            cleanup_temp_dir(temp_dir)

    if temp_ref_audio and ref_audio and os.path.exists(ref_audio):
        os.remove(ref_audio)
    clean_memory()


# ─── Main Menu ────────────────────────────────────────────────

def main_menu():
    clear_screen()
    console.print()
    print_banner()

    # Compact vertical menu table
    menu = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    menu.add_column("Key", style="bold cyan", justify="center", width=3, no_wrap=True)
    menu.add_column("Icon", style="white", justify="center", width=2, no_wrap=True)
    menu.add_column("Mode", style="bold white", width=16, no_wrap=True)
    menu.add_column("Description", style="grey62", width=50, no_wrap=True)
    menu.add_column("Status", justify="center", width=3, no_wrap=True)

    for key, info in MODELS.items():
        available = get_smart_path(info["folder"]) is not None
        dot = "[green]\u25cf[/green]" if available else "[red]\u25cb[/red]"
        menu.add_row(key, info["icon"], info["name"], info["description"], dot)

    menu.add_row("q", "", "[muted]Exit[/muted]", "", "")
    console.print(menu)

    choice = instant_menu_choice(
        "[bold white]Select[/bold white] [muted](q=quit)[/muted]: ",
        {"1", "2", "3", "q"},
    )

    if choice is None:
        return  # Backspace/Escape → re-display menu

    if choice == "q":
        console.print("\n[muted]Goodbye![/muted]\n")
        sys.exit()

    mode = MODELS[choice]["mode"]
    if mode == "custom":
        run_custom_session(choice)
    elif mode == "design":
        run_design_session(choice)
    elif mode == "clone_manager":
        run_clone_manager(choice)


# ─── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        while True:
            try:
                main_menu()
            except Exception as e:
                console.print(f"\n[error]Unexpected error:[/error] {e}")
                console.print("[muted]Returning to main menu...[/muted]\n")
    except KeyboardInterrupt:
        console.print("\n\n[muted]Interrupted. Goodbye![/muted]\n")
