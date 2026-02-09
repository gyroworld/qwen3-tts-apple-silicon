#!/usr/bin/env python3
"""
Qwen3-TTS Manager — A beautiful Rich-based CLI for text-to-speech on Apple Silicon.
Run: python app.py
"""

import os
import platform
import sys
import gc
import warnings

# Apple Silicon only: fail fast with a clear message (before Rich/MLX)
if sys.platform != "darwin" or platform.machine() != "arm64":
    msg = (
        "This app runs only on Apple Silicon Macs (M1/M2/M3/M4).\n"
        f"Your system: {sys.platform} / {platform.machine()}"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress transformers / tokenizers warnings that bypass Python's warnings module
# (e.g. Mistral regex, unregistered model_type). These are harmless for TTS.
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.ERROR)

from rich.panel import Panel
from rich.table import Table
from rich import box

# Allow importing qwen3_tts when running app.py from project root
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from qwen3_tts.config import BASE_OUTPUT_DIR, MODELS
from qwen3_tts.ui import console, instant_menu_choice, clear_screen, print_banner
from qwen3_tts.io import get_smart_path
from qwen3_tts.sessions import run_custom_session, run_design_session, run_clone_manager

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
        "[bold white]Select[/bold white]: ",
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
