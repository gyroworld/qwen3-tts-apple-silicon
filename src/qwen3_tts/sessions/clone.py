"""Voice Cloning session: saved voices, enroll, quick clone, delete, update."""

import gc
import os

from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box

from qwen3_tts.config import MODELS, VOICES_DIR
from qwen3_tts.ui import console, clear_screen, instant_menu_choice
from qwen3_tts.io import (
    get_smart_path,
    ensure_model,
    load_model_with_progress,
    make_temp_dir,
    cleanup_temp_dir,
    save_audio_file,
    get_text_input,
    clean_path,
    convert_audio_if_needed,
)
from qwen3_tts.voices import get_saved_voices, enroll_new_voice, delete_voice, update_voice
from qwen3_tts.transcription import APPLE_SPEECH_AVAILABLE, _offer_apple_transcribe


def clean_memory():
    gc.collect()


def run_clone_manager(model_key):
    from mlx_audio.tts.generate import generate_audio

    clear_screen()
    console.print()
    console.print(Rule("[bold magenta]Voice Cloning[/bold magenta]", style="magenta"))
    console.print()

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

    if ref_text == "." and APPLE_SPEECH_AVAILABLE:
        console.print()
        apple_result = _offer_apple_transcribe(
            ref_audio,
            "Transcribe this voice with Apple?",
        )
        if apple_result:
            ref_text = apple_result
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
