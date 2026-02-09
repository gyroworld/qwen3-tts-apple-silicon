"""Voice enrollment, list, pick, delete, update. Depends on config, ui, io, transcription."""

import os
import re
import shutil
import time

from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box

from qwen3_tts import config
from qwen3_tts.ui import console, clear_screen, instant_menu_choice, confirm_overwrite
from qwen3_tts.io import clean_path, convert_audio_if_needed

VOICES_DIR = config.VOICES_DIR


def get_saved_voices():
    """Return a sorted list of enrolled voice names."""
    if not os.path.exists(VOICES_DIR):
        return []
    voices = [f.replace(".wav", "") for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    return sorted(voices)


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


def enroll_new_voice():
    """Enroll a new voice for later cloning."""
    from qwen3_tts.transcription import APPLE_SPEECH_AVAILABLE, _offer_apple_transcribe

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
    from qwen3_tts.transcription import APPLE_SPEECH_AVAILABLE, _offer_apple_transcribe

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
