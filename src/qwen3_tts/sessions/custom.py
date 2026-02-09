"""Custom Voice session: preset speakers, emotion, speed."""

import gc

from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box

from qwen3_tts.config import MODELS, SPEAKER_MAP, EMOTION_PRESETS, SPEED_PRESETS
from qwen3_tts.ui import console, clear_screen, instant_menu_choice, normalize_whitespace
from qwen3_tts.io import (
    ensure_model,
    load_model_with_progress,
    make_temp_dir,
    cleanup_temp_dir,
    save_audio_file,
    get_text_input,
)


def clean_memory():
    gc.collect()


def run_custom_session(model_key):
    from mlx_audio.tts.generate import generate_audio

    info = MODELS[model_key]
    model_path = ensure_model(info)
    if not model_path:
        return

    model = load_model_with_progress(model_path, info["name"])
    if not model:
        return

    clear_screen()

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
        if choice in all_speakers:
            speaker = choice
            console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
            break
        console.print("  [warning]Unknown speaker \u2014 try again.[/warning]")
        continue

    clear_screen()
    console.print()
    console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
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
    clear_screen()
    console.print()
    console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
    console.print(f"  [success]\u2713 Emotion:[/success] {base_instruct}")
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

    clear_screen()
    console.print()
    console.print(f"  [success]\u2713 Speaker:[/success] {speaker}")
    console.print(f"  [success]\u2713 Emotion:[/success] {base_instruct}")
    console.print(f"  [success]\u2713 Speed:[/success] {speed}x")
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
