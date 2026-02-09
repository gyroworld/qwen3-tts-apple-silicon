"""Voice Design session: describe a voice, then generate."""

import gc

from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from qwen3_tts.config import MODELS
from qwen3_tts.ui import console, clear_screen
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


def run_design_session(model_key):
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
