"""Theme, console, and input helpers. No dependency on config or io."""

import sys
from io import StringIO

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

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

    result = pt_prompt(ANSI(_rich_to_ansi(prompt_markup)), key_bindings=kb)

    if result == "__interrupt__":
        console.print()
        raise KeyboardInterrupt

    if result is not None:
        sys.stdout.write("\x1b[1A\r\x1b[2K")
        console.print(f"{prompt_markup}[bold cyan]{result}[/bold cyan]")

    return result


def clear_screen():
    """Clear the terminal screen."""
    console.clear()


def normalize_whitespace(s):
    """Return string with leading/trailing whitespace removed; None and empty become empty string."""
    if s is None:
        return ""
    return s.strip()


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
