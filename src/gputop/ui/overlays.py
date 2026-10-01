"""Modal overlays: the filter prompt, the kill confirmation and the help screen.

Each is a :class:`~textual.screen.ModalScreen` that returns a value, so the app's
reaction to it is a single ``push_screen_wait`` call and the overlay itself holds no
state that could disagree with the app.  That matters most for the kill dialog: what the
user confirmed has to be the thing the app acts on, with no intermediate copy to fall out
of step.

Keys inside a modal are bound with ``priority=True``.  Without it, a modal that is
waiting for ``k`` would also see the app's own ``k`` binding and trigger a second
confirmation underneath itself.
"""

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from gputop.control import SIGNAL_HELP, SIGNALS
from gputop.model.process import GpuProcess
from gputop.ui.format import fmt_bytes

#: Width of the modal boxes.  Fixed rather than content-driven: a confirmation whose box
#: changes size as the user types is a confirmation that reflows under the cursor.
FILTER_WIDTH = 64
KILL_WIDTH = 76
HELP_WIDTH = 74


class _Modal[T](ModalScreen[T]):
    """Shared look for every overlay.

    The palette comes from the app's CSS variables rather than from a constructor
    argument, so an overlay opened during a theme change is themed by the same source as
    the screen underneath it instead of by whatever was current when the class was
    written.
    """

    DEFAULT_CSS: ClassVar[str] = """
    _Modal {
        align: center middle;
        background: $surface 60%;
    }
    .modal-box {
        background: $panel;
        border: round $accent;
        padding: 1 2;
        height: auto;
    }
    .modal-title { color: $accent; text-style: bold; }
    .modal-hint { color: $text-muted; }
    """


class FilterPrompt(_Modal[str | None]):
    """Ask for the text to filter the process table by.

    Dismisses with the entered text, or ``None`` when cancelled.  The initial value is
    the current filter, so the prompt opens with the existing text selected rather than
    making the user retype it to refine it.
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    DEFAULT_CSS: ClassVar[str] = f"""
    FilterPrompt .modal-box {{ width: {FILTER_WIDTH}; }}
    """

    def __init__(self, current: str = "") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Static("Filter processes", classes="modal-title")
            yield Input(
                value=self._current,
                placeholder="name, command, user or pid",
                id="filter-input",
            )
            yield Static("enter apply  ·  esc clear", classes="modal-hint")

    def on_mount(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply the filter."""
        event.stop()
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """Leave the filter as it was."""
        self.dismiss(None)


class KillConfirm(_Modal[str | None]):
    """Confirm which signal to send to a selected process.

    Dismisses with the signal name, or ``None`` when cancelled.  Both signals are offered
    as explicit keys rather than a single "kill" button, because they are not variants of
    one action: one can be caught by the process and one cannot.
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("t", "send('SIGTERM')", "SIGTERM", show=True, priority=True),
        Binding("k", "send('SIGKILL')", "SIGKILL", show=True, priority=True),
        Binding("escape,c", "cancel", "Cancel", show=True, priority=True),
    ]

    DEFAULT_CSS: ClassVar[str] = f"""
    KillConfirm .modal-box {{ width: {KILL_WIDTH}; }}
    KillConfirm .modal-body {{ height: auto; }}
    """

    def __init__(self, process: GpuProcess) -> None:
        super().__init__()
        self._process = process

    def compose(self) -> ComposeResult:
        process = self._process
        with Vertical(classes="modal-box"):
            yield Static(f"Send a signal to pid {process.pid}?", classes="modal-title")
            yield Static(self._body(), id="kill-body", classes="modal-body")
            # Plain Statics rather than a row container: a ``Horizontal`` fills the
            # height it is given, which pushed the key hints off the bottom of a dialog
            # that had already wrapped onto two lines.
            yield Static(
                "  ".join(f"[b]{name[3:]}[/b] {SIGNAL_HELP[name]}" for name, _ in SIGNALS),
                classes="modal-hint",
            )
            yield Static(
                "  ".join(f"[b]{key}[/b] {desc}" for key, desc in _KEY_HELP),
                classes="modal-hint",
            )

    def _body(self) -> str:
        """The full description of the target: who it is, and what it is using."""
        process = self._process
        return "\n".join(
            [
                f"  user     {process.user}",
                f"  command  {process.command}",
                f"  engine   {process.engine_percent:.1f}%",
                f"  memory   {fmt_bytes(process.memory_used)}"
                f"  (VRAM {fmt_bytes(process.vram_used)}, GTT {fmt_bytes(process.gtt_used)})",
            ]
        )

    def action_send(self, signal_name: str) -> None:
        """Confirm ``signal_name``."""
        self.dismiss(signal_name)

    def action_cancel(self) -> None:
        """Change nothing."""
        self.dismiss(None)


#: Key hints shown in the confirmation, built from the bindings so they cannot drift.
_KEY_HELP: tuple[tuple[str, str], ...] = (
    ("t", "SIGTERM"),
    ("k", "SIGKILL"),
    ("esc", "cancel"),
)


class HelpOverlay(_Modal[None]):
    """The key reference."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape,q,question_mark,h", "close", "Close", show=False, priority=True),
    ]

    DEFAULT_CSS: ClassVar[str] = f"""
    HelpOverlay .modal-box {{ width: {HELP_WIDTH}; }}
    """

    def __init__(self, entries: tuple[tuple[str, str], ...]) -> None:
        super().__init__()
        self._entries = entries

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Static("gputop keys", classes="modal-title")
            for keys, description in self._entries:
                yield Static(f"[b]{keys:<16}[/b] {description}", classes="modal-hint")
            yield Static("esc close", classes="modal-hint")

    def action_close(self) -> None:
        """Dismiss."""
        self.dismiss(None)


def default_help(signal_help: str, *, blocks_note: str = "") -> tuple[tuple[str, str], ...]:
    """The full key reference.

    Args:
        signal_help: What ``k`` does in *this* session.  It is a parameter rather than a
            fixed line because the answer depends on the configuration, and a key
            reference that lists a key twice -- once as an action and once as a note --
            is worse than one that says plainly that the action is unavailable.
        blocks_note: A sentence about the blocks panel when it is not showing, so the reason
            a key reference mentions nothing about it is visible here rather than something
            the user has to infer.
    """
    entries = (
        ("q", "quit"),
        ("tab / shift+tab", "next / previous GPU"),
        ("1 … 9", "jump to GPU n"),
        ("o", "overview: every GPU at once"),
        ("click a tab", "switch GPU with the mouse"),
        ("click a panel", "move focus to it"),
        ("m", "cycle colour theme"),
        ("+ / -", "sampling interval"),
        ("p", "show or hide the process table"),
        ("s / S", "next / previous sort column"),
        ("r", "reverse the sort"),
        ("R", "sample now"),
        ("/", "filter processes"),
        ("esc", "clear the filter"),
        ("a", "show or hide idle processes"),
        ("t", "process tree on / off"),
        ("c", "full command line"),
        ("up / down", "move the selection"),
        ("k", signal_help),
        ("h", "this help"),
    )
    if blocks_note:
        return (*entries, ("", blocks_note))
    return entries


__all__ = ["FilterPrompt", "HelpOverlay", "KillConfirm", "default_help"]
