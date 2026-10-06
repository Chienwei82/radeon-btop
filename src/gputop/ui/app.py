"""The btop-style Textual interface.

Threading contract, which the rest of the program depends on:

* The sampler thread owns all sampling state and publishes immutable snapshots through a
  queue.  This file never performs filesystem I/O -- no ``open``, no ``os.readlink``, no
  sysfs read.  Everything the UI displays was already collected by the sampler.
* The UI thread only ever *reads* what it receives.  Nothing here mutates a snapshot, and
  nothing here is read by the sampler, so no lock is needed and the app is correct on the
  free-threaded build.
* The one thing the UI does own is a short-lived render buffer, rebuilt each frame on the
  UI thread alone.

The one deliberate exception to "no I/O" is :func:`gputop.control.terminate`, reached
through the ``k`` key: it is a single non-blocking syscall behind a disabled-by-default
config flag and an explicit confirmation, and handing it to a worker would add a message
round trip and a way to fail silently to avoid waiting for nothing.

View state -- the sort column, the filter, the theme, the interval -- is held in this
module, derived from the config at construction and written back to the config file when
it changes.  What the table *shows* is decided by :mod:`gputop.ui.procview`; this module
only decides what to ask it for.
"""

import contextlib
import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import Footer, Static

from gputop import control
from gputop.config import MAX_INTERVAL_MS, MIN_INTERVAL_MS, Config, StateConfig
from gputop.model.alerts import AlertState, evaluate, worst
from gputop.model.blocks import BlocksStatus
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.process import GpuProcess
from gputop.model.snapshot import GpuSnapshot
from gputop.sampler import Sampler, SamplerOptions
from gputop.sessionlog import SessionLog, open_log
from gputop.state import save_state
from gputop.ui.devices import GpuSwitcher, overview_panel
from gputop.ui.overlays import FilterPrompt, HelpOverlay, KillConfirm, default_help
from gputop.ui.panels import (
    blocks_heading,
    blocks_panel,
    footer_panel,
    header_panel,
    power_profile_panel,
    sensors_panel,
    utilisation_rows,
)
from gputop.ui.proctable import ProcessTable
from gputop.ui.procview import (
    FilterState,
    SortColumn,
    SortState,
    build_rows,
    busiest_processes,
    filter_processes,
)
from gputop.ui.theme import Gradient, Theme, get_theme, theme_names
from gputop.ui.widgets import BrailleGraph, CpuMeter

#: Below this width the layout collapses to a single column.
NARROW_WIDTH = 72
#: Below this height the per-process table is hidden to keep the graphs legible.
SHORT_HEIGHT = 20
#: Below this height the memory graphs are hidden; an empty bordered box reads as a bug.
MIN_HEIGHT_FOR_MEMORY_GRAPHS = 28

#: Rows the interface needs around the utilisation graph whatever its size: the docked
#: device header, the GPU tab bar when shown, the graph panel's title and readout, the
#: memory row, the status line, the process table and the footer.
#:
#: Measured by growing the terminal until the pinned graph stopped gaining rows, rather
#: than estimated: an estimate that is a row or two low would let the pinned height push
#: the footer off the bottom, which is the one failure the responsive layout exists to
#: prevent.
MIN_HEIGHT_AROUND_UTIL_GRAPH = 22

#: Fewest rows a pinned utilisation graph is ever given.  Matches the ``min-height`` the
#: graph carries in CSS, so a pinned height can never be smaller than the default one.
MIN_UTIL_GRAPH = 3

#: Rows the utilisation panel needs besides its graph: the top border, the title, the two
#: readout lines and the bottom border.  A pinned graph height is added to this to form the
#: panel's ``min-height``; without it the plot grows and the readout is what gets cropped.
UTIL_PANEL_CHROME_ROWS = 5

#: Terminal height at which the sensor panel shows every one of its rows without scrolling.
#:
#: Measured against the reference card, not estimated.  Below this the panel's last lines
#: (PCIe link, power profile, throttle reasons) are cut off, and it scrolls.
MIN_HEIGHT_FOR_SENSORS_COMPLETE = 38

#: Terminal height at which the optional row of panels may be added.
#:
#: One threshold for both panels, and it is higher than either panel needs on its own.  The
#: cost is not the row's height but what it takes from the sensor panel beside it: the sensor
#: panel scrolls, so a row added below the main one does not lose its content, it pushes the
#: last lines out of view.  Measured, the sensors need 38 rows and the row costs 8 more, so
#: 46 is the point below which showing it would cost a required panel something.  An
#: optional panel must never be the reason a required one needs scrolling.
MIN_HEIGHT_FOR_OPTIONAL_ROW = MIN_HEIGHT_FOR_SENSORS_COMPLETE + 8

#: Narrowest the optional panels are laid out at.  Their bars are sized from the panel, and a
#: panel narrower than its own label column produces a bar of no useful length.
MIN_OPTIONAL_PANEL_WIDTH = 30

#: How often the saved state is checked for changes.  Long enough that holding down a key
#: does not write the file sixty times a second, short enough that a crash costs at most
#: this much of the session's state.
STATE_SAVE_INTERVAL = 2.0

#: Step the ``+`` and ``-`` keys move the sampling interval by, in milliseconds.
INTERVAL_STEP_MS = 100

#: How long the refresh timer is re-armed for after the ``R`` key asks for a sample now.
#: Long enough for the sampler to wake, finish a tick and publish, short enough that the
#: redraw reads as immediate.  The normal cadence is restored on the far side of it.
REFRESH_NUDGE_S = 0.15

#: How long a reply to a key press stays in the status line, in seconds.
#:
#: Not "until the next sample", which is what this used to be.  At the default 1 s interval
#: that is a second, which is readable; at the 100 ms the ``-`` key reaches it is a tenth of
#: one, which no human reads and which makes the tests that check for the text race the very
#: sample that erases it.  A fixed lifetime is also the honest description of what the
#: status line is for: a reply, not a state.
NOTICE_LIFETIME_S = 1.5

NO_DEVICE_HELP = (
    "No amdgpu device found.\n\n"
    "gputop only supports the amdgpu driver. Check that your GPU is bound to it:\n"
    "    ls -l /sys/class/drm/card*/device/driver"
)


def short_bdf(bdf: str) -> str:
    """Shorten a PCI address to the bus and device, e.g. ``0c:00.0``.

    The domain is almost always ``0000`` on a consumer machine, so dropping it buys room
    for the column while keeping the part that actually distinguishes two GPUs.
    """
    parts = bdf.split(":")
    return ":".join(parts[1:]) if len(parts) == 3 else (bdf or "N/A")


class Panel(Vertical):
    """A titled panel that can take focus.

    Clicking a panel focuses it.  With no tab bar and no menu, focus is the only way to
    aim the keyboard at a region of the screen, so the mouse has to be able to set it too
    -- and the focus marker lives in the *title* rather than in a border colour, because
    a border that changes colour to say "focused" is invisible unless you already know to
    look for it.
    """

    can_focus = True
    can_focus_children = True

    def __init__(
        self,
        *,
        title: str,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
        classes: str | None = None,
    ) -> None:
        # The ``panel`` class is added unconditionally rather than passed at each call
        # site.  It is what identifies a bordered box for the alert flash, and the two
        # memory panels are plain ``Vertical``\\s carrying the same class; adding it here
        # means the flash can select on one name and cannot miss a panel, instead of
        # selecting on the type and silently skipping the two that are not ``Panel``.
        super().__init__(id=id, classes=f"panel {classes}" if classes else "panel")
        self._title = title
        self._focused = False

    def compose(self) -> ComposeResult:
        yield Static(self._rendered_title(), id=self.title_id, classes="panel-title")
        yield from self.compose_body()

    def compose_body(self) -> ComposeResult:
        """The panel's content.  Overridden by each concrete panel."""
        return iter(())

    @property
    def title_id(self) -> str:
        """CSS id of this panel's heading, derived from the panel's own id."""
        return f"{self.id}-title"

    def set_title(self, title: str) -> None:
        """Change the heading text, keeping the focus marker."""
        self._title = title
        with contextlib.suppress(NoMatches):
            self.query_one(f"#{self.title_id}", Static).update(self._rendered_title())

    def on_focus(self) -> None:
        """Show the focus marker."""
        self._mark_focused(True)

    def on_blur(self) -> None:
        """Hide the focus marker."""
        self._mark_focused(False)

    def on_click(self) -> None:
        """Move focus here, unless a focusable child has already claimed it.

        A click inside the process table selects a row and the table takes focus; taking
        it back would undo exactly what the user just did.
        """
        focused = self.screen.focused
        if focused is None or (focused is not self and self not in focused.ancestors):
            self.focus()

    def _mark_focused(self, focused: bool) -> None:
        if focused == self._focused:
            return
        self._focused = focused
        self.set_title(self._title)

    def _rendered_title(self) -> str:
        return f"▸ {self._title}" if self._focused else self._title


class UtilPanel(Panel):
    """The utilisation graph and its readouts."""

    def __init__(
        self,
        theme: Theme,
        gradient: Gradient,
        *,
        title: str,
        capacity: int,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
    ) -> None:
        super().__init__(title=title, id=id)
        self._theme = theme
        self._gradient = gradient
        self._capacity = capacity

    def compose_body(self) -> ComposeResult:
        yield BrailleGraph(
            theme=self._theme, gradient=self._gradient, id="util-graph", capacity=self._capacity
        )
        yield Static("", id="util-readout")


class SensorPanel(Panel):
    """The sensor readouts."""

    def compose_body(self) -> ComposeResult:
        yield Static("", id="sensors-body")


class BlocksPanel(Panel):
    """Per-block utilisation from radeontop.

    A panel with nothing to say is hidden by :meth:`set_available` rather than rendered
    empty.  An empty bordered box is read as a bug, and on a machine without radeontop it
    would be a permanent one.
    """

    def compose_body(self) -> ComposeResult:
        yield Static("", id="blocks-body")


class ProfilePanel(Panel):
    """The read-only power profile and overdrive table."""

    def compose_body(self) -> ComposeResult:
        yield Static("", id="profile-body")


class ProcessPanel(Panel):
    """The per-process table, under the panel's own heading.

    Each panel owns its own body rather than having the app nest children inside a
    ``with`` block: a panel that has to be declared in two places is a panel whose title
    and whose contents can drift apart, and this app has already been bitten by a layout
    that looked right and was not.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        sort: SortState,
        full_command: bool,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
    ) -> None:
        super().__init__(title="PROCESSES", id=id)
        self._theme = theme
        self._sort = sort
        self._full_command = full_command

    def compose_body(self) -> ComposeResult:
        yield ProcessTable(
            self._theme,
            sort=self._sort,
            full_command=self._full_command,
            id="process-table-panel",
        )


class GpuTopApp(App[None]):
    """The gputop terminal interface."""

    TITLE = "gputop"
    SUB_TITLE = "AMD Radeon monitor"

    CSS: ClassVar[str] = """
    Screen { layout: vertical; background: $surface; }
    /* Docked so the device name is never scrolled out of view on a short terminal:
       it is the one piece of text that has to survive every layout. */
    #device-header { dock: top; height: auto; padding: 0 1; }
    #gpu-switcher { height: 1; padding: 0 1; display: none; }
    #main-row { height: 1fr; min-height: 6; }
    #graphs { width: 2fr; }
    .panel-title { color: $accent; text-style: bold; height: 1; }
    #util-panel { height: 2fr; min-height: 6; border: round $accent; padding: 0 1; }
    #util-graph { height: 1fr; min-height: 3; }
    #util-readout { height: auto; }
    /* Hidden outright on a short terminal: a border with no graph inside looks broken. */
    #memory-row { height: 1fr; min-height: 5; }
    #vram-panel, #gtt-panel { width: 1fr; border: round $accent; padding: 0 1; }
    #vram-graph, #gtt-graph { height: 1fr; min-height: 3; }
    #sensors-panel {
        width: 1fr;
        min-width: 40;
        border: round $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    #overview { width: 1fr; height: 1fr; padding: 0 1; display: none; }
    /* A row of its own below the main row rather than a fourth column beside it.  The main
       row is a three-way split whose widths the responsive layout already computes; adding
       to it would mean re-deriving those on every terminal width, and the two new panels
       are the first things that should yield when space runs short.  `height: auto` lets
       the row take only the rows it needs, and the two 1fr rows above and below absorb the
       difference. */
    #extra-row { height: auto; display: none; }
    #blocks-panel, #profile-panel { width: 1fr; height: auto; border: round $accent; padding: 0 1; }
    #blocks-body, #profile-body { height: auto; }
    /* Not docked: two bottom-docked rows compete with the Footer for the same edge and
       the panel draws over the status line.  In normal flow it always gets its row. */
    #status-row { height: auto; }
    #status { height: auto; padding: 0 1; }
    #process-panel { height: 1fr; min-height: 5; border: round $accent; padding: 0 1; }
    """

    # Only the handful a user reaches for are shown in the footer.  The full list is one
    # `h` away, and a footer that wraps is a footer nobody reads.
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        # Priority beats the Screen's own tab/shift-tab focus traversal, which would
        # otherwise make the most-used multi-GPU key move the cursor instead.
        Binding("tab", "next_device", "GPU", show=False, priority=True),
        Binding("shift+tab", "previous_device", "GPU", show=False, priority=True),
        Binding("o", "toggle_overview", "All GPUs"),
        Binding("slash", "edit_filter", "Filter"),
        Binding("h,question_mark", "help", "Help"),
        Binding("1", "focus_device(0)", "GPU 1", show=False),
        Binding("2", "focus_device(1)", "GPU 2", show=False),
        Binding("3", "focus_device(2)", "GPU 3", show=False),
        Binding("4", "focus_device(3)", "GPU 4", show=False),
        Binding("5", "focus_device(4)", "GPU 5", show=False),
        Binding("6", "focus_device(5)", "GPU 6", show=False),
        Binding("7", "focus_device(6)", "GPU 7", show=False),
        Binding("8", "focus_device(7)", "GPU 8", show=False),
        Binding("9", "focus_device(8)", "GPU 9", show=False),
        Binding("R", "refresh_now", "Resample", show=False),
        Binding("plus", "faster", "Faster", show=False),
        Binding("minus", "slower", "Slower", show=False),
        Binding("m", "cycle_theme", "Theme", show=False),
        Binding("p", "toggle_processes", "Procs", show=False),
        Binding("s", "next_sort", "Sort", show=False),
        Binding("S", "previous_sort", "Sort", show=False),
        Binding("r", "reverse_sort", "Reverse", show=False),
        Binding("escape", "clear_filter", "Clear", show=False),
        Binding("a", "toggle_idle", "Idle", show=False),
        Binding("t", "toggle_tree", "Tree", show=False),
        Binding("c", "toggle_full_command", "Cmdline", show=False),
        Binding("up", "select_previous", "Up", show=False),
        Binding("down", "select_next", "Down", show=False),
        Binding("k", "kill_selected", "Signal", show=False),
    ]

    def __init__(
        self,
        config: Config,
        options: SamplerOptions,
        *,
        state_path: Path | None = None,
        log_target: Path | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._options = options
        self._state_path = state_path
        self._log_target = log_target
        self._recorder: SessionLog | None = None
        self._last_recorded: float | None = None
        self._sampler = Sampler(options)
        self._show_processes = config.process.show
        self._limit = config.process.max_rows
        self._interval_s = options.validated().interval_s
        self._save_timer: Timer | None = None
        self._refresh_timer: Timer | None = None
        self._state_dirty = False
        # Two notices, because there are two kinds of thing worth saying.  ``_notice`` is
        # a standing condition the user has to act on and stays until something replaces
        # it; ``_transient_notice`` is the reply to a key press and is dropped by the next
        # sample.  One slot for both meant a transient message ("no process selected")
        # outlasted the session, because nothing ever cleared it.
        self._notice: str | None = None
        self._transient_notice: str | None = None
        self._notice_until_ns: int | None = None
        # What the layout and the panel borders were last set to.  Both are pure functions
        # of values that change far less often than the sampler ticks, and writing an
        # inline style that is already set makes Textual re-resolve the stylesheet and
        # repaint the widget -- so the steady state has to write nothing at all.
        self._layout_key: tuple[object, ...] | None = None
        self._border_colour: str | None = None

        self._theme: Theme = get_theme(config.ui.theme)
        self._gradient = Gradient(self._theme)
        self._monochrome = config.ui.no_color
        names = theme_names()
        self._theme_index = max(
            0, names.index(self._theme.name) if self._theme.name in names else 0
        )

        # View state.  The config file is the starting point; a saved session overrides
        # it, and the CLI has already been applied over the top by the caller.
        state = config.state
        self._sort = SortState.parse(
            state.sort or config.process.sort, reverse=state.sort_reverse
        )
        self._view = FilterState(
            text=state.filter,
            show_idle=state.show_idle or config.process.show_idle,
            min_engine_percent=config.process.min_usage_percent,
            hide_kernel_threads=config.process.hide_kernel_threads,
        )
        self._tree = state.process_tree or config.process.process_tree
        self._full_command = state.full_command or config.process.full_command
        self._overview = state.overview
        # Floored here for the same reason ``SortState.parse`` validates its column: a
        # config file and a persisted state file are both things a user can edit by hand.
        # A negative index was not caught by the clamp in ``_render``, which only caps the
        # top, so ``devices[-1]`` silently focused the *last* GPU while the tab bar
        # highlighted nothing and the footer said "GPU 0/2" -- and the bad value was then
        # written back out, so it survived the next restart too.
        self._index = max(0, state.gpu_index)

        self._cpu = CpuMeter()
        self._started_monotonic = time.monotonic()
        self._last_snapshot: GpuSnapshot | None = None
        self._narrow = False
        self._short = False
        # Whether the alert border is currently in its lit phase of the flash cycle.  Kept
        # as state rather than derived from the clock so that the flash is driven by the
        # timer, not by how often samples happen to arrive: a sampler running slower than
        # the flash rate would otherwise produce a border that never changes.
        self._flash_on = False
        self._alert_state: AlertState = AlertState.clear()
        self._flash_timer: Timer | None = None
        # ``is_running`` alone is not enough: it is still False while the first screen is
        # mounting, and true again nowhere.  The pair brackets exactly the window in
        # which the widgets this app draws into exist.
        #
        # The guard is not paranoia.  Textual closes the screens *before* it dispatches
        # ``Unmount`` and only then stops the timers, so an interval that fires during
        # shutdown queries a screen that has already been pruned -- and an exception
        # raised inside a timer callback is fatal to the app, not merely noisy.
        self._live = False

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # All three graphs are fed from the sampler's ring buffer and hold every sample in
        # it, so their capacity is the one configured number: ``general.history_points``.
        # It used to be left at the widget's own default, which happened to match, so
        # lowering history_points silently left the graphs holding more than could be
        # filled -- and raising it could not have worked at all, because there was nothing
        # raising it.
        capacity = self._config.history_length
        yield Static("", id="device-header")
        yield GpuSwitcher(self._theme, id="gpu-switcher")
        with Horizontal(id="main-row"):
            with Vertical(id="graphs"):
                yield UtilPanel(
                    self._theme,
                    self._gradient,
                    title="GPU utilisation",
                    id="util-panel",
                    capacity=capacity,
                )
                with Horizontal(id="memory-row"):
                    with Vertical(id="vram-panel", classes="panel"):
                        yield Static("VRAM", classes="panel-title")
                        yield BrailleGraph(
                            theme=self._theme,
                            gradient=self._gradient,
                            id="vram-graph",
                            capacity=capacity,
                        )
                    with Vertical(id="gtt-panel", classes="panel"):
                        yield Static("GTT", classes="panel-title")
                        yield BrailleGraph(
                            theme=self._theme,
                            gradient=self._gradient,
                            id="gtt-graph",
                            capacity=capacity,
                        )
            yield SensorPanel(title="Sensors", id="sensors-panel")
            yield Static("", id="overview")
        with Horizontal(id="extra-row"):
            yield BlocksPanel(title="Blocks", id="blocks-panel")
            yield ProfilePanel(title="Power profile", id="profile-panel")
        with Horizontal(id="status-row"):
            yield Static("", id="status")
        yield ProcessPanel(
            self._theme,
            sort=self._sort,
            full_command=self._full_command,
            id="process-panel",
        )
        yield Footer()

    def discover(self) -> tuple[object, ...]:
        """Discover devices outside the event loop.

        Discovery reads sysfs.  It belongs to startup, not to a frame, so
        :func:`run_app` calls this before ``App.run``; the fallback in
        :meth:`on_mount` exists only for direct construction in tests.
        """
        return self._sampler.discover()

    def on_mount(self) -> None:
        # The palette is re-asserted here because Textual builds the stylesheet's variable
        # map before this constructor has run, so the first pass had no theme to merge.
        # Without it the screen renders in Textual's stock palette until a theme change.
        self._apply_css_variables()
        # Always start, even when discover() already ran: that call only enumerates
        # devices, and skipping start() here left the sampling thread unlaunched so the
        # UI rendered but never updated.  start() is idempotent.
        self._sampler.start()
        if not self._sampler.devices:
            self.query_one("#device-header", Static).update(NO_DEVICE_HELP)
            return

        # A standing condition, so ``_notice`` rather than ``notify_line``: the user has
        # to act on it and it does not go away on its own.  The alternative -- letting the
        # filter quietly watch every card -- is the exact thing the warning exists to stop.
        if self._sampler.filter_warning:
            self._notice = self._sampler.filter_warning
        self._apply_colour_mode()
        self._open_recorder()
        # 1 Hz is the floor for the app's own CPU figure: a faster sample would mostly
        # measure the timer's jitter, and the number is for a human reading it.
        self.set_interval(1.0, self._tick_cpu)
        self._refresh_timer = self.set_interval(self._interval_s, self._refresh)
        self._start_flash_timer()
        if self._state_path is not None:
            # A first run still writes: the theme and the interval it starts with are
            # choices the next start should reproduce, and "the file was never created"
            # is indistinguishable from "nothing was ever recorded".
            self._state_dirty = not self._state_path.is_file()
            self._save_timer = self.set_interval(STATE_SAVE_INTERVAL, self._flush_state)
        self._live = True
        self._refresh()
        self._focus_default()

    def on_unmount(self) -> None:
        """Stop the sampler and persist whatever the last frame was showing."""
        self._live = False
        for timer in (self._refresh_timer, self._save_timer, self._flash_timer):
            if timer is not None:
                timer.stop()
        self._sampler.stop()
        self._flush_state()
        self._close_recorder()

    # -- session recording -------------------------------------------------

    def _open_recorder(self) -> None:
        """Open the recording target, if ``--log`` named one.

        Opened on mount rather than in ``__init__`` so that a path which cannot be written
        produces a notice in the status bar the user will actually read, instead of an
        exception on a screen that has not been drawn yet.
        """
        if self._log_target is None:
            return
        log = open_log(self._log_target, zstd_level=self._config.log.zstd_level)
        if log.problem is not None:
            self._notice = f"cannot record to {self._log_target}: {log.problem.value}"
            return
        self._recorder = log

    def _record(self, snapshot: GpuSnapshot) -> None:
        """Append the newest snapshot to the recording.

        Only the newest snapshot of a tick is recorded, for the same reason the UI renders
        only that one: during a slow repaint the queue can hold several, and writing them
        all would fill a recording with samples nobody ever saw.

        ``log.interval_s`` thins the stream further for a session that will run for hours: a
        100 ms sampler produces 36 000 rows an hour, most of them the same idle card, and a
        file nobody can plot is a file nobody reads.  The check lives here rather than in
        the writer so the recorder stays a pure sink with no clock of its own.
        """
        if self._recorder is None:
            return
        minimum = self._config.log.interval_s
        if minimum > 0:
            now = time.monotonic()
            if self._last_recorded is not None and now - self._last_recorded < minimum:
                return
            self._last_recorded = now
        if not self._recorder.write(snapshot):
            self._notice = f"recording stopped: {self._recorder.problem}"
            self._recorder = None

    def _close_recorder(self) -> None:
        """Close the recording, writing the compression frame's epilogue.

        Called from unmount so every exit path closes the file.  For a ``.zst`` target an
        unclosed stream is a truncated archive rather than a readable one, which is the one
        outcome worse than not having recorded at all.
        """
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    # -- alerts ------------------------------------------------------------

    def _start_flash_timer(self) -> None:
        """Start the alert flash timer, at the half-period of the configured rate.

        The timer is started once at mount and simply stops toggling when nothing is over
        its limit, rather than being started and stopped as conditions come and go.  A
        timer whose existence tracks a condition has to be torn down on every clear, and a
        clear that arrives between the check and the teardown leaves a timer firing forever.
        One timer that checks a boolean cannot get out of step with that boolean.
        """
        hz = self._config.alerts.flash_hz
        if hz <= 0:
            # 0 means "hold a steady border", which is a valid choice for someone who finds
            # a flashing panel intolerable, so it is not treated as "no alerts".
            return
        # Two toggles per cycle: lit, unlit, lit...
        self._flash_timer = self.set_interval(0.5 / hz, self._tick_flash)

    def _tick_flash(self) -> None:
        """Advance the flash phase and repaint the borders."""
        if not (self._live and self._drawable()):
            return
        if not self._alert_state.active:
            return
        self._flash_on = not self._flash_on
        self._apply_alert_borders()

    def _apply_alert_borders(self) -> None:
        """Colour every panel border according to the current alert state.

        With alerting off or nothing crossed, the border is the theme accent -- the normal
        case, and the one that must leave no trace of having been alarmed.  While a threshold
        is crossed the border alternates between the alert colour and the track colour, which
        is the flash.

        All panels share one border colour rather than each flashing independently.  A screen
        with three differently-timed borders is noise; one border that changes says "this
        screen", which is what the user needs to know.

        The colour is written to each panel's own style rather than pushed through a CSS
        variable, because :meth:`Stylesheet.set_variables` *replaces* the variable map
        rather than merging into it.  Driving a once-per-second flash through it deletes
        Textual's own design tokens -- including the ``$background`` its default ``App``
        rule references -- and the whole stylesheet then fails to resolve.

        The write is skipped when the colour has not changed, which is every sample while
        nothing is over its limit.  Re-asserting an identical inline style still marks the
        widget dirty in Textual, and a per-sample repaint of every panel is the single
        largest avoidable cost in the frame.
        """
        colour = self._alert_border_colour()
        if colour == self._border_colour:
            return
        self._border_colour = colour
        for panel in self.query(".panel"):
            panel.styles.border = ("round", colour)

    def _alert_border_colour(self) -> str:
        """The colour every panel border should currently be.

        Two states: the theme accent when nothing is over a limit, and an alternation
        between the alert colour and the track colour while something is.  The dark phase
        is the track rather than the accent so the flash reads as "this border went away"
        rather than as a second thing to look at.
        """
        state = self._alert_state
        if not self._config.alerts.enabled or not state.active:
            return self._theme.accent
        return self._theme.alert if self._flash_on else self._theme.track

    def _evaluate_alerts(self, snapshot: GpuSnapshot) -> AlertState:
        """Recompute the alert state for the whole snapshot.

        Every device is judged, not just the focused one: a card the user is not looking at
        can be the one that is overheating, and a border that only watches the focused card
        would report the machine as fine while it burns.
        """
        if not self._config.alerts.enabled:
            return AlertState.clear()
        thresholds = self._config.alerts.thresholds()
        return worst(evaluate(metrics, thresholds) for metrics in snapshot.devices)

    def _focus_default(self) -> None:
        """Start with the process table focused.

        It is the only panel with a cursor, and a monitor whose main interaction is "move
        the selection" should not require a click before the arrow keys do anything.
        """
        if self._show_processes and not self._short:
            self.query_one(ProcessTable).focus()

    def on_resize(self) -> None:
        """Re-evaluate the layout when the terminal changes size.

        Textual calls ``Widget.on_resize`` with no argument; declaring an ``event``
        parameter made this handler raise a ``TypeError`` on every resize, so a terminal
        that shrank kept its previous layout until the next sample happened to redraw.
        """
        self._apply_responsive_layout()

    # -- colour -------------------------------------------------------------

    def _apply_colour_mode(self) -> None:
        """Reconcile truecolor detection with ``--no-color``."""
        if self._monochrome:
            # Textual's ANSI mode is the correct graceful fallback: it keeps the
            # interface legible on a terminal that cannot do 24-bit colour.
            self.ansi_color = True
            return
        system = getattr(self.console, "color_system", None)
        truecolor = system in (None, "truecolor")
        if not truecolor:
            self._gradient = Gradient(self._theme, truecolor=False)
        for widget in self.query(BrailleGraph):
            widget.set_truecolor(truecolor)
        self.refresh(layout=True)

    def set_theme(self, name: str) -> None:
        """Switch theme at runtime."""
        self._theme = get_theme(name)
        names = theme_names()
        self._theme_index = names.index(self._theme.name) if self._theme.name in names else 0
        system = getattr(self.console, "color_system", None)
        self._gradient = Gradient(self._theme, truecolor=system in (None, "truecolor"))
        for graph in self.query(BrailleGraph):
            graph.set_theme(self._theme, truecolor=self._gradient.truecolor)
        for switcher in self.query(GpuSwitcher):
            switcher.set_theme(self._theme)
        for table in self.query(ProcessTable):
            table.set_theme(self._theme)
        self._apply_css_variables()
        self._refresh()
        self._mark_state_dirty()

    def _apply_css_variables(self) -> None:
        """Re-resolve CSS after a theme change.

        Textual caches the variable map, so assigning a new palette is not enough on its
        own; this asks for the re-parse.  It goes through :meth:`refresh_css` rather than
        the stylesheet directly because ``refresh_css`` re-reads
        :meth:`get_css_variables` -- the override that merges gputop's palette into
        Textual's own tokens -- where the direct call would overwrite them.
        """
        self.refresh_css()

    def get_css_variables(self) -> dict[str, str]:
        """Merge the gputop palette into Textual's own CSS variables.

        This is Textual's documented extension point, and the reason it is used instead of
        :meth:`Stylesheet.set_variables` is that the two do opposite things: ``super()``
        returns the full design-token map, which ``set_variables`` *assigns over*, so a
        call carrying three names leaves three entries where there were 168.  Every one of
        Textual's own rules that referenced a now-missing variable -- ``$background`` in the
        default ``App`` rule, ``$panel`` and ``$text-muted`` in gputop's own overlay
        stylesheet -- then failed to resolve, and the app raised ``UnresolvedVariableError``
        on the next stylesheet re-parse, which is any modal or any theme change.

        Overriding the getter also means the theme is in effect from the first frame.
        Setting variables from an action applied nothing at start-up, so ``$accent``
        resolved to Textual's stock orange until the user pressed ``m``.
        """
        variables = super().get_css_variables()
        # Textual asks for these before ``__init__`` has run, so there may be no palette
        # yet.  Returning the framework's own tokens unchanged is the right answer to
        # "no palette yet"; ``on_mount`` re-asserts the real one.
        theme = getattr(self, "_theme", None)
        if theme is None:
            return variables
        variables.update(
            {
                "accent": theme.accent,
                "surface": theme.track,
                "text": theme.text,
                # Used by the modal stylesheet, which is the one place a themed surface is
                # visible as a large block rather than as a border or a graph background.
                "panel": theme.track,
                "text-muted": theme.muted,
            }
        )
        return variables

    # -- refresh ------------------------------------------------------------

    def _tick_cpu(self) -> None:
        """Recompute the app's own CPU usage once a second."""
        self._cpu.sample()
        if self._drawable() and self._last_snapshot is not None:
            self.query_one("#status", Static).update(self._status(self._last_snapshot))

    def _refresh(self) -> None:
        """Drain queued snapshots and redraw the newest.

        Only the newest snapshot is rendered.  During a slow repaint the queue can hold
        several; rendering them all would show stale data and waste the frame.
        """
        if not (self._live and self.is_running):
            return
        snapshot = self._sampler.latest()
        if snapshot is None:
            return
        self._last_snapshot = snapshot
        self._record(snapshot)
        self._render(snapshot)

    def _render(self, snapshot: GpuSnapshot) -> None:
        """Draw one snapshot into the widgets."""
        if not snapshot.devices:
            return
        self._index = min(self._index, len(snapshot.devices) - 1)
        metrics = snapshot.devices[self._index]
        # The switcher and the overview compare the *stored* device index, which is not the
        # same number as the position: a device whose sample fails is dropped from
        # ``snapshot.devices`` by the sampler's per-device error handling, so the two only
        # agree while every card is readable.  When they disagreed the focused tab lost
        # its badge, its bold name and its caret, and the overview lost its "focused"
        # marker, with nothing on screen to say which GPU was actually shown.
        focused = metrics.device.index

        self.query_one("#device-header", Static).update(
            header_panel(
                metrics,
                self._theme,
                time.monotonic() - self._started_monotonic,
                self._interval_s,
            )
        )
        self.query_one(GpuSwitcher).update_devices(snapshot.devices, focused)

        graphs_inner = max(30, self.query_one("#graphs").size.width - 4)
        self.query_one("#util-readout", Static).update(
            _join(
                utilisation_rows(
                    metrics, self._theme, available=graphs_inner, gradient=self._gradient
                )
            )
        )
        # Bar width follows the panel, not a constant: a fixed width overflowed the
        # sensor panel and ran each value into the next row.
        sensor_inner = max(24, self.query_one("#sensors-panel").size.width - 4)
        self.query_one("#sensors-body", Static).update(
            _join(
                sensors_panel(
                    metrics,
                    self._theme,
                    available=sensor_inner,
                    gradient=self._gradient,
                )
            )
        )

        if self._overview:
            self.query_one("#overview", Static).update(
                overview_panel(
                    snapshot.devices,
                    self._theme,
                    max(30, self.size.width - 4),
                    index=focused,
                    gradient=self._gradient,
                )
            )

        self._render_extra(snapshot, metrics)

        # The graphs are fed from the sampler's history, so a fresh process starts with a
        # graph that fills in rather than one that only ever shows the latest point.
        self._feed_graphs(snapshot)
        self._render_processes(snapshot)

        self._alert_state = self._evaluate_alerts(snapshot)
        if self._alert_state.active:
            # Summarised here rather than in the footer because the footer is already full
            # and an alert that cannot be read is the same as an alert that did not happen.
            self._notice = self._alert_state.summary or self._notice
        self._apply_alert_borders()

        self.query_one("#status", Static).update(self._status(snapshot))
        # The key-press reply has been on screen long enough to read; the standing notices
        # above have not, and are untouched.
        if self._notice_until_ns is not None and time.monotonic_ns() >= self._notice_until_ns:
            self._transient_notice = None
            self._notice_until_ns = None
        self._apply_responsive_layout()

    def _render_extra(self, snapshot: GpuSnapshot, metrics: AmdgpuMetrics) -> None:
        """Draw the two optional panels, hiding each one that has nothing to say.

        The two are decided **independently**, which is the whole point of putting them in
        one row.  The profile panel is read from sysfs and works on every card that publishes
        SCPP; the blocks panel needs a privileged child process that is off by default.
        Gating the profile panel on the blocks panel meant that on any machine where
        radeontop is absent -- which is every machine running gputop unprivileged -- the
        power-profile table was silently withheld from a user who had every right to see it.

        Each panel is hidden rather than shown empty.  A bordered box with a "waiting"
        caption reads as a bug, and an ``N/A`` beside each of fourteen blocks answers no
        question anyone asked.  When the blocks panel is hidden for want of data, the reason
        appears in the status line and the help overlay instead.
        """
        height = self.size.height
        blocks = metrics.blocks
        # Both panels wait for the same height: below it the row is not affordable without
        # the sensor panel starting to scroll.  See MIN_HEIGHT_FOR_OPTIONAL_ROW.
        show_blocks = blocks is not None and height >= MIN_HEIGHT_FOR_OPTIONAL_ROW
        has_tables = metrics.profiles.present or metrics.odc.present
        show_profile = has_tables and height >= MIN_HEIGHT_FOR_OPTIONAL_ROW

        blocks_widget = self.query_one("#blocks-panel", BlocksPanel)
        blocks_widget.display = show_blocks
        self.query_one("#profile-panel", ProfilePanel).display = show_profile
        self.query_one("#extra-row").display = show_blocks or show_profile

        if show_blocks:
            width = max(
                MIN_OPTIONAL_PANEL_WIDTH, self.query_one("#blocks-panel").size.width - 4
            )
            blocks_widget.set_title(blocks_heading(blocks))
            self.query_one("#blocks-body", Static).update(
                _join(blocks_panel(blocks, self._theme, width, gradient=self._gradient))
            )
        if show_profile:
            width = max(
                MIN_OPTIONAL_PANEL_WIDTH, self.query_one("#profile-panel").size.width - 4
            )
            self.query_one("#profile-body", Static).update(
                _join(power_profile_panel(metrics, self._theme, width, gradient=self._gradient))
            )

    def _render_processes(self, snapshot: GpuSnapshot) -> None:
        """Rebuild the process table for the current view state."""
        panel = self.query_one(ProcessTable)
        panel.set_full_command(self._full_command)
        if not panel.display:
            return
        # In overview mode the table stops being scoped to one GPU: the point of the mode
        # is to see everything at once, and a table showing only the focused card's
        # clients would quietly contradict the row of cards above it.
        scope = None if self._overview else self._current_bdf()
        filters = self._view.for_device(scope)
        # The bdf scoping is left to ``filter_processes``, which already applies it via
        # ``FilterState.matches``.  It used to be pre-applied here as well, which built a
        # second copy of the same rule: they agreed, and a change to either one would have
        # changed what the table shows with nothing on screen to say so.
        kept = filter_processes(snapshot.processes, filters)
        rows = build_rows(kept, sort=self._sort, filters=filters, tree=self._tree)
        top = busiest_processes(kept, sort=self._sort)
        panel.update_rows(
            rows[: self._limit],
            sort=self._sort,
            filters=filters,
            top=top,
            total=len(kept),
            tree=self._tree,
        )

    def _current_bdf(self) -> str | None:
        """The PCI address of the focused device, when one is selected."""
        if self._last_snapshot is None or not self._last_snapshot.devices:
            return None
        index = min(self._index, len(self._last_snapshot.devices) - 1)
        return self._last_snapshot.devices[index].device.bdf

    def _feed_graphs(self, snapshot: GpuSnapshot) -> None:
        """Push the current device's history into the three graphs."""
        history = self._sampler.history()
        index = self._index

        def series(getter: Callable[[AmdgpuMetrics], float | None]) -> tuple[float | None, ...]:
            values: list[float | None] = [
                getter(entry.devices[index]) for entry in history if index < len(entry.devices)
            ]
            values.append(getter(snapshot.devices[index]))
            return tuple(values)

        self.query_one("#util-graph", BrailleGraph).set_series(
            series(lambda m: m.gpu_busy_percent)
        )
        self.query_one("#vram-graph", BrailleGraph).set_series(series(lambda m: m.vram.percent))
        self.query_one("#gtt-graph", BrailleGraph).set_series(series(lambda m: m.gtt.percent))

    def _status(self, snapshot: GpuSnapshot) -> Text:
        """Build the status line."""
        text = footer_panel(
            self._theme,
            sequence=snapshot.sequence,
            metrics_source=snapshot.source.metrics_abi,
            tick_ms=snapshot.stats.last_duration_ms,
            cpu=self._cpu.format(),
            visible=snapshot.visible_process_count,
            total=snapshot.total_process_count,
            warnings=snapshot.warnings,
            blocks_status=snapshot.blocks_status,
            blocks_hint=snapshot.blocks_hint,
        )
        if len(snapshot.devices) > 1:
            text.append("  ·  ", style=self._theme.track)
            text.append(
                f"GPU {self._index + 1}/{len(snapshot.devices)}",
                style=self._theme.accent,
            )
        notice = self._transient_notice or self._notice
        if notice:
            text.append("  ·  ", style=self._theme.track)
            text.append(notice, style=self._theme.warn)
        return text

    def notify_line(self, message: str) -> None:
        """Show a one-line notice in the status bar until the next sample.

        Transient by design: a reply to a key press belongs to that key press.  A condition
        the user has to act on -- a recording that could not be opened, a state file that
        could not be written -- is assigned to ``_notice`` instead, and stays.
        """
        self._transient_notice = message
        self._notice_until_ns = time.monotonic_ns() + int(NOTICE_LIFETIME_S * 1e9)

    # -- responsiveness -----------------------------------------------------

    def _apply_responsive_layout(self) -> None:
        """Show or hide panels to suit the current terminal size.

        The graphs are the point of the app, so they are never what gets dropped: on a
        short terminal the process table goes first, and on a narrow one the sensor panel
        moves underneath the graphs instead of squeezing them to nothing.

        The result is a pure function of the terminal size, the overview toggle, the
        process-table toggle and the device count, so it is only recomputed when one of
        those changes.  It used to re-assert seven inline styles on every sample, which
        is a stylesheet re-resolve and a repaint of the two biggest panels per tick to
        arrive at the layout that was already there.
        """
        width = self.size.width
        height = self.size.height
        narrow = width < NARROW_WIDTH
        short = height < SHORT_HEIGHT

        key = (width, height, self._overview, self._show_processes, len(self._devices()))
        if key == self._layout_key:
            return
        self._layout_key = key

        main = self.query_one("#main-row")
        graphs = self.query_one("#graphs")
        sensors = self.query_one("#sensors-panel")
        overview = self.query_one("#overview")

        main.styles.layout = "vertical" if narrow else "horizontal"
        graphs.styles.width = "1fr"
        sensors.styles.width = "1fr"
        sensors.styles.min_width = 0 if narrow else 40
        # Stacked, the sensors get a bounded share and scroll; letting them size to their
        # content pushed the graphs off the top of the screen entirely.
        graphs.styles.height = "2fr" if narrow else "1fr"
        sensors.styles.height = "1fr"

        # Overview replaces the detail view rather than sitting beside it: a compact card
        # squeezed into half the width is not compact, it is unreadable.
        overview.display = self._overview
        graphs.display = not self._overview
        sensors.display = not self._overview

        # On a short terminal the graphs keep their space; the table yields.
        self.query_one("#process-panel").display = self._show_processes and not short
        self.query_one("#memory-row").display = height >= MIN_HEIGHT_FOR_MEMORY_GRAPHS
        self._apply_graph_height(height)
        # One tab is not a switcher.  Showing it anyway would spend a row on a fact the
        # header already states and teach a mouse user to click a dead end.
        self.query_one(GpuSwitcher).display = len(self._devices()) > 1

        self._narrow = narrow
        self._short = short

    def _apply_graph_height(self, terminal_height: int) -> None:
        """Size the utilisation graph from ``ui.graph_height``.

        ``0`` -- the default -- leaves the graph at ``1fr``, which is the fluid behaviour
        the interface has always had: the graph takes whatever the layout can spare, so it
        is large on a big terminal and small on a small one without anyone configuring
        anything.

        A positive value pins it to that many rows, which is the point of the key: a user
        who wants a tall graph and is tired of it shrinking every time they resize.  Two
        things have to go with it, and neither is obvious:

        * The panel has to grow to *hold* the graph.  The graph is not the whole panel --
          there is a title above it and the GPU and VRAM readout lines below -- and a
          ``height`` on one child of a ``Vertical`` that does not fit simply crops the
          others.  Pinning the graph alone grew the plot and silently deleted the two
          numbers under it, which is worse than not honouring the key at all.  So the
          panel gets a matching ``min-height`` and the readout is always there.

        * The request is clamped to what the terminal can afford, because a fixed height
          that overflows does not squeeze, it pushes the process table and the footer off
          the bottom of the screen -- the one thing the responsive layout exists to
          prevent.  The constant is measured, not estimated; see
          :data:`MIN_HEIGHT_AROUND_UTIL_GRAPH`.
        """
        panel = self.query_one("#util-panel")
        wanted = self._config.ui.graph_height
        if wanted <= 0:
            panel.styles.min_height = 0
            self.query_one("#util-graph", BrailleGraph).styles.height = "1fr"
            return
        capped = min(
            wanted, max(terminal_height - MIN_HEIGHT_AROUND_UTIL_GRAPH, MIN_UTIL_GRAPH)
        )
        self.query_one("#util-graph", BrailleGraph).styles.height = capped
        panel.styles.min_height = capped + UTIL_PANEL_CHROME_ROWS

    def _devices(self) -> tuple[AmdgpuMetrics, ...]:
        """The devices of the most recent snapshot."""
        if self._last_snapshot is None:
            return ()
        return self._last_snapshot.devices

    # -- actions ------------------------------------------------------------

    def action_refresh_now(self) -> None:
        """Force an immediate sample rather than waiting for the next interval.

        The sampler is *asked* to sample rather than made to: calling ``sample_once()``
        here would run a second tick on the UI thread beside the sampler's own, and two
        ticks sharing the sequence counter, the history ring and the per-process delta
        baseline corrupt the engine utilisation figures.  Waking the loop keeps the single
        owner the whole design rests on.
        """
        self._sampler.request_tick()
        # The loop wakes and publishes in about a tick, so the refresh timer is re-armed
        # short to pick it up: on the default interval this is invisible, but the user may
        # have dialled the period up to ten seconds and pressing R should not then appear
        # to do nothing for ten seconds.
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(REFRESH_NUDGE_S, self._nudge_refresh)

    def _nudge_refresh(self) -> None:
        """Redraw after a refresh-key nudge, then restore the normal cadence.

        Restoring here rather than leaving the timer on the nudge delay matters: a timer
        left at that delay would redraw several times a second for the rest of the session,
        which is the opposite of what one key press asked for.
        """
        self._refresh()
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(self._interval_s, self._refresh)

    def action_toggle_processes(self) -> None:
        """Show or hide the per-process table."""
        self._show_processes = not self._show_processes
        self._redraw()

    def action_cycle_theme(self) -> None:
        """Advance to the next available theme."""
        names = theme_names()
        next_index = (self._theme_index + 1) % len(names)
        self.set_theme(names[next_index])

    def action_focus_device(self, index: int) -> None:
        """Switch the focused device."""
        if self._last_snapshot and index < len(self._last_snapshot.devices):
            self._index = index
            self._redraw()
            self._mark_state_dirty()

    def action_next_device(self) -> None:
        """Focus the next device, wrapping at the end."""
        self._step_device(1)

    def action_previous_device(self) -> None:
        """Focus the previous device, wrapping at the start."""
        self._step_device(-1)

    def _step_device(self, step: int) -> None:
        """Move the focus by ``step`` devices."""
        count = len(self._devices())
        if count <= 1:
            return
        self._index = (self._index + step) % count
        self._redraw()
        self._mark_state_dirty()

    def action_toggle_overview(self) -> None:
        """Show every GPU at once, or go back to the focused one."""
        self._overview = not self._overview
        self._redraw()
        self._mark_state_dirty()

    def action_next_sort(self) -> None:
        """Order by the next column."""
        self._set_sort(self._sort.cycled(1))

    def action_previous_sort(self) -> None:
        """Order by the previous column."""
        self._set_sort(self._sort.cycled(-1))

    def action_reverse_sort(self) -> None:
        """Flip the sort direction."""
        self._set_sort(self._sort.toggled())

    def sort_by(self, column: SortColumn) -> None:
        """Order by ``column``; the active column reverses if it is chosen again."""
        self._set_sort(self._sort.with_column(column))

    def _set_sort(self, state: SortState) -> None:
        """Adopt a new sort and redraw the table.

        Only the table is affected by the sort, so the panels around it are left alone
        rather than repainted for nothing.
        """
        self._sort = state
        self._redraw_processes()
        self._mark_state_dirty()

    def action_clear_filter(self) -> None:
        """Drop the filter text."""
        if self._view.text:
            self._set_view(self._view.with_text(""))

    def action_edit_filter(self) -> None:
        """Prompt for filter text and apply it.

        The result comes back through a callback rather than by awaiting the screen:
        ``push_screen_wait`` has to be called from a worker, and an action is not one.
        A key press is not worth spawning a task to block on.
        """
        self.push_screen(FilterPrompt(self._view.text), self._apply_filter_result)

    def _apply_filter_result(self, result: str | None) -> None:
        """Adopt whatever the filter prompt returned, or keep what was there."""
        if result is None:
            return
        self._set_view(self._view.with_text(result))

    def action_toggle_idle(self) -> None:
        """Show every client, or only the ones doing something."""
        self._set_view(self._view.toggled_idle())

    def action_toggle_tree(self) -> None:
        """Show the process tree, or a flat list."""
        self._tree = not self._tree
        self._redraw_processes()
        self._mark_state_dirty()

    def action_toggle_full_command(self) -> None:
        """Switch between the truncated name and the full command line."""
        self._full_command = not self._full_command
        self._redraw_processes()
        self._mark_state_dirty()

    def action_faster(self) -> None:
        """Sample more often."""
        self._set_interval(self._interval_s - INTERVAL_STEP_MS / 1000.0)

    def action_slower(self) -> None:
        """Sample less often."""
        self._set_interval(self._interval_s + INTERVAL_STEP_MS / 1000.0)

    def _set_interval(self, seconds: float) -> None:
        """Adopt a new sampling interval and restart the refresh timer."""
        clamped = max(MIN_INTERVAL_MS / 1000.0, min(MAX_INTERVAL_MS / 1000.0, seconds))
        if abs(clamped - self._interval_s) < 1e-9:
            return
        self._interval_s = clamped
        self._options = replace(self._options, interval_s=clamped)
        self._sampler.set_interval(clamped)
        # The old timer has to be stopped, not just replaced: Textual's set_interval
        # *adds* a timer, and two of them would sample at twice the requested rate.
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(clamped, self._refresh)
        # The header states the interval, so the whole screen is repainted rather than
        # just that one line: the sampler has already been told, and the next sample will
        # follow on its own.
        self._redraw()
        self._mark_state_dirty()

    def action_select_next(self) -> None:
        """Move the process selection down, bringing the table into focus."""
        self._move_selection(1)

    def action_select_previous(self) -> None:
        """Move the process selection up, bringing the table into focus."""
        self._move_selection(-1)

    def _move_selection(self, step: int) -> None:
        """Move the table cursor and focus the table so the arrows keep working.

        The guard is on ``#process-panel``, the *wrapper* the responsive layout hides,
        not on the table inside it: the table's own ``display`` is never touched, so
        checking it made this guard dead code and the arrows kept moving a selection
        nobody could see.
        """
        if not self.query_one("#process-panel").display:
            return
        panel = self.query_one(ProcessTable)
        panel.step_cursor(step)
        panel.focus()

    def action_kill_selected(self) -> None:
        """Ask before sending a signal to the selected process."""
        # Same guard as the arrows: a panel the user cannot see is a selection they
        # cannot check, and signalling it is not something to do blind.
        if not self.query_one("#process-panel").display:
            return
        panel = self.query_one(ProcessTable)
        process = panel.selected
        if not self._config.process.allow_kill:
            self.notify_line("signalling disabled: set process.allow_kill = true")
            self._redraw_status()
            return
        if process is None:
            self.notify_line("no process selected")
            self._redraw_status()
            return
        reason = control.refusal(process, own_pid=self._pid)
        if reason is not None:
            self.notify_line(reason)
            self._redraw_status()
            return
        self.push_screen(KillConfirm(process), lambda name: self._send(process, name))

    def _send(self, process: GpuProcess, signal_name: str | None) -> None:
        """Deliver the confirmed signal, or do nothing if it was cancelled.

        The target is the one captured when the dialog opened, not the row that happens
        to be under the cursor when it closes: a list that reorders under a modal is a
        dialog that can describe one process and signal another.
        """
        if signal_name is None:
            return
        outcome = control.terminate(process, signal_name, own_pid=self._pid)
        self.notify_line(
            f"{signal_name} → {process.pid} {process.name}" if outcome is None else outcome
        )
        self._redraw_status()

    def action_help(self) -> None:
        """Show the key reference."""
        self.push_screen(
            HelpOverlay(default_help(self._signal_help(), blocks_note=self._blocks_help_note()))
        )

    def _blocks_help_note(self) -> str:
        """A sentence about the blocks panel, only when it is not currently showing."""
        snapshot = self._last_snapshot
        if snapshot is None or snapshot.blocks_status is BlocksStatus.OK:
            return ""
        if self.query_one("#blocks-panel").display:
            return ""
        return f"blocks: {snapshot.blocks_hint or snapshot.blocks_status.hint}"

    def _signal_help(self) -> str:
        """What ``k`` does here, which depends on whether it is allowed to."""
        if self._config.process.allow_kill:
            return "send SIGTERM or SIGKILL to the selected process"
        return "disabled: set process.allow_kill = true"

    # -- view state ---------------------------------------------------------

    @property
    def _pid(self) -> int:
        """gputop's own process id, so a stray selection can never signal the monitor."""
        return os.getpid()

    def _set_view(self, filters: FilterState) -> None:
        """Adopt new filter state and redraw."""
        self._view = filters
        self._redraw_processes()
        self._mark_state_dirty()

    def _redraw(self) -> None:
        """Repaint from the snapshot already in hand.

        Only :meth:`action_refresh_now` waits for a new sample.  A key that changes what
        is *shown* -- which GPU, which view, whether the table exists -- has everything
        it needs already, and re-rendering is what makes the key feel instant instead of
        up to a second late.
        """
        if self._drawable() and self._last_snapshot is not None:
            self._render(self._last_snapshot)

    def _drawable(self) -> bool:
        """Whether there is still a screen to draw into."""
        return self._live and self.is_running

    def _redraw_processes(self) -> None:
        """Rebuild the process table from the last snapshot."""
        if self._drawable() and self._last_snapshot is not None:
            self._render_processes(self._last_snapshot)

    def _redraw_status(self) -> None:
        """Repaint the status line so a notice appears immediately."""
        if self._drawable() and self._last_snapshot is not None:
            self.query_one("#status", Static).update(self._status(self._last_snapshot))

    def _mark_state_dirty(self) -> None:
        """Note that the saved state is out of date."""
        if self._state_path is not None:
            self._state_dirty = True

    def current_state(self) -> StateConfig:
        """The session state as it would be written to disk."""
        return StateConfig(
            theme=self._theme.name,
            interval_ms=round(self._interval_s * 1000),
            sort=self._sort.column,
            sort_reverse=self._sort.reverse,
            filter=self._view.text,
            show_idle=self._view.show_idle,
            process_tree=self._tree,
            full_command=self._full_command,
            overview=self._overview,
            gpu_index=self._index,
        )

    def save_state_now(self) -> str | None:
        """Write the session state, returning a message when it could not be written."""
        if self._state_path is None:
            return None
        self._state_dirty = False
        return save_state(self.current_state(), self._state_path)

    def _flush_state(self) -> None:
        """Write the session state if anything changed since the last write."""
        if self._state_dirty:
            problem = self.save_state_now()
            if problem is not None:
                # A full disk or a read-only home must not turn into a crash on the way
                # out; the monitor has been doing its job until this instant.
                self._state_dirty = False
                self._notice = problem

    # -- messages -----------------------------------------------------------

    def on_gpu_switcher_device_clicked(self, event: GpuSwitcher.DeviceClicked) -> None:
        """Switch GPU from a click on the tab bar."""
        event.stop()
        self.action_focus_device(event.index)

    def on_process_table_sort_requested(self, event: ProcessTable.SortRequested) -> None:
        """Sort by a clicked column heading."""
        event.stop()
        self.sort_by(event.column)

    def on_process_table_row_selected(self, event: ProcessTable.RowSelected) -> None:
        """Accept a row chosen by mouse or keyboard.

        The widget has already moved its own cursor; the app only needs to redraw the
        status line so anything referring to the selection is current.
        """
        event.stop()
        self._redraw_status()


def _join(rows: list[Text]) -> Text:
    """Stack panel rows into a single ``Text`` block.

    A Rich ``Group`` of several ``Text`` objects is not a layout container: with
    ``no_wrap`` set the rows bleed into one another rather than stacking, which is what
    made the sensor rows overlap and run "500 MHz" into the next row's label.  One
    ``Text`` with explicit newlines renders as the block it is meant to be.
    """
    combined = Text(no_wrap=True, overflow="crop", end="")
    for index, row in enumerate(rows):
        if index:
            combined.append("\n")
        combined.append_text(row)
    return combined


def run_app(
    config: Config,
    options: SamplerOptions,
    *,
    state_path: Path | None = None,
    log_target: Path | None = None,
) -> int:
    """Run the Textual application and return a process exit status.

    Device discovery happens here, before the event loop starts.  It is the only sysfs
    read the main thread performs, and doing it up front keeps the loop free of I/O for
    the whole run.

    Args:
        config: The loaded configuration.
        options: Sampler options merged from the same configuration.
        state_path: Where the session state is written, or ``None`` to write none.  The
            tests pass ``None`` so that running them cannot rewrite a developer's real
            settings; the CLI passes the resolved default.
        log_target: Where to record the session, or ``None`` to record nothing.
    """
    app = GpuTopApp(config, options, state_path=state_path, log_target=log_target)
    app.discover()
    with contextlib.suppress(KeyboardInterrupt):
        app.run()
    return 0


__all__ = [
    "MIN_HEIGHT_FOR_MEMORY_GRAPHS",
    "MIN_HEIGHT_FOR_OPTIONAL_ROW",
    "MIN_HEIGHT_FOR_SENSORS_COMPLETE",
    "MIN_OPTIONAL_PANEL_WIDTH",
    "NARROW_WIDTH",
    "SHORT_HEIGHT",
    "BlocksPanel",
    "GpuTopApp",
    "Panel",
    "ProfilePanel",
    "run_app",
    "short_bdf",
]
