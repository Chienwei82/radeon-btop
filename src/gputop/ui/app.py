"""The btop-style Textual interface.

Threading contract, which the rest of the program depends on:

* The sampler thread owns all sampling state and publishes immutable snapshots through a
  queue.  This file never performs I/O -- no ``open``, no ``os.readlink``, no sysfs read.
  Everything the UI displays was already collected by the sampler.
* The UI thread only ever *reads* what it receives.  Nothing here mutates a snapshot, and
  nothing here is read by the sampler, so no lock is needed and the app is correct on the
  free-threaded build.
* The one thing the UI does own is a short-lived render buffer, rebuilt each frame on the
  UI thread alone.
"""

import contextlib
import time
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Static

from gputop.config import Config
from gputop.model.process import GpuProcess
from gputop.model.snapshot import GpuSnapshot
from gputop.sampler import Sampler, SamplerOptions
from gputop.ui.format import NA, fmt_bytes, fmt_percent
from gputop.ui.panels import (
    footer_panel,
    header_panel,
    sensors_panel,
    utilisation_rows,
)
from gputop.ui.theme import Gradient, Theme, get_theme, theme_names
from gputop.ui.widgets import BrailleGraph, CpuMeter

#: Process table columns, in order.
PROCESS_COLUMNS = (
    ("PID", 8),
    ("Process", 22),
    ("User", 12),
    ("GPU", 12),
    ("Engine", 10),
    ("VRAM", 11),
    ("GTT", 11),
    ("CPU", 11),
)

#: Below this width the layout collapses to a single column.
NARROW_WIDTH = 72
#: Below this height the per-process table is hidden to keep the graphs legible.
SHORT_HEIGHT = 20
#: Below this height the memory graphs are hidden; an empty bordered box reads as a bug.
MIN_HEIGHT_FOR_MEMORY_GRAPHS = 28

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
    return ":".join(parts[1:]) if len(parts) == 3 else (bdf or NA)


class ProcessTable(Static):
    """The per-client table, backed by a Textual ``DataTable``."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="process-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        for label, width in PROCESS_COLUMNS:
            table.add_column(label, width=width)

    def update_snapshot(self, processes: tuple[GpuProcess, ...], limit: int) -> None:
        """Replace the rows with the current busiest clients."""
        table = self.query_one(DataTable)
        table.clear()
        for process in processes[:limit]:
            table.add_row(
                str(process.pid),
                process.name[:22],
                process.user[:12],
                short_bdf(process.bdf),
                fmt_percent(process.engine_percent, 1),
                fmt_bytes(process.vram_used),
                fmt_bytes(process.gtt_used),
                fmt_bytes(process.cpu_used),
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
    /* Not docked: two bottom-docked rows compete with the Footer for the same edge and
       the panel draws over the status line.  In normal flow it always gets its row. */
    #status-row { height: auto; }
    #status { height: auto; padding: 0 1; }
    #process-panel { height: 1fr; min-height: 5; border: round $accent; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("p", "toggle_processes", "Processes"),
        Binding("m", "cycle_theme", "Theme"),
        Binding("1", "focus_device(0)", "GPU 1"),
        Binding("2", "focus_device(1)", "GPU 2"),
        Binding("3", "focus_device(2)", "GPU 3"),
    ]

    def __init__(self, config: Config, options: SamplerOptions) -> None:
        super().__init__()
        self._config = config
        self._options = options
        self._sampler = Sampler(options)
        self._show_processes = config.process.show
        self._limit = config.process.max_rows

        self._theme: Theme = get_theme(config.ui.theme)
        self._gradient = Gradient(self._theme)
        self._monochrome = config.ui.no_color
        self._theme_index = max(
            0, theme_names().index(self._theme.name) if self._theme.name in theme_names() else 0
        )

        self._index = 0
        self._cpu = CpuMeter()
        self._started_monotonic = time.monotonic()
        self._last_snapshot: GpuSnapshot | None = None
        self._narrow = False
        self._short = False

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="device-header")
        with Horizontal(id="main-row"):
            with Vertical(id="graphs"):
                with Vertical(id="util-panel", classes="panel"):
                    yield Static("GPU utilisation", classes="panel-title")
                    yield BrailleGraph(
                        theme=self._theme,
                        gradient=self._gradient,
                        id="util-graph",
                    )
                    yield Static("", id="util-readout")
                with Horizontal(id="memory-row"):
                    with Vertical(id="vram-panel", classes="panel"):
                        yield Static("VRAM", classes="panel-title")
                        yield BrailleGraph(
                            theme=self._theme, gradient=self._gradient, id="vram-graph"
                        )
                    with Vertical(id="gtt-panel", classes="panel"):
                        yield Static("GTT", classes="panel-title")
                        yield BrailleGraph(
                            theme=self._theme,
                            gradient=self._gradient,
                            id="gtt-graph",
                        )
            with VerticalScroll(id="sensors-panel", classes="panel"):
                yield Static("Sensors", classes="panel-title")
                yield Static("", id="sensors-body")
        with Horizontal(id="status-row"):
            yield Static("", id="status")
        with Vertical(id="process-panel"):
            yield ProcessTable()
        yield Footer()

    def discover(self) -> tuple[object, ...]:
        """Discover devices outside the event loop.

        Discovery reads sysfs.  It belongs to startup, not to a frame, so
        :func:`run_app` calls this before ``App.run``; the fallback in
        :meth:`on_mount` exists only for direct construction in tests.
        """
        return self._sampler.discover()

    def on_mount(self) -> None:
        # Always start, even when discover() already ran: that call only enumerates
        # devices, and skipping start() here left the sampling thread unlaunched so the
        # UI rendered but never updated.  start() is idempotent.
        self._sampler.start()
        if not self._sampler.devices:
            self.query_one("#device-header", Static).update(NO_DEVICE_HELP)
            return

        self._apply_colour_mode()
        # 1 Hz is the floor for the app's own CPU figure: a faster sample would mostly
        # measure the timer's jitter, and the number is for a human reading it.
        self.set_interval(1.0, self._tick_cpu)
        self.set_interval(self._options.interval_s, self._refresh)
        self._refresh()

    def on_unmount(self) -> None:
        self._sampler.stop()

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
        for widget in self.query(BrailleGraph):
            widget.set_theme(self._theme, truecolor=self._gradient.truecolor)
        self._apply_css_variables()
        self._refresh()

    def _apply_css_variables(self) -> None:
        """Push the palette into CSS so borders and titles follow the theme."""
        self.stylesheet.set_variables(
            {
                "accent": self._theme.accent,
                "surface": self._theme.track,
                "text": self._theme.text,
            }
        )

    # -- refresh ------------------------------------------------------------

    def _tick_cpu(self) -> None:
        """Recompute the app's own CPU usage once a second."""
        self._cpu.sample()
        if self._last_snapshot is not None:
            self.query_one("#status", Static).update(self._status(self._last_snapshot))

    def _refresh(self) -> None:
        """Drain queued snapshots and redraw the newest.

        Only the newest snapshot is rendered.  During a slow repaint the queue can hold
        several; rendering them all would show stale data and waste the frame.
        """
        snapshot: GpuSnapshot | None = None
        while True:
            candidate = self._sampler.latest()
            if candidate is None:
                break
            snapshot = candidate
        if snapshot is None:
            return
        self._last_snapshot = snapshot
        self._render(snapshot)

    def _render(self, snapshot: GpuSnapshot) -> None:
        """Draw one snapshot into the widgets."""
        if not snapshot.devices:
            return
        self._index = min(self._index, len(snapshot.devices) - 1)
        metrics = snapshot.devices[self._index]

        self.query_one("#device-header", Static).update(
            header_panel(
                metrics,
                self._theme,
                time.monotonic() - self._started_monotonic,
                self._options.interval_s,
            )
        )
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

        # The graphs are fed from the sampler's history, so a fresh process starts with a
        # graph that fills in rather than one that only ever shows the latest point.
        self._feed_graphs(snapshot)

        panel = self.query_one(ProcessTable)
        panel.display = self._show_processes and not self._short
        if panel.display:
            panel.update_snapshot(snapshot.processes, self._limit)

        self.query_one("#status", Static).update(self._status(snapshot))
        self._apply_responsive_layout()

    def _feed_graphs(self, snapshot: GpuSnapshot) -> None:
        """Push the current device's history into the three graphs."""
        history = self._sampler.history()
        index = self._index

        def series(getter: object) -> tuple[float | None, ...]:
            values: list[float | None] = []
            for entry in history:
                if index >= len(entry.devices):
                    continue
                values.append(getter(entry.devices[index]))  # type: ignore[operator]
            values.append(getter(snapshot.devices[index]))  # type: ignore[operator]
            return tuple(values)

        self.query_one("#util-graph", BrailleGraph).set_series(
            series(lambda m: m.gpu_busy_percent)
        )
        self.query_one("#vram-graph", BrailleGraph).set_series(series(lambda m: m.vram.percent))
        self.query_one("#gtt-graph", BrailleGraph).set_series(series(lambda m: m.gtt.percent))

    def _status(self, snapshot: GpuSnapshot) -> Text:
        """Build the status line."""
        return footer_panel(
            self._theme,
            sequence=snapshot.sequence,
            metrics_source=snapshot.source.metrics_abi,
            tick_ms=snapshot.stats.last_duration_ms,
            cpu=self._cpu.format(),
            visible=snapshot.visible_process_count,
            total=snapshot.total_process_count,
            warnings=snapshot.warnings,
        )

    # -- responsiveness -----------------------------------------------------

    def _apply_responsive_layout(self) -> None:
        """Show or hide panels to suit the current terminal size.

        The graphs are the point of the app, so they are never what gets dropped: on a
        short terminal the process table goes first, and on a narrow one the sensor panel
        moves underneath the graphs instead of squeezing them to nothing.
        """
        width = self.size.width
        height = self.size.height
        narrow = width < NARROW_WIDTH
        short = height < SHORT_HEIGHT

        main = self.query_one("#main-row")
        graphs = self.query_one("#graphs")
        sensors = self.query_one("#sensors-panel")

        main.styles.layout = "vertical" if narrow else "horizontal"
        graphs.styles.width = "1fr"
        sensors.styles.width = "1fr"
        sensors.styles.min_width = 0 if narrow else 40
        # Stacked, the sensors get a bounded share and scroll; letting them size to their
        # content pushed the graphs off the top of the screen entirely.
        graphs.styles.height = "2fr" if narrow else "1fr"
        sensors.styles.height = "1fr"

        # On a short terminal the graphs keep their space; the table yields.
        self.query_one("#process-panel").display = self._show_processes and not short
        self.query_one("#memory-row").display = height >= MIN_HEIGHT_FOR_MEMORY_GRAPHS

        self._narrow = narrow
        self._short = short

    # -- actions ------------------------------------------------------------

    def action_refresh_now(self) -> None:
        """Force an immediate sample rather than waiting for the next interval."""
        self._sampler.sample_once()
        self._refresh()

    def action_toggle_processes(self) -> None:
        """Show or hide the per-process table."""
        self._show_processes = not self._show_processes
        self._apply_responsive_layout()
        self._refresh()

    def action_cycle_theme(self) -> None:
        """Advance to the next available theme."""
        names = theme_names()
        next_index = (self._theme_index + 1) % len(names)
        self.set_theme(names[next_index])

    def action_focus_device(self, index: int) -> None:
        """Switch the focused device."""
        if self._last_snapshot and index < len(self._last_snapshot.devices):
            self._index = index
            self._refresh()


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


def run_app(config: Config, options: SamplerOptions) -> int:
    """Run the Textual application and return a process exit status.

    Device discovery happens here, before the event loop starts.  It is the only sysfs
    read the main thread performs, and doing it up front keeps the loop free of I/O for
    the whole run.
    """
    app = GpuTopApp(config, options)
    app.discover()
    with contextlib.suppress(KeyboardInterrupt):
        app.run()
    return 0
