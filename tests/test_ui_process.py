"""The per-process panel and everything around it: sorting, filtering, the tree,
multi-GPU switching, the mouse, and the confirmation before a signal is sent.

The decisions these exercise are unit-tested in :mod:`tests.test_procview`; what is
tested here is that the *keys, the mouse and the widgets* reach them, and that the state
survives a restart.
"""

import asyncio
import shutil
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets._data_table import RowDoesNotExist
from rich.text import Text

from gputop import control
from gputop.config import (
    Config,
    GeneralConfig,
    GpuConfig,
    ProcessConfig,
    StateConfig,
    UiConfig,
    apply_state,
)
from gputop.model.process import SORT_COLUMNS
from gputop.sampler import SamplerOptions
from gputop.state import STATE_SECTION, load_state, read_document
from gputop.ui.app import GpuTopApp
from gputop.ui.devices import GpuSwitcher, tab_spans
from gputop.ui.proctable import ProcessTable
from gputop.ui.procview import SortState
from tests.conftest import engine_line, make_gpu, make_process
from tests.snapshot import (
    run_app,
    screen_lines,
    wait_for_stable_layout,
    wait_until,
)

GiB = 1024**3
MiB = 1024**2

FIXED_SYSFS = {
    "mem_info_vis_vram_used": str(4 * GiB),
    "mem_info_vis_vram_total": str(16 * GiB),
    "mem_info_gtt_used": str(1 * GiB),
    "mem_info_gtt_total": str(8 * GiB),
    "gpu_busy_percent": "37",
}


def build_dgpu(
    drm_root: Path, *, bdf: str = "0000:0c:00.0", card: int = 0, busy: int = 37
) -> None:
    """A discrete card with a small dedicated pool."""
    make_gpu(
        drm_root,
        card=card,
        bdf=bdf,
        device_id=0x73BF,
        metrics={"average_gfx_activity": busy, "temperature_edge": 49},
        hwmon={"power1_average": "45000000", "power1_cap": "211000000"},
        extra_sysfs={**FIXED_SYSFS, "gpu_busy_percent": str(busy)},
    )


def build_igpu(drm_root: Path) -> None:
    """An APU: small VRAM carve-out, large GTT, which is what makes it an iGPU."""
    make_gpu(
        drm_root,
        card=1,
        bdf="0000:03:00.0",
        device_id=0x164E,
        extra_sysfs={
            "mem_info_vis_vram_used": str(128 * MiB),
            "mem_info_vis_vram_total": str(512 * MiB),
            "mem_info_gtt_used": str(3 * GiB),
            "mem_info_gtt_total": str(12 * GiB),
            "mem_info_vram_vendor": "N/A",
            "gpu_busy_percent": "6",
        },
    )


def build_clients(proc_root: Path) -> None:
    """Four clients with a deliberate parent/child relationship and split memory."""
    make_process(
        proc_root,
        4242,
        name="renderfarm",
        cmdline=["/usr/bin/renderfarm", "--scene", "42"],
        client_id=1,
        fdinfo=[
            f"drm-resident-vram: {256 * MiB} B",
            f"drm-resident-gtt: {64 * MiB} B",
            engine_line("gfx", 900_000_000),
        ],
    )
    make_process(
        proc_root,
        22871,
        name="firefox",
        ppid=4242,
        cmdline=["/usr/lib/firefox/firefox"],
        client_id=2,
        fdinfo=[f"drm-resident-vram: {84 * MiB} B", engine_line("gfx", 400_000_000)],
    )
    make_process(
        proc_root,
        30114,
        name="python3",
        cmdline=["/usr/bin/python3.13", "train.py"],
        client_id=3,
        fdinfo=[f"drm-resident-gtt: {900 * MiB} B", engine_line("compute", 700_000_000)],
    )
    make_process(
        proc_root,
        1042,
        name="Xorg",
        ppid=1,
        cmdline=["/usr/lib/xorg/Xorg", "-nolisten", "tcp"],
        client_id=4,
        fdinfo=[f"drm-resident-vram: {38 * MiB} B", engine_line("gfx", 5_000_000)],
    )
    # An idle client: holds a descriptor, uses nothing.  The table hides it by default.
    make_process(proc_root, 9999, name="dbus-daemon", client_id=5)


#: ``(pid, client id, engine, extra nanoseconds)`` for :func:`ramp`.
#:
#: The ratios matter more than the absolute numbers: they have to stay ordered under
#: *any* sampling window the test happens to get, and well under 100% so that the clamp
#: never turns two of them into a tie.
BUSY: tuple[tuple[int, int, str, int], ...] = (
    (4242, 1, "gfx", 8_000_000),
    (22871, 2, "gfx", 4_000_000),
    (30114, 3, "compute", 6_000_000),
    (1042, 4, "gfx", 200_000),
)


def ramp(proc_root: Path) -> None:
    """Advance every client's engine counter by a known amount.

    ``drm-engine-*`` is *cumulative*, so a fixture whose counters never move has no
    utilisation at all -- which is honest for a static tree and useless for testing
    anything that depends on the GPU being busy.
    """
    for pid, _client, engine, delta_ns in BUSY:
        path = proc_root / str(pid) / "fdinfo" / "9"
        text = path.read_text()
        key = f"drm-engine-{engine}: "
        base = int(text.split(key)[1].split(" ")[0])
        path.write_text(text.replace(f"{key}{base} ns", f"{key}{base + delta_ns} ns"))


async def _ramp_forever(proc_root: Path, stop: asyncio.Event) -> None:
    """Keep advancing the counters until ``stop`` is set.

    One bump is not enough.  The sampler's tick and the interface's repaint both run at
    the configured interval, so a delta that exists for a single tick is regularly
    discarded before anything is painted -- which is FR2 working as designed, and exactly
    what makes a one-shot fixture flaky rather than wrong.
    """
    while not stop.is_set():
        ramp(proc_root)
        await asyncio.sleep(0.03)


def _is_busy(app: GpuTopApp) -> bool:
    """Whether the snapshot on screen shows any client doing GPU work."""
    snapshot = app._last_snapshot
    return snapshot is not None and any(p.engine_percent > 0 for p in snapshot.processes)


async def make_busy(app: GpuTopApp, proc_root: Path, pilot: Pilot[Any]) -> None:
    """Make the clients busy, and wait until that is what the interface is showing."""
    stop = asyncio.Event()
    pump = asyncio.create_task(_ramp_forever(proc_root, stop))
    try:
        await wait_until(lambda: _is_busy(app), what="a sample showing GPU work")
        await pilot.pause()
    finally:
        stop.set()
        await pump


def build_one_gpu(drm_root: Path, proc_root: Path) -> None:
    """A discrete GPU with clients on it."""
    build_dgpu(drm_root)
    build_clients(proc_root)


def build_with_kernel_thread(drm_root: Path, proc_root: Path) -> None:
    """The same machine plus a client with no argument vector, i.e. a kernel thread."""
    build_one_gpu(drm_root, proc_root)
    make_process(
        proc_root,
        7777,
        name="kworker/18:2",
        cmdline=(),
        client_id=7,
        fdinfo=[f"drm-resident-vram: {4 * MiB} B", engine_line("gfx", 1_000_000)],
    )


def build_two_gpus(drm_root: Path, proc_root: Path) -> None:
    """A discrete card *and* an APU, with a client on each."""
    build_dgpu(drm_root)
    build_igpu(drm_root)
    build_clients(proc_root)
    make_process(
        proc_root,
        5555,
        name="compiz",
        pdev="0000:03:00.0",
        client_id=6,
        fdinfo=[f"drm-resident-gtt: {400 * MiB} B", engine_line("gfx", 250_000_000)],
    )


def make_app(
    drm_root: Path,
    proc_root: Path,
    *,
    build: Callable[[Path, Path], None] = build_one_gpu,
    state_path: Path | None = None,
    state_file: Path | None = None,
    interval_ms: int = 100,
    **process: object,
) -> GpuTopApp:
    """Build an app bound to the fake trees.

    ``state_file`` is a previously saved ``[state]`` section to start from, which is how
    "what the user had last time" is exercised without going through a whole first run.
    """
    build(drm_root, proc_root)
    config = Config(
        general=GeneralConfig(interval_ms=interval_ms, history_points=64),
        gpu=GpuConfig(),
        process=ProcessConfig(show=True, max_rows=12, **process),  # type: ignore[arg-type]
        ui=UiConfig(),
        state=StateConfig(),
    )
    if state_file is not None:
        config = apply_state(config, load_state(state_file))
    return GpuTopApp(
        config,
        # The options carry the merged interval: in production ``sampler_options`` builds
        # them from the config *and* the command line, and the app reads the interval from
        # them, because ``gputop -i 0.5`` reaches nothing else.
        SamplerOptions(drm_root=drm_root, proc_root=proc_root, interval_s=interval_ms / 1000),
        state_path=state_path,
    )


def capture[T](app: GpuTopApp, body: Callable[[Pilot[Any]], Awaitable[T]], size=(140, 44)) -> T:
    """Run the app until it has settled, then capture something from it.

    Every test in this file is about the process table, so the readiness condition is
    that the table has data: the sampler's first sample establishes an engine baseline
    and its second carries the first delta, and a body that runs before either has an
    empty table to assert about.
    """

    async def guarded(pilot: Pilot[Any]) -> T:
        await wait_until(
            lambda: app._last_snapshot is not None and bool(app._last_snapshot.processes),
            what="a first sample with processes",
        )
        return await body(pilot)

    return asyncio.run(run_app(app, size=size, body=guarded, settle=3))


async def press(pilot: Pilot[Any], *keys: str) -> None:
    """Press keys, then let the screen repaint before anything is captured.

    Without the pause a capture can read the compositor one frame behind the action, and
    the test then fails only when the machine is busy enough for the repaint to lose the
    race -- which is the worst possible failure mode for a test.
    """
    for key in keys:
        await pilot.press(key)
        await wait_for_stable_layout(pilot.app)


def table_of(app: GpuTopApp) -> ProcessTable:
    """The process table widget."""
    return app.query_one(ProcessTable)


def _marked_style(table: ProcessTable) -> str:
    """The style of the busiest row's first cell, which carries the accent colour.

    Unmarked rows hold plain strings; the marker is the only cell that is styled, so that
    is how the marked row is found without reaching into the widget's private state.
    """
    for row in table.rows:
        cell = table.grid.get_row(f"{row.process.bdf}:{row.process.client_id}")[0]
        # Marked rows are the only ones whose cells carry a style; plain (unmarked) cells
        # have style == "" (not None), so test truthiness rather than ``is not None``.
        if isinstance(cell, Text) and cell.style:
            return str(cell.style.color)
    return ""


def _marked_heading(table: ProcessTable) -> str:
    """The column heading currently carrying the sort marker."""
    for label in table.headings.values():
        if "▴" in label or "▾" in label:
            return label
    return ""


#: Rows of column heading above the first data row.  ``get_row_at`` counts rows alone
#: while a click offset is measured from the widget's corner, so the two differ by this.
HEADING_HEIGHT = 1


def row_offset(app: GpuTopApp, pid: int) -> int:
    """The offset within the table widget where a click selects ``pid``.

    Guessing an offset would mean re-deriving the heading height and the scroll position,
    which is exactly the arithmetic a click is about; asking the widget is both simpler
    and the thing actually under test.  The one constant this still assumes is checked by
    the test that uses it: a click that missed leaves the selection unchanged and fails.
    """
    grid = table_of(app).grid
    for offset in range(grid.size.height):
        try:
            cells = grid.get_row_at(offset)
        except RowDoesNotExist:
            # Past the last populated row: the empty space below the list.
            continue
        if cells and str(cells[0]) == str(pid):
            return offset + HEADING_HEIGHT
    raise AssertionError(f"pid {pid} is not on screen")


def select(app: GpuTopApp, pid: int) -> None:
    """Put the cursor on a specific process, wherever it happens to be in the order."""
    table = table_of(app)
    for index, row in enumerate(table.rows):
        if row.process.pid == pid:
            table.move_cursor(index)
            return
    raise AssertionError(f"pid {pid} is not in the table")


def shown_pids(app: GpuTopApp) -> list[int]:
    """The pids the table is currently showing, in the order it shows them."""
    rows = table_of(app).rows
    return [row.process.pid for row in rows]


def screen(app: GpuTopApp) -> str:
    """The rendered screen as one string."""
    return "\n".join(screen_lines(app, 140))


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    """``s`` and ``S`` cycle the column; ``r`` flips the direction."""

    def test_default_is_total_engine_load(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> SortState:
            return app._sort

        assert capture(app, body) == SortState(column="total", reverse=True)

    def test_s_advances_the_column(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> SortState:
            await press(pilot, "s")
            return app._sort

        assert capture(app, body).column == SORT_COLUMNS[1]

    def test_s_walks_every_column(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[str]:
            seen = [app._sort.column]
            for _ in SORT_COLUMNS:
                await press(pilot, "s")
                seen.append(app._sort.column)
            return seen

        seen = capture(app, body)
        assert set(seen) == set(SORT_COLUMNS)
        assert seen[0] == seen[-1]

    def test_shift_s_goes_backwards(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> SortState:
            await press(pilot, "S")
            return app._sort

        state = capture(app, body)
        assert state.column == SORT_COLUMNS[-1]

    def test_r_reverses_without_changing_the_column(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> SortState:
            await press(pilot, "r")
            return app._sort

        state = capture(app, body)
        assert state.column == "total"
        assert state.reverse is False

    def test_r_twice_returns_to_the_start(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> SortState:
            await press(pilot, "r")
            await press(pilot, "r")
            return app._sort

        assert capture(app, body) == SortState(column="total", reverse=True)

    def test_the_heading_carries_the_marker(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> dict[str, str]:
            await press(pilot, "s")
            return table_of(app).headings

        headings = capture(app, body)
        assert headings["pid"] == "PID ▾"
        assert headings["gfx"] == "GFX %"

    def test_the_marker_moves_when_the_column_does(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> dict[str, str]:
            await press(pilot, "s")
            return table_of(app).headings

        headings = capture(app, body)
        marked = [column for column, label in headings.items() if "▾" in label]
        assert marked == ["pid"]

    def test_rows_follow_the_sort(self, drm_root: Path, proc_root: Path) -> None:
        """One ``s`` lands on PID, which starts descending: the highest pid first."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[int]:
            await press(pilot, "s")
            return shown_pids(app)

        pids = capture(app, body)
        assert pids == sorted(pids, reverse=True)

    def test_reversing_puts_the_lowest_pid_first(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[int]:
            await press(pilot, "s")
            await press(pilot, "r")
            return shown_pids(app)

        pids = capture(app, body)
        assert pids == sorted(pids)

    def test_the_panel_says_how_it_is_sorted(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "r")
            return screen(app)

        assert "sort total ▴" in capture(app, body)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    """``/`` prompts, ``escape`` clears, ``a`` reveals the idle clients."""

    def test_idle_clients_are_hidden_by_default(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return screen(app)

        text = capture(app, body)
        assert "renderfarm" in text
        assert "dbus-daemon" not in text

    def test_a_reveals_them(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "a")
            return screen(app)

        text = capture(app, body)
        assert "dbus-daemon" in text
        assert "idle shown" in text

    def test_a_twice_hides_them_again(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "a")
            await press(pilot, "a")
            return screen(app)

        assert "dbus-daemon" not in capture(app, body)

    def test_the_prompt_filters(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            return screen(app)

        text = capture(app, body)
        assert "firefox" in text
        assert "renderfarm" not in text

    def test_the_filter_is_stated_in_the_heading(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            return screen(app)

        assert "filter 'fire'" in capture(app, body)

    def test_escape_clears_the_filter(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            narrow = screen(app)
            await press(pilot, "escape")
            return narrow, screen(app)

        narrow, wide = capture(app, body)
        assert "renderfarm" not in narrow
        assert "renderfarm" in wide

    def test_cancelling_the_prompt_keeps_the_old_filter(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            await press(pilot, "slash")
            for key in "zzz":
                await pilot.press(key)
            await press(pilot, "escape")
            return app._view.text

        assert capture(app, body) == "fire"

    def test_the_prompt_opens_with_the_current_text(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            await press(pilot, "slash")
            await pilot.pause()
            return screen(app)

        assert "fire" in capture(app, body)

    def test_the_minimum_usage_threshold_is_honoured(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A threshold of one percent drops everything the ramped clients do not reach."""
        app = make_app(drm_root, proc_root, min_usage_percent=1.0)

        async def body(pilot: Pilot[Any]) -> str:
            await make_busy(app, proc_root, pilot)
            return screen(app)

        text = capture(app, body)
        assert "renderfarm" in text
        assert "Xorg" not in text


# ---------------------------------------------------------------------------
# The tree and the command line
# ---------------------------------------------------------------------------


class TestTreeAndCommand:
    """``t`` nests by parent, ``c`` shows the full command line."""

    def test_the_tree_is_off_by_default(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> bool:
            return app._tree

        assert capture(app, body) is False

    def test_t_nests_the_children(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "t")
            return screen(app)

        text = capture(app, body)
        assert "└─" in text or "├─" in text
        assert "tree" in text

    def test_t_puts_firefox_under_renderfarm(self, drm_root: Path, proc_root: Path) -> None:
        """``PPid`` 4242 is renderfarm, so the row must be indented under it."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[tuple[int, int]]:
            await press(pilot, "t")
            return [(row.process.pid, row.depth) for row in table_of(app).rows]

        depths = dict(capture(app, body))
        assert depths[4242] == 0
        assert depths[22871] == 1

    def test_t_twice_returns_to_a_flat_list(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "t")
            await press(pilot, "t")
            return screen(app)

        text = capture(app, body)
        assert "└─" not in text
        assert "├─" not in text

    def test_the_short_name_is_shown_by_default(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return screen(app)

        text = capture(app, body)
        assert "renderfarm" in text
        assert "/usr/bin/renderfarm" not in text

    def test_c_shows_the_full_command_line(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "c")
            return screen(app)

        assert "/usr/bin/renderfarm" in capture(app, body)

    def test_c_twice_returns_to_the_name(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "c")
            await press(pilot, "c")
            return screen(app)

        assert "/usr/bin/renderfarm" not in capture(app, body)

    def test_a_kernel_thread_never_reaches_the_table(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "a")
            return screen(app)

        # dbus-daemon has a command line here, so it is a userspace process and does show.
        assert "dbus-daemon" in capture(app, body)


# ---------------------------------------------------------------------------
# The busiest client
# ---------------------------------------------------------------------------


class TestHighlight:
    """The row using the most GPU is marked, and the panel says who."""

    def test_the_busiest_client_is_named(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await make_busy(app, proc_root, pilot)
            return screen(app)

        assert "top renderfarm" in capture(app, body)

    def test_nothing_is_named_when_nothing_is_busy(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return screen(app)

        text = capture(app, body)
        assert "top " not in text
        assert "idle" in text


# ---------------------------------------------------------------------------
# Multi-GPU
# ---------------------------------------------------------------------------


class TestMultipleGpus:
    """Switching, scoping the table, and the compact overview."""

    def test_the_tab_bar_appears_with_two_gpus(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(_pilot: Pilot[Any]) -> bool:
            return app.query_one(GpuSwitcher).display

        assert capture(app, body) is True

    def test_the_tab_bar_is_hidden_with_one_gpu(self, drm_root: Path, proc_root: Path) -> None:
        """A single tab is not a switcher, and the header already says which GPU."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> bool:
            return app.query_one(GpuSwitcher).display

        assert capture(app, body) is False

    def test_the_tab_bar_names_both_gpus(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(_pilot: Pilot[Any]) -> str:
            return screen(app)

        text = capture(app, body)
        assert "AMD Radeon RX 6800" in text
        assert "AMD Radeon Graphics (Raphael)" in text

    def test_tab_moves_to_the_next_gpu(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "tab")
            return app._index

        assert capture(app, body) == 1

    def test_tab_wraps_back_to_the_first(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "tab")
            await press(pilot, "tab")
            return app._index

        assert capture(app, body) == 0

    def test_shift_tab_moves_backwards(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "tab")
            await press(pilot, "shift+tab")
            return app._index

        assert capture(app, body) == 0

    def test_a_number_key_jumps_to_a_gpu(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "2")
            return app._index

        assert capture(app, body) == 1

    def test_a_number_key_past_the_end_does_nothing(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "5")
            return app._index

        assert capture(app, body) == 0

    def test_the_header_follows_the_focused_gpu(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "tab")
            return screen(app)

        text = capture(app, body)
        assert "0000:03:00.0" in text
        assert "GPU 2/2" in text

    def test_the_table_is_scoped_to_the_focused_gpu(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``drm-pdev`` routes a client to one card, so the table follows the switch."""
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "tab")
            return screen(app)

        text = capture(app, body)
        assert "compiz" in text
        assert "renderfarm" not in text

    def test_the_memory_panel_leads_with_gtt_on_the_apu(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "tab")
            return screen(app)

        text = capture(app, body)
        assert text.index("GTT") < text.index("VRAM")

    def test_the_overview_shows_every_gpu_at_once(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "o")
            return screen(app)

        text = capture(app, body)
        assert "RX 6800" in text
        assert "Raphael" in text
        assert "focused" in text

    def test_the_overview_hides_the_detail_panels(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> tuple[bool, bool, bool]:
            await press(pilot, "o")
            return (
                app.query_one("#graphs").display,
                app.query_one("#sensors-panel").display,
                app.query_one("#overview").display,
            )

        graphs, sensors, overview = capture(app, body)
        assert (graphs, sensors, overview) == (False, False, True)

    def test_the_overview_shows_clients_from_every_gpu(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The mode that says "everything at once" must not scope the table to one card."""
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "o")
            return screen(app)

        text = capture(app, body)
        assert "renderfarm" in text
        assert "compiz" in text

    def test_the_overview_toggles_back(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> bool:
            await press(pilot, "o")
            await press(pilot, "o")
            return app.query_one("#overview").display

        assert capture(app, body) is False

    def test_the_overview_survives_a_narrow_terminal(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            await press(pilot, "o")
            return max(len(line) for line in screen_lines(app, 64))

        assert capture(app, body, size=(64, 44)) <= 64


# ---------------------------------------------------------------------------
# The mouse
# ---------------------------------------------------------------------------


class TestMouse:
    """Selecting rows, switching GPUs and moving focus, all with a pointer."""

    def test_a_click_on_a_tab_switches_gpu(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            switcher = app.query_one(GpuSwitcher)
            await wait_until(lambda: switcher.size.width > 0, what="the tab bar to be sized")
            # Click a cell the geometry says belongs to the second tab.
            await pilot.click(switcher, offset=(tab_spans(switcher.devices)[1][0] + 2, 0))
            await pilot.pause()
            return app._index

        assert capture(app, body) == 1

    def test_a_click_on_the_focused_tab_is_harmless(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            switcher = app.query_one(GpuSwitcher)
            await wait_until(lambda: switcher.size.width > 0, what="the tab bar to be sized")
            await pilot.click(switcher, offset=(1, 0))
            await pilot.pause()
            return app._index

        assert capture(app, body) == 0

    def test_a_click_past_the_tabs_does_nothing(self, drm_root: Path, proc_root: Path) -> None:
        """The bar is narrower than its row; the empty space past it is not a tab."""
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> int:
            switcher = app.query_one(GpuSwitcher)
            await wait_until(lambda: switcher.size.width > 0, what="the tab bar to be sized")
            last_tab = tab_spans(switcher.devices)[-1]
            assert last_tab[1] < switcher.size.width, "the fixture must leave empty space"
            await pilot.click(switcher, offset=(last_tab[1] + 2, 0))
            await pilot.pause()
            return app._index

        assert capture(app, body) == 0

    def test_a_click_on_a_panel_gives_it_focus(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await pilot.click("#util-graph")
            return screen(app)

        assert "▸ GPU utilisation" in capture(app, body)

    def test_the_focus_marker_moves_with_the_focus(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            await pilot.click("#util-graph")
            first = screen(app)
            await pilot.click("#process-table")
            return first, screen(app)

        first, second = capture(app, body)
        assert "▸ GPU utilisation" in first
        assert "▸ GPU utilisation" not in second

    def test_a_click_selects_a_row(self, drm_root: Path, proc_root: Path) -> None:
        """The clicked row becomes the selection, whichever row that happens to be.

        The y offset is found from the table's own hit map rather than guessed, so the
        test does not depend on the header height or on whether the list is scrolled.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> int:
            table = table_of(app)
            await wait_until(lambda: table.size.height > 3, what="the table to be sized")
            target = table.rows[1].process
            await pilot.click("#process-table", offset=(2, row_offset(app, target.pid)))
            await wait_until(
                lambda: table.selected is not None and table.selected.pid == target.pid,
                what="the clicked row to be selected",
            )
            selected = table.selected
            assert selected is not None
            return selected.pid

        assert capture(app, body) == 4242

    def test_the_arrow_keys_move_the_selection(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[int]:
            table = table_of(app)
            seen = [table.cursor_row]
            for step in range(1, 4):
                await press(pilot, "down")
                await wait_until(
                    lambda: table.cursor_row >= step,  # noqa: B023 - step is read at once
                    what=f"the cursor past row {step - 1}",
                )
                seen.append(table.cursor_row)
            return seen

        assert capture(app, body) == [0, 1, 2, 3]

    def test_the_selection_does_not_run_off_the_end(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int, int]:
            for _ in range(30):
                await press(pilot, "down")
            return table_of(app).cursor_row, len(shown_pids(app))

        cursor, rows = capture(app, body)
        assert cursor == rows - 1

    def test_the_selection_follows_the_process_not_the_row(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A row index would drift onto a different process as the table changes."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int | None, int | None]:
            table_of(app).move_cursor(2)
            chosen = table_of(app).selected
            app._sort = SortState(column="pid", reverse=False)
            app._redraw_processes()
            after = table_of(app).selected
            return chosen.pid if chosen else None, after.pid if after else None

        before, after = capture(app, body)
        assert before is not None
        assert after == before

    def test_the_selection_survives_a_rebuild_after_a_click(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A click on an already-highlighted row is not what moves the cursor.

        ``DataTable`` posts ``RowSelected`` only when the clicked row is the one already
        under the cursor; anything else moves the cursor and posts ``RowHighlighted``.
        The rebuild then found no identity to restore and dropped the selection onto row
        0 -- onto whichever process happened to sort first.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int | None, int | None]:
            table = table_of(app)
            table.grid.cursor_coordinate = table.grid.cursor_coordinate._replace(row=2)
            await pilot.pause()
            chosen = table.selected
            # A different order, so the chosen client is no longer where it was left.
            app._sort = SortState(column="pid", reverse=True)
            app._redraw_processes()
            await pilot.pause()
            return (
                chosen.pid if chosen else None,
                table.selected.pid if table.selected else None,
            )

        chosen, after = capture(app, body)
        assert chosen is not None
        assert after == chosen

    def test_a_theme_change_recolours_the_marked_row(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The marker's colour is in the cell's style, not in its text.

        Keying the rebuild on the cell text alone meant an unchanged sample short-circuited
        the refresh, so pressing ``m`` left the busiest row in the old palette until some
        other cell's characters happened to change.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            table = table_of(app)
            await make_busy(app, proc_root, pilot)
            await wait_until(lambda: _marked_style(table) != "", what="the marker to be styled")
            before = _marked_style(table)
            app.set_theme("dracula")
            await pilot.pause()
            return before, _marked_style(table)

        before, after = capture(app, body)
        assert before != ""
        assert after != ""
        assert before != after

    def test_re_sorting_moves_the_heading_arrow(self, drm_root: Path, proc_root: Path) -> None:
        """The arrow has to leave the column it was on even when no row changes."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> tuple[str, str, str]:
            table = table_of(app)
            app.action_next_sort()
            first = _marked_heading(table)
            app.action_next_sort()
            second = _marked_heading(table)
            return first, second, "".join(table.headings.values())

        first, second, headings = capture(app, body)
        assert first != ""
        assert first != second
        assert second in headings


# ---------------------------------------------------------------------------
# Signalling
# ---------------------------------------------------------------------------


class TestSignalling:
    """Disabled by default, confirmed when enabled, and never sent silently."""

    def test_the_key_is_inert_by_default(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "k")
            return screen(app)

        text = capture(app, body)
        assert "allow_kill" in text

    def test_nothing_is_sent_when_disabled(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _spy(monkeypatch)
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "k")

        capture(app, body)
        assert sent == []

    def test_the_dialog_describes_the_target_before_confirming(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> tuple[int, str]:
            pid = table_of(app).selected
            await press(pilot, "k")
            await pilot.pause()
            assert pid is not None
            return pid.pid, screen(app)

        pid, text = capture(app, body)
        assert str(pid) in text
        assert "SIGTERM" in text
        assert "SIGKILL" in text
        assert "t SIGTERM" in text

    def test_escape_cancels(self, drm_root: Path, proc_root: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        sent = _spy(monkeypatch)
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "k")
            await pilot.pause()
            await press(pilot, "escape")

        capture(app, body)
        assert sent == []

    def test_t_sends_sigterm(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _spy(monkeypatch)
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> tuple[int, str]:
            pid = table_of(app).selected
            await press(pilot, "k")
            await pilot.pause()
            await press(pilot, "t")
            assert pid is not None
            # Waited for rather than assumed: the confirmation is a transient notice, so
            # a capture taken immediately after the key can read the compositor one frame
            # behind the action and miss it.  It now lives NOTICE_LIFETIME_S seconds, which
            # is long enough to read at the 100 ms interval this app runs at.
            await wait_until(
                lambda: app._transient_notice is not None, what="the confirmation notice"
            )
            return pid.pid, screen(app)

        pid, text = capture(app, body)
        assert sent == [(pid, signal.SIGTERM)]
        assert "SIGTERM" in text

    def test_k_sends_sigkill(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _spy(monkeypatch)
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> int:
            pid = table_of(app).selected
            assert pid is not None
            await press(pilot, "k")
            await pilot.pause()
            await press(pilot, "k")
            return pid.pid

        assert sent == [(capture(app, body), signal.SIGKILL)]

    def test_the_key_below_the_dialog_is_ignored(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A modal must not let the app's own bindings fire underneath it."""
        sent = _spy(monkeypatch)
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "k")
            await pilot.pause()
            await press(pilot, "s")

        capture(app, body)
        assert sent == []

    def test_a_kernel_thread_is_refused_before_the_dialog(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No dialog is offered for a target that has no userspace task to signal."""
        sent = _spy(monkeypatch)
        app = make_app(
            drm_root,
            proc_root,
            build=build_with_kernel_thread,
            allow_kill=True,
            hide_kernel_threads=False,
        )

        async def body(pilot: Pilot[Any]) -> str:
            select(app, 7777)
            await press(pilot, "k")
            return screen(app)

        text = capture(app, body)
        assert "kernel thread" in text
        assert "Send a signal" not in text
        assert sent == []


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record every signal the app would send."""
    sent: list[tuple[int, int]] = []

    def record(pid: int, number: int) -> None:
        sent.append((pid, number))

    monkeypatch.setattr(control.os, "kill", record)
    return sent


# ---------------------------------------------------------------------------
# Saved state
# ---------------------------------------------------------------------------


class TestPersistence:
    """What the app writes on the way out, and reads back on the way in."""

    def test_a_state_file_is_written(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)
        capture(app, _nothing)
        assert STATE_SECTION in read_document(path)

    def test_the_theme_is_remembered(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "m")
            return app._theme.name

        assert capture(app, body) == "dracula"
        assert read_document(path)[STATE_SECTION]["theme"] == "dracula"

    def test_the_sort_is_remembered(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "s")
            await press(pilot, "s")
            await press(pilot, "r")

        capture(app, body)
        section = read_document(path)[STATE_SECTION]
        assert section["process_sort"] == "user"
        assert section["process_sort_reverse"] is True

    def test_the_interval_is_remembered(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "minus")
            await pilot.pause()

        capture(app, body)
        assert read_document(path)[STATE_SECTION]["interval_ms"] == 200

    def test_plus_and_minus_step_the_interval(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, interval_ms=500)

        async def body(pilot: Pilot[Any]) -> tuple[float, float, float]:
            start = app._interval_s
            await press(pilot, "plus")
            await pilot.pause()
            faster = app._interval_s
            await press(pilot, "minus")
            await pilot.pause()
            return start, faster, app._interval_s

        start, faster, slower = capture(app, body)
        assert start == pytest.approx(0.5)
        assert faster == pytest.approx(0.4)
        assert slower == pytest.approx(0.5)

    def test_the_interval_is_clamped(self, drm_root: Path, proc_root: Path) -> None:
        """A faster tick would spend more time in procfs than in the data it collects."""
        app = make_app(drm_root, proc_root, interval_ms=200)

        async def body(pilot: Pilot[Any]) -> float:
            for _ in range(20):
                await press(pilot, "plus")
            await pilot.pause()
            return app._interval_s

        assert capture(app, body) == pytest.approx(0.1)

    def test_the_filter_is_remembered(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "slash")
            for key in "fire":
                await pilot.press(key)
            await press(pilot, "enter")
            await pilot.pause()

        capture(app, body)
        assert read_document(path)[STATE_SECTION]["process_filter"] == "fire"

    def test_a_saved_state_is_applied_on_startup(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        path.write_text(
            "[state]\ntheme = 'gruvbox'\nprocess_sort = 'vram'\n"
            "process_filter = 'fire'\nshow_process_tree = true\nfull_command = true\n"
        )
        app = make_app(drm_root, proc_root, state_file=path)

        async def body(_pilot: Pilot[Any]) -> tuple[str, SortState, str, bool, bool]:
            return (
                app._theme.name,
                app._sort,
                app._view.text,
                app._tree,
                app._full_command,
            )

        theme, sort, text, tree, full = capture(app, body)
        assert theme == "gruvbox"
        assert sort.column == "vram"
        assert text == "fire"
        assert tree is True
        assert full is True

    def test_a_saved_gpu_index_is_applied(self, drm_root: Path, proc_root: Path) -> None:
        app = GpuTopApp(
            Config(
                general=GeneralConfig(interval_ms=100),
                process=ProcessConfig(show=True),
                ui=UiConfig(),
                state=StateConfig(gpu_index=1),
            ),
            SamplerOptions(drm_root=drm_root, proc_root=proc_root, interval_s=0.1),
        )
        build_two_gpus(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> int:
            return app._index

        assert capture(app, body) == 1

    def test_nothing_is_written_without_a_state_path(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Running the test suite must not rewrite a developer's real settings."""
        app = make_app(drm_root, proc_root)
        capture(app, _nothing)
        assert app.save_state_now() is None

    def test_a_write_failure_is_reported_not_raised(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        app = make_app(drm_root, proc_root, state_path=tmp_path / "missing" / "x" / "y.toml")
        assert isinstance(app.save_state_now(), str | type(None))
        # The directory is created on demand, so this is the success path.
        assert app.save_state_now() is None

    def test_the_saved_state_round_trips(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        app = make_app(drm_root, proc_root, state_path=path)

        async def body(pilot: Pilot[Any]) -> None:
            await press(pilot, "m")
            await press(pilot, "s")
            await press(pilot, "a")
            await press(pilot, "t")
            await press(pilot, "c")
            await press(pilot, "o")
            await pilot.pause()

        capture(app, body)
        section = read_document(path)[STATE_SECTION]
        assert section["theme"] == "dracula"
        assert section["show_idle_processes"] is True
        assert section["show_process_tree"] is True
        assert section["full_command"] is True
        assert section["overview"] is True


async def _nothing(_pilot: Pilot[Any]) -> None:
    """A body that does nothing, for tests that only care about the exit path."""
    return None


class TestMiscellaneous:
    """The remaining small behaviours that changed with this milestone."""

    def test_help_opens_and_closes(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            await press(pilot, "h")
            await pilot.pause()
            opened = screen(app)
            await press(pilot, "escape")
            await pilot.pause()
            return opened, screen(app)

        opened, closed = capture(app, body)
        assert "gputop keys" in opened
        assert "gputop keys" not in closed

    def test_help_lists_the_new_keys(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "h")
            await pilot.pause()
            return screen(app)

        text = capture(app, body)
        for key in (
            "reverse the sort",
            "filter processes",
            "process tree",
            "every GPU at once",
        ):
            assert key in text

    def test_help_says_when_signalling_is_disabled(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "h")
            await pilot.pause()
            return screen(app)

        assert "allow_kill" in capture(app, body)

    def test_help_says_when_signalling_is_enabled(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "h")
            await pilot.pause()
            return screen(app)

        assert "SIGKILL to the selected process" in capture(app, body)

    def test_toggling_the_panel_hides_it(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            before = screen(app)
            await press(pilot, "p")
            return before, screen(app)

        before, after = capture(app, body)
        assert "PROCESSES" in before
        assert "PROCESSES" not in after

    def test_no_line_overflows_a_narrow_terminal(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(pilot: Pilot[Any]) -> list[str]:
            await press(pilot, "t")
            return screen_lines(app, 80)

        for line in capture(app, body, size=(80, 44)):
            assert len(line) <= 80, repr(line)

    def test_no_line_overflows_a_short_terminal(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=build_two_gpus)

        async def body(_pilot: Pilot[Any]) -> list[str]:
            return screen_lines(app, 140)

        for line in capture(app, body, size=(140, 18)):
            assert len(line) <= 140, repr(line)

    def test_the_table_survives_a_resize(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int, int]:
            before = table_of(app).cursor_row
            await pilot.resize_terminal(60, 20)
            await pilot.pause()
            return before, table_of(app).cursor_row

        before, after = capture(app, body, size=(140, 44))
        assert before == after


# ---------------------------------------------------------------------------
# Process text is data, never Rich markup
# ---------------------------------------------------------------------------


class TestProcessTextIsNotMarkup:
    """``DataTable`` parses ``str`` cells as markup; ``/proc`` text is not markup.

    A command line containing ``[/]`` raised ``MarkupError`` and killed the redraw of
    the whole table, and one containing ``[red]`` was silently styled and mangled.  The
    kill dialog had the same problem at the moment the user confirmed a signal.
    """

    def test_a_command_line_containing_markup_renders_verbatim(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        build_dgpu(drm_root)
        make_process(
            proc_root,
            5150,
            name="mapper",
            cmdline=["mapper", "[/]", "[red]"],
            fdinfo=["drm-resident-vram: 1048576 B"],
        )
        app = make_app(drm_root, proc_root, build=lambda _d, _p: None)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "c")
            return screen(app)

        text = capture(app, body)
        assert "[/]" in text
        assert "[red]" in text

    def test_the_kill_dialog_does_the_same(self, drm_root: Path, proc_root: Path) -> None:
        build_dgpu(drm_root)
        make_process(
            proc_root,
            5150,
            name="mapper",
            cmdline=["/usr/bin/mapper", "--map", "[/]"],
            fdinfo=["drm-resident-vram: 1048576 B"],
        )
        app = make_app(drm_root, proc_root, build=lambda _d, _p: None, allow_kill=True)

        async def body(pilot: Pilot[Any]) -> str:
            await press(pilot, "k")
            await pilot.pause()
            return screen(app)

        text = capture(app, body)
        assert "[/]" in text
        assert "SIGTERM" in text


# ---------------------------------------------------------------------------
# The selection's lifecycle
# ---------------------------------------------------------------------------


class TestSelectionLifecycle:
    """A selection follows the process, and stops when the process does."""

    def test_a_selection_that_exits_stops_being_selected(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The selection does not silently transfer to whichever row inherited the cursor.

        A kill dialog aimed at "the selected process" would otherwise aim at whatever
        happened to sort first after the target exited -- naming one process and
        signalling another.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> bool:
            select(app, 4242)
            assert table_of(app).selected is not None
            shutil.rmtree(proc_root / "4242")
            await wait_until(
                lambda: (
                    app._last_snapshot is not None
                    and all(p.pid != 4242 for p in app._last_snapshot.processes)
                ),
                what="a sample without the exited process",
            )
            await wait_until(
                lambda: table_of(app).selected is None,
                what="the selection to stop rather than move",
            )
            return True

        assert capture(app, body) is True

    def test_the_user_can_select_again_afterwards(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The empty selection lasts until the user picks a row, not a sample."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> int | None:
            select(app, 4242)
            shutil.rmtree(proc_root / "4242")
            await wait_until(
                lambda: table_of(app).selected is None,
                what="the selection to stop",
            )
            await press(pilot, "down")
            chosen = table_of(app).selected
            return chosen.pid if chosen is not None else None

        pid = capture(app, body)
        assert pid is not None
        assert pid != 4242


# ---------------------------------------------------------------------------
# Heading clicks and hidden panels
# ---------------------------------------------------------------------------


class TestHeadingClicks:
    """A click on a column heading sorts by that column.

    ``StringKey`` has no ``__str__``, so comparing ``str(column_key)`` against a column
    name compared an object repr and matched nothing: heading clicks were silently
    ignored.
    """

    def test_clicking_a_heading_sorts_by_that_column(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> str:
            # y=0 inside the grid is the heading row; x=2 is inside the first column.
            await pilot.click("#process-table", offset=(2, 0))
            await wait_until(
                lambda: app._sort.column == "pid", what="the heading click to sort"
            )
            return app._sort.column

        assert capture(app, body) == "pid"


class TestHiddenPanelKeys:
    """Keys aimed at the process panel are inert while the panel is hidden.

    The responsive layout hides ``#process-panel`` on a short terminal.  The guard used
    to check the table *inside* it, which is never hidden, so the arrows kept moving a
    selection nobody could see and ``k`` could still signal it.
    """

    def _make(self, drm_root: Path, proc_root: Path) -> GpuTopApp:
        return make_app(drm_root, proc_root, allow_kill=True)

    def test_the_arrows_do_not_move_a_hidden_selection(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = self._make(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int, int]:
            await pilot.resize_terminal(60, 18)
            await pilot.pause()
            assert not app.query_one("#process-panel").display
            before = table_of(app).cursor_row
            await press(pilot, "down")
            return before, table_of(app).cursor_row

        before, after = capture(app, body, size=(60, 18))
        assert before == after == 0

    def test_k_does_not_signal_a_hidden_selection(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = self._make(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> bool:
            await press(pilot, "k")
            await pilot.pause()
            return app._transient_notice is None

        assert capture(app, body, size=(60, 18)) is True
