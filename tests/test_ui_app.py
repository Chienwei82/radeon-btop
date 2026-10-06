"""Integration tests for the app: composition, responsiveness and interaction.

These run against a synthetic sysfs tree so every value on screen is deterministic, which
is what makes a screen snapshot meaningful.  Readiness is awaited on the app's own tick
count rather than a sleep, so the suite is neither slow nor timing-dependent.
"""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Footer

from gputop.config import Config, GeneralConfig, GpuConfig, ProcessConfig, UiConfig
from gputop.model.alerts import AlertLevel
from gputop.readers import discovery, fsutil, procfs
from gputop.sampler import SamplerOptions
from gputop.ui.app import MIN_HEIGHT_AROUND_UTIL_GRAPH, MIN_UTIL_GRAPH, GpuTopApp, short_bdf
from gputop.ui.theme import THEMES, theme_names
from gputop.ui.widgets import BrailleGraph
from tests.conftest import engine_line, make_gpu, make_process
from tests.snapshot import run_app, screen_lines, wait_until, widget_text

GiB = 1024**3
MiB = 1024**2

#: A fixed, fully populated metric table so the screen is reproducible.
FIXED_METRICS = {
    "temperature_edge": 49,
    "temperature_hotspot": 56,
    "temperature_mem": 52,
    "average_gfx_activity": 37,
    "average_socket_power": 45,
    "current_gfxclk": 2200,
    "current_uclk": 1000,
    "pcie_link_width": 16,
    "pcie_link_speed": 80,
    "voltage_soc": 918,
    "voltage_gfx": 768,
    "voltage_mem": 900,
}

FIXED_HWMON = {
    "power1_average": "45000000",
    "power1_cap": "211000000",
    "fan1_input": "2400",
    "fan1_max": "3000",
    "pwm1": "204",
}

FIXED_SYSFS = {
    "mem_info_vis_vram_used": str(4 * GiB),
    "mem_info_vis_vram_total": str(16 * GiB),
    "mem_info_gtt_used": str(1 * GiB),
    "mem_info_gtt_total": str(8 * GiB),
    "pp_dpm_sclk": "0: 500Mhz\n1: 2475Mhz *\n",
    "pp_power_profile_mode": " 0 BOOTUP_DEFAULT*:\n 1 3D_FULL_SCREEN :\n",
}


def build_hardware(drm_root: Path, proc_root: Path) -> None:
    """Create one deterministic GPU and one deterministic client.

    The client is given resident memory rather than left at zero: a real DRM client
    always holds some, and the table hides clients that are demonstrably doing nothing,
    so a zero-memory fixture would describe a process gputop deliberately does not show.
    """
    make_gpu(
        drm_root,
        bdf="0000:0c:00.0",
        device_id=0x73BF,
        metrics=FIXED_METRICS,
        hwmon=FIXED_HWMON,
        extra_sysfs=FIXED_SYSFS,
    )
    make_process(
        proc_root,
        4242,
        name="renderfarm",
        cmdline=["/usr/bin/renderfarm", "--scene", "42"],
        fdinfo=[
            f"drm-resident-vram: {256 * MiB} B",
            f"drm-resident-gtt: {64 * MiB} B",
            engine_line("gfx", 500_000_000),
        ],
    )


def make_config(**ui: object) -> Config:
    """Build a config with a short interval and the process table enabled."""
    return Config(
        general=GeneralConfig(interval_ms=100, history_points=64),
        gpu=GpuConfig(),
        process=ProcessConfig(show=True, max_rows=5),
        ui=UiConfig(**ui),  # type: ignore[arg-type]
    )


def make_app(drm_root: Path, proc_root: Path, *, build: bool = True, **ui: object) -> GpuTopApp:
    """Build an app bound to the fake trees."""
    if build:
        build_hardware(drm_root, proc_root)
    return GpuTopApp(
        make_config(**ui),
        SamplerOptions(drm_root=drm_root, proc_root=proc_root, interval_s=0.1),
    )


def _record(calls: list[str], label: str, original: object) -> object:
    """Wrap a reader so a call from the UI thread is flagged.

    The patch is process-wide and the sampler thread legitimately does all of this I/O
    concurrently, so the wrapper only records main-thread calls -- the thread running the
    event loop -- and otherwise delegates so sampling keeps working normally.
    """

    def guard(*args: object, **kwargs: object) -> object:
        if threading.current_thread() is threading.main_thread():
            calls.append(label)
        return original(*args, **kwargs)  # type: ignore[operator]

    return guard


def capture[T](
    app: GpuTopApp, size: tuple[int, int], body: Callable[[Pilot[Any]], Awaitable[T]]
) -> T:
    """Run the app until it has settled, then capture from it."""
    return asyncio.run(run_app(app, size=size, body=body, settle=3))


class TestNoDevice:
    """A machine without an AMD GPU must explain itself, not crash."""

    def test_help_is_shown(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, build=False)

        async def body(_pilot: Pilot[Any]) -> str:
            return widget_text(app.query_one("#device-header"))

        text = capture(app, (80, 24), body)
        assert "No amdgpu device" in text
        assert "amdgpu" in text


class TestScreenSnapshot:
    """A full-screen snapshot against deterministic hardware."""

    def screen(self, drm_root: Path, proc_root: Path, size: tuple[int, int] = (120, 40)) -> str:
        """Return the rendered screen as one string."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return "\n".join(screen_lines(app, size[0]))

        return capture(app, size, body)

    def test_header_shows_identity_and_timing(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "AMD Radeon RX 6800" in text
        assert "amdgpu" in text
        assert "0000:0c:00.0" in text
        assert "0x73bf" in text
        assert "up " in text
        assert "every 0.1s" in text

    def test_panels_have_rounded_borders(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "╭" in text and "╮" in text
        assert "╰" in text and "╯" in text

    def test_the_alert_flash_reaches_every_bordered_panel(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Two bordered boxes were not ``Panel`` instances and so were never selected.

        ``#vram-panel`` and ``#gtt-panel`` are plain ``Vertical``\\s with their border in
        the stylesheet, and the flash queried ``query(Panel)``.  Two of seven boxes kept
        the calm colour while the rest flashed, which is the one thing a border flash is
        not supposed to do: the docstring's own rule is that "one border that changes says
        'this screen'".
        """
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> list[str]:
            panels = list(app.query(".panel"))
            assert panels, "no bordered panels found"
            app._alert_state = replace(app._alert_state, level=AlertLevel.ALERT)
            app._flash_on = True
            app._apply_alert_borders()
            borders = [str(p.styles.border) for p in panels]
            # The two that were never ``Panel`` instances, named explicitly: they are the
            # ones a type-based query could not reach.
            for panel_id in ("vram-panel", "gtt-panel"):
                assert str(app.query_one(f"#{panel_id}").styles.border) == borders[0]
            return borders

        borders = capture(app, (120, 40), body)
        assert len(borders) == 7
        assert all(border == borders[0] for border in borders), (
            "not every panel border carries the alert colour"
        )

    def test_all_panels_are_present(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        for title in ("GPU utilisation", "VRAM", "GTT", "Sensors", "PID"):
            assert title in text, f"missing panel: {title}"

    def test_sensor_values_are_shown(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "49°C" in text
        assert "2,200 MHz" in text
        assert "45.0 W" in text
        assert "cap 211 W" in text
        assert "Gen3 x16" in text
        assert "throttle none" in text

    def test_memory_readings_are_shown(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "4.0 GiB" in text and "16.0 GiB" in text

    def test_process_row_is_listed(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "4242" in text
        assert "renderfarm" in text

    def test_footer_reports_the_apps_own_cpu(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root)
        assert "cpu " in text
        assert "metrics v1.3" in text

    def test_no_line_overflows_the_terminal(self, drm_root: Path, proc_root: Path) -> None:
        """A row longer than the panel wraps and destroys the layout."""
        for line in self.screen(drm_root, proc_root).splitlines():
            assert len(line) <= 120, repr(line)


class TestGraphs:
    """The three braille graphs."""

    def graph(
        self, drm_root: Path, proc_root: Path, widget_id: str
    ) -> tuple[tuple[float | None, ...], str]:
        """Run the app and capture one graph's samples and render.

        The render happens inside the running app: an unmounted widget reports a zero
        size, so reading it afterwards would compare against nothing.
        """
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> tuple[tuple[float | None, ...], str]:
            graph = app.query_one(widget_id, BrailleGraph)
            return graph.values, graph.render().plain

        return capture(app, (120, 40), body)

    def test_util_graph_receives_samples(self, drm_root: Path, proc_root: Path) -> None:
        values, _ = self.graph(drm_root, proc_root, "#util-graph")
        assert values
        assert all(value is None or 0.0 <= value <= 100.0 for value in values)

    def test_memory_graphs_receive_samples(self, drm_root: Path, proc_root: Path) -> None:
        for widget_id in ("#vram-graph", "#gtt-graph"):
            values, _ = self.graph(drm_root, proc_root, widget_id)
            assert values, f"{widget_id} got no samples"

    def test_graph_renders_uniform_rectangular_rows(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        _values, rendered = self.graph(drm_root, proc_root, "#util-graph")
        lines = rendered.splitlines()
        assert len(lines) > 1, "a graph must be more than one row tall"
        assert len({len(line) for line in lines}) == 1, "rows must be uniform width"

    def test_vram_graph_reflects_the_known_value(self, drm_root: Path, proc_root: Path) -> None:
        """4 GiB of 16 GiB is 25%, so the plot must not be flat at the bottom."""
        values, rendered = self.graph(drm_root, proc_root, "#vram-graph")
        assert values[-1] == pytest.approx(25.0, abs=0.5)
        assert any(char != "⠀" for char in rendered)

    def test_util_graph_reflects_the_known_value(self, drm_root: Path, proc_root: Path) -> None:
        values, _ = self.graph(drm_root, proc_root, "#util-graph")
        assert values[-1] == pytest.approx(37.0, abs=0.5)


class TestGraphHeight:
    """``ui.graph_height``: 0 fills the layout, a positive value pins the graph."""

    def _graph_height(
        self,
        drm_root: Path,
        proc_root: Path,
        size: tuple[int, int],
        **ui: object,
    ) -> int:
        app = make_app(drm_root, proc_root, **ui)

        async def body(_pilot: Pilot[Any]) -> int:
            return app.query_one("#util-graph", BrailleGraph).size.height

        return capture(app, size, body)

    def test_the_default_is_unchanged_by_the_key(self, drm_root: Path, proc_root: Path) -> None:
        """``0`` is the documented default and must render exactly as before it existed.

        The measured baseline: the graph is terminal-driven, not fixed, so pinning 10 rows
        by default would have made it taller than it has always been on a big terminal.
        """
        assert self._graph_height(drm_root, proc_root, (120, 40)) == 6
        assert self._graph_height(drm_root, proc_root, (120, 40), graph_height=0) == 6

    def test_a_positive_value_pins_the_graph(self, drm_root: Path, proc_root: Path) -> None:
        """The point of the key: a tall graph that does not shrink on every resize."""
        assert self._graph_height(drm_root, proc_root, (120, 60), graph_height=14) == 14

    def test_a_pinned_graph_is_clamped_rather_than_allowed_to_overflow(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A fixed height that overflows pushes the process table and footer off-screen.

        That is the one thing the responsive layout exists to prevent, so the request is
        reduced to what the terminal can spare rather than honoured literally.
        """
        height = self._graph_height(drm_root, proc_root, (120, 30), graph_height=40)

        assert height == 30 - MIN_HEIGHT_AROUND_UTIL_GRAPH

    @pytest.mark.parametrize("rows", [24, 20])
    def test_a_pinned_graph_never_pushes_the_footer_off_the_screen(
        self, drm_root: Path, proc_root: Path, rows: int
    ) -> None:
        """The failure mode, asserted rather than inferred from the height arithmetic.

        The graph is also asserted to have real height, so this cannot pass vacuously by
        a layout that simply ignored the key -- which would keep the footer on screen for
        the wrong reason.
        """
        app = make_app(drm_root, proc_root, graph_height=40)

        async def body(_pilot: Pilot[Any]) -> tuple[int, int, int]:
            footer = app.query_one(Footer).region
            return (
                footer.y + footer.height,
                app.screen.size.height,
                app.query_one("#util-graph", BrailleGraph).size.height,
            )

        bottom, screen, graph = capture(app, (100, rows), body)

        assert graph >= MIN_UTIL_GRAPH
        assert bottom <= screen, f"footer ends at row {bottom} of {screen}"

    def test_a_pinned_graph_never_goes_below_the_minimum(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A terminal with nothing to spare still gets a readable graph, not a sliver."""
        assert (
            self._graph_height(drm_root, proc_root, (80, 20), graph_height=40) >= MIN_UTIL_GRAPH
        )

    @pytest.mark.parametrize("pinned", [12, 18, 40])
    def test_a_pinned_graph_does_not_crop_the_readout(
        self, drm_root: Path, proc_root: Path, pinned: int
    ) -> None:
        """The panel has to grow to *hold* the graph, not merely contain it.

        The graph is not the whole panel: there is a title above it and the GPU and VRAM
        readout lines below.  Pinning the graph alone grew the plot and silently deleted
        the two numbers under it -- and every other measurement still looked correct,
        because a cropped child of a fixed-height box reports the size it was given.
        """
        app = make_app(drm_root, proc_root, graph_height=pinned)

        async def body(_pilot: Pilot[Any]) -> bool:
            readout = app.query_one("#util-readout")
            panel = app.query_one("#util-panel")
            return readout.size.height > 0 and readout.region.bottom <= panel.region.bottom

        assert capture(app, (120, 60), body), f"readout cropped at graph_height={pinned}"


class TestGraphHistoryCapacity:
    """How many samples a graph holds is one number, not two."""

    @pytest.mark.parametrize("points", [16, 64, 128])
    def test_the_graphs_hold_exactly_as_many_points_as_the_sampler_keeps(
        self, drm_root: Path, proc_root: Path, points: int
    ) -> None:
        """``history_points`` drives the graphs as well as the buffer.

        There used to be a second ``graph_history`` key under ``[ui]`` that nothing read.
        Wiring the capacity to the one live number is what makes lowering it do anything:
        the graph's own deque was left at a fixed 300, so a config asking for 64 samples
        kept a deque of 300 that could only ever be 64 full.
        """
        app = make_app(drm_root, proc_root)
        app._config = replace(
            app._config,
            general=replace(app._config.general, history_points=points),
            gpu=app._config.gpu,
        )
        app._sampler._options = replace(app._sampler.options, history_length=points)

        async def body(_pilot: Pilot[Any]) -> list[int]:
            return [
                app.query_one(widget_id, BrailleGraph)._capacity
                for widget_id in ("#util-graph", "#vram-graph", "#gtt-graph")
            ]

        assert set(capture(app, (120, 40), body)) == {points}


class TestThemes:
    """Runtime theme switching and the config surface."""

    def test_the_palette_reaches_css_without_destroying_textuals_own(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The palette is merged, never assigned over the framework's variable map.

        ``Stylesheet.set_variables`` *replaces* the map: calling it with three names left
        three entries where Textual had 168, so every rule referencing a design token --
        ``$background`` in Textual's own default ``App`` rule -- failed to resolve on the
        next re-parse, which is any modal.  The app's own comment next to the alert flash
        describes this exact hazard, and then the theme change did it anyway.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[int]:
            before = len(app.stylesheet._variables)
            await pilot.press("m")
            after = len(app.stylesheet._variables)
            await pilot.press("question_mark")
            return [before, after, len(app.stylesheet._variables)]

        before, after, after_modal = capture(app, (120, 40), body)
        assert before > 100, "Textual's design tokens should be present at start-up"
        assert after == before, "a theme change discarded the variable map"
        assert after_modal == before, "pushing a screen discarded the variable map"

    def test_the_theme_is_in_effect_before_any_key_is_pressed(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``$accent`` used to resolve to Textual's stock orange until the user pressed ``m``.

        The variables were only pushed from an action, so a fresh launch rendered in the
        framework's palette rather than the configured one.
        """
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> list[str]:
            return [app.get_css_variables()["accent"]]

        assert capture(app, (120, 40), body) == [app._theme.accent]

    def test_a_modal_pushes_without_an_unresolved_variable_error(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The re-parse that a screen push triggers is where a lost variable blows up."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[str]:
            await pilot.press("question_mark")
            await pilot.pause()
            return [app.screen.__class__.__name__]

        assert capture(app, (120, 40), body)  # would raise on an unresolved variable

    def test_cycle_visits_every_theme(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> list[str]:
            seen = [app._theme.name]
            for _ in range(len(theme_names())):
                await pilot.press("m")
                seen.append(app._theme.name)
            return seen

        seen = capture(app, (120, 40), body)
        assert seen[0] == "default"
        assert set(seen) == set(theme_names())

    def test_set_theme_reaches_the_graphs(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> list[str]:
            app.set_theme("dracula")
            return [graph._gradient.at(100.0) for graph in app.query(BrailleGraph)]

        colours = capture(app, (120, 40), body)
        assert colours
        assert all(colour == THEMES["dracula"].high for colour in colours)

    def test_unknown_theme_name_falls_back(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, theme="nonsense")
        assert app._theme.name == "default"

    def test_theme_from_config_is_applied(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root, theme="gruvbox")
        assert app._theme.name == "gruvbox"

    def test_no_color_keeps_the_interface_legible(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """--no-color must produce the same content in terminal ANSI colours."""
        app = make_app(drm_root, proc_root, no_color=True)

        async def body(_pilot: Pilot[Any]) -> tuple[bool, str]:
            return bool(app.ansi_color), "\n".join(screen_lines(app, 120))

        ansi, text = capture(app, (120, 40), body)
        assert ansi is True
        assert "AMD Radeon RX 6800" in text
        assert "2,200 MHz" in text


class TestResponsiveness:
    """The layout adapts without ever dropping the graphs."""

    def screen(self, drm_root: Path, proc_root: Path, size: tuple[int, int]) -> str:
        """Render the app at a given terminal size."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return "\n".join(screen_lines(app, size[0]))

        return capture(app, size, body)

    def test_wide_layout_shows_everything(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root, (140, 44))
        assert "Sensors" in text
        assert "PID" in text

    def test_the_layout_is_not_recomputed_while_nothing_changes(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Layout is a pure function of size and toggles, so a steady state writes nothing.

        It used to re-assert seven inline styles on every sample.  Each one marks a
        widget dirty in Textual, which costs a stylesheet re-resolve and a repaint of the
        two largest panels per tick -- for a layout that had not changed.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[int, int]:
            await wait_until(lambda: app._layout_key is not None, what="first layout")
            first = app._layout_key
            # Two more samples; the key must not move and the styles must not be rewritten.
            await wait_until(lambda: app._sampler.stats.ticks >= 3, what="three samples")
            assert app._layout_key == first
            return app._sampler.stats.ticks, 0 if first is None else 1

        ticks, laid_out = capture(app, (120, 40), body)
        assert ticks >= 3
        assert laid_out == 1

    def test_narrow_layout_keeps_every_panel(self, drm_root: Path, proc_root: Path) -> None:
        """A narrow terminal stacks the panels rather than losing them."""
        text = self.screen(drm_root, proc_root, (64, 44))
        assert "GPU utilisation" in text
        assert "Sensors" in text

    def test_narrow_layout_fits_the_width(self, drm_root: Path, proc_root: Path) -> None:
        width = 64
        for line in self.screen(drm_root, proc_root, (width, 44)).splitlines():
            assert len(line) <= width, repr(line)

    def test_short_layout_drops_the_process_table(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """On a short terminal the table yields so the graphs stay legible."""
        text = self.screen(drm_root, proc_root, (120, 16))
        assert "GPU utilisation" in text
        assert "PID" not in text

    def test_very_small_terminal_still_renders(self, drm_root: Path, proc_root: Path) -> None:
        text = self.screen(drm_root, proc_root, (40, 12))
        assert "AMD Radeon RX 6800" in text


class TestActions:
    """Key bindings."""

    def test_toggle_processes(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[bool, bool]:
            before = app._show_processes
            await pilot.press("p")
            return before, app._show_processes

        before, after = capture(app, (120, 40), body)
        assert before is True
        assert after is False

    def test_refresh_now_produces_a_sample(self, drm_root: Path, proc_root: Path) -> None:
        """``R`` forces an immediate sample, without waiting for the interval."""
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> int:
            before = app._sampler.stats.ticks
            await pilot.press("R")
            return app._sampler.stats.ticks - before

        assert capture(app, (120, 40), body) >= 1

    def test_focus_device_is_a_no_op_when_absent(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> int:
            await pilot.press("3")
            return app._index

        assert capture(app, (120, 40), body) == 0


class TestShortBdf:
    """The GPU column has to distinguish two identical cards."""

    def test_drops_the_domain(self) -> None:
        assert short_bdf("0000:0c:00.0") == "0c:00.0"

    def test_distinguishes_two_gpus(self) -> None:
        assert short_bdf("0000:0c:00.0") != short_bdf("0000:03:00.0")

    def test_empty_is_na(self) -> None:
        assert short_bdf("") == "N/A"


class TestUiThreadDoesNoIO:
    """The UI thread must not do the data layer's I/O.

    The whole design rests on the sampler owning every read.  A stray sysfs or procfs read
    from a render path would put a disk wait on the frame budget and make the app stutter
    exactly when the machine is busy.
    """

    def test_no_reader_is_called_while_rendering(
        self, drm_root: Path, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build_hardware(drm_root, proc_root)
        app = make_app(drm_root, proc_root)
        # Mirror run_app: discover before the event loop, so the measurement window
        # contains rendering only and not the one-time startup fallback in on_mount.
        app.discover()

        calls: list[str] = []

        async def body(_pilot: Pilot[Any]) -> None:
            # Install the guard only once the app is settled, then measure one frame.
            for module, names in (
                (
                    fsutil,
                    ("read_text", "read_int", "read_bytes", "read_hex_int", "read_first_int"),
                ),
                (procfs, ("parse_fdinfo",)),
                (discovery, ("discover_devices",)),
            ):
                for name in names:
                    monkeypatch.setattr(
                        module,
                        name,
                        _record(calls, f"{module.__name__}.{name}", getattr(module, name)),
                    )
            # A render is queue draining, widget updates and history projection.
            app._refresh()
            if app._last_snapshot is not None:
                app._feed_graphs(app._last_snapshot)

        asyncio.run(run_app(app, size=(120, 40), body=body, settle=3))
        assert calls == [], f"the UI thread performed I/O: {sorted(set(calls))}"

    def test_discovery_runs_before_the_event_loop(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``run_app`` discovers up front, so the loop itself never reads sysfs."""
        build_hardware(drm_root, proc_root)
        app = make_app(drm_root, proc_root)
        assert app._sampler.devices == ()
        assert len(app.discover()) == 1
