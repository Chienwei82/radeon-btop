"""Tests for the panel text builders.

These are pure functions, so every assertion is an exact snapshot.  The policy they
enforce -- an unavailable reading is never rendered as a value -- is the one thing a
monitor must never get wrong.
"""

import pytest
from rich.text import Text

from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import (
    AmdgpuMetrics,
    Clock,
    Fan,
    MemoryPool,
    PcieLink,
    Power,
    Temperature,
    ThrottleInfo,
)
from gputop.ui.devices import TAB_GAP, TAB_SEPARATOR, device_at, gpu_tabs, tab_spans
from gputop.ui.format import NA, fmt_duration
from gputop.ui.panels import (
    LABEL_WIDTH,
    fit_bar_width,
    footer_panel,
    header_panel,
    pool_row,
    sensors_panel,
    stat_row,
    utilisation_rows,
)
from gputop.ui.theme import DEFAULT_THEME, DRACULA_THEME, Gradient

MiB = 1024**2
GiB = 1024**3


def make_device(**overrides: object) -> AmdgpuDevice:
    """Build a device record."""
    defaults: dict[str, object] = {
        "index": 0,
        "card": "card0",
        "render": "renderD128",
        "bdf": "0000:0c:00.0",
        "vendor_id": 0x1002,
        "device_id": 0x73BF,
        "device_dir": "/sys/devices/fake",
        "hwmon_dir": None,
        "kind": "dgpu",
        "name": "AMD Radeon Test",
        "vram_vendor": "samsung",
        "boot_vga": False,
    }
    defaults.update(overrides)
    return AmdgpuDevice(**defaults)  # type: ignore[arg-type]


def make_metrics(**overrides: object) -> AmdgpuMetrics:
    """Build a metrics record."""
    defaults: dict[str, object] = {
        "device": make_device(),
        "sequence": 1,
        "timestamp_ns": 0,
        "interval_s": 1.0,
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def plain(text: Text) -> str:
    """Strip styling from a rendered row."""
    return text.plain


class TestFitBarWidth:
    """Bar sizing, which is what keeps a row inside its panel."""

    def test_uses_the_space_left_after_the_value(self) -> None:
        assert fit_bar_width(50, 10) == 50 - 14 - 10 - 1

    def test_falls_back_to_a_minimum(self) -> None:
        """A narrow panel degrades to a short bar rather than pushing the value away."""
        assert fit_bar_width(20, 15) == 6

    def test_never_negative(self) -> None:
        assert fit_bar_width(5, 40) >= 6


class TestStatRow:
    """A single labelled bar."""

    def test_renders_label_bar_and_value(self) -> None:
        row = plain(stat_row("GPU", "42%", 42.0, DEFAULT_THEME, available=40))
        assert row.startswith("GPU")
        assert "42%" in row
        assert len(row) <= 40

    def test_missing_value_shows_na_not_zero(self) -> None:
        row = plain(stat_row("GPU", NA, None, DEFAULT_THEME, available=40))
        assert NA in row
        assert "0%" not in row

    def test_missing_value_draws_an_empty_bar(self) -> None:
        row = plain(stat_row("GPU", NA, None, DEFAULT_THEME, available=40))
        assert "█" not in row

    def test_zero_is_a_real_reading(self) -> None:
        """0% must render as 0%, distinct from N/A."""
        row = plain(stat_row("GPU", "0%", 0.0, DEFAULT_THEME, available=40))
        assert "0%" in row
        assert NA not in row

    def test_full_bar(self) -> None:
        # 40 available - 14 label - 5 value - 1 space = a 20 cell bar.
        row = plain(stat_row("GPU", "100%", 100.0, DEFAULT_THEME, available=40))
        assert "█" * 20 in row
        assert "░" not in row

    def test_never_exceeds_the_available_width(self) -> None:
        """A row is exactly `available` wide unless the minimum bar already overflows.

        Below the floor the row is deliberately allowed to exceed the panel rather than
        drop the value, because a clipped number is worse than a slightly long one.
        """
        value = "1,234 MHz"
        floor = LABEL_WIDTH + 6 + 1 + len(value)
        for available in (30, 45, 60, 80):
            row = plain(stat_row("Core", value, 55.0, DEFAULT_THEME, available=available))
            assert len(row) <= available
        cramped = plain(stat_row("Core", value, 55.0, DEFAULT_THEME, available=20))
        assert len(cramped) <= floor

    def test_colour_follows_the_value(self) -> None:
        """The gradient has to reach the bar, not just the empty track."""
        low = stat_row("GPU", "5%", 5.0, DEFAULT_THEME, available=40)
        high = stat_row("GPU", "95%", 95.0, DEFAULT_THEME, available=40)
        assert bar_colour(low) != bar_colour(high)

    def test_theme_changes_the_colour(self) -> None:
        default = stat_row("GPU", "50%", 50.0, DEFAULT_THEME, available=40)
        dracula = stat_row("GPU", "50%", 50.0, DRACULA_THEME, available=40)
        assert bar_colour(default) != bar_colour(dracula)

    def test_empty_track_uses_the_theme_track_colour(self) -> None:
        row = stat_row("GPU", "5%", 5.0, DEFAULT_THEME, available=40)
        assert colour_at(row, row.plain.index("\u2591")) == DEFAULT_THEME.track.lower()


def colour_at(text: Text, index: int) -> str:
    """The colour applied to a given character."""
    for start, end, style in text._spans:
        if start <= index < end and style.color is not None:
            return style.color.triplet.hex
    return ""


def bar_colour(text: Text) -> str:
    """The colour of the first filled bar cell."""
    return colour_at(text, text.plain.index("\u2588"))


class TestPoolRow:
    """Memory rows."""

    def test_shows_used_total_and_percentage(self) -> None:
        row = plain(
            pool_row("VRAM", MemoryPool(used=4 * GiB, total=16 * GiB), DEFAULT_THEME, 60)
        )
        assert "4.0 GiB" in row and "16.0 GiB" in row
        assert "25%" in row

    def test_missing_pool_shows_na(self) -> None:
        row = plain(pool_row("VRAM", MemoryPool(), DEFAULT_THEME, 60))
        assert NA in row
        assert "0%" not in row

    def test_zero_total_is_not_zero_percent(self) -> None:
        row = plain(pool_row("VRAM", MemoryPool(used=0, total=0), DEFAULT_THEME, 60))
        assert NA in row

    def test_fits_the_panel(self) -> None:
        row = plain(
            pool_row("VRAM", MemoryPool(used=4 * GiB, total=16 * GiB), DEFAULT_THEME, 44)
        )
        assert len(row) <= 44


class TestHeaderPanel:
    """The device header."""

    def test_shows_identity_and_timing(self) -> None:
        text = plain(header_panel(make_metrics(), DEFAULT_THEME, 3725.0, 1.0))
        assert "AMD Radeon Test" in text
        assert "dGPU" in text
        assert "amdgpu" in text
        assert "0000:0c:00.0" in text
        assert "0x73bf" in text
        assert "renderD128" in text
        assert "1h 02m" in text
        assert "every 1s" in text

    def test_marks_an_integrated_gpu(self) -> None:
        text = plain(
            header_panel(make_metrics(device=make_device(kind="igpu")), DEFAULT_THEME, 5.0, 1.0)
        )
        assert "iGPU" in text

    def test_marks_the_boot_device(self) -> None:
        text = plain(
            header_panel(
                make_metrics(device=make_device(boot_vga=True)), DEFAULT_THEME, 1.0, 1.0
            )
        )
        assert "boot" in text

    def test_shows_the_metrics_source(self) -> None:
        text = plain(header_panel(make_metrics(metrics_abi="v1.3"), DEFAULT_THEME, 1.0, 1.0))
        assert "v1.3" in text

    def test_falls_back_to_sysfs_without_the_binary_table(self) -> None:
        text = plain(header_panel(make_metrics(), DEFAULT_THEME, 1.0, 1.0))
        assert "sysfs" in text

    def test_three_lines(self) -> None:
        assert plain(header_panel(make_metrics(), DEFAULT_THEME, 1.0, 1.0)).count("\n") == 2


class TestUtilisationRows:
    """Which memory pool gets the bar."""

    def test_discrete_shows_vram(self) -> None:
        rows = [plain(r) for r in utilisation_rows(make_metrics(), DEFAULT_THEME, 50)]
        assert any("VRAM" in r for r in rows)
        assert not any(r.startswith("GTT") for r in rows)

    def test_integrated_shows_gtt(self) -> None:
        """On an APU, GTT is the working set; VRAM is a small carve-out."""
        metrics = make_metrics(
            device=make_device(kind="igpu"),
            gtt=MemoryPool(used=4 * GiB, total=16 * GiB),
            vram=MemoryPool(used=64 * MiB, total=512 * MiB),
        )
        rows = utilisation_rows(metrics, DEFAULT_THEME, 50)
        first = rows[1].plain
        assert first.startswith("GTT")
        # VRAM is still listed, but demoted below the primary pool.
        assert any(r.plain.startswith("VRAM") for r in rows[2:])

    def test_includes_the_gpu_row(self) -> None:
        rows = utilisation_rows(make_metrics(gpu_busy_percent=10.0), DEFAULT_THEME, 50)
        assert rows[0].plain.startswith("GPU")
        assert "10%" in rows[0].plain


class TestSensorsPanel:
    """Every sensor the specification calls for."""

    def full(self) -> AmdgpuMetrics:
        """A device with every reading present."""
        return make_metrics(
            gpu_busy_percent=50.0,
            sclk=Clock(current=2200, maximum=2475, source="gpu_metrics"),
            mclk=Clock(current=1000, maximum=1000, source="gpu_metrics"),
            socclk=Clock(current=800, maximum=1200, source="gpu_metrics"),
            temperatures=(
                Temperature("edge", 49, "gpu_metrics"),
                Temperature("junction", 56, "gpu_metrics"),
            ),
            power=Power(draw_w=45.0, cap_w=211.0, source="gpu_metrics"),
            fan=Fan(rpm=2400, max_rpm=3000, pwm=80.0),
            voltages_mv=(("soc", 918), ("gfx", 768)),
            pcie=PcieLink(width=16, speed=80),
            power_profile="BOOTUP_DEFAULT",
            throttle=ThrottleInfo(raw=0, active=frozenset()),
        )

    def test_every_required_sensor_is_present(self) -> None:
        text = "\n".join(plain(r) for r in sensors_panel(self.full(), DEFAULT_THEME, 50))
        for label in ("Core", "Memory", "SoC", "temp edge", "temp junction", "Power", "Fan"):
            assert label in text, f"missing {label}"

    def test_power_is_shown_against_the_cap(self) -> None:
        text = "\n".join(plain(r) for r in sensors_panel(self.full(), DEFAULT_THEME, 50))
        assert "45.0 W" in text
        assert "cap 211 W" in text

    def test_pcie_and_profile(self) -> None:
        text = "\n".join(plain(r) for r in sensors_panel(self.full(), DEFAULT_THEME, 50))
        assert "Gen3 x16" in text
        assert "BOOTUP_DEFAULT" in text

    def test_quiet_throttle(self) -> None:
        text = "\n".join(plain(r) for r in sensors_panel(self.full(), DEFAULT_THEME, 50))
        assert "throttle none" in text

    def test_active_throttle_names_the_reasons(self) -> None:
        metrics = make_metrics(throttle=ThrottleInfo(raw=1 << 32, active={"TEMP_GPU"}))
        text = "\n".join(plain(r) for r in sensors_panel(metrics, DEFAULT_THEME, 50))
        assert "TEMP_GPU" in text

    def test_fan_is_hidden_on_an_igpu(self) -> None:
        """An APU has no fan, so the row would be permanent noise."""
        metrics = self.full()
        igpu = make_metrics(device=make_device(kind="igpu"))
        text = "\n".join(plain(r) for r in sensors_panel(igpu, DEFAULT_THEME, 50))
        assert "Fan" not in text
        assert "Fan" in "\n".join(plain(r) for r in sensors_panel(metrics, DEFAULT_THEME, 50))

    def test_memory_clock_is_hidden_on_an_igpu(self) -> None:
        text = "\n".join(
            plain(r)
            for r in sensors_panel(
                make_metrics(device=make_device(kind="igpu")), DEFAULT_THEME, 50
            )
        )
        assert "\nMemory" not in f"\n{text}"

    def test_bare_device_never_claims_zero(self) -> None:
        """The critical property: absence is ``N/A``, not ``0``."""
        text = "\n".join(plain(r) for r in sensors_panel(make_metrics(), DEFAULT_THEME, 50))
        assert "0 MHz" not in text
        assert "0.0 W" not in text
        assert text.count(NA) >= 4

    def test_rows_stay_within_the_panel(self) -> None:
        for available in (36, 44, 60, 90):
            for row in sensors_panel(self.full(), DEFAULT_THEME, available):
                assert len(plain(row)) <= available

    def test_temperature_uses_a_fixed_scale(self) -> None:
        """49 C is 39% of a 125 C span, not 49% of an arbitrary range."""
        rows = sensors_panel(self.full(), DEFAULT_THEME, 50)
        edge = next(r for r in rows if r.plain.startswith("temp edge"))
        assert "49°C" in edge.plain
        filled = edge.plain.count("█")
        total = filled + edge.plain.count("░")
        assert filled / total == pytest.approx(49 / 125, abs=0.02)


class TestFooterPanel:
    """The status line."""

    def test_reports_the_apps_own_cpu(self) -> None:
        text = plain(
            footer_panel(
                DEFAULT_THEME,
                sequence=7,
                metrics_source="v1.3",
                tick_ms=12.0,
                cpu="3.2%",
            )
        )
        assert "cpu 3.2%" in text

    def test_flags_a_partial_process_view(self) -> None:
        text = plain(
            footer_panel(
                DEFAULT_THEME,
                sequence=1,
                metrics_source=None,
                tick_ms=1.0,
                cpu="1%",
                visible=15,
                total=580,
            )
        )
        assert "partial 15/580 procs" in text

    def test_no_partial_note_when_visibility_is_complete(self) -> None:
        text = plain(
            footer_panel(
                DEFAULT_THEME,
                sequence=1,
                metrics_source=None,
                tick_ms=1.0,
                cpu="1%",
                visible=580,
                total=580,
            )
        )
        assert "partial" not in text

    def test_surfaces_the_first_warning(self) -> None:
        text = plain(
            footer_panel(
                DEFAULT_THEME,
                sequence=1,
                metrics_source=None,
                tick_ms=1.0,
                cpu="1%",
                warnings=("card0: PermissionError", "other"),
            )
        )
        assert "PermissionError" in text
        assert "other" not in text


class TestFmtDuration:
    """Uptime formatting."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (9, "9s"),
            (65, "1m 05s"),
            (3725, "1h 02m"),
            (90_061, "1d 01h"),
        ],
    )
    def test_formats(self, seconds: int, expected: str) -> None:
        assert fmt_duration(seconds) == expected

    def test_negative_clamps_to_zero(self) -> None:
        assert fmt_duration(-5.0) == "0s"

    def test_none_is_na(self) -> None:
        assert fmt_duration(None) == NA


class TestGradientPlumbing:
    """A shared ramp must not change the text, only the colour."""

    def test_shared_gradient_matches_a_built_one(self) -> None:
        built = plain(stat_row("GPU", "50%", 50.0, DEFAULT_THEME, 40))
        shared = plain(
            stat_row("GPU", "50%", 50.0, DEFAULT_THEME, 40, gradient=Gradient(DEFAULT_THEME))
        )
        assert built == shared


class TestTabGeometry:
    """The tab bar's hit testing is arithmetic over the line it just drew.

    A span that disagrees with the drawing by one cell per tab is invisible in a
    screenshot and puts a click on the wrong GPU -- and the mouse tests that use
    ``tab_spans`` to pick a cell cannot catch it, because they ask the same function
    where to click.  These assert the two against each other instead.
    """

    def _devices(self, count: int) -> list[AmdgpuMetrics]:
        return [
            make_metrics(
                device=make_device(index=index, card=f"card{index}", name=f"GPU{index}"),
                gpu_busy_percent=50.0,
            )
            for index in range(count)
        ]

    @pytest.mark.parametrize("count", [1, 2, 3, 9, 10, 12])
    def test_the_spans_add_up_to_the_drawn_line(self, count: int) -> None:
        devices = self._devices(count)
        # Wide enough not to truncate: the bar crops itself at the terminal width, and
        # this is about the arithmetic, not about cropping.
        line = gpu_tabs(devices, DEFAULT_THEME, index=0, available=1000)
        spans = tab_spans(devices)
        assert sum(end - start for start, end, _ in spans) == len(line.plain)

    @pytest.mark.parametrize("count", [1, 2, 3, 9, 10, 12])
    def test_each_span_starts_on_its_own_number_badge(self, count: int) -> None:
        """The badge is the first thing drawn, so the span has to reach it.

        A span carries the gutter that precedes its tab, so the badge sits one cell past
        the gap -- and the number widens past one digit at the tenth GPU, which is exactly
        where an assumption of a fixed badge width stops holding.
        """
        devices = self._devices(count)
        line = gpu_tabs(devices, DEFAULT_THEME, index=0, available=1000).plain
        for start, _end, position in tab_spans(devices):
            badge = str(position + 1)
            at = start + (TAB_GAP if position else 0) + 1
            assert line[at : at + len(badge)] == badge

    def test_the_gutter_belongs_to_the_tab_that_follows_it(self) -> None:
        devices = self._devices(3)
        line = gpu_tabs(devices, DEFAULT_THEME, index=0, available=1000).plain
        first_end = tab_spans(devices)[0][1]
        assert set(line[first_end : first_end + TAB_GAP]) <= {" ", TAB_SEPARATOR}
        assert device_at(devices, first_end) == 1
        assert device_at(devices, first_end - 1) == 0

    @pytest.mark.parametrize("name", ["显卡", "Ünïcödé GPU", "🎮"])
    def test_a_wide_device_name_is_measured_in_cells_not_code_points(self, name: str) -> None:
        """``[gpu.names]`` is user-supplied, and a CJK or emoji name is a normal thing to put in it.

        ``tab_width`` used ``len()``, which is code points: ``len("显卡")`` is 2 where the
        name occupies 4 cells.  Every boundary after it was short by the difference, so
        the last GPU's tab started early and the right-hand end of the bar belonged to no
        tab at all -- a click there hit nothing, and a click just before it hit the wrong
        GPU.  The whole test suite was blind to it because every fixture name is ASCII.
        """
        from rich.cells import cell_len

        devices = [
            make_metrics(
                device=make_device(index=0, card="card0", name=name),
                gpu_busy_percent=50.0,
            ),
            make_metrics(
                device=make_device(index=1, card="card1", name="Second"),
                gpu_busy_percent=10.0,
            ),
        ]
        line = gpu_tabs(devices, DEFAULT_THEME, index=0, available=1000)
        spans = tab_spans(devices)
        # Spans are in cells, so they have to sum to the drawn cell width, not its
        # code-point length.
        assert sum(end - start for start, end, _ in spans) == cell_len(line.plain)
        # And the far end of the bar has to belong to the last tab rather than to nothing.
        assert device_at(devices, cell_len(line.plain) - 1) == 1
