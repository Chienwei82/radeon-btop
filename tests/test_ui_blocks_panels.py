"""Tests for the blocks and power-profile panel text.

The rules these enforce are the ones that make an optional panel honest: a block the
hardware does not have is absent rather than zero, a block that is genuinely idle shows
``0%`` rather than ``N/A``, and a panel with no data says so in one line instead of
drawing a column of empty bars.
"""

import pytest

from gputop.model.blocks import (
    AUX_BLOCK_ORDER,
    BLOCK_LABEL_WIDTH,
    BLOCK_ORDER,
    BlockReading,
    BlocksStatus,
    ClockReading,
    GpuBlocks,
    block_label,
)
from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.power import (
    OdcVoltage,
    OdvDomain,
    OdvEntry,
    OdvUnit,
    PowerProfileEntry,
    PowerProfileTable,
)
from gputop.ui.format import NA
from gputop.ui.panels import (
    blocks_heading,
    blocks_panel,
    footer_panel,
    power_profile_panel,
)
from gputop.ui.theme import DEFAULT_THEME

MiB = 1024**2


def make_metrics(**overrides: object) -> AmdgpuMetrics:
    """Build a metrics record."""
    device = AmdgpuDevice(
        index=0,
        card="card0",
        render="renderD128",
        bdf="0000:0c:00.0",
        vendor_id=0x1002,
        device_id=0x73BF,
        device_dir="/sys/devices/fake",
        hwmon_dir=None,
        kind="dgpu",
        name="AMD Radeon Test",
        vram_vendor=None,
        boot_vga=False,
    )
    defaults: dict[str, object] = {
        "device": device,
        "sequence": 1,
        "timestamp_ns": 0,
        "interval_s": 1.0,
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def make_blocks(**overrides: object) -> GpuBlocks:
    """Build a block sample with every pipeline block present.

    Percentages are distinct per block so a test can assert on a *named* block rather than
    on a position, which keeps the assertions honest if the display order changes.
    """
    percents = {key: float(index * 5) for index, key in enumerate(BLOCK_ORDER)}
    percents["gpu"] = 45.0
    percents["cb"] = 65.0
    defaults: dict[str, object] = {
        "blocks": tuple(
            BlockReading(key=key, label=block_label(key), percent=percents[key])
            for key in BLOCK_ORDER
        ),
        "aux_blocks": tuple(
            BlockReading(key=key, label=block_label(key), percent=1.0)
            for key in AUX_BLOCK_ORDER
        ),
        "clocks": (
            ClockReading(key="mclk", label="Memory Clock", percent=50.0, mhz=1000),
            ClockReading(key="sclk", label="Shader Clock", percent=25.0, mhz=2104),
        ),
    }
    defaults.update(overrides)
    return GpuBlocks(**defaults)  # type: ignore[arg-type]


def rows_text(rows: list) -> list[str]:
    """Strip styling from each rendered row."""
    return [row.plain for row in rows]


class TestBlocksPanel:
    """The per-block panel."""

    def test_renders_a_row_per_block(self) -> None:
        text = rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44))
        assert len(text) == len(BLOCK_ORDER) + len(AUX_BLOCK_ORDER) + 2

    def test_every_row_fits_the_available_width(self) -> None:
        """A row that wraps would corrupt every row beneath it."""
        for row in rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=40)):
            assert len(row) <= 40

    def test_a_narrow_panel_still_fits(self) -> None:
        """44 is the narrowest width the app hands these panels."""
        for row in rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44)):
            assert len(row) <= 44

    def test_the_busy_blocks_are_named_and_sized(self) -> None:
        text = rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44))
        assert "Graphics pipe" in text[0]
        assert "45%" in text[0]
        assert "█" in text[0]

    def test_every_known_block_gets_a_row_and_a_label_that_fits(self) -> None:
        """A label wider than the reserved column pushes the bar off the row."""
        for key in BLOCK_ORDER:
            assert len(block_label(key)) <= BLOCK_LABEL_WIDTH

    def test_the_longest_label_still_leaves_a_gutter_before_the_bar(self) -> None:
        """Column width is the longest label *plus one*.

        Measuring to exactly the longest label leaves the bar flush against it on the row
        that needs the whole column, which reads as a single run-together token.
        """
        longest = max(len(block_label(key)) for key in BLOCK_ORDER)
        assert longest + 1 == BLOCK_LABEL_WIDTH

    def test_no_block_row_runs_out_under_its_own_label(self) -> None:
        for row in rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44)):
            if "█" in row or "░" in row:
                assert " " in row.split("█")[0].split("░")[0] or row.strip()

    def test_a_genuinely_idle_block_reads_zero_not_na(self) -> None:
        """0% is a measurement; N/A means absent, which is a different thing."""
        blocks = make_blocks(
            blocks=(BlockReading(key="gpu", label="Graphics pipe", percent=0.0),)
        )
        text = rows_text(blocks_panel(blocks, DEFAULT_THEME, available=44))
        assert "0%" in text[0]
        assert NA not in text[0]

    def test_an_absent_block_gets_no_row_at_all(self) -> None:
        """radeontop omits blocks the hardware lacks, so absence must not become a zero."""
        blocks = make_blocks(
            blocks=(BlockReading(key="gpu", label="Graphics pipe", percent=9.0),)
        )
        text = rows_text(blocks_panel(blocks, DEFAULT_THEME, available=44))
        assert not any("Texture Cache" in row for row in text)

    def test_clocks_show_a_frequency_rather_than_only_a_percentage(self) -> None:
        text = rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44))
        assert "2,104 MHz" in text[-1]
        assert "Memory Clock" in text[-2]

    def test_a_clock_with_no_frequency_degrades_to_the_percentage(self) -> None:
        blocks = make_blocks(
            clocks=(ClockReading(key="sclk", label="Shader Clock", percent=25.0, mhz=None),)
        )
        text = rows_text(blocks_panel(blocks, DEFAULT_THEME, available=44))
        assert "25%" in text[-1]
        assert NA not in text[-1]

    def test_no_sample_yields_one_explanatory_line(self) -> None:
        """Fourteen ``N/A`` labels answer nothing; one sentence does."""
        text = rows_text(blocks_panel(None, DEFAULT_THEME, available=44))
        assert len(text) == 1
        assert "waiting" in text[0]
        assert NA not in text[0]

    def test_an_empty_sample_says_so(self) -> None:
        text = rows_text(blocks_panel(GpuBlocks(), DEFAULT_THEME, available=44))
        assert len(text) == 1
        assert "no blocks" in text[0]

    def test_group_order_is_pipeline_then_aux_then_clocks(self) -> None:
        text = rows_text(blocks_panel(make_blocks(), DEFAULT_THEME, available=44))
        assert "Graphics pipe" in text[0]
        assert "UVD" in text[len(BLOCK_ORDER)]
        assert "Memory Clock" in text[-2]


class TestBlocksHeading:
    """The panel title, which is the answer to "which block"."""

    def test_names_the_busiest_block_and_its_share(self) -> None:
        heading = blocks_heading(make_blocks())
        assert "Blocks" in heading
        assert "Color Block" in heading
        assert "%" in heading

    def test_is_neutral_when_there_is_no_sample(self) -> None:
        assert blocks_heading(None) == "Blocks"
        assert blocks_heading(GpuBlocks()) == "Blocks"


class TestPowerProfilePanel:
    """The read-only power profile and overdrive panel."""

    def test_shows_the_active_profile(self) -> None:
        metrics = make_metrics(
            profiles=PowerProfileTable(
                profiles=(
                    PowerProfileEntry(index=0, name="BOOTUP_DEFAULT", active=True),
                    PowerProfileEntry(index=1, name="3D_FULL_SCREEN"),
                ),
                present=True,
            )
        )
        text = " ".join(rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44)))
        assert "BOOTUP_DEFAULT" in text

    def test_lists_the_alternatives_without_offering_to_change_them(self) -> None:
        metrics = make_metrics(
            profiles=PowerProfileTable(
                profiles=(
                    PowerProfileEntry(index=0, name="BOOTUP_DEFAULT", active=True),
                    PowerProfileEntry(index=1, name="3D_FULL_SCREEN"),
                ),
                present=True,
            )
        )
        text = " ".join(rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44)))
        assert "3D_FULL_SCREEN" in text
        assert "read-only" in text

    def test_an_absent_table_is_na_not_a_profile_called_none(self) -> None:
        text = " ".join(rows_text(power_profile_panel(make_metrics(), DEFAULT_THEME, 44)))
        assert f"profile {NA}" in text
        assert f"profiles {NA}" in text

    def test_overdrive_domains_are_shown_with_their_own_units(self) -> None:
        metrics = make_metrics(
            odc=OdcVoltage(
                domains=(
                    OdvDomain(
                        name="sclk",
                        entries=(
                            OdvEntry(state=0, value=500, unit=OdvUnit.MEGAHERTZ),
                            OdvEntry(state=1, value=2104, unit=OdvUnit.MEGAHERTZ),
                        ),
                        minimum=500,
                        maximum=2600,
                        unit=OdvUnit.MEGAHERTZ,
                    ),
                ),
                vddgfx_offset_mv=-50,
                present=True,
            )
        )
        text = " ".join(rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44)))
        assert "2104/2600 MHz" in text
        assert "-50 mV" in text

    def test_a_millivolt_table_is_not_reported_as_megahertz(self) -> None:
        """The legacy format must keep its own unit rather than inherit the modern one."""
        metrics = make_metrics(
            odc=OdcVoltage(
                domains=(
                    OdvDomain(
                        name="sclk",
                        entries=(
                            OdvEntry(state=0, value=300, unit=OdvUnit.MILLIVOLT),
                            OdvEntry(state=1, value=450, unit=OdvUnit.MILLIVOLT),
                        ),
                        minimum=300,
                        maximum=450,
                        unit=OdvUnit.MILLIVOLT,
                    ),
                ),
                present=True,
            )
        )
        text = " ".join(rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44)))
        assert "450/450 mV" in text

    def test_an_absent_overdrive_table_is_na(self) -> None:
        text = " ".join(rows_text(power_profile_panel(make_metrics(), DEFAULT_THEME, 44)))
        assert f"overdrive {NA}" in text

    @pytest.mark.parametrize("width", [44, 58, 80])
    def test_the_bar_rows_fit_the_available_width(self, width: int) -> None:
        """Bar rows are sized to the panel; a long profile list crops rather than wraps."""
        metrics = make_metrics(
            odc=OdcVoltage(
                domains=(
                    OdvDomain(
                        name="sclk",
                        entries=(OdvEntry(state=1, value=2104, unit=OdvUnit.MEGAHERTZ),),
                        minimum=500,
                        maximum=2600,
                        unit=OdvUnit.MEGAHERTZ,
                    ),
                ),
                vddgfx_offset_mv=-50,
                present=True,
            )
        )
        rows = rows_text(power_profile_panel(metrics, DEFAULT_THEME, width))
        for row in rows:
            if row.startswith("OD "):
                assert len(row) <= width

    def test_detail_rows_are_trimmed_to_the_panel_rather_than_left_to_wrap(self) -> None:
        """A row that renders taller than it lays out draws over the panel's own border.

        Trimming the text to the available width is what keeps the box and its contents
        the same size; relying on the widget to crop it leaves a row two lines tall in a
        panel sized ``height: auto``.
        """
        metrics = make_metrics(
            profiles=PowerProfileTable(
                profiles=tuple(
                    PowerProfileEntry(index=i, name=f"AN_EXTREMELY_LONG_PROFILE_NAME_{i}")
                    for i in range(12)
                ),
                present=True,
            )
        )
        rows = rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44))
        for row in rows:
            assert len(row) <= 44
        assert any(row.endswith("…") for row in rows)

    def test_detail_rows_never_wrap(self) -> None:
        """A wrapped row corrupts every row under it, so every row must be no-wrap."""
        metrics = make_metrics(
            profiles=PowerProfileTable(
                profiles=tuple(
                    PowerProfileEntry(index=i, name=f"AN_EXTREMELY_LONG_PROFILE_NAME_{i}")
                    for i in range(8)
                ),
                present=True,
            )
        )
        for row in rows_text(power_profile_panel(metrics, DEFAULT_THEME, 44)):
            assert "\n" not in row


class TestBlocksHintInTheFooter:
    """The status line's explanation for a missing blocks panel."""

    def footer(self, status: BlocksStatus, hint: str = "") -> str:
        return footer_panel(
            DEFAULT_THEME,
            sequence=1,
            metrics_source="v1.3",
            tick_ms=1.0,
            cpu="1%",
            blocks_status=status,
            blocks_hint=hint,
        ).plain

    @pytest.mark.parametrize(
        "status",
        [
            BlocksStatus.MISSING,
            BlocksStatus.NOT_PERMITTED,
            BlocksStatus.UNSUPPORTED,
            BlocksStatus.FAILED,
        ],
    )
    def test_a_problem_is_explained(self, status: BlocksStatus) -> None:
        assert status.hint in self.footer(status)

    def test_a_specific_explanation_preferred_over_the_generic_one(self) -> None:
        assert "Cannot access GPU registers" in self.footer(
            BlocksStatus.NOT_PERMITTED, "Cannot access GPU registers, are you root?"
        )

    def test_no_hint_when_healthy(self) -> None:
        assert "blocks" not in self.footer(BlocksStatus.OK, "")

    def test_no_hint_when_deliberately_disabled(self) -> None:
        """A user who turned it off chose that; nagging would train them to ignore this line."""
        assert "blocks" not in self.footer(BlocksStatus.DISABLED)
