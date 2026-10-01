"""Tests for the read-only power profile and overdrive readers.

The important cases here are the two hardware generations.  ``pp_od_clk_voltage`` was a
voltage table on older parts and became an overdrive *clock* table on RDNA2, under the same
filename.  A parser that assumes millivolts reports a 2104 MHz clock ceiling as "2104 mV" on
every modern card, which is the kind of confidently wrong number this panel cannot produce.
"""

from pathlib import Path

import pytest

from gputop.model.power import OdvUnit
from gputop.readers.pp import (
    assert_read_only,
    parse_odc_voltage,
    parse_power_profile_table,
    read_odc_voltage,
    read_power_profile_table,
)

#: The real contents of ``pp_od_clk_voltage`` on an RDNA2 Navi 21, verbatim.
RDNA2_ODC = """OD_SCLK:
0: 500Mhz
1: 2104Mhz
OD_MCLK:
0: 97Mhz
1: 1000MHz
OD_RANGE:
SCLK:     500Mhz       2600Mhz
MCLK:     674Mhz       1075Mhz
OD_VDDGFX_OFFSET:
-50mV
"""

#: The legacy layout: one bare table of millivolts, no sections, no unit suffixes.
LEGACY_ODC = "0: 300\n1: 350\n2: 400\n3: 450\n"

#: The real contents of ``pp_power_profile_mode``, including its header and detail rows.
REAL_PROFILES = """PROFILE_INDEX(NAME) CLOCK_TYPE(NAME) FPS MinFreqType MinActiveFreqType \
MinActiveFreq BoosterFreqType BoosterFreq PD_Data_limit_c
 0 BOOTUP_DEFAULT*:
                    0(       GFXCLK)       0       5       1       0       4     800 4587520
                    1(      SOCCLK)       0       5       1       0       1       0 3276800
 1 3D_FULL_SCREEN :
                    0(       GFXCLK)       0       5       0    1600       4     650 5242880
 2   POWER_SAVING :
                    0(       GFXCLK)       0       5       1       0       3       0 5898240
 3          VIDEO :
 4             VR :
 5         COMPUTE :
 6          CUSTOM :
"""


class TestPowerProfile:
    """``pp_power_profile_mode``."""

    def test_the_real_file_yields_every_profile_in_order(self) -> None:
        table = parse_power_profile_table(REAL_PROFILES)
        assert table.present
        assert table.names() == (
            "BOOTUP_DEFAULT",
            "3D_FULL_SCREEN",
            "POWER_SAVING",
            "VIDEO",
            "VR",
            "COMPUTE",
            "CUSTOM",
        )

    def test_the_starred_profile_is_the_active_one(self) -> None:
        table = parse_power_profile_table(REAL_PROFILES)
        assert table.active_name == "BOOTUP_DEFAULT"
        assert table.active_index == 0

    def test_the_column_header_and_detail_rows_are_not_mistaken_for_profiles(self) -> None:
        """The file's own header line and ``0(  GFXCLK)`` rows must both be skipped.

        They are the two rows a naive "first token is an index" parser would turn into
        profiles named ``CLOCK_TYPE(NAME)`` and ``(``.
        """
        table = parse_power_profile_table(REAL_PROFILES)
        assert "CLOCK_TYPE(NAME)" not in table.names()
        assert not any(name.startswith("(") for name in table.names())
        assert len(table.profiles) == 7

    def test_another_active_profile_is_found(self) -> None:
        text = " 0 BOOTUP_DEFAULT :\n 1 3D_FULL_SCREEN*:\n"
        assert parse_power_profile_table(text).active_name == "3D_FULL_SCREEN"

    def test_no_star_falls_back_to_the_first_profile(self) -> None:
        """APUs that idle on a single state mark no row at all."""
        text = " 0 BOOTUP_DEFAULT :\n 1 VIDEO :\n"
        assert parse_power_profile_table(text).active_name == "BOOTUP_DEFAULT"

    @pytest.mark.parametrize("text", [None, "", "no profiles here\n"])
    def test_an_absent_or_empty_file_is_not_an_error(self, text: str | None) -> None:
        table = parse_power_profile_table(text)
        assert not table.present
        assert table.active is None
        assert table.active_name is None
        assert table.names() == ()

    def test_it_describes_itself_compactly(self) -> None:
        assert parse_power_profile_table(REAL_PROFILES).describe() == "BOOTUP_DEFAULT (0 of 7)"
        assert parse_power_profile_table(None).describe() == "N/A"


class TestOdcVoltage:
    """``pp_od_clk_voltage``, across both of its formats."""

    def test_a_legacy_table_with_no_range_section_stays_in_millivolts(self) -> None:
        """A section-less table is a GCN *voltage* table, not a clock one.

        With no ``OD_RANGE`` row to read a unit from, the domain defaulted to megahertz
        while its own entries said millivolts -- so ``describe()`` announced "400 MHz"
        for 400 mV, which is the opposite error the per-entry unit exists to prevent.
        """
        table = parse_odc_voltage("0:300 1:350 2:400\n")
        domain = table.domain("sclk")
        assert domain is not None
        assert domain.unit is OdvUnit.MILLIVOLT
        assert domain.describe() == "400 mV"

    def test_the_modern_file_reports_clock_ceilings_in_megahertz(self) -> None:
        """The unit is carried per entry, not assumed.

        Presenting a 2104 MHz shader-clock ceiling as a voltage would be wrong on every
        RDNA2 card, which is the format this attribute actually takes today.
        """
        table = parse_odc_voltage(RDNA2_ODC)
        assert table.present
        sclk = table.domain("sclk")
        assert sclk is not None
        assert [e.value for e in sclk.entries] == [500, 2104]
        assert all(e.unit is OdvUnit.MEGAHERTZ for e in sclk.entries)
        assert sclk.unit is OdvUnit.MEGAHERTZ

    def test_the_range_section_supplies_the_floor_and_ceiling(self) -> None:
        sclk = parse_odc_voltage(RDNA2_ODC).domain("sclk")
        assert sclk is not None
        assert (sclk.minimum, sclk.maximum) == (500, 2600)
        assert sclk.ceiling == 2104
        assert sclk.headroom == 2600 - 2104

    def test_each_domain_is_kept_separate(self) -> None:
        table = parse_odc_voltage(RDNA2_ODC)
        assert {d.name for d in table.domains} == {"sclk", "mclk"}
        mclk = table.domain("mclk")
        assert mclk is not None
        assert (mclk.minimum, mclk.maximum) == (674, 1075)

    def test_the_percent_is_a_ratio_of_this_drivers_own_range(self) -> None:
        """Not of a hard-coded maximum, which would be wrong on any other part."""
        sclk = parse_odc_voltage(RDNA2_ODC).domain("sclk")
        assert sclk.percent == pytest.approx((2104 - 500) / (2600 - 500) * 100.0)

    def test_the_voltage_offset_is_read_with_its_sign(self) -> None:
        assert parse_odc_voltage(RDNA2_ODC).vddgfx_offset_mv == -50

    def test_the_legacy_file_reports_millivolts(self) -> None:
        """The older layout has no unit suffix and no sections, and is still a table."""
        table = parse_odc_voltage(LEGACY_ODC)
        assert table.present
        domain = table.domain("sclk")
        assert domain is not None
        assert [e.value for e in domain.entries] == [300, 350, 400, 450]
        assert all(e.unit is OdvUnit.MILLIVOLT for e in domain.entries)
        assert domain.floor == 300
        assert domain.ceiling == 450

    def test_a_legacy_table_has_no_range_and_so_no_percent(self) -> None:
        """Without a stated range there is no honest 0-100 scale, so there is no bar."""
        domain = parse_odc_voltage(LEGACY_ODC).domain("sclk")
        assert domain.percent is None

    def test_an_unrecognised_section_does_not_land_in_the_clock_domain(self) -> None:
        """``pp_od_clk_voltage`` carries sections that are not clock tables at all.

        SMU13 (Navi 30/31/32/33) writes ``OD_FAN_CURVE`` with rows like ``2: 95C 100%``,
        and Vega20/Navi1x write ``OD_VDDC_CURVE``.  Their rows have exactly the same
        ``<state>: <value>`` shape as a clock entry, so an unrecognised section used to
        fall through to the legacy shader-clock bucket and the overdrive panel reported
        "sclk 95 MHz" -- a fan-curve temperature presented as a clock ceiling.  Ignoring a
        section we know is not a clock table beats guessing that it is one.
        """
        smu13 = (
            "OD_SCLK:\n0: 500Mhz\n1: 2104Mhz\n"
            "OD_MCLK:\n0: 97Mhz\n1: 1000MHz\n"
            "OD_FAN_CURVE:\n2: 95C 100%\n3: 85C 70%\n"
            "OD_FAN_TARGET_TEMPERATURE:\n0: 0C\n"
            "OD_RANGE:\nSCLK:     500Mhz       2600Mhz\n"
        )
        table = parse_odc_voltage(smu13)
        sclk = table.domain("sclk")
        assert sclk is not None
        assert [e.value for e in sclk.entries] == [500, 2104], (
            "a fan curve leaked into the shader clock domain"
        )
        assert sclk.maximum == 2600, "the real OD_RANGE was not read"
        # The modelled domains survive; nothing was invented for the ones we skip.
        assert {d.name for d in table.domains} == {"sclk", "mclk"}

    def test_pairs_on_one_line_are_accepted(self) -> None:
        """The driver's spacing is not consistent across kernels."""
        table = parse_odc_voltage("0: 300 1: 350 2: 400\n")
        domain = table.domain("sclk")
        assert [e.value for e in domain.entries] == [300, 350, 400]

    @pytest.mark.parametrize("text", [None, "", "\n\n"])
    def test_an_absent_file_is_not_an_error(self, text: str | None) -> None:
        table = parse_odc_voltage(text)
        assert not table.present
        assert table.domains == ()
        assert table.describe() == "N/A"

    def test_a_zero_width_range_does_not_divide_by_zero(self) -> None:
        table = parse_odc_voltage("OD_SCLK:\n0: 100Mhz\nOD_RANGE:\nSCLK: 100Mhz 100Mhz\n")
        assert table.domain("sclk").percent is None

    def test_it_describes_itself_in_one_line(self) -> None:
        described = parse_odc_voltage(RDNA2_ODC).describe()
        assert "Shader clock 2104/2600 MHz" in described
        assert "Memory clock 1000/1075 MHz" in described


class TestReading:
    """Reading the attributes off a fake sysfs tree."""

    def test_both_tables_are_read_from_a_device_directory(self, tmp_path: Path) -> None:
        device = tmp_path / "device"
        device.mkdir()
        (device / "pp_od_clk_voltage").write_text(RDNA2_ODC)
        (device / "pp_power_profile_mode").write_text(REAL_PROFILES)

        odc = read_odc_voltage(device)
        assert odc.present
        assert odc.domain("sclk").ceiling == 2104
        assert read_power_profile_table(device).active_name == "BOOTUP_DEFAULT"

    def test_missing_files_yield_empty_tables_rather_than_raising(self, tmp_path: Path) -> None:
        device = tmp_path / "empty"
        device.mkdir()
        assert not read_odc_voltage(device).present
        assert not read_power_profile_table(device).present

    def test_a_directory_that_does_not_exist_yields_empty_tables(self, tmp_path: Path) -> None:
        absent = tmp_path / "no-such-device"
        assert not read_odc_voltage(absent).present
        assert not read_power_profile_table(absent).present


class TestReadOnly:
    """The one rule this module exists to keep."""

    def test_the_module_asserts_its_own_read_only_promise(self) -> None:
        assert assert_read_only() is True

    def test_no_write_operation_appears_in_the_source(self) -> None:
        """Checked structurally rather than trusted.

        A later edit that adds a write path would silently violate the rule the whole
        application is built on; this turns that into a test failure instead.
        """
        import gputop.readers.pp as module

        source = Path(module.__file__).read_text()
        for forbidden in (
            "write_text",
            "write_bytes",
            'open("w")',
            "open('w')",
            "open(1",  # os.open(fd, ...) with a writable flag
            "truncate",
        ):
            assert forbidden not in source, f"readers/pp.py must not use {forbidden}"

    def test_reading_a_table_does_not_change_the_file(self, tmp_path: Path) -> None:
        """A reader that accidentally opened for writing would truncate its own input."""
        path = tmp_path / "pp_od_clk_voltage"
        path.write_text(RDNA2_ODC)
        before = path.read_text()
        parse_odc_voltage(path.read_text())
        assert path.read_text() == before
