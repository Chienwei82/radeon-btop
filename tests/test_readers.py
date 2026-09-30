"""Tests for the low-level sysfs readers: fsutil, hwmon, dpm and throttle decoding."""

from pathlib import Path

import pytest

from gputop.readers.dpm import (
    parse_dpm_table,
    parse_power_profile,
    read_clock,
    read_clock_max,
    read_link,
    read_performance_level,
)
from gputop.readers.fsutil import (
    link_target,
    parse_drm_size,
    read_bytes,
    read_hex_int,
    read_int,
    read_text,
    resolve,
)
from gputop.readers.hwmon import (
    canonical_label,
    read_fan,
    read_power,
    read_temperatures,
    sorted_temperature_names,
)
from gputop.readers.throttle import ThrottleStatus, decode_throttle
from tests.conftest import write


class TestFsutil:
    """Every read helper is total: failure yields ``None``, never an exception."""

    def test_read_text_of_a_missing_file(self, tmp_path: Path) -> None:
        assert read_text(tmp_path / "absent") is None

    def test_read_text_of_a_directory(self, tmp_path: Path) -> None:
        assert read_text(tmp_path) is None

    def test_read_text_strips_whitespace(self, tmp_path: Path) -> None:
        write(tmp_path / "f", "  value \n")
        assert read_text(tmp_path / "f") == "value"

    def test_read_text_survives_invalid_utf8(self, tmp_path: Path) -> None:
        """sysfs is byte oriented; invalid UTF-8 must not raise."""
        write(tmp_path / "f", b"\xff\xfe\x00bad")
        assert read_text(tmp_path / "f") is not None

    def test_read_int(self, tmp_path: Path) -> None:
        write(tmp_path / "f", "42\n")
        assert read_int(tmp_path / "f") == 42

    def test_read_int_of_garbage(self, tmp_path: Path) -> None:
        write(tmp_path / "f", "not a number")
        assert read_int(tmp_path / "f") is None

    def test_read_int_of_empty(self, tmp_path: Path) -> None:
        write(tmp_path / "f", "")
        assert read_int(tmp_path / "f") is None

    def test_read_hex_int(self, tmp_path: Path) -> None:
        write(tmp_path / "vendor", "0x1002\n")
        assert read_hex_int(tmp_path / "vendor") == 0x1002

    def test_read_hex_int_of_garbage(self, tmp_path: Path) -> None:
        write(tmp_path / "vendor", "0xZZZZ")
        assert read_hex_int(tmp_path / "vendor") is None

    def test_read_bytes(self, tmp_path: Path) -> None:
        write(tmp_path / "f", b"\x78\x00\x01")
        assert read_bytes(tmp_path / "f") == b"\x78\x00\x01"

    def test_read_bytes_respects_the_limit(self, tmp_path: Path) -> None:
        write(tmp_path / "f", b"x" * 100)
        assert read_bytes(tmp_path / "f", limit=10) == b"x" * 10

    def test_read_bytes_of_a_missing_file(self, tmp_path: Path) -> None:
        assert read_bytes(tmp_path / "absent") is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("0", 0), ("4116 KiB", 4116 * 1024), ("1 MiB", 1024**2), ("2 GiB", 2 * 1024**3)],
    )
    def test_parse_drm_size_uses_binary_prefixes(self, text: str, expected: int) -> None:
        """The driver writes IEC prefixes, so KiB must be 1024 and not 1000."""
        assert parse_drm_size(text) == expected

    def test_parse_drm_size_handles_the_tab_spacing(self) -> None:
        """``drm-memory-gtt:\\t0 KiB`` carries an irregular separator."""
        assert parse_drm_size("  0 KiB") == 0

    def test_parse_drm_size_of_a_bare_number(self) -> None:
        assert parse_drm_size("512") == 512

    def test_parse_drm_size_of_an_unknown_unit(self) -> None:
        assert parse_drm_size("4 furlongs") is None

    def test_parse_drm_size_of_garbage(self) -> None:
        assert parse_drm_size("banana") is None
        assert parse_drm_size(None) is None

    def test_link_target_of_a_symlink(self, tmp_path: Path) -> None:
        (tmp_path / "link").symlink_to("/some/target")
        assert link_target(tmp_path / "link") == "/some/target"

    def test_link_target_of_a_regular_file(self, tmp_path: Path) -> None:
        write(tmp_path / "f", "x")
        assert link_target(tmp_path / "f") is None

    def test_resolve_of_a_missing_path_does_not_raise(self, tmp_path: Path) -> None:
        assert resolve(tmp_path / "absent") is not None


class TestHwmon:
    """Sensor label handling across vendor quirks."""

    def test_temperatures_are_labelled(self, tmp_path: Path) -> None:
        write(tmp_path / "temp1_input", "49000")
        write(tmp_path / "temp1_label", "edge")
        write(tmp_path / "temp2_input", "56000")
        write(tmp_path / "temp2_label", "junction")
        write(tmp_path / "temp3_input", "52000")
        write(tmp_path / "temp3_label", "mem")

        assert read_temperatures(tmp_path) == {"edge": 49.0, "junction": 56.0, "mem": 52.0}

    def test_missing_label_falls_back_to_index(self, tmp_path: Path) -> None:
        write(tmp_path / "temp4_input", "50000")
        assert read_temperatures(tmp_path) == {"temp4": 50.0}

    def test_unreadable_input_is_skipped(self, tmp_path: Path) -> None:
        write(tmp_path / "temp1_input", "garbage")
        write(tmp_path / "temp2_input", "50000")
        write(tmp_path / "temp2_label", "mem")

        assert read_temperatures(tmp_path) == {"mem": 50.0}

    def test_none_directory(self) -> None:
        assert read_temperatures(None) == {}

    def test_missing_directory(self, tmp_path: Path) -> None:
        assert read_temperatures(tmp_path / "absent") == {}

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("edge", "edge"),
            ("junction", "junction"),
            ("hotspot", "junction"),
            ("mem", "mem"),
            ("soc", "soc"),
            ("vrmem", "mem"),
            ("", "temp1"),
        ],
    )
    def test_canonical_label(self, label: str, expected: str) -> None:
        assert canonical_label(label, 1) == expected

    def test_unknown_label_is_preserved(self) -> None:
        """An unrecognised label is surfaced rather than discarded."""
        assert canonical_label("weird-sensor", 5) == "weird-sensor"

    def test_sensor_order_is_stable_and_known_first(self) -> None:
        order = sorted_temperature_names({"core": 1, "edge": 2, "zzz": 3, "junction": 4})
        # Known sensors in canonical order, unknown ones last.
        assert order == ["edge", "junction", "core", "zzz"]

    def test_power_prefers_the_average(self, tmp_path: Path) -> None:
        write(tmp_path / "power1_average", "33000000")
        write(tmp_path / "power1_input", "31000000")
        assert read_power(tmp_path)[0] == 33.0

    def test_power_falls_back_to_input(self, tmp_path: Path) -> None:
        write(tmp_path / "power1_input", "31000000")
        assert read_power(tmp_path)[0] == 31.0

    def test_power_cap(self, tmp_path: Path) -> None:
        write(tmp_path / "power1_cap", "211000000")
        assert read_power(tmp_path)[1] == 211.0

    def test_power_cap_falls_back_to_max(self, tmp_path: Path) -> None:
        write(tmp_path / "power1_cap_max", "225000000")
        assert read_power(tmp_path)[1] == 225.0

    def test_power_absent(self) -> None:
        assert read_power(None) == (None, None)

    def test_fan_reading(self, tmp_path: Path) -> None:
        write(tmp_path / "fan1_input", "2400")
        write(tmp_path / "fan1_max", "3000")
        write(tmp_path / "pwm1", "128")

        rpm, max_rpm, pwm = read_fan(tmp_path)
        assert (rpm, max_rpm) == (2400, 3000)
        assert pwm is not None and pwm == pytest.approx(50.2, abs=0.1)

    def test_fan_absent(self) -> None:
        assert read_fan(None) == (None, None, None)


class TestDpm:
    """Clock tables and performance level, which carry the awkward real-world format."""

    def test_parses_the_current_row(self) -> None:
        """The active row is the one marked with ``*``."""
        table = "0: 500Mhz *\n1: 2475Mhz \n"
        assert parse_dpm_table(table) == (500, 2475)

    def test_lowercase_h_in_the_unit(self) -> None:
        """The kernel writes ``Mhz``, not ``MHz`` -- an easy and silent mistake."""
        assert parse_dpm_table("0: 100Mhz *\n1: 2000Mhz \n") == (100, 2000)

    @pytest.mark.parametrize("unit", ["Mhz", "MHz", "Uhz", "GHz"])
    def test_all_units(self, unit: str) -> None:
        assert parse_dpm_table(f"0: 100{unit} *\n1: 200{unit} \n") == (100, 200)

    def test_maximum_is_the_highest_row_not_the_last(self) -> None:
        table = "0: 100Mhz\n1: 2400Mhz\n2: 800Mhz *\n"
        assert parse_dpm_table(table) == (800, 2400)

    def test_unmarked_table_uses_the_highest_row_as_current(self) -> None:
        """A single-state APU table has no marker; the only row is the current one."""
        assert parse_dpm_table("0: 400Mhz \n") == (400, 400)

    def test_empty_table(self) -> None:
        assert parse_dpm_table("") == (None, None)
        assert parse_dpm_table(None) == (None, None)

    def test_garbage_table(self) -> None:
        assert parse_dpm_table("this is not a table") == (None, None)

    def test_clock_prefers_the_direct_attribute(self, tmp_path: Path) -> None:
        write(tmp_path / "current_sclk", "2200\n")
        write(tmp_path / "pp_dpm_sclk", "0: 500Mhz\n1: 2400Mhz *\n")
        assert read_clock(tmp_path, "current_sclk", "pp_dpm_sclk") == (2200, 2)

    def test_clock_falls_back_to_the_dpm_table(self, tmp_path: Path) -> None:
        """``current_sclk`` is absent on recent drivers, so the table must cover it."""
        write(tmp_path / "pp_dpm_sclk", "0: 500Mhz *\n1: 2400Mhz \n")
        assert read_clock(tmp_path, "current_sclk", "pp_dpm_sclk") == (500, 1)

    def test_clock_absent_everywhere(self, tmp_path: Path) -> None:
        assert read_clock(tmp_path, "current_sclk", "pp_dpm_sclk") == (None, 0)

    def test_clock_max_from_the_table(self, tmp_path: Path) -> None:
        write(tmp_path / "pp_dpm_sclk", "0: 500Mhz *\n1: 2400Mhz \n")
        assert read_clock_max(tmp_path, "current_sclk_max", "pp_dpm_sclk") == 2400

    def test_clock_max_from_the_attribute(self, tmp_path: Path) -> None:
        write(tmp_path / "current_sclk_max", "2600\n")
        assert read_clock_max(tmp_path, "current_sclk_max", "pp_dpm_sclk") == 2600

    def test_performance_level(self, tmp_path: Path) -> None:
        write(tmp_path / "power_dpm_force_performance_level", "auto\n")
        assert read_performance_level(tmp_path) == "auto"

    def test_performance_level_absent(self, tmp_path: Path) -> None:
        assert read_performance_level(tmp_path) is None

    def test_power_profile_finds_the_starred_entry(self) -> None:
        """The profile table is long; only the starred line is active."""
        table = (
            "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME) ...\n"
            " 0 BOOTUP_DEFAULT*:\n"
            " 1 3D_FULL_SCREEN :\n"
            " 2 POWER_SAVING :\n"
        )
        assert parse_power_profile(table) == "BOOTUP_DEFAULT"

    def test_power_profile_picks_the_later_starred_entry(self) -> None:
        table = " 0 BOOTUP_DEFAULT :\n 1 3D_FULL_SCREEN*:\n 2 POWER_SAVING :\n"
        assert parse_power_profile(table) == "3D_FULL_SCREEN"

    def test_power_profile_without_a_marker(self) -> None:
        assert parse_power_profile(" 0 BOOTUP_DEFAULT :\n 1 VIDEO :\n") == "BOOTUP_DEFAULT"

    def test_power_profile_absent(self) -> None:
        assert parse_power_profile(None) is None

    def test_pcie_link(self, tmp_path: Path) -> None:
        write(tmp_path / "current_link_width", "16\n")
        write(tmp_path / "current_link_speed", "32\n")
        assert read_link(tmp_path) == (16, 32)


class TestThrottle:
    """Throttle bits are sparse, so a range-based decoder would be wrong."""

    def test_no_throttling(self) -> None:
        assert decode_throttle(0) == frozenset()

    def test_single_bit(self) -> None:
        assert decode_throttle(1 << 32) == {"TEMP_GPU"}

    def test_multiple_sparse_bits(self) -> None:
        raw = (1 << 0) | (1 << 21) | (1 << 47)
        assert decode_throttle(raw) == {"PPT0", "EDC_CPU", "PROCHOT_GFX"}

    def test_unknown_bits_are_ignored_not_fatal(self) -> None:
        """A future kernel adding a reason must not break decoding of known ones."""
        raw = (1 << 32) | (1 << 60)
        assert decode_throttle(raw) == {"TEMP_GPU"}

    def test_negative_is_treated_as_none(self) -> None:
        assert decode_throttle(-1) == frozenset()

    def test_flags_are_distinct_bit_positions(self) -> None:
        """No two reasons may share a bit, which would make decoding ambiguous."""
        values = [int(member) for member in ThrottleStatus]
        assert len(values) == len(set(values))
        assert all(bin(v).count("1") == 1 for v in values)
