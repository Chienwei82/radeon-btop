"""Binary ``gpu_metrics`` parser tests.

These are the highest-risk tests in the project: a silently wrong offset produces
plausible-looking but wrong numbers, which is the worst possible failure for a monitor.
The layout assertions therefore use literal byte offsets captured from real hardware
rather than re-deriving expectations from the registry under test.
"""

import struct

import pytest

from gputop.readers.metrics_bin import (
    ABI_REGISTRY,
    SENTINEL_U16,
    SENTINEL_U32,
    SENTINEL_U64,
    GpuMetricsParser,
    unrecognised_header,
)
from tests.conftest import V1_3_OFFSETS, V1_3_STRUCTURE_SIZE, build_metric_table

V1_3 = ABI_REGISTRY[(1, 3)]


def test_v1_3_offsets_match_hardware() -> None:
    """The aligner must reproduce the byte offsets read off a real Navi 44 table."""
    for name, expected in V1_3_OFFSETS.items():
        assert V1_3.field_map[name] == expected, f"{name} at wrong offset"


def test_v1_3_calcsize_matches_declared_structure_size() -> None:
    """``Struct.size`` must equal the driver-declared size so truncation is detectable."""
    assert V1_3.body.size == V1_3_STRUCTURE_SIZE


@pytest.mark.parametrize("revision", sorted(ABI_REGISTRY))
def test_every_abi_calcsize_is_self_consistent(revision: tuple[int, int]) -> None:
    """Each registered layout must place every field inside its own declared size."""
    abi = ABI_REGISTRY[revision]
    sizes = {"B": 1, "H": 2, "I": 4, "Q": 8}
    assert abi.body.size >= 4, "every layout must at least span the header"
    for spec in abi.fields:
        offset = abi.field_map[spec.name]
        assert offset + sizes[spec.code] <= abi.body.size
    # Fields must be laid out in declaration order without overlapping.
    ordered = [abi.field_map[spec.name] for spec in abi.fields]
    assert ordered == sorted(ordered)


def test_parse_v1_3_reads_every_field() -> None:
    """A fully populated v1.3 table decodes to the documented values."""
    blob = build_metric_table(
        V1_3,
        {
            "temperature_edge": 49,
            "temperature_hotspot": 56,
            "temperature_mem": 52,
            "average_gfx_activity": 37,
            "average_umc_activity": 12,
            "average_socket_power": 33,
            "energy_accumulator": 135_145_864,
            "system_clock_counter": 987_654_321,
            "current_gfxclk": 500,
            "current_uclk": 1000,
            "throttle_status": 0,
            "current_fan_speed": 737,
            "pcie_link_width": 16,
            "pcie_link_speed": 80,
            "voltage_soc": 918,
            "voltage_gfx": 768,
            "voltage_mem": 900,
        },
    )
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.abi_label == "v1.3"
    assert parsed.structure_size == V1_3_STRUCTURE_SIZE
    assert [(t.label, t.celsius) for t in parsed.temperatures] == [
        ("edge", 49),
        ("junction", 56),
        ("mem", 52),
    ]
    assert parsed.gpu_activity_percent == 37.0
    assert parsed.umc_activity_percent == 12.0
    assert parsed.power_draw_w == 33.0
    assert parsed.energy_mj == 135_145_864
    assert parsed.sclk_mhz == 500
    assert parsed.mclk_mhz == 1000
    assert parsed.fan_rpm == 737
    assert parsed.voltages_mv == (("soc", 918), ("gfx", 768), ("mem", 900))
    assert parsed.pcie is not None
    assert parsed.pcie.width == 16
    assert parsed.pcie.generation == 5
    assert parsed.pcie.describe() == "Gen5 x16"
    assert parsed.throttle is not None
    assert not parsed.throttle.is_throttling


def test_unsupported_fields_become_none_not_zero() -> None:
    """Sentinel values must become ``None``, never a plausible-looking 0 or 65535."""
    blob = build_metric_table(
        V1_3,
        {
            "temperature_hbm": SENTINEL_U16,
            "gfx_activity_acc": SENTINEL_U32,
            "firmware_timestamp": SENTINEL_U64,
            "average_umc_activity": SENTINEL_U16,
        },
    )
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert "l3" not in {t.label for t in parsed.temperatures}
    assert parsed.umc_activity_percent is None


def test_clock_of_zero_is_kept_because_idle_is_real() -> None:
    """A zero clock is a legitimate idle reading, not an unsupported field."""
    blob = build_metric_table(V1_3, {"current_gfxclk": 0, "average_gfx_activity": 0})
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.sclk_mhz == 0
    assert parsed.gpu_activity_percent == 0.0


def test_implausible_clock_is_rejected() -> None:
    """A clock beyond any real ASIC is dropped rather than shown as a 4-digit number.

    60000 MHz is chosen because it fits the 16-bit field -- a value that does not fit
    could never be produced by the driver and would fail in the fixture instead.
    """
    blob = build_metric_table(V1_3, {"current_gfxclk": 60_000})
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.sclk_mhz is None


def test_absent_voltage_rail_is_rejected() -> None:
    """RDNA 3 has no discrete GFX voltage rail; a near-zero value must not be shown."""
    blob = build_metric_table(V1_3, {"voltage_soc": 918, "voltage_gfx": 0, "voltage_mem": 900})
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.voltages_mv == (("soc", 918), ("mem", 900))


@pytest.mark.parametrize("revision", sorted(ABI_REGISTRY))
def test_round_trip_every_registered_abi(revision: tuple[int, int]) -> None:
    """Every registered ABI parses its own template without raising."""
    abi = ABI_REGISTRY[revision]
    blob = build_metric_table(abi, {"throttle_status": 0})
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.abi_label == abi.label
    assert parsed.structure_size == abi.body.size


def test_unknown_format_revision_returns_none() -> None:
    """A v3.x table on a future kernel must be refused, not guessed at."""
    blob = build_metric_table(V1_3, structure_size=V1_3.body.size)
    header = bytearray(blob[:4])
    header[2] = 3  # format_revision = 3
    header[3] = 0
    patched = bytes(header) + blob[4:]

    assert GpuMetricsParser().parse(patched) is None
    assert unrecognised_header(patched) == (V1_3.body.size, 3, 0)


def test_unknown_content_revision_returns_none() -> None:
    """An unregistered content revision within a known format is refused too."""
    blob = build_metric_table(V1_3, structure_size=V1_3.body.size)
    header = bytearray(blob[:4])
    header[3] = 9  # content_revision = 9
    patched = bytes(header) + blob[4:]

    assert GpuMetricsParser().parse(patched) is None


def test_truncated_blob_returns_none_rather_than_raising() -> None:
    """A file shorter than its own declared size is rejected."""
    blob = build_metric_table(V1_3)
    assert GpuMetricsParser().parse(blob[:-1]) is None


def test_shorter_than_header_returns_none() -> None:
    """A file too small to hold a header cannot be decoded."""
    assert GpuMetricsParser().parse(b"") is None
    assert GpuMetricsParser().parse(b"\x78\x00") is None
    assert unrecognised_header(b"\x78\x00") is None


def test_structure_larger_than_the_file_is_refused() -> None:
    """A header claiming more bytes than the file holds is malformed.

    Decoding it would mean inventing the missing tail, and every field is a
    correctness claim, so the parser falls back to sysfs instead.
    """
    blob = build_metric_table(V1_3, {"temperature_edge": 44})
    header = bytearray(blob[:4])
    header[0:2] = (V1_3.body.size + 16).to_bytes(2, "little")

    assert GpuMetricsParser().parse(bytes(header) + blob[4:]) is None


def test_layout_longer_than_the_table_decodes_what_is_present() -> None:
    """A table shorter than the layout we know still yields the fields it does contain.

    This covers the reverse of forward compatibility: if the kernel ever trims the
    structure, or our layout is ahead of it, the shared prefix must still decode.
    """
    trimmed = 60
    blob = build_metric_table(
        V1_3,
        {"temperature_edge": 44, "average_gfx_activity": 21},
        structure_size=trimmed,
    )[:trimmed]

    parsed = GpuMetricsParser().parse(blob)
    assert parsed is not None
    # Fields inside the surviving prefix decode normally.
    assert parsed.temperatures[0].celsius == 44
    assert parsed.gpu_activity_percent == 21.0
    assert parsed.sclk_mhz == 0  # current_gfxclk sits at offset 54, inside the prefix
    # Fields past the trim point are absent, not zero -- otherwise a truncated table
    # would claim "fan stopped, nothing throttling" as if it had been measured.
    assert parsed.fan_rpm is None  # offset 72
    assert parsed.throttle is None  # offset 68
    assert parsed.voltages_mv == ()  # offset 104


def test_trailing_bytes_from_a_newer_kernel_are_ignored() -> None:
    """Extra trailing bytes are tolerated, since a newer driver may append fields."""
    abi = V1_3
    blob = build_metric_table(abi, {"current_gfxclk": 900}, trailing=32)
    parsed = GpuMetricsParser().parse(blob)

    assert parsed is not None
    assert parsed.sclk_mhz == 900
    assert parsed.structure_size == abi.body.size + 32


def test_parse_file_on_missing_path_returns_none(tmp_path) -> None:
    """A missing file is ordinary, not an error."""
    assert GpuMetricsParser().parse_file(tmp_path / "nope") is None


def test_parse_file_on_unreadable_path_returns_none(tmp_path) -> None:
    """An unreadable path is ordinary, not an error."""
    directory = tmp_path / "a_directory"
    directory.mkdir()
    assert GpuMetricsParser().parse_file(directory) is None


def test_real_hardware_blob_decodes() -> None:
    """A capture of the reference machine's exact metric values decodes correctly.

    These are the readings a Navi 44 produced while a browser was running: 49/56/52
    degrees on edge/junction/mem, 33 W, 918/768/900 mV, PCIe Gen5 x16.  Regression-guarding
    them catches any aligner change that would silently shift every 64-bit field.
    """
    blob = build_metric_table(
        V1_3,
        {
            "temperature_edge": 49,
            "temperature_hotspot": 56,
            "temperature_mem": 52,
            "average_gfx_activity": 8,
            "average_socket_power": 33,
            "current_gfxclk": 500,
            "current_uclk": 1000,
            "pcie_link_width": 16,
            "pcie_link_speed": 80,
            "voltage_soc": 918,
            "voltage_gfx": 768,
            "voltage_mem": 900,
        },
    )
    parsed = GpuMetricsParser().parse(blob)
    assert parsed is not None
    assert [t.celsius for t in parsed.temperatures] == [49, 56, 52]
    assert parsed.voltages_mv == (("soc", 918), ("gfx", 768), ("mem", 900))
    assert parsed.pcie is not None and parsed.pcie.describe() == "Gen5 x16"
    assert parsed.structure_size == 120


def test_header_structure_size_excludes_nothing() -> None:
    """``structure_size`` counts the header, so a body-only size would be off by four."""
    blob = build_metric_table(V1_3)
    declared = struct.unpack_from("<H", blob, 0)[0]
    assert declared == len(blob) == V1_3_STRUCTURE_SIZE
