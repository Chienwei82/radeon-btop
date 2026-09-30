"""Parser for the versioned binary ``gpu_metrics`` sysfs file.

Layout, verified against a live Navi 21 / RX 6800 (``amdgpu`` 6.19.4, kernel 7.0.0-34)::

    struct metrics_table_header {   /* offset 0, 4 bytes */
        __u16 structure_size;       /* total buffer size INCLUDING this header */
        __u8  format_revision;      /* 1 = v1.x, 2 = v2.x, 3 = v3.x          */
        __u8  content_revision;     /* minor revision within the format        */
    };

The body follows at offset 4 and is fixed-width little-endian.  ``structure_size``
bounds the body, which is what makes the parser forward compatible: a driver reporting a
larger struct is truncated safely, a smaller one yields ``None`` for the missing tail,
and an unrecognised ``(format, content)`` pair is never guessed at.

Struct padding is reproduced with an explicit aligner rather than ``struct``'s native
alignment, because the C layout has to be reproduced identically on every architecture
and a silent misalignment would corrupt every 64-bit field.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from gputop.model.metrics import PcieLink, Temperature, ThrottleInfo
from gputop.readers.fsutil import read_bytes
from gputop.readers.throttle import decode_throttle

#: The common 4-byte header prefixing every metric table.
HEADER = struct.Struct("<HBB")
HEADER_SIZE = HEADER.size

#: Values written when a field is not implemented by the ASIC.  Observed on the reference
#: Navi 21: 0xFFFF for ``temperature_hbm`` (it has GDDR6, not HBM), 0xFFFFFFFF for the
#: activity accumulators and 0xFFFFFFFFFFFFFFFF for ``firmware_timestamp``.
#:
#: These are sentinels, not measurements: a card with HBM would populate that field, so a
#: missing reading must never be rendered as 65535 C or 65535 MHz.
SENTINEL_U16 = 0xFFFF
SENTINEL_U32 = 0xFFFFFFFF
SENTINEL_U64 = 0xFFFFFFFFFFFFFFFF

# Plausibility bounds.  These are per field family on purpose: a generic clamp would
# wrongly discard a legitimate idle reading such as ``sclk == 0``.
MAX_CLOCK_MHZ = 10_000
MAX_TEMPERATURE_C = 150
MAX_POWER_W = 10_000
MIN_VOLTAGE_MV = 50
MAX_VOLTAGE_MV = 2_000

_SIZE = {"B": 1, "H": 2, "I": 4, "Q": 8}

_TEMP_ORDER = ("edge", "junction", "mem", "soc", "core", "l3")


@dataclass(frozen=True, slots=True)
class _Field:
    """One named field in a metric table layout."""

    name: str
    code: str


def _aligner(
    fields: tuple[_Field, ...], start: int = HEADER_SIZE
) -> tuple[str, dict[str, int]]:
    """Build a ``struct`` format string plus a field-offset map.

    Padding is inserted so each field lands where a C compiler would place it, measured
    from the start of the buffer (offset 0, *including* the header).  That is what makes
    the 64-bit accumulators line up on multiples of eight.

    The leading header bytes are emitted as explicit pad.  They consume no values, so
    ``unpack_from`` still yields exactly one value per field, but they do make
    ``Struct.size`` equal the driver's declared ``structure_size`` -- which is what makes
    the truncation check in :meth:`GpuMetricsParser.parse` trustworthy.
    """
    fmt = f"{start}x"
    offsets: dict[str, int] = {}
    offset = start
    for spec in fields:
        size = _SIZE[spec.code]
        remainder = offset % size
        if remainder:
            fmt += f"{size - remainder}x"
            offset += size - remainder
        offsets[spec.name] = offset
        fmt += spec.code
        offset += size
    return fmt, offsets


@dataclass(frozen=True, slots=True)
class MetricsAbi:
    """One entry in the ABI registry."""

    fmt_revision: int
    content_revision: int
    body: struct.Struct
    field_map: Mapping[str, int]
    fields: tuple[_Field, ...]

    @property
    def label(self) -> str:
        """Human readable ABI label, e.g. ``v1.3``."""
        return f"v{self.fmt_revision}.{self.content_revision}"


_V1_TEMPS = (
    _Field("temperature_edge", "H"),
    _Field("temperature_hotspot", "H"),
    _Field("temperature_mem", "H"),
    _Field("temperature_vrgfx", "H"),
    _Field("temperature_vrsoc", "H"),
    _Field("temperature_vrmem", "H"),
)
_V1_ACTIVITIES = (
    _Field("average_gfx_activity", "H"),
    _Field("average_umc_activity", "H"),
    _Field("average_mm_activity", "H"),
)
_V1_CLOCK_NAMES = ("gfxclk", "socclk", "uclk", "vclk0", "dclk0", "vclk1", "dclk1")

_V1_0_FIELDS: tuple[_Field, ...] = (
    _Field("system_clock_counter", "Q"),
    *_V1_TEMPS,
    *_V1_ACTIVITIES,
    _Field("average_socket_power", "H"),
    _Field("energy_accumulator", "I"),
    *(_Field(f"average_{name}_frequency", "H") for name in _V1_CLOCK_NAMES),
    *(_Field(f"current_{name}", "H") for name in _V1_CLOCK_NAMES),
    _Field("throttle_status", "I"),
    _Field("current_fan_speed", "H"),
    _Field("pcie_link_width", "B"),
    _Field("pcie_link_speed", "B"),
)

_V1_1_FIELDS: tuple[_Field, ...] = (
    *_V1_TEMPS,
    *_V1_ACTIVITIES,
    _Field("average_socket_power", "H"),
    _Field("energy_accumulator", "Q"),
    _Field("system_clock_counter", "Q"),
    *(_Field(f"average_{name}_frequency", "H") for name in _V1_CLOCK_NAMES),
    *(_Field(f"current_{name}", "H") for name in _V1_CLOCK_NAMES),
    _Field("throttle_status", "I"),
    _Field("current_fan_speed", "H"),
    _Field("pcie_link_width", "H"),
    _Field("pcie_link_speed", "H"),
    _Field("padding", "H"),
    _Field("gfx_activity_acc", "I"),
    _Field("mem_activity_acc", "I"),
    _Field("temperature_hbm", "H"),
)

_V1_2_FIELDS: tuple[_Field, ...] = (*_V1_1_FIELDS, _Field("firmware_timestamp", "Q"))

_V1_3_FIELDS: tuple[_Field, ...] = (
    *_V1_2_FIELDS,
    _Field("voltage_soc", "H"),
    _Field("voltage_gfx", "H"),
    _Field("voltage_mem", "H"),
    _Field("padding1", "H"),
    # Present on the reference GPU (reads zero) and required to reach the declared
    # structure_size of 120, so it is decoded but never used as the throttle source.
    _Field("indep_throttle_status", "Q"),
)

_V2_0_FIELDS: tuple[_Field, ...] = (
    _Field("system_clock_counter", "Q"),
    _Field("temperature_gfx", "H"),
    _Field("temperature_soc", "H"),
    _Field("temperature_core", "H"),
    _Field("temperature_l3", "H"),
    _Field("average_gfx_activity", "H"),
    _Field("average_mm_activity", "H"),
    _Field("average_socket_power", "H"),
    _Field("average_cpu_power", "H"),
    _Field("average_soc_power", "H"),
    _Field("average_gfx_power", "H"),
    _Field("average_core_power", "H"),
    _Field("average_gfxclk_frequency", "H"),
    _Field("average_socclk_frequency", "H"),
    _Field("average_uclk_frequency", "H"),
    _Field("average_fclk_frequency", "H"),
    _Field("average_vclk_frequency", "H"),
    _Field("average_dclk_frequency", "H"),
    _Field("current_gfxclk", "H"),
    _Field("current_socclk", "H"),
    _Field("current_uclk", "H"),
    _Field("current_fclk", "H"),
    _Field("current_vclk", "H"),
    _Field("current_dclk", "H"),
    _Field("current_coreclk", "H"),
    _Field("current_l3clk", "H"),
    _Field("throttle_status", "I"),
    _Field("fan_pwm", "H"),
    _Field("padding", "H"),
)

_V2_1_FIELDS: tuple[_Field, ...] = (
    _Field("temperature_gfx", "H"),
    _Field("temperature_soc", "H"),
    _Field("temperature_core", "H"),
    _Field("temperature_l3", "H"),
    _Field("average_gfx_activity", "H"),
    _Field("average_mm_activity", "H"),
    _Field("system_clock_counter", "Q"),
    _Field("average_socket_power", "H"),
    _Field("average_cpu_power", "H"),
    _Field("average_soc_power", "H"),
    _Field("average_gfx_power", "H"),
    _Field("average_core_power", "H"),
    _Field("average_gfxclk_frequency", "H"),
    _Field("average_socclk_frequency", "H"),
    _Field("average_uclk_frequency", "H"),
    _Field("average_fclk_frequency", "H"),
    _Field("average_vclk_frequency", "H"),
    _Field("average_dclk_frequency", "H"),
    _Field("current_gfxclk", "H"),
    _Field("current_socclk", "H"),
    _Field("current_uclk", "H"),
    _Field("current_fclk", "H"),
    _Field("current_vclk", "H"),
    _Field("current_dclk", "H"),
    _Field("current_coreclk", "H"),
    _Field("current_l3clk", "H"),
    _Field("throttle_status", "I"),
    _Field("fan_pwm", "H"),
    _Field("padding", "H"),
)

_V2_2_FIELDS: tuple[_Field, ...] = (*_V2_1_FIELDS, _Field("indep_throttle_status", "Q"))


def _build(revision: int, content: int, fields: tuple[_Field, ...]) -> MetricsAbi:
    """Assemble one registry entry from its field description."""
    fmt, offsets = _aligner(fields)
    return MetricsAbi(
        fmt_revision=revision,
        content_revision=content,
        body=struct.Struct("<" + fmt),
        field_map=MappingProxyType(offsets),
        fields=fields,
    )


def _build_registry() -> Mapping[tuple[int, int], MetricsAbi]:
    """Register every layout gputop understands.

    Adding support for a new kernel revision is a single entry here plus one test
    fixture; no parser logic changes.
    """
    entries = (
        (1, 0, _V1_0_FIELDS),
        (1, 1, _V1_1_FIELDS),
        (1, 2, _V1_2_FIELDS),
        (1, 3, _V1_3_FIELDS),
        (2, 0, _V2_0_FIELDS),
        (2, 1, _V2_1_FIELDS),
        (2, 2, _V2_2_FIELDS),
    )
    registry = {
        (rev, content): _build(rev, content, fields) for rev, content, fields in entries
    }
    return MappingProxyType(registry)


#: All metric table layouts gputop understands.
ABI_REGISTRY: Mapping[tuple[int, int], MetricsAbi] = _build_registry()

#: Maps a v2.x APU temperature field onto its canonical label.
_V2_TEMPERATURE_FIELDS = (
    ("temperature_gfx", "edge"),
    ("temperature_soc", "soc"),
    ("temperature_core", "core"),
    ("temperature_l3", "l3"),
)
_V1_TEMPERATURE_FIELDS = (
    ("temperature_edge", "edge"),
    ("temperature_hotspot", "junction"),
    ("temperature_mem", "mem"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedGpuMetrics:
    """The decoded metric table, normalised onto canonical metric names.

    Every field is optional and already sentinel- and plausibility-filtered, so ``None``
    means "this hardware does not provide it" rather than "zero".
    """

    abi_label: str
    format_revision: int
    content_revision: int
    structure_size: int
    gpu_activity_percent: float | None = None
    umc_activity_percent: float | None = None
    mm_activity_percent: float | None = None
    temperatures: tuple[Temperature, ...] = ()
    power_draw_w: float | None = None
    energy_mj: int | None = None
    sclk_mhz: int | None = None
    socclk_mhz: int | None = None
    mclk_mhz: int | None = None
    vclk_mhz: int | None = None
    dclk_mhz: int | None = None
    fclk_mhz: int | None = None
    fan_rpm: int | None = None
    fan_pwm_percent: float | None = None
    throttle: ThrottleInfo | None = None
    pcie: PcieLink | None = None
    voltages_mv: tuple[tuple[str, int], ...] = ()


def _filter_u16(value: int | None) -> int | None:
    """Drop the unsupported-field sentinel from a 16-bit value."""
    return None if value is None or value == SENTINEL_U16 else value


def _filter_u32(value: int | None) -> int | None:
    """Drop the unsupported-field sentinel from a 32-bit value."""
    return None if value is None or value == SENTINEL_U32 else value


def _filter_u64(value: int | None) -> int | None:
    """Drop the unsupported-field sentinel from a 64-bit value."""
    return None if value is None or value == SENTINEL_U64 else value


def _clock(value: int | None) -> int | None:
    """Keep only plausible clock frequencies; zero stays valid because idle is real."""
    if value is None or not 0 <= value <= MAX_CLOCK_MHZ:
        return None
    return value


def _temperature(value: int | None, label: str) -> Temperature | None:
    """Build a temperature only from a plausible Celsius reading."""
    if value is None or not 0 <= value <= MAX_TEMPERATURE_C:
        return None
    return Temperature(label=label, celsius=value, source="gpu_metrics")


def _voltage(value: int | None, name: str) -> tuple[str, int] | None:
    """Reject voltages outside a physically plausible rail range."""
    if value is None or not MIN_VOLTAGE_MV <= value <= MAX_VOLTAGE_MV:
        return None
    return (name, value)


def _percent(value: int | None) -> float | None:
    """Convert a 0-100 activity reading."""
    if value is None or not 0 <= value <= 100:
        return None
    return float(value)


def _power(value: int | None) -> float | None:
    """Convert a whole-watt power reading."""
    if value is None or not 0 <= value <= MAX_POWER_W:
        return None
    return float(value)


class GpuMetricsParser:
    """Decodes the binary ``gpu_metrics`` file.

    The class holds no mutable state, so one instance may be shared freely between
    threads.
    """

    def parse_file(self, path: Path) -> ParsedGpuMetrics | None:
        """Parse a ``gpu_metrics`` file.

        Args:
            path: Path to the sysfs attribute.

        Returns:
            The decoded table, or ``None`` when the file is missing, unreadable,
            truncated below the header, or reports an unrecognised ABI.  The caller then
            falls back to the plain sysfs and hwmon readers.
        """
        blob = read_bytes(path)
        if blob is None:
            return None
        return self.parse(blob)

    def parse(self, blob: bytes) -> ParsedGpuMetrics | None:
        """Parse a metric table from raw bytes.

        Args:
            blob: The raw file contents.

        Returns:
            The decoded table, or ``None`` if the header is implausible or the
            ``(format, content)`` pair is not registered.  An unknown version is never
            guessed at, because misreading an unknown layout would report confident and
            entirely wrong numbers instead of falling back.
        """
        if len(blob) < HEADER_SIZE:
            return None
        structure_size, fmt_revision, content_revision = HEADER.unpack_from(blob)
        if not HEADER_SIZE < structure_size <= len(blob):
            return None
        abi = ABI_REGISTRY.get((fmt_revision, content_revision))
        if abi is None:
            return None

        if abi.body.size <= len(blob):
            values = abi.body.unpack_from(blob)
            present = len(blob)
        else:
            # The table is shorter than the layout we know about.  Zero-fill so struct
            # still yields one value per field, but remember how many bytes were really
            # there: anything past that must surface as unavailable, not as a measured 0.
            present = structure_size
            padded = blob + bytes(abi.body.size - len(blob))
            values = abi.body.unpack_from(padded)
        return self._to_model(values, abi, structure_size, present)

    def _to_model(
        self,
        values: tuple[int, ...],
        abi: MetricsAbi,
        structure_size: int,
        present: int,
    ) -> ParsedGpuMetrics:
        """Map a raw unpacked tuple onto the normalised model.

        Args:
            values: One unpacked value per field in ``abi.fields``.
            abi: The layout used to decode.
            structure_size: The driver-declared size, for reporting.
            present: How many bytes of the table were actually present.  Fields starting at
                or after this offset are treated as absent, so a truncated table reports
                ``None`` rather than a fabricated zero.
        """
        raw = {
            spec.name: int(value)
            for spec, value in zip(abi.fields, values, strict=False)
            if abi.field_map[spec.name] < present
        }

        def u16(name: str) -> int | None:
            value = raw.get(name)
            return _filter_u16(value) if value is not None else None

        def u32(name: str) -> int | None:
            value = raw.get(name)
            return _filter_u32(value) if value is not None else None

        def u64(name: str) -> int | None:
            value = raw.get(name)
            return _filter_u64(value) if value is not None else None

        temperatures: dict[str, Temperature] = {}
        for source_name, label in (*_V1_TEMPERATURE_FIELDS, *_V2_TEMPERATURE_FIELDS):
            if source_name not in raw:
                continue
            reading = _temperature(u16(source_name), label)
            if reading is not None:
                temperatures.setdefault(label, reading)
        ordered = tuple(temperatures[name] for name in _TEMP_ORDER if name in temperatures)

        width = u16("pcie_link_width")
        speed = u16("pcie_link_speed")
        pcie = PcieLink(width=width, speed=speed) if width and speed else None

        throttle_raw = u32("throttle_status")
        throttle = (
            ThrottleInfo(raw=throttle_raw, active=decode_throttle(throttle_raw))
            if throttle_raw is not None
            else None
        )

        voltages = tuple(
            item
            for item in (
                _voltage(u16("voltage_soc"), "soc"),
                _voltage(u16("voltage_gfx"), "gfx"),
                _voltage(u16("voltage_mem"), "mem"),
            )
            if item is not None
        )

        return ParsedGpuMetrics(
            abi_label=abi.label,
            format_revision=abi.fmt_revision,
            content_revision=abi.content_revision,
            structure_size=structure_size,
            gpu_activity_percent=_percent(u16("average_gfx_activity")),
            umc_activity_percent=_percent(u16("average_umc_activity")),
            mm_activity_percent=_percent(u16("average_mm_activity")),
            temperatures=ordered,
            power_draw_w=_power(u16("average_socket_power")),
            energy_mj=u64("energy_accumulator"),
            sclk_mhz=_clock(u16("current_gfxclk")),
            socclk_mhz=_clock(u16("current_socclk")),
            mclk_mhz=_clock(u16("current_uclk")),
            vclk_mhz=_clock(u16("current_vclk0")),
            dclk_mhz=_clock(u16("current_dclk0")),
            fclk_mhz=_clock(u16("current_fclk")),
            fan_rpm=u16("current_fan_speed"),
            fan_pwm_percent=_percent(u16("fan_pwm")),
            throttle=throttle,
            pcie=pcie,
            voltages_mv=voltages,
        )


def unrecognised_header(blob: bytes) -> tuple[int, int, int] | None:
    """Return ``(structure_size, format, content)`` for an unparsed metric table.

    Used by ``gputop --dump-metrics`` so that a user on an unreleased kernel can report
    a single line that makes the next release support their card.
    """
    if len(blob) < HEADER_SIZE:
        return None
    return HEADER.unpack_from(blob)
