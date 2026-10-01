"""Per-device metric snapshots.

Every field is optional.  ``None`` means "not available on this hardware / kernel /
permission level" and must be rendered as ``N/A`` by the UI -- never as ``0``.
"""

from dataclasses import dataclass, field

from gputop.model.aliases import Bytes, Celsius, Mhz, Nanoseconds, Percent, Rpm, Watts
from gputop.model.blocks import GpuBlocks
from gputop.model.device import AmdgpuDevice
from gputop.model.power import OdcVoltage, PowerProfileTable


@dataclass(frozen=True, slots=True)
class MemoryPool:
    """A used/total memory pool such as VRAM or GTT."""

    used: Bytes | None = None
    total: Bytes | None = None
    source: str = "none"

    @property
    def percent(self) -> Percent | None:
        """Fraction of the pool in use, or ``None`` when the total is unknown/zero."""
        if self.total is None or self.total <= 0 or self.used is None:
            return None
        return min(100.0, max(0.0, self.used / self.total * 100.0))


@dataclass(frozen=True, slots=True)
class Clock:
    """A clock frequency with its current value and ceiling."""

    current: Mhz | None = None
    maximum: Mhz | None = None
    source: str = "none"

    @property
    def percent(self) -> Percent | None:
        """Utilisation of the clock range, useful for drawing a bar."""
        if self.maximum is None or self.maximum <= 0 or self.current is None:
            return None
        return min(100.0, max(0.0, self.current / self.maximum * 100.0))


@dataclass(frozen=True, slots=True)
class Power:
    """Board power draw against the enforced cap."""

    draw_w: Watts | None = None
    cap_w: Watts | None = None
    energy_mj: int | None = None
    source: str = "none"

    @property
    def percent(self) -> Percent | None:
        """Power draw as a percentage of the cap."""
        if self.cap_w is None or self.cap_w <= 0 or self.draw_w is None:
            return None
        return min(100.0, max(0.0, self.draw_w / self.cap_w * 100.0))


@dataclass(frozen=True, slots=True)
class Fan:
    """Fan state; APUs and passively cooled cards report ``None`` throughout."""

    rpm: Rpm | None = None
    max_rpm: Rpm | None = None
    pwm: Percent | None = None

    @property
    def percent(self) -> Percent | None:
        """Fan duty as a percentage, preferring the measured RPM over the PWM value."""
        if self.rpm is not None and self.max_rpm and self.max_rpm > 0:
            return min(100.0, max(0.0, self.rpm / self.max_rpm * 100.0))
        return self.pwm

    @property
    def stopped(self) -> bool:
        """True when the fan is present but not spinning."""
        return self.rpm == 0 or self.pwm == 0


@dataclass(frozen=True, slots=True)
class Temperature:
    """A single temperature reading."""

    label: str
    celsius: Celsius | None = None
    source: str = "none"


@dataclass(frozen=True, slots=True)
class ThrottleInfo:
    """Decoded throttle status bits."""

    raw: int
    active: frozenset[str]

    @property
    def is_throttling(self) -> bool:
        """Whether any throttle reason is currently asserted."""
        return bool(self.active)


@dataclass(frozen=True, slots=True)
class PcieLink:
    """Negotiated PCIe link width and speed."""

    width: int
    speed: int

    @property
    def generation(self) -> int:
        """PCIe generation, derived from the encoded speed value."""
        return self.speed // 16 if self.speed >= 16 else 0

    def describe(self) -> str:
        """Human readable form such as ``Gen5 x16``."""
        gen = self.generation
        if gen <= 0:
            return f"x{self.width}"
        return f"Gen{gen} x{self.width}"


@dataclass(frozen=True, slots=True, kw_only=True)
class AmdgpuMetrics:
    """One immutable sample of a single GPU's state."""

    device: AmdgpuDevice
    sequence: int
    timestamp_ns: Nanoseconds
    interval_s: float
    gpu_busy_percent: Percent | None = None
    mem_busy_percent: Percent | None = None
    vram: MemoryPool = MemoryPool()
    gtt: MemoryPool = MemoryPool()
    sclk: Clock = Clock()
    mclk: Clock = Clock()
    socclk: Clock = Clock()
    voltages_mv: tuple[tuple[str, int], ...] = ()
    temperatures: tuple[Temperature, ...] = ()
    power: Power = Power()
    fan: Fan = Fan()
    performance_level: str | None = None
    power_profile: str | None = None
    throttle: ThrottleInfo | None = None
    pcie: PcieLink | None = None
    metrics_abi: str | None = None
    unavailable: frozenset[str] = frozenset()
    #: Per-block utilisation, or ``None`` when radeontop is not producing it.  ``None`` is
    #: the common case and is not an error; see :mod:`gputop.readers.radeontop`.
    blocks: GpuBlocks | None = None
    #: The overdrive table, read-only.  Absent on hardware the driver does not publish it for.
    odc: OdcVoltage = field(default_factory=OdcVoltage)
    #: Every selectable SCPP power profile, read-only.  ``power_profile`` above is the
    #: active name; this is the whole table it was chosen from.
    profiles: PowerProfileTable = field(default_factory=PowerProfileTable)
