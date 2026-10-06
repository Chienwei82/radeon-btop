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


#: PCIe generation against its signalling rate, in the kernel's unit of 0.1 GT/s.
#: ``kgd_pp_interface.h`` documents ``pcie_link_speed`` as "in 0.1 GT/s" in every
#: ``gpu_metrics`` revision from v1_0 on, so 80 is 8.0 GT/s -- Gen3 -- and not an
#: encoding of 5.  Reading the generation off the rate is what keeps a Gen3 card
#: from being reported as Gen5; reading it off an assumed multiplier cannot.
_PCIE_GENERATIONS: tuple[tuple[int, int], ...] = (
    (25, 1),  # 2.5 GT/s
    (50, 2),  # 5.0 GT/s
    (80, 3),  # 8.0 GT/s
    (160, 4),  # 16.0 GT/s
    (320, 5),  # 32.0 GT/s
    (640, 6),  # 64.0 GT/s
)

#: How far off a standard signalling rate the kernel may report and still be called a
#: generation, as a fraction of that rate.  Generous enough for a driver that rounds
#: 8.0 GT/s up, tight enough that an unrecognised rate is reported as the speed it is
#: rather than snapped to the nearest generation.
_PCIE_RATE_TOLERANCE = 0.1


@dataclass(frozen=True, slots=True)
class PcieLink:
    """Negotiated PCIe link width and speed.

    ``speed`` is ``gpu_metrics.pcie_link_speed`` verbatim: tenths of a gigatransfer
    per second, the unit the kernel itself uses.  Every source that builds a
    ``PcieLink`` -- the binary metric table and the ``current_link_*`` sysfs files --
    converts into it, so :attr:`generation` means the same thing for all of them.
    """

    width: int
    speed: int

    @property
    def gt_per_second(self) -> float:
        """Signalling rate of the negotiated link, in GT/s."""
        return self.speed / 10

    @property
    def generation(self) -> int:
        """PCIe generation, or ``0`` when the rate is not a standard one."""
        if self.speed <= 0:
            return 0
        rate, generation = min(_PCIE_GENERATIONS, key=lambda entry: abs(entry[0] - self.speed))
        if abs(rate - self.speed) > rate * _PCIE_RATE_TOLERANCE:
            return 0
        return generation

    def describe(self) -> str:
        """Human readable form such as ``Gen3 x16``.

        A rate that matches no generation is shown as the speed it is, never rounded
        up to the closest one: a wrong generation is a claim about the hardware, and
        this figure exists to tell you what the link is doing right now.
        """
        gen = self.generation
        if gen <= 0:
            return f"{self.gt_per_second:g} GT/s x{self.width}"
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
