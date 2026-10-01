"""SCPP power profile and overdrive table values.

These live in ``model`` rather than in ``readers`` because they are data, not parsing: the
readers turn sysfs text into these and the UI renders them, and neither should have to know
which layer the other is in.

Both tables are reported by the amdgpu driver and neither is universally present, so a
missing table is an ordinary answer rather than an error.  ``present`` distinguishes "the
file was absent" from "the file was empty", because the first is a fact about the hardware
and the second is a fact about a driver bug.

Nothing here performs I/O, and nothing in this package ever writes to sysfs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class OdvUnit(StrEnum):
    """What an overdrive entry is measured in."""

    #: Legacy tables: a bare integer in millivolts.
    MILLIVOLT = "mV"
    #: RDNA2 and later: a clock ceiling in megahertz.
    MEGAHERTZ = "MHz"


def _fmt(value: int | None, unit: str) -> str:
    """Format an optional overdrive value, rendering ``None`` as ``N/A``."""
    return "N/A" if value is None else f"{value} {unit}"


@dataclass(frozen=True, slots=True)
class PowerProfileEntry:
    """One selectable SCPP profile."""

    #: The profile's numeric id, which is what the kernel accepts on a write.
    index: int
    name: str
    active: bool = False


@dataclass(frozen=True, slots=True)
class PowerProfileTable:
    """A parsed ``pp_power_profile_mode``.

    The full table is kept rather than just the active name because the active name answers
    "what is it now" and not "what could it be", and the second question is the one that
    makes the panel worth a panel.

    gputop never writes to this file.  The ids are carried anyway because a panel that lists
    ``3  3D_FULL_SCREEN`` next to a ``0  BOOTUP_DEFAULT*`` row is showing what the driver
    considers the alternatives, and that is useful without being actionable from here.
    """

    profiles: tuple[PowerProfileEntry, ...] = ()
    #: True when the file was read and contained at least one parseable row.
    present: bool = False

    @property
    def active(self) -> PowerProfileEntry | None:
        """The starred profile, or the first one when the driver marks none."""
        if not self.profiles:
            return None
        for profile in self.profiles:
            if profile.active:
                return profile
        return self.profiles[0]

    @property
    def active_name(self) -> str | None:
        """The active profile's name, or ``None`` when the table is absent."""
        entry = self.active
        return None if entry is None else entry.name

    @property
    def active_index(self) -> int | None:
        """The active profile's id, or ``None`` when the table is absent."""
        entry = self.active
        return None if entry is None else entry.index

    def names(self) -> tuple[str, ...]:
        """Every profile name, in table order."""
        return tuple(profile.name for profile in self.profiles)

    def describe(self) -> str:
        """A one-line summary, e.g. ``3D_FULL_SCREEN (2 of 7)``."""
        entry = self.active
        if entry is None:
            return "N/A"
        return f"{entry.name} ({entry.index} of {len(self.profiles)})"


@dataclass(frozen=True, slots=True)
class OdvEntry:
    """One overdrive table entry: a value for one DPM state."""

    state: int
    value: int
    unit: OdvUnit


@dataclass(frozen=True, slots=True)
class OdvDomain:
    """One clock domain's overdrive table.

    Attributes:
        name: ``sclk``, ``mclk`` or ``socclk``.
        entries: Per-state values, in the order the driver listed them.
        minimum: The domain's floor, from ``OD_RANGE``.
        maximum: The domain's ceiling, from ``OD_RANGE``.
        unit: What ``minimum`` and ``maximum`` are measured in.
    """

    name: str
    entries: tuple[OdvEntry, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    unit: OdvUnit = OdvUnit.MEGAHERTZ

    @property
    def label(self) -> str:
        """The domain's display name."""
        return _DOMAIN_LABELS.get(self.name, self.name.upper())

    @property
    def ceiling(self) -> int | None:
        """The highest value listed in the table, which is the overdriven ceiling."""
        if not self.entries:
            return None
        return max(entry.value for entry in self.entries)

    @property
    def floor(self) -> int | None:
        """The lowest value listed in the table."""
        if not self.entries:
            return None
        return min(entry.value for entry in self.entries)

    @property
    def headroom(self) -> int | None:
        """How much is left between the current ceiling and the domain's maximum."""
        top, maximum = self.ceiling, self.maximum
        if top is None or maximum is None:
            return None
        return maximum - top

    @property
    def percent(self) -> float | None:
        """Where the ceiling sits between the domain's floor and maximum.

        The bar is a ratio of *this driver's* stated range rather than of a hard-coded
        maximum, so it stays correct on a part whose ceiling is not the one gputop was
        written against.  It is the ceiling that is drawn, not a live measurement: the
        driver publishes what the card is *set* to allow, which is what a user checking an
        overdrive setting wants to see.
        """
        low, high, top = self.minimum, self.maximum, self.ceiling
        if low is None or high is None or top is None or high <= low:
            return None
        return min(100.0, max(0.0, (top - low) / (high - low) * 100.0))

    def describe(self) -> str:
        """A compact one-line summary such as ``2104/2600 MHz``."""
        unit = self.unit.value
        low, high, top = self.minimum, self.maximum, self.ceiling
        if top is None:
            return f"{_fmt(low, unit)} - {_fmt(high, unit)}"
        if low is None or high is None:
            return f"{top} {unit}"
        return f"{top}/{high} {unit}"


#: Display names for the overdrive domains.  A domain gputop has not heard of renders as its
#: own uppercased key rather than being hidden.
_DOMAIN_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "sclk": "Shader clock",
        "mclk": "Memory clock",
        "socclk": "SoC clock",
    }
)


@dataclass(frozen=True, slots=True)
class OdcVoltage:
    """A parsed ``pp_od_clk_voltage``.

    Despite the attribute's name this is a table of overdrive ceilings, in whatever unit the
    card uses: the amdgpu driver repurposed a voltage table into an overdrive *clock* table
    on RDNA2 and later.  The unit is therefore carried per entry rather than assumed, which
    is what keeps a modern card from being reported in millivolts.
    """

    domains: tuple[OdvDomain, ...] = ()
    #: ``OD_VDDGFX_OFFSET``: the signed graphics-rail voltage offset, when present.
    vddgfx_offset_mv: int | None = None
    present: bool = False

    def domain(self, name: str) -> OdvDomain | None:
        """Look up one domain by key, e.g. ``sclk``."""
        for entry in self.domains:
            if entry.name == name:
                return entry
        return None

    def describe(self) -> str:
        """A one-line summary of every domain."""
        if not self.domains:
            return "N/A"
        return " · ".join(f"{d.label} {d.describe()}" for d in self.domains)


__all__ = [
    "OdcVoltage",
    "OdvDomain",
    "OdvEntry",
    "OdvUnit",
    "PowerProfileEntry",
    "PowerProfileTable",
]
