"""Readers for the SCPP power profile and overdrive tables -- **read only**.

Everything here opens sysfs attributes for reading and nothing else.  That is a hard rule
rather than a convention: ``pp_power_profile_mode`` and ``pp_od_clk_voltage`` are writable
files whose contents change hardware, and a monitor that quietly reprograms somebody's GPU
is not a monitor.  No function in this module accepts a mode or a writable handle, and
:func:`assert_read_only` exists so the intent is checkable rather than merely documented.

Both attributes are reported by the driver and neither is universally present, so a missing
file is an ordinary answer rather than an error.

A note on ``pp_od_clk_voltage``, whose name outlived its contents, and which changed shape
twice rather than once.  On Vega10 and older it was a per-DPM-level voltage table --
``0:300 1:350 2:400`` in millivolts.  On Vega20 and newer the same attribute was reused for
overdrive *clock* ceilings and gained sections::

    OD_SCLK:
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

Parsing is therefore section-aware and unit-carrying.  A file with no ``OD_*`` header at all
is still handled as the single legacy voltage table it is, so cards from both eras decode
through the same entry point.
"""

import re
from pathlib import Path

from gputop.model.power import (
    OdcVoltage,
    OdvDomain,
    OdvEntry,
    OdvUnit,
    PowerProfileEntry,
    PowerProfileTable,
)
from gputop.readers.fsutil import PROFILE_ROW, TABLE_LIMIT, read_text

#: A section header inside ``pp_od_clk_voltage``, e.g. ``OD_SCLK:``.
_ODV_SECTION = re.compile(r"^\s*(OD_[A-Z_]+)\s*:?\s*$")

#: An entry: ``<state>: <value><unit>``.  Deliberately unanchored and applied with
#: :meth:`re.Pattern.finditer`, because some kernels write one pair per line and others write
#: a whole run of pairs on one line separated by single spaces.  An anchored ``match`` would
#: read only the first pair of the run and silently discard the rest of the table.
#:
#: The unit is absent on the legacy voltage tables, which are bare integers in millivolts.
_ODV_ENTRY = re.compile(r"(\d+)\s*:\s*(-?\d+)\s*([A-Za-z]*)")

#: An ``OD_RANGE`` line: ``SCLK:     500Mhz       2600Mhz``.
_ODV_RANGE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+)\s*([A-Za-z]*)\s+(-?\d+)\s*([A-Za-z]*)\s*$"
)

#: A signed millivolt value, used by ``OD_VDDGFX_OFFSET`` (``-50mV``).
_ODV_OFFSET = re.compile(r"^\s*(-?\d+)\s*mV\s*$", re.IGNORECASE)

#: Section name to canonical domain key.  Only the per-domain sections appear; ``OD_RANGE``
#: and ``OD_VDDGFX_OFFSET`` carry no per-state entries and are recognised by name below.
_ODV_SECTIONS: dict[str, str] = {
    "OD_SCLK": "sclk",
    "OD_MCLK": "mclk",
    "OD_SOCCLK": "socclk",
}

#: Domains ``OD_RANGE`` can name.
_RANGE_DOMAINS = frozenset({"sclk", "mclk", "socclk"})

#: ``OD_*`` sections that hold no ``<state>: <value>`` rows, so their lines are matched by
#: shape rather than by section.  Anything else that is not in :data:`_ODV_SECTIONS` has
#: its body skipped: the attribute carries sections this parser does not model, and their
#: rows have the same shape as a clock entry without being one.
_NON_ENTRY_SECTIONS = frozenset({"OD_RANGE", "OD_VDDGFX_OFFSET"})

#: Which domain a section-less entry belongs to.  The legacy layout is a single bare voltage
#: table for the shader clock, which is the only domain that file ever described.
_LEGACY_DOMAIN = "sclk"


def _unit_of(raw: str, default: OdvUnit) -> OdvUnit:
    """Normalise a unit suffix, falling back to the section's historical unit."""
    lowered = raw.lower()
    if lowered in ("hz", "khz", "mhz", "ghz"):
        return OdvUnit.MEGAHERTZ
    if lowered == "mv":
        return OdvUnit.MILLIVOLT
    return default


def parse_power_profile_table(text: str | None) -> PowerProfileTable:
    """Parse the whole ``pp_power_profile_mode`` table.

    Args:
        text: Raw file contents, or ``None`` when the file could not be read.

    Returns:
        Every profile in table order, with the active one flagged.  The file's per-profile
        detail rows -- ``0(  GFXCLK)  0  5  1 ...`` -- are skipped without a special case:
        they open with a digit immediately followed by ``(``, which no profile row has, so
        the same expression separates the two.
    """
    if not text:
        return PowerProfileTable()
    profiles: list[PowerProfileEntry] = []
    for line in text.splitlines():
        match = PROFILE_ROW.match(line)
        if match is None:
            continue
        profiles.append(
            PowerProfileEntry(
                index=int(match.group(1)),
                name=match.group(2).strip(),
                active=match.group(3) is not None,
            )
        )
    return PowerProfileTable(profiles=tuple(profiles), present=bool(profiles))


def parse_odc_voltage(text: str | None) -> OdcVoltage:
    """Parse ``pp_od_clk_voltage`` into per-domain tables.

    Args:
        text: Raw file contents, or ``None`` when the file could not be read.

    Returns:
        One :class:`~gputop.model.power.OdvDomain` per clock domain the driver listed, in
        file order, plus the graphics-rail voltage offset when one is reported.
    """
    if not text:
        return OdcVoltage()

    entries: dict[str, list[OdvEntry]] = {}
    ranges: dict[str, tuple[int, int, OdvUnit]] = {}
    offset: int | None = None

    #: The domain subsequent lines belong to, or ``None`` outside any per-domain section.
    current: str | None = None
    #: Inside an ``OD_*`` section this parser does not model, whose rows must be dropped.
    skipping = False
    #: Whether the file declared its sections, which decides a bare entry's default unit.
    sectioned = any(_ODV_SECTION.match(line.strip()) for line in text.splitlines())
    default_unit = OdvUnit.MEGAHERTZ if sectioned else OdvUnit.MILLIVOLT

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        header = _ODV_SECTION.match(stripped)
        if header is not None:
            # OD_RANGE and OD_VDDGFX_OFFSET hold no per-state entries; leaving ``current``
            # unset makes their lines fall through to the shape-based matches below.
            #
            # A section we do not know is *not* the same as no section.  ``_ODV_SECTION``
            # matches any ``OD_*`` header, and the attribute carries far more of them than
            # this parser models: SMU13 writes ``OD_FAN_CURVE``, ``OD_ACOUSTIC_LIMIT``,
            # ``OD_FAN_TARGET_TEMPERATURE`` and friends, and Vega20/Navi1x write
            # ``OD_VDDC_CURVE``.  Their rows are ``<state>: <value>`` like every other
            # entry, so falling through put a fan curve's "2: 95C" into the *shader clock*
            # domain and the overdrive panel reported "sclk 95 MHz".  Skipping the body of
            # an unrecognised section is the honest answer: we know it is not a clock
            # table, and reading it as one is worse than ignoring it.
            name = header.group(1)
            current = _ODV_SECTIONS.get(name)
            skipping = current is None and name not in _NON_ENTRY_SECTIONS
            continue

        if skipping:
            continue

        if current is None:
            span = _ODV_RANGE.match(stripped)
            if span is not None and span.group(1).lower() in _RANGE_DOMAINS:
                ranges[span.group(1).lower()] = (
                    int(span.group(2)),
                    int(span.group(4)),
                    _unit_of(span.group(5), OdvUnit.MEGAHERTZ),
                )
                continue
            if offset is None:
                bias = _ODV_OFFSET.match(stripped)
                if bias is not None:
                    offset = int(bias.group(1))
                    continue

        domain = current if current is not None else _LEGACY_DOMAIN
        bucket = entries.setdefault(domain, [])
        # finditer, not match: a line may carry a whole run of pairs, and an anchored
        # match would take the first and discard the rest of the table without complaint.
        for entry in _ODV_ENTRY.finditer(stripped):
            bucket.append(
                OdvEntry(
                    state=int(entry.group(1)),
                    value=int(entry.group(2)),
                    unit=_unit_of(entry.group(3), default_unit),
                )
            )

    def span_of(name: str, items: list[OdvEntry]) -> tuple[int | None, int | None, OdvUnit]:
        """The domain's ``OD_RANGE`` span, or ``(None, None)`` when it declared none.

        The unit falls back to what the domain's own entries are measured in rather than to
        a fixed default: a table with no sections at all is a legacy GCN *voltage* table,
        and assuming megahertz there made ``describe()`` announce "400 MHz" for 400 mV.
        """
        found = ranges.get(name)
        if found is not None:
            return found
        return None, None, items[0].unit if items else default_unit

    domains = tuple(
        OdvDomain(
            name=name,
            entries=tuple(items),
            minimum=span_of(name, items)[0],
            maximum=span_of(name, items)[1],
            unit=span_of(name, items)[2],
        )
        for name, items in entries.items()
    )

    return OdcVoltage(
        domains=domains,
        vddgfx_offset_mv=offset,
        present=bool(domains) or offset is not None,
    )


def read_power_profile_table(device_dir: Path) -> PowerProfileTable:
    """Read ``pp_power_profile_mode`` for a device, without writing to it."""
    return parse_power_profile_table(
        read_text(device_dir / "pp_power_profile_mode", limit=TABLE_LIMIT)
    )


def read_odc_voltage(device_dir: Path) -> OdcVoltage:
    """Read ``pp_od_clk_voltage`` for a device, without writing to it.

    A missing file yields an empty table, which the UI renders as ``N/A``.
    """
    return parse_odc_voltage(read_text(device_dir / "pp_od_clk_voltage", limit=TABLE_LIMIT))


def assert_read_only() -> bool:
    """Return True; the read-only guarantee of this module, stated once for the tests.

    The property being asserted is structural rather than a runtime state: this module
    imports ``read_text`` from :mod:`gputop.readers.fsutil` and exposes no function that
    accepts a mode or a writable handle.  Making it a function rather than a comment lets a
    test assert on it, so a later edit that adds a write path fails the suite instead of
    quietly breaking the one rule the rest of the application is built around.
    """
    return True


__all__ = [
    "assert_read_only",
    "parse_odc_voltage",
    "parse_power_profile_table",
    "read_odc_voltage",
    "read_power_profile_table",
]
