"""DPM clock tables, performance level and SCPP power profile.

The ``current_sclk`` / ``current_mclk`` / ``current_link_*`` names are widely written down
as sysfs attributes, and the reference machine does not have them -- but enumerating what
``amdgpu`` actually registers shows they never did: they are fields of the driver's
internal ``struct amdgpu_dpm``, not sysfs attributes, and appear in no in-tree release
(checked at v5.4 and at v7.2).  They are probed anyway, because one failing ``open()`` is
cheaper than being wrong about a tree that does publish them, and every clock lookup falls
back to parsing the ``pp_dpm_*`` table and taking the row marked ``*`` as the current state.

Where a clock really comes from is ``gpu_metrics`` (``current_gfxclk`` and friends), the
hwmon ``freq*_input`` pair, or the ``pp_dpm_*`` level table.
"""

import re
from pathlib import Path

from gputop.readers.fsutil import PROFILE_ROW, TABLE_LIMIT, parse_first_int, read_text

# The kernel writes the unit as ``Mhz``/``Uhz``/``Ghz`` -- a lowercase ``h`` -- and marks
# the active row with a trailing ``*``.  ``Khz`` is included for the same reason: a unit
# outside the set makes the whole row unmatched, which silently empties the table rather
# than skipping one line of it.
_DPM_ROW = re.compile(r"^(\S+):\s*(\d+)\s*([KMUG]?[Hh]z)\s*(\*)?\s*$")

#: ``current_link_speed`` is a human-readable string, not a number: the driver writes
#: ``16.0 GT/s PCIe``.  Taking the first integer out of it keeps ``16`` and throws away
#: the decimal, which turns 16.0 GT/s into 1.6 -- so the rate has to be matched whole.
_LINK_SPEED_ROW = re.compile(r"(\d+(?:\.\d+)?)\s*GT/s")


def parse_dpm_table(text: str | None) -> tuple[int | None, int | None]:
    """Parse a ``pp_dpm_*`` table into ``(current_mhz, max_mhz)``.

    The active row is the one flagged with ``*``.  When no row is flagged -- which happens
    on APUs that idle on a single state -- the last row is treated as the ceiling and the
    flagged-or-first row as the current value.

    Args:
        text: Raw file contents.

    Returns:
        ``(None, None)`` when the table is missing or unparseable.
    """
    if not text:
        return None, None

    first: int | None = None
    current: int | None = None
    maximum: int | None = None
    for line in text.splitlines():
        match = _DPM_ROW.match(line.strip())
        if match is None:
            continue
        mhz = int(match.group(2))
        if first is None:
            first = mhz
        if match.group(4) is not None and current is None:
            current = mhz
        if maximum is None or mhz > maximum:
            maximum = mhz

    if current is None:
        # No row was flagged.  The first row is the lowest DPM state, and an unmarked table
        # is what an idling APU publishes; reading the ceiling here reported an idle card
        # as pinned at its maximum clock, and its clock bar at 100%.
        current = first
    return current, maximum


def read_clock(device_dir: Path, attribute: str, dpm_file: str) -> tuple[int | None, int]:
    """Read one clock using the full fallback chain.

    Args:
        device_dir: The device's sysfs directory.
        attribute: The direct attribute name, e.g. ``current_sclk``.
        dpm_file: The DPM table to fall back to, e.g. ``pp_dpm_sclk``.

    Returns:
        ``(current_mhz, confidence)`` where confidence is 2 for a direct attribute read,
        1 for the DPM table and 0 when nothing was readable.
    """
    current, _maximum, confidence = read_clock_pair(device_dir, attribute, dpm_file)
    return current, confidence


def read_clock_pair(
    device_dir: Path,
    attribute: str,
    dpm_file: str,
    max_attribute: str | None = None,
) -> tuple[int | None, int | None, int]:
    """Read a clock's current value and its ceiling, touching each file at most once.

    The obvious implementation -- :func:`read_clock` for the value and :func:`read_clock_max`
    for the ceiling -- reads the DPM table twice, and on recent drivers, where
    ``current_sclk``/``current_sclk_max`` no longer exist, *that* table is the only source
    either of them comes from.  Every clock domain therefore cost four reads where two
    suffice.  The single source wins here because the ceiling cannot come from anywhere
    the current value did not.

    Args:
        device_dir: The device's sysfs directory.
        attribute: The direct current-value attribute, e.g. ``current_sclk``.
        dpm_file: The DPM table to fall back to.
        max_attribute: Direct ceiling attribute, defaulting to ``current_<suffix>_max``.

    Returns:
        ``(current_mhz, maximum_mhz, confidence)`` with the same confidence scale as
        :func:`read_clock`.  ``maximum_mhz`` is ``None`` only when nothing at all was
        readable.
    """
    if max_attribute is None:
        max_attribute = f"{attribute}_max"
    current = parse_first_int(read_text(device_dir / attribute))
    if current is not None:
        maximum = parse_first_int(read_text(device_dir / max_attribute))
        if maximum is None:
            # The ceiling attribute can be absent while the current one is present.  The
            # DPM table still carries it, and without this the bar that draws current
            # against maximum had no maximum to draw against and read as unknown.
            _table_current, maximum = parse_dpm_table(
                read_text(device_dir / dpm_file, limit=TABLE_LIMIT)
            )
        return current, maximum, 2
    table_current, maximum = parse_dpm_table(
        read_text(device_dir / dpm_file, limit=TABLE_LIMIT)
    )
    return table_current, maximum, 1 if table_current is not None else 0


def read_clock_max(device_dir: Path, attribute: str, dpm_file: str) -> int | None:
    """Read a clock ceiling via ``attribute`` then the highest DPM row."""
    value = parse_first_int(read_text(device_dir / attribute))
    if value is not None:
        return value
    _current, maximum = parse_dpm_table(read_text(device_dir / dpm_file, limit=TABLE_LIMIT))
    return maximum


def read_performance_level(device_dir: Path) -> str | None:
    """Read the active DPM performance level, e.g. ``auto`` or ``manual``."""
    for attribute in ("power_dpm_force_performance_level", "pp_dpm_performance_level"):
        value = read_text(device_dir / attribute)
        if value:
            return value
    return None


def parse_power_profile(text: str | None) -> str | None:
    """Extract the active SCPP profile name from ``pp_power_profile_mode``.

    The file is a large table listing every profile, clock type and limit; the active
    profile is the one whose line ends with ``*`` immediately after its name.  When no
    profile is starred the first one is returned.
    """
    if not text:
        return None
    fallback: str | None = None
    for line in text.splitlines():
        match = PROFILE_ROW.match(line)
        if match is None:
            continue
        name = match.group(2).strip()
        if fallback is None:
            fallback = name
        if match.group(3) is not None:
            return name
    return fallback


def read_power_profile(device_dir: Path) -> str | None:
    """Read the active SCPP power profile name."""
    return parse_power_profile(
        read_text(device_dir / "pp_power_profile_mode", limit=TABLE_LIMIT)
    )


def parse_link_speed(text: str | None) -> int | None:
    """Parse a ``current_link_speed`` file into the kernel's 0.1 GT/s unit.

    ``16.0 GT/s PCIe`` becomes ``160``, matching what ``gpu_metrics`` reports for the
    same link, so both sources can drive the same :class:`PcieLink`.

    Args:
        text: Raw file contents.

    Returns:
        The rate in tenths of a GT/s, or ``None`` when the file is missing or does not
        carry a rate.
    """
    if not text:
        return None
    match = _LINK_SPEED_ROW.search(text)
    if match is None:
        return None
    return round(float(match.group(1)) * 10)


def read_link(device_dir: Path) -> tuple[int | None, int | None]:
    """Read the negotiated ``(width, speed)`` of the PCIe link.

    ``speed`` comes back in tenths of a GT/s, the unit :class:`PcieLink` expects.
    """
    width = parse_first_int(read_text(device_dir / "current_link_width"))
    speed = parse_link_speed(read_text(device_dir / "current_link_speed"))
    return width, speed
