"""DPM clock tables, performance level and SCPP power profile.

The ``current_sclk``/``current_mclk`` attributes are documented widely but are **absent**
on recent drivers (verified absent on ``amdgpu`` 6.19), so every clock lookup has a
three-level fallback ending in parsing the ``pp_dpm_*`` table and taking the row marked
with ``*`` as the current state.
"""

import re
from pathlib import Path

from gputop.readers.fsutil import parse_first_int, read_text

# The kernel writes the unit as ``Mhz``/``Uhz``/``Ghz`` -- a lowercase ``h`` -- and marks
# the active row with a trailing ``*``.
_DPM_ROW = re.compile(r"^(\S+):\s*(\d+)\s*([MUG]?[Hh]z)\s*(\*)?\s*$")
_PROFILE_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z0-9_]+)\s*(\*)?:?\s*$")


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

    current: int | None = None
    maximum: int | None = None
    for line in text.splitlines():
        match = _DPM_ROW.match(line.strip())
        if match is None:
            continue
        mhz = int(match.group(2))
        marked = match.group(4) is not None
        if marked and current is None:
            current = mhz
        if maximum is None or mhz > maximum:
            maximum = mhz

    if current is None and maximum is not None:
        current = maximum
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
    value = parse_first_int(read_text(device_dir / attribute))
    if value is not None:
        return value, 2
    current, _maximum = parse_dpm_table(read_text(device_dir / dpm_file))
    if current is not None:
        return current, 1
    return None, 0


def read_clock_max(device_dir: Path, attribute: str, dpm_file: str) -> int | None:
    """Read a clock ceiling via ``attribute`` then the highest DPM row."""
    value = parse_first_int(read_text(device_dir / attribute))
    if value is not None:
        return value
    _current, maximum = parse_dpm_table(read_text(device_dir / dpm_file))
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
        match = _PROFILE_ROW.match(line)
        if match is None:
            continue
        name = match.group(2)
        if fallback is None:
            fallback = name
        if match.group(3) is not None:
            return name
    return fallback


def read_power_profile(device_dir: Path) -> str | None:
    """Read the active SCPP power profile name."""
    return parse_power_profile(read_text(device_dir / "pp_power_profile_mode"))


def read_link(device_dir: Path) -> tuple[int | None, int | None]:
    """Read the negotiated ``(width, speed)`` of the PCIe link."""
    width = parse_first_int(read_text(device_dir / "current_link_width"))
    speed = parse_first_int(read_text(device_dir / "current_link_speed"))
    return width, speed
