"""hwmon sensor reader.

Temperature semantics differ between vendors and generations, so ``tempN_label`` is
consulted and any unrecognised label is surfaced verbatim rather than dropped.
"""

import re
from collections.abc import Iterable
from pathlib import Path

from gputop.readers.fsutil import (
    parse_millidegrees,
    parse_power_microwatts,
    read_int,
    read_text,
)

TEMP_FILE_PATTERN = re.compile(r"^temp(\d+)_input$")

#: Canonical names for the labels AMD uses, keyed by the substring that identifies them.
#:
#: Sorted longest-first at import rather than written in that order, because the order is
#: load-bearing and hand-maintained order drifts: a future entry added at the end could
#: shadow an existing one, and the comment claiming otherwise would not catch it.  The
#: entries are a substring search, not exact names, because AMD's labels carry prefixes
#: ("temp1: mem", "junction") that vary between boards.
LABEL_MAP: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("junction", "junction"),
            ("hotspot", "junction"),
            ("edge", "edge"),
            ("mem", "mem"),
            ("soc", "soc"),
            ("core", "core"),
            ("l3", "l3"),
            ("vr", "vr"),
        ),
        key=lambda pair: -len(pair[0]),
    )
)

#: Ordering used when presenting the sensor panel.
_TEMP_ORDER = ("edge", "junction", "mem", "soc", "core", "l3")


def canonical_label(label: str | None, index: int) -> str:
    """Map an hwmon label onto a canonical name.

    Args:
        label: The raw ``tempN_label`` contents, or ``None`` when the file is absent.
        index: The sensor number, used to build ``tempN`` as a last resort.

    Returns:
        A canonical name such as ``edge`` or ``junction``; unknown labels are returned
        lower-cased and stripped, and a missing label becomes ``tempN``.
    """
    if not label:
        return f"temp{index}"
    cleaned = label.strip().lower()
    for needle, canonical in LABEL_MAP:
        if needle in cleaned:
            return canonical
    return cleaned


def _sensor_sort_key(name: str) -> tuple[int, int]:
    """Sort known sensors first in a stable display order, unknown ones last."""
    try:
        return (0, _TEMP_ORDER.index(name))
    except ValueError:
        return (1, 0)


def read_temperatures(hwmon_dir: Path | None) -> dict[str, float]:
    """Read every ``tempN_input`` in a hwmon device.

    Args:
        hwmon_dir: The device's hwmon directory, or ``None``.

    Returns:
        Mapping of canonical label to degrees Celsius.  Empty when the directory is
        missing or unreadable -- an absent sensor is normal, not an error.
    """
    if hwmon_dir is None:
        return {}
    try:
        entries = sorted(entry.name for entry in hwmon_dir.iterdir())
    except OSError:
        return {}

    readings: dict[str, float] = {}
    for name in entries:
        match = TEMP_FILE_PATTERN.match(name)
        if match is None:
            continue
        index = int(match.group(1))
        value = parse_millidegrees(read_text(hwmon_dir / name))
        if value is None:
            continue
        label = canonical_label(read_text(hwmon_dir / f"temp{index}_label"), index)
        # Keep the first reading for a label; duplicate labels are not expected.
        readings.setdefault(label, value)
    return readings


def read_power(hwmon_dir: Path | None) -> tuple[float | None, float | None]:
    """Read instantaneous power draw and the enforced cap, in watts.

    ``power1_average`` is preferred because it is a true average over a driver-defined
    window and is far less jittery than ``power1_input``.
    """
    if hwmon_dir is None:
        return None, None
    draw = parse_power_microwatts(read_text(hwmon_dir / "power1_average"))
    if draw is None:
        draw = parse_power_microwatts(read_text(hwmon_dir / "power1_input"))
    cap = parse_power_microwatts(read_text(hwmon_dir / "power1_cap"))
    if cap is None:
        cap = parse_power_microwatts(read_text(hwmon_dir / "power1_cap_max"))
    return draw, cap


def read_fan(hwmon_dir: Path | None) -> tuple[int | None, int | None, float | None]:
    """Read ``(rpm, max_rpm, pwm_percent)`` from a hwmon device."""
    if hwmon_dir is None:
        return None, None, None
    rpm = read_int(hwmon_dir / "fan1_input")
    max_rpm = read_int(hwmon_dir / "fan1_max")
    pwm_raw = read_int(hwmon_dir / "pwm1")
    pwm = (pwm_raw / 255.0 * 100.0) if pwm_raw is not None else None
    return rpm, max_rpm, pwm


def sorted_temperature_names(names: Iterable[str]) -> list[str]:
    """Return sensor names in a stable, human-friendly presentation order.

    Args:
        names: The sensor labels to order.  Only the keys are used, so both a
            ``dict[str, float]`` of hwmon readings and a ``dict[str, Temperature]`` of
            already-merged readings can be passed.
    """
    return sorted(names, key=_sensor_sort_key)
