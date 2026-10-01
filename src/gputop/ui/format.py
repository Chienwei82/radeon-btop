"""Formatting helpers shared by the widgets.

Keeping every "how is a number shown" decision in one place is what makes the ``N/A``
policy enforceable: a missing reading must never render as ``0``.
"""

NA = "N/A"


def fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
    """Format an optional number, rendering ``None`` as ``N/A``.

    This is the single choke point for the "unavailable is not zero" rule.
    """
    if value is None:
        return NA
    if digits == 0:
        return f"{int(value):,}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def fmt_temp(value: float | None) -> str:
    """Format a Celsius reading with a degree sign."""
    return NA if value is None else f"{value:.0f}°C"


def fmt_duration(seconds: float | None) -> str:
    """Format an elapsed duration compactly, e.g. ``1h 02m`` or ``9s``.

    Days are shown once there are any.  The docstring used to say they were omitted -- on
    the grounds that an uptime in days is not useful at a glance -- while the code below
    the comment had been printing ``3d 07h`` all along, and a reader trusting the comment
    would have deleted the branch that was doing the useful thing.
    """
    if seconds is None:
        return NA
    total = int(max(0.0, seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_bytes(value: int | None) -> str:
    """Format a byte count with binary units, matching ``free``/``btop`` convention."""
    if value is None:
        return NA
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        # The last unit is the escape hatch rather than an extra case after the loop: a
        # petabyte-scale value must still land somewhere instead of falling out of the
        # bottom, and the test after the loop existed to catch that without doing it.
        if abs(size) < 1024.0 or unit == "TiB":
            precision = 0 if unit == "B" else 1
            return f"{size:,.{precision}f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable: the loop returns on its final unit")  # pragma: no cover


def fmt_percent(value: float | None, digits: int = 0) -> str:
    """Format a 0-100 percentage."""
    return NA if value is None else f"{value:,.{digits}f}%"


def bar(fraction: float | None, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    """Render a proportional bar.

    Args:
        fraction: 0-100, or ``None`` when the metric is unavailable.
        width: Total bar width in characters.
        filled: Character for the used portion.
        empty: Character for the remaining portion.

    Returns:
        A bar of exactly ``width`` characters, or ``N/A`` when the fraction is unknown.
    """
    if fraction is None:
        return NA
    clamped = min(100.0, max(0.0, fraction))
    count = round(clamped / 100.0 * width)
    return filled * count + empty * (width - count)
