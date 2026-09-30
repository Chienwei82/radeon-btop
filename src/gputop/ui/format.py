"""Formatting helpers shared by the widgets.

Keeping every "how is a number shown" decision in one place is what makes the ``N/A``
policy enforceable: a missing reading must never render as ``0``.
"""

from gputop.model.metrics import AmdgpuMetrics, MemoryPool, Temperature

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

    The day field is omitted: this renders a monitor's uptime, and an uptime measured in
    days is not useful at a glance next to everything else on the line.
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
        if abs(size) < 1024.0 or unit == "TiB":
            precision = 0 if unit == "B" else 1
            return f"{size:,.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TiB"


def fmt_percent(value: float | None, digits: int = 0) -> str:
    """Format a 0-100 percentage."""
    return NA if value is None else f"{value:,.{digits}f}%"


def fmt_pool(pool: MemoryPool) -> str:
    """Format a memory pool as ``used / total (percent)``."""
    percent = pool.percent
    suffix = f" ({percent:.0f}%)" if percent is not None else ""
    return f"{fmt_bytes(pool.used)} / {fmt_bytes(pool.total)}{suffix}"


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


def throttle_text(metrics: AmdgpuMetrics) -> str:
    """Describe throttle state, naming the active reasons."""
    throttle = metrics.throttle
    if throttle is None:
        return NA
    if not throttle.is_throttling:
        return "none"
    return ", ".join(sorted(throttle.active))


def temperatures_text(metrics: AmdgpuMetrics) -> str:
    """Render every sensor on one line, e.g. ``edge 49°C  junction 56°C``."""
    if not metrics.temperatures:
        return NA
    return "  ".join(f"{t.label} {fmt_temp(t.celsius)}" for t in _ordered(metrics.temperatures))


def _ordered(temps: tuple[Temperature, ...]) -> tuple[Temperature, ...]:
    """Keep the sampler's ordering; present as a seam for a future preference."""
    return temps
