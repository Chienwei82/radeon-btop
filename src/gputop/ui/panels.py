"""Text builders for each panel.

These return :class:`rich.text.Text` rather than markup strings because the bars need
per-character colour, which markup cannot express.  Keeping them as pure functions makes
every panel directly testable without mounting a widget.
"""

from collections.abc import Sequence

from rich.style import Style
from rich.text import Text

from gputop.model.metrics import AmdgpuMetrics, Clock, MemoryPool
from gputop.ui.format import (
    NA,
    fmt,
    fmt_bytes,
    fmt_duration,
    fmt_percent,
    fmt_temp,
)
from gputop.ui.theme import Gradient, Theme, threshold_style

#: Bar glyphs, matching the block characters the graphs use so the panels and the
#: braille plots read as one visual system.
BAR_FILLED = "█"
BAR_EMPTY = "░"


def _rule(theme: Theme, width: int) -> Text:
    """A horizontal divider."""
    return Text("─" * max(0, width), Style(color=theme.track))


#: Columns reserved for the left-hand label, wide enough for "temp junction".
LABEL_WIDTH = 14


def fit_bar_width(
    available: int, value_reserve: int, *, label: int = LABEL_WIDTH, minimum: int = 6
) -> int:
    """Choose a bar width that fits the space a row actually has.

    The bar is sized from the *measured* value text rather than a constant, which is what
    stops a long reading from wrapping into the next row.
    """
    return max(minimum, available - label - value_reserve - 1)


def stat_row(
    label: str,
    value: str,
    fraction: float | None,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> Text:
    """Build one ``label  bar  value`` line that fits ``available`` columns.

    Args:
        label: Left-hand label.
        value: The formatted reading.
        fraction: 0-100 for the bar, or ``None`` when the reading is unavailable.
        theme: Palette.
        available: Total columns the row may occupy.
        gradient: Ramp used to colour the bar; built from ``theme`` when omitted.

    An unavailable reading draws an empty bar and prints ``N/A``.  A full bar for a
    missing value would be the single most misleading thing this UI could do.
    """
    ramp = gradient or Gradient(theme)
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(label.ljust(LABEL_WIDTH), Style(color=theme.muted))
    width = fit_bar_width(available, len(value) + 1)
    if fraction is None:
        text.append(BAR_EMPTY * width, Style(color=theme.track))
        text.append(f" {NA}", Style(color=theme.muted))
        return text
    clamped = min(100.0, max(0.0, fraction))
    filled = round(clamped / 100.0 * width)
    text.append(BAR_FILLED * filled, Style(color=ramp.at(clamped)))
    text.append(BAR_EMPTY * (width - filled), Style(color=theme.track))
    text.append(f" {value}", Style(color=theme.text))
    return text


def pool_row(
    label: str,
    pool: MemoryPool,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> Text:
    """Build a memory pool line: bar, used/total and percentage, fitting ``available``."""
    ramp = gradient or Gradient(theme)
    percent = pool.percent
    detail = f"{fmt_bytes(pool.used)} / {fmt_bytes(pool.total)}"
    tail = f"  {percent:.0f}%" if percent is not None else f"  {NA}"

    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(label.ljust(LABEL_WIDTH), Style(color=theme.muted))
    width = fit_bar_width(available, len(detail) + len(tail) + 1)
    if percent is None:
        text.append(BAR_EMPTY * width, Style(color=theme.track))
    else:
        filled = round(min(100.0, percent) / 100.0 * width)
        text.append(BAR_FILLED * filled, Style(color=ramp.at(percent)))
        text.append(BAR_EMPTY * (width - filled), Style(color=theme.track))
    text.append(f" {detail}", Style(color=theme.text))
    text.append(
        tail,
        Style(color=threshold_style(theme, percent) if percent is not None else theme.muted),
    )
    return text


def header_panel(
    metrics: AmdgpuMetrics,
    theme: Theme,
    uptime_s: float,
    interval_s: float,
) -> Text:
    """Build the device header: name, driver, PCI id, uptime and interval."""
    device = metrics.device
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(device.name, Style(color=theme.text, bold=True))
    kind = "iGPU" if device.is_integrated else "dGPU"
    text.append(f"  {kind}", Style(color=theme.accent))
    if device.boot_vga:
        text.append("  boot", Style(color=theme.warn))

    text.append("\n")
    text.append("amdgpu", Style(color=theme.muted))
    text.append("  ·  ", Style(color=theme.track))
    text.append(device.bdf, Style(color=theme.muted))
    text.append("  ·  ", Style(color=theme.track))
    text.append(f"0x{device.device_id:04x}", Style(color=theme.muted))
    if device.render:
        text.append("  ·  ", Style(color=theme.track))
        text.append(device.render, Style(color=theme.muted))

    text.append("\n")
    text.append(f"up {fmt_duration(uptime_s)}".ljust(24), Style(color=theme.muted))
    text.append(f"every {interval_s:g}s", Style(color=theme.muted))
    text.append("  ·  ", Style(color=theme.track))
    text.append(
        metrics.metrics_abi or "sysfs",
        Style(color=theme.ok if metrics.metrics_abi else theme.warn),
    )
    return text


def utilisation_rows(
    metrics: AmdgpuMetrics,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> list[Text]:
    """Build the live utilisation rows for the currently focused device."""
    rows = [
        stat_row(
            "GPU",
            fmt_percent(metrics.gpu_busy_percent),
            metrics.gpu_busy_percent,
            theme,
            available,
            gradient,
        )
    ]
    # On an integrated GPU the working set is shared system memory, so GTT is the pool
    # worth the bar; VRAM is a small fixed carve-out and says little about real usage.
    primary = metrics.gtt if metrics.device.is_integrated else metrics.vram
    primary_label = "GTT" if metrics.device.is_integrated else "VRAM"
    rows.append(pool_row(primary_label, primary, theme, available, gradient))
    if metrics.device.is_integrated:
        # Keep VRAM visible on an APU too, but demoted: it is the smaller pool.
        rows.append(pool_row("VRAM", metrics.vram, theme, available, gradient))
    return rows


def memory_rows(
    metrics: AmdgpuMetrics, theme: Theme, width: int = 24, gradient: Gradient | None = None
) -> list[Text]:
    """Build both memory rows regardless of device kind."""
    return [
        pool_row("VRAM", metrics.vram, theme, width, gradient),
        pool_row("GTT", metrics.gtt, theme, width, gradient),
    ]


def _clock_row(
    label: str, clock: Clock, theme: Theme, available: int, gradient: Gradient | None
) -> Text:
    """Build a clock row; the bar is relative to the clock's own ceiling."""
    value = NA if clock.current is None else f"{fmt(clock.current)} MHz"
    return stat_row(label, value, clock.percent, theme, available, gradient)


def sensors_panel(
    metrics: AmdgpuMetrics,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> list[Text]:
    """Build the sensor rows: clocks, temperatures, power, fan and throttle.

    Every row is bar-or-``N/A``.  Temperatures have no meaningful 0-100 scale, so they
    use a fixed 0-125 C span, which is the point at which every AMD part throttles hard.
    """
    ramp = gradient or Gradient(theme)
    rows: list[Text] = [
        _clock_row("Core", metrics.sclk, theme, available, ramp),
    ]
    if not metrics.device.is_integrated:
        rows.append(_clock_row("Memory", metrics.mclk, theme, available, ramp))
    rows.append(_clock_row("SoC", metrics.socclk, theme, available, ramp))

    for temp in metrics.temperatures:
        fraction = None if temp.celsius is None else temp.celsius / 125.0 * 100.0
        rows.append(
            stat_row(
                f"temp {temp.label}",
                fmt_temp(temp.celsius),
                fraction,
                theme,
                available,
                ramp,
            )
        )

    power_percent = metrics.power.percent
    rows.append(
        stat_row(
            "Power",
            f"{fmt(metrics.power.draw_w, digits=1)} W",
            power_percent,
            theme,
            available,
            ramp,
        )
    )
    cap = metrics.power.cap_w
    rows.append(
        Text(
            f"{' ' * LABEL_WIDTH}cap {fmt(cap)} W" if cap is not None else f"        cap {NA}",
            Style(color=theme.muted),
        )
    )

    if metrics.device.supports_fan:
        rows.append(
            stat_row(
                "Fan",
                f"{fmt(metrics.fan.rpm)} rpm",
                metrics.fan.percent,
                theme,
                available,
                ramp,
            )
        )

    if metrics.voltages_mv:
        volts = "  ".join(f"{name} {value}mV" for name, value in metrics.voltages_mv)
        rows.append(Text(f"{' ' * LABEL_WIDTH}{volts}", Style(color=theme.muted)))

    if metrics.pcie is not None:
        rows.append(
            Text(f"{' ' * LABEL_WIDTH}PCIe {metrics.pcie.describe()}", Style(color=theme.muted))
        )

    profile = metrics.power_profile or metrics.performance_level
    rows.append(
        Text(
            f"{' ' * LABEL_WIDTH}profile {profile}"
            if profile
            else f"{' ' * LABEL_WIDTH}profile {NA}",
            Style(color=theme.muted),
        )
    )

    throttle = metrics.throttle
    if throttle is None:
        rows.append(Text(f"{' ' * LABEL_WIDTH}throttle {NA}", Style(color=theme.muted)))
    elif throttle.is_throttling:
        rows.append(
            Text(
                f"{' ' * LABEL_WIDTH}throttle {', '.join(sorted(throttle.active))}",
                Style(color=theme.alert),
            )
        )
    else:
        rows.append(Text(f"{' ' * LABEL_WIDTH}throttle none", Style(color=theme.ok)))
    return rows


def footer_panel(
    theme: Theme,
    *,
    sequence: int,
    metrics_source: str | None,
    tick_ms: float,
    cpu: str,
    visible: int | None = None,
    total: int | None = None,
    warnings: Sequence[str] = (),
) -> Text:
    """Build the status line above the key hints."""
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(f"#{sequence}", Style(color=theme.muted))
    text.append("  ·  ", Style(color=theme.track))
    text.append(
        f"metrics {metrics_source or 'sysfs'}",
        Style(color=theme.ok if metrics_source else theme.warn),
    )
    text.append("  ·  ", Style(color=theme.track))
    text.append(f"tick {tick_ms:.0f} ms", Style(color=theme.muted))
    text.append("  ·  ", Style(color=theme.track))
    # The app's own consumption: the honest check that the UI thread is idle.
    cpu_style = Style(color=theme.ok if _cpu_number(cpu) < 20 else theme.warn)
    text.append(f"cpu {cpu}", cpu_style)
    if visible is not None and total is not None and total > visible:
        text.append("  ·  ", Style(color=theme.track))
        text.append(
            f"partial {visible}/{total} procs",
            Style(color=theme.warn),
        )
    for warning in warnings[:1]:
        text.append("  ·  ", Style(color=theme.track))
        text.append(warning, Style(color=theme.alert))
    return text


def _cpu_number(text: str) -> float:
    """Parse the leading number out of a formatted CPU reading."""
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return 0.0


def device_tab(metrics: AmdgpuMetrics, active: bool) -> Text:
    """Build a compact tab for the device switcher."""
    theme_style = Style(bold=True) if active else Style()
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(
        f" {metrics.device.name} ",
        theme_style + Style(color="white" if active else "grey62"),
    )
    return text
