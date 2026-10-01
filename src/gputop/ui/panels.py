"""Text builders for each panel.

These return :class:`rich.text.Text` rather than markup strings because the bars need
per-character colour, which markup cannot express.  Keeping them as pure functions makes
every panel directly testable without mounting a widget.
"""

from collections.abc import Sequence

from rich.style import Style
from rich.text import Text

from gputop.model.blocks import BLOCK_LABEL_WIDTH, BlocksStatus, ClockReading, GpuBlocks
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


#: Columns reserved for the left-hand label, wide enough for "temp junction".
LABEL_WIDTH = 14

#: Marks a row as an overdrive ceiling rather than a live measurement.  Kept as a constant
#: because the profile panel has to reserve room for it when it sizes that column.
OD_PREFIX = "OD "


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
    *,
    label_width: int = LABEL_WIDTH,
) -> Text:
    """Build one ``label  bar  value`` line that fits ``available`` columns.

    Args:
        label: Left-hand label.
        value: The formatted reading.
        fraction: 0-100 for the bar, or ``None`` when the reading is unavailable.
        theme: Palette.
        available: Total columns the row may occupy.
        gradient: Ramp used to colour the bar; built from ``theme`` when omitted.
        label_width: Columns reserved for the label.  The blocks panel passes a wider one
            because hardware block names are longer than sensor names; the label is not
            truncated to fit, so a caller that passes too small a value overflows the row.

    An unavailable reading draws an empty bar and prints ``N/A``.  A full bar for a
    missing value would be the single most misleading thing this UI could do.
    """
    ramp = gradient or Gradient(theme)
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(label.ljust(label_width), Style(color=theme.muted))
    width = fit_bar_width(available, len(value) + 1, label=label_width)
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
    # ``fmt`` already renders a missing cap as ``N/A``, so there is no second arm to keep
    # in step -- and no way for one of them to end up indented differently from the rest
    # of the panel.
    rows.append(Text(f"{' ' * LABEL_WIDTH}cap {fmt(cap)} W", Style(color=theme.muted)))

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


def blocks_panel(
    blocks: GpuBlocks | None,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> list[Text]:
    """Build the per-block utilisation panel.

    Every block is a bar on a 0-100 scale, and the groups run together without separators:
    the model has already grouped them by pipeline stage, and a blank row between groups
    costs a line of a terminal that may not have one to spare.

    A block at exactly zero still draws an empty bar with a ``0%`` label.  That is not the
    same as a missing block -- radeontop omits a block the hardware has no unit for, so
    every row here describes something that exists.

    Args:
        blocks: The sample, or ``None`` when radeontop has not produced one.  ``None``
            renders a single explanatory row rather than a column of empty bars, because
            fourteen ``N/A`` labels answer nothing.
        theme: Palette.
        available: Columns the panel may occupy.
        gradient: Ramp for the bars; built from ``theme`` when omitted.
    """
    ramp = gradient or Gradient(theme)
    if blocks is None:
        return [Text("waiting for radeontop", Style(color=theme.muted))]

    rows: list[Text] = []
    for group in blocks.display_rows():
        for reading in group:
            if isinstance(reading, ClockReading) and reading.mhz is not None:
                # A clock has an absolute frequency worth more than a bare percentage, so
                # the bar shows utilisation and the text shows the number being read.
                value = f"{reading.mhz:,} MHz"
            else:
                # Falling back to the percentage rather than to N/A: a clock whose
                # frequency radeontop did not report still has a utilisation figure, and
                # printing N/A beside a bar that is plainly filled contradicts itself.
                value = fmt_percent(reading.percent)
            rows.append(
                stat_row(
                    reading.label,
                    value,
                    reading.percent,
                    theme,
                    available,
                    ramp,
                    label_width=BLOCK_LABEL_WIDTH,
                )
            )
    if not rows:
        return [Text("no blocks reported", Style(color=theme.muted))]
    return rows


def blocks_heading(blocks: GpuBlocks | None) -> str:
    """A one-line heading naming the busiest block, or a neutral title.

    Putting the answer in the panel's title is what makes this panel worth opening: "which
    block is busy" is the question, and the bars are the evidence.
    """
    if blocks is None:
        return "Blocks"
    busiest = blocks.busiest()
    if busiest is None:
        return "Blocks"
    return f"Blocks · {busiest.label} {busiest.percent:.0f}%"


def power_profile_panel(
    metrics: AmdgpuMetrics,
    theme: Theme,
    available: int,
    gradient: Gradient | None = None,
) -> list[Text]:
    """Build the read-only power-profile and overdrive panel.

    Everything shown here is a value the driver published, read once per sample.  Nothing in
    this panel, or anywhere else in gputop, writes to these files: the panel exists to show
    what the card is set to, and a monitor that could change the setting from a keystroke
    would be a different and much more dangerous program.

    Args:
        metrics: The sample carrying ``odc`` and ``profiles``.
        theme: Palette.
        available: Columns the panel may occupy.
        gradient: Ramp for the overdrive bars; built from ``theme`` when omitted.
    """
    ramp = gradient or Gradient(theme)
    rows: list[Text] = []
    # Detail rows are indented to the blocks panel's wider label column rather than the
    # sensors' 14, so the two panels sitting side by side share one left edge.
    indent = " " * BLOCK_LABEL_WIDTH

    def detail(label: str, value: str, colour: str) -> Text:
        """One indented ``label value`` row, trimmed to the panel width.

        Trimmed here rather than left to the widget's own cropping.  A row that is merely
        "no-wrap, crop" still measures as two lines tall in Textual, and a panel sized
        ``height: auto`` from a row that renders taller than it lays out draws the overflow
        over its own bottom border.  Fitting the text to the width is what keeps the box
        and its contents the same size.
        """
        text = f"{indent}{label} {value}".rstrip()
        if len(text) > available:
            # An ellipsis, so a truncated list reads as truncated rather than as a list
            # that happens to end there.
            text = text[: max(0, available - 1)].rstrip() + "…"
        return Text(text, Style(color=colour), no_wrap=True, overflow="crop")

    table = metrics.profiles
    active = table.active_name or metrics.power_profile
    rows.append(
        detail(
            "profile",
            active if active else NA,
            theme.text if active else theme.muted,
        )
    )
    if table.profiles:
        entry = table.active
        rows.append(
            detail(
                "index",
                f"{entry.index if entry is not None else NA} of {len(table.profiles)}",
                theme.muted,
            )
        )
        others = [p.name for p in table.profiles if p.name != active]
        if others:
            # Trimmed, not wrapped: a wrapped row draws over the panel's bottom border.  The
            # complete list is in the JSON dump and the session log, so shortening it here
            # costs nothing that cannot be got elsewhere.
            rows.append(detail("available", ", ".join(others), theme.muted))
    else:
        rows.append(detail("profiles", NA, theme.muted))

    odc = metrics.odc
    if odc.domains:
        for domain in odc.domains:
            rows.append(
                stat_row(
                    f"{OD_PREFIX}{domain.label}",
                    domain.describe(),
                    domain.percent,
                    theme,
                    available,
                    ramp,
                    # Three columns wider than the block labels: the "OD " prefix is part
                    # of this row's label, and reserving space for it is what stops the
                    # bar starting flush against the longest one.
                    label_width=BLOCK_LABEL_WIDTH + len(OD_PREFIX),
                )
            )
        if odc.vddgfx_offset_mv is not None:
            sign = "+" if odc.vddgfx_offset_mv >= 0 else ""
            rows.append(detail("vddgfx", f"{sign}{odc.vddgfx_offset_mv} mV", theme.muted))
    else:
        rows.append(detail("overdrive", NA, theme.muted))
    rows.append(detail("read-only", "", theme.muted))
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
    blocks_status: BlocksStatus = BlocksStatus.OK,
    blocks_hint: str = "",
) -> Text:
    """Build the status line above the key hints.

    The blocks hint is filtered here rather than by the caller so that the policy -- explain
    a problem, but stay silent about a deliberate choice -- lives with the line that shows
    it and can be asserted on directly.

    Args:
        blocks_status: Whether the optional block panel is working.
        blocks_hint: A more specific explanation to show in place of the status's own.

    Deliberately silent for :attr:`BlocksStatus.OK` and :attr:`BlocksStatus.DISABLED`: a
    user who turned the panel off chose that, and reminding them every second would train
    them to ignore the part of the status line that matters.  The other statuses are all
    cases where the user asked for data and is not getting it.
    """
    hint = ""
    if blocks_status not in (BlocksStatus.OK, BlocksStatus.DISABLED):
        hint = blocks_hint or blocks_status.hint
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
    if warnings:
        # The first warning, then a count for the rest.  Showing only the first was silent
        # truncation: the sampler raises one warning per failing device, so a two-GPU
        # machine with one unreadable card reported that card and the other disappeared
        # with no trace.  The count keeps the line from pushing the blocks hint off the
        # right edge while still admitting that something was withheld.
        text.append("  ·  ", Style(color=theme.track))
        text.append(warnings[0], Style(color=theme.alert))
        if len(warnings) > 1:
            text.append(f" (+{len(warnings) - 1} more)", Style(color=theme.warn))
    if hint:
        # After warnings, and in the muted colour rather than the alert colour: a missing
        # radeontop is a fact about the installation, not a fault in the GPU.
        text.append("  ·  ", Style(color=theme.track))
        text.append(f"blocks: {hint}", Style(color=theme.muted))
    return text


def _cpu_number(text: str) -> float:
    """Parse the leading number out of a formatted CPU reading."""
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return 0.0
