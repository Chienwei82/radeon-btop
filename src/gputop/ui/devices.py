"""Multi-GPU presentation: the device tab bar and the all-at-once overview.

Two widgets serve the same question at different scales.  The tab bar answers "which GPU
am I looking at, and how busy is the other one", which is a one-line question and belongs
where the eye already is.  The overview answers "is anything wrong with *any* of them",
which is the question you have when something is wrong and want the next screen to have
the same shape on every machine.

The builders are pure functions returning :class:`~rich.text.Text`, like every other
panel here, so the tab geometry -- which is what click handling depends on -- can be
asserted without a terminal.
"""

from collections.abc import Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.message import Message
from textual.widget import Widget

from gputop.model.metrics import AmdgpuMetrics, MemoryPool
from gputop.ui.format import fmt, fmt_bytes, fmt_percent, fmt_temp
from gputop.ui.panels import BAR_EMPTY, BAR_FILLED, LABEL_WIDTH, fit_bar_width, stat_row
from gputop.ui.theme import Gradient, Theme

#: Separator between tabs.  A box-drawing character rather than a space so an idle tab is
#: still a visible target to click.
TAB_SEPARATOR = "│"

#: Cells used by the mini usage bar inside a tab.  Four is what fits next to a name; the
#: number beside it carries the precise value.
TAB_BAR_CELLS = 4

#: Smallest a tab may become before the name is truncated rather than the bar dropped.
MIN_TAB_WIDTH = 18

#: Cells a tab spends on everything that is neither the name, the kind, the percentage nor
#: the badge number: the space either side of each of the three badges (6), the mini bar
#: (4), the gap before the percentage (1) and the trailing active-marker cell (2).
#:
#: The badge's *digits* are deliberately not counted here -- their width depends on which
#: device this is, which is what :func:`tab_width` adds.
TAB_CHROME = 13

#: Cells the separator costs, for every tab but the first.
TAB_GAP = 3


def device_busy(metrics: AmdgpuMetrics) -> float | None:
    """Utilisation used for a tab's mini bar, preferring the busy metric.

    ``mem_busy_percent`` is the fallback because on a card that reports neither, a
    memory-bound workload is still visibly doing something, and an empty bar next to a
    name would read as "this GPU is fine".
    """
    for value in (metrics.gpu_busy_percent, metrics.mem_busy_percent):
        if value is not None:
            return value
    return None


def _mini_bar(percent: float | None, theme: Theme, cells: int = TAB_BAR_CELLS) -> Text:
    """A tiny usage bar, or a flat track when the reading is unavailable."""
    text = Text(no_wrap=True, end="")
    if percent is None:
        text.append(BAR_EMPTY * cells, Style(color=theme.track))
        return text
    filled = round(min(100.0, max(0.0, percent)) / 100.0 * cells)
    text.append(BAR_FILLED * filled, Style(color=theme.accent))
    text.append(BAR_EMPTY * (cells - filled), Style(color=theme.track))
    return text


def tab_label(metrics: AmdgpuMetrics, theme: Theme, *, active: bool, number: int) -> Text:
    """Build one device tab.

    The focused tab is marked with a reversed number and a trailing caret rather than a
    highlighted background: a 40-cell band of colour for the whole run of the program is
    a lot of attention to spend on a state that is already three cells away in the
    header.
    """
    text = Text(no_wrap=True, overflow="crop", end="")
    marker_style = Style(color=theme.track, bgcolor=theme.accent) if active else Style()
    text.append(f" {number} ", marker_style)
    kind = "iGPU" if metrics.device.is_integrated else "dGPU"
    name_style = Style(bold=True, color=theme.text) if active else Style(color=theme.muted)
    text.append(f" {metrics.device.name} ", name_style)
    text.append(f" {kind} ", Style(color=theme.accent if active else theme.track))
    busy = device_busy(metrics)
    text.append_text(_mini_bar(busy, theme))
    text.append(" ")
    text.append(fmt_percent(busy), Style(color=theme.text if active else theme.muted))
    text.append(" ◂" if active else "  ", Style(color=theme.accent))
    return text


def gpu_tabs(
    devices: Sequence[AmdgpuMetrics], theme: Theme, *, index: int, available: int = 200
) -> Text:
    """Build the whole device switcher line.

    Args:
        devices: Every device, in the sampler's order.
        theme: Palette.
        index: Which device is focused.
        available: Columns the line may occupy.  Tabs are truncated rather than wrapped:
            a wrapped tab bar reads as a second row of devices that do not exist.
    """
    text = Text(no_wrap=True, overflow="crop", end="")
    for position, metrics in enumerate(devices):
        if position:
            text.append(f" {TAB_SEPARATOR} ", Style(color=theme.track))
        text.append_text(
            tab_label(metrics, theme, active=metrics.device.index == index, number=position + 1)
        )
    if len(text) > available:
        text.truncate(max(0, available), overflow="crop")
    return text


def tab_width(devices: Sequence[AmdgpuMetrics], position: int) -> int:
    """Cells occupied by one tab, separator included.

    Computed from the same fields :func:`tab_label` prints, so the two cannot drift: a
    hit test that disagrees with what was drawn puts a click on the wrong GPU, which is
    the one bug no screenshot would ever catch.  The badge's digit count is part of that,
    and it grows at the tenth GPU -- a machine with a dozen cards would otherwise put
    every boundary one or two cells out of place.

    The name is measured in *cells*, not code points, because that is the unit the
    terminal lays out in and the unit :func:`tab_label` produces.  ``[gpu.names]`` is
    user-supplied and a CJK name is a normal thing to put in it; ``len("显卡")`` is 2
    where the name occupies 4 cells, so every boundary after it fell short by two and the
    right-hand end of the last tab was not clickable at all.
    """
    metrics = devices[position]
    kind = "iGPU" if metrics.device.is_integrated else "dGPU"
    percent = fmt_percent(device_busy(metrics))
    number = len(str(position + 1))
    width = TAB_CHROME + number + cell_len(metrics.device.name) + cell_len(kind)
    width += cell_len(percent)
    width = max(MIN_TAB_WIDTH, width)
    return width + (TAB_GAP if position else 0)


def tab_spans(devices: Sequence[AmdgpuMetrics]) -> tuple[tuple[int, int, int], ...]:
    """Every tab's geometry as ``(start, end, position)``."""
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for position in range(len(devices)):
        width = tab_width(devices, position)
        spans.append((offset, offset + width, position))
        offset += width
    return tuple(spans)


def device_at(devices: Sequence[AmdgpuMetrics], x: int) -> int:
    """The device position covering cell ``x``, or ``-1`` when past the last tab."""
    for start, end, position in tab_spans(devices):
        if start <= x < end:
            return position
    return -1


def _pool(label: str, pool: MemoryPool, theme: Theme, inner: int, gradient: Gradient) -> Text:
    """A memory row sized for the overview card."""
    percent = pool.percent
    detail = f"{fmt_bytes(pool.used)} / {fmt_bytes(pool.total)}"
    text = Text(no_wrap=True, overflow="crop", end="")
    text.append(label.ljust(LABEL_WIDTH), Style(color=theme.muted))
    width = fit_bar_width(inner, len(detail) + 1)
    if percent is None:
        text.append(BAR_EMPTY * width, Style(color=theme.track))
    else:
        filled = round(min(100.0, percent) / 100.0 * width)
        text.append(BAR_FILLED * filled, Style(color=gradient.at(percent)))
        text.append(BAR_EMPTY * (width - filled), Style(color=theme.track))
    text.append(f" {detail}", Style(color=theme.text))
    return text


def _overview_card(
    metrics: AmdgpuMetrics,
    theme: Theme,
    inner: int,
    gradient: Gradient,
    *,
    active: bool,
) -> list[Text]:
    """Build one device's overview card, as a fixed number of rows.

    Every card is the same height on purpose: side by side, ragged cards would leave
    holes in the layout exactly where a second GPU's numbers should be.
    """
    device = metrics.device
    rows: list[Text] = []

    header = Text(no_wrap=True, overflow="crop", end="")
    header.append(device.name, Style(color=theme.text, bold=True))
    header.append("  iGPU" if device.is_integrated else "  dGPU", Style(color=theme.accent))
    if device.boot_vga:
        header.append("  boot", Style(color=theme.warn))
    if active:
        header.append("  ◂ focused", Style(color=theme.accent))
    rows.append(header)

    rows.append(
        stat_row(
            "GPU",
            fmt_percent(metrics.gpu_busy_percent),
            metrics.gpu_busy_percent,
            theme,
            inner,
            gradient,
        )
    )
    # On an integrated GPU the working set is system memory, so GTT leads for the same
    # reason the main memory panel leads with it.
    primary, secondary = (
        (metrics.gtt, metrics.vram)
        if device.is_integrated
        else (
            metrics.vram,
            metrics.gtt,
        )
    )
    rows.append(
        _pool("GTT" if device.is_integrated else "VRAM", primary, theme, inner, gradient)
    )
    rows.append(
        _pool("VRAM" if device.is_integrated else "GTT", secondary, theme, inner, gradient)
    )

    temps = "  ".join(f"{t.label} {fmt_temp(t.celsius)}" for t in metrics.temperatures[:3])
    rows.append(
        Text(
            " ".ljust(LABEL_WIDTH) + (temps or "N/A"),
            Style(color=theme.muted if temps else theme.track),
        )
    )

    power = f"{fmt(metrics.power.draw_w, digits=1)} W"
    if metrics.power.cap_w is not None:
        power += f" / {metrics.power.cap_w} W"
    if device.supports_fan and metrics.fan.rpm is not None:
        power += f"   fan {metrics.fan.rpm:,} rpm"
    rows.append(Text(" ".ljust(LABEL_WIDTH) + power, Style(color=theme.muted)))

    rows.append(Text(" ".ljust(LABEL_WIDTH) + device.bdf, Style(color=theme.muted)))
    return rows


def overview_panel(
    devices: Sequence[AmdgpuMetrics],
    theme: Theme,
    available: int,
    *,
    index: int = 0,
    gradient: Gradient | None = None,
) -> Text:
    """Render every GPU side by side, one compact card each.

    Args:
        devices: Every device the sampler found.
        theme: Palette.
        available: Total columns available.
        index: The focused device, marked in its card.
        gradient: Ramp for the bars; built from ``theme`` when omitted.
    """
    if not devices:
        return Text("")
    ramp = gradient or Gradient(theme)
    count = len(devices)
    gutter = 1
    # A single very wide card would waste the point of a compact overview, so the width is
    # capped and the row is allowed to overflow into a second line of cards if a machine
    # has more GPUs than fit.
    card = min(46, max(28, (available - gutter * (count - 1)) // count))
    inner = card - 4
    cards = [
        _overview_card(metrics, theme, inner, ramp, active=metrics.device.index == index)
        for metrics in devices
    ]
    return _join_cards(cards, card, gutter, theme)


def _join_cards(cards: Sequence[Sequence[Text]], card: int, gutter: int, theme: Theme) -> Text:
    """Place the cards next to each other, boxed and of equal height.

    The border is drawn by hand rather than with a ``Table`` or a ``Panel`` because the
    cards have to share a baseline *and* keep their own internal styling, and a layout
    container would put its own frame between them, eating the columns the bars need.
    Every line is built at exactly ``card`` cells wide, which is what keeps the boxes
    closing on the right instead of drifting by a character per card.
    """
    height = max(len(rows) for rows in cards)
    edge = Style(color=theme.accent)
    out = Text(no_wrap=True, overflow="crop", end="")

    for index in range(height + 2):
        for position, rows in enumerate(cards):
            if position:
                out.append(" " * gutter)
            if index == 0:
                out.append("╭" + "─" * (card - 2) + "╮", edge)
            elif index == height + 1:
                out.append("╰" + "─" * (card - 2) + "╯", edge)
            else:
                body = rows[index - 1].copy()
                body.truncate(card - 4, overflow="crop")
                body.pad_right(max(0, card - 4 - len(body.plain)))
                out.append("│ ", edge)
                out.append_text(body)
                out.append(" │", edge)
        if index < height + 1:
            out.append("\n")
    return out


class GpuSwitcher(Widget):
    """The clickable device tab bar.

    A row of :class:`~textual.widgets.Button` widgets would give focus rings and
    accessibility handling for free, but it would also give every tab its own background,
    padding and minimum width -- three things that make a six-GPU bar wrap on an
    80-column terminal.  One widget drawing one styled line keeps the bar to a single row
    however many devices exist, and the hit testing is then arithmetic over that line.
    """

    DEFAULT_CSS = """
    GpuSwitcher { height: 1; width: 1fr; }
    """

    class DeviceClicked(Message):
        """Posted when a tab is clicked.

        Args:
            index: Position in the device list, 0-based.
        """

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(
        self,
        theme: Theme,
        *,
        devices: Sequence[AmdgpuMetrics] = (),
        index: int = 0,
        name: str | None = None,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._theme = theme
        self._devices = tuple(devices)
        self._index = index

    def set_theme(self, theme: Theme) -> None:
        """Switch palette and repaint."""
        self._theme = theme
        self.refresh()

    def update_devices(self, devices: Sequence[AmdgpuMetrics], index: int) -> None:
        """Replace the device list and the focused index."""
        self._devices = tuple(devices)
        self._index = index
        self.refresh()

    @property
    def devices(self) -> tuple[AmdgpuMetrics, ...]:
        """The devices currently shown."""
        return self._devices

    def device_at(self, x: int) -> int:
        """The device position covering cell ``x``, or ``-1``."""
        return device_at(self._devices, x)

    def render(self) -> Text:
        """Draw the bar."""
        return gpu_tabs(
            self._devices, self._theme, index=self._index, available=self.size.width
        )

    def on_click(self, event: object) -> None:
        """Switch to the clicked device."""
        position = self.device_at(int(getattr(event, "x", -1)))
        if position >= 0:
            self.post_message(self.DeviceClicked(position))


__all__ = [
    "MIN_TAB_WIDTH",
    "TAB_BAR_CELLS",
    "TAB_GAP",
    "TAB_SEPARATOR",
    "GpuSwitcher",
    "device_at",
    "device_busy",
    "gpu_tabs",
    "overview_panel",
    "tab_label",
    "tab_spans",
    "tab_width",
]
