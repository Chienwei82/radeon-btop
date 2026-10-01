"""Reusable graph widgets built on Unicode braille.

Braille gives 2x4 dots per character cell, so a 120x10 panel resolves to 240x40 sample
points -- fine enough that a one-second-per-sample history looks continuous instead of
stair-stepped, which is the entire reason ``btop``-style tools use it.

The same widget serves the tall utilisation graph, the memory graphs and the one-line
sensor sparklines; only the height and scaling differ.
"""

import time
from collections import deque
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Literal

from rich.style import Style
from rich.text import Text
from textual.widget import Widget

from gputop.ui.theme import Gradient, Theme

#: First code point of the braille block; dot bits are OR-ed into this base.
BRAILLE_BASE = 0x2800

#: ``_BRAILLE_BITS[row][column]`` -> the bit that lights that dot.
#:
#: The Unicode braille pattern is not a linear grid, which is the usual source of
#: off-by-one braille renderers: dots 1-3 and 7 form the left column, dots 4-6 and 8 the
#: right one.
BRAILLE_BITS: tuple[tuple[int, int], ...] = (
    (0x01, 0x08),  # row 0: dots 1 and 4
    (0x02, 0x10),  # row 1: dots 2 and 5
    (0x04, 0x20),  # row 2: dots 3 and 6
    (0x40, 0x80),  # row 3: dots 7 and 8
)

#: How many samples a graph retains.  At the default 1 s interval this is five minutes,
#: which matches the sampler's default history length.
DEFAULT_CAPACITY = 300

ScaleMode = Literal["percent", "auto"]


@lru_cache(maxsize=512)
def _style_for(colour: str) -> Style:
    """Return a shared :class:`~rich.style.Style` for one colour.

    ``Style`` is immutable, so one instance per colour can be shared by every cell of
    every graph.  Building one per cell was a measurable share of the frame time: a
    60x20 panel is 1200 cells, and the ramp has at most 255 distinct colours in it.
    """
    return Style(color=colour)


def braille_text(
    samples: Sequence[float | None],
    width: int,
    height: int,
    *,
    gradient: Gradient,
    track: str,
    scale: ScaleMode = "percent",
) -> Text:
    """Rasterise a series into styled braille text.

    Split out from the widget so the rasteriser can be tested at explicit dimensions
    without mounting anything; :meth:`BrailleGraph.render` is a thin wrapper over this.

    Args:
        samples: Values oldest-first.  ``None`` is a gap and is left empty rather than
            interpolated across, so a reading the hardware did not provide shows as a
            hole instead of a straight line between two unrelated points.
        width: Cells across.
        height: Cells down; each cell is 2x4 dots.
        gradient: Colour ramp indexed by value.
        track: Colour for unfilled cells.
        scale: ``"percent"`` fixes the axis to 0-100; ``"auto"`` fits the data.

    Returns:
        Styled text exactly ``width`` columns by ``height`` rows.

    Filled cells are accumulated per *dot column* rather than per sample.  A 300-sample
    history in a 58-cell panel puts roughly five samples in each dot column, and every one
    of them paints the same run of dots from its own value down to the bottom -- so the
    drawing only depends on the highest point and the peak value of each column.  Doing
    the reduction first is the same output for a fraction of the work.
    """
    if width <= 0 or height <= 0:
        return Text("")

    present = [value for value in samples if value is not None]
    if not present:
        return _placeholder(width, height, track)

    low, high = _axis(present, scale)
    span = high - low
    dot_width = width * 2
    dot_height = height * 4

    # Indexed ``[parity][cell_x]``: the two dot columns of a cell are independent.
    top_of: list[list[int]] = [[dot_height] * width, [dot_height] * width]
    peak_of: list[list[float]] = [[low] * width, [low] * width]

    last = len(samples) - 1
    for index, value in enumerate(samples):
        if value is None:
            continue
        x = (index * (dot_width - 1)) // last if last else dot_width - 1
        cell_x = x >> 1
        if span <= 0:
            top = dot_height - 1
        else:
            fraction = (value - low) / span
            top = round((1.0 - min(1.0, max(0.0, fraction))) * (dot_height - 1))
        parity = x & 1
        if top < top_of[parity][cell_x]:
            top_of[parity][cell_x] = top
        if value > peak_of[parity][cell_x]:
            peak_of[parity][cell_x] = value

    masks = [[0] * width for _ in range(height)]
    peaks = [[low] * width for _ in range(height)]
    for parity in (0, 1):
        tops = top_of[parity]
        column_peaks = peak_of[parity]
        for cell_x in range(width):
            top = tops[cell_x]
            if top >= dot_height:
                continue
            value = column_peaks[cell_x]
            for dot_y in range(top, dot_height):
                cell_y = dot_y >> 2
                masks[cell_y][cell_x] |= BRAILLE_BITS[dot_y & 3][parity]
                if value > peaks[cell_y][cell_x]:
                    peaks[cell_y][cell_x] = value

    track_style = _style_for(track)
    rows: list[Text] = []
    for cell_y in range(height):
        row = Text(no_wrap=True, overflow="crop", end="")
        mask_row = masks[cell_y]
        peak_row = peaks[cell_y]
        for cell_x in range(width):
            mask = mask_row[cell_x]
            if mask == 0:
                row.append(chr(BRAILLE_BASE), track_style)
            else:
                row.append(chr(BRAILLE_BASE + mask), _style_for(gradient.at(peak_row[cell_x])))
        rows.append(row)
    # Joining rows explicitly is what makes this a plot.  Appending every cell in one
    # loop produced a single long strip that the layout then wrapped, which smeared the
    # series across the panel instead of drawing it.
    return Text("\n").join(rows)


def _axis(present: Sequence[float], scale: ScaleMode) -> tuple[float, float]:
    """Choose the value range for the y axis."""
    if scale == "percent":
        return 0.0, 100.0
    low = min(present)
    high = max(present)
    if high - low < 1e-9:
        # A flat series would otherwise produce an infinite or zero span; give it a
        # small symmetric window so it renders as a line in the middle.
        padding = max(abs(high) * 0.05, 1.0)
        return low - padding, high + padding
    margin = (high - low) * 0.15
    return low - margin, high + margin


def _placeholder(width: int, height: int, track: str) -> Text:
    """Render the empty state.

    A blank rectangle would be indistinguishable from an idle GPU, so the widget says it
    is waiting instead.
    """
    message = "waiting for samples…"
    pad = " " * max(0, (width - len(message)) // 2)
    text = Text((pad + message).ljust(width)[:width], no_wrap=True, overflow="crop", end="")
    text.stylize(Style(color=track))
    return text


class BrailleGraph(Widget):
    """A filled-area braille graph of a numeric series.

    Args:
        theme: Palette to draw with.
        capacity: Maximum number of samples retained.
        scale: ``"percent"`` fixes the axis to 0-100, which is right for utilisation.
            ``"auto"`` fits the data range, which is right for temperatures and clocks
            where the interesting movement is a small slice of the absolute value.
        gradient: The colour ramp.  Built from ``theme`` when omitted.
    """

    DEFAULT_CSS = """
    BrailleGraph { width: 1fr; height: auto; min-height: 1; }
    """

    def __init__(
        self,
        *,
        theme: Theme,
        capacity: int = DEFAULT_CAPACITY,
        scale: ScaleMode = "percent",
        gradient: Gradient | None = None,
        name: str | None = None,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._theme = theme
        self._gradient = gradient or Gradient(theme)
        self._scale_mode: ScaleMode = scale
        self._capacity = max(1, capacity)
        self._values: deque[float | None] = deque(maxlen=self._capacity)

    # -- data ---------------------------------------------------------------

    def push(self, value: float | None) -> None:
        """Append one sample.  ``None`` is a gap, not a zero."""
        self._values.append(value)
        self.refresh()

    def set_series(self, values: Iterable[float | None]) -> None:
        """Replace the whole series, keeping only the most recent samples.

        ``values`` may be any iterable, including a one-shot generator: it is materialised
        once, before the tail is taken.
        """
        items = list(values)
        self._values.clear()
        self._values.extend(items[-self._capacity :])
        self.refresh()

    def clear(self) -> None:
        """Drop every sample."""
        self._values.clear()
        self.refresh()

    @property
    def values(self) -> tuple[float | None, ...]:
        """The retained samples, oldest first."""
        return tuple(self._values)

    # -- theming ------------------------------------------------------------

    def set_theme(self, theme: Theme, *, truecolor: bool = True) -> None:
        """Switch palette at runtime and repaint."""
        self._theme = theme
        self._gradient = Gradient(theme, truecolor=truecolor)
        self.refresh()

    def set_truecolor(self, truecolor: bool) -> None:
        """Switch between the truecolor ramp and the 256-colour snapped one."""
        self._gradient = Gradient(self._theme, truecolor=truecolor)
        self.refresh()

    # -- rendering ----------------------------------------------------------

    def render(self) -> Text:
        """Rasterise the series into braille cells."""
        return braille_text(
            self._values,
            self.size.width,
            self.size.height,
            gradient=self._gradient,
            track=self._theme.track,
            scale=self._scale_mode,
        )

    # -- convenience --------------------------------------------------------


class CpuMeter:
    """Measures this process's own CPU consumption.

    Uses :func:`time.process_time_ns`, which counts CPU time across *all* threads of the
    process.  That is the number worth showing: on a machine where the procfs scan
    dominates, the sampler thread is the interesting consumer, and thread-local timing
    would report a reassuring near-zero.

    Only called from the UI thread, once a second, so it carries no synchronisation.
    """

    __slots__ = ("_last_cpu_ns", "_last_wall_ns", "_primed", "percent")

    def __init__(self) -> None:
        self._last_cpu_ns = time.process_time_ns()
        self._last_wall_ns = time.monotonic_ns()
        self._primed = False
        self.percent = 0.0

    def sample(self) -> float:
        """Recompute and return CPU usage as a percentage of one core.

        The first call only establishes the baseline: a window measured from object
        construction is a few microseconds wide, so the ratio is dominated by noise and
        has been observed to report well over 100%.
        """
        if not self._primed:
            self._primed = True
            self._last_cpu_ns = time.process_time_ns()
            self._last_wall_ns = time.monotonic_ns()
            return self.percent
        cpu_now = time.process_time_ns()
        wall_now = time.monotonic_ns()
        cpu_delta = cpu_now - self._last_cpu_ns
        wall_delta = wall_now - self._last_wall_ns
        self._last_cpu_ns = cpu_now
        self._last_wall_ns = wall_now
        if wall_delta <= 0:
            return self.percent
        # A value above 100 means more than one core was busy, which is possible and
        # worth showing rather than clamping away.
        self.percent = 100.0 * cpu_delta / wall_delta
        return self.percent

    def format(self) -> str:
        """Format the last sampled value, keeping a decimal only when it matters."""
        return f"{self.percent:.1f}%" if self.percent < 10.0 else f"{self.percent:.0f}%"
