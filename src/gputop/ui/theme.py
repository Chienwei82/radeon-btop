"""Colour themes and the utilisation gradient.

A theme is a small set of named colours.  The graph gradient is *computed*, not stored:
it is interpolated between three stops (idle, mid, saturated) so the colour tracks the
value the way a thermometre does, instead of jumping between three flat colours.

Two rendering targets are supported:

* **truecolor** -- the full interpolated gradient, 256 stops.
* **256 colours** -- the same stops snapped to the xterm 6x6x6 cube.  Emitting a smooth
  24-bit gradient into a 256-colour terminal would band, because only 216 cube entries
  exist; snapping first keeps the ramp visually monotonic.
* **monochrome** (``--no-color``) -- no styling at all.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Channel levels of the xterm 256-colour 6x6x6 cube.  Index * 40 + 16 is the palette slot.
CUBE_LEVELS = (0, 95, 135, 175, 215, 255)

#: Number of gradient stops.  Both counts are deliberately *odd* so that a stop lands
#: exactly on the 0.5 position, which makes 50% render as the theme's mid colour rather
#: than a hair either side of it.  Truecolor gets a smooth ramp; a 256-colour terminal
#: needs far fewer stops because the palette cannot represent them anyway.
TRUECOLOR_STOPS = 255
ANSI_STOPS = 25


@dataclass(frozen=True, slots=True)
class Theme:
    """A named palette.

    Attributes:
        name: Identifier used in config and on the command line.
        low: Gradient colour for an idle workload.
        mid: Gradient colour for a half-loaded workload.
        high: Gradient colour for a saturated workload.
        accent: Border and heading colour.
        text: Primary text colour.
        muted: Secondary text colour for labels and units.
        track: The unfilled part of a bar and the graph background.
        ok: Healthy state.
        warn: Attention state, e.g. a fan at its limit.
        alert: Failure state, e.g. throttling.
    """

    name: str
    low: str
    mid: str
    high: str
    accent: str
    text: str
    muted: str
    track: str
    ok: str
    warn: str
    alert: str


DEFAULT_THEME = Theme(
    name="default",
    low="#4ec9b0",
    mid="#dcdcaa",
    high="#f7768e",
    accent="#61afef",
    text="#e6e6e6",
    muted="#7f848e",
    track="#3a3f4b",
    ok="#4ec9b0",
    warn="#e5c07b",
    alert="#f7768e",
)

DRACULA_THEME = Theme(
    name="dracula",
    low="#50fa7b",
    mid="#f1fa8c",
    high="#ff5555",
    accent="#bd93f9",
    text="#f8f8f2",
    muted="#6272a4",
    track="#44475a",
    ok="#50fa7b",
    warn="#f1fa8c",
    alert="#ff5555",
)

GRUVBOX_THEME = Theme(
    name="gruvbox",
    low="#b8bb26",
    mid="#fabd2f",
    high="#fb4934",
    accent="#83a598",
    text="#ebdbb2",
    muted="#928374",
    track="#504945",
    ok="#b8bb26",
    warn="#fabd2f",
    alert="#fb4934",
)

THEMES: Mapping[str, Theme] = MappingProxyType(
    {
        "default": DEFAULT_THEME,
        "dracula": DRACULA_THEME,
        "gruvbox": GRUVBOX_THEME,
    }
)

DEFAULT_THEME_NAME = "default"


def get_theme(name: str) -> Theme:
    """Look up a theme by name, falling back to the default.

    An unknown name must not raise: it comes from a config file or a command line the
    user can typo, and refusing to start over a colour name would be absurd.
    """
    return THEMES.get(name.strip().lower(), DEFAULT_THEME)


def theme_names() -> tuple[str, ...]:
    """Return the available theme names in a stable order."""
    return tuple(THEMES)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` into a channel triple."""
    text = value.lstrip("#")
    if len(text) != 6:
        # A malformed colour in a custom theme should degrade, not explode mid-render.
        return 255, 255, 255
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        # Right length, wrong alphabet: ``#zzzzzz`` reaches int() just as surely as a
        # short string does, and the promise above has to cover both.
        return 255, 255, 255


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Format a channel triple as ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def snap_to_cube(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Snap a colour onto the xterm 6x6x6 cube.

    Each channel is moved to the nearest cube level.  This is what keeps a gradient
    monotonic once a 24-bit value has to survive a trip through a 256-colour terminal.
    """
    return tuple(  # type: ignore[return-value]
        min(CUBE_LEVELS, key=lambda level: abs(level - channel)) for channel in rgb
    )


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _mix(
    start: tuple[int, int, int], end: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Blend two colours; ``t`` of 0 yields ``start`` and 1 yields ``end``."""
    return (
        round(_lerp(start[0], end[0], t)),
        round(_lerp(start[1], end[1], t)),
        round(_lerp(start[2], end[2], t)),
    )


class Gradient:
    """A precomputed colour ramp for one theme and one colour depth.

    The ramp is built once and indexed by value, so drawing a graph performs no
    interpolation at paint time.  That matters: the graph is rebuilt on every resize and
    every sample, and doing per-dot maths there shows up in the frame time.
    """

    __slots__ = ("_stops", "truecolor")

    def __init__(self, theme: Theme, *, truecolor: bool = True) -> None:
        """Build the ramp.

        Args:
            theme: The palette to interpolate between.
            truecolor: When false, every stop is snapped to the 256-colour cube.
        """
        self.truecolor = truecolor
        count = TRUECOLOR_STOPS if truecolor else ANSI_STOPS
        low = _hex_to_rgb(theme.low)
        mid = _hex_to_rgb(theme.mid)
        high = _hex_to_rgb(theme.high)

        stops: list[str] = []
        for index in range(count):
            position = index / (count - 1)
            rgb = (
                _mix(low, mid, position * 2)
                if position <= 0.5
                else _mix(mid, high, (position - 0.5) * 2)
            )
            if not truecolor:
                rgb = snap_to_cube(rgb)
            stops.append(_rgb_to_hex(rgb))
        self._stops = tuple(stops)

    def __len__(self) -> int:
        return len(self._stops)

    def at(self, percent: float) -> str:
        """Return the colour for a 0-100 value.

        The value is clamped rather than rejected: a slightly out-of-range reading is
        ordinary on a busy machine, and the top of the ramp is the correct answer.
        """
        clamped = min(100.0, max(0.0, percent))
        index = round(clamped / 100.0 * (len(self._stops) - 1))
        return self._stops[index]

    def stops(self) -> tuple[str, ...]:
        """Expose the ramp, for tests and for the theme preview."""
        return self._stops


def threshold_style(theme: Theme, percent: float | None) -> str:
    """Pick a status colour from a percentage.

    Used where a gradient would be misleading -- a discrete state, not a magnitude.
    """
    if percent is None:
        return theme.muted
    if percent >= 90.0:
        return theme.alert
    if percent >= 70.0:
        return theme.warn
    return theme.ok
