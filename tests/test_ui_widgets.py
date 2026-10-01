"""Tests for the braille graph, the gradient and the CPU meter.

The rasteriser is a pure function, so every case here is an exact snapshot rather than a
rendering assertion.
"""

import itertools

import pytest
from rich.style import Style
from rich.text import Text

from gputop.ui.theme import (
    ANSI_STOPS,
    CUBE_LEVELS,
    DEFAULT_THEME,
    DRACULA_THEME,
    GRUVBOX_THEME,
    THEMES,
    TRUECOLOR_STOPS,
    Gradient,
    get_theme,
    snap_to_cube,
    theme_names,
    threshold_style,
)
from gputop.ui.widgets import BRAILLE_BASE, BRAILLE_BITS, BrailleGraph, CpuMeter, braille_text


def render(samples: list[float | None], width: int, height: int, **kwargs) -> Text:
    """Render a series with the default theme."""
    kwargs.setdefault("gradient", Gradient(DEFAULT_THEME))
    kwargs.setdefault("track", DEFAULT_THEME.track)
    return braille_text(samples, width, height, **kwargs)


class TestBrailleBits:
    """The braille bit layout, which is the usual source of off-by-one renderers."""

    def test_bits_are_distinct(self) -> None:
        flat = [bit for row in BRAILLE_BITS for bit in row]
        assert len(flat) == 8
        assert len(set(flat)) == 8

    def test_documented_dot_mapping(self) -> None:
        """Dots 1-3 and 7 are the left column; dots 4-6 and 8 the right."""
        assert BRAILLE_BITS == ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

    def test_all_dots_is_the_last_code_point(self) -> None:
        assert chr(BRAILLE_BASE + 0xFF) == "⣿"

    def test_no_dots_is_the_first(self) -> None:
        assert chr(BRAILLE_BASE) == "⠀"


class TestBrailleGeometry:
    """Output shape and axis behaviour."""

    def test_exact_dimensions(self) -> None:
        text = render([50.0] * 10, 12, 4)
        lines = text.plain.splitlines()
        assert len(lines) == 4
        assert all(len(line) == 12 for line in lines)

    def test_zero_size_is_empty(self) -> None:
        assert render([1.0], 0, 5).plain == ""
        assert render([1.0], 5, 0).plain == ""

    def test_empty_series_shows_a_placeholder(self) -> None:
        """A blank rectangle would be indistinguishable from an idle GPU."""
        text = render([], 30, 2)
        assert "waiting for samples" in text.plain
        assert all(len(line) == 30 for line in text.plain.splitlines())

    def test_all_none_shows_a_placeholder(self) -> None:
        assert "waiting for samples" in render([None, None], 30, 1).plain

    @pytest.mark.parametrize(
        ("value", "lit_rows"),
        [
            (0.0, [3]),  # 0% marks the bottom row only
            (25.0, [2, 3]),
            (50.0, [2, 3]),
            (100.0, [0, 1, 2, 3]),  # 100% fills the column
        ],
    )
    def test_percent_scale_fills_from_the_bottom(
        self, value: float, lit_rows: list[int]
    ) -> None:
        """On a fixed 0-100 scale the fill grows upward from the bottom row."""
        lines = render([value], 1, 4).plain.splitlines()
        lit = [index for index, char in enumerate(lines) if char != chr(BRAILLE_BASE)]
        assert lit == lit_rows

    def test_zero_is_drawn_at_the_bottom(self) -> None:
        line = render([0.0], 1, 4).plain
        assert line[3] != chr(BRAILLE_BASE), "0% should still mark the bottom row"
        assert line[0] == chr(BRAILLE_BASE)

    def test_full_fills_every_row(self) -> None:
        """A 1-cell-wide graph can only light one dot column, so check rows, not dots."""
        lines = render([100.0], 1, 4).plain.splitlines()
        assert all(char != chr(BRAILLE_BASE) for char in lines)

    def test_two_cells_light_both_dot_columns(self) -> None:
        """Width 2 is the minimum that exercises the full 2x4 dot cell."""
        text = render([100.0] * 40, 2, 4)
        assert "⣿" in text.plain

    def test_a_lone_sample_sits_at_the_right_edge(self) -> None:
        """With one sample there is no history, so it represents 'now'."""
        text = render([100.0], 4, 4)
        for line in text.plain.splitlines():
            assert line[3] != chr(BRAILLE_BASE), "newest sample belongs on the right"
            assert line[:3] == chr(BRAILLE_BASE) * 3

    def test_auto_scale_centres_a_flat_series(self) -> None:
        """A perfectly flat series cannot span an axis, so it is drawn mid-height."""
        lines = render([50.0, 50.0, 50.0], 1, 8, scale="auto").plain.splitlines()
        lit = [i for i, char in enumerate(lines) if char != chr(BRAILLE_BASE)]
        assert lit == [4, 5, 6, 7], "a flat series should sit on the vertical centre"

    def test_auto_scale_expands_a_narrow_band(self) -> None:
        """Two degrees on a 0-100 axis would be invisible; auto scaling must show it."""
        lines = render([50.0, 51.0], 2, 4, scale="auto").plain.splitlines()
        assert lines[0] != chr(BRAILLE_BASE), "the high point should reach the top"

    def test_percent_scale_hides_a_narrow_band(self) -> None:
        """The fixed scale deliberately does not amplify small changes."""
        lines = render([50.0, 51.0], 2, 4, scale="percent").plain.splitlines()
        assert lines[0] == chr(BRAILLE_BASE) * 2, "neither sample reaches the top row"

    def test_gaps_are_left_empty(self) -> None:
        """A missing reading must be a hole, not an interpolated straight line."""
        text = render([100.0, None, None, 100.0], 4, 2)
        assert text.plain.splitlines()[0][1:3] == chr(BRAILLE_BASE) * 2

    def test_oldest_sample_is_leftmost(self) -> None:
        text = render([100.0, 0.0, 0.0, 0.0], 4, 4)
        lines = text.plain.splitlines()
        # The oldest sample occupies the left dot column of its cell.
        first_column = "".join(line[0] for line in lines)
        assert first_column == "\u2847" * 4  # dots 1, 2, 3 and 7
        assert lines[0][3] == chr(BRAILLE_BASE), "newest sample is at zero here"

    def test_out_of_range_values_are_clamped(self) -> None:
        high = render([150.0], 1, 4).plain.splitlines()
        low = render([-20.0], 1, 4).plain.splitlines()
        assert all(char != chr(BRAILLE_BASE) for char in high)
        assert low[3] != chr(BRAILLE_BASE)
        assert low[0] == chr(BRAILLE_BASE)

    def test_more_samples_than_columns_takes_the_peak(self) -> None:
        """Compressing must preserve a spike, otherwise a brief load is invisible."""
        values = [0.0] * 500 + [100.0] + [0.0] * 499
        lines = render(values, 10, 4).plain.splitlines()
        top_row_lit = [i for i, char in enumerate(lines[0]) if char != chr(BRAILLE_BASE)]
        assert top_row_lit, "the isolated 100% sample must still reach the top row"
        # Everything else is idle and should stay on the bottom row.
        assert lines[0].count(chr(BRAILLE_BASE)) > 5


class TestBrailleColours:
    """The gradient has to reach the dots, not just the empty track."""

    def test_filled_dots_use_the_ramp(self) -> None:
        text = render([90.0], 1, 4)
        styles = [style.color.triplet.hex for _, _, style in text._spans if style.color]
        assert any(colour != DEFAULT_THEME.track for colour in styles)

    def test_empty_dots_use_the_track(self) -> None:
        text = render([0.0], 1, 4)
        top_style = text._spans[0][2]
        assert top_style.color is not None
        assert top_style.color.name.lower().lstrip("#") == DEFAULT_THEME.track.lstrip("#")

    def test_higher_value_uses_a_warmer_colour(self) -> None:
        """Green at idle, red when saturated: the point of the gradient."""
        gradient = Gradient(DEFAULT_THEME)
        cool = gradient.at(5.0)
        hot = gradient.at(95.0)
        assert _greenness(cool) > _greenness(hot)


def _greenness(colour: str) -> int:
    """Green minus red, as a crude 'coolness' score."""
    text = colour.lstrip("#")
    return int(text[2:4], 16) - int(text[0:2], 16)


class TestGradient:
    """Ramp construction and the 256-colour fallback."""

    def test_truecolor_stop_count(self) -> None:
        assert len(Gradient(DEFAULT_THEME, truecolor=True)) == TRUECOLOR_STOPS

    def test_ansi_uses_fewer_stops(self) -> None:
        """A 256-colour terminal cannot show 256 distinct colours, so it gets a ramp
        sized to the palette rather than one that would band."""
        assert len(Gradient(DEFAULT_THEME, truecolor=False)) == ANSI_STOPS

    def test_endpoints_match_the_theme(self) -> None:
        assert Gradient(DEFAULT_THEME).at(0.0) == DEFAULT_THEME.low
        assert Gradient(DEFAULT_THEME).at(100.0) == DEFAULT_THEME.high

    def test_midpoint_is_the_middle_stop(self) -> None:
        assert Gradient(DEFAULT_THEME).at(50.0) == DEFAULT_THEME.mid

    def test_clamps_out_of_range(self) -> None:
        gradient = Gradient(DEFAULT_THEME)
        assert gradient.at(-10.0) == gradient.at(0.0)
        assert gradient.at(200.0) == gradient.at(100.0)

    def test_ramp_progresses_from_cool_to_warm(self) -> None:
        """Colour must move steadily, or a rising graph looks like noise."""
        stops = Gradient(DEFAULT_THEME).stops()
        warmth = [int(s.lstrip("#")[0:2], 16) - int(s.lstrip("#")[2:4], 16) for s in stops]
        assert warmth[-1] > warmth[0], "the ramp must end warmer than it starts"
        steps = [b - a for a, b in itertools.pairwise(warmth)]
        # Mostly non-decreasing; a couple of flat steps are fine on a long ramp.
        assert sum(step >= 0 for step in steps) >= len(steps) * 0.9

    def test_ansi_stops_are_cube_values(self) -> None:
        for stop in Gradient(DEFAULT_THEME, truecolor=False).stops():
            for offset in (0, 2, 4):
                assert int(stop.lstrip("#")[offset : offset + 2], 16) in CUBE_LEVELS

    def test_truecolor_keeps_full_precision(self) -> None:
        stops = Gradient(DEFAULT_THEME, truecolor=True).stops()
        assert any(
            int(s.lstrip("#")[offset : offset + 2], 16) not in CUBE_LEVELS
            for s in stops
            for offset in (0, 2, 4)
        )

    def test_snap_to_cube_picks_the_nearest_level(self) -> None:
        assert snap_to_cube((10, 10, 10)) == (0, 0, 0)
        assert snap_to_cube((250, 250, 250)) == (255, 255, 255)
        assert snap_to_cube((100, 0, 0)) == (95, 0, 0)


class TestThemes:
    """Theme registry behaviour."""

    def test_required_themes_exist(self) -> None:
        assert {"default", "dracula", "gruvbox"} <= set(theme_names())

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_theme("  DRACULA ").name == "dracula"

    def test_unknown_theme_falls_back(self) -> None:
        """A typo in a config file must not stop the app from starting."""
        assert get_theme("nonsense").name == "default"

    @pytest.mark.parametrize("name", ["default", "dracula", "gruvbox"])
    def test_every_theme_builds_a_ramp(self, name: str) -> None:
        assert len(Gradient(get_theme(name))) == TRUECOLOR_STOPS

    @pytest.mark.parametrize("name", ["default", "dracula", "gruvbox"])
    def test_themes_are_visually_distinct(self, name: str) -> None:
        theme = THEMES[name]
        assert theme.low != theme.high

    @pytest.mark.parametrize("name", ["default", "dracula", "gruvbox"])
    def test_every_colour_is_valid_hex(self, name: str) -> None:
        theme = THEMES[name]
        for value in (theme.low, theme.mid, theme.high, theme.accent, theme.text, theme.muted):
            assert len(value.lstrip("#")) == 6
            int(value.lstrip("#"), 16)

    def test_threshold_colour_escalates(self) -> None:
        theme = DEFAULT_THEME
        assert threshold_style(theme, None) == theme.muted
        assert threshold_style(theme, 10.0) == theme.ok
        assert threshold_style(theme, 75.0) == theme.warn
        assert threshold_style(theme, 95.0) == theme.alert

    def test_dracula_and_gruvbox_have_expected_midpoints(self) -> None:
        assert Gradient(DRACULA_THEME).at(50.0) == DRACULA_THEME.mid
        assert Gradient(GRUVBOX_THEME).at(50.0) == GRUVBOX_THEME.mid


class TestGraphWidget:
    """The widget wrapper around the pure rasteriser."""

    def test_push_and_read_back(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME)
        graph.push(10.0)
        graph.push(None)
        assert graph.values == (10.0, None)

    def test_capacity_is_enforced(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME, capacity=3)
        for value in range(5):
            graph.push(float(value))
        assert graph.values == (2.0, 3.0, 4.0)

    def test_set_series_keeps_the_tail(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME, capacity=3)
        graph.set_series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert graph.values == (3.0, 4.0, 5.0)

    def test_clear(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME)
        graph.push(1.0)
        graph.clear()
        assert graph.values == ()

    def test_unmounted_render_is_empty_not_an_error(self) -> None:
        """A zero-sized widget must not divide by zero during the first frame."""
        assert BrailleGraph(theme=DEFAULT_THEME).render().plain == ""

    def test_theme_switch_rebuilds_the_ramp(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME)
        graph.set_theme(DRACULA_THEME)
        assert graph._gradient.at(100.0) == DRACULA_THEME.high

    def test_truecolor_toggle(self) -> None:
        graph = BrailleGraph(theme=DEFAULT_THEME)
        graph.set_truecolor(False)
        assert not graph._gradient.truecolor


class TestCpuMeter:
    """The app's own CPU accounting."""

    def test_starts_at_zero(self) -> None:
        assert CpuMeter().percent == 0.0

    def test_first_sample_only_establishes_a_baseline(self) -> None:
        """A window from construction is microseconds wide and reports noise."""
        meter = CpuMeter()
        assert meter.sample() == 0.0

    def test_sampling_after_a_window_produces_a_value(self) -> None:
        meter = CpuMeter()
        meter.sample()
        busy = sum(i * i for i in range(50_000))
        value = meter.sample()
        assert value >= 0.0
        assert busy > 0

    def test_idle_process_reports_near_zero(self) -> None:
        """A real one-second window, as the app uses, must read as near zero."""
        import time

        meter = CpuMeter()
        meter.sample()
        time.sleep(0.05)
        meter.sample()
        assert meter.percent < 50.0

    def test_format_scales_with_magnitude(self) -> None:
        meter = CpuMeter()
        meter.percent = 0.42
        assert meter.format() == "0.4%"
        meter.percent = 42.0
        assert meter.format() == "42%"
        meter.percent = 180.0
        assert meter.format() == "180%"

    def test_more_than_one_core_is_not_clamped(self) -> None:
        """A multi-threaded sampler really can exceed 100%, and that is worth seeing."""
        meter = CpuMeter()
        meter.percent = 250.0
        assert meter.format() == "250%"


class TestRasteriserEquivalence:
    """The rasteriser draws per *dot column* rather than per sample.

    Folding the samples into each column before painting is what made a 300-sample graph
    about twice as fast, and it is an optimisation only if the cells come out identical.
    The reference below is the original per-sample loop, kept here as the oracle: if a
    future change to either implementation alters the picture, these fail.
    """

    @staticmethod
    def _reference(samples: list[float | None], width: int, height: int, **kwargs) -> Text:
        """The pre-optimisation implementation: fill from each sample down to the floor."""
        from gputop.ui.widgets import _axis, _placeholder

        gradient = kwargs["gradient"]
        track = kwargs["track"]
        if width <= 0 or height <= 0:
            return Text("")
        present = [value for value in samples if value is not None]
        if not present:
            return _placeholder(width, height, track)
        low, high = _axis(present, kwargs.get("scale", "percent"))
        span = high - low
        dot_width, dot_height = width * 2, height * 4
        masks = [[0] * width for _ in range(height)]
        peaks = [[low] * width for _ in range(height)]
        last = len(samples) - 1
        for index, value in enumerate(samples):
            if value is None:
                continue
            x = (index * (dot_width - 1)) // last if last else dot_width - 1
            cell_x = x >> 1
            if span <= 0:
                top = dot_height - 1
            else:
                top = round((1.0 - min(1.0, max(0.0, (value - low) / span))) * (dot_height - 1))
            for dot_y in range(top, dot_height):
                cell_y = dot_y >> 2
                masks[cell_y][cell_x] |= BRAILLE_BITS[dot_y & 3][x & 1]
                if value > peaks[cell_y][cell_x]:
                    peaks[cell_y][cell_x] = value
        rows = []
        for cell_y in range(height):
            row = Text(no_wrap=True, overflow="crop", end="")
            for cell_x in range(width):
                mask = masks[cell_y][cell_x]
                row.append(
                    chr(BRAILLE_BASE + mask),
                    Style(color=track if mask == 0 else gradient.at(peaks[cell_y][cell_x])),
                )
            rows.append(row)
        return Text("\n").join(rows)

    @pytest.mark.parametrize("width,height", [(1, 1), (5, 3), (24, 4), (58, 10), (60, 20)])
    @pytest.mark.parametrize("scale", ["percent", "auto"])
    def test_same_picture_as_the_per_sample_rasteriser(
        self, width: int, height: int, scale: str
    ) -> None:
        gradient = Gradient(DEFAULT_THEME)
        kwargs = {"gradient": gradient, "track": DEFAULT_THEME.track, "scale": scale}
        series = [
            [float(index % 101) for index in range(300)],
            [None if index % 17 == 0 else float(index * 3 % 97) for index in range(300)],
            [50.0] * 120,
            [0.0, 100.0] * 60,
            [-20.0, 140.0] * 40,
            [None] * 30,
        ]
        for samples in series:
            mine = braille_text(samples, width, height, **kwargs)
            theirs = self._reference(samples, width, height, **kwargs)
            assert mine.plain == theirs.plain, (len(samples), width, height, scale)
            assert _spans(mine) == _spans(theirs), (len(samples), width, height, scale)

    def test_a_single_sample_still_renders(self) -> None:
        """The reduction must not divide by an empty series."""
        assert render([42.0], 4, 2).plain.strip()

    def test_gaps_stay_empty(self) -> None:
        """A gap is a hole, not a bridge: an absent sample paints no dot column."""
        text = render([80.0, None, None, None], 8, 2)
        rows = text.plain.splitlines()
        assert all(cell == "⠀" for row in rows for cell in row[2:])


def _spans(text: Text) -> list[tuple[int, int, str]]:
    """Styled spans as comparable tuples, so two renderings can be compared exactly."""
    return [(span.start, span.end, str(span.style)) for span in text._spans]
