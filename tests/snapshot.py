"""Screen-snapshot helpers for tests.

Textual's SVG screenshot is not what a regression test wants: it embeds the theme, so an
intentional colour change would fail an image comparison for no useful reason.  These
helpers capture the *text* layout instead, which is what the UI's structure actually is,
and optionally the colours of a region, which is what the braille graph must get right.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.pilot import Pilot

#: Width used when rendering a renderable to plain text for a snapshot.
CAPTURE_WIDTH = 120

#: How long a mount or sample wait may take before the test is considered stuck.
WAIT_TIMEOUT = 10.0
WAIT_INTERVAL = 0.02


async def wait_until(
    predicate: Callable[[], bool], *, what: str = "condition", timeout: float = WAIT_TIMEOUT
) -> None:
    """Poll until ``predicate`` holds.

    A bare ``pilot.pause()`` yields the loop once, which is not the same as "the app has
    mounted": under load the widgets are not queryable yet, and a test that reaches for
    one too early fails intermittently with ``NoMatches`` rather than for a real reason.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(WAIT_INTERVAL)
    raise AssertionError(f"timed out waiting for {what} after {timeout}s")


async def run_app[T](
    app: App[object],
    *,
    size: tuple[int, int] = (120, 40),
    body: Callable[[Pilot[Any]], Awaitable[T]] | None = None,
    settle: int = 0,
) -> T:
    """Run an app to a settled state, then capture something from it.

    Args:
        app: The application under test.
        size: Terminal size to run at.
        body: Optional coroutine invoked once the app is ready; its result is returned.
        settle: Extra samples to wait for before running ``body``.

    Waiting is on the app's own tick count rather than a sleep, so the tests are neither
    slow nor timing-dependent.
    """
    async with app.run_test(size=size) as pilot:
        await wait_until(lambda: len(app.screen.children) > 0, what="mount")
        if settle or body is not None:
            ticks = _ticks_of(app)
            await wait_until(
                lambda: _ticks_of(app) >= ticks + settle,
                what=f"{settle} more samples",
            )
        # Layout settles a frame after the widgets are sized; capturing earlier can
        # snapshot a half-laid-out screen that the running app would never show.
        await wait_for_stable_layout(app)
        if body is None:
            await pilot.pause()
            return None  # type: ignore[return-value]
        return await body(pilot)


async def wait_for_stable_layout(app: App[object], timeout: float = WAIT_TIMEOUT) -> None:
    """Wait until the widget geometry stops changing between animation frames.

    Public because a test that presses a key and then reads the screen needs the same
    guarantee as one that only reads it: without it the capture can land between the
    action and the frame that shows it, and the test fails only on a busy machine.
    """
    deadline = time.monotonic() + timeout
    previous: tuple[object, ...] | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        current = tuple(
            (str(widget.id), widget.size.width, widget.size.height)
            for widget in app.screen.walk_children()
        )
        if current == previous:
            return
        previous = current
    raise AssertionError("layout did not settle")


def _ticks_of(app: App[object]) -> int:
    """Sampler ticks so far, or 0 when the app exposes no sampler."""
    sampler = getattr(app, "_sampler", None)
    if sampler is None:
        return 0
    return sampler.stats.ticks


def renderable_text(renderable: object, width: int = CAPTURE_WIDTH) -> str:
    """Render any Rich renderable to plain text.

    Args:
        renderable: A ``Text``, ``Group``, ``Table`` or anything else Rich can print.
        width: Columns to render at.

    Returns:
        The rendered text with no ANSI escapes, so a snapshot stays readable in a diff.
    """
    console = Console(width=width, no_color=True, legacy_windows=False, force_terminal=False)
    with console.capture() as captured:
        console.print(renderable)
    return captured.get()


def widget_text(widget: object) -> str:
    """Render a widget's current content to plain text.

    A Textual ``Static`` does not return a ``Text`` from ``render()`` -- it returns a
    ``RichVisual`` wrapper -- so this goes through a Console rather than assuming the
    concrete type.
    """
    render = getattr(widget, "render", None)
    if render is None:
        raise TypeError(f"{widget!r} has no render()")
    result = render()
    if isinstance(result, Text):
        return result.plain
    return renderable_text(result)


def text_segments(text: Text) -> list[tuple[str, str]]:
    """Split styled text into ``(character, colour)`` pairs.

    Braille encodes eight dots in one character, so a snapshot of the graph has to be
    compared per character *and* per colour to be worth anything.
    """
    pairs: list[tuple[str, str]] = []
    for start, end, style in text._spans:
        colour = str(style.color) if style and style.color else ""
        pairs.extend((text.plain[index], colour) for index in range(start, end))
    return pairs


def screen_lines(app: App[object], width: int = CAPTURE_WIDTH) -> list[str]:
    """Return the visible screen as plain-text lines.

    Args:
        app: A running app.
        width: Columns to render at.

    The strip text is taken *literally*.  It used to be re-rendered through
    ``Console.render_str``, which parses Rich markup: a screen showing a process whose
    command contained ``[/]`` raised ``MarkupError`` here -- in the test helper, not in
    the app -- and any test about markup-like text could not even capture the screen.
    """
    lines: list[str] = []
    for strip in app.screen._compositor.render_strips():
        lines.append(strip.text.rstrip())
    return lines
