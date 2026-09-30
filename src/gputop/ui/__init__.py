"""The Textual user interface.

Attribute access is deferred via :pep:`562` so that importing a leaf module such as
:mod:`gputop.ui.theme` does not drag in the whole application.  Without this,
``gputop.config`` cannot validate a theme name without importing the app, which imports
the config right back.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gputop.ui.app import GpuTopApp
    from gputop.ui.widgets import BrailleGraph, CpuMeter

__all__ = ["BrailleGraph", "CpuMeter", "GpuTopApp", "run_app"]


def __getattr__(name: str) -> Any:
    """Resolve the UI entry points on first access."""
    if name in ("GpuTopApp", "run_app"):
        from gputop.ui import app

        return getattr(app, name)
    if name in ("BrailleGraph", "CpuMeter"):
        from gputop.ui import widgets

        return getattr(widgets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
