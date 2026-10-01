"""Threshold alerts.

An alert here is a *statement about a reading*, not a notification: it is recomputed from
scratch for every sample, so a condition that clears stops being reported the moment it
clears, with no timer to expire and no latch to forget to clear.  That property is what
makes the panel border safe to flash -- it can only ever be flashing because something is
actually wrong right now.

Two design choices are worth stating.

**A missing reading never alerts.**  Every threshold is skipped when the value it guards is
``None``.  A card whose junction sensor is not exposed must not flash forever because it
cannot prove it is cool.

**The worst temperature wins, not the average.**  An average over edge, junction and memory
hides precisely the case that matters: a hotspot well above its neighbours while the mean
sits comfortably below the threshold.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

from gputop.model.metrics import AmdgpuMetrics


class AlertLevel(IntEnum):
    """How bad it is, ordered so that comparisons work.

    Two states, not three.  An earlier draft carried a ``WARN`` between them for readings
    that were merely *approaching* a limit, but that needs a second threshold per metric
    and therefore a second knob for a user to get wrong; nothing ever produced it, and an
    unreachable state with a docstring describing what it would have meant is worse than
    its absence.  If a warning band is wanted later it should arrive as two thresholds per
    metric, not as a level no code sets.

    The ordering is deliberate so ``max()`` over several breaches is the panel's level.
    """

    OK = 0
    ALERT = 1

    @property
    def label(self) -> str:
        """Short display name."""
        return "ok" if self is AlertLevel.OK else "alert"


@dataclass(frozen=True, slots=True, kw_only=True)
class Thresholds:
    """The configured alert thresholds.

    Attributes:
        temp_c: Junction/hotspot temperature in degrees Celsius that counts as ``ALERT``.
        power_percent: Power draw as a percentage of the enforced cap.
        vram_percent: VRAM in use, as a percentage of the pool.
    """

    temp_c: float = 90.0
    power_percent: float = 95.0
    vram_percent: float = 92.0

    @property
    def is_exceeded(self) -> bool:
        """Whether any threshold is at its absolute ceiling, disabling alerting by omission.

        Not used to disable alerting -- that is what ``alerts.enabled`` is for -- but useful
        in the panel to show a threshold is configured at all rather than silently never
        firing.
        """
        return self.temp_c > 0 or self.power_percent > 0 or self.vram_percent > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertBreach:
    """One metric that crossed its threshold."""

    #: What was measured, e.g. ``temp junction``.
    metric: str
    #: The formatted reading, e.g. ``94°C``.
    reading: str
    #: The threshold it crossed, e.g. ``90°C``.
    limit: str
    level: AlertLevel

    def describe(self) -> str:
        """A one-line description, e.g. ``temp junction 94°C > 90°C``."""
        return f"{self.metric} {self.reading} > {self.limit}"


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertState:
    """Every threshold currently crossed for one device."""

    level: AlertLevel = AlertLevel.OK
    breaches: tuple[AlertBreach, ...] = ()

    @property
    def active(self) -> bool:
        """Whether anything is over its limit."""
        return self.level is not AlertLevel.OK

    @property
    def summary(self) -> str:
        """A one-line description naming every breach, or ``""`` when clear."""
        return "; ".join(breach.describe() for breach in self.breaches)

    @classmethod
    def clear(cls) -> AlertState:
        """The state of a device with nothing to report."""
        return cls(level=AlertLevel.OK)


def hottest_temperature(metrics: AmdgpuMetrics) -> tuple[str, float] | None:
    """Return ``(label, celsius)`` for the hottest sensor, or ``None`` when there are none.

    The hot-spot ordering matters less than the tie-break being stable: two sensors reading
    the same temperature must always produce the same label, or the panel title flickers
    between two names for an unchanged reading.
    """
    readings = [(t.label, t.celsius) for t in metrics.temperatures if t.celsius is not None]
    if not readings:
        return None
    return max(readings, key=lambda pair: (pair[1], pair[0]))


def evaluate(metrics: AmdgpuMetrics, thresholds: Thresholds) -> AlertState:
    """Compare one sample against the thresholds.

    Args:
        metrics: The sample to judge.
        thresholds: The configured limits.

    Returns:
        The worst level crossed, with every breach that contributed to it.

    Two readings are treated as ordinary rather than as alerts.  A metric whose value is
    ``None`` is skipped, because a card whose junction sensor is not exposed must not flash
    forever because it cannot prove it is cool.  And a threshold at or below zero disables
    its own check, which is how a user turns off one alert without editing three places --
    ``alerts.temp_c = 0`` means "never tell me about temperature", not "tell me the
    temperature is above zero", which is what a bare ``>=`` comparison would do.
    """
    breaches: list[AlertBreach] = []

    hottest = hottest_temperature(metrics)
    if hottest is not None and thresholds.temp_c > 0 and hottest[1] >= thresholds.temp_c:
        breaches.append(
            AlertBreach(
                metric=f"temp {hottest[0]}",
                reading=f"{hottest[1]:.0f}C",
                limit=f"{thresholds.temp_c:.0f}C",
                level=AlertLevel.ALERT,
            )
        )

    draw = metrics.power.percent
    if draw is not None and thresholds.power_percent > 0 and draw >= thresholds.power_percent:
        breaches.append(
            AlertBreach(
                metric="power",
                reading=f"{draw:.0f}%",
                limit=f"{thresholds.power_percent:.0f}%",
                level=AlertLevel.ALERT,
            )
        )

    vram = metrics.vram.percent
    if vram is not None and thresholds.vram_percent > 0 and vram >= thresholds.vram_percent:
        breaches.append(
            AlertBreach(
                metric="vram",
                reading=f"{vram:.0f}%",
                limit=f"{thresholds.vram_percent:.0f}%",
                level=AlertLevel.ALERT,
            )
        )

    if not breaches:
        return AlertState.clear()
    level = max(breach.level for breach in breaches)
    return AlertState(
        level=level,
        breaches=tuple(breach for breach in breaches if breach.level is level),
    )


def worst(states: Iterable[AlertState]) -> AlertState:
    """Combine several devices' states into the one the border should show.

    Multi-GPU is why this exists: two quiet cards must not be able to hide a third one that
    is over its limit, because the border is a single shared signal.
    """
    collected = [state for state in states if state.active]
    if not collected:
        return AlertState.clear()
    level = max(state.level for state in collected)
    breaches = tuple(breach for state in collected for breach in state.breaches)
    return AlertState(level=level, breaches=breaches)


__all__ = [
    "AlertBreach",
    "AlertLevel",
    "AlertState",
    "Thresholds",
    "evaluate",
    "hottest_temperature",
    "worst",
]
