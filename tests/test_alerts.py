"""Tests for the threshold alert system.

An alert is the one thing in gputop that shouts, so most of what follows is about *not*
shouting: a reading that is missing must stay silent, a reading sitting exactly on the
threshold must fire, and the sensor named in the message must be the hottest one rather
than the first or the mean -- a mean over edge, junction and memory hides precisely the
hotspot that matters.

The power-profile reader carries the opposite guarantee, and it is asserted structurally
rather than by comment: the module is scanned for write operations, and the scanner is
itself tested against a planted write, so a future edit which adds one fails this suite
instead of quietly reprogramming somebody's GPU.
"""

import ast
import io
import os
import re
import tokenize
from dataclasses import fields
from pathlib import Path

import pytest

from gputop.config import AlertsConfig, Config, load_config
from gputop.model.alerts import (
    AlertBreach,
    AlertLevel,
    AlertState,
    Thresholds,
    evaluate,
    hottest_temperature,
    worst,
)
from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import AmdgpuMetrics, MemoryPool, Power, Temperature
from gputop.readers import pp
from tests.conftest import write

GiB = 1024**3


def make_device(**overrides: object) -> AmdgpuDevice:
    """Build a device record."""
    defaults: dict[str, object] = {
        "index": 0,
        "card": "card0",
        "render": "renderD128",
        "bdf": "0000:0c:00.0",
        "vendor_id": 0x1002,
        "device_id": 0x73BF,
        "device_dir": "/sys/devices/fake",
        "hwmon_dir": None,
        "kind": "dgpu",
        "name": "AMD Radeon Test",
        "vram_vendor": "samsung",
        "boot_vga": False,
    }
    defaults.update(overrides)
    return AmdgpuDevice(**defaults)  # type: ignore[arg-type]


def make_metrics(**overrides: object) -> AmdgpuMetrics:
    """Build a metrics record."""
    defaults: dict[str, object] = {
        "device": make_device(),
        "sequence": 1,
        "timestamp_ns": 0,
        "interval_s": 1.0,
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def temps(*readings: tuple[str, float | None]) -> tuple[Temperature, ...]:
    """Build a temperature tuple from ``(label, celsius)`` pairs, in the order given.

    Args:
        readings: One ``(label, celsius)`` pair per sensor, ``celsius`` being ``None`` for
            a sensor this card does not expose.

    Returns:
        The sensors as :class:`~gputop.model.metrics.Temperature` records.
    """
    return tuple(
        Temperature(label=label, celsius=celsius, source="gpu_metrics")
        for label, celsius in readings
    )


def breach(metric: str, level: AlertLevel) -> AlertBreach:
    """A synthetic breach, for combining states no real sample can produce."""
    return AlertBreach(metric=metric, reading="99C", limit="90C", level=level)


class TestEvaluateQuietSample:
    """The common case: nothing is crossed, so nothing is reported."""

    def test_a_bare_device_alerts_nothing(self) -> None:
        """A device that has reported no readings at all must never be guessed about."""
        state = evaluate(make_metrics(), Thresholds())
        assert state.level is AlertLevel.OK
        assert state.breaches == ()
        assert state.active is False
        assert state.summary == ""

    def test_a_comfortable_sample_alerts_nothing(self) -> None:
        state = evaluate(
            make_metrics(
                temperatures=temps(("edge", 49.0), ("junction", 56.0)),
                power=Power(draw_w=45.0, cap_w=211.0, source="gpu_metrics"),
                vram=MemoryPool(used=4 * GiB, total=16 * GiB),
            ),
            Thresholds(),
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()
        assert state.summary == ""

    def test_a_sample_just_under_every_threshold_alerts_nothing(self) -> None:
        """All three limits missed by a hair, so one of them cannot be the near miss."""
        state = evaluate(
            make_metrics(
                temperatures=temps(("junction", 89.0)),
                power=Power(draw_w=94.0, cap_w=100.0),
                vram=MemoryPool(used=91, total=100),
            ),
            Thresholds(),
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()


class TestTemperatureThreshold:
    """The temperature limit is inclusive, to the degree."""

    def test_exactly_on_the_threshold_breaches(self) -> None:
        """``>=`` semantics: a card sitting on its limit is already over it."""
        state = evaluate(
            make_metrics(temperatures=temps(("edge", 90.0))), Thresholds(temp_c=90.0)
        )
        assert state.level is AlertLevel.ALERT
        assert len(state.breaches) == 1
        assert state.breaches[0].metric == "temp edge"
        assert state.breaches[0].reading == "90C"
        assert state.breaches[0].limit == "90C"
        assert state.breaches[0].level is AlertLevel.ALERT

    def test_one_degree_below_does_not(self) -> None:
        state = evaluate(
            make_metrics(temperatures=temps(("edge", 89.0))), Thresholds(temp_c=90.0)
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    @pytest.mark.parametrize(
        ("celsius", "breached"),
        [
            (0.0, False),
            (89.0, False),
            (89.9, False),
            (89.99, False),
            (90.0, True),
            (90.01, True),
            (120.0, True),
        ],
    )
    def test_the_boundary_is_inclusive(self, celsius: float, breached: bool) -> None:
        state = evaluate(
            make_metrics(temperatures=temps(("junction", celsius))), Thresholds(temp_c=90.0)
        )
        assert (state.level is AlertLevel.ALERT) is breached

    def test_the_limit_is_rendered_next_to_the_reading(self) -> None:
        """The user has to be able to see which number to change."""
        state = evaluate(
            make_metrics(temperatures=temps(("junction", 95.0))), Thresholds(temp_c=90.0)
        )
        assert state.summary == "temp junction 95C > 90C"


class TestHottestSensor:
    """The breach names the sensor that is actually hot."""

    def hotspot(self) -> AmdgpuMetrics:
        """One hot junction among two cool neighbours."""
        return make_metrics(
            temperatures=temps(("edge", 50.0), ("junction", 95.0), ("mem", 60.0))
        )

    def test_the_hot_spot_is_named(self) -> None:
        state = evaluate(self.hotspot(), Thresholds(temp_c=90.0))
        assert [b.metric for b in state.breaches] == ["temp junction"]
        assert state.breaches[0].reading == "95C"

    def test_it_is_not_the_first_sensor(self) -> None:
        """``edge`` comes first in the tuple and is the coolest of the three."""
        state = evaluate(self.hotspot(), Thresholds(temp_c=90.0))
        assert state.breaches[0].metric != "temp edge"

    def test_it_is_not_the_mean(self) -> None:
        """The average of 50, 95 and 60 is 68 C: cool, and precisely the wrong answer."""
        assert hottest_temperature(self.hotspot()) == ("junction", 95.0)
        mean = (50.0 + 95.0 + 60.0) / 3
        assert mean == pytest.approx(68.33, abs=0.01)
        state = evaluate(self.hotspot(), Thresholds(temp_c=90.0))
        assert [b.metric for b in state.breaches] == ["temp junction"]

    def test_the_position_in_the_tuple_does_not_matter(self) -> None:
        """Sysfs enumerates sensors in no useful order, so no position may win."""
        readings = (
            (("edge", 50.0), ("junction", 95.0), ("mem", 60.0)),
            (("mem", 60.0), ("edge", 50.0), ("junction", 95.0)),
            (("junction", 95.0), ("mem", 60.0), ("edge", 50.0)),
        )
        for order in readings:
            state = evaluate(make_metrics(temperatures=temps(*order)), Thresholds(temp_c=90.0))
            assert [b.metric for b in state.breaches] == ["temp junction"], order

    def test_a_single_sensor_is_its_own_hottest(self) -> None:
        state = make_metrics(temperatures=temps(("edge", 49.0)))
        assert hottest_temperature(state) == ("edge", 49.0)
        assert hottest_temperature(make_metrics()) is None


class TestTieBreak:
    """Two sensors at the same temperature must not make the panel title flicker."""

    def tied(self) -> AmdgpuMetrics:
        """Edge and junction reading identically."""
        return make_metrics(temperatures=temps(("edge", 90.0), ("junction", 90.0)))

    def test_the_same_label_comes_back_twice(self) -> None:
        first = evaluate(self.tied(), Thresholds(temp_c=90.0))
        second = evaluate(self.tied(), Thresholds(temp_c=90.0))
        assert len(first.breaches) == 1
        assert [b.metric for b in first.breaches] == [b.metric for b in second.breaches]

    def test_the_winner_does_not_depend_on_the_order(self) -> None:
        forward = evaluate(
            make_metrics(temperatures=temps(("edge", 90.0), ("junction", 90.0))),
            Thresholds(temp_c=90.0),
        )
        backward = evaluate(
            make_metrics(temperatures=temps(("junction", 90.0), ("edge", 90.0))),
            Thresholds(temp_c=90.0),
        )
        assert forward.breaches[0].metric == backward.breaches[0].metric

    def test_the_rule_itself_is_pinned(self) -> None:
        """Any deterministic rule satisfies the contract; this one is recorded so that
        changing it shows up as a failing test rather than as a title that flickers once
        and is never noticed.
        """
        state = evaluate(self.tied(), Thresholds(temp_c=90.0))
        assert state.breaches[0].metric == "temp junction"

    def test_a_three_way_tie_is_also_stable(self) -> None:
        readings = temps(("edge", 90.0), ("junction", 90.0), ("mem", 90.0))
        first = evaluate(make_metrics(temperatures=readings), Thresholds(temp_c=90.0))
        second = evaluate(make_metrics(temperatures=readings), Thresholds(temp_c=90.0))
        assert len(first.breaches) == 1
        assert [b.metric for b in first.breaches] == [b.metric for b in second.breaches]


class TestPowerThreshold:
    """Power is a fraction of the cap, so a missing cap is a missing reading."""

    def test_exactly_on_the_threshold_breaches(self) -> None:
        state = evaluate(
            make_metrics(power=Power(draw_w=95.0, cap_w=100.0)),
            Thresholds(power_percent=95.0),
        )
        assert state.level is AlertLevel.ALERT
        assert [(b.metric, b.reading, b.limit) for b in state.breaches] == [
            ("power", "95%", "95%")
        ]

    def test_just_below_does_not(self) -> None:
        state = evaluate(
            make_metrics(power=Power(draw_w=94.0, cap_w=100.0)),
            Thresholds(power_percent=95.0),
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    def test_no_cap_means_no_percentage_and_no_alert(self) -> None:
        """A card that publishes no enforced cap has nothing for a percentage to divide by."""
        power = Power(draw_w=210.0, cap_w=None)
        assert power.percent is None
        state = evaluate(make_metrics(power=power), Thresholds(power_percent=10.0))
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    def test_no_draw_means_no_percentage_and_no_alert(self) -> None:
        power = Power(draw_w=None, cap_w=211.0)
        assert power.percent is None
        state = evaluate(make_metrics(power=power), Thresholds(power_percent=10.0))
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    def test_a_zero_cap_never_divides(self) -> None:
        """Some drivers publish a cap of 0 before the limit has been programmed."""
        power = Power(draw_w=120.0, cap_w=0.0)
        assert power.percent is None
        assert evaluate(make_metrics(power=power), Thresholds()).level is AlertLevel.OK

    def test_a_draw_above_the_cap_is_clamped(self) -> None:
        """Over the cap is 100%, not a 300% reading no threshold could ever reach."""
        assert Power(draw_w=300.0, cap_w=100.0).percent == 100.0
        state = evaluate(
            make_metrics(power=Power(draw_w=300.0, cap_w=100.0)),
            Thresholds(power_percent=95.0),
        )
        assert [b.metric for b in state.breaches] == ["power"]
        assert state.breaches[0].reading == "100%"


class TestVramThreshold:
    """VRAM is a fraction of the pool."""

    def test_exactly_on_the_threshold_breaches(self) -> None:
        state = evaluate(
            make_metrics(vram=MemoryPool(used=92, total=100)),
            Thresholds(vram_percent=92.0),
        )
        assert state.level is AlertLevel.ALERT
        assert [(b.metric, b.reading, b.limit) for b in state.breaches] == [
            ("vram", "92%", "92%")
        ]

    def test_just_below_does_not(self) -> None:
        state = evaluate(
            make_metrics(vram=MemoryPool(used=91, total=100)),
            Thresholds(vram_percent=92.0),
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    def test_a_real_pool_over_the_limit(self) -> None:
        state = evaluate(
            make_metrics(vram=MemoryPool(used=15 * GiB, total=16 * GiB)), Thresholds()
        )
        assert [b.metric for b in state.breaches] == ["vram"]
        assert state.breaches[0].reading == "94%"

    def test_a_pool_with_no_total_never_breaches(self) -> None:
        pool = MemoryPool(used=4 * GiB, total=None)
        assert pool.percent is None
        state = evaluate(make_metrics(vram=pool), Thresholds(vram_percent=1.0))
        assert state.breaches == ()

    def test_an_unreadable_pool_is_not_an_empty_one(self) -> None:
        """``MemoryPool()`` must not read as 0%, which would never alert and always lie."""
        assert MemoryPool().percent is None
        state = evaluate(make_metrics(vram=MemoryPool()), Thresholds(vram_percent=1.0))
        assert state.breaches == ()

    def test_a_zero_sized_pool_never_divides(self) -> None:
        assert MemoryPool(used=0, total=0).percent is None
        empty = evaluate(make_metrics(vram=MemoryPool(used=0, total=0)), Thresholds())
        assert empty.breaches == ()

    def test_an_over_filled_pool_is_clamped(self) -> None:
        assert MemoryPool(used=200, total=100).percent == 100.0


class TestSimultaneousBreaches:
    """Several metrics can be over at once, and all of them have to be reported."""

    def hot(self) -> AmdgpuMetrics:
        """Over on temperature, power and VRAM at the same time."""
        return make_metrics(
            temperatures=temps(("edge", 50.0), ("junction", 95.0)),
            power=Power(draw_w=205.0, cap_w=210.0),
            vram=MemoryPool(used=95, total=100),
        )

    def test_all_three_are_kept(self) -> None:
        state = evaluate(self.hot(), Thresholds())
        assert state.level is AlertLevel.ALERT
        assert sorted(b.metric for b in state.breaches) == ["power", "temp junction", "vram"]

    def test_every_kept_breach_shares_the_reported_level(self) -> None:
        """The list and the level can never disagree about how bad this is."""
        state = evaluate(self.hot(), Thresholds())
        assert len(state.breaches) == 3
        assert all(b.level is state.level for b in state.breaches)
        assert state.level is max(b.level for b in state.breaches)

    def test_only_the_crossed_ones_are_kept(self) -> None:
        """Lifting the VRAM limit out of reach must drop the VRAM breach, not mute it."""
        state = evaluate(self.hot(), Thresholds(vram_percent=100.0))
        assert sorted(b.metric for b in state.breaches) == ["power", "temp junction"]

    def test_the_summary_names_every_breach(self) -> None:
        state = evaluate(self.hot(), Thresholds())
        assert state.summary == "temp junction 95C > 90C; power 98% > 95%; vram 95% > 92%"

    def test_the_summary_is_empty_when_nothing_is_over(self) -> None:
        state = evaluate(
            make_metrics(
                temperatures=temps(("junction", 50.0)),
                power=Power(draw_w=10.0, cap_w=210.0),
                vram=MemoryPool(used=1 * GiB, total=16 * GiB),
            ),
            Thresholds(),
        )
        assert state.summary == ""


class TestMissingReadings:
    """Absence is silence, never a value of zero."""

    #: The most sensitive threshold that is still switched on, used to prove the silence
    #: below comes from the reading and not from an unreachable limit.
    KEEN = Thresholds(temp_c=0.1, power_percent=0.1, vram_percent=0.1)

    def test_a_none_temperature_never_alerts(self) -> None:
        """Same thresholds on both sides; the only difference is the reading."""
        missing = make_metrics(temperatures=temps(("edge", None), ("junction", None)))
        present = make_metrics(temperatures=temps(("edge", 0.2), ("junction", 0.3)))
        assert evaluate(missing, self.KEEN).breaches == ()
        assert [b.metric for b in evaluate(present, self.KEEN).breaches] == ["temp junction"]

    def test_no_sensors_at_all_is_silence(self) -> None:
        assert hottest_temperature(make_metrics()) is None
        state = evaluate(make_metrics(), self.KEEN)
        assert state.breaches == ()

    def test_a_missing_sensor_does_not_mask_a_hot_one(self) -> None:
        """The junction is absent and the edge is at 95 C: the edge must still be named."""
        state = evaluate(
            make_metrics(temperatures=temps(("junction", None), ("edge", 95.0))),
            Thresholds(temp_c=90.0),
        )
        assert [b.metric for b in state.breaches] == ["temp edge"]

    def test_one_readable_sensor_alerts_on_its_own(self) -> None:
        """A card exposing only edge must alert on edge, not stay quiet for want of a
        second opinion.
        """
        state = evaluate(
            make_metrics(temperatures=temps(("edge", 95.0))), Thresholds(temp_c=90.0)
        )
        assert [b.metric for b in state.breaches] == ["temp edge"]

    def test_a_device_with_nothing_readable_never_alerts(self) -> None:
        """Every guarded value missing at once, against limits nothing could miss: an
        empty card must be silent while the identical limits on a populated one are not.
        """
        populated = make_metrics(
            temperatures=temps(("junction", 95.0)),
            power=Power(draw_w=205.0, cap_w=210.0),
            vram=MemoryPool(used=95, total=100),
        )
        empty = evaluate(make_metrics(), self.KEEN)
        assert empty.level is AlertLevel.OK
        assert empty.breaches == ()
        assert len(evaluate(populated, self.KEEN).breaches) == 3


class TestZeroDisables:
    """A threshold of ``0`` switches off its own check rather than firing on everything.

    The difference matters: a bare ``>=`` against zero would alert on every sample, so
    "turn the temperature alert off" would instead mean "flash constantly".
    """

    def hot(self) -> AmdgpuMetrics:
        """Over on all three metrics, by a wide margin."""
        return make_metrics(
            temperatures=temps(("edge", 50.0), ("junction", 95.0)),
            power=Power(draw_w=205.0, cap_w=210.0),
            vram=MemoryPool(used=95, total=100),
        )

    def test_all_three_disabled_is_silence(self) -> None:
        state = evaluate(
            self.hot(), Thresholds(temp_c=0.0, power_percent=0.0, vram_percent=0.0)
        )
        assert state.level is AlertLevel.OK
        assert state.breaches == ()

    def test_disabling_one_leaves_the_others_alerting(self) -> None:
        state = evaluate(self.hot(), Thresholds(temp_c=0.0))
        assert sorted(b.metric for b in state.breaches) == ["power", "vram"]

    def test_a_switched_off_threshold_still_shows_in_the_panel(self) -> None:
        """``is_exceeded`` is how the UI knows a threshold is off rather than unset."""
        assert Thresholds(temp_c=0.0).is_exceeded is True
        assert Thresholds(temp_c=0.0, power_percent=0.0, vram_percent=0.0).is_exceeded is False

    def test_a_configured_zero_reaches_the_model_and_disables(self, tmp_path: Path) -> None:
        """End to end: the number a user types is the one that switches the alert off."""
        config = load_config(
            write(tmp_path / "gputop.toml", "[alerts]\ntemp_c = 0\npower_percent = 0\n")
        )
        assert config.alerts.temp_c == 0.0
        assert config.warnings == ()
        state = evaluate(self.hot(), config.alerts.thresholds())
        assert [b.metric for b in state.breaches] == ["vram"]


class TestWorst:
    """One shared border for every device, so no card can hide behind another."""

    def test_no_devices_is_ok(self) -> None:
        state = worst([])
        assert state.level is AlertLevel.OK
        assert state.breaches == ()
        assert state.active is False
        assert state.summary == ""

    def test_an_empty_generator_is_ok(self) -> None:
        assert worst(state for state in []).level is AlertLevel.OK

    def test_every_quiet_device_is_ok(self) -> None:
        assert worst([AlertState.clear() for _ in range(3)]).level is AlertLevel.OK

    def test_the_highest_level_wins(self) -> None:
        """``max`` over the devices is the level the border shows."""
        states = [
            AlertState.clear(),
            AlertState(level=AlertLevel.ALERT, breaches=(breach("power", AlertLevel.ALERT),)),
            AlertState(
                level=AlertLevel.ALERT,
                breaches=(breach("temp junction", AlertLevel.ALERT),),
            ),
        ]
        assert worst(states).level is AlertLevel.ALERT

    def test_a_quiet_gpu_cannot_hide_an_overheating_one(self) -> None:
        """The card nobody is looking at can be the one that is burning."""
        hot = evaluate(make_metrics(temperatures=temps(("junction", 95.0))), Thresholds())
        combined = worst([AlertState.clear(), AlertState.clear(), hot])
        assert combined.level is AlertLevel.ALERT
        assert [b.metric for b in combined.breaches] == ["temp junction"]

    def test_breaches_from_every_device_are_kept(self) -> None:
        hot = evaluate(make_metrics(temperatures=temps(("junction", 95.0))), Thresholds())
        loaded = evaluate(
            make_metrics(
                power=Power(draw_w=205.0, cap_w=210.0),
                vram=MemoryPool(used=95, total=100),
            ),
            Thresholds(),
        )
        combined = worst([hot, loaded])
        assert combined.level is AlertLevel.ALERT
        assert sorted(b.metric for b in combined.breaches) == ["power", "temp junction", "vram"]

    def test_the_summary_names_every_breach(self) -> None:
        states = [
            AlertState(
                level=AlertLevel.ALERT, breaches=(breach("temp junction", AlertLevel.ALERT),)
            ),
            AlertState(level=AlertLevel.ALERT, breaches=(breach("power", AlertLevel.ALERT),)),
        ]
        assert worst(states).summary == "temp junction 99C > 90C; power 99C > 90C"

    def test_an_inactive_device_contributes_nothing(self) -> None:
        """``active`` is decided by the level, so a clear state adds nothing at all."""
        quiet = AlertState(
            level=AlertLevel.OK, breaches=(breach("temp edge", AlertLevel.ALERT),)
        )
        combined = worst([quiet, AlertState.clear()])
        assert combined.level is AlertLevel.OK
        assert combined.breaches == ()

    def test_a_single_breached_machine_is_active_on_its_own(self) -> None:
        """One card over its limit is enough; the others do not have to agree."""
        state = AlertState(
            level=AlertLevel.ALERT, breaches=(breach("power", AlertLevel.ALERT),)
        )
        assert state.active is True
        assert worst([state]).level is AlertLevel.ALERT

    def test_there_is_only_one_breach_level(self) -> None:
        """``WARN`` was removed rather than left as a state nothing could produce.

        A level with no producer is a level whose colour branch is dead code and whose
        docstring describes behaviour that does not exist.  Pinning the count here means
        adding one back is a deliberate act rather than an accident.
        """
        assert [level.name for level in AlertLevel] == ["OK", "ALERT"]


class TestThresholdsRecord:
    """The config record and the model record have to carry the same numbers."""

    def test_the_config_produces_the_model_record(self) -> None:
        config = AlertsConfig(temp_c=70.0, power_percent=80.0, vram_percent=60.0)
        assert config.thresholds() == Thresholds(
            temp_c=70.0, power_percent=80.0, vram_percent=60.0
        )

    def test_the_defaults_agree(self) -> None:
        assert AlertsConfig().thresholds() == Thresholds()
        assert Thresholds() == Thresholds(temp_c=90.0, power_percent=95.0, vram_percent=92.0)

    def test_the_flash_rate_stays_in_the_config(self) -> None:
        """``flash_hz`` is a presentation choice, so it never reaches the model."""
        assert "flash_hz" in {f.name for f in fields(AlertsConfig)}
        assert "flash_hz" not in {f.name for f in fields(Thresholds)}
        assert AlertsConfig(flash_hz=3.0).thresholds() == Thresholds()

    def test_is_exceeded(self) -> None:
        assert Thresholds().is_exceeded is True
        assert Thresholds(temp_c=0.0, power_percent=0.0, vram_percent=0.0).is_exceeded is False

    def test_the_numbers_a_user_types_are_the_numbers_judged_by(self) -> None:
        thresholds = AlertsConfig(temp_c=70.0).thresholds()
        state = evaluate(make_metrics(temperatures=temps(("junction", 71.0))), thresholds)
        assert [b.metric for b in state.breaches] == ["temp junction"]
        assert state.breaches[0].limit == "70C"


class TestAlertsConfigLoading:
    """A hand-written threshold is clamped into range, not rejected."""

    def load(self, tmp_path: Path, body: str) -> Config:
        """Write ``body`` as a config file and load it."""
        return load_config(write(tmp_path / "gputop.toml", body))

    def out_of_range(self, tmp_path: Path) -> Config:
        """Every alert field, one past the end of its range."""
        return self.load(
            tmp_path,
            "[alerts]\n"
            "temp_c = 500.0\n"
            "power_percent = -5.0\n"
            "vram_percent = 250\n"
            "flash_hz = 99.0\n",
        )

    def test_each_value_is_clamped(self, tmp_path: Path) -> None:
        alerts = self.out_of_range(tmp_path).alerts
        assert alerts.temp_c == 125.0
        assert alerts.power_percent == 0.0
        assert alerts.vram_percent == 100.0
        assert alerts.flash_hz == 20.0

    def test_every_clamp_is_explained(self, tmp_path: Path) -> None:
        """A silently altered threshold is a threshold the user cannot trust."""
        config = self.out_of_range(tmp_path)
        assert {w.split("=", 1)[0] for w in config.warnings} == {
            "alerts.temp_c",
            "alerts.power_percent",
            "alerts.vram_percent",
            "alerts.flash_hz",
        }
        assert all("out of range" in w for w in config.warnings)

    def test_a_clamped_config_still_feeds_the_model(self, tmp_path: Path) -> None:
        config = self.out_of_range(tmp_path)
        assert config.alerts.thresholds() == Thresholds(
            temp_c=125.0, power_percent=0.0, vram_percent=100.0
        )

    def test_flash_hz_zero_is_accepted(self, tmp_path: Path) -> None:
        """0 Hz means a steady border, not a value out of range."""
        config = self.load(tmp_path, "[alerts]\nflash_hz = 0\n")
        assert config.alerts.flash_hz == 0.0
        assert config.warnings == ()

    def test_a_zero_threshold_is_in_range_too(self, tmp_path: Path) -> None:
        config = self.load(
            tmp_path, "[alerts]\ntemp_c = 0\npower_percent = 0\nvram_percent = 0\n"
        )
        assert config.alerts.temp_c == 0.0
        assert config.alerts.power_percent == 0.0
        assert config.alerts.vram_percent == 0.0
        assert config.warnings == ()

    def test_in_range_values_pass_through_untouched(self, tmp_path: Path) -> None:
        config = self.load(
            tmp_path,
            "[alerts]\ntemp_c = 80\npower_percent = 70\nvram_percent = 60\nflash_hz = 2.5\n",
        )
        assert config.alerts.temp_c == 80.0
        assert config.alerts.flash_hz == 2.5
        assert config.alerts.thresholds() == Thresholds(
            temp_c=80.0, power_percent=70.0, vram_percent=60.0
        )
        assert config.warnings == ()

    def test_alerting_can_be_switched_off(self, tmp_path: Path) -> None:
        config = self.load(tmp_path, "[alerts]\nenabled = false\n")
        assert config.alerts.enabled is False
        assert config.warnings == ()

    def test_an_unknown_key_is_a_warning_and_is_ignored(self, tmp_path: Path) -> None:
        """A config written for a future release must still work today."""
        config = self.load(tmp_path, "[alerts]\ntemp_c = 80\nfrom_the_future = 1\n")
        assert config.alerts.temp_c == 80.0
        assert getattr(config.alerts, "from_the_future", None) is None
        assert any("alerts.from_the_future" in w for w in config.warnings)


class TestPpReadOnly:
    """``pp_power_profile_mode`` and ``pp_od_clk_voltage`` are readable, never writable."""

    def test_the_module_asserts_its_own_rule(self) -> None:
        assert pp.assert_read_only() is True

    def test_the_source_contains_no_write_operation(self) -> None:
        violations = scan_for_writes(Path(pp.__file__).read_text())
        assert violations == [], violations

    def test_the_module_never_imports_a_writer(self) -> None:
        """The other way into a write is a library that does one for you."""
        imported = _imported_names(Path(pp.__file__).read_text())
        assert {
            n for n in imported if n.split(".")[0] in {"os", "shutil", "subprocess"}
        } == set()

    def test_the_module_reads_through_the_shared_reader(self) -> None:
        """Reads go through one audited helper, so a fresh ``open()`` has to be justified."""
        source = Path(pp.__file__).read_text()
        # The whole import list, not just "is ``read_text`` there": a check that only
        # tests for one name passes just as happily when something else arrives beside it.
        assert _imported_from(source, "gputop.readers.fsutil") == {
            "PROFILE_ROW",
            "TABLE_LIMIT",
            "read_text",
        }
        assert "open(" not in source

    def test_the_module_can_still_read(self, tmp_path: Path) -> None:
        """A module that did nothing at all would also pass the scan, so prove it reads."""
        write(tmp_path / "pp_power_profile_mode", "0 BOOTUP_DEFAULT\n2 3D_FULL_SCREEN*\n")
        table = pp.read_power_profile_table(tmp_path)
        assert table.active_name == "3D_FULL_SCREEN"
        assert table.present is True


# ---------------------------------------------------------------------------
# The read-only scanner
# ---------------------------------------------------------------------------

#: Method names that mutate a file, matched on the last dotted segment so that
#: ``path.write_text`` and a bare ``write_text`` are both caught.  ``replace``, ``compile``
#: and the ``read_*`` family are deliberately absent: they are pure text operations, and
#: flagging them is exactly the false positive that gets a scanner switched off.  The
#: filesystem spellings of ``replace`` are matched by name in :data:`_WRITE_FUNCTIONS`.
_WRITE_NAMES = frozenset(
    {
        "chmod",
        "chown",
        "hardlink_to",
        "link_to",
        "mkdir",
        "makedirs",
        "mkstemp",
        "rename",
        "rmdir",
        "symlink_to",
        "touch",
        "truncate",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
)

#: Fully qualified writers, matched exactly.  These are the names whose last segment
#: ("remove", "system", "rmtree") is too ordinary to condemn on sight.
_WRITE_FUNCTIONS = frozenset(
    {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.makedirs",
        "os.remove",
        "os.removedirs",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "os.write",
        "pathlib.Path.chmod",
        "pathlib.Path.mkdir",
        "pathlib.Path.rename",
        "pathlib.Path.replace",
        "pathlib.Path.rmdir",
        "pathlib.Path.symlink_to",
        "pathlib.Path.touch",
        "pathlib.Path.truncate",
        "pathlib.Path.unlink",
        "pathlib.Path.write_bytes",
        "pathlib.Path.write_text",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.move",
        "shutil.rmtree",
    }
)

#: Process launchers and dynamic code execution, by exact name.  ``compile`` sits here
#: rather than in :data:`_WRITE_NAMES` because ``re.compile`` is a parser's best friend.
_LAUNCHERS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "os.execl",
        "os.execv",
        "os.execve",
        "os.popen",
        "os.spawnl",
        "os.spawnv",
        "os.spawnve",
        "os.system",
    }
)

#: ``os.open`` flags that ask for anything other than reading.
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

#: Characters that make a mode a writing mode: ``w`` write, ``a`` append, ``x`` create,
#: ``+`` update.
_WRITE_MODE_CHARS = frozenset("wax+")

#: ``open`` spellings that are functions taking the path first, as opposed to a method on
#: an object that already holds one.
_FUNCTIONS_WITH_A_PATH = frozenset({"open", "codecs.open", "gzip.open", "io.open"})

#: A ``>`` in a string followed by the start of a path is a shell redirect.  Requiring the
#: target to look like a path is what keeps ordinary prose -- "``95C > 90C``" -- from being
#: mistaken for one.
_REDIRECT_IN_STRING = re.compile(r">>?\s*[/~'\"]")


def _dotted(node: ast.expr) -> str:
    """Render a call target as a dotted name, e.g. ``os.remove`` or ``Path.write_text``."""
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _qualified(target: str) -> str:
    """Spell a target with its module, so ``Path.write_text`` matches ``pathlib.Path``."""
    return f"pathlib.{target}" if target.startswith("Path.") else target


def _literal(node: ast.expr) -> object:
    """The value of a constant expression, or ``None`` when it is not constant."""
    return node.value if isinstance(node, ast.Constant) else None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    """The expression passed as ``name=`` at a call site, if any."""
    return next((kw.value for kw in node.keywords if kw.arg == name), None)


def _os_constant(name: str) -> int | None:
    """The integer a dotted constant such as ``os.O_RDONLY`` names, if it is one."""
    module, _, attribute = name.rpartition(".")
    if module != "os":
        return None
    value = getattr(os, attribute, None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _static_int(node: ast.expr) -> int | None:
    """Resolve an integer built from literals and ``os.O_*`` constants.

    Args:
        node: The expression to resolve.

    Returns:
        The value, or ``None`` when it depends on something that is not a constant.  A
        flag this test cannot read is not a flag it is willing to call read-only.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return None if isinstance(node.value, bool) else node.value
    if isinstance(node, ast.Attribute):
        return _os_constant(_dotted(node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _static_int(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd)):
        left, right = _static_int(node.left), _static_int(node.right)
        if left is None or right is None:
            return None
        return left | right if isinstance(node.op, ast.BitOr) else left & right
    return None


def _open_mode(node: ast.Call, target: str) -> str | None:
    """The mode an ``open`` call uses, or ``None`` when it is not statically readable.

    The two spellings disagree about argument order, so the index differs: a bare
    ``open(path, "w")`` takes the path first, while ``path.open("w")`` is already
    holding the path and takes the mode first.
    """
    explicit = _keyword(node, "mode")
    if explicit is not None:
        value = _literal(explicit)
        return value if isinstance(value, str) else None
    index = 1 if target in _FUNCTIONS_WITH_A_PATH else 0
    if len(node.args) > index:
        value = _literal(node.args[index])
        return value if isinstance(value, str) else None
    return "r"


def _open_flags(node: ast.Call) -> int | None:
    """The flags an ``os.open`` call uses, or ``None`` when they are not readable."""
    explicit = _keyword(node, "flags")
    raw = explicit if explicit is not None else (node.args[1] if len(node.args) > 1 else None)
    return os.O_RDONLY if raw is None else _static_int(raw)


def _imported_names(source: str) -> set[str]:
    """Every module and symbol a module imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.name for alias in node.names)
    return names


def _imported_from(source: str, module: str) -> set[str]:
    """The names a module imports from one specific module."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _stray_angles(source: str) -> list[int]:
    """Line numbers of a ``>`` in code that cannot be a comparison.

    Python has no redirect operator, so a ``>`` is a comparison or a syntax error.  This
    cannot fire on valid source today and is kept as a cheap guard rather than relied on
    for anything; the reachable version of the same check is :data:`_REDIRECT_IN_STRING`,
    because a shell redirect can only reach a file from inside a string.
    """
    operand_types = frozenset(
        {tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.FSTRING_END}
    )
    operand_ops = frozenset({")", "]", "}"})
    skip = frozenset({tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE})
    found: list[int] = []
    previous: tokenize.TokenInfo | None = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in skip:
            continue
        if token.type == tokenize.OP and token.string in {">", ">>", ">>="}:
            ends_operand = previous is not None and (
                previous.type in operand_types
                or (previous.type == tokenize.OP and previous.string in operand_ops)
            )
            if not ends_operand:
                found.append(token.start[0])
        previous = token
    return found


def scan_for_writes(source: str) -> list[str]:
    """Everything in ``source`` that could write to a file, or shell out to do so.

    Args:
        source: Python source text.

    Returns:
        One line per defect found, empty when the source only reads.  Each entry is a
        specific thing rather than a smell: a filesystem mutator, a writing file mode, a
        write flag on ``os.open``, a process launcher, or a ``>`` redirect aimed at a path.
        A module that merely *talks* about writing -- in a docstring, a comment, or a
        string constant that is not a command -- is not flagged, because prose is not a
        syscall and a scanner that condemns prose is a scanner that gets switched off.
    """
    violations: list[str] = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _dotted(node.func)
        leaf = target.rpartition(".")[2]
        line = node.lineno

        if leaf in _WRITE_NAMES or _qualified(target) in _WRITE_FUNCTIONS:
            violations.append(f"line {line}: writes through {target}()")
            continue
        if target in _LAUNCHERS or target.startswith("subprocess."):
            violations.append(f"line {line}: launches a process through {target}()")
            continue
        if target == "os.open":
            flags = _open_flags(node)
            if flags is None:
                violations.append(f"line {line}: os.open with unreadable flags")
            elif flags & _WRITE_FLAGS:
                violations.append(f"line {line}: os.open with the write flag {flags:#x}")
            continue
        if leaf == "open":
            mode = _open_mode(node, target)
            if mode is None:
                violations.append(f"line {line}: {target}() with an unreadable mode")
            elif _WRITE_MODE_CHARS & set(mode):
                violations.append(f"line {line}: {target}(..., {mode!r})")

    violations += [f"line {line}: a '>' outside a comparison" for line in _stray_angles(source)]

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _REDIRECT_IN_STRING.search(node.value)
        ):
            violations.append(f"line {node.lineno}: a '>' redirect in {node.value!r}")
    return violations


class TestScanForWrites:
    """The scanner itself, so the read-only guarantee does not rest on a check that cannot
    fail.
    """

    #: A module that imports every writer the scanner knows how to recognise.
    HEADER = "import os\nimport shutil\nimport subprocess\nfrom pathlib import Path\n\n"

    def plant(self, snippet: str) -> list[str]:
        """Compile one statement into a module and scan it."""
        return scan_for_writes(
            f"{self.HEADER}def f(path, other, handle, lines, source):\n    {snippet}\n"
        )

    @pytest.mark.parametrize(
        "snippet",
        [
            "path.write_text('x')",
            "path.write_bytes(b'x')",
            "handle.writelines(lines)",
            "handle.write('x')",
            "path.unlink()",
            "path.mkdir()",
            "path.touch()",
            "path.rename(other)",
            "Path.replace(other)",
            "os.remove(path)",
            "os.replace(path, other)",
            "os.truncate(path, 0)",
            "shutil.rmtree(path)",
            "shutil.copy(path, other)",
            "open(path, 'w')",
            "open(path, mode='a+')",
            "path.open('wb')",
            "os.open(path, os.O_WRONLY | os.O_CREAT)",
            "os.open(path, os.O_RDWR)",
            "subprocess.run(['sh', '-c', 'cat > /sys/class/drm/card0/device/pp_od_clk'])",
            "os.system('echo 2 > /sys/class/drm/card0/device/pp_power_profile_mode')",
            "eval(compile(source, '<s>', 'exec'))",
            "command = 'cat > /sys/class/drm/card0/device/pp_od_clk_voltage'",
        ],
    )
    def test_it_catches_a_write(self, snippet: str) -> None:
        assert self.plant(snippet) != [], snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            'path.read_text(encoding="utf-8")',
            "path.read_bytes()",
            'open(path, "rb")',
            "open(path)",
            "path.open('r', encoding='utf-8')",
            "os.open(path, os.O_RDONLY)",
            "os.open(path, os.O_RDONLY | os.O_CLOEXEC)",
            "os.stat(path).st_size > 0",
            "if len(lines) > 1 and '>' in text: pass",
            'text = "a > b"',
            'text.replace("a", "b")',
            "os.replace  # a comment naming a writer is not one",
        ],
    )
    def test_it_does_not_condemn_a_read(self, snippet: str) -> None:
        header = "import os\nfrom pathlib import Path\n\n"
        planted = f"{header}def f(path, lines, text):\n    {snippet}\n    return 0\n"
        assert scan_for_writes(planted) == [], snippet

    def test_prose_about_writing_is_allowed(self) -> None:
        """A docstring may name the very thing the module forbids."""
        source = (
            "from pathlib import Path\n\n"
            "def read(path: Path) -> str:\n"
            '    """Read a file.  This module never calls write_text or open(path, "w")."""\n'
            "    # Nor write_bytes(), nor os.open(path, os.O_WRONLY).\n"
            "    return path.read_text()\n"
        )
        assert scan_for_writes(source) == []
