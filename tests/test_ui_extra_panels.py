"""App-level tests for the panels added after milestone 3.

These mount the real Textual app against a fake sysfs tree rather than calling the text
builders directly, because what is being checked here is layout and lifetime: that a panel
with nothing to say is hidden rather than empty, that the alert border changes and changes
back, and that a session recording is closed on the way out.

No test here starts a real ``radeontop``.  The subprocess itself is covered in
``tests/test_radeontop.py``; what these tests own is the seam from a published sample to the
screen, and the parts of the app that must behave when there is no sample at all.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot

from gputop.config import AlertsConfig, BlocksConfig, Config, LogConfig
from gputop.model.blocks import BlocksStatus
from gputop.sampler import Sampler, SamplerOptions
from gputop.sessionlog import read_log
from gputop.ui.app import (
    MIN_HEIGHT_FOR_OPTIONAL_ROW,
    MIN_HEIGHT_FOR_SENSORS_COMPLETE,
    GpuTopApp,
)
from tests.snapshot import run_app, screen_lines
from tests.test_ui_app import build_hardware
from tests.test_ui_app import make_config as base_config

#: Wide and tall enough that no responsive rule hides the optional row.
ROOMY = (160, 50)


def make_app(
    drm_root: Path,
    proc_root: Path,
    *,
    config: Config | None = None,
    log_target: Path | None = None,
    size: tuple[int, int] = ROOMY,
) -> GpuTopApp:
    """Build an app bound to the fake trees."""
    build_hardware(drm_root, proc_root)
    return GpuTopApp(
        config if config is not None else base_config(),
        SamplerOptions(drm_root=drm_root, proc_root=proc_root, interval_s=0.1),
        log_target=log_target,
    )


async def _until(predicate: Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    """Poll ``predicate`` until it holds, so the test waits on state and not on a sleep."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def render[T](
    app: GpuTopApp,
    body: Callable[[Pilot[Any]], Awaitable[T]],
    size: tuple[int, int] = ROOMY,
) -> T:
    """Mount the app, let it settle, and return ``body``'s result."""
    return asyncio.run(run_app(app, size=size, body=body, settle=2))


def screen_of(app: GpuTopApp, size: tuple[int, int] = ROOMY) -> str:
    """The rendered screen as plain text."""

    async def body(_pilot: Pilot[Any]) -> str:
        return "\n".join(screen_lines(app, size[0]))

    return render(app, body, size)


def with_config(**sections: Any) -> Config:
    """The standard test config with individual sections replaced."""
    config = base_config()
    return replace(config, **sections)


class TestOptionalRowIsHiddenWithoutBlocks:
    """A panel with nothing to say must not appear at all."""

    def test_neither_optional_panel_is_on_screen_by_default(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Blocks is off by default, so the row has nothing to draw."""
        app = make_app(drm_root, proc_root)
        text = screen_of(app)
        assert "Blocks ·" not in text
        assert "Shader Interpolator" not in text

    def test_a_disabled_panel_is_silent_rather_than_explained(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The user turned it off. Reminding them every second teaches them to ignore it."""
        app = make_app(
            drm_root, proc_root, config=with_config(blocks=BlocksConfig(enabled=False))
        )
        assert "blocks" not in screen_of(app)

    def test_the_blocks_panel_reserves_no_space_when_hidden(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The blocks panel needs a privileged child that is off by default."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> bool:
            return app.query_one("#blocks-panel").display

        assert render(app, body) is False

    def test_the_profile_panel_appears_without_any_radeontop(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The two panels are independent, and this is the reason that matters.

        The profile panel is read from sysfs and works on any card that publishes SCPP.  It
        was once gated on the blocks panel, so on every machine without a privileged
        radeontop -- which is every machine running gputop unprivileged -- a user who had
        every right to see the power profile silently got nothing.
        """
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> tuple[bool, bool, bool]:
            assert app.query_one("#blocks-panel").display is False
            return (
                app.query_one("#profile-panel").display,
                app.query_one("#extra-row").display,
                app._sampler.devices != (),
            )

        profile, row, _has_devices = render(app, body)
        assert profile is True
        assert row is True

    #: The readings the sensor panel exists for.  A card reporting three temperatures and
    #: a voltage rail produces more rows than fit at this height even without the optional
    #: row, and the panel scrolls; what must not happen is the *primary* readings being the
    #: ones that disappear when an optional panel is switched on.
    PRIMARY_SENSOR_READINGS = (
        "Core",
        "Memory",
        "SoC",
        "temp edge",
        "temp junction",
        "Power",
        "cap 211 W",
    )

    @pytest.mark.parametrize(
        "height",
        [MIN_HEIGHT_FOR_OPTIONAL_ROW, MIN_HEIGHT_FOR_OPTIONAL_ROW + 6],
    )
    def test_the_optional_row_never_costs_a_primary_sensor_reading(
        self, drm_root: Path, proc_root: Path, height: int
    ) -> None:
        """The invariant behind ``MIN_HEIGHT_FOR_OPTIONAL_ROW``.

        The sensor panel scrolls, so a row added below the main one does not lose its
        content -- it pushes the last lines out of view.  The last lines are the trailing
        facts (PCIe link, throttle reasons); the first are the clocks, temperatures and
        power.  Adding an optional panel must never cost the second group, which is why the
        threshold is measured against the sensor panel and not against the optional one.
        """
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return "\n".join(screen_lines(app, 120))

        text = render(app, body, size=(120, height))
        for expected in self.PRIMARY_SENSOR_READINGS:
            assert expected in text, f"{expected!r} scrolled off at height {height}"

    @pytest.mark.parametrize(
        "height",
        [MIN_HEIGHT_FOR_SENSORS_COMPLETE - 2, MIN_HEIGHT_FOR_SENSORS_COMPLETE + 4],
    )
    def test_the_sensor_panel_is_complete_on_a_tall_enough_terminal(
        self, drm_root: Path, proc_root: Path, height: int
    ) -> None:
        """Pins the constant the optional row's threshold is derived from."""
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return "\n".join(screen_lines(app, 120))

        text = render(app, body, size=(120, height))
        complete = all(m in text for m in ("Gen3 x16", "throttle none", "cap 211 W"))
        assert complete is (height >= MIN_HEIGHT_FOR_SENSORS_COMPLETE)

    def test_both_optional_panels_wait_for_the_same_height(self) -> None:
        """One threshold, because one row: either it is affordable or it is not."""
        assert MIN_HEIGHT_FOR_OPTIONAL_ROW > MIN_HEIGHT_FOR_SENSORS_COMPLETE


class TestSessionRecording:
    """``--log`` at the app level: opened on mount, closed on unmount."""

    def test_a_session_is_recorded_to_a_csv(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "session.csv"
        screen_of(make_app(drm_root, proc_root, log_target=target))
        lines = list(read_log(target))
        assert lines[0].startswith("time,sequence")
        assert len(lines) > 1

    def test_a_compressed_session_is_closed_into_a_valid_frame(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        """An unclosed ``.zst`` stream is a truncated archive rather than a readable file."""
        target = tmp_path / "session.csv.zst"
        screen_of(make_app(drm_root, proc_root, log_target=target))
        assert target.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
        assert len(list(read_log(target))) > 1

    def test_no_target_means_no_recorder(self, drm_root: Path, proc_root: Path) -> None:
        app = make_app(drm_root, proc_root)
        screen_of(app)
        assert app._recorder is None

    def test_an_unwritable_target_is_a_notice_rather_than_a_crash(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "a-regular-file"
        blocker.write_text("not a directory")
        app = make_app(drm_root, proc_root, log_target=blocker / "inside" / "s.csv")
        text = screen_of(app)
        assert "cannot record" in text
        assert app._recorder is None

    def test_log_interval_thins_a_fast_sampler(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        """A 100 ms sampler writes 36 000 rows an hour; most are the same idle card.

        ``log.interval_s`` is the knob for that, and it has to actually thin the stream
        rather than being a documented setting that does nothing.
        """
        target = tmp_path / "thin.csv"
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(log=LogConfig(interval_s=30.0)),
            log_target=target,
        )
        screen_of(app)
        # The run lasts a couple of seconds, so a 30 s floor allows exactly one record.
        assert len(list(read_log(target))) == 2  # header plus one row

    def test_a_zero_interval_records_every_sample(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "all.csv"
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(log=LogConfig(interval_s=0.0)),
            log_target=target,
        )
        screen_of(app)
        assert len(list(read_log(target))) > 2

    def test_the_recorder_is_closed_after_the_run(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        app = make_app(drm_root, proc_root, log_target=tmp_path / "s.csv")
        screen_of(app)
        assert app._recorder is None  # cleared by _close_recorder on unmount


class TestAlertBorders:
    """The threshold flash, driven by real samples from a fake card."""

    #: The fixture card's hottest sensor is ``junction`` at 56 C, so a threshold below that
    #: is a card that is over its limit.  Set as a threshold rather than by rewriting the
    #: fixture because ``gpu_metrics`` outranks hwmon -- a test that wrote a hotter sensor
    #: into hwmon would still see the binary table's 56 C and wonder why nothing fired.
    HOT_C = 50.0

    def test_a_breached_threshold_is_reported_and_lights_the_border(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(alerts=AlertsConfig(enabled=True, temp_c=self.HOT_C)),
        )

        async def body(_pilot: Pilot[Any]) -> tuple[bool, str, str]:
            # The phase is pinned rather than left as the flash timer found it, so this
            # asserts the lit colour specifically rather than either end of the cycle.
            app._flash_on = True
            return (
                app._alert_state.active,
                app._alert_state.summary,
                app._alert_border_colour(),
            )

        active, summary, colour = render(app, body)
        assert active
        assert "temp junction" in summary
        assert colour == app._theme.alert

    def test_the_border_alternates_rather_than_sitting_on_one_colour(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(alerts=AlertsConfig(enabled=True, temp_c=self.HOT_C)),
        )

        async def body(_pilot: Pilot[Any]) -> tuple[str, str, str]:
            lit = app._alert_border_colour()
            app._flash_on = not app._flash_on
            unlit = app._alert_border_colour()
            app._flash_on = not app._flash_on
            return lit, unlit, app._alert_border_colour()

        lit, unlit, again = render(app, body)
        assert lit != unlit
        assert again == lit  # a full cycle returns to where it started

    def test_the_flash_timer_actually_toggles_the_border(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The colour logic is worthless if nothing drives it.

        ``flash_hz = 4`` is used so the half-period is 125 ms and the test does not have to
        wait a noticeable fraction of a second; the point is that the timer runs at all and
        that it stops the moment nothing is over its limit.
        """
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(
                alerts=AlertsConfig(enabled=True, temp_c=self.HOT_C, flash_hz=4.0)
            ),
        )

        async def body(pilot: Pilot[Any]) -> bool:
            await _until(lambda: app._alert_state.active, "an alert")
            start = app._flash_on
            await _until(lambda: app._flash_on != start, "a flash toggle", timeout=3.0)
            assert app.query_one("#blocks-panel").styles.border is not None
            return True

        assert render(app, body) is True

    def test_the_flash_timer_stops_when_nothing_is_over(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A border that keeps blinking with nothing wrong is worse than none."""
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(alerts=AlertsConfig(enabled=True, flash_hz=4.0)),
        )

        async def body(pilot: Pilot[Any]) -> bool:
            before = app._flash_on
            for _ in range(20):
                await pilot.pause(0.05)
            return app._flash_on == before

        assert render(app, body) is True

    def test_nothing_crossed_leaves_the_border_on_the_theme_accent(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        app = make_app(drm_root, proc_root)

        async def body(_pilot: Pilot[Any]) -> str:
            return app._alert_border_colour()

        assert render(app, body) == app._theme.accent

    def test_an_ordinary_card_does_not_alert_on_its_normal_readings(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """56 C junction and 45 W of a 211 W cap are both comfortably inside the defaults."""
        app = make_app(
            drm_root, proc_root, config=with_config(alerts=AlertsConfig(enabled=True))
        )

        async def body(_pilot: Pilot[Any]) -> bool:
            return app._alert_state.active

        assert render(app, body) is False

    def test_disabling_alerts_stops_the_border_changing(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """An alert that cannot be turned off is not a setting."""
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(alerts=AlertsConfig(enabled=False, temp_c=1.0)),
        )

        async def body(_pilot: Pilot[Any]) -> tuple[str, bool]:
            return app._alert_border_colour(), app._alert_state.active

        colour, active = render(app, body)
        assert active is False
        assert colour == app._theme.accent

    def test_the_border_is_not_re_written_when_nothing_changed(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Re-asserting an identical inline style still makes Textual repaint the panel.

        Profiling put per-sample border writes -- seven inline styles a second across
        every panel, to arrive at the colour already on screen -- among the largest
        avoidable costs in the frame.  The write is skipped when the colour has not
        changed, so a steady state writes nothing at all, and a real change still lands.
        """
        app = make_app(drm_root, proc_root)

        async def body(pilot: Pilot[Any]) -> tuple[str, str]:
            # Two more samples, so the border would have been re-applied twice more.
            await _until(lambda: app._sampler.stats.ticks >= 3, "three samples")
            applied = app._border_colour
            await _until(lambda: app._sampler.stats.ticks >= 6, "six samples")
            assert app._border_colour == applied
            # Clearing the recorded colour must force the next call to write again.
            app._border_colour = "stale"
            app._apply_alert_borders()
            return applied, app._border_colour

        before, after = render(app, body)
        assert before == after == app._theme.accent

    def test_a_zero_threshold_disables_that_one_alert(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``temp_c = 0`` means "never tell me", not "tell me above zero"."""
        app = make_app(
            drm_root,
            proc_root,
            config=with_config(alerts=AlertsConfig(enabled=True, temp_c=0.0)),
        )

        async def body(_pilot: Pilot[Any]) -> bool:
            return app._alert_state.active

        assert render(app, body) is False


class TestSamplerBlockSeam:
    """The contract between the radeontop pool and everything downstream."""

    def test_no_block_source_yields_none_rather_than_a_zeroed_sample(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``None`` and an empty sample are different answers and must not be confused."""
        build_hardware(drm_root, proc_root)
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                interval_s=0.05,
                collect_processes=False,
            )
        )
        sampler.discover()
        try:
            snapshot = sampler.sample_once()
            assert snapshot.devices
            assert snapshot.devices[0].blocks is None
            assert snapshot.blocks_status is not BlocksStatus.OK
            assert snapshot.blocks_hint or snapshot.blocks_status.hint
        finally:
            sampler.stop()

    def test_disabling_blocks_short_circuits_the_pool(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A disabled panel must not spawn anything, or open anything."""
        build_hardware(drm_root, proc_root)
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                interval_s=0.05,
                collect_processes=False,
            )
        )
        sampler.discover()
        sampler.start_blocks()
        try:
            assert sampler.sample_once().blocks_status is BlocksStatus.DISABLED
        finally:
            sampler.stop()

    def test_the_power_tables_are_read_into_the_sample(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The profile panel's data comes from the sampler, not from the UI."""
        build_hardware(drm_root, proc_root)
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                interval_s=0.05,
                collect_processes=False,
            )
        )
        sampler.discover()
        try:
            metrics = sampler.sample_once().devices[0]
            assert metrics.profiles.present
            assert metrics.profiles.active_name == "BOOTUP_DEFAULT"
            # The fixture has no pp_od_clk_voltage, which must be empty rather than wrong.
            assert not metrics.odc.present
        finally:
            sampler.stop()


class TestProfilePanelIsReadOnly:
    """The panel must not touch the files it reports."""

    def test_rendering_opens_nothing_from_the_ui_thread(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The UI performs no I/O; this is the same guarantee the rest of the app relies on.

        Asserted by wrapping both readers and recording calls that arrive from the event
        loop thread.  The sampler thread legitimately does all of this concurrently, so
        only main-thread calls are flagged and everything else is delegated untouched.
        """
        import threading

        import gputop.readers.pp as pp

        calls: list[str] = []

        def guard(name: str, original: object) -> object:
            def wrapper(*args: object, **kwargs: object) -> object:
                if threading.current_thread() is threading.main_thread():
                    calls.append(name)
                return original(*args, **kwargs)  # type: ignore[operator]

            return wrapper

        originals = {
            "read_power_profile_table": pp.read_power_profile_table,
            "read_odc_voltage": pp.read_odc_voltage,
        }
        pp.read_power_profile_table = guard(  # type: ignore[assignment]
            "power_profile_table", originals["read_power_profile_table"]
        )
        pp.read_odc_voltage = guard(  # type: ignore[assignment]
            "odc_voltage", originals["read_odc_voltage"]
        )
        try:
            screen_of(make_app(drm_root, proc_root))
        finally:
            pp.read_power_profile_table = originals["read_power_profile_table"]  # type: ignore[assignment]
            pp.read_odc_voltage = originals["read_odc_voltage"]  # type: ignore[assignment]
        assert calls == []
