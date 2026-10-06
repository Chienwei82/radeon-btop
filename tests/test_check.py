"""Tests for ``gputop --check``: the report, and every reason it can give.

The point of ``--check`` is the *reasons*, so most of what is tested here is the
explanation rather than the reading: a report that says "missing" without saying why has
not answered the question it was asked.
"""

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from gputop.check import (
    Availability,
    Reason,
    _explain,
    _specs,
    _table,
    at_least,
    describe_version,
    device_findings,
    gather,
    kernel_version,
    process_ownership,
    system_findings,
)
from gputop.cli import main
from gputop.model.blocks import BlocksStatus
from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.snapshot import GpuSnapshot
from tests.conftest import make_gpu, make_process

#: One rendered table row: label, status, source, then the reading.
_ROW = re.compile(r"^\S.*?\s{2}(ok|missing|disabled)\s{2}")


def bare_device(drm_root: Path, *, device_id: int = 0x73BF, card: int = 0) -> Path:
    """A GPU directory with nothing in it but the identity files."""
    return make_gpu(drm_root, card=card, device_id=device_id, driver="amdgpu")


def device_for(drm_root: Path, proc_root: Path, **kwargs) -> AmdgpuDevice:
    """Discover the single GPU in a synthetic tree."""
    from gputop.readers.discovery import discover_devices

    found = discover_devices(drm_root)
    assert found, "the fixture tree has no device"
    return found[0]


def sample_for(device: AmdgpuDevice) -> AmdgpuMetrics:
    """An empty metric record for one device: everything missing."""
    return AmdgpuMetrics(device=device, sequence=1, timestamp_ns=0, interval_s=1.0)


#: Clock metrics, whose ``source`` column names a real sysfs file rather than a family.
CLOCK_KEYS = ("sclk", "mclk", "socclk")


def snapshot_for(*devices: AmdgpuDevice, processes: tuple[object, ...] = ()) -> GpuSnapshot:
    """A snapshot carrying the given devices and no readings."""
    return GpuSnapshot(
        sequence=1,
        timestamp_ns=0,
        devices=tuple(sample_for(device) for device in devices),
        processes=processes,
        visible_process_count=len(processes),
        total_process_count=1 if processes else 0,
        source=MappingProxyType({"per_metric": MappingProxyType({})}),
    )


class TestKernelVersion:
    """The kernel comparison that turns "absent" into "too old"."""

    @pytest.mark.parametrize(
        "release,expected",
        [
            ("7.0.0-34-generic", (7, 0)),
            ("6.8.0-amd64", (6, 8)),
            ("5.19.0-9-amd64", (5, 19)),
            ("4.19.0", (4, 19)),
            ("6.10.0-rc3", (6, 10)),
        ],
    )
    def test_parsed(self, release: str, expected: tuple[int, int]) -> None:
        assert kernel_version(release) == expected

    @pytest.mark.parametrize("release", ["", "not-a-kernel", "linux.x.y"])
    def test_unparseable_is_zero_not_an_error(self, release: str) -> None:
        """An unknown version must never claim a metric is too new."""
        assert kernel_version(release) == (0, 0)

    def test_zero_never_counts_as_new_enough(self) -> None:
        assert at_least((0, 0), (5, 19)) is False
        assert at_least((5, 19), (5, 19)) is True
        assert at_least((6, 1), (5, 19)) is True

    def test_an_unknown_kernel_is_not_reported_as_too_old(self, tmp_path: Path) -> None:
        """``at_least`` answers False for an unparsed version, and that is the wrong answer here.

        The docstring on :func:`kernel_version` promises the explanation "falls back to
        the weaker, still-true 'not published'".  It did not: the single comparison said
        "needs Linux 5.10 or newer, this is 0.0" -- the strongest possible claim about a
        kernel version nobody is running.
        """
        reason, detail = _explain(
            tmp_path, ("gpu_metrics", "gpu_busy_percent"), (0, 0), min_kernel=(5, 10)
        )
        assert reason is Reason.NOT_PUBLISHED
        assert "0.0" not in detail

    def test_a_genuinely_old_kernel_is_still_reported(self, tmp_path: Path) -> None:
        reason, detail = _explain(tmp_path, ("gpu_metrics",), (5, 1), min_kernel=(5, 10))
        assert reason is Reason.NEEDS_NEWER_KERNEL
        assert "5.1" in detail

    def test_describe(self) -> None:
        assert describe_version((6, 8)) == "6.8"


class TestDeviceFindings:
    """Each metric's verdict on a tree that does not have it."""

    def test_a_bare_card_reports_everything_missing(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        findings = device_findings(device, sample_for(device), (7, 0))
        assert findings, "the report covers no metrics at all"
        assert all(f.availability is Availability.MISSING for f in findings)
        assert all(f.reason is not None for f in findings)

    def test_a_fully_populated_card_reports_everything_present(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        from gputop.model.metrics import Clock, Fan, MemoryPool, PcieLink, Power, Temperature

        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        metrics = AmdgpuMetrics(
            device=device,
            sequence=1,
            timestamp_ns=0,
            interval_s=1.0,
            gpu_busy_percent=50.0,
            mem_busy_percent=25.0,
            vram=MemoryPool(used=1 << 30, total=1 << 34),
            gtt=MemoryPool(used=1 << 20, total=1 << 33),
            sclk=Clock(current=2000, maximum=2500, source="gpu_metrics"),
            mclk=Clock(current=900, maximum=1000, source="dpm"),
            socclk=Clock(current=800, maximum=1200, source="dpm"),
            temperatures=(Temperature("edge", 50.0, "hwmon"),),
            power=Power(draw_w=100.0, cap_w=200.0),
            fan=Fan(rpm=1000, max_rpm=3000, pwm=50.0),
            performance_level="auto",
            power_profile="BOOTUP_DEFAULT",
            pcie=PcieLink(width=16, speed=80),
            voltages_mv=(("gfx", 800),),
        )
        findings = device_findings(device, metrics, (7, 0))
        missing = {f.metric.key: f for f in findings if f.availability is not Availability.OK}
        # throttle and the overdrive table are the two this record does not carry.
        assert set(missing) == {"throttle", "overdrive"}, sorted(missing)

    def test_the_source_column_names_the_file_that_actually_answered(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The DPM tables are ``pp_dpm_sclk``/``pp_dpm_mclk``/``pp_dpm_socclk``.

        Deriving the name by stripping ``clk`` out of the key produced ``pp_dpm_s``,
        ``pp_dpm_m`` and ``pp_dpm_soc`` -- three filenames that do not exist, in the one
        column of the report people paste into bug reports.
        """
        from gputop.model.metrics import Clock

        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        metrics = AmdgpuMetrics(
            device=device,
            sequence=1,
            timestamp_ns=0,
            interval_s=1.0,
            sclk=Clock(current=2000, maximum=2500, source="dpm"),
            mclk=Clock(current=900, maximum=1000, source="dpm"),
            socclk=Clock(current=800, maximum=1200, source="dpm"),
        )

        findings = device_findings(device, metrics, (7, 0))
        clocks = {f.metric.key: f.metric.source for f in findings if f.metric.key in CLOCK_KEYS}

        assert clocks["sclk"] == "pp_dpm_sclk"
        assert clocks["mclk"] == "pp_dpm_mclk"
        assert clocks["socclk"] == "pp_dpm_socclk"

    def test_an_integrated_gpu_is_told_it_has_no_fan(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """An APU genuinely has no fan; that is hardware, not a missing file."""
        # An APU is an APU for a reason: the carve-out is small and the VRAM vendor reads
        # N/A, which is what the classifier votes on.
        make_gpu(
            drm_root,
            device_id=0x1638,
            driver="amdgpu",
            extra_sysfs={
                "mem_info_vram_vendor": "N/A",
                "mem_info_vis_vram_total": str(512 * 1024 * 1024),
            },
        )
        device = device_for(drm_root, proc_root)
        assert device.is_integrated
        findings = device_findings(device, sample_for(device), (7, 0))
        fan = next(f for f in findings if f.metric.key == "fan")
        assert fan.availability is Availability.MISSING
        assert fan.reason is Reason.UNSUPPORTED_GENERATION

    def test_an_old_kernel_is_blamed_on_the_kernel(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The same empty tree reads differently at 5.4 than at 7.0.

        5.4 rather than something merely old, because the boundary is a fact about the
        driver: ``gpu_metrics`` was registered in 5.10 (absent at v5.9, present at v5.10 in
        ``drivers/gpu/drm/amd/pm/amdgpu_pm.c``).  A kernel from 5.15 already has the
        attribute and the honest answer there is "the driver does not publish it here".
        """
        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        old = device_findings(device, sample_for(device), (5, 4))
        gpu = next(f for f in old if f.metric.key == "gpu_utilisation")
        assert gpu.reason is Reason.NEEDS_NEWER_KERNEL
        assert "5.10" in gpu.detail

        current = device_findings(device, sample_for(device), (7, 0))
        again = next(f for f in current if f.metric.key == "gpu_utilisation")
        assert again.reason is Reason.NOT_PUBLISHED

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
    def test_an_unreadable_attribute_is_reported_as_a_permission_problem(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A permission problem is the one cause the user can fix without a new kernel."""
        device_dir = bare_device(drm_root)
        target = device_dir / "gpu_busy_percent"
        target.write_text("50\n")
        target.chmod(0o000)
        try:
            device = device_for(drm_root, proc_root)
            findings = device_findings(device, sample_for(device), (7, 0))
            gpu = next(f for f in findings if f.metric.key == "gpu_utilisation")
            assert gpu.reason is Reason.PERMISSION
            assert "gpu_busy_percent" in gpu.detail
        finally:
            target.chmod(0o644)

    def test_every_metric_has_a_reason_and_a_source(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        findings = device_findings(device, sample_for(device), (7, 0))
        for finding in findings:
            assert finding.metric.source, finding.metric.key
            assert finding.metric.label, finding.metric.key
            if finding.availability is Availability.MISSING:
                assert finding.reason is not None, finding.metric.key


class TestSystemFindings:
    """The findings that are about the machine rather than one GPU."""

    def test_no_devices_at_all(self, drm_root: Path, proc_root: Path) -> None:
        findings = system_findings(
            snapshot_for(),
            kernel=(7, 0),
            proc_root=proc_root,
            blocks_status=BlocksStatus.DISABLED,
            blocks_hint="",
            blocks_enabled=False,
            blocks_binary="radeontop",
        )
        processes = next(f for f in findings if f.metric.key == "processes")
        assert processes.availability is Availability.MISSING
        assert processes.reason is Reason.NO_CLIENTS

    def test_the_blocks_panel_is_reported_as_turned_off(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        findings = system_findings(
            snapshot_for(),
            kernel=(7, 0),
            proc_root=proc_root,
            blocks_status=BlocksStatus.DISABLED,
            blocks_hint="",
            blocks_enabled=False,
            blocks_binary="radeontop",
        )
        blocks = next(f for f in findings if f.metric.key == "blocks")
        assert blocks.availability is Availability.DISABLED
        assert blocks.reason is Reason.DISABLED

    @pytest.mark.parametrize(
        "status,expected",
        [
            (BlocksStatus.MISSING, Reason.NOT_INSTALLED),
            (BlocksStatus.NOT_PERMITTED, Reason.PERMISSION),
            (BlocksStatus.UNSUPPORTED, Reason.UNSUPPORTED_GENERATION),
            (BlocksStatus.FAILED, Reason.NEEDS_NEWER_DRIVER),
        ],
    )
    def test_each_blocks_failure_maps_to_its_own_remedy(
        self, status: BlocksStatus, expected: Reason, tmp_path: Path
    ) -> None:
        """'Install radeontop' and 'you need root' are different problems."""
        findings = system_findings(
            snapshot_for(),
            kernel=(7, 0),
            proc_root=tmp_path,
            blocks_status=status,
            blocks_hint="because",
            blocks_enabled=True,
            blocks_binary="radeontop",
        )
        blocks = next(f for f in findings if f.metric.key == "blocks")
        assert blocks.availability is Availability.MISSING
        assert blocks.reason is expected

    def test_a_running_blocks_pool_is_reported_as_available(self, tmp_path: Path) -> None:
        findings = system_findings(
            snapshot_for(),
            kernel=(7, 0),
            proc_root=tmp_path,
            blocks_status=BlocksStatus.OK,
            blocks_hint="",
            blocks_enabled=True,
            blocks_binary="radeontop",
        )
        blocks = next(f for f in findings if f.metric.key == "blocks")
        assert blocks.availability is Availability.OK

    def test_an_old_kernel_blames_per_process_accounting(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        make_process(proc_root, 100)
        findings = system_findings(
            snapshot_for(),
            kernel=(5, 4),
            proc_root=proc_root,
            blocks_status=BlocksStatus.DISABLED,
            blocks_hint="",
            blocks_enabled=False,
            blocks_binary="radeontop",
        )
        processes = next(f for f in findings if f.metric.key == "processes")
        assert processes.reason is Reason.NEEDS_NEWER_KERNEL
        assert "5.19" in processes.detail

    def test_process_ownership_counts_processes(self, proc_root: Path) -> None:
        for pid in (100, 101, 102):
            make_process(proc_root, pid, drm_node="/dev/null")
        total, hidden = process_ownership(proc_root)
        assert total == 3
        assert hidden == 0  # the fixtures are created by this user

    def test_process_ownership_survives_a_missing_procfs(self, tmp_path: Path) -> None:
        assert process_ownership(tmp_path / "nope") == (0, 0)

    def test_an_unscanned_process_table_does_not_call_the_gpu_idle(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``--no-processes`` (or ``[process] show = false``) means nobody looked.

        The finding used to say "none of the N visible processes holds a /dev/dri
        descriptor, which is what an idle GPU looks like" and status ``ok`` -- an empty
        table because the scan was switched off, reported as a positive answer about the
        hardware.  That is the question people most often bring to ``--check``.
        """
        make_process(proc_root, 100)
        findings = system_findings(
            snapshot_for(),
            kernel=(6, 1),
            proc_root=proc_root,
            blocks_status=BlocksStatus.DISABLED,
            blocks_hint="",
            blocks_enabled=False,
            blocks_binary="radeontop",
            processes_scanned=False,
        )
        processes = next(f for f in findings if f.metric.key == "processes")
        assert processes.availability is Availability.DISABLED
        assert "idle" not in processes.detail
        assert "switched off" in processes.detail

    def test_a_scanned_but_quiet_gpu_still_reports_idle(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The real idle verdict must survive: this is the behaviour, not a bug."""
        make_process(proc_root, 100)
        findings = system_findings(
            snapshot_for(),
            kernel=(6, 1),
            proc_root=proc_root,
            blocks_status=BlocksStatus.DISABLED,
            blocks_hint="",
            blocks_enabled=False,
            blocks_binary="radeontop",
            processes_scanned=True,
        )
        processes = next(f for f in findings if f.metric.key == "processes")
        assert processes.availability is Availability.OK
        assert "idle GPU" in processes.detail

    def test_blocks_that_were_never_run_are_not_called_missing(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        """``--check`` never spawns radeontop, so the pool's status is not an observation.

        It starts at ``MISSING`` and is only resolved in ``start()``, so the report told
        every user to install a binary they already had.  ``None`` now means "not run",
        and the binary is resolved without starting anything: a ``$PATH`` lookup and a
        ``stat``, no process and no privileges.
        """
        binary = tmp_path / "radeontop"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        findings = system_findings(
            snapshot_for(),
            kernel=(6, 1),
            proc_root=proc_root,
            blocks_status=None,
            blocks_hint="",
            blocks_enabled=True,
            blocks_binary=str(binary),
        )
        blocks = next(f for f in findings if f.metric.key == "blocks")
        assert blocks.reason is not Reason.NOT_INSTALLED

    def test_blocks_that_were_never_run_still_report_a_genuine_absence(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        findings = system_findings(
            snapshot_for(),
            kernel=(6, 1),
            proc_root=proc_root,
            blocks_status=None,
            blocks_hint="",
            blocks_enabled=True,
            blocks_binary=str(tmp_path / "definitely-not-here"),
        )
        blocks = next(f for f in findings if f.metric.key == "blocks")
        assert blocks.reason is Reason.NOT_INSTALLED

    def test_an_exited_process_is_not_reported_as_another_users(
        self, proc_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PID that vanished mid-scan is not a permissions problem.

        ``process_ownership`` counted *any* ``OSError`` from ``stat`` as hidden, so on a
        busy machine -- where processes exit between ``scandir`` and ``stat`` constantly --
        the report claimed "N of M processes belong to another user and cannot be
        inspected" about processes that no longer existed.
        """
        import gputop.check as check

        make_process(proc_root, 100)
        make_process(proc_root, 101)
        real_scandir = check.os.scandir

        class Entry:
            """A scan entry whose ``stat`` fails the way a dead PID's does."""

            def __init__(self, entry: object, gone: bool) -> None:
                self._entry = entry
                self._gone = gone
                self.name = getattr(entry, "name", "")

            def stat(self, **kwargs: object) -> object:
                if self._gone:
                    raise FileNotFoundError(2, "No such file or directory")
                return self._entry.stat(**kwargs)  # type: ignore[attr-defined]

        class FakeScan:
            """A ``scandir`` result: iterable, and a context manager like the real one."""

            def __init__(self, entries: list[Entry]) -> None:
                self._entries = entries

            def __iter__(self) -> object:
                return iter(self._entries)

            def __enter__(self) -> object:
                return iter(self._entries)

            def __exit__(self, *exc: object) -> None:
                return None

        def fake_scandir(path: object) -> object:
            return FakeScan(
                [Entry(e, e.name == "101") for e in real_scandir(path)]  # type: ignore[arg-type]
            )

        monkeypatch.setattr(check.os, "scandir", fake_scandir)
        total, hidden = process_ownership(proc_root)
        assert hidden == 0, "a vanished process was blamed on permissions"
        assert total == 1, "a vanished process is still counted in the total"


class TestSubReadings:
    """A missing part of a reading is absent, not zero and not the word "None"."""

    def _specs(self) -> dict[str, object]:
        return {spec.key: spec for spec in _specs()}

    def _base(self, drm_root: Path) -> AmdgpuMetrics:
        bare_device(drm_root)
        return sample_for(device_for(drm_root, drm_root))

    def test_a_missing_fan_tach_is_not_reported_as_zero_rpm(self, drm_root: Path) -> None:
        """``m.fan.rpm or 0`` turned "no sensor" into a stopped fan.

        :attr:`Fan.stopped` treats a zero as *present and not spinning*, so the row was
        making a claim about the hardware rather than reporting a gap in the data.
        """
        from gputop.model.metrics import Fan

        value = self._specs()["fan"].read(  # type: ignore[attr-defined]
            replace(self._base(drm_root), fan=Fan(pwm=50.0))
        )
        assert value == "50 % duty"

    def test_a_missing_duty_does_not_print_the_word_none(self, drm_root: Path) -> None:
        from gputop.model.metrics import Fan

        value = self._specs()["fan"].read(  # type: ignore[attr-defined]
            replace(self._base(drm_root), fan=Fan(rpm=797, max_rpm=3000))
        )
        assert value == "797 RPM"

    def test_a_genuinely_stopped_fan_still_reports_zero(self, drm_root: Path) -> None:
        from gputop.model.metrics import Fan

        value = self._specs()["fan"].read(  # type: ignore[attr-defined]
            replace(self._base(drm_root), fan=Fan(rpm=0, max_rpm=3000, pwm=0.0))
        )
        assert value == "0 RPM, 0 % duty"

    def test_a_missing_power_cap_omits_the_clause(self, drm_root: Path) -> None:
        from gputop.model.metrics import Power

        value = self._specs()["power"].read(  # type: ignore[attr-defined]
            replace(self._base(drm_root), power=Power(draw_w=152.0))
        )
        assert value == "152 W"

    def test_a_published_power_cap_is_shown(self, drm_root: Path) -> None:
        from gputop.model.metrics import Power

        value = self._specs()["power"].read(  # type: ignore[attr-defined]
            replace(self._base(drm_root), power=Power(draw_w=152.0, cap_w=211.0))
        )
        assert value == "152 W (cap 211 W)"

    def test_no_row_ever_contains_the_word_none(self, drm_root: Path) -> None:
        """The whole class of bug, checked over every metric at once."""
        from gputop.model.metrics import Fan, Power, Temperature

        metrics = replace(
            self._base(drm_root),
            fan=Fan(),
            power=Power(draw_w=10.0),
            temperatures=(Temperature(label="edge", celsius=None, source="none"),),
        )
        for key, spec in self._specs().items():
            rendered = spec.read(metrics)  # type: ignore[attr-defined]
            assert rendered is None or "None" not in rendered, key


class TestRendering:
    """The text and JSON shapes."""

    def test_the_table_is_aligned(self, drm_root: Path, proc_root: Path) -> None:
        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        findings = device_findings(device, sample_for(device), (7, 0))
        lines = _table(findings)
        # Data rows start at column 0; the notes under them are indented.  The first two
        # lines are the header and its rule.
        rows = [line for line in lines if line and not line[0].isspace()][2:]
        parsed = [_ROW.match(line) for line in rows]
        assert all(parsed), "every finding gets a row"
        assert len({match.start(1) for match in parsed if match}) == 1, (
            "statuses do not line up"
        )
        assert any("->" in line for line in lines), "missing metrics carry no explanation"

    def test_an_available_metric_carries_no_caveat(self) -> None:
        """A note on a row that says 'ok' reads as doubt about the number."""
        from gputop.check import Finding, Metric

        good = Finding(
            Metric(key="k", label="Metric", source="src", value="1", extra="only sometimes"),
            Availability.OK,
        )
        assert not any("only sometimes" in line for line in _table([good]))

    def test_a_missing_metric_shows_its_caveat_as_well_as_its_reason(self) -> None:
        """``extra`` used to render only when there was no reason, and there always was one.

        So the standing "what to do about it" lines never appeared: not "add --blocks",
        not "APUs have no fan", not the GCN overdrive note.  Each of those strings was set,
        and none of them was ever shown.
        """
        from gputop.check import Finding, Metric

        finding = Finding(
            Metric(
                key="k",
                label="Metric",
                source="src",
                value=None,
                extra="turn it on with --blocks",
            ),
            Availability.DISABLED,
            Reason.DISABLED,
            "turned off",
        )
        lines = _table([finding])
        assert any("turned off" in line for line in lines)
        assert any("turn it on with --blocks" in line for line in lines)

    def test_a_caveat_without_a_reason_still_shows_exactly_once(self) -> None:
        """The unconditional ``extra`` line is the only place the caveat is printed.

        A fallback that folded ``extra`` into the note printed it twice for a finding
        that carried no reason of its own.
        """
        from gputop.check import Finding, Metric

        finding = Finding(
            Metric(
                key="k",
                label="Metric",
                source="src",
                value=None,
                extra="turn it on with --blocks",
            ),
            Availability.DISABLED,
        )
        lines = _table([finding])
        assert sum("turn it on with --blocks" in line for line in lines) == 1

    def test_json_is_shaped_for_a_consumer(self, drm_root: Path, proc_root: Path) -> None:
        bare_device(drm_root)
        device = device_for(drm_root, proc_root)
        report = gather(
            snapshot_for(device),
            drm_root=drm_root,
            proc_root=proc_root,
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["ok"] is True
        assert payload["devices"][0]["card"]
        assert payload["devices"][0]["metrics"][0]["status"] == "missing"
        assert payload["devices"][0]["metrics"][0]["reason"]


class TestCommand:
    """The CLI surface: ``--check``, ``--check --json`` and the exit status."""

    def test_it_runs_and_exits_zero_when_a_gpu_is_present(
        self, drm_root: Path, proc_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tests.conftest import write

        make_gpu(
            drm_root,
            metrics={"average_gfx_activity": 30, "current_gfxclk": 2000},
            hwmon={"temp1_input": "49000"},
            extra_sysfs={"mem_info_vis_vram_total": str(8 << 30)},
        )
        write(drm_root / "card0" / "device" / "gpu_busy_percent", "30\n")

        status = main(["--check", "--drm-root", str(drm_root), "--proc-root", str(proc_root)])
        out = capsys.readouterr().out
        assert status == 0
        assert "gputop check" in out
        assert "METRIC" in out
        assert "card0" in out

    def test_it_exits_three_when_there_is_no_gpu(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = main(
            ["--check", "--drm-root", str(tmp_path / "empty"), "--proc-root", str(tmp_path)]
        )
        out = capsys.readouterr().out
        assert status == 3
        assert "No amdgpu device" in out

    def test_json_output_is_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = main(
            [
                "--check",
                "--json",
                "--drm-root",
                str(tmp_path / "empty"),
                "--proc-root",
                str(tmp_path),
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert status == 3
        assert payload["ok"] is False
        assert "environment" in payload

    def test_the_report_never_starts_radeontop(
        self, drm_root: Path, proc_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A diagnostic must not launch a process that wants privileges."""
        make_gpu(drm_root)
        main(
            [
                "--check",
                "--blocks",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
            ]
        )
        out = capsys.readouterr().out
        # The pool was configured but never started, so there is no sample and no child.
        assert "Per-block panel" in out
