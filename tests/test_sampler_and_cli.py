"""Tests for the model layer, config loading and the sampler/CLI plumbing."""

import io
import json
from pathlib import Path

import pytest

from gputop.config import Config, load_config
from gputop.model.history import RingBuffer
from gputop.model.metrics import Clock, Fan, MemoryPool, PcieLink, Power
from gputop.model.process import EngineUsage, GpuProcess
from gputop.model.snapshot import GpuSnapshot, SamplerStats, SourceReport
from gputop.sampler import Sampler, SamplerOptions
from tests.conftest import make_gpu, make_process, write


class TestModel:
    """Derived values must guard against the degenerate cases, not divide by zero."""

    def test_memory_pool_percent(self) -> None:
        assert MemoryPool(used=50, total=200).percent == 25.0

    def test_memory_pool_percent_is_none_without_a_total(self) -> None:
        assert MemoryPool(used=50, total=None).percent is None

    def test_memory_pool_percent_is_none_when_total_is_zero(self) -> None:
        """An APU can report a zero-sized pool; that must not divide by zero."""
        assert MemoryPool(used=0, total=0).percent is None

    def test_memory_pool_percent_clamps(self) -> None:
        assert MemoryPool(used=300, total=200).percent == 100.0

    def test_clock_percent(self) -> None:
        assert Clock(current=500, maximum=1000).percent == 50.0

    def test_clock_percent_needs_a_maximum(self) -> None:
        assert Clock(current=500, maximum=None).percent is None

    def test_power_percent(self) -> None:
        assert Power(draw_w=50, cap_w=200).percent == 25.0

    def test_power_percent_needs_a_cap(self) -> None:
        assert Power(draw_w=50, cap_w=None).percent is None

    def test_fan_percent_prefers_rpm(self) -> None:
        assert Fan(rpm=1500, max_rpm=3000).percent == 50.0

    def test_fan_percent_falls_back_to_pwm(self) -> None:
        assert Fan(rpm=None, max_rpm=None, pwm=40.0).percent == 40.0

    def test_fan_stopped_detects_a_stalled_fan(self) -> None:
        assert Fan(rpm=0).stopped
        assert not Fan(rpm=1200, max_rpm=3000).stopped

    def test_pcie_generation(self) -> None:
        assert PcieLink(width=16, speed=80).generation == 5
        assert PcieLink(width=16, speed=80).describe() == "Gen5 x16"

    def test_pcie_without_a_generation(self) -> None:
        assert PcieLink(width=4, speed=2).describe() == "x4"


class TestEngineUsage:
    """Delta maths, including the clamp that keeps the figure honest."""

    def test_percent_of_a_known_delta(self) -> None:
        usage = EngineUsage(engine="gfx", total_ns=500, delta_ns=250, window_ns=1000)
        assert usage.percent == 25.0

    def test_percent_is_none_without_a_window(self) -> None:
        assert EngineUsage(engine="gfx", total_ns=5, delta_ns=0, window_ns=0).percent is None

    def test_percent_clamps_at_one_hundred(self) -> None:
        usage = EngineUsage(engine="gfx", total_ns=5000, delta_ns=5000, window_ns=1000)
        assert usage.percent == 100.0

    def test_process_percent_sums_engines(self) -> None:
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="0000:0c:00.0",
            client_id=1,
            engines=(
                EngineUsage("gfx", 400, 400, 1000),
                EngineUsage("compute", 300, 300, 1000),
            ),
        )
        assert process.engine_percent == 70.0

    def test_process_percent_is_zero_without_engines(self) -> None:
        process = GpuProcess(pid=1, name="x", user="u", bdf="b", client_id=1)
        assert process.engine_percent == 0.0

    def test_memory_used_includes_the_cpu_pool(self) -> None:
        """On an APU the CPU-visible pool dominates, so excluding it would mislead."""
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="b",
            client_id=1,
            vram_used=10,
            gtt_used=20,
            cpu_used=70,
        )
        assert process.memory_used == 100

    def test_engine_percent_for_a_bucket(self) -> None:
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="b",
            client_id=1,
            engines=(EngineUsage("gfx", 250, 250, 1000),),
        )
        assert process.engine_percent_for("gfx") == 25.0
        assert process.engine_percent_for("enc") is None


class TestRingBuffer:
    """The history buffer is bounded and oldest-first."""

    def test_bounded_capacity(self) -> None:
        buffer = RingBuffer[int](capacity=3)
        for value in range(1, 6):
            buffer.append(value)
        assert buffer.items() == (3, 4, 5)
        assert len(buffer) == 3

    def test_capacity_is_at_least_one(self) -> None:
        assert RingBuffer[int](capacity=0).capacity == 1

    def test_last(self) -> None:
        buffer = RingBuffer[int](capacity=2)
        assert buffer.last() is None
        buffer.append(7)
        assert buffer.last() == 7

    def test_items_are_a_snapshot(self) -> None:
        """Returning a tuple means a reader cannot be affected by later appends."""
        buffer = RingBuffer[int](capacity=2)
        buffer.append(1)
        snapshot = buffer.items()
        buffer.append(2)
        assert snapshot == (1,)


class TestConfig:
    """Config loading is total and never rejects an unknown key by crashing."""

    def test_missing_file_yields_defaults_with_a_warning(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.toml")
        assert config.general.interval_ms == 1000
        assert any("not found" in w for w in config.warnings)

    def test_valid_file_is_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(
            path,
            "[general]\ninterval_ms = 250\nhistory_points = 60\n"
            "[process]\nmax_rows = 5\nsort = 'memory'\n",
        )
        config = load_config(path)

        assert config.general.interval_ms == 250
        assert config.history_length == 60
        assert config.process.max_rows == 5
        assert config.warnings == ()

    def test_unknown_key_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        """A config written for a future release must still work today."""
        path = tmp_path / "gputop.toml"
        write(path, "[general]\ninterval_ms = 500\nfrom_the_future = 1\n")

        config = load_config(path)

        assert config.general.interval_ms == 500
        assert any("from_the_future" in w for w in config.warnings)

    def test_malformed_toml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "this is not = = toml")
        config = load_config(path)
        assert config.warnings

    def test_out_of_range_interval_is_clamped(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[general]\ninterval_ms = 5\n")
        config = load_config(path)
        assert config.general.interval_ms == 100
        assert any("out of range" in w for w in config.warnings)

    def test_invalid_kind_falls_back_to_auto(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[gpu]\nkind = 'nonsense'\n")
        assert load_config(path).gpu.kind == "auto"

    def test_gpu_names_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[gpu.names]\n'0000:0c:00.0' = 'My GPU'\n")
        assert load_config(path).gpu.names == {"0000:0c:00.0": "My GPU"}

    def test_defaults_have_no_warnings(self) -> None:
        assert Config().warnings == ()


class TestSampler:
    """The sampler ties the readers together and must survive hostile input."""

    def _sampler(self, drm_root: Path, proc_root: Path) -> Sampler:
        return Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=True,
                interval_s=0.05,
            )
        )

    def test_empty_tree_produces_an_empty_snapshot(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        sampler = self._sampler(drm_root, proc_root)
        assert sampler.discover() == ()

        snapshot = sampler.sample_once()

        assert snapshot.devices == ()
        assert snapshot.processes == ()
        assert snapshot.sequence == 1

    def test_snapshot_combines_devices_and_processes(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        make_process(proc_root, 100)
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        snapshot = sampler.sample_once()

        assert len(snapshot.devices) == 1
        assert len(snapshot.processes) == 1
        assert snapshot.total_process_count == 1
        assert snapshot.stats.ticks == 1
        assert not snapshot.is_partial_process_view

    def test_partial_process_visibility_is_reported(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Hiding other users' processes must be visible, not silently misleading."""
        make_gpu(drm_root, bdf="0000:0c:00.0")
        make_process(proc_root, 100)
        (proc_root / "200").mkdir()
        (proc_root / "200" / "fd").symlink_to(proc_root / "999" / "fd")
        (proc_root / "200" / "fdinfo").symlink_to(proc_root / "999" / "fdinfo")

        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()
        snapshot = sampler.sample_once()

        assert snapshot.total_process_count == 2
        assert snapshot.visible_process_count == 1
        assert snapshot.is_partial_process_view

    def test_a_device_without_binary_metrics_still_produces_a_sample(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A pre-5.19 kernel has no ``gpu_metrics``; the sysfs path must carry it."""
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            metrics=None,
            metrics_abi=None,
            extra_sysfs={
                "pp_dpm_sclk": "0: 500Mhz *\n1: 2400Mhz \n",
                "mem_info_vis_vram_used": "1048576",
                "mem_info_vis_vram_total": "8589934592",
                "gpu_busy_percent": "42",
            },
        )
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        metrics = sampler.sample_once().devices[0]

        assert metrics.metrics_abi is None
        assert metrics.sclk.current == 500
        assert metrics.sclk.maximum == 2400
        assert metrics.sclk.source == "dpm"
        assert metrics.gpu_busy_percent == 42.0
        assert metrics.vram.used == 1048576
        # 1 MiB of 8 GiB is 0.0122%, not 12.2%.
        assert metrics.vram.percent == pytest.approx(0.0122, abs=0.0001)

    def test_one_broken_device_does_not_hide_the_others(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A failing device is isolated so the healthy one is still reported."""
        make_gpu(drm_root, card=0, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        make_gpu(drm_root, card=1, bdf="0000:03:00.0", metrics={"temperature_edge": 40})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        # Make one device's directory unusable without touching the other.
        broken = sampler.devices[0].device_dir / "gpu_metrics"
        broken.unlink()
        broken.mkdir()

        snapshot = sampler.sample_once()

        # Both devices still produce a sample; the broken one simply has no binary data.
        assert len(snapshot.devices) == 2
        assert sorted(str(m.metrics_abi) for m in snapshot.devices) == ["None", "v1.3"]
        broken_metrics = next(m for m in snapshot.devices if m.metrics_abi is None)
        assert broken_metrics.sclk.current is None
        healthy = next(m for m in snapshot.devices if m.metrics_abi == "v1.3")
        assert healthy.temperatures[0].celsius == 49

    def test_options_are_validated(self) -> None:
        options = SamplerOptions(interval_s=0.001, history_length=0).validated()
        assert options.interval_s == 0.05
        assert options.history_length == 1

    def test_sample_once_is_repeatable(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        first = sampler.sample_once()
        second = sampler.sample_once()

        assert first.sequence == 1
        assert second.sequence == 2
        assert first.timestamp_ns <= second.timestamp_ns

    def test_thread_publishes_snapshots(self, drm_root: Path, proc_root: Path) -> None:
        """The background thread hands immutable snapshots over the queue."""
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.start()
        try:
            snapshot = sampler.wait(timeout=5.0)
        finally:
            sampler.stop()

        assert snapshot is not None
        assert len(snapshot.devices) == 1
        assert sampler.history()

    def test_source_report_tracks_the_binary_abi(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        snapshot = sampler.sample_once()

        assert snapshot.source.metrics_abi == "v1.3"
        assert snapshot.source.using_binary_metrics


class TestSnapshot:
    """Snapshot lookups used by the UI."""

    def test_device_by_index(self) -> None:
        snapshot = GpuSnapshot(sequence=1, timestamp_ns=0)
        assert snapshot.device_by_index(0) is None

    def test_partial_visibility_flags(self) -> None:
        partial = GpuSnapshot(
            sequence=1, timestamp_ns=0, visible_process_count=1, total_process_count=5
        )
        assert partial.is_partial_process_view

    def test_default_stats_are_zeroed(self) -> None:
        assert SamplerStats().ticks == 0

    def test_source_report_defaults(self) -> None:
        assert not SourceReport().using_binary_metrics


class TestCli:
    """The JSON dump is the tool's scriptable contract."""

    def _run(self, argv: list[str]) -> dict:
        from gputop.cli import main

        buffer = io.StringIO()
        status = main(
            [*argv, "--drm-root", str(self._drm), "--proc-root", str(self._proc)],
            buffer,
        )
        assert status == 0, f"exit {status}"
        return json.loads(buffer.getvalue())

    def test_devices_listing(self, drm_root: Path, proc_root: Path) -> None:
        self._drm, self._proc = drm_root, proc_root
        make_gpu(drm_root, bdf="0000:0c:00.0", device_id=0x73BF)

        payload = self._run(["--devices"])

        assert payload["count"] == 1
        assert payload["devices"][0]["bdf"] == "0000:0c:00.0"
        assert payload["devices"][0]["name"] == "AMD Radeon Navi 44"

    def test_dump_emits_json(self, drm_root: Path, proc_root: Path) -> None:
        self._drm, self._proc = drm_root, proc_root
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            metrics={"temperature_edge": 49, "average_gfx_activity": 12},
        )
        make_process(proc_root, 100)

        payload = self._run(["--dump", "--interval", "0.05", "--no-processes"])

        assert payload["metrics_abi"] == "v1.3"
        assert payload["using_binary_metrics"] is True
        device = payload["devices"][0]
        assert device["temperatures"][0] == {
            "label": "edge",
            "celsius": 49,
            "source": "gpu_metrics",
        }
        assert device["gpu_busy_percent"] == 12.0

    def test_dump_without_a_device_reports_and_exits_nonzero(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        from gputop.cli import main

        buffer = io.StringIO()
        status = main(
            ["--dump", "--drm-root", str(drm_root), "--proc-root", str(proc_root)], buffer
        )

        assert status == 3
        payload = json.loads(buffer.getvalue())
        assert payload["devices"] == []
        assert any("no amdgpu device" in w for w in payload["warnings"])

    def test_pretty_output_is_indented(self, drm_root: Path, proc_root: Path) -> None:
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0")
        buffer = io.StringIO()
        main(
            [
                "--devices",
                "--pretty",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
            ],
            buffer,
        )
        assert "\n  " in buffer.getvalue()

    def test_unavailable_metrics_are_null_not_zero(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A consumer must be able to tell "not available" from "measured zero"."""
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        buffer = io.StringIO()
        main(
            [
                "--dump",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        device = json.loads(buffer.getvalue())["devices"][0]

        # No VRAM attributes in the fake tree, so these must be null rather than 0.
        assert device["vram"] == {"used": None, "total": None, "percent": None}
        # Absent entirely: this layout has no PCIe or voltage fields at all.
        assert device["pcie"] is None
        assert device["voltages_mv"] == {}
        # throttle_status exists in v1.3 and reads zero, which means "not throttling" --
        # a real reading, so it must be reported rather than nulled out.
        assert device["throttle"] == {"raw": 0, "active": [], "is_throttling": False}
