"""The sampling thread.

Design constraints, in priority order:

1. **The sampler thread owns all mutable state.**  Nothing it writes is read by the UI
   thread.  Results cross the boundary only as immutable :class:`GpuSnapshot` values
   pushed onto a :class:`queue.SimpleQueue`.
2. **No locks.**  A lock would be correct on both builds but would reintroduce the GIL-like
   serialisation the free-threaded build exists to remove.  Single-owner state needs none.
3. **A tick never dies.**  Every reader is already total, and this layer additionally
   catches exceptions per device and per sub-system so that one bad sample degrades to a
   warning instead of silently stopping the monitor.

The thread sleeps on a fixed period rather than "N ms after the previous work finished",
so a slow tick cannot permanently shorten the interval, and the sleep is clamped so a slow
tick cannot make the loop spin to "catch up" and thrash the disk.
"""

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, SimpleQueue
from types import MappingProxyType
from typing import Self

from gputop.model.device import AmdgpuDevice, DeviceKind
from gputop.model.history import DEFAULT_HISTORY_LENGTH, RingBuffer
from gputop.model.metrics import (
    AmdgpuMetrics,
    Clock,
    Fan,
    MemoryPool,
    PcieLink,
    Power,
    Temperature,
)
from gputop.model.process import GpuProcess
from gputop.model.snapshot import GpuSnapshot, SamplerStats, SourceReport
from gputop.readers.discovery import discover_devices
from gputop.readers.dpm import (
    parse_power_profile,
    read_clock,
    read_clock_max,
    read_link,
    read_performance_level,
)
from gputop.readers.fsutil import read_int, read_text
from gputop.readers.hwmon import (
    read_fan,
    read_power,
    read_temperatures,
    sorted_temperature_names,
)
from gputop.readers.metrics_bin import GpuMetricsParser, ParsedGpuMetrics
from gputop.readers.procfs import ProcessCollector

DEFAULT_INTERVAL_S = 1.0

#: Cap on retained warnings so a permanently missing file cannot grow the snapshot without
#: bound.  A repeated warning is reported once, not on every tick.
MAX_WARNINGS = 8

#: Preference order for sensor presentation; ``sorted_temperature_names`` refines this.
_SENSOR_ORDER = ("edge", "junction", "mem", "soc", "core", "l3")


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerOptions:
    """Immutable sampler configuration."""

    interval_s: float = DEFAULT_INTERVAL_S
    history_length: int = DEFAULT_HISTORY_LENGTH
    collect_processes: bool = True
    drm_root: Path = Path("/sys/class/drm")
    proc_root: Path = Path("/proc")
    kind_override: DeviceKind | None = None
    name_overrides: Mapping[str, str] = field(default_factory=dict)

    def validated(self) -> SamplerOptions:
        """Return a copy with a usable interval, history length and root paths.

        The interval floor of 50 ms exists because a faster tick would spend more time in
        procfs than in the data it collects, and the procfs scan is the expensive part.
        """
        return SamplerOptions(
            interval_s=max(0.05, self.interval_s),
            history_length=max(1, self.history_length),
            collect_processes=self.collect_processes,
            drm_root=Path(self.drm_root),
            proc_root=Path(self.proc_root),
            kind_override=self.kind_override,
            name_overrides=dict(self.name_overrides),
        )


class Sampler:
    """Background sampler that publishes immutable snapshots.

    Args:
        options: Sampling configuration; see :class:`SamplerOptions`.

    The instance may be shared between threads provided only one thread calls
    :meth:`start` and :meth:`stop`, which is how both the CLI and the TUI use it.
    """

    def __init__(self, options: SamplerOptions | None = None) -> None:
        self._options = (options or SamplerOptions()).validated()
        self._queue: SimpleQueue[GpuSnapshot] = SimpleQueue()
        self._history: RingBuffer[GpuSnapshot] = RingBuffer(self._options.history_length)
        # Published for readers.  Rebound to a fresh immutable tuple after every tick,
        # so history() never iterates a deque the sampler thread is appending to.
        self._history_view: tuple[GpuSnapshot, ...] = ()
        self._parser = GpuMetricsParser()
        self._collector = ProcessCollector(self._options.proc_root)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._devices: tuple[AmdgpuDevice, ...] = ()
        self._sequence = 0
        self._stats = SamplerStats()
        self._warnings: list[str] = []
        self._seen_warnings: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Discover devices and start the sampling thread.

        Discovery happens here, on the caller's thread, so that a machine with no AMD GPU
        reports that immediately rather than after the first tick.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._devices = discover_devices(
            self._options.drm_root,
            kind_override=self._options.kind_override,
            name_overrides=dict(self._options.name_overrides),
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gputop-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to finish and wait for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- consuming ---------------------------------------------------------

    def latest(self) -> GpuSnapshot | None:
        """Return the newest published snapshot without blocking."""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def wait(self, timeout: float | None = None) -> GpuSnapshot | None:
        """Block until a snapshot arrives.

        Args:
            timeout: Maximum wait in seconds, or ``None`` to block indefinitely.

        Returns:
            A snapshot, or ``None`` if the timeout elapsed.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def history(self) -> tuple[GpuSnapshot, ...]:
        """Return the retained history, oldest first.

        Safe to call from any thread.  The sampler thread rebinds ``_history_view`` to
        a fresh immutable tuple after every tick, and readers only ever touch that
        tuple.  Returning the underlying ring buffer would mean iterating a ``deque``
        that the sampler thread is concurrently appending to, which is precisely the
        shared mutable state this design otherwise avoids.
        """
        return self._history_view

    @property
    def devices(self) -> tuple[AmdgpuDevice, ...]:
        """Devices discovered by :meth:`start`."""
        return self._devices

    @property
    def options(self) -> SamplerOptions:
        """The validated configuration in effect."""
        return self._options

    @property
    def stats(self) -> SamplerStats:
        """Rolling health counters, for display only.

        The value is a single frozen dataclass, so reading it from another thread cannot
        observe a half-updated state.
        """
        return self._stats

    def sample_once(self) -> GpuSnapshot:
        """Run exactly one tick synchronously and publish the result.

        This is what ``--dump`` and the tests use, so sampling is verifiable without any
        threading at all.
        """
        snapshot = self._tick()
        self._queue.put(snapshot)
        return snapshot

    def discover(self) -> tuple[AmdgpuDevice, ...]:
        """Discover devices without starting the thread, for one-shot use."""
        self._devices = discover_devices(
            self._options.drm_root,
            kind_override=self._options.kind_override,
            name_overrides=dict(self._options.name_overrides),
        )
        return self._devices

    # -- the thread body ---------------------------------------------------

    def _run(self) -> None:
        """Tick until stopped, sleeping on a fixed period."""
        period = self._options.interval_s
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self._queue.put(self._tick())
            except Exception:
                self._stats = _with_tick_error(self._stats)
            next_deadline += period
            delay = next_deadline - time.monotonic()
            self._stop.wait(min(max(delay, 0.0), period))

    def _tick(self) -> GpuSnapshot:
        """Perform one sample and return the immutable result."""
        started_ns = time.monotonic_ns()
        self._sequence += 1
        warnings: list[str] = []
        durations: list[float] = []
        samples: list[AmdgpuMetrics] = []

        for device in self._devices:
            device_started = time.monotonic_ns()
            try:
                samples.append(self._sample_device(device))
            except Exception as error:
                self._record_warning(
                    f"{device.card}: {type(error).__name__}: {error}", warnings
                )
            durations.append((time.monotonic_ns() - device_started) / 1e6)

        processes: tuple[GpuProcess, ...] = ()
        visible = 0
        total = 0
        if self._options.collect_processes:
            proc_started = time.monotonic_ns()
            try:
                collected, visible, total = self._collector.collect(time.monotonic_ns())
                processes = collected
            except Exception as error:
                self._record_warning(f"procfs: {type(error).__name__}: {error}", warnings)
            durations.append((time.monotonic_ns() - proc_started) / 1e6)

        elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
        self._stats = _advance(self._stats, durations, elapsed_ms)

        snapshot = GpuSnapshot(
            sequence=self._sequence,
            timestamp_ns=time.monotonic_ns(),
            devices=tuple(samples),
            processes=processes,
            visible_process_count=visible,
            total_process_count=total,
            stats=self._stats,
            source=SourceReport(
                metrics_abi=_dominant_abi(samples),
                per_metric=MappingProxyType({}),
            ),
            warnings=tuple(warnings),
        )
        # The history is owned by whichever thread runs the tick, which is always the same
        # thread, so no lock is needed.  Recording here rather than at the publish site
        # means ``sample_once`` populates history too.
        self._history.append(snapshot)
        # Snapshot the ring into an immutable tuple for cross-thread readers.
        self._history_view = self._history.items()
        return snapshot

    def _record_warning(self, message: str, current: list[str]) -> None:
        """Record a warning once, so a permanent fault does not flood the snapshot."""
        if message in self._seen_warnings:
            return
        self._seen_warnings.add(message)
        if len(current) < MAX_WARNINGS:
            current.append(message)

    # -- per device -------------------------------------------------------

    def _sample_device(self, device: AmdgpuDevice) -> AmdgpuMetrics:
        """Read every source for one device and normalise them into a single snapshot."""
        parsed = self._parser.parse_file(device.device_dir / "gpu_metrics")
        hwmon_temps = read_temperatures(device.hwmon_dir)
        hwmon_power, power_cap = read_power(device.hwmon_dir)
        hwmon_fan_rpm, fan_max, hwmon_fan_pwm = read_fan(device.hwmon_dir)

        sclk, sclk_conf = read_clock(device.device_dir, "current_sclk", "pp_dpm_sclk")
        mclk, mclk_conf = read_clock(device.device_dir, "current_mclk", "pp_dpm_mclk")
        socclk, socclk_conf = read_clock(device.device_dir, "current_socclk", "pp_dpm_socclk")

        vram_total = read_int(device.device_dir / "mem_info_vis_vram_total")
        if vram_total is None:
            vram_total = read_int(device.device_dir / "mem_info_vram_total")

        sclk, sclk_source = _prefer_binary(sclk, parsed.sclk_mhz if parsed else None, sclk_conf)
        mclk, mclk_source = _prefer_binary(mclk, parsed.mclk_mhz if parsed else None, mclk_conf)
        socclk, socclk_source = _prefer_binary(
            socclk, parsed.socclk_mhz if parsed else None, socclk_conf
        )

        return AmdgpuMetrics(
            device=device,
            sequence=self._sequence,
            timestamp_ns=time.monotonic_ns(),
            interval_s=self._options.interval_s,
            gpu_busy_percent=_first_not_none(
                parsed.gpu_activity_percent if parsed else None,
                _percent_attribute(device.device_dir, "gpu_busy_percent"),
            ),
            mem_busy_percent=_first_not_none(
                parsed.umc_activity_percent if parsed else None,
                _percent_attribute(device.device_dir, "mem_busy_percent"),
            ),
            vram=MemoryPool(
                used=read_int(device.device_dir / "mem_info_vis_vram_used"),
                total=vram_total,
                source="sysfs",
            ),
            gtt=MemoryPool(
                used=read_int(device.device_dir / "mem_info_gtt_used"),
                total=read_int(device.device_dir / "mem_info_gtt_total"),
                source="sysfs",
            ),
            sclk=Clock(
                current=sclk,
                maximum=read_clock_max(device.device_dir, "current_sclk_max", "pp_dpm_sclk"),
                source=sclk_source,
            ),
            mclk=Clock(
                current=mclk,
                maximum=read_clock_max(device.device_dir, "current_mclk_max", "pp_dpm_mclk"),
                source=mclk_source,
            ),
            socclk=Clock(
                current=socclk,
                maximum=read_clock_max(
                    device.device_dir, "current_socclk_max", "pp_dpm_socclk"
                ),
                source=socclk_source,
            ),
            voltages_mv=parsed.voltages_mv if parsed else (),
            temperatures=_merge_temperatures(hwmon_temps, parsed),
            power=Power(
                draw_w=_first_not_none(parsed.power_draw_w if parsed else None, hwmon_power),
                cap_w=power_cap,
                energy_mj=parsed.energy_mj if parsed else None,
                source="gpu_metrics" if parsed and parsed.power_draw_w is not None else "hwmon",
            ),
            fan=Fan(
                rpm=_first_not_none(parsed.fan_rpm if parsed else None, hwmon_fan_rpm),
                max_rpm=fan_max,
                pwm=_first_not_none(parsed.fan_pwm_percent if parsed else None, hwmon_fan_pwm),
            ),
            performance_level=read_performance_level(device.device_dir),
            power_profile=parse_power_profile(
                read_text(device.device_dir / "pp_power_profile_mode")
            ),
            throttle=parsed.throttle if parsed else None,
            pcie=_pcie_link(device, parsed),
            metrics_abi=parsed.abi_label if parsed else None,
            unavailable=frozenset(),
        )


# -- helpers --------------------------------------------------------------


def _first_not_none[T](*values: T | None) -> T | None:
    """Return the first argument that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def _prefer_binary(
    sysfs_value: int | None, binary_value: int | None, confidence: int
) -> tuple[int | None, str]:
    """Choose a clock value and label its source.

    ``gpu_metrics`` wins when it has a value; otherwise the sysfs attribute (confidence
    2) wins over the DPM table (confidence 1).
    """
    if binary_value is not None:
        return binary_value, "gpu_metrics"
    if sysfs_value is not None and confidence == 2:
        return sysfs_value, "sysfs"
    if sysfs_value is not None:
        return sysfs_value, "dpm"
    return None, "none"


def _pcie_link(device: AmdgpuDevice, parsed: ParsedGpuMetrics | None) -> PcieLink | None:
    """Resolve the PCIe link, preferring the binary source."""
    if parsed is not None and parsed.pcie is not None:
        return parsed.pcie
    width, speed = read_link(device.device_dir)
    if width and speed:
        return PcieLink(width=width, speed=speed)
    return None


def _merge_temperatures(
    hwmon: dict[str, float], parsed: ParsedGpuMetrics | None
) -> tuple[Temperature, ...]:
    """Merge hwmon and ``gpu_metrics`` sensors, preferring the binary source.

    hwmon is read first so a sensor that only hwmon knows about -- several boards expose
    VRM rails there -- is not lost when ``gpu_metrics`` is available.
    """
    merged: dict[str, Temperature] = {
        name: Temperature(label=name, celsius=value, source="hwmon")
        for name, value in hwmon.items()
    }
    if parsed is not None:
        for temp in parsed.temperatures:
            merged[temp.label] = temp
    return tuple(merged[name] for name in sorted_temperature_names(merged.keys()))


def _percent_attribute(device_dir: Path, attribute: str) -> float | None:
    """Read a 0-100 sysfs activity attribute."""
    value = read_int(device_dir / attribute)
    if value is None or not 0 <= value <= 100:
        return None
    return float(value)


def _dominant_abi(samples: list[AmdgpuMetrics]) -> str | None:
    """Return the ABI used by the first device that has one, for the footer."""
    for entry in samples:
        if entry.metrics_abi is not None:
            return entry.metrics_abi
    return None


def _advance(stats: SamplerStats, durations: list[float], elapsed_ms: float) -> SamplerStats:
    """Fold this tick's timings into the rolling counters."""
    samples = durations or [elapsed_ms]
    return SamplerStats(
        ticks=stats.ticks + 1,
        tick_errors=stats.tick_errors,
        read_errors=stats.read_errors,
        last_duration_ms=elapsed_ms,
        avg_duration_ms=(stats.avg_duration_ms * stats.ticks + sum(samples))
        / (stats.ticks + 1),
    )


def _with_tick_error(stats: SamplerStats) -> SamplerStats:
    """Record a failed tick while leaving the timing history intact."""
    return SamplerStats(
        ticks=stats.ticks,
        tick_errors=stats.tick_errors + 1,
        read_errors=stats.read_errors,
        last_duration_ms=stats.last_duration_ms,
        avg_duration_ms=stats.avg_duration_ms,
    )


__all__ = ["DEFAULT_INTERVAL_S", "Sampler", "SamplerOptions"]
