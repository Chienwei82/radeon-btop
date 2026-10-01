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
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from types import MappingProxyType
from typing import Self

from gputop.model.blocks import BlocksStatus, GpuBlocks
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
from gputop.readers.discovery import discover_devices, select_devices
from gputop.readers.dpm import (
    parse_power_profile,
    read_clock_pair,
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
from gputop.readers.pp import read_odc_voltage, read_power_profile_table
from gputop.readers.procfs import ProcessCollector
from gputop.readers.radeontop import (
    STARTUP_GRACE_S,
    RadeontopOptions,
    RadeontopPool,
    wait_for_first_sample,
)

DEFAULT_INTERVAL_S = 1.0

#: Cap on retained warnings so a permanently missing file cannot grow the snapshot without
#: bound.  A repeated warning is reported once, not on every tick.
MAX_WARNINGS = 8

#: Cap on the set of already-reported warning messages, so a fault whose text embeds a
#: varying value -- a PID, a path -- cannot grow it without bound over a long session.
#: Reaching it is not a real possibility for a healthy machine: the messages come from a
#: fixed set of per-device and per-subsystem reads.
MAX_SEEN_WARNINGS = 256


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
    #: Restrict monitoring to these cards, from the ``gpu.devices`` config key.  Empty is
    #: every card discovery finds.  Names are matched by :func:`select_devices` against
    #: each device's PCI address in its full and shortened forms and its ``cardN`` node.
    only_devices: tuple[str, ...] = ()
    #: How to run radeontop for the optional blocks panel.
    radeontop: RadeontopOptions = field(default_factory=RadeontopOptions)

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
            only_devices=tuple(self.only_devices),
            radeontop=self.radeontop.validated(),
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
        # Read by the tick loop every iteration, so ``set_interval`` can change the rate
        # without a second timer that would race the one the UI owns.
        self._interval_s: float = self._options.interval_s
        self._history: RingBuffer[GpuSnapshot] = RingBuffer(self._options.history_length)
        # Published for readers.  Rebound to a fresh immutable tuple after every tick,
        # so history() never iterates a deque the sampler thread is appending to.
        self._history_view: tuple[GpuSnapshot, ...] = ()
        self._parser = GpuMetricsParser()
        self._collector = ProcessCollector(self._options.proc_root)
        self._blocks = RadeontopPool(self._options.radeontop)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Poked to cut the tick loop's sleep short, by :meth:`stop` (so shutdown is
        # immediate) and by :meth:`set_interval` (so a new rate is not stuck behind the
        # remainder of the old one).
        self._resync = threading.Event()
        self._devices: tuple[AmdgpuDevice, ...] = ()
        #: Why ``gpu.devices`` named a card this machine does not have, if it did.
        #:
        #: A start-up condition rather than a per-tick warning, on the same reasoning as
        #: :attr:`blocks_hint`: the card set is enumerated once and cannot change without a
        #: restart, so the condition is permanent, and a permanent message in the snapshot
        #: warning list would hold one of its eight slots for the whole session and push
        #: out a real read error that happened once.  Read once at start-up by whichever
        #: of the three entry points is running.
        self._filter_warning = ""
        self._sequence = 0
        self._stats = SamplerStats()
        self._seen_warnings: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Discover devices and start the sampling thread.

        Discovery happens here, on the caller's thread, so that a machine with no AMD GPU
        reports that immediately rather than after the first tick.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._devices = self._discover()
        # Configured from the discovered devices, so it is set up here rather than in
        # __init__: the children are spawned once, and spawning them against a device list
        # that does not exist yet would start a process per card of nothing.
        self._configure_blocks()
        self._blocks.start()
        self._stop.clear()
        self._resync.clear()
        self._thread = threading.Thread(target=self._run, name="gputop-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to finish and wait for it."""
        self._stop.set()
        self._resync.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            # The join timed out -- a tick was still inside /proc or a radeontop pipe.
            # The handle is deliberately *kept*: clearing it would let a later start()
            # spawn a second tick thread beside this one, and start() clears _stop, so the
            # orphan would never see the stop and would run for the life of the process,
            # two threads racing on the sequence counter, the history and the collector
            # baseline.  Leaving the handle set makes start() refuse instead.
            return
        self._thread = None
        # Stopped after the sampling thread, not before: the pool is polled from inside a
        # tick, so tearing it down while a tick is in flight would race the child handles.
        self._blocks.stop()

    def set_interval(self, interval_s: float) -> float:
        """Change the sampling period of an already running sampler.

        The tick loop reads ``self._interval_s`` at the top of every iteration, so
        changing it takes effect on the next tick without stopping the thread and
        without a second timer racing the first.  A negative or absurd value is clamped
        rather than rejected, because the only caller is a key press and the clamp is
        exactly the behaviour the key advertises.
        """
        self._interval_s = max(0.05, min(interval_s, 600.0))
        self._options = replace(self._options, interval_s=self._interval_s)
        # Cut the loop's current sleep short: re-reading the period only helps once the
        # loop is awake, and without this a change from ten seconds to 100 ms waits out
        # the remainder of the ten.
        self._resync.set()
        return self._interval_s

    def request_tick(self) -> None:
        """Ask a running sampler for a sample now, without waiting for the period.

        The tick loop sleeps on :attr:`_resync`, so setting it is enough to cut the wait
        short.  This is what the UI's refresh key uses, and it is deliberately not
        :meth:`sample_once`: that runs a whole tick on the caller's thread, which for the
        UI thread means a second tick running beside the sampler's own, sharing the
        sequence counter, the history ring and the collector's delta baseline.
        """
        self._resync.set()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- consuming ---------------------------------------------------------

    def latest(self) -> GpuSnapshot | None:
        """Return the newest published snapshot without blocking.

        The queue is drained rather than popped once.  A slow repaint can leave several
        snapshots queued, and returning the *oldest* of them would show stale data while
        discarding newer samples; draining makes this the counterpart the docstring claims
        it is and leaves the queue holding only what arrives after the caller looked.
        """
        newest: GpuSnapshot | None = None
        while True:
            try:
                newest = self._queue.get_nowait()
            except Empty:
                return newest

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
    def filter_warning(self) -> str:
        """Why a ``gpu.devices`` entry matched no card, or ``""`` when all of them did.

        Set by :meth:`discover` and :meth:`start`, so it is meaningful only once one of
        those has run.
        """
        return self._filter_warning

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

        It must not be called while the sampling thread is running.  A tick mutates the
        sequence counter, the history ring, the rolling statistics and the collector's
        delta baseline, all of which the design above gives to a single owner; running two
        concurrently computes engine deltas against the wrong baseline and loses a tick from
        the statistics.  :meth:`request_tick` is the way to ask a *running* sampler for a
        sample now.
        """
        snapshot = self._tick()
        self._queue.put(snapshot)
        return snapshot

    def discover(self) -> tuple[AmdgpuDevice, ...]:
        """Discover devices without starting the thread, for one-shot use."""
        self._devices = self._discover()
        # The pool is configured but not started.  A one-shot caller decides for itself
        # whether spawning radeontop is worth the wait, so ``--dump`` only does so when the
        # blocks panel was actually asked for.
        self._configure_blocks()
        return self._devices

    def _discover(self) -> tuple[AmdgpuDevice, ...]:
        """Enumerate the cards and apply the ``gpu.devices`` filter.

        Both call sites share this so the filter cannot be honoured on one path and
        ignored on the other, which is the failure mode a filter has: ``--dump`` listing
        three cards the interface is not watching reads as two different machines.
        """
        found = discover_devices(
            self._options.drm_root,
            kind_override=self._options.kind_override,
            name_overrides=dict(self._options.name_overrides),
        )
        selected, unmatched = select_devices(found, self._options.only_devices)
        if unmatched:
            available = ", ".join(sorted(device.bdf or device.card for device in found))
            self._filter_warning = (
                f"gpu.devices: no such device: {', '.join(unmatched)}"
                f"{f' (found: {available})' if available else ' (no devices found)'}"
            )
        else:
            self._filter_warning = ""
        return selected

    def _configure_blocks(self) -> None:
        """Point the radeontop pool at the currently discovered devices."""
        self._blocks.configure(
            [(str(device.index), device.card, device.bdf) for device in self._devices]
        )

    def start_blocks(self) -> BlocksStatus:
        """Start the radeontop pool against the discovered devices."""
        self._configure_blocks()
        self._blocks.start()
        return self._blocks.status()

    def await_blocks(self, timeout_s: float = STARTUP_GRACE_S) -> bool:
        """Wait for the block pool to produce its first sample.

        For the one-shot CLI paths, which have no sampling thread to accumulate the wait
        for.  A timeout is not an error: the caller still emits a snapshot, just without
        block data and with the status saying why.
        """
        return wait_for_first_sample(self._blocks, timeout_s)

    def blocks_status(self) -> BlocksStatus:
        """The radeontop pool's current state, for ``--check`` and for diagnostics."""
        return self._blocks.status()

    def blocks_hint(self) -> str:
        """The human explanation of :meth:`blocks_status`; empty when all is well."""
        return self._blocks_hint(self.blocks_status())

    # -- the thread body ---------------------------------------------------

    def _run(self) -> None:
        """Tick until stopped, sleeping on a fixed period.

        The period is re-read from ``self._interval_s`` at the top of every iteration
        rather than captured into a local once.  A cached copy looks harmless and is not:
        :meth:`set_interval` updated the field and every caller read the new value back, so
        the ``+`` and ``-`` keys reported the interval they had just set while the loop
        carried on sampling at the rate the thread started with.

        The sleep waits on ``_resync`` rather than on ``_stop`` alone.  Re-reading the
        period is not enough on its own: a change made while the loop is inside its sleep
        waits out whatever is left of the *old* period, so dropping from ten seconds to
        100 ms could take ten seconds to take effect.  ``set_interval`` pokes ``_resync``
        to cut that short, and ``stop`` pokes it too, so shutting down is still immediate.
        """
        period = self._interval_s
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            # Armed before the tick, so a change made during the tick is not lost between
            # here and the wait.  A change made before this line is picked up by the read
            # below anyway, so there is no window in which it is dropped entirely.
            self._resync.clear()
            current = self._interval_s
            if current != period:
                period = current
                next_deadline = time.monotonic()
            try:
                self._queue.put(self._tick())
            except Exception:
                self._stats = _with_tick_error(self._stats)
            next_deadline += period
            delay = next_deadline - time.monotonic()
            self._resync.wait(min(max(delay, 0.0), period))

    def _tick(self) -> GpuSnapshot:
        """Perform one sample and return the immutable result."""
        started_ns = time.monotonic_ns()
        self._sequence += 1
        warnings: list[str] = []
        durations: list[float] = []
        samples: list[AmdgpuMetrics] = []

        # Polled before the devices are read so a block sample that is already waiting is
        # attached to the same tick that will display it, rather than one tick late.
        blocks_status = self._blocks.poll()

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
            blocks_status=blocks_status,
            blocks_hint=self._blocks_hint(blocks_status),
            wall_clock_ns=time.time_ns(),
        )
        # The history is owned by whichever thread runs the tick, which is always the same
        # thread, so no lock is needed.  Recording here rather than at the publish site
        # means ``sample_once`` populates history too.
        self._history.append(snapshot)
        # Snapshot the ring into an immutable tuple for cross-thread readers.
        self._history_view = self._history.items()
        return snapshot

    def _record_warning(self, message: str, current: list[str]) -> None:
        """Record a warning once, so a permanent fault does not flood the snapshot.

        Also the only producer of :attr:`SamplerStats.read_errors`, which is the count a
        user looks at to decide whether a metric that says ``N/A`` means "absent" or
        "unreadable"; leaving it permanently at zero would make it look like the read path
        had never failed.
        """
        if message in self._seen_warnings:
            return
        self._seen_warnings.add(message)
        if len(self._seen_warnings) > MAX_SEEN_WARNINGS:
            # Bounded rather than capped per snapshot: this set is a de-duplicator, and a
            # message that embeds a changing value (a PID, a path) would otherwise add a
            # fresh entry every few seconds for as long as the monitor ran.
            self._seen_warnings.clear()
        self._stats = _with_read_error(self._stats)
        if len(current) < MAX_WARNINGS:
            current.append(message)

    def _blocks_hint(self, status: BlocksStatus) -> str:
        """The explanation shown where the blocks panel would be.

        The hint is attached to every snapshot rather than raised as a warning, because it
        is a permanent condition rather than a fault: it would occupy one of the eight
        warning slots for the whole session and push out a real error that happened once.
        """
        if status.ok:
            return ""
        return self._blocks.diagnostic() or status.hint

    def blocks_for(self, device: AmdgpuDevice) -> GpuBlocks | None:
        """The latest block sample for a device, or ``None`` when there is none.

        Only reads the pool's immutable published values; it does no I/O and does not wait.
        """
        return self._blocks.blocks_for(str(device.index))

    # -- per device -------------------------------------------------------

    def _sample_device(self, device: AmdgpuDevice) -> AmdgpuMetrics:
        """Read every source for one device and normalise them into a single snapshot."""
        parsed = self._parser.parse_file(device.device_dir / "gpu_metrics")
        hwmon_temps = read_temperatures(device.hwmon_dir)
        hwmon_power, power_cap = read_power(device.hwmon_dir)
        hwmon_fan_rpm, fan_max, hwmon_fan_pwm = read_fan(device.hwmon_dir)

        # Current value and ceiling come from one pass over the fallback chain: on recent
        # drivers the direct attributes are gone, so reading the DPM table twice per
        # domain was six of the reads this function made.
        sclk, sclk_max, sclk_conf = read_clock_pair(
            device.device_dir, "current_sclk", "pp_dpm_sclk"
        )
        mclk, mclk_max, mclk_conf = read_clock_pair(
            device.device_dir, "current_mclk", "pp_dpm_mclk"
        )
        socclk, socclk_max, socclk_conf = read_clock_pair(
            device.device_dir, "current_socclk", "pp_dpm_socclk"
        )

        vram_total = read_int(device.device_dir / "mem_info_vis_vram_total")
        if vram_total is None:
            vram_total = read_int(device.device_dir / "mem_info_vram_total")

        profiles = read_power_profile_table(device.device_dir)

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
            sclk=Clock(current=sclk, maximum=sclk_max, source=sclk_source),
            mclk=Clock(current=mclk, maximum=mclk_max, source=mclk_source),
            socclk=Clock(current=socclk, maximum=socclk_max, source=socclk_source),
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
            power_profile=profiles.active_name
            or parse_power_profile(read_text(device.device_dir / "pp_power_profile_mode")),
            throttle=parsed.throttle if parsed else None,
            pcie=_pcie_link(device, parsed),
            metrics_abi=parsed.abi_label if parsed else None,
            unavailable=frozenset(),
            blocks=self.blocks_for(device),
            odc=read_odc_voltage(device.device_dir),
            profiles=profiles,
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


def _with_read_error(stats: SamplerStats) -> SamplerStats:
    """Record a failed per-device or per-subsystem read."""
    return SamplerStats(
        ticks=stats.ticks,
        tick_errors=stats.tick_errors,
        read_errors=stats.read_errors + 1,
        last_duration_ms=stats.last_duration_ms,
        avg_duration_ms=stats.avg_duration_ms,
    )


__all__ = ["DEFAULT_INTERVAL_S", "Sampler", "SamplerOptions"]
