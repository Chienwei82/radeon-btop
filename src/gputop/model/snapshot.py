"""The immutable snapshot object handed from the sampler thread to the UI."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from gputop.model.aliases import Nanoseconds
from gputop.model.blocks import BlocksStatus
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.process import GpuProcess

_EMPTY: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceReport:
    """Which source each metric family was resolved from for one sample."""

    metrics_abi: str | None = None
    per_metric: Mapping[str, str] = field(default=_EMPTY)

    @property
    def using_binary_metrics(self) -> bool:
        """True when the rich ``gpu_metrics`` path produced this sample."""
        return self.metrics_abi is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerStats:
    """Rolling health counters for the sampler thread."""

    ticks: int = 0
    tick_errors: int = 0
    read_errors: int = 0
    last_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuSnapshot:
    """A complete, immutable view of all GPUs and their clients at one instant.

    Instances are created by the sampler thread and never mutated afterwards, which is
    what makes them safe to publish across a thread boundary on the free-threaded build.
    """

    sequence: int
    timestamp_ns: Nanoseconds
    devices: tuple[AmdgpuMetrics, ...] = ()
    processes: tuple[GpuProcess, ...] = ()
    visible_process_count: int = 0
    total_process_count: int = 0
    stats: SamplerStats = SamplerStats()
    source: SourceReport = SourceReport()
    warnings: tuple[str, ...] = ()
    #: Whether the optional radeontop-backed block data is available, and if not, why.
    #:
    #: Carried on the snapshot rather than queried by the UI, because the UI performs no I/O
    #: and must not spawn a process to find out whether it could.  One status covers the
    #: whole machine: a per-device status would let the panel render partial data with no
    #: indication that a card was missing, which is the more misleading of the two options.
    blocks_status: BlocksStatus = BlocksStatus.DISABLED
    #: A one-line explanation of ``blocks_status`` when it is not :attr:`BlocksStatus.OK`.
    blocks_hint: str = ""
    #: Wall-clock nanoseconds, recorded at the same instant as ``timestamp_ns``.
    #:
    #: Kept alongside the monotonic clock rather than derived from it because a monotonic
    #: clock has no defined relationship to a calendar: only the process start offset links
    #: the two, and that offset drifts with NTP adjustments over a long session.  This is
    #: what a session recording writes into its time column; nothing that compares intervals
    #: should read it.
    wall_clock_ns: int = 0

    def device_by_index(self, index: int) -> AmdgpuMetrics | None:
        """Return the sample for a device index, or ``None`` when out of range."""
        for metrics in self.devices:
            if metrics.device.index == index:
                return metrics
        return None

    @property
    def is_partial_process_view(self) -> bool:
        """True when ``/proc`` permissions hid processes from this user."""
        return self.total_process_count > self.visible_process_count
