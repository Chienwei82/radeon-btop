"""Per-process GPU accounting derived from ``/proc/<pid>/fdinfo``."""

from dataclasses import dataclass
from typing import Literal

from gputop.model.aliases import Bdf, Bytes, ClientId, Nanoseconds, Percent


@dataclass(frozen=True, slots=True)
class EngineUsage:
    """Cumulative and per-interval time spent on one GPU engine.

    ``total_ns`` comes straight from the ``drm-engine-<name>`` key and only ever grows.
    ``delta_ns`` is the difference against the previous sample and ``window_ns`` is the
    wall-clock duration of the sampling window, so utilisation is self-contained.

    The kernel only exposes *cumulative* engine time -- there is no period field -- so a
    client seen for the first time has no baseline and reports ``None`` rather than ``0``.
    """

    engine: str
    total_ns: Nanoseconds
    delta_ns: Nanoseconds
    window_ns: Nanoseconds

    @property
    def percent(self) -> Percent | None:
        """Utilisation over the sampling window, or ``None`` without a baseline.

        The result is clamped to 100: ``drm-engine-*`` counts time *on* the engine, and
        concurrent work can legitimately push a delta past the wall-clock window.
        """
        if self.window_ns <= 0 or self.total_ns <= 0:
            return None
        return min(100.0, max(0.0, self.delta_ns / self.window_ns * 100.0))


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuProcess:
    """One DRM client, deduplicated by ``(drm-pdev, drm-client-id)``.

    Attributes:
        pid: PID of a process holding a descriptor onto the client.  When several
            processes share one client id, the lowest PID is reported.
        name: Process name from ``/proc/<pid>/comm``, which the kernel truncates to
            15 characters and which is *not* unique.
        cmdline: Full argument vector from ``/proc/<pid>/cmdline``.  Empty for a kernel
            thread, which is the only root-free way to tell one apart -- see
            :attr:`is_kernel_thread`.
        ppid: Parent PID from ``/proc/<pid>/status``, or ``None`` when unreadable.  This
            is what the process tree is built from.
        user: Owning user name.
        bdf: PCI address from ``drm-pdev``; routes the client to a specific GPU.
        client_id: The ``drm-client-id`` value.
        engines: Per-engine usage, empty when the client has never been scheduled.
        vram_used: ``drm-resident-vram`` if present, else ``drm-total-vram``.
        vram_shared: ``drm-shared-vram``.
        gtt_used: ``drm-resident-gtt`` if present, else ``drm-total-gtt``.
        gtt_shared: ``drm-shared-gtt``.
        cpu_used: ``drm-resident-cpu`` if present, else ``drm-total-cpu``.  This is the
            pool an APU uses for system memory and is the one that matters on an iGPU,
            where it is usually larger than VRAM.
    """

    pid: int
    name: str
    user: str
    bdf: Bdf
    client_id: ClientId
    cmdline: tuple[str, ...] = ()
    ppid: int | None = None
    engines: tuple[EngineUsage, ...] = ()
    vram_used: Bytes | None = None
    vram_shared: Bytes | None = None
    gtt_used: Bytes | None = None
    gtt_shared: Bytes | None = None
    cpu_used: Bytes | None = None

    @property
    def identity(self) -> tuple[Bdf, ClientId]:
        """The row's stable key: the dedup pair the kernel itself reported.

        Used to keep a selection or a highlight attached to the same client across
        samples, where the row index moves as soon as anything else changes.
        """
        return (self.bdf, self.client_id)

    @property
    def command(self) -> str:
        """The full command line, falling back to the truncated ``comm`` name.

        ``comm`` is capped at 15 characters by the kernel, which is why a process table
        built from it alone cannot distinguish ``/usr/bin/python3`` from
        ``/usr/bin/python3.13``.  The command line is the disambiguating view, and it is
        empty for kernel threads -- hence the fallback rather than an empty column.
        """
        return " ".join(self.cmdline) if self.cmdline else self.name

    @property
    def is_kernel_thread(self) -> bool:
        """Whether this looks like a kernel thread.

        A kernel thread has no argument vector, so ``/proc/<pid>/cmdline`` reads back
        empty.  That is the only distinction available without root: ``status`` has no
        "kernel" flag, and the thread group leader check needs nothing more than this.
        """
        return not self.cmdline

    @property
    def is_idle(self) -> bool:
        """Whether the client is doing nothing measurable this interval.

        A client that merely *exists* -- a compositor holding a descriptor, a daemon that
        opened a render node at startup -- is not using the GPU.  Those rows are noise in
        a list whose purpose is to answer "what is my GPU doing", so they are hidden by
        default and can be brought back with one key.

        Engine utilisation is ``0.0`` rather than ``None`` for a client that has never
        been scheduled, so a freshly started gputop shows a real table immediately
        instead of an empty one.
        """
        return self.engine_percent <= 0.0 and self.memory_used <= 0

    @property
    def engine_percent(self) -> Percent:
        """Sum of per-engine utilisation, clamped to 100.

        A client genuinely running on both gfx and compute is using both, so the value is
        a sum rather than a maximum.  The clamp is required because a multi-engine delta
        can legitimately exceed the wall-clock window.
        """
        total = 0.0
        seen = False
        for usage in self.engines:
            value = usage.percent
            if value is not None:
                total += value
                seen = True
        if not seen:
            return 0.0
        return min(100.0, total)

    @property
    def memory_used(self) -> Bytes:
        """Total resident memory attributed to this client.

        CPU-visible memory is included because on an APU it is the dominant pool, and
        excluding it would report an iGPU client as using almost nothing.
        """
        return (self.vram_used or 0) + (self.gtt_used or 0) + (self.cpu_used or 0)

    def engine_percent_for(self, bucket: str) -> Percent | None:
        """Utilisation for a normalised engine bucket, or ``None`` when unseen.

        Several raw engines land in one bucket -- ``sdma0`` and ``sdma1`` are both
        ``dma`` -- and a client using both is using both, so the bucket sums them the
        way :attr:`engine_percent` sums every engine.  Taking the maximum instead
        under-reported the column against the total the rows are ordered by: two SDMA
        engines at 30% + 20% read as 30% here and 50% there.
        """
        values = [
            u.percent for u in self.engines if u.engine == bucket and u.percent is not None
        ]
        if not values:
            return None
        return min(100.0, sum(values))

    def bucket_percent(self, *buckets: str) -> Percent | None:
        """Summed utilisation across several buckets, or ``None`` when none are known.

        A column that groups engines -- encode and decode share one, because a card
        doing both is doing *video* work and the split is rarely the interesting part --
        has to add them the same way :attr:`engine_percent` does.  Reporting ``None``
        rather than ``0`` keeps the "no baseline yet" case distinguishable from "this
        client is not using the engine".
        """
        total = 0.0
        seen = False
        for bucket in buckets:
            value = self.engine_percent_for(bucket)
            if value is not None:
                total += value
                seen = True
        if not seen:
            return None
        return min(100.0, total)

    @property
    def enc_dec_percent(self) -> Percent | None:
        """Video encode plus decode utilisation, the VCN column of the table."""
        return self.bucket_percent("enc", "dec")


#: Engine buckets, in display order.  Kept beside the model so the table, the filter and
#: the tests agree on spelling without duplicating string literals.
ENGINE_BUCKETS: tuple[str, ...] = ("gfx", "compute", "dma", "enc", "dec", "other")

#: Every value the process table can be ordered by.
#:
#: The vocabulary lives in the model rather than in the view because the *config file*
#: has to validate it: ``process.sort`` is a string a user types, and checking it against
#: a set the UI also uses is the only way for the two to stay in step.  ``total`` is the
#: sum across engines and is sortable but never displayed -- the four engine columns
#: already say it, and a tenth column repeating the sum would cost width for nothing.
type SortColumn = Literal[
    "total", "pid", "user", "command", "gfx", "compute", "dma", "encdc", "vram", "gtt"
]

SORT_COLUMNS: tuple[SortColumn, ...] = (
    "total",
    "pid",
    "user",
    "command",
    "gfx",
    "compute",
    "dma",
    "encdc",
    "vram",
    "gtt",
)
