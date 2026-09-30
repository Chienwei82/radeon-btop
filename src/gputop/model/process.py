"""Per-process GPU accounting derived from ``/proc/<pid>/fdinfo``."""

from dataclasses import dataclass

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
        name: Process name from ``/proc/<pid>/comm``.
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
    engines: tuple[EngineUsage, ...] = ()
    vram_used: Bytes | None = None
    vram_shared: Bytes | None = None
    gtt_used: Bytes | None = None
    gtt_shared: Bytes | None = None
    cpu_used: Bytes | None = None

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
        """Utilisation for a normalised engine bucket, or ``None`` when unseen."""
        values = [
            u.percent for u in self.engines if u.engine == bucket and u.percent is not None
        ]
        if not values:
            return None
        return max(values)
