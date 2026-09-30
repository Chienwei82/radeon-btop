"""Per-process GPU accounting from ``/proc/<pid>/fdinfo``.

The kernel exposes two things that make this possible without root:

* ``drm-client-id`` identifies a DRM client.  Several file descriptors -- and even
  several processes -- can reference the same client, so all records sharing a
  ``(drm-pdev, drm-client-id)`` pair are merged into one row, taking the **maximum** of
  each field rather than the sum.  Summing would double-count.
* ``drm-engine-<name>: <ns>`` is *cumulative* engine time.  There is no period field on
  this kernel, so utilisation must be derived from a delta against the previous sample.

``drm-pdev`` routes each client to a specific GPU, which is what makes multi-GPU and
dGPU+APU accounting correct rather than a single blended number.
"""

import os
import pwd
import re
from dataclasses import dataclass, field
from pathlib import Path

from gputop.model.aliases import Bdf, ClientId, Nanoseconds
from gputop.model.process import EngineUsage, GpuProcess
from gputop.readers.fsutil import parse_drm_size

_ENGINE_ACTIVE_KEY = re.compile(r"^drm-engine-active-([a-z0-9_]+)$")
_ENGINE_PERIOD_KEY = re.compile(r"^drm-engine-period-([a-z0-9_]+)$")
_ENGINE_KEY = re.compile(r"^drm-engine-([a-z0-9_]+)$")

#: A descriptor is on a GPU when its target is under ``/dev/dri``.  Matching on the
#: substring rather than a suffix matters: the target is ``/dev/dri/renderD128``, so a
#: suffix test against ``/dri/renderD`` can never match.
DRM_NODE_MARKERS = ("/dev/dri/renderD", "/dev/dri/card")

#: Driver engine names mapped onto the display buckets.  Anything unrecognised becomes
#: ``other`` but keeps its raw name, so a new engine still shows up instead of vanishing.
_ENGINE_BUCKETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gfx",), "gfx"),
    (("compute",), "compute"),
    (("sdma0", "sdma1", "sdma", "dma"), "dma"),
    (("vangogh", "vcnenc", "venc", "vcn0.enc", "encode"), "enc"),
    (("jpeg", "vdec", "vdpdec", "uvd", "va", "decode", "vcn0.dec"), "dec"),
    (("video", "vcn", "vcn0"), "other"),
)

_BUCKET_BY_ENGINE: dict[str, str] = {
    engine: bucket for engines, bucket in _ENGINE_BUCKETS for engine in engines
}

#: Memory keys parsed out of one fdinfo file.  The driver emits a full cross product of
#: ``drm-{total,shared,resident,purgeable,memory}-{vram,gtt,cpu,...}`` plus the
#: AMD-specific ``amd-requested-*`` / ``amd-evicted-*``; matching the whole family by
#: pattern keeps the parser correct when a new pool name is added.
_MEMORY_KEY = re.compile(
    r"^(?:drm-(?:total|shared|resident|purgeable|memory)|amd-requested|amd-evicted)-([a-z]+)$"
)


def normalise_engine(raw_name: str) -> str:
    """Map a driver engine name onto a display bucket.

    Args:
        raw_name: The name as it appears in ``drm-engine-<name>``.

    Returns:
        One of ``gfx``, ``compute``, ``dma``, ``enc``, ``dec`` or ``other``.
    """
    name = raw_name.strip().lower()
    if name in _BUCKET_BY_ENGINE:
        return _BUCKET_BY_ENGINE[name]
    if name.startswith("sdma") or name.startswith("dma"):
        return "dma"
    if "enc" in name:
        return "enc"
    if "dec" in name or "jpeg" in name:
        return "dec"
    return "other"


@dataclass(slots=True)
class _ClientAccumulator:
    """Mutable scratch state used only while merging one scan's worth of records."""

    pid: int
    name: str
    user: str
    bdf: Bdf
    client_id: ClientId
    engines: dict[str, int] = field(default_factory=dict)
    memory: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class FdinfoRecord:
    """One parsed ``fdinfo`` file for a DRM file descriptor."""

    driver: str
    client_id: ClientId
    pdev: Bdf
    engines: dict[str, int]
    memory: dict[str, int]


def _leading_int(value: str) -> int | None:
    """Extract the leading integer from a value that may carry a unit.

    The amdgpu driver writes engine counters as ``173321 ns``, so parsing the whole
    string with ``int()`` would raise and take the entire sample down with it.  A reader
    must never raise, so the unit is discarded and a non-numeric value yields ``None``.
    """
    token = value.split()
    if not token:
        return None
    try:
        return int(token[0])
    except ValueError:
        return None


def parse_fdinfo(text: str) -> FdinfoRecord | None:
    """Parse the DRM accounting keys out of one ``fdinfo`` file.

    Args:
        text: The file contents.

    Returns:
        A record, or ``None`` when the file is not DRM accounting data or belongs to a
        different driver.
    """
    driver: str | None = None
    client_id: ClientId | None = None
    pdev: Bdf | None = None
    engines: dict[str, int] = {}
    active: dict[str, int] = {}
    period: dict[str, int] = {}
    memory: dict[str, int] = {}

    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()

        if key == "drm-driver":
            driver = value
        elif key == "drm-client-id":
            client_id = _leading_int(value)
            if client_id is None:
                continue
        elif key == "drm-pdev":
            pdev = value
        elif key.startswith("drm-engine-active-"):
            match = _ENGINE_ACTIVE_KEY.match(key)
            if match:
                number = _leading_int(value)
                if number is not None:
                    active[match.group(1)] = number
        elif key.startswith("drm-engine-period-"):
            match = _ENGINE_PERIOD_KEY.match(key)
            if match:
                number = _leading_int(value)
                if number is not None:
                    period[match.group(1)] = number
        elif key.startswith("drm-engine-"):
            match = _ENGINE_KEY.match(key)
            if match:
                number = _leading_int(value)
                if number is not None:
                    engines[match.group(1)] = number
        elif _MEMORY_KEY.match(key):
            size = parse_drm_size(value)
            if size is not None:
                memory[key] = size

    if driver is None or client_id is None:
        return None
    if driver != "amdgpu":
        return None

    # Newer kernels report active/period pairs directly, which are authoritative when
    # present; otherwise fall back to the cumulative total.
    for name, busy in active.items():
        window = period.get(name, 0)
        if window > 0:
            engines[name] = busy
    return FdinfoRecord(
        driver=driver,
        client_id=client_id,
        pdev=pdev or "",
        engines=engines,
        memory=memory,
    )


class ProcessCollector:
    """Scans ``/proc`` for DRM clients and derives per-engine utilisation.

    The instance owns the previous-sample baseline used for deltas.  It is intended to
    be confined to the sampler thread and never read concurrently; the class therefore
    needs no locking, which keeps it usable on the free-threaded build.

    Args:
        proc_root: The procfs mount point, overridable for tests.
        drm_name: Name of the user database, resolved through ``pwd`` for ownership.
    """

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root
        # Per-client cumulative engine counters from the previous scan.  One window for
        # the whole scan, so a single timestamp is enough.
        self._previous: dict[tuple[Bdf, ClientId], dict[str, int]] = {}
        # ``None`` rather than 0 means "no sample yet": a timestamp of exactly 0 is a
        # legitimate value, so it cannot double as the has-a-baseline flag.
        self._last_scan_ns: Nanoseconds | None = None
        # Rebuilt for every scan.  Instance state, not module state, so that two
        # collectors (as used by the tests) never share mutable data.
        self._fdinfo_names: dict[Path, frozenset[str]] = {}

    def reset(self) -> None:
        """Forget the delta baseline, e.g. after a device rescan."""
        self._previous.clear()
        self._last_scan_ns = None

    def collect(self, now_ns: Nanoseconds) -> tuple[tuple[GpuProcess, ...], int, int]:
        """Scan ``/proc`` and return the current DRM clients.

        Args:
            now_ns: Monotonic timestamp of this sample, used as the delta window.

        Returns:
            ``(processes, visible_count, total_count)`` where ``total_count`` is the
            number of process directories seen and ``visible_count`` the number whose
            file descriptors could actually be inspected.  The two differing is what
            reveals that ``/proc`` permissions hid other users' processes, which the UI
            reports rather than hiding.
        """
        clients, total_count, visible_count = self._scan_clients()
        processes = self._build_processes(clients, now_ns)
        return processes, visible_count, total_count

    def _fdinfo_names_for(self, fdinfo_dir: Path) -> frozenset[str]:
        """List the ``fdinfo`` entries available for one process.

        Memoised per scan so that a process holding several DRM descriptors does not
        trigger a ``scandir`` per descriptor.  The cache is cleared at the start of every
        scan, because a descriptor opened by a live process must not stay invisible for
        the rest of the run.
        """
        cached = self._fdinfo_names.get(fdinfo_dir)
        if cached is not None:
            return cached
        try:
            names = frozenset(entry.name for entry in os.scandir(fdinfo_dir))
        except OSError, ValueError:
            names = frozenset()
        self._fdinfo_names[fdinfo_dir] = names
        return names

    def _scan_clients(self) -> tuple[dict[tuple[Bdf, ClientId], _ClientAccumulator], int, int]:
        """Walk ``/proc`` and merge every DRM record by ``(pdev, client-id)``."""
        clients: dict[tuple[Bdf, ClientId], _ClientAccumulator] = {}
        total_count = 0
        visible_count = 0

        try:
            entries = list(os.scandir(self._proc_root))
        except OSError, ValueError:
            return clients, 0, 0

        self._fdinfo_names.clear()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            total_count += 1
            pid = int(entry.name)
            try:
                records = self._read_pid(pid)
            except OSError:
                # The process exited between scandir and the read; skip it.
                continue
            except PermissionError:
                # Another user's process: count it, but do not pretend we saw it.
                continue
            if records:
                visible_count += 1
            self._merge(clients, pid, records)

        return clients, total_count, visible_count

    def _read_pid(self, pid: int) -> list[FdinfoRecord]:
        """Return every DRM fdinfo record held by one process."""
        fd_dir = self._proc_root / str(pid) / "fd"
        fdinfo_dir = self._proc_root / str(pid) / "fdinfo"
        try:
            fd_entries = list(os.scandir(fd_dir))
        except OSError, ValueError:
            return []

        records: list[FdinfoRecord] = []
        for fd_entry in fd_entries:
            # ``os.scandir`` yields str paths, so re-wrap to use the Path API.
            fd_path = Path(fd_entry.path)
            try:
                target = fd_path.readlink().as_posix()
            except OSError:
                continue
            if not any(marker in target for marker in DRM_NODE_MARKERS):
                continue
            fd_name = fd_entry.name
            if fd_name not in self._fdinfo_names_for(fdinfo_dir):
                continue
            try:
                text = (fdinfo_dir / fd_name).read_text(errors="replace")
            except OSError, ValueError:
                continue
            record = parse_fdinfo(text)
            if record is not None:
                records.append(record)
        return records

    def _merge(
        self,
        clients: dict[tuple[Bdf, ClientId], _ClientAccumulator],
        pid: int,
        records: list[FdinfoRecord],
    ) -> None:
        """Fold one process's records into the shared per-client accumulators."""
        if not records:
            return
        name, user = _process_identity(self._proc_root / str(pid))
        for record in records:
            key = (record.pdev, record.client_id)
            acc = clients.get(key)
            if acc is None:
                acc = _ClientAccumulator(
                    pid=pid, name=name, user=user, bdf=record.pdev, client_id=record.client_id
                )
                clients[key] = acc
            elif pid < acc.pid:
                # Prefer the lowest PID so the row is stable across scans.
                acc.pid, acc.name, acc.user = pid, name, user
            # Maximum, never sum: two fds onto one client report the same totals.
            for engine, total_ns in record.engines.items():
                previous = acc.engines.get(engine)
                if previous is None or total_ns > previous:
                    acc.engines[engine] = total_ns
            for key_name, size in record.memory.items():
                previous_size = acc.memory.get(key_name)
                if previous_size is None or size > previous_size:
                    acc.memory[key_name] = size

    def _build_processes(
        self, clients: dict[tuple[Bdf, ClientId], _ClientAccumulator], now_ns: Nanoseconds
    ) -> tuple[GpuProcess, ...]:
        """Turn accumulators into frozen rows with delta-derived engine usage."""
        window_ns = max(0, now_ns - self._last_scan_ns) if self._last_scan_ns is not None else 0
        processes: list[GpuProcess] = []
        current: dict[tuple[Bdf, ClientId], dict[str, int]] = {}

        for key, acc in clients.items():
            previous_engines = self._previous.get(key, {})
            usage: list[EngineUsage] = []
            for engine, total_ns in sorted(acc.engines.items()):
                baseline = previous_engines.get(engine)
                delta = 0
                if baseline is not None and total_ns >= baseline:
                    delta = total_ns - baseline
                usage.append(
                    EngineUsage(
                        engine=normalise_engine(engine),
                        total_ns=total_ns,
                        delta_ns=delta,
                        window_ns=window_ns if baseline is not None else 0,
                    )
                )
            current[key] = dict(acc.engines)

            processes.append(
                GpuProcess(
                    pid=acc.pid,
                    name=acc.name,
                    user=acc.user,
                    bdf=acc.bdf,
                    client_id=acc.client_id,
                    engines=tuple(usage),
                    vram_used=_preferred(acc.memory, "drm-resident-vram", "drm-total-vram"),
                    vram_shared=acc.memory.get("drm-shared-vram"),
                    gtt_used=_preferred(acc.memory, "drm-resident-gtt", "drm-total-gtt"),
                    gtt_shared=acc.memory.get("drm-shared-gtt"),
                    cpu_used=_preferred(acc.memory, "drm-resident-cpu", "drm-total-cpu"),
                )
            )

        self._previous = current
        self._last_scan_ns = now_ns
        processes.sort(key=_sort_key)
        return tuple(processes)


def _preferred(memory: dict[str, int], first: str, second: str) -> int | None:
    """Return resident memory when the kernel provides it, else the total."""
    return memory.get(first, memory.get(second))


def _sort_key(process: GpuProcess) -> tuple[float, int, int]:
    """Order rows busiest-first, breaking ties by memory and then PID for stability."""
    return (-process.engine_percent, -process.memory_used, process.pid)


def _process_identity(pid_dir: Path) -> tuple[str, str]:
    """Return ``(name, user)`` for a process, tolerating a vanished process."""
    name = ""
    try:
        name = (pid_dir / "comm").read_text(errors="replace").strip()
    except OSError, ValueError:
        name = ""
    if not name:
        try:
            status = (pid_dir / "status").read_text(errors="replace")
        except OSError, ValueError:
            status = ""
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.partition(":")[2].strip()
                break
    if not name:
        name = pid_dir.name

    try:
        user = pwd.getpwuid(pid_dir.stat().st_uid).pw_name
    except OSError, ValueError, KeyError:
        user = "?"
    return name, user
