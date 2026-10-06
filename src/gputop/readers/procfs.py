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

**Cost.**  This is the only part of a sample whose price grows with the size of the
*system* rather than with the number of GPUs: it visits every process and reads one
symlink per open descriptor.  Profiling a synthetic 1000-process machine (``tools/
profile_gputop.py procfs --scaling``) put it at ~25 us per process, roughly 25 ms per
tick for a typical desktop.  Everything here is therefore written against ``os`` string
APIs rather than ``pathlib``: a ``Path`` per descriptor cost more than the syscall it was
wrapping.  The measured result of that change is recorded in the README.
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

#: A descriptor is on a GPU when its target lives under ``/dev/dri``.  Matching on the
#: substring rather than a suffix matters: the target is ``/dev/dri/renderD128``, so a
#: suffix test against ``/dri/renderD`` can never match.  One prefix test rather than one
#: per node type, because everything under that directory *is* a DRM node, and the
#: parser below still insists the driver is amdgpu.
DRM_DIR = "/dev/dri/"

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
    cmdline: tuple[str, ...] = ()
    ppid: int | None = None
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

    # amdgpu emits only ``drm-engine-<name>``.  The active/period pair is accepted
    # defensively, for a driver that reports a busy window directly rather than a
    # cumulative total, and it wins when present because it needs no baseline.
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
        # The scan works in ``str`` paths, not ``Path`` ones: it visits every process on
        # the machine, so a handful of microseconds of ``pathlib`` construction per
        # descriptor was measurable against the syscall it was wrapping.  Kept as both
        # forms so the public constructor still takes a ``Path``.
        self._root = os.fspath(proc_root)
        # Per-client cumulative engine counters from the previous scan.  One window for
        # the whole scan, so a single timestamp is enough.
        self._previous: dict[tuple[Bdf, ClientId], dict[str, int]] = {}
        # ``None`` rather than 0 means "no sample yet": a timestamp of exactly 0 is a
        # legitimate value, so it cannot double as the has-a-baseline flag.
        self._last_scan_ns: Nanoseconds | None = None
        # uid -> user name, resolved once.  ``pwd.getpwuid`` is an NSS lookup, and a scan
        # would otherwise repeat it for every client on every tick.
        self._users: dict[int, str] = {}

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
            reports rather than hiding.  ``visible_count`` counts *inspectable* processes,
            not ones that happen to hold a DRM descriptor: a plain process is perfectly
            visible and must not be reported as hidden.
        """
        clients, total_count, visible_count = self._scan_clients()
        processes = self._build_processes(clients, now_ns)
        return processes, visible_count, total_count

    def _scan_clients(self) -> tuple[dict[tuple[Bdf, ClientId], _ClientAccumulator], int, int]:
        """Walk ``/proc`` and merge every DRM record by ``(pdev, client-id)``."""
        clients: dict[tuple[Bdf, ClientId], _ClientAccumulator] = {}
        total_count = 0
        visible_count = 0

        try:
            # ``scandir`` yields the full path on each entry, so it is reused instead of
            # rebuilding ``/proc/<pid>`` from the number.
            entries = list(os.scandir(self._root))
        except OSError, ValueError:
            return clients, 0, 0

        read_pid = self._read_pid
        merge = self._merge
        for entry in entries:
            name = entry.name
            if not name.isdigit():
                continue
            total_count += 1
            pid = int(name)
            records = read_pid(entry.path)
            if records is None:
                # Could not inspect this process at all: another user's descriptor
                # directory, or the process exited between the scan and the read.  This
                # is the case ``visible_count`` exists to count.  (``_read_pid`` swallows
                # its own OSError/ValueError, so there is nothing to catch here.)
                continue
            visible_count += 1
            if records:
                merge(clients, pid, entry.path, records)

        return clients, total_count, visible_count

    def _read_pid(self, pid_dir: str) -> list[FdinfoRecord] | None:
        """Return every DRM fdinfo record held by one process.

        Args:
            pid_dir: The process directory as a string path, already known to exist.

        Returns:
            The records -- empty for a process holding no DRM descriptor -- or ``None``
            when the process could not be inspected at all: its descriptor directory is
            unreadable, which is the ordinary state of another user's process without
            root.  The distinction is what ``visible_count`` is made of; a bare ``[]``
            for both would make "has no GPU descriptor" and "hidden by permissions"
            indistinguishable.

        The descriptor's *target* is what identifies a GPU client, so every open
        descriptor costs one ``readlink``.  The previous implementation also listed the
        ``fdinfo`` directory to confirm the descriptor had accounting data; the read
        itself answers the same question -- an ``ENOENT`` here means the same thing -- so
        the extra ``scandir`` per client process is gone.
        """
        try:
            fd_entries = list(os.scandir(pid_dir + "/fd"))
        except OSError, ValueError:
            return None

        records: list[FdinfoRecord] = []
        fdinfo_dir = pid_dir + "/fdinfo/"
        for fd_entry in fd_entries:
            try:
                target = os.readlink(fd_entry.path)
            except OSError:
                continue
            if DRM_DIR not in target:
                continue
            try:
                with open(fdinfo_dir + fd_entry.name, "rb") as handle:
                    text = handle.read().decode(errors="replace")
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
        pid_dir: str,
        records: list[FdinfoRecord],
    ) -> None:
        """Fold one process's records into the shared per-client accumulators."""
        if not records:
            return
        name, user, ppid, cmdline = _process_identity(pid_dir, self._users)
        for record in records:
            key = (record.pdev, record.client_id)
            acc = clients.get(key)
            if acc is None:
                acc = _ClientAccumulator(
                    pid=pid,
                    name=name,
                    user=user,
                    bdf=record.pdev,
                    client_id=record.client_id,
                    cmdline=cmdline,
                    ppid=ppid,
                )
                clients[key] = acc
            elif pid < acc.pid:
                # Prefer the lowest PID so the row is stable across scans.
                acc.pid, acc.name, acc.user = pid, name, user
                acc.cmdline, acc.ppid = cmdline, ppid
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
                    cmdline=acc.cmdline,
                    ppid=acc.ppid,
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


def _process_identity(
    pid_dir: str, users: dict[int, str]
) -> tuple[str, str, int | None, tuple[str, ...]]:
    """Return ``(name, user, ppid, cmdline)`` for a process, tolerating a vanished one.

    ``status`` is read because ``PPid`` lives there and only there, and the process tree is
    built from it.  It is read for processes that hold a DRM descriptor and for no others,
    so the extra read is bounded by the size of the table, not by the size of ``/proc``.
    ``comm`` is preferred for the name because it needs no parsing, and is still the
    fallback chain the kernel itself documents.

    Args:
        pid_dir: Process directory as a string path.
        users: uid -> name cache owned by the collector, so an NSS lookup happens once per
            uid per run rather than once per client per tick.
    """
    name = ""
    ppid: int | None = None
    try:
        with open(pid_dir + "/comm", "rb") as handle:
            name = handle.read().decode(errors="replace").strip()
    except OSError, ValueError:
        name = ""
    try:
        with open(pid_dir + "/status", "rb") as handle:
            status = handle.read().decode(errors="replace")
    except OSError, ValueError:
        status = ""
    for line in status.splitlines():
        if line.startswith("Name:") and not name:
            name = line.partition(":")[2].strip()
        elif line.startswith("PPid:"):
            ppid = _leading_int(line.partition(":")[2])
    if not name:
        name = os.path.basename(pid_dir)

    try:
        uid = os.stat(pid_dir).st_uid
    except OSError, ValueError:
        return name, "?", ppid, _read_cmdline(pid_dir)
    user = users.get(uid)
    if user is None:
        try:
            user = pwd.getpwuid(uid).pw_name
        except OSError, ValueError, KeyError:
            user = "?"
        users[uid] = user

    return name, user, ppid, _read_cmdline(pid_dir)


def _read_cmdline(pid_dir: str) -> tuple[str, ...]:
    """Read ``/proc/<pid>/cmdline`` as a list of arguments.

    The file is NUL-separated with a trailing NUL, and it is **empty** for a kernel
    thread -- which is exactly how the caller recognises one without root.  An unreadable
    file is indistinguishable from an empty one here, and both mean "no argument vector",
    so an empty tuple is the right answer either other way.
    """
    try:
        with open(pid_dir + "/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError, ValueError:
        return ()
    return tuple(part for part in raw.decode(errors="replace").split("\0") if part)
