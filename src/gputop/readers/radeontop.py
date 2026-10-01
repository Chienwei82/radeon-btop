"""Per-block utilisation, by supervising a ``radeontop`` subprocess.

Why a subprocess at all
-----------------------

The amdgpu driver exposes its GRBM and SRBM busy counters only through the radeon ioctl on a
handle that needs elevated privileges.  gputop's whole design is that it never requires root,
so it cannot take that dependency for the whole application.  But the counters are genuinely
useful -- they are the only way to see *which* part of the graphics pipe is saturated, which
is the question "why is my GPU at 100%" actually reduces to.  So they are read the one way
that does not require gputop itself to be privileged: by running ``radeontop`` as a child
process and parsing its dump output.

The child is dropped to the invoking user by radeontop's own startup, so gputop gains
nothing by being root -- and if gputop *is* run as root, so is the child.  The privilege
requirement is therefore exactly whatever radeontop itself needs, which is the correct
boundary: it stays a per-machine choice rather than becoming this program's.

Degradation
-----------

A missing binary, a kernel that refuses the ioctl, an unsupported card, and a crash are four
different problems with four different remedies, so they are four different
:class:`~gputop.model.blocks.BlocksStatus` values rather than one ``unavailable``.  None of
them stop the sampler, and the panel is hidden rather than showing empty bars.

Threading
---------

Each supervised process gets one reader thread that drains stdout, and one that drains
stderr.  Both are required: an undrained pipe fills and blocks the child, which would freeze
its sampling silently.  The threads publish by rebinding a single attribute to a frozen
value, so the sampler thread reads them without a lock -- the same rule
:mod:`gputop.sampler.engine` follows for its history.
"""

import contextlib
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from gputop.model.blocks import (
    AUX_BLOCK_ORDER,
    BLOCK_LABELS,
    BLOCK_ORDER,
    CLOCK_KEYS,
    MEMORY_KEYS,
    BlockReading,
    BlocksStatus,
    ClockReading,
    GpuBlocks,
    block_label,
)

#: The dump line's prefix: ``<seconds>.<microseconds>: bus 0c, ...``
_HEADER = re.compile(r"^\s*(\d+)\.(\d+)\s*:\s*bus\s+([0-9a-fA-F]+)\s*,?\s*(.*)$")

#: One ``key value...`` field of the dump line.
_FIELD = re.compile(r"^([a-z][a-z0-9_]*)\s+(.*)$")

#: A number inside a field's value list.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: radeontop's own dump interval granularity.  ``-i`` is parsed with ``atoi`` and clamped
#: to a minimum of 1 second, so a sub-second interval is not expressible and pretending
#: otherwise would mean silently sampling slower than the user asked for.
MIN_DUMP_INTERVAL_S = 1

#: Default samples per second requested from radeontop.  Its own default; raising it costs
#: CPU and buys accuracy nobody can see on a bar that refreshes once a second.
DEFAULT_TICKS = 120

#: How long to wait for the first line before deciding the child is not going to produce
#: one.  radeontop busy-waits for its initial sample set before its first dump, so a short
#: grace period is not enough.
STARTUP_GRACE_S = 3.0

#: stderr lines retained for failure classification.  radeontop is chatty on failure and
#: nearly silent on success, so a small window is plenty.
STDERR_WINDOW = 20

#: stdout lines retained before the first dump line, for the same reason.  Small because
#: radeontop says everything it is going to say in the first few lines or not at all.
STDOUT_HEAD_WINDOW = 8

#: How long to wait for the stderr drain to finish after the child dies, before classifying
#: on whatever has arrived.  See :meth:`RadeontopSource._on_child_exit`.
STDERR_DRAIN_TIMEOUT_S = 1.0

#: Restart attempts before a child is declared failed.  radeontop exits non-zero-ish by
#: dying immediately when it cannot use the card; retrying that forever would be a busy loop.
MAX_RESTARTS = 2

#: How long a child must stay alive before it is considered to have proven itself and the
#: restart budget is restored.  Sized well above radeontop's own start-up (which busy-waits
#: for its first sample set) so that a child which can never work is not mistaken for one
#: that merely took a while.
RESTART_RESET_S = 60.0

#: Substrings radeontop prints when it cannot use the card, mapped to what to report.  It
#: writes some of them to stdout and some to stderr, so both streams are searched.  The
#: ordering matters: the permission message is checked first because it is the one a user
#: can actually act on.
_STDERR_SIGNALS: tuple[tuple[str, BlocksStatus], ...] = (
    ("cannot access gpu registers", BlocksStatus.NOT_PERMITTED),
    ("are you root", BlocksStatus.NOT_PERMITTED),
    ("failed to open drm node", BlocksStatus.NOT_PERMITTED),
    ("permission denied", BlocksStatus.NOT_PERMITTED),
    ("failed to find drm devices", BlocksStatus.UNSUPPORTED),
    ("unsupported bus type", BlocksStatus.UNSUPPORTED),
    ("failed to open", BlocksStatus.UNSUPPORTED),
    ("no vram support", BlocksStatus.UNSUPPORTED),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RadeontopOptions:
    """How to run radeontop.

    Attributes:
        enabled: Whether the blocks panel may be shown at all.
        binary: Program name looked up on ``$PATH``, or an absolute path.
        ticks: Samples per second requested from radeontop.
        interval_s: Seconds between dumps.  Whole seconds only; see
            :data:`MIN_DUMP_INTERVAL_S`.
        drm_root: Directory the DRM nodes live in, so ``-p`` can be preferred over ``-b``.
        restart: Whether a child that dies is restarted.
    """

    #: Off by default; see :class:`~gputop.config.BlocksConfig` for why.
    enabled: bool = False
    binary: str = "radeontop"
    ticks: int = DEFAULT_TICKS
    interval_s: int = 1
    drm_root: Path = Path("/dev/dri")
    restart: bool = True

    def validated(self) -> RadeontopOptions:
        """Clamp to ranges radeontop would accept anyway.

        Clamping rather than rejecting means a config file written by hand with
        ``interval_s = 0`` produces a working program and a sane interval, instead of an
        error the user has to read a man page to resolve.
        """
        return RadeontopOptions(
            enabled=self.enabled,
            binary=self.binary,
            ticks=max(1, min(self.ticks, 10_000)),
            interval_s=max(MIN_DUMP_INTERVAL_S, min(self.interval_s, 3600)),
            drm_root=Path(self.drm_root),
            restart=self.restart,
        )


@dataclass(frozen=True, slots=True)
class RadeontopDump:
    """One parsed line of radeontop's dump output.

    Attributes:
        seconds: Wall-clock seconds from the line's own timestamp.
        microseconds: Microseconds within that second.
        bus: The PCI bus radeontop sampled, hexadecimal as printed.
        blocks: Utilisation per block key, as a percentage.
        clocks: ``(key, percent, megahertz)`` for the memory and shader clocks.
        memory: ``(key, percent, megabytes)`` for VRAM and GTT.
    """

    seconds: int = 0
    microseconds: int = 0
    bus: str = ""
    blocks: dict[str, float] = field(default_factory=dict)
    clocks: dict[str, tuple[float, float]] = field(default_factory=dict)
    memory: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_blocks(self, *, now_ns: int | None = None) -> GpuBlocks:
        """Convert to the model type the rest of the program passes around.

        Args:
            now_ns: The sampler's current *wall-clock* time, used to record staleness.
                Left ``None`` the sample claims no age, which is correct for a line the
                reader thread has only just parsed.

                The clock matters here.  ``dump.c`` stamps each line with
                ``gettimeofday()`` -- a wall-clock epoch -- while the rest of gputop
                measures intervals with ``time.monotonic()``, which is a different
                origin.  Passing a monotonic value here produced an age of about 56 years
                for every sample.  The one production call site correctly omits the
                argument, so this has never been observed; it is spelled out because the
                failure is silent and enormous.
        """
        ordered = [
            BlockReading(key=key, label=block_label(key), percent=self.blocks[key])
            for key in BLOCK_ORDER
            if key in self.blocks
        ]
        aux = [
            BlockReading(key=key, label=block_label(key), percent=self.blocks[key])
            for key in AUX_BLOCK_ORDER
            if key in self.blocks
        ]
        clocks: list[ClockReading] = []
        for key in CLOCK_KEYS:
            pair = self.clocks.get(key)
            if pair is None:
                continue
            percent, mhz = pair
            clocks.append(
                ClockReading(key=key, label=block_label(key), percent=percent, mhz=round(mhz))
            )
        timestamp = self.seconds * 1_000_000_000 + self.microseconds * 1_000
        age = 0 if now_ns is None else max(0, now_ns - timestamp)
        return GpuBlocks(
            blocks=tuple(ordered),
            aux_blocks=tuple(aux),
            clocks=tuple(clocks),
            timestamp_ns=timestamp,
            age_ns=age,
        )


def parse_dump_line(line: str) -> RadeontopDump | None:
    """Parse one line of ``radeontop -d`` output.

    The line is parsed by field name rather than by position, and that is not a stylistic
    choice.  ``dump.c`` emits ``tc``, ``smx``, ``cr``, ``uvd``, ``vce0``, ``vram``, ``gtt``
    and the clocks only when the card reports those bits, so a positional parser would read
    ``vgt``'s value as ``ta``'s on every modern card and report confidently wrong numbers.
    Naming the fields makes a missing block a missing block.

    Args:
        line: One line of output, with or without its trailing newline.

    Returns:
        The parsed dump, or ``None`` when the line is a banner, a diagnostic, or garbage.
        ``None`` is routine: radeontop writes its ``"Dumping to ..., until termination."``
        banner to **stdout** with a plain ``printf``, interleaved with the data, and a card
        it cannot read writes a sentence instead of a data line.  (The comment here used
        to say the banner went to stderr, which is what made the surrounding stream
        handling look wrong when it was right.)
    """
    header = _HEADER.match(line)
    if header is None:
        return None
    seconds, micros, bus, rest = header.groups()

    dump = RadeontopDump(
        seconds=int(seconds),
        microseconds=int(micros),
        bus=bus.lower(),
    )
    for chunk in rest.split(","):
        field_match = _FIELD.match(chunk.strip())
        if field_match is None:
            continue
        key, payload = field_match.group(1), field_match.group(2)
        numbers = _NUMBER.findall(payload)
        if not numbers:
            continue
        values = [float(n) for n in numbers]
        if key in BLOCK_LABELS:
            dump.blocks[key] = values[0]
        elif key in MEMORY_KEYS:
            # ``vram 13.13% 4096.00mb`` -- percentage then megabytes.
            dump.memory[key] = (values[0], values[1] if len(values) > 1 else 0.0)
        elif key in CLOCK_KEYS:
            # ``sclk 25.00% 2.100ghz`` -- percentage then gigahertz.
            ghz = values[1] if len(values) > 1 else 0.0
            dump.clocks[key] = (values[0], ghz * 1000.0)
    return dump


def bus_from_bdf(bdf: str) -> str | None:
    """Extract the PCI bus from a domain:bus:device.function address.

    radeontop's ``-b`` takes the bus as it appears in ``lspci`` output, which is the
    two-digit hexadecimal field -- ``0000:0c:00.0`` gives ``0c``.

    Returns:
        The bus as a hexadecimal string, or ``None`` when the address is malformed.
    """
    parts = bdf.split(":")
    if len(parts) != 3 or not parts[1]:
        return None
    bus = parts[1].lower()
    return bus if all(c in "0123456789abcdef" for c in bus) else None


def build_argv(
    options: RadeontopOptions,
    *,
    bus: str | None = None,
    node: Path | None = None,
) -> list[str]:
    """Build the command line for one radeontop child.

    Dump mode is ``-d - -i <interval> -l 0``: stdout rather than a file, so the child's
    output can be consumed as a stream instead of re-read from disk, and a line limit of
    zero, which is radeontop's spelling of "until terminated".

    Args:
        options: Validated run options.
        bus: Hexadecimal PCI bus, used when no DRM node path is available.
        node: The card's DRM node, preferred over ``bus`` because it is unambiguous.

    Returns:
        The argument vector, program name first.
    """
    argv = [
        options.binary,
        "-d",
        "-",
        "-i",
        str(options.interval_s),
        "-l",
        "0",
        "-t",
        str(options.ticks),
    ]
    if node is not None:
        argv += ["-p", str(node)]
    elif bus is not None:
        argv += ["-b", bus]
    return argv


def classify_failure(lines: tuple[str, ...]) -> BlocksStatus | None:
    """Map captured diagnostic output onto a status, or ``None`` when nothing matches.

    The argument is *both* streams, not stderr alone.  radeontop splits its complaints: the
    packaged build writes "Cannot access GPU registers, are you root?" and "Failed to open
    DRM node" to **stdout**, and only "Failed to find DRM devices" to stderr.  Reading only
    stderr therefore classifies a privilege problem as an unsupported card -- the least
    actionable of the available answers, and the one this function exists to avoid.

    Matching is safe against a stream that also carries data: a dump line is numbers, commas
    and a ``%``, and none of the signals below can occur in one.

    The signals are tried in declaration order, which puts the actionable permission
    failure ahead of the generic "failed to open", because radeontop reports every problem
    it has and the first one that explains the cause is the useful one.
    """
    blob = " ".join(lines).lower()
    if not blob.strip():
        return None
    for needle, status in _STDERR_SIGNALS:
        if needle in blob:
            return status
    return None


def resolve_binary(name: str) -> Path | None:
    """Find the radeontop executable.

    Args:
        name: A program name to look up on ``$PATH``, or an absolute path.

    Returns:
        The resolved path, or ``None`` when it is absent or not executable.
    """
    candidate = Path(name)
    if candidate.is_absolute() or os.sep in name:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        return None
    found = shutil.which(name)
    return Path(found) if found else None


def _drm_node(options: RadeontopOptions, card: str | None) -> Path | None:
    """Return the DRM node for a card when one exists, else ``None``.

    Passing ``-p`` is preferred over ``-b`` because the bus alone is ambiguous on a machine
    with two cards behind the same PCI bridge, and a wrong card's numbers would be worse
    than no numbers.  The existence check is what makes the fake sysfs trees in the test
    suite work without inventing DRM nodes.
    """
    if not card:
        return None
    node = options.drm_root / card
    return node if node.exists() else None


class RadeontopSource:
    """One supervised radeontop process, for one GPU.

    **The one lock in the codebase.**  :mod:`gputop.sampler.engine` takes no locks because
    the sampling state has exactly one owner and one reader that only ever sees immutable
    values.  A subprocess handle does not have that property: the sampler thread starts and
    polls the child, and the UI thread may stop it during unmount.  Those two can genuinely
    race on process creation and teardown, so this class guards them with a short-held lock.

    The lock is held only across ``Popen`` and ``terminate``/``wait`` -- a handful of
    syscalls per session and per restart -- and never while reading a sample.  The values the
    sampler reads (``_status``, ``_blocks``) are still published by single rebinds of frozen
    objects, so the hot path keeps the property the rest of the program depends on.
    """

    def __init__(
        self, options: RadeontopOptions, *, key: str, bus: str | None, node: Path | None
    ) -> None:
        """Create the source.

        Args:
            options: Validated run options.
            key: Identifier for the device this serves; also the pool's lookup key.
            bus: Hexadecimal PCI bus, used when ``node`` is absent.
            node: DRM node, used in preference to ``bus``.
        """
        self._options = options
        self._key = key
        self._bus = bus
        self._node = node
        self._process: subprocess.Popen[str] | None = None
        self._threads: list[threading.Thread] = []
        self._stderr_thread: threading.Thread | None = None
        self._stderr: deque[str] = deque(maxlen=STDERR_WINDOW)
        # The first lines of stdout.  Before radeontop manages the card it writes its
        # complaints there, so this is a diagnostics window that happens to share a pipe
        # with the data; it stops filling once a dump line is seen.
        self._stdout_head: deque[str] = deque(maxlen=STDOUT_HEAD_WINDOW)
        self._saw_dump_line = False
        self._stop = threading.Event()
        # Published by rebinding, so readers never see a half-updated pair.  ``_status``
        # is the value a caller reads; ``_blocks`` is the data to go with it.
        self._status: BlocksStatus = BlocksStatus.MISSING
        self._blocks: GpuBlocks = GpuBlocks()
        self._restarts = 0
        self._started_at = time.monotonic()
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        """The device key this source serves."""
        return self._key

    @property
    def status(self) -> BlocksStatus:
        """Why this source is or is not producing data."""
        return self._status

    @property
    def blocks(self) -> GpuBlocks:
        """The most recent sample, empty when nothing has been parsed yet."""
        return self._blocks

    def stderr_text(self) -> str:
        """The retained stderr lines, for a diagnostic message."""
        return " ".join(self._stderr)

    def start(self) -> None:
        """Launch the child, or record why it could not be launched.

        Safe to call more than once; a source that is already running is left alone.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            binary = resolve_binary(self._options.binary)
            if binary is None:
                self._status = BlocksStatus.MISSING
                return
            argv = build_argv(self._options, bus=self._bus, node=self._node)
            try:
                # argv is built from validated options, never passed through a shell.
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    # A new process group so a terminal interrupt reaches the child too,
                    # rather than leaving it orphaned and still holding the GPU.
                    start_new_session=True,
                )
            except OSError, ValueError:
                self._status = BlocksStatus.FAILED
                return
            # The previous child's pipes are closed before it is replaced.  Restarting is
            # ordinary -- a driver reload or a hot-unplugged eGPU causes it -- and without
            # this every restart would leave two descriptors open, since nothing else holds
            # a reference to the process that is being dropped.
            self._close_pipes(self._process)
            self._process = process
            self._started_at = time.monotonic()
            self._stop.clear()
            self._status = BlocksStatus.OK
            self._stderr_thread = None
            # Everything below belongs to the child being replaced, not to this one.
            # ``_saw_dump_line`` in particular: carrying it over made every complaint the
            # new child writes before its first sample get discarded, so a replacement
            # that cannot reach the GPU reported "failed" instead of the reason it printed.
            self._stderr.clear()
            self._stdout_head.clear()
            self._saw_dump_line = False
            self._spawn_reader(self._read_stdout, "gputop-radeontop-out")
            self._spawn_reader(self._read_stderr, "gputop-radeontop-err")

    @staticmethod
    def _close_pipes(process: subprocess.Popen[str] | None) -> None:
        """Close a finished child's stdout and stderr.

        Closed explicitly rather than left to the garbage collector: ``Popen`` keeps both
        pipes open until then, and a session that starts and stops children repeatedly --
        every restart, every device switch -- leaks two descriptors each time and
        eventually runs the monitor out of them.
        """
        if process is None:
            return
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError, ValueError):
                    pipe.close()

    def _spawn_reader(self, target: Callable[[], None], name: str) -> None:
        """Start one drain thread, remembering the stderr one so exit can wait on it."""
        # Readers from a previous child are finished by the time a restart happens -- they
        # return when their pipe closes -- but nothing removed them, so a machine whose
        # radeontop flaps (a driver reload, an eGPU re-enumerating) accumulated two dead
        # Thread objects per restart for the life of the process, and stop() then joined
        # that whole history.  Pruned here rather than in start() so the list never
        # outlives the children it refers to.
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        if name.endswith("-err"):
            self._stderr_thread = thread
        thread.start()

    def stop(self) -> None:
        """Terminate the child and let its threads finish.

        Termination is polite first: radeontop installs a SIGTERM handler so it can flush
        the frame it is writing.  It is killed only if it does not go quietly, because a
        child that ignores SIGTERM would otherwise outlive the monitor that started it.

        The lock is taken here for the same reason :meth:`start` takes it.  Without it, a
        restart already in flight on the sampler thread installs a new child between this
        method's read of ``_process`` and its final write, and the result is a radeontop
        nobody terminates, still holding the card, with its pipes leaked.  The handle is
        cleared *inside* the lock, so a concurrent :meth:`start` sees a free slot rather
        than a process this method believes it owns.
        """
        self._stop.set()
        with self._lock:
            process = self._process
            threads = tuple(self._threads)
            self._process = None
            self._threads.clear()
            self._stderr_thread = None
            if process is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, OSError):
                        process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2.0)
        # Joined outside the lock: a reader can block for as long as its child takes to
        # notice SIGTERM, and holding the lock across that would stall ``start`` for a
        # restart that the sampler is entitled to make meanwhile.
        for thread in threads:
            with contextlib.suppress(RuntimeError):
                thread.join(timeout=2.0)
        self._close_pipes(process)

    # -- drain loops -------------------------------------------------------

    def _read_stdout(self) -> None:
        """Parse dump lines until the child closes stdout or the source is stopped."""
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                # A reader can outlive the child it was spawned for: a restart installs a
                # replacement while the previous child's stdout is still reaching EOF.
                # Everything below would then describe the child that is running now,
                # including marking a perfectly healthy one as failed.  Bail out instead.
                if self._stop.is_set() or self._process is not process:
                    return
                sample = parse_dump_line(line)
                if sample is not None:
                    # From here on stdout is data; anything before this was a complaint.
                    self._saw_dump_line = True
                    self._blocks = sample.to_blocks()
                elif not self._saw_dump_line:
                    self._stdout_head.append(line.strip())
        except OSError, ValueError:
            # ``stop`` closes the pipes after a bounded join, so a reader that outlived its
            # join finds the handle closed underneath it.  That is the shutdown this thread
            # exists for, not a failure to report.
            return
        self._on_child_exit(process)

    def _read_stderr(self) -> None:
        """Retain stderr so a failure can be explained rather than merely reported."""
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                self._stderr.append(line.strip())
        except OSError, ValueError:
            # See ``_read_stdout``: the pipe was closed under us during shutdown.
            return

    def _on_child_exit(self, process: subprocess.Popen[str] | None = None) -> None:
        """Classify why a child stopped producing output.

        The stderr drain is awaited first, and that ordering is load-bearing rather than
        tidy.  radeontop reports *every* problem it has, most usefully last: it prints
        "Failed to find DRM devices", then "Failed to open DRM node", then finally "Cannot
        access GPU registers, are you root?".  stdout closes as soon as the child dies while
        stderr is still being delivered, so classifying on stdout EOF alone reads a partial
        buffer and reports the least actionable of the three -- telling a user their card is
        unsupported when the real answer is that they need root.

        Joining the stderr thread is bounded, because a child that exits while leaving its
        stderr pipe open would otherwise hang the sampler thread here.

        Args:
            process: The child that exited.  Defaults to the current one; a reader passes
                its own, because a thread from a previous child says nothing about the
                replacement that is already running.
        """
        if self._stop.is_set():
            return
        target = self._process if process is None else process
        if target is None or target is not self._process:
            return
        stderr_thread = self._stderr_thread
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=STDERR_DRAIN_TIMEOUT_S)
        status = classify_failure((*self._stdout_head, *self._stderr))
        if status is not None:
            self._status = status
            return
        self._status = BlocksStatus.FAILED

    def poll(self) -> BlocksStatus:
        """Re-check liveness, restarting the child when that is allowed.

        Called from the sampler thread on every tick.  A child that exits during a normal
        session -- a driver reload, a hot-unplugged eGPU -- should not permanently silence
        the panel, so it is restarted a bounded number of times and then left failed.
        """
        if self._stop.is_set():
            return self._status
        process = self._process
        if process is not None and process.poll() is None:
            # A child that has outlived the reset window has proven itself, so the next
            # failure it suffers gets the full budget again.  Without this a long session
            # accumulates restarts towards the cap over hours and ends up unable to recover
            # from a single ordinary hiccup.
            if time.monotonic() - self._started_at >= RESTART_RESET_S:
                self._restarts = 0
            return self._status
        if process is not None:
            self._on_child_exit(process)
        if self._status.ok or not self._options.restart:
            return self._status
        # Only a transient failure is worth retrying; a missing binary or a refused ioctl
        # will refuse again just as promptly.
        if self._status in (
            BlocksStatus.MISSING,
            BlocksStatus.NOT_PERMITTED,
            BlocksStatus.UNSUPPORTED,
        ):
            return self._status
        if self._restarts >= MAX_RESTARTS:
            self._status = BlocksStatus.FAILED
            return self._status
        self._restarts += 1
        self.start()
        return self._status


class RadeontopPool:
    """One :class:`RadeontopSource` per GPU, plus the panel's overall verdict.

    The pool exists so that the rest of the program deals with a single object: the sampler
    creates one, calls :meth:`poll` each tick, and asks it for a status and a per-device
    sample.  It does not need to know that a subprocess is involved, or that there may be
    several.
    """

    def __init__(self, options: RadeontopOptions | None = None) -> None:
        self._options = (options or RadeontopOptions()).validated()
        self._sources: dict[str, RadeontopSource] = {}
        self._order: tuple[str, ...] = ()

    @property
    def options(self) -> RadeontopOptions:
        """The validated options in effect."""
        return self._options

    @property
    def enabled(self) -> bool:
        """Whether the blocks panel may be shown."""
        return self._options.enabled

    def configure(self, devices: list[tuple[str, str, str | None]]) -> None:
        """Create one source per device.

        Args:
            devices: ``(key, card_name, bdf)`` triples, in the sampler's device order.
        """
        self.stop()
        self._sources.clear()
        if not self._options.enabled:
            self._order = ()
            return
        self._order = tuple(key for key, _card, _bdf in devices)
        for key, card, bdf in devices:
            self._sources[key] = RadeontopSource(
                self._options,
                key=key,
                bus=bus_from_bdf(bdf) if bdf else None,
                node=_drm_node(self._options, card),
            )

    def start(self) -> None:
        """Start every source."""
        if not self._options.enabled:
            return
        for source in self._sources.values():
            source.start()

    def stop(self) -> None:
        """Stop every source."""
        for source in self._sources.values():
            source.stop()

    def keys(self) -> tuple[str, ...]:
        """The device keys this pool serves, in device order."""
        return self._order

    def status(self) -> BlocksStatus:
        """The worst status across every source.

        "Worst" is ordered by how actionable it is rather than by severity: a card that
        radeontop cannot read at all is reported as unsupported even if another card works,
        because a panel that silently covers only half the GPUs is the more misleading of the
        two outcomes.
        """
        if not self._options.enabled:
            return BlocksStatus.DISABLED
        if not self._sources:
            return BlocksStatus.MISSING
        statuses = {source.status for source in self._sources.values()}
        for candidate in (
            BlocksStatus.NOT_PERMITTED,
            BlocksStatus.MISSING,
            BlocksStatus.UNSUPPORTED,
            BlocksStatus.FAILED,
        ):
            if candidate in statuses:
                return candidate
        return BlocksStatus.OK

    def blocks_for(self, key: str) -> GpuBlocks | None:
        """The latest sample for a device, or ``None`` when it has none yet.

        ``None`` and an empty :class:`GpuBlocks` are different answers: the first means the
        child has not produced a line for this card yet, the second means it produced one
        that had no blocks in it.
        """
        source = self._sources.get(key)
        if source is None or not source.status.ok:
            return None
        if source.blocks.is_empty:
            return None
        return source.blocks

    def poll(self) -> BlocksStatus:
        """Refresh every source and return the overall status."""
        if not self._options.enabled:
            return BlocksStatus.DISABLED
        for source in self._sources.values():
            source.poll()
        return self.status()

    def diagnostic(self) -> str:
        """A one-line explanation of the current status, or ``""`` when healthy."""
        status = self.status()
        hint = status.hint
        if hint:
            return hint
        details = [
            source.stderr_text() for source in self._sources.values() if source.stderr_text()
        ]
        return details[0] if details else ""


def wait_for_first_sample(pool: RadeontopPool, timeout_s: float = STARTUP_GRACE_S) -> bool:
    """Block until every source has produced a line, or the timeout expires.

    Only used by the one-shot CLI paths, which have no sampler thread to wait on.  A test
    and a ``--dump`` both need the panel to have real numbers the first time it renders.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pool.poll()
        if all(pool.blocks_for(key) is not None for key in pool.keys()):  # noqa: SIM118
            return True
        time.sleep(0.05)
    return False


__all__ = [
    "DEFAULT_TICKS",
    "MIN_DUMP_INTERVAL_S",
    "RadeontopDump",
    "RadeontopOptions",
    "RadeontopPool",
    "RadeontopSource",
    "build_argv",
    "bus_from_bdf",
    "classify_failure",
    "parse_dump_line",
    "resolve_binary",
    "wait_for_first_sample",
]
