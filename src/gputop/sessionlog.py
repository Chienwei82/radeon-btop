"""Session recording to CSV, JSON Lines, or either under zstd.

Recording is an append-only stream, chosen for one property above prettiness: a session that
is interrupted still leaves a valid, readable file.  CSV is the format a spreadsheet opens
without being asked anything; JSON Lines is the format ``jq`` and a line-oriented reader
handle.  A single JSON array would be tidier to read and would lose everything on a crash,
so it is not offered.

Compression is inferred from the filename.  ``session.csv.zst`` is a CSV that happens to be
compressed; the base name decides the format and the suffix decides the transport, so the
two choices are independent and neither has to be declared twice.

Every writer here is best-effort with respect to *failure*: a full disk or an unwritable
directory produces one warning and then silence, because a recording that takes the monitor
down with it has made things worse.  The file is flushed per record, so a crash costs at
most one row.
"""

import contextlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, cast

from gputop.model.blocks import BLOCK_ORDER
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.snapshot import GpuSnapshot

#: Suffixes that mean "compress this".  ``.zst`` is the conventional one and ``.zstd`` is
#: accepted because it is the spelled-out form and people type both.
COMPRESSION_SUFFIXES: tuple[str, ...] = (".zst", ".zstd")


class LogFormat(StrEnum):
    """The on-disk record format."""

    CSV = "csv"
    #: One JSON object per line.  Streams, greps and truncates without a parser.
    JSONL = "jsonl"


class LogProblem(StrEnum):
    """Why a log could not be opened."""

    UNKNOWN_SUFFIX = "unknown suffix"
    OPEN_FAILED = "could not be opened"
    UNWRITABLE = "not writable"


def split_target(path: Path) -> tuple[LogFormat, bool, str]:
    """Decide format and compression from a log filename.

    Args:
        path: The requested target, e.g. ``session.csv``, ``session.json.zst``.

    Returns:
        ``(format, compressed, stem)``.  ``stem`` is the name the format is chosen from,
        with any compression suffix already removed.

    Raises:
        ValueError: When the format suffix is neither ``.csv`` nor ``.json``, since there is
            no defensible default -- silently choosing CSV for a file the user named
            ``.txt`` would be a guess about data they care enough to name.
    """
    name = path.name
    compressed = any(name.lower().endswith(suffix) for suffix in COMPRESSION_SUFFIXES)
    stem = name
    if compressed:
        for suffix in COMPRESSION_SUFFIXES:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
    suffix = Path(stem).suffix.lower()
    if suffix == ".csv":
        return LogFormat.CSV, compressed, stem
    if suffix in (".json", ".jsonl", ".ndjson"):
        return LogFormat.JSONL, compressed, stem
    raise ValueError(f"cannot tell the log format from {name!r}; use .csv or .json")


def _open_stream(handle: IO[bytes], *, level: int, compressed: bool) -> IO[bytes]:
    """Wrap ``handle`` for text output, compressing when asked.

    The import is local so that a Python built without zstd support only fails when someone
    actually requests a ``.zst`` file, rather than at start-up for every user.
    """
    if not compressed:
        return handle
    from compression import zstd  # optional dependency; see the module docstring

    return cast("IO[bytes]", zstd.ZstdFile(handle, mode="ab", level=level))


#: The CSV header, built once so the column order and the row builder cannot drift.
#:
#: Block columns come from :data:`~gputop.model.blocks.BLOCK_ORDER` and are present whether
#: or not radeontop is running: a card that has no Texture Cache block leaves that cell
#: empty rather than shifting every later column.  A stable column set is the whole point
#: of the CSV header -- a consumer should be able to read row 4000 with the same code that
#: read row 1, including on a session where radeontop was installed halfway through.
def _csv_columns() -> tuple[str, ...]:
    base = (
        "time",
        "sequence",
        "device_index",
        "card",
        "bdf",
        "name",
        "gpu_percent",
        "mem_percent",
        "vram_used",
        "vram_total",
        "gtt_used",
        "gtt_total",
        "sclk_mhz",
        "mclk_mhz",
        "socclk_mhz",
        "power_w",
        "power_cap_w",
        "fan_rpm",
        "profile",
        "odc_sclk",
        "odc_mclk",
        "throttling",
    )
    temps = ("edge", "junction", "mem", "soc", "core", "l3")
    return (
        base
        + tuple(f"temp_{name}" for name in temps)
        + tuple(f"block_{k}" for k in BLOCK_ORDER)
    )


CSV_COLUMNS: tuple[str, ...] = _csv_columns()


def _csv_row(metrics: AmdgpuMetrics, snapshot: GpuSnapshot) -> list[str]:
    """Flatten one device's sample into CSV cells, in :data:`CSV_COLUMNS` order.

    An absent reading is an empty cell, never ``0`` and never ``N/A``.  Empty is what every
    CSV reader treats as "no value", whereas a literal ``N/A`` in a numeric column is a
    parse error the consumer has to special-case.
    """
    temps = {t.label: t.celsius for t in metrics.temperatures}
    blocks = metrics.blocks.by_key() if metrics.blocks is not None else {}

    def number(value: float | None, digits: int = 2) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    cells = [
        _iso(snapshot),
        str(metrics.sequence),
        str(metrics.device.index),
        metrics.device.card,
        metrics.device.bdf,
        metrics.device.name,
        number(metrics.gpu_busy_percent),
        number(metrics.mem_busy_percent),
        "" if metrics.vram.used is None else str(metrics.vram.used),
        "" if metrics.vram.total is None else str(metrics.vram.total),
        "" if metrics.gtt.used is None else str(metrics.gtt.used),
        "" if metrics.gtt.total is None else str(metrics.gtt.total),
        number(metrics.sclk.current),
        number(metrics.mclk.current),
        number(metrics.socclk.current),
        number(metrics.power.draw_w, 1),
        number(metrics.power.cap_w, 1),
        "" if metrics.fan.rpm is None else str(metrics.fan.rpm),
        metrics.power_profile or metrics.performance_level or "",
        _describe(metrics, "sclk"),
        _describe(metrics, "mclk"),
        "" if metrics.throttle is None else ",".join(sorted(metrics.throttle.active)),
    ]
    cells += [
        number(temps.get(name)) for name in ("edge", "junction", "mem", "soc", "core", "l3")
    ]
    cells += [number(blocks.get(key)) for key in BLOCK_ORDER]
    return cells


def _iso(snapshot: GpuSnapshot) -> str:
    """Format a sample's wall-clock stamp as ISO 8601.

    The snapshot carries both clocks: ``timestamp_ns`` is monotonic and is what the rest of
    the program compares, while ``wall_clock_ns`` is only for putting the sample on a
    timeline a person or a spreadsheet can read.  Recording the second at the source rather
    than reconstructing it here keeps this module free of any dependency on the sampler.

    ISO 8601 with an explicit offset, which sorts lexicographically in the same order as
    chronologically -- that is what makes a recorded session usable by anything that reads
    the file as lines.  :meth:`datetime.isoformat` produces exactly that, so there is no
    format string here to keep in step with it.
    """
    from datetime import UTC, datetime

    if snapshot.wall_clock_ns <= 0:
        return ""
    moment = datetime.fromtimestamp(snapshot.wall_clock_ns / 1e9, tz=UTC)
    return moment.astimezone().isoformat(timespec="milliseconds")


def _describe(metrics: AmdgpuMetrics, domain: str) -> str:
    """One overdrive domain's summary, or an empty cell when the card reports none."""
    entry = metrics.odc.domain(domain)
    return "" if entry is None else entry.describe()


def _cell(value: Any) -> str:
    """Render one CSV cell, quoting only when the value requires it."""
    text = "" if value is None else str(value)
    if any(ch in text for ch in (",", '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text


@dataclass(slots=True)
class SessionLog:
    """An open recording target.

    Use :func:`open_log` rather than constructing this directly; it resolves the format from
    the filename and reports a problem as a value instead of an exception, because a log
    that cannot be opened must not stop the monitor from starting.
    """

    path: Path
    format: LogFormat
    compressed: bool
    problem: LogProblem | None = None
    #: Whether the CSV header still has to be written.
    #:
    #: Seeded from whether the file already had content, because the file is opened in
    #: append mode: recording twice to the same path is an ordinary thing to do, and a
    #: second header line lands in the middle of the data.  Every strict CSV reader then
    #: absorbs that line as a record.  A file that already exists but is empty -- a previous
    #: run that wrote nothing, or was interrupted before its first flush -- still gets a
    #: header, because there is nothing to conflict with.
    header_written: bool = False
    _records: int = 0
    _binary: IO[bytes] | None = None
    _stream: IO[bytes] | None = None
    _text: _LineWriter | None = None
    _broken: str | None = None

    @property
    def records(self) -> int:
        """How many records have been written."""
        return self._records

    @property
    def is_open(self) -> bool:
        """Whether the target is usable."""
        return self.problem is None and self._text is not None and self._broken is None

    def _fail(self, problem: LogProblem, detail: str) -> None:
        """Record the first failure; later ones are ignored.

        First-failure-wins is what keeps a full disk from producing one warning per sample
        and burying the monitor's status line.
        """
        if self.problem is None:
            self.problem = problem
            self._broken = detail

    def write(self, snapshot: GpuSnapshot) -> bool:
        """Append one snapshot.

        Args:
            snapshot: The sample to record.  Every device in it becomes its own row, so a
                row is always exactly one GPU at one instant.

        Returns:
            ``True`` when the snapshot was recorded.  ``False`` means the log has been
            disabled for the rest of the session; the caller does not need to check it.

            A snapshot with no devices is *not* a failure and returns ``True``: there is
            simply nothing to record, and the next sample records normally.  Reporting
            ``False`` there would tell a caller that "the log is dead" while it is still
            perfectly usable.
        """
        if not self.is_open:
            return False
        try:
            for metrics in snapshot.devices:
                if self.format is LogFormat.CSV:
                    self._write_csv(snapshot, metrics)
                else:
                    self._write_jsonl(snapshot, metrics)
                self._records += 1
        except (OSError, ValueError, TypeError) as exc:
            self._fail(LogProblem.UNWRITABLE, str(exc))
            return False
        return True

    def _write_csv(self, snapshot: GpuSnapshot, metrics: AmdgpuMetrics) -> None:
        """Write the header once, then one row per device."""
        assert self._text is not None
        if not self.header_written:
            self._text.write(",".join(_cell(c) for c in CSV_COLUMNS) + "\n")
            self.header_written = True
        cells = _csv_row(metrics, snapshot)
        self._text.write(",".join(_cell(cell) for cell in cells) + "\n")
        self._text.flush()

    def _write_jsonl(self, snapshot: GpuSnapshot, metrics: AmdgpuMetrics) -> None:
        """Write one JSON object per device."""
        assert self._text is not None
        self._text.write(json.dumps(_json_record(snapshot, metrics), sort_keys=False) + "\n")
        self._text.flush()

    def close(self) -> None:
        """Finish the stream, writing the compression frame's epilogue."""
        for stream in (self._text, self._stream, self._binary):
            if stream is not None:
                # Closing a zstd stream whose underlying file already failed still attempts
                # to write a frame epilogue; there is nothing useful to do about the
                # resulting exception, and the data is already lost either way.
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        self._text = None
        self._stream = None
        self._binary = None


def _json_record(snapshot: GpuSnapshot, metrics: AmdgpuMetrics) -> dict[str, Any]:
    """Build the JSON object for one device's sample."""
    return {
        "time": _iso(snapshot),
        "monotonic_ns": snapshot.timestamp_ns,
        "sequence": snapshot.sequence,
        "device": {
            "index": metrics.device.index,
            "card": metrics.device.card,
            "bdf": metrics.device.bdf,
            "name": metrics.device.name,
            "integrated": metrics.device.is_integrated,
        },
        "gpu_percent": metrics.gpu_busy_percent,
        "mem_percent": metrics.mem_busy_percent,
        "vram": {
            "used": metrics.vram.used,
            "total": metrics.vram.total,
            "percent": metrics.vram.percent,
        },
        "gtt": {
            "used": metrics.gtt.used,
            "total": metrics.gtt.total,
            "percent": metrics.gtt.percent,
        },
        "clocks_mhz": {
            "sclk": metrics.sclk.current,
            "mclk": metrics.mclk.current,
            "socclk": metrics.socclk.current,
        },
        "power": {
            "draw_w": metrics.power.draw_w,
            "cap_w": metrics.power.cap_w,
            "percent": metrics.power.percent,
        },
        "fan_rpm": metrics.fan.rpm,
        "temperatures_c": {t.label: t.celsius for t in metrics.temperatures},
        "profile": metrics.power_profile or metrics.performance_level,
        "throttling": (None if metrics.throttle is None else sorted(metrics.throttle.active)),
        "odc": metrics.odc.describe(),
        "blocks": (
            None
            if metrics.blocks is None
            else {
                "busiest": metrics.blocks.busiest_key(),
                "percent": dict(metrics.blocks.by_key()),
            }
        ),
    }


def open_log(path: Path, *, zstd_level: int = 3) -> SessionLog:
    """Open a recording target.

    Args:
        path: Where to write.  The suffix selects the format and, optionally, zstd.
        zstd_level: Compression level for a ``.zst`` target; 0 uses the library default.

    Returns:
        A :class:`SessionLog`, either open or carrying a :class:`LogProblem`.  A problem is
        reported in the value rather than raised so that ``--log /root/nope.csv`` degrades
        to "recording disabled, here's why" instead of refusing to open the monitor.
    """
    try:
        fmt, compressed, _stem = split_target(path)
    except ValueError as exc:
        return SessionLog(
            path=path,
            format=LogFormat.CSV,
            compressed=False,
            problem=LogProblem.UNKNOWN_SUFFIX,
            _broken=str(exc),
        )

    log = SessionLog(path=path, format=fmt, compressed=compressed)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        binary = path.open("ab")
    except OSError as exc:
        log.problem = LogProblem.OPEN_FAILED
        log._broken = str(exc)
        return log
    log._binary = binary
    # Seeded here rather than at first write so that reopening an existing recording
    # appends rows instead of writing a second header in among them.
    log.header_written = binary.tell() > 0
    try:
        log._stream = _open_stream(binary, level=zstd_level, compressed=compressed)
        log._text = _LineWriter(log._stream)
    except (OSError, ValueError, ImportError) as exc:
        binary.close()
        log.problem = LogProblem.OPEN_FAILED
        log._broken = str(exc)
        return log
    return log


class _LineWriter:
    """A minimal buffered text wrapper over a binary stream.

    :class:`io.TextIOWrapper` would do this, but it cannot be constructed over a
    ``ZstdFile`` without taking ownership of the underlying handle, and closing it would
    then close the file this class is also responsible for.  Wrapping by hand keeps the
    ownership split explicit and the flush-per-record behaviour visible.
    """

    __slots__ = ("_buffer", "_stream")

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._buffer: list[str] = []

    def write(self, text: str) -> int:
        """Append text, returning the character count as a file object must."""
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        """Push buffered text through and out to the file."""
        if not self._buffer:
            return
        payload = "".join(self._buffer).encode("utf-8")
        self._buffer.clear()
        self._stream.write(payload)
        self._stream.flush()

    def close(self) -> None:
        """Flush, then leave the underlying stream open -- this layer does not own it."""
        self.flush()


def read_log(path: Path) -> Iterator[str]:
    """Read back a recorded file, transparently decompressing.

    Exists so the tests, and any user wanting to check a recording, do not each reimplement
    the suffix rule that :func:`split_target` already encodes.
    """
    compressed = any(path.name.lower().endswith(suffix) for suffix in COMPRESSION_SUFFIXES)
    with path.open("rb") as handle:
        if not compressed:
            for line in handle:
                yield line.decode("utf-8", errors="replace").rstrip("\n")
            return
        from compression import zstd

        with zstd.ZstdFile(handle, mode="rb") as decompressed:
            for line in decompressed:
                yield line.decode("utf-8", errors="replace").rstrip("\n")


@contextmanager
def open_log_context(path: Path, *, zstd_level: int = 3) -> Iterator[SessionLog]:
    """Open a log and close it on exit, yielding the value whatever happens.

    Closing matters more than it looks for a ``.zst`` target: the compression library writes
    the frame epilogue on close, and a log left unclosed is a truncated archive rather than
    a readable file.  This is what guarantees that on every path out of the program.
    """
    log = open_log(path, zstd_level=zstd_level)
    try:
        yield log
    finally:
        log.close()


__all__ = [
    "COMPRESSION_SUFFIXES",
    "CSV_COLUMNS",
    "LogFormat",
    "LogProblem",
    "SessionLog",
    "open_log",
    "open_log_context",
    "read_log",
    "split_target",
]
