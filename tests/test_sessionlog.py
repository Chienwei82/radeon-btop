"""Tests for the session recorder: CSV, JSON Lines, and the zstd transport.

Every assertion reads the file back off disk rather than inspecting the writer's
internals, because the contract the module actually promises is about bytes: a
spreadsheet, ``jq`` and a session that was interrupted all only ever see the file.  The
two rules worth pinning down are that the header appears exactly once, and that an
absent reading is an empty cell -- never ``0`` and never ``N/A`` -- so a consumer can
tell "this hardware did not report it" from "this hardware was idle".
"""

import csv
import io
import json
from compression import zstd
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gputop.model.blocks import BlockReading, GpuBlocks
from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import (
    AmdgpuMetrics,
    Clock,
    Fan,
    MemoryPool,
    Power,
    Temperature,
    ThrottleInfo,
)
from gputop.model.snapshot import GpuSnapshot
from gputop.sessionlog import (
    CSV_COLUMNS,
    LogFormat,
    LogProblem,
    open_log,
    open_log_context,
    read_log,
    split_target,
)

#: A fixed wall clock, so the ``time`` column is reproducible across runs.
WALL_CLOCK_NS = 1_700_000_000_000_000_000

#: The four bytes that start every zstd frame; anything else is not a zstd file.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

#: The columns that carry a reading, and must be empty for a device that reported none.
MEASUREMENT_COLUMNS: tuple[str, ...] = (
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
    "odc_sclk",
    "odc_mclk",
    "throttling",
    "temp_edge",
    "temp_junction",
    "temp_mem",
    "temp_soc",
    "temp_core",
    "temp_l3",
    "block_gpu",
    "block_ee",
)

#: The top-level keys of one JSON Lines record.
JSON_KEYS: frozenset[str] = frozenset(
    {
        "blocks",
        "clocks_mhz",
        "device",
        "fan_rpm",
        "gpu_percent",
        "gtt",
        "mem_percent",
        "monotonic_ns",
        "odc",
        "power",
        "profile",
        "sequence",
        "temperatures_c",
        "throttling",
        "time",
        "vram",
    }
)

MiB = 1024**2


def make_device(**overrides: object) -> AmdgpuDevice:
    """Build a device record."""
    defaults: dict[str, object] = {
        "index": 0,
        "card": "card0",
        "render": "renderD128",
        "bdf": "0000:0c:00.0",
        "vendor_id": 0x1002,
        "device_id": 0x73BF,
        "device_dir": Path("/sys/devices/fake"),
        "hwmon_dir": None,
        "kind": "dgpu",
        "name": "AMD Radeon RX 6800",
        "vram_vendor": "samsung",
        "boot_vga": False,
    }
    defaults.update(overrides)
    return AmdgpuDevice(**defaults)  # type: ignore[arg-type]


def make_metrics(**overrides: object) -> AmdgpuMetrics:
    """Build a metrics record with nothing reported."""
    defaults: dict[str, object] = {
        "device": make_device(),
        "sequence": 1,
        "timestamp_ns": 0,
        "interval_s": 1.0,
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def make_bare_metrics(**overrides: object) -> AmdgpuMetrics:
    """A device on which nothing is readable: every reading absent, never zero."""
    defaults: dict[str, object] = {
        "device": make_device(),
        "sequence": 1,
        "timestamp_ns": 0,
        "interval_s": 1.0,
        "gpu_busy_percent": None,
        "mem_busy_percent": None,
        "vram": MemoryPool(used=None, total=None, source="none"),
        "gtt": MemoryPool(used=None, total=None, source="none"),
        "sclk": Clock(current=None, maximum=None, source="none"),
        "mclk": Clock(current=None, maximum=None, source="none"),
        "socclk": Clock(current=None, maximum=None, source="none"),
        "power": Power(draw_w=None, cap_w=None, source="none"),
        "fan": Fan(rpm=None, max_rpm=None, pwm=None),
        "temperatures": (),
        "throttle": None,
        "blocks": None,
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def make_rich_metrics(**overrides: object) -> AmdgpuMetrics:
    """A device with every reading present, so the rendered values can be pinned."""
    defaults: dict[str, object] = {
        "device": make_device(),
        "sequence": 7,
        "timestamp_ns": 7_000_000,
        "interval_s": 1.0,
        "gpu_busy_percent": 42.5,
        "mem_busy_percent": 17.25,
        "vram": MemoryPool(used=2 * MiB, total=16 * MiB, source="gpu_metrics"),
        "gtt": MemoryPool(used=None, total=None, source="none"),
        "sclk": Clock(current=2200, maximum=2475, source="gpu_metrics"),
        "mclk": Clock(current=1000, maximum=1000, source="gpu_metrics"),
        "socclk": Clock(current=800, maximum=1200, source="gpu_metrics"),
        "power": Power(draw_w=45.0, cap_w=211.0, source="gpu_metrics"),
        "fan": Fan(rpm=2400, max_rpm=3000, pwm=80.0),
        "temperatures": (
            Temperature("edge", 49, "gpu_metrics"),
            Temperature("junction", 56, "gpu_metrics"),
        ),
        "power_profile": "BOOTUP_DEFAULT",
        "throttle": ThrottleInfo(raw=1 << 32, active=frozenset({"TEMP_GPU", "SW_THERMAL"})),
        "blocks": GpuBlocks(
            blocks=(
                BlockReading(key="gpu", label="Graphics pipe", percent=77.0),
                BlockReading(key="vgt", label="Vertex Grouper", percent=12.0),
            )
        ),
    }
    defaults.update(overrides)
    return AmdgpuMetrics(**defaults)  # type: ignore[arg-type]


def make_snapshot(*devices: AmdgpuMetrics, sequence: int = 1) -> GpuSnapshot:
    """Bundle devices into a snapshot with a fixed wall clock."""
    return GpuSnapshot(
        sequence=sequence,
        timestamp_ns=1_000_000 * sequence,
        devices=devices,
        wall_clock_ns=WALL_CLOCK_NS + sequence,
    )


def read_rows(path: Path) -> list[list[str]]:
    """Parse a recorded CSV the way a consumer would, with the stdlib reader."""
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))


def read_record(path: Path) -> dict[str, object]:
    """Parse a one-line JSON record back into a dict."""
    return json.loads(next(iter(read_log(path))))


class TestSplitTarget:
    """The filename decides the format and the transport, independently."""

    def test_csv_is_csv_and_uncompressed(self) -> None:
        assert split_target(Path("s.csv")) == (LogFormat.CSV, False, "s.csv")

    def test_json_is_jsonl_and_uncompressed(self) -> None:
        assert split_target(Path("s.json")) == (LogFormat.JSONL, False, "s.json")

    def test_zst_suffix_compresses_without_changing_the_format(self) -> None:
        assert split_target(Path("s.csv.zst")) == (LogFormat.CSV, True, "s.csv")
        assert split_target(Path("s.json.zst")) == (LogFormat.JSONL, True, "s.json")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("s.csv", LogFormat.CSV),
            ("s.json", LogFormat.JSONL),
            ("s.jsonl", LogFormat.JSONL),
            ("s.ndjson", LogFormat.JSONL),
        ],
    )
    def test_every_accepted_format_suffix(self, name: str, expected: LogFormat) -> None:
        assert split_target(Path(name))[0] is expected

    @pytest.mark.parametrize("suffix", [".zst", ".zstd"])
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("s.ndjson", LogFormat.JSONL), ("s.json", LogFormat.JSONL), ("s.csv", LogFormat.CSV)],
    )
    def test_both_spellings_of_zstd_compress(
        self, name: str, expected: LogFormat, suffix: str
    ) -> None:
        assert split_target(Path(f"{name}{suffix}")) == (expected, True, name)

    def test_ignores_case_in_the_suffix(self) -> None:
        assert split_target(Path("S.CSV.ZST")) == (LogFormat.CSV, True, "S.CSV")

    def test_only_the_basename_is_parsed(self) -> None:
        """A directory called ``logs.csv`` must not decide the format."""
        assert split_target(Path("/var/log/session.json"))[0] is LogFormat.JSONL

    @pytest.mark.parametrize("name", ["s.txt", "s.log", "s", "s.tsv"])
    def test_an_unusable_suffix_raises(self, name: str) -> None:
        """No defensible default: guessing CSV for a file the user named is a lie."""
        with pytest.raises(ValueError, match="cannot tell the log format"):
            split_target(Path(name))


class TestCsvWriting:
    """The CSV path: a stable column set, and absence that stays absent."""

    def test_header_is_written_exactly_once(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        with open_log_context(path) as log:
            for sequence in (1, 2, 3):
                metrics = make_metrics(sequence=sequence)
                assert log.write(make_snapshot(metrics, sequence=sequence)) is True
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        assert lines[0].startswith("time,sequence,device_index,")
        assert lines.count(lines[0]) == 1

    def test_header_names_every_column(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        assert read_rows(path)[0] == list(CSV_COLUMNS)

    def test_every_row_has_the_full_column_count(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_rich_metrics(), make_bare_metrics()))
        rows = read_rows(path)
        assert len(rows) == 3
        for row in rows:
            assert len(row) == len(CSV_COLUMNS)

    def test_absent_reading_is_an_empty_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "absent.csv"
        with open_log_context(path) as log:
            assert log.write(make_snapshot(make_bare_metrics())) is True
        row = read_rows(path)[1]
        cells = dict(zip(CSV_COLUMNS, row, strict=True))
        for column in MEASUREMENT_COLUMNS:
            assert cells[column] == "", f"{column} should be empty, got {cells[column]!r}"

    def test_absent_reading_is_never_zero_or_na(self, tmp_path: Path) -> None:
        """The property that keeps a spreadsheet from inventing data."""
        path = tmp_path / "absent.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_bare_metrics()))
        (row,) = read_rows(path)[1:]
        text = ",".join(row)
        assert "N/A" not in text
        assert "nan" not in text.lower()
        for column in MEASUREMENT_COLUMNS:
            assert not row[CSV_COLUMNS.index(column)].startswith("0")

    def test_present_reading_appears(self, tmp_path: Path) -> None:
        path = tmp_path / "full.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_rich_metrics()))
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        assert cells["sequence"] == "7"
        assert cells["card"] == "card0"
        assert cells["bdf"] == "0000:0c:00.0"
        assert cells["name"] == "AMD Radeon RX 6800"
        assert cells["gpu_percent"] == "42.50"
        assert cells["mem_percent"] == "17.25"
        assert cells["vram_used"] == str(2 * MiB)
        assert cells["vram_total"] == str(16 * MiB)
        assert cells["sclk_mhz"] == "2200.00"
        assert cells["power_w"] == "45.0"
        assert cells["power_cap_w"] == "211.0"
        assert cells["fan_rpm"] == "2400"
        assert cells["profile"] == "BOOTUP_DEFAULT"
        assert cells["temp_edge"] == "49.00"
        assert cells["block_gpu"] == "77.00"

    def test_a_zero_reading_is_recorded_as_zero(self, tmp_path: Path) -> None:
        """Zero is a measurement; it must not be confused with an absent cell."""
        path = tmp_path / "zero.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_bare_metrics(gpu_busy_percent=0.0, fan=Fan(rpm=0))))
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        assert cells["gpu_percent"] == "0.00"
        assert cells["fan_rpm"] == "0"
        # ... and the readings that really are absent stay empty next to them.
        assert cells["power_w"] == ""

    def test_multiple_throttle_reasons_are_one_cell(self, tmp_path: Path) -> None:
        """The reason list is joined with commas, so it has to be quoted."""
        path = tmp_path / "throttled.csv"
        throttle = ThrottleInfo(raw=0, active=frozenset({"TEMP_GPU", "SW"}))
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics(throttle=throttle)))
        rows = read_rows(path)
        assert len(rows[1]) == len(CSV_COLUMNS)
        cells = dict(zip(CSV_COLUMNS, rows[1], strict=True))
        assert cells["throttling"] == "SW,TEMP_GPU"

    def test_a_device_name_containing_a_comma_is_quoted(self, tmp_path: Path) -> None:
        path = tmp_path / "comma.csv"
        metrics = make_metrics(device=make_device(name="AMD Radeon RX 6800, rev C3"))
        with open_log_context(path) as log:
            log.write(make_snapshot(metrics))
        text = path.read_text(encoding="utf-8")
        assert '"AMD Radeon RX 6800, rev C3"' in text
        rows = read_rows(path)
        assert len(rows[1]) == len(CSV_COLUMNS)
        cells = dict(zip(CSV_COLUMNS, rows[1], strict=True))
        assert cells["name"] == "AMD Radeon RX 6800, rev C3"
        assert cells["bdf"] == "0000:0c:00.0"

    def test_a_device_name_containing_a_quote_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "quote.csv"
        metrics = make_metrics(device=make_device(name='Radeon "XT" 6800'))
        with open_log_context(path) as log:
            log.write(make_snapshot(metrics))
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        assert cells["name"] == 'Radeon "XT" 6800'

    def test_the_time_column_is_an_iso_stamp(self, tmp_path: Path) -> None:
        path = tmp_path / "time.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        stamp = datetime.fromisoformat(cells["time"])
        assert stamp.year == 2023

    def test_the_time_column_is_utc_so_lines_stay_in_order(self, tmp_path: Path) -> None:
        """Local time repeats an hour when DST falls back, un-sorting the file.

        The module promises that lines sort chronologically; only a fixed offset
        delivers that in every timezone and across a DST change.
        """
        path = tmp_path / "utc.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        stamp = datetime.fromisoformat(cells["time"])
        assert stamp.utcoffset() == timedelta(0)
        assert stamp.isoformat().endswith("+00:00")

    def test_a_missing_wall_clock_leaves_an_empty_cell(self, tmp_path: Path) -> None:
        """Not guessing a time is better than writing a plausible wrong one."""
        path = tmp_path / "notime.csv"
        snapshot = GpuSnapshot(sequence=1, timestamp_ns=1, devices=(make_metrics(),))
        with open_log_context(path) as log:
            log.write(snapshot)
        cells = dict(zip(CSV_COLUMNS, read_rows(path)[1], strict=True))
        assert cells["time"] == ""

    def test_records_counts_devices_not_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        with open_log_context(path) as log:
            assert log.records == 0
            second = make_metrics(device=make_device(index=1, card="card1", bdf="0000:0d:00.0"))
            assert log.write(make_snapshot(make_metrics(), second)) is True
            assert log.records == 2
            assert log.write(make_snapshot(make_metrics())) is True
            assert log.records == 3
        rows = read_rows(path)
        assert len(rows) == 4
        assert [row[CSV_COLUMNS.index("device_index")] for row in rows[1:]] == ["0", "1", "0"]

    def test_the_file_is_readable_before_close(self, tmp_path: Path) -> None:
        """Flushing per record is what makes a crash cost at most one row."""
        path = tmp_path / "s.csv"
        log = open_log(path)
        try:
            log.write(make_snapshot(make_metrics()))
            assert len(read_rows(path)) == 2
        finally:
            log.close()

    def test_a_snapshot_with_no_devices_adds_no_records(self, tmp_path: Path) -> None:
        """An empty sample must not corrupt the file or disable the recording.

        ``write`` returns ``False`` here even though the log stays usable, which
        contradicts its own docstring ("``False`` means the log has been disabled for the
        rest of the session"); the assertions below stick to the durable facts.
        """
        path = tmp_path / "empty.csv"
        with open_log_context(path) as log:
            log.write(GpuSnapshot(sequence=1, timestamp_ns=1))
            assert log.records == 0
            assert log.is_open
            assert log.write(make_snapshot(make_metrics())) is True
        assert len(read_rows(path)) == 2


class TestJsonLines:
    """The JSON Lines path: one object per device, per line."""

    def test_one_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics(), make_metrics(device=make_device(index=1))))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_two_devices_produce_two_distinct_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        second = make_device(index=1, card="card1", bdf="0000:0d:00.0", name="AMD Radeon 780M")
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics(), make_metrics(device=second)))
        text = path.read_text(encoding="utf-8")
        first, other = (json.loads(line) for line in text.splitlines())
        assert first["device"]["index"] == 0
        assert first["device"]["card"] == "card0"
        assert other["device"]["index"] == 1
        assert other["device"]["card"] == "card1"
        assert other["device"]["name"] == "AMD Radeon 780M"
        assert first != other

    def test_the_record_has_the_expected_top_level_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_rich_metrics(), sequence=7))
        record = read_record(path)
        assert set(record) == JSON_KEYS
        assert record["sequence"] == 7
        assert record["monotonic_ns"] == 7_000_000

    def test_the_record_round_trips_through_json_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_rich_metrics(), sequence=7))
        record = read_record(path)
        assert record["device"] == {
            "index": 0,
            "card": "card0",
            "bdf": "0000:0c:00.0",
            "name": "AMD Radeon RX 6800",
            "integrated": False,
        }
        assert record["gpu_percent"] == 42.5
        assert record["mem_percent"] == 17.25
        assert record["vram"] == {"used": 2 * MiB, "total": 16 * MiB, "percent": 12.5}
        assert record["gtt"] == {"used": None, "total": None, "percent": None}
        assert record["clocks_mhz"] == {"sclk": 2200.0, "mclk": 1000.0, "socclk": 800.0}
        assert record["power"] == {
            "draw_w": 45.0,
            "cap_w": 211.0,
            "percent": pytest.approx(45.0 / 211.0 * 100),
        }
        assert record["fan_rpm"] == 2400
        assert record["temperatures_c"] == {"edge": 49.0, "junction": 56.0}
        assert record["profile"] == "BOOTUP_DEFAULT"
        assert record["throttling"] == ["SW_THERMAL", "TEMP_GPU"]
        assert record["blocks"] == {"busiest": "gpu", "percent": {"gpu": 77.0, "vgt": 12.0}}

    def test_an_absent_reading_is_null(self, tmp_path: Path) -> None:
        """The JSON equivalent of an empty cell: null, never 0 and never a string."""
        path = tmp_path / "absent.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_bare_metrics()))
        record = read_record(path)
        assert record["gpu_percent"] is None
        assert record["mem_percent"] is None
        assert record["fan_rpm"] is None
        assert record["vram"] == {"used": None, "total": None, "percent": None}
        assert record["clocks_mhz"] == {"sclk": None, "mclk": None, "socclk": None}
        assert record["power"] == {"draw_w": None, "cap_w": None, "percent": None}
        assert record["temperatures_c"] == {}
        assert record["throttling"] is None
        assert record["blocks"] is None

    def test_read_log_reads_jsonl_back(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        lines = list(read_log(path))
        assert len(lines) == 1
        assert json.loads(lines[0])["device"]["card"] == "card0"

    def test_no_header_is_written(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
            log.write(make_snapshot(make_metrics(sequence=2), sequence=2))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(line.startswith("{") for line in lines)


class TestZstdCompression:
    """The ``.zst`` transport: a real frame, readable without this module."""

    def test_the_raw_file_starts_with_the_zstd_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv.zst"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        assert path.read_bytes()[:4] == ZSTD_MAGIC

    def test_the_raw_bytes_decompress_to_the_recorded_text(self, tmp_path: Path) -> None:
        """Proves it is a frame, not merely a file with a suggestive name."""
        plain = tmp_path / "plain.csv"
        packed = tmp_path / "packed.csv.zst"
        for path in (plain, packed):
            with open_log_context(path) as log:
                log.write(make_snapshot(make_rich_metrics()))
        assert zstd.decompress(packed.read_bytes()) == plain.read_bytes()

    def test_read_log_decompresses_transparently(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.csv"
        packed = tmp_path / "packed.csv.zst"
        for path in (plain, packed):
            with open_log_context(path) as log:
                for sequence in (1, 2, 3):
                    metrics = make_rich_metrics(sequence=sequence)
                    log.write(make_snapshot(metrics, sequence=sequence))
        assert list(read_log(packed)) == list(read_log(plain))
        assert len(list(read_log(packed))) == 4

    def test_closing_writes_a_complete_frame(self, tmp_path: Path) -> None:
        """An unclosed compressor is a truncated archive, so ``close`` is the contract."""
        path = tmp_path / "s.csv.zst"
        log = open_log(path)
        log.write(make_snapshot(make_metrics()))
        log.close()
        lines = list(read_log(path))
        assert len(lines) == 2
        assert lines[0].split(",")[0] == "time"
        assert zstd.decompress(path.read_bytes()).decode("utf-8") == "\n".join(lines) + "\n"

    def test_the_context_manager_closes_on_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv.zst"
        with pytest.raises(RuntimeError), open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
            raise RuntimeError("sampler died")
        assert zstd.decompress(path.read_bytes()).startswith(b"time,sequence,")

    def test_read_log_handles_the_spelled_out_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json.zstd"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        assert path.read_bytes()[:4] == ZSTD_MAGIC
        assert json.loads(next(iter(read_log(path))))["device"]["card"] == "card0"

    @pytest.mark.parametrize("level", [0, 1, 3, 9, 19, 22])
    def test_every_valid_level_round_trips(self, tmp_path: Path, level: int) -> None:
        path = tmp_path / f"level{level}.csv.zst"
        log = open_log(path, zstd_level=level)
        assert log.is_open
        assert log.problem is None
        log.write(make_snapshot(make_metrics()))
        log.close()
        lines = list(read_log(path))
        assert len(lines) == 2
        assert lines[0].split(",")[0] == "time"

    def test_a_level_outside_the_range_is_reported_not_raised(self, tmp_path: Path) -> None:
        """23 is one past the library maximum; the log degrades instead of exploding."""
        log = open_log(tmp_path / "s.csv.zst", zstd_level=23)
        assert log.problem is LogProblem.OPEN_FAILED
        assert log.is_open is False


class TestFailureHandling:
    """A recording that cannot proceed must not take the monitor down."""

    def test_an_unopenable_path_reports_a_problem(self, tmp_path: Path) -> None:
        """The parent is a regular file, so the directory can never be created."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        log = open_log(blocker / "session.csv")
        assert log.problem is LogProblem.OPEN_FAILED
        assert log.is_open is False
        assert log.problem is not LogProblem.UNKNOWN_SUFFIX

    def test_an_unopenable_path_does_not_raise(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        with open_log_context(blocker / "session.csv") as log:
            assert log.is_open is False
            assert log.write(make_snapshot(make_metrics())) is False

    def test_an_unknown_suffix_reports_a_problem(self, tmp_path: Path) -> None:
        log = open_log(tmp_path / "session.txt")
        assert log.problem is LogProblem.UNKNOWN_SUFFIX
        assert log.is_open is False
        assert not (tmp_path / "session.txt").exists()

    def test_an_unknown_suffix_does_not_raise(self, tmp_path: Path) -> None:
        with open_log_context(tmp_path / "session.txt") as log:
            assert log.write(make_snapshot(make_metrics())) is False
            assert log.records == 0

    def test_the_format_of_an_unknown_suffix_is_still_reported(self, tmp_path: Path) -> None:
        """The unusable value still carries a format, so callers can log something."""
        log = open_log(tmp_path / "session.txt")
        assert isinstance(log.format, LogFormat)

    def test_a_failed_write_disables_the_log(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        log = open_log(path)
        assert log.is_open
        assert log._binary is not None
        log._binary.close()  # a full disk, seen from the writer's side
        assert log.write(make_snapshot(make_metrics())) is False
        assert log.is_open is False
        assert log.problem is LogProblem.UNWRITABLE
        assert log.records == 0

    def test_writes_after_a_failure_are_harmless(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        log = open_log(path)
        assert log._binary is not None
        log._binary.close()
        assert log.write(make_snapshot(make_metrics())) is False
        for _ in range(3):
            assert log.write(make_snapshot(make_metrics())) is False
        assert log.is_open is False
        assert log.records == 0

    def test_closing_a_broken_log_does_not_raise(self, tmp_path: Path) -> None:
        """Even the compressed case, where closing would try to write an epilogue."""
        log = open_log(tmp_path / "s.csv.zst")
        assert log._binary is not None
        log._binary.close()
        assert log.write(make_snapshot(make_metrics())) is False
        log.close()
        assert log.is_open is False

    def test_closing_a_log_that_never_opened_does_not_raise(self, tmp_path: Path) -> None:
        open_log(tmp_path / "session.txt").close()
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        open_log(blocker / "session.csv").close()

    def test_the_first_problem_is_the_one_reported(self, tmp_path: Path) -> None:
        """One warning per session, not one per sample."""
        log = open_log(tmp_path / "session.txt")
        assert log.problem is LogProblem.UNKNOWN_SUFFIX
        log.write(make_snapshot(make_metrics()))
        log.write(make_snapshot(make_metrics()))
        assert log.problem is LogProblem.UNKNOWN_SUFFIX


class TestRoundTrip:
    """What the whole module is for: write a session, read the same session back."""

    def test_three_snapshots_round_trip_through_zstd(self, tmp_path: Path) -> None:
        path = tmp_path / "session.csv.zst"
        with open_log_context(path) as log:
            for sequence in (1, 2, 3):
                metrics = make_rich_metrics(sequence=sequence, gpu_busy_percent=sequence * 10.0)
                assert log.write(make_snapshot(metrics, sequence=sequence)) is True
            assert log.records == 3
        lines = list(read_log(path))
        assert len(lines) == 4
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        assert len(rows) == 4
        header, *body = rows
        assert header[:6] == ["time", "sequence", "device_index", "card", "bdf", "name"]
        assert all(len(row) == len(CSV_COLUMNS) for row in body)
        assert [row[CSV_COLUMNS.index("sequence")] for row in body] == ["1", "2", "3"]
        busy = CSV_COLUMNS.index("gpu_percent")
        assert [row[busy] for row in body] == ["10.00", "20.00", "30.00"]
        assert len({tuple(row) for row in body}) == 3

    def test_the_round_trip_survives_a_json_session(self, tmp_path: Path) -> None:
        path = tmp_path / "session.json.zst"
        with open_log_context(path) as log:
            for sequence in (1, 2, 3):
                metrics = make_rich_metrics(sequence=sequence, gpu_busy_percent=sequence * 10.0)
                log.write(make_snapshot(metrics, sequence=sequence))
        records = [json.loads(line) for line in read_log(path)]
        assert [record["sequence"] for record in records] == [1, 2, 3]
        assert [record["gpu_percent"] for record in records] == [10.0, 20.0, 30.0]
        assert len({record["gpu_percent"] for record in records}) == 3

    def test_reopening_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        """The data of the first session survives; a log must never lose rows."""
        path = tmp_path / "session.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        first = path.read_text(encoding="utf-8")
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics(sequence=2), sequence=2))
        assert path.read_text(encoding="utf-8").startswith(first)
        data = [row for row in read_rows(path) if row[0] != "time"]
        assert [row[CSV_COLUMNS.index("sequence")] for row in data] == ["1", "2"]

    def test_a_restart_does_not_repeat_the_header(self, tmp_path: Path) -> None:
        """Recording twice to one path is ordinary; two headers in one file is not.

        The file is opened in append mode so a second session extends the first, which
        means "already has a header" has to come from the file rather than from the
        in-memory object.  A second header line would land in the middle of the data, and
        every strict CSV reader absorbs it as a record.
        """
        path = tmp_path / "session.csv"
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics(sequence=2), sequence=2))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines.count(lines[0]) == 1
        assert len(read_rows(path)) == 3

    def test_an_existing_but_empty_file_still_gets_a_header(self, tmp_path: Path) -> None:
        """A previous run that wrote nothing left no header to conflict with."""
        path = tmp_path / "session.csv"
        path.write_text("")
        with open_log_context(path) as log:
            log.write(make_snapshot(make_metrics()))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("time,sequence")
        assert len(lines) == 2
