"""Per-process accounting tests: fdinfo parsing, deduplication and delta maths."""

import time
from pathlib import Path

import pytest

from gputop.readers.procfs import ProcessCollector, normalise_engine, parse_fdinfo
from tests.conftest import engine_line, make_process, write

NS = 1_000_000_000


def _rewrite_fdinfo(proc_root: Path, pid: int, lines: str, client_id: int = 42) -> None:
    """Replace a fake process's fdinfo body, keeping the standard DRM header keys.

    ``client_id`` defaults to 42, the value :func:`make_process` writes, so a test that
    only bumps a counter keeps the same ``(pdev, client-id)`` key across both samples.
    """
    write(
        proc_root / str(pid) / "fdinfo" / "9",
        f"drm-driver:\tamdgpu\ndrm-client-id:\t{client_id}\ndrm-pdev:\t0000:0c:00.0\n{lines}",
    )


def _set_gfx(proc_root: Path, pid: int, nanoseconds: int, client_id: int = 42) -> None:
    """Set a process's cumulative gfx engine counter."""
    _rewrite_fdinfo(proc_root, pid, f"{engine_line('gfx', nanoseconds)}\n", client_id=client_id)


def _set_compute(proc_root: Path, pid: int, nanoseconds: int, client_id: int = 42) -> None:
    """Set a process's cumulative compute engine counter."""
    _rewrite_fdinfo(
        proc_root,
        pid,
        f"{engine_line('gfx', 0)}\n{engine_line('compute', nanoseconds)}\n",
        client_id=client_id,
    )


def test_parse_fdinfo_reads_a_real_shaped_record() -> None:
    """A record in the kernel's exact format parses into every field."""
    text = "\n".join(
        [
            "pos:\t0",
            "flags:\t02100002",
            "drm-driver:\tamdgpu",
            "drm-client-id:\t71",
            "drm-pdev:\t0000:0c:00.0",
            "drm-total-vram:\t4116 KiB",
            "drm-resident-vram:\t4116 KiB",
            "drm-shared-vram:\t0 B",
            "drm-total-gtt:\t4108 KiB",
            "drm-resident-gtt:\t4108 KiB",
            engine_line("gfx", 173_321),
            engine_line("compute", 4_096),
        ]
    )
    record = parse_fdinfo(text)

    assert record is not None
    assert record.driver == "amdgpu"
    assert record.client_id == 71
    assert record.pdev == "0000:0c:00.0"
    assert record.engines == {"gfx": 173_321, "compute": 4_096}
    # The kernel writes binary IEC prefixes; 4116 KiB is 4214784 bytes, not 4116000.
    assert record.memory["drm-resident-vram"] == 4116 * 1024
    assert record.memory["drm-total-gtt"] == 4108 * 1024


def test_engine_counter_unit_does_not_raise() -> None:
    """The ``ns`` suffix on engine counters must not blow up the parse."""
    record = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-client-id:\t1\n" + engine_line("gfx", 1_000)
    )
    assert record is not None
    assert record.engines["gfx"] == 1_000


def test_non_numeric_engine_value_is_ignored() -> None:
    """A garbage counter yields no entry rather than an exception."""
    record = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-client-id:\t1\ndrm-engine-gfx:\tnot-a-number ns"
    )
    assert record is not None
    assert record.engines == {}


def test_non_amdgpu_driver_is_rejected() -> None:
    """Only the amdgpu driver is in scope; there is no vendor abstraction layer."""
    record = parse_fdinfo("drm-driver:\ti915\ndrm-client-id:\t1\n")
    assert record is None


def test_file_without_drm_keys_is_rejected() -> None:
    """An ordinary fdinfo file is not a DRM record."""
    assert parse_fdinfo("pos:\t0\nflags:\t02100002\n") is None


def test_non_numeric_client_id_is_rejected() -> None:
    """A malformed client id means the record cannot be keyed, so it is dropped."""
    assert parse_fdinfo("drm-driver:\tamdgpu\ndrm-client-id:\tboom\n") is None


def test_engine_active_and_period_override_cumulative() -> None:
    """When the kernel provides active/period pairs, they are authoritative."""
    record = parse_fdinfo(
        "\n".join(
            [
                "drm-driver:\tamdgpu",
                "drm-client-id:\t5",
                "drm-engine-gfx:\t999_999",
                "drm-engine-active-gfx:\t250",
                "drm-engine-period-gfx:\t1000",
            ]
        )
    )
    assert record is not None
    assert record.engines["gfx"] == 250


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gfx", "gfx"),
        ("compute", "compute"),
        ("sdma0", "dma"),
        ("sdma1", "dma"),
        ("vangogh", "enc"),
        ("vcnenc", "enc"),
        ("jpeg", "dec"),
        ("vdec", "dec"),
        ("vdpdec", "dec"),
        ("uvd", "dec"),
        ("something-new", "other"),
    ],
)
def test_engine_normalisation(raw: str, expected: str) -> None:
    """Engine names fold into stable display buckets, unknown ones into ``other``."""
    assert normalise_engine(raw) == expected


def test_collection_finds_a_client(proc_root: Path) -> None:
    """A process holding a DRM descriptor is reported."""
    make_process(proc_root, 100, fdinfo=["drm-resident-vram:\t8192 KiB"])

    processes, visible, total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert total == 1
    assert visible == 1
    assert len(processes) == 1
    assert processes[0].pid == 100
    assert processes[0].name == "testproc"
    assert processes[0].vram_used == 8192 * 1024


def test_non_drm_descriptors_are_ignored(proc_root: Path) -> None:
    """A process with no descriptor under ``/dev/dri`` is not a GPU client."""
    pid_dir = proc_root / "100"
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "fdinfo").mkdir()
    write(pid_dir / "comm", "daemon\n")
    for index, target in enumerate(("/dev/null", "/etc/passwd", "/tmp/sock"), start=20):
        (pid_dir / "fd" / str(index)).symlink_to(target)

    processes, visible, total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert processes == ()
    assert total == 1
    # Inspectable, so it counts as *visible*.  ``visible`` is about /proc permissions,
    # not about holding a GPU descriptor: the two counts differing is what reveals that
    # other users' processes are hidden, and a plain daemon is not hidden from anyone.
    assert visible == 1


def test_duplicate_client_ids_are_merged_not_summed(proc_root: Path) -> None:
    """Two descriptors onto one DRM client must collapse to a single row.

    Summing would double the memory, which is the classic double-counting bug this
    deduplication exists to prevent.  Both processes deliberately report identical
    counters, so only ``max`` can produce the single reported value.
    """
    for pid in (100, 200):
        make_process(proc_root, pid)
        write(
            proc_root / str(pid) / "fdinfo" / "9",
            "drm-driver:\tamdgpu\ndrm-client-id:\t42\ndrm-pdev:\t0000:0c:00.0\n"
            "drm-resident-vram:\t4096 KiB\ndrm-engine-gfx: 10000 ns\n",
        )

    processes, _visible, _total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert len(processes) == 1
    # Maximum, not the sum of 4096 + 4096.
    assert processes[0].vram_used == 4096 * 1024
    # The lowest pid is reported so the row is stable between scans.
    assert processes[0].pid == 100


def test_same_client_id_on_different_gpus_stays_separate(proc_root: Path) -> None:
    """``drm-pdev`` is part of the key, so one id on two GPUs is two rows.

    This is what makes multi-GPU and dGPU+APU attribution correct instead of blended.
    """
    make_process(proc_root, 100)
    write(
        proc_root / "100" / "fdinfo" / "9",
        "drm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-pdev:\t0000:0c:00.0\n",
    )
    make_process(proc_root, 200, drm_node="/dev/dri/renderD129")
    write(
        proc_root / "200" / "fdinfo" / "9",
        "drm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-pdev:\t0000:03:00.0\n",
    )

    processes, _visible, _total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert len(processes) == 2
    assert {p.bdf for p in processes} == {"0000:0c:00.0", "0000:03:00.0"}


def test_engine_utilisation_is_none_on_first_sample(proc_root: Path) -> None:
    """Without a baseline there is no delta, so utilisation is unknown, not zero."""
    make_process(proc_root, 100, fdinfo={engine_line("gfx", 500_000_000)})

    processes, _visible, _total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    usage = processes[0].engines[0]
    assert usage.total_ns == 500_000_000
    assert usage.delta_ns == 0
    assert usage.percent is None


def test_engine_delta_computes_utilisation(proc_root: Path) -> None:
    """A known delta over a known window yields the correct percentage."""
    collector = ProcessCollector(proc_root)
    make_process(proc_root, 100, fdinfo={engine_line("gfx", 100_000_000)})
    collector.collect(0)
    _set_gfx(proc_root, 100, 350_000_000)

    # One second window, 250 ms of engine time => 25%.
    processes, _visible, _total = collector.collect(NS)

    assert processes[0].engines[0].delta_ns == 250_000_000
    assert processes[0].engines[0].window_ns == NS
    assert processes[0].engine_percent == 25.0


def test_engine_delta_is_clamped_to_one_hundred(proc_root: Path) -> None:
    """Concurrent engines can outrun the wall clock; the reported figure caps at 100."""
    collector = ProcessCollector(proc_root)
    make_process(proc_root, 100, fdinfo={engine_line("gfx", 0)})
    collector.collect(0)
    _set_gfx(proc_root, 100, 5 * NS)

    processes, _visible, _total = collector.collect(NS)

    assert processes[0].engine_percent == 100.0


def test_counter_going_backwards_does_not_produce_negative_usage(proc_root: Path) -> None:
    """A counter reset (client recreated) must not yield negative utilisation."""
    collector = ProcessCollector(proc_root)
    make_process(proc_root, 100, fdinfo={engine_line("gfx", 400_000_000)})
    collector.collect(0)
    _set_gfx(proc_root, 100, 1_000_000)

    processes, _visible, _total = collector.collect(NS)

    assert processes[0].engines[0].delta_ns == 0
    assert processes[0].engine_percent == 0.0


def test_multi_engine_usage_sums_then_clamps(proc_root: Path) -> None:
    """A client busy on two engines sums, and still caps at 100."""
    collector = ProcessCollector(proc_root)
    make_process(
        proc_root,
        100,
        fdinfo={engine_line("gfx", 0), engine_line("compute", 0)},
    )
    collector.collect(0)
    # One rewrite so the gfx counter is not clobbered by setting compute.
    _rewrite_fdinfo(
        proc_root,
        100,
        f"{engine_line('gfx', 400_000_000)}\n{engine_line('compute', 900_000_000)}\n",
    )

    processes, _visible, _total = collector.collect(NS)

    assert processes[0].engine_percent == 100.0  # 40% + 90% clamped


def test_empty_proc_root_yields_nothing(tmp_path: Path) -> None:
    """A missing or empty procfs yields empty results, never an exception."""
    collector = ProcessCollector(tmp_path / "does-not-exist")
    processes, visible, total = collector.collect(time.monotonic_ns())
    assert processes == ()
    assert (visible, total) == (0, 0)


def test_non_numeric_pid_directories_are_skipped(proc_root: Path) -> None:
    """``/proc`` contains non-numeric entries that are not processes."""
    (proc_root / "self").mkdir()
    (proc_root / "sys").mkdir()
    make_process(proc_root, 100)

    _processes, _visible, total = ProcessCollector(proc_root).collect(time.monotonic_ns())
    assert total == 1


def test_vanished_process_is_skipped(proc_root: Path) -> None:
    """A process that exits mid-scan is skipped, not fatal.

    ``fd`` is created but the process entry is a dangling symlink, which is exactly the
    shape of a process that exited between ``scandir`` and ``readlink``.
    """
    make_process(proc_root, 100)
    ghost = proc_root / "200"
    ghost.mkdir()
    (ghost / "fd").symlink_to(proc_root / "999" / "fd")
    (ghost / "fdinfo").symlink_to(proc_root / "999" / "fdinfo")

    processes, _visible, total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert total == 2
    assert len(processes) == 1


def test_reset_forgets_the_delta_baseline(proc_root: Path) -> None:
    """After a rescan the baseline is dropped, so usage is unknown again."""
    collector = ProcessCollector(proc_root)
    make_process(proc_root, 100, fdinfo={engine_line("gfx", 0)})
    collector.collect(0)
    _set_gfx(proc_root, 100, 10_000_000)
    assert collector.collect(NS)[0][0].engine_percent == 1.0

    collector.reset()
    assert collector.collect(2 * NS)[0][0].engine_percent == 0.0


def test_processes_are_sorted_busiest_first(proc_root: Path) -> None:
    """Rows are ordered by utilisation so the interesting ones stay on screen."""
    collector = ProcessCollector(proc_root)
    for pid in (100, 200, 300):
        make_process(proc_root, pid, fdinfo={engine_line("gfx", 0)})
        write(
            proc_root / str(pid) / "fdinfo" / "9",
            f"drm-driver:\tamdgpu\ndrm-client-id:\t{pid}\ndrm-pdev:\t0000:0c:00.0\n"
            "drm-engine-gfx: 0 ns\n",
        )
    collector.collect(0)
    for pid, busy in ((100, 100_000_000), (200, 700_000_000), (300, 400_000_000)):
        _set_gfx(proc_root, pid, busy, client_id=pid)

    processes, _visible, _total = collector.collect(NS)

    assert [p.pid for p in processes] == [200, 300, 100]


def test_cpu_memory_pool_is_captured(proc_root: Path) -> None:
    """CPU-visible memory is captured, since it dominates on an APU."""
    make_process(proc_root, 100, fdinfo=["drm-resident-cpu:\t16384 KiB"])

    processes, _visible, _total = ProcessCollector(proc_root).collect(time.monotonic_ns())

    assert processes[0].cpu_used == 16384 * 1024
    assert processes[0].memory_used == 16384 * 1024
