#!/usr/bin/env python3
"""Measure what gputop costs: the sampling tick, the interface render, and memory.

Modes, each printing a report to stdout and optionally a ``.prof`` file that
``python -m pstats`` can read:

    tools/profile_gputop.py tick       --ticks 30
    tools/profile_gputop.py procfs     --scaling
    tools/profile_gputop.py render     --seconds 10 --size 120x40
    tools/profile_gputop.py memory     --seconds 10

* ``tick`` runs the sampler synchronously on the live machine and then times each
  sub-system separately, so the parts can be compared without a profiler's overhead
  distorting the total.
* ``procfs`` times the ``/proc`` scan alone.  It is the only part of a tick whose cost
  grows with the *system* rather than with the number of GPUs, so ``--scaling`` measures
  it against synthetic trees of 5 to 4000 processes.
* ``render`` drives the real application headlessly through ``App.run_test`` and profiles
  the event loop: layout, painting and the sampler thread together, which is what a user
  actually costs.
* ``memory`` reports resident set size and peak Python allocation for a live interface.

``--synthetic P:C`` builds a throwaway ``/proc`` with ``P`` processes of which ``C`` hold
DRM descriptors, and points the scan at it.  The live ``/proc`` on a container is a poor
substitute: it holds five processes and no GPU client, so a profile taken against it says
nothing about a desktop.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import os
import pstats
import resource
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from gputop.config import Config  # noqa: E402
from gputop.readers.procfs import ProcessCollector  # noqa: E402
from gputop.sampler import Sampler, SamplerOptions  # noqa: E402

MIB = 1024 * 1024


# --------------------------------------------------------------------------- resources


def rss_bytes() -> int:
    """Current resident set size, from ``/proc/self/statm`` (cheap; no subprocess)."""
    try:
        pages = int(Path("/proc/self/statm").read_bytes().split()[1])
    except OSError, IndexError, ValueError:
        return 0
    return pages * os.sysconf("SC_PAGE_SIZE")


def peak_rss_bytes() -> int:
    """Peak resident set size for the process, from ``getrusage``."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def cpu_seconds() -> float:
    """User+system CPU consumed by every thread of this process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def describe(values: list[float], unit: str = "ms") -> str:
    """One-line distribution summary, so outliers are visible in the output."""
    if not values:
        return "no samples"
    ordered = sorted(values)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return (
        f"mean {statistics.fmean(ordered):9.2f}{unit}  "
        f"p50 {ordered[len(ordered) // 2]:9.2f}{unit}  "
        f"p95 {ordered[index]:9.2f}{unit}  "
        f"max {ordered[-1]:9.2f}{unit}  n={len(ordered)}"
    )


def report_top(profile: cProfile.Profile, count: int) -> None:
    """Print the hottest functions, and separately where the time was called from."""
    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("tottime")
    print()
    print(f"top {count} by self time:")
    stats.print_stats(count)
    stats.sort_stats("cumulative")
    print(f"top {count} by cumulative time:")
    stats.print_stats(count)


# ------------------------------------------------------------------------- fake trees


def build_proc_tree(root: Path, processes: int, clients: int) -> Path:
    """Create a ``/proc`` stand-in with DRM clients in it.

    Reuses the test fixtures, because the point of this tree is to be *the same* shape the
    tests pin: a profile against a differently-shaped tree measures the wrong thing.
    """
    from tests.conftest import make_process

    root.mkdir(parents=True, exist_ok=True)
    for pid in range(processes):
        if pid < clients:
            make_process(
                root,
                pid + 100,
                name=f"gpu-client-{pid}",
                client_id=pid + 1,
                fdinfo=[f"drm-engine-gfx: {pid * 1000} ns", "drm-resident-vram:\t256 MiB"],
            )
        else:
            make_process(root, pid + 100, name=f"proc-{pid}", drm_node="/dev/null")
    return root


def synthetic_request(value: str | None) -> tuple[int, int] | None:
    """Parse ``--synthetic P:C`` into a process/client count."""
    if not value:
        return None
    processes, _, clients = value.partition(":")
    return int(processes), int(clients or processes)


# ----------------------------------------------------------------------- tick profiling


@dataclass(slots=True)
class Report:
    """What one mode measured."""

    label: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    rss_mib: float = 0.0
    peak_rss_mib: float = 0.0
    ticks: int = 0
    detail: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    def print(self) -> None:
        print()
        print(f"== {self.label} ==")
        print(f"wall            {self.wall_s:.2f} s over {self.ticks} ticks")
        share = self.cpu_s / self.wall_s * 100 if self.wall_s else 0.0
        print(f"cpu             {self.cpu_s:.3f} s ({share:.1f}% of wall, all threads)")
        print(f"rss             {self.rss_mib:.1f} MiB (peak {self.peak_rss_mib:.1f} MiB)")
        for line in self.lines:
            print(line)
        for key, value in self.detail.items():
            print(f"{key:<15} {value}")


def timed(count: int, call: object, *, pause: float = 0.0) -> list[float]:
    """Run ``call`` ``count`` times, returning the per-call duration in milliseconds."""
    durations: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        call()  # type: ignore[operator]
        elapsed = (time.perf_counter() - started) * 1000
        durations.append(elapsed)
        if pause:
            time.sleep(max(0.0, pause - elapsed / 1000))
    return durations


def profile_ticks(args: argparse.Namespace) -> Report:
    """Run the sampler on the live machine, then time each sub-system separately."""
    options = SamplerOptions(
        interval_s=args.interval,
        collect_processes=not args.no_processes,
        drm_root=args.drm_root,
        proc_root=args.proc_root,
    )
    sampler = Sampler(options)
    devices = sampler.discover()
    if not devices:
        raise SystemExit(f"no amdgpu device under {args.drm_root}")
    sampler.sample_once()

    collector = ProcessCollector(args.proc_root)
    collector.collect(time.monotonic_ns())

    gc.collect()
    before_rss = rss_bytes()
    wall_start, cpu_start = time.monotonic(), cpu_seconds()
    full = timed(args.ticks, sampler.sample_once, pause=args.interval)
    wall = time.monotonic() - wall_start
    cpu = cpu_seconds() - cpu_start

    # Sub-systems, measured without the interval pause and without a profiler attached:
    # this is the number that says whether a tick can keep up with its own deadline.
    per_device = timed(args.ticks, lambda: sampler._sample_device(devices[0]))
    per_proc = timed(args.ticks, lambda: collector.collect(time.monotonic_ns()))

    tracemalloc.start()
    tracemalloc.reset_peak()
    sampler.sample_once()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    snapshot = sampler.latest() or sampler.sample_once()
    return Report(
        label="sampler tick (live machine)",
        wall_s=wall,
        cpu_s=cpu,
        rss_mib=rss_bytes() / MIB,
        peak_rss_mib=peak_rss_bytes() / MIB,
        ticks=args.ticks,
        lines=[
            f"tick            {describe(full)}",
            f"one device      {describe(per_device)}",
            f"procfs scan     {describe(per_proc)}",
            "",
            f"per tick: device {statistics.fmean(per_device):.2f} ms, "
            f"procfs {statistics.fmean(per_proc):.2f} ms, "
            f"overhead {statistics.fmean(full) - statistics.fmean(per_device) - statistics.fmean(per_proc):.2f} ms",
        ],
        detail={
            "devices": str(len(devices)),
            "proc dirs": str(snapshot.total_process_count),
            "python heap peak": f"{peak / MIB:.2f} MiB for one tick",
            "rss growth": f"{(rss_bytes() - before_rss) / MIB:+.1f} MiB over the run",
            "budget at 1 Hz": f"{statistics.fmean(full) / (args.interval * 1000) * 100:.1f}% of a second",
        },
    )


def profile_procfs(args: argparse.Namespace) -> Report:
    """Time the ``/proc`` scan alone, and its scaling with system size."""
    report = Report(label="procfs scan")
    profile = cProfile.Profile() if args.profile else None

    if args.scaling:
        # The one measurement that cannot be taken against this container's /proc, which
        # holds five processes and no GPU client.
        with TemporaryDirectory() as tmp:
            for count in (5, 25, 100, 400, 1000, 4000):
                clients = max(1, count // 20)
                root = build_proc_tree(Path(tmp) / f"p{count}", count, clients)
                collector = ProcessCollector(root)
                collector.collect(time.monotonic_ns())
                durations = timed(5, partial(collector.collect, time.monotonic_ns()))
                mean = statistics.fmean(durations)
                report.lines.append(
                    f"{count:>6} processes ({clients:>4} GPU clients): "
                    f"{mean:8.2f} ms  = {mean * 1000 / count:6.1f} us per process"
                )
        report.detail["note"] = "scan cost is linear in process count"
        report.print()
        return report

    collector = ProcessCollector(args.proc_root)
    gc.collect()
    before = rss_bytes()
    wall_start, cpu_start = time.monotonic(), cpu_seconds()
    if profile is not None:
        profile.enable()
    durations: list[float] = []
    processes: tuple[object, ...] = ()
    visible = total = 0
    for _ in range(args.ticks):
        processes, visible, total = collector.collect(time.monotonic_ns())
        durations.extend(
            timed(1, lambda: collector.collect(time.monotonic_ns()), pause=args.interval)
        )
    if profile is not None:
        profile.disable()
    report.wall_s = time.monotonic() - wall_start
    report.cpu_s = cpu_seconds() - cpu_start
    report.ticks = len(durations)
    report.rss_mib = rss_bytes() / MIB
    report.peak_rss_mib = peak_rss_bytes() / MIB
    report.lines = [f"scan            {describe(durations)}"]
    report.detail = {
        "processes seen": str(total),
        "with DRM clients": str(visible),
        "clients in table": str(len(processes)),
        "rss growth": f"{(rss_bytes() - before) / MIB:+.1f} MiB over the run",
    }
    if profile is not None:
        if args.save:
            profile.dump_stats(args.save)
            report.detail["profile written"] = str(args.save)
        report_top(profile, args.top)
    return report


# ---------------------------------------------------------------------- render profiling


async def _drive(args: argparse.Namespace, app: object) -> int:
    """Run an application headlessly for a fixed wall-clock window."""
    width, height = (int(part) for part in args.size.lower().split("x", 1))
    frames = 0
    runner = app.run_test(size=(width, height))  # type: ignore[attr-defined]
    async with runner as pilot:
        await pilot.pause()
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            await asyncio.sleep(0.05)
            frames += 1
    return frames


def _build_app(args: argparse.Namespace) -> object:
    """Construct the real application against the requested roots."""
    from gputop.ui.app import GpuTopApp

    options = SamplerOptions(
        interval_s=args.interval,
        drm_root=args.drm_root,
        proc_root=args.proc_root,
    )
    config = Config()
    return GpuTopApp(config, options, state_path=None, log_target=args.log)


def profile_render(args: argparse.Namespace) -> Report:
    """Profile the whole interface: event loop, layout, painting and the sampler."""
    app = _build_app(args)
    gc.collect()
    before = rss_bytes()
    wall_start, cpu_start = time.monotonic(), cpu_seconds()
    profile = cProfile.Profile() if args.profile else None
    if profile is not None:
        profile.enable()
    frames = asyncio.run(_drive(args, app))
    if profile is not None:
        profile.disable()
    wall = time.monotonic() - wall_start
    cpu = cpu_seconds() - cpu_start
    sampler = app._sampler  # type: ignore[attr-defined]
    ticks = sampler.stats.ticks
    report = Report(
        label="interface render (headless, live sampler thread)",
        wall_s=wall,
        cpu_s=cpu,
        rss_mib=rss_bytes() / MIB,
        peak_rss_mib=peak_rss_bytes() / MIB,
        ticks=ticks,
        lines=[f"cpu per second  {cpu / max(wall, 1e-9):.3f} s of CPU per wall second"],
        detail={
            "terminal": args.size,
            "samples drawn": str(ticks),
            "pilot frames": str(frames),
            "avg sample cost": f"{sampler.stats.avg_duration_ms:.2f} ms (sampler self-report)",
            "cpu per sample": f"{cpu / ticks * 1000:.1f} ms" if ticks else "n/a",
            "rss growth": f"{(rss_bytes() - before) / MIB:+.1f} MiB over the run",
        },
    )
    if profile is not None:
        if args.save:
            profile.dump_stats(args.save)
            report.detail["profile written"] = str(args.save)
        report_top(profile, args.top)
    return report


def profile_memory(args: argparse.Namespace) -> Report:
    """Report the memory a live interface holds, sampled while it runs."""
    samples: list[int] = []

    async def watch() -> None:
        app = _build_app(args)
        runner = app.run_test(size=(120, 40))  # type: ignore[attr-defined]
        async with runner as pilot:
            await pilot.pause()
            started = time.monotonic()
            while time.monotonic() - started < args.seconds:
                await asyncio.sleep(0.25)
                samples.append(rss_bytes())

    app = _build_app(args)
    del app
    gc.collect()
    before = rss_bytes()
    tracemalloc.start()
    tracemalloc.reset_peak()
    cpu_start = cpu_seconds()
    asyncio.run(watch())
    cpu = cpu_seconds() - cpu_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Report(
        label="memory (interface running)",
        wall_s=args.seconds,
        cpu_s=cpu,
        rss_mib=rss_bytes() / MIB,
        peak_rss_mib=peak_rss_bytes() / MIB,
        lines=[f"rss over time   {describe([value / MIB for value in samples], 'MiB')}"],
        detail={
            "rss before app": f"{before / MIB:.1f} MiB",
            "rss now": f"{rss_bytes() / MIB:.1f} MiB",
            "peak rss": f"{peak_rss_bytes() / MIB:.1f} MiB",
            "python alloc peak": f"{peak / MIB:.1f} MiB",
            "free-threaded": str(not sys._is_gil_enabled()),
            "samples": str(len(samples)),
        },
    )


# ------------------------------------------------------------------------------ driver

MODES = {
    "tick": profile_ticks,
    "procfs": profile_procfs,
    "render": profile_render,
    "memory": profile_memory,
}


def main(argv: list[str] | None = None) -> int:
    """Run one profiling mode and print its report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--ticks", type=int, default=30, help="ticks to run (tick, procfs)")
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="seconds to run (render, memory)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="sampling interval, seconds"
    )
    parser.add_argument(
        "--size", default="120x40", help="terminal size for render, e.g. 120x40"
    )
    parser.add_argument("--drm-root", type=Path, default=Path("/sys/class/drm"))
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--log", type=Path, default=None, help="also record the session here")
    parser.add_argument("--no-processes", action="store_true", help="skip the /proc scan")
    parser.add_argument(
        "--synthetic",
        metavar="P:C",
        default=None,
        help="build a fake /proc with P processes, C of them GPU clients",
    )
    parser.add_argument("--scaling", action="store_true", help="procfs: sweep process counts")
    parser.add_argument(
        "--profile", action="store_true", help="attach cProfile and print the hot spots"
    )
    parser.add_argument("--save", type=Path, default=None, help="write the profile for pstats")
    parser.add_argument("--top", type=int, default=15, help="how many profile rows to print")
    args = parser.parse_args(argv)

    if args.synthetic:
        processes, clients = synthetic_request(args.synthetic)
        tmp = TemporaryDirectory()
        args.proc_root = build_proc_tree(Path(tmp.name) / "proc", processes, clients)
        print(
            f"synthetic /proc: {processes} processes, {clients} GPU clients at {args.proc_root}"
        )

    if args.profile and args.save is None:
        args.save = Path(f"gputop-{args.mode}.prof")

    report = MODES[args.mode](args)
    report.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
