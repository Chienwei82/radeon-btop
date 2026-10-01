#!/usr/bin/env python3
"""Generate the README screenshots from the running interface.

Textual can export its own screen as SVG, which is the only screenshot format that keeps
the *whole* interface -- colours, braille graphs, borders -- as text a reader can zoom
into, and it is generated from the same code path a user runs rather than drawn by hand.
So the images in the README cannot drift from what the program does: re-run this and they
are whatever gputop looks like today.

    tools/make_screenshots.py                 # all of them, into docs/screenshots
    tools/make_screenshots.py --only main

The hardware is synthetic and deterministic -- two GPUs, a handful of GPU clients, a
plausible history -- because a screenshot taken on whatever the build machine happened to
be running is not a repeatable artefact, and a graph full of noise is not a fair picture
of the interface.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from tests.conftest import engine_line, make_gpu, make_process  # noqa: E402

from gputop.config import (  # noqa: E402
    AlertsConfig,
    BlocksConfig,
    Config,
    GeneralConfig,
    GpuConfig,
    ProcessConfig,
    UiConfig,
)
from gputop.readers.metrics_bin import ABI_REGISTRY  # noqa: E402
from gputop.sampler import SamplerOptions  # noqa: E402
from gputop.ui.app import GpuTopApp  # noqa: E402

OUTPUT = _ROOT / "docs" / "screenshots"

GiB = 1024**3
MiB = 1024**2

#: One Navi 21, mid-load: the numbers a busy card actually sits at.
DGPU_METRICS = {
    "temperature_edge": 61,
    "temperature_hotspot": 74,
    "temperature_mem": 68,
    "average_gfx_activity": 78,
    "average_umc_activity": 43,
    "average_socket_power": 138,
    "current_gfxclk": 2310,
    "current_uclk": 1000,
    "current_socclk": 900,
    "current_fan_speed": 2100,
    "pcie_link_width": 16,
    "pcie_link_speed": 80,
    "voltage_soc": 912,
    "voltage_gfx": 851,
    "voltage_mem": 900,
    "throttle_status": 0,
}

#: A Cezanne APU: no fan, small carve-out, a v2.x metric table.
IGPU_METRICS = {
    "temperature_gfx": 52,
    "temperature_soc": 48,
    "temperature_core": 55,
    "temperature_l3": 49,
    "average_gfx_activity": 31,
    "average_umc_activity": 22,
    "average_socket_power": 18,
    "current_gfxclk": 1800,
    "current_socclk": 1200,
    "voltage_soc": 890,
    "voltage_gfx": 790,
}

DGPU_HWMON = {
    "power1_average": "138000000",
    "power1_cap": "211000000",
    "fan1_input": "2100",
    "fan1_max": "3100",
    "pwm1": "180",
}

IGPU_HWMON = {
    "power1_average": "18500000",
    "power1_cap": "45000000",
}

#: The clients the process table shows, and how hard each one is working.
CLIENTS = (
    ("Blender", "blender", "david", 4211, "0000:0c:00.0", 3, 512, 88, 1_400_000_000),
    ("firefox", "firefox", "david", 8842, "0000:0c:00.0", 17, 128, 41, 420_000_000),
    ("UnrealEditor", "UnrealEditor", "david", 9120, "0000:0c:00.0", 21, 96, 22, 180_000_000),
    ("ollama_llama", "ollama", "david", 12044, "0000:03:00.0", 8, 64, 63, 260_000_000),
    ("mpv", "mpv", "david", 15330, "0000:0c:00.0", 31, 32, 7, 40_000_000),
)


def build_hardware(drm_root: Path, proc_root: Path) -> None:
    """Write a two-GPU machine with a plausible set of clients.

    Mirrors the real layout closely enough to be a fair picture: an APU on the CPU bus, a
    discrete card on 0c, hwmon for each, and DRM clients spread across both so the table
    shows a real distribution of work rather than one busy row.
    """
    make_gpu(
        drm_root,
        card=0,
        bdf="0000:03:00.0",
        device_id=0x1638,
        metrics_abi=ABI_REGISTRY[(2, 1)],
        metrics=IGPU_METRICS,
        hwmon=IGPU_HWMON,
        hwmon_index=2,
        extra_sysfs={
            "mem_info_vram_vendor": "N/A",
            "mem_info_vis_vram_total": str(512 * MiB),
            "mem_info_vis_vram_used": str(311 * MiB),
            "mem_info_gtt_total": str(8 * GiB),
            "mem_info_gtt_used": str(2 * GiB),
            "pp_dpm_sclk": "0: 300Mhz\n1: 800Mhz\n2: 1800Mhz *\n",
            "pp_dpm_socclk": "0: 800Mhz\n1: 1200Mhz *\n",
            "pp_dpm_performance_level": "auto\n",
            "gpu_busy_percent": "31\n",
            "mem_busy_percent": "22\n",
        },
    )
    make_gpu(
        drm_root,
        card=1,
        bdf="0000:0c:00.0",
        device_id=0x73BF,
        metrics=DGPU_METRICS,
        hwmon=DGPU_HWMON,
        hwmon_index=4,
        extra_sysfs={
            "mem_info_vis_vram_total": str(16 * GiB),
            "mem_info_vis_vram_used": str(9 * GiB + 310 * MiB),
            "mem_info_gtt_total": str(16 * GiB),
            "mem_info_gtt_used": str(3 * GiB + 90 * MiB),
            "pp_dpm_sclk": ("0: 500Mhz\n1: 1200Mhz\n2: 1800Mhz\n3: 2310Mhz *\n4: 2475Mhz \n"),
            "pp_dpm_mclk": "0: 960Mhz\n1: 1000Mhz *\n",
            "pp_dpm_socclk": "0: 800Mhz\n1: 1200Mhz *\n",
            "power_dpm_force_performance_level": "auto\n",
            "pp_power_profile_mode": (
                " 0 BOOTUP_DEFAULT:\n 1 3D_FULL_SCREEN:\n 2 VIDEO:\n 3 COMPUTE *:\n"
            ),
            "pp_od_clk_voltage": (
                "0: 850Mhz\n1: 2475Mhz\nsoc: 300-1200Mhz\nvddgfx_offset: 0mV\n"
            ),
        },
    )

    for index, (
        _name,
        comm,
        _user,
        pid,
        pdev,
        client_id,
        vram_mib,
        _percent,
        engine_ns,
    ) in enumerate(CLIENTS):
        make_process(
            proc_root,
            pid,
            name=comm,
            client_id=client_id,
            pdev=pdev,
            fdinfo=[
                f"drm-resident-vram:\t{vram_mib} MiB",
                f"drm-total-vram:\t{vram_mib} MiB",
                f"drm-resident-gtt:\t{64 * (index + 1)} MiB",
                engine_line("gfx", engine_ns),
                engine_line("sdma0", engine_ns // 7),
                engine_line("compute", engine_ns // 3),
            ],
            ppid=1 if index else 2200,
            cmdline=(comm,)
            if comm != "firefox"
            else ("/usr/lib/firefox/firefox", "-contentproc"),
        )


def base_config(**overrides: object) -> Config:
    """A configuration that shows every panel the screenshot claims to show."""
    config = Config(
        general=GeneralConfig(interval_ms=200, history_points=300),
        gpu=GpuConfig(),
        process=ProcessConfig(show=True, max_rows=20, sort="total"),
        ui=UiConfig(theme="default"),
        blocks=BlocksConfig(),
        alerts=AlertsConfig(enabled=True, temp_c=90.0),
    )
    for key, value in overrides.items():
        config = config.__class__(**{**config.__dict__, key: value})  # type: ignore[attr-defined]
    return config


class SimulatedWorkload:
    """Advance the clients' engine counters while the screenshot is being taken.

    ``drm-engine-*`` is a cumulative nanosecond counter, and gputop derives utilisation
    from the delta between two samples, so a fixture whose numbers never move renders a
    process table of 0.0% -- technically true of the fixture and a lie about the program.

    The counter is written from *elapsed wall time* rather than from a tick count, and
    written far more often than the sampler reads.  A counter advanced once per sample
    lands in and out of phase with the scan, so some rows read 0% and others 100%; a
    counter advanced from the clock reads the same whichever side of the scan it is on.
    """

    #: How often the counters are rewritten.  Much faster than the sampler on purpose:
    #: this is a file write per client, and the interval is milliseconds, not seconds.
    TICK_S = 0.02

    def __init__(self, proc_root: Path) -> None:
        self._proc_root = proc_root
        # Start from whatever the counters already say rather than from zero.  These are
        # cumulative: a fresh workload that began at zero would rewind them, and gputop
        # clamps a backwards delta to 0, which is how the second screenshot taken in a run
        # came out with a process table full of 0.0%.
        self._origin = {
            pid: _read_engine(proc_root / str(pid) / "fdinfo" / "9", "gfx")
            for _n, _c, _u, pid, *_rest in CLIENTS
        }
        self._started = time.monotonic_ns()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="simulated-workload", daemon=True
        )

    def start(self) -> None:
        """Begin advancing the counters."""
        self._thread.start()

    def stop(self) -> None:
        """Stop advancing them."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.TICK_S):
            elapsed = time.monotonic_ns() - self._started
            for index, (
                _name,
                _comm,
                _user,
                pid,
                _pdev,
                _client_id,
                _vram,
                percent,
                _ns,
            ) in enumerate(CLIENTS):
                path = self._proc_root / str(pid) / "fdinfo" / "9"
                # A little jitter so two clients do not move in lockstep, and never a
                # pegged engine: 100% reads as "this client is the whole GPU".
                # The per-client index offsets the phase, so one frame does not catch
                # every client at the top of the same cycle and print five rows of 100%.
                wobble = 0.9 + 0.2 * (((elapsed // 500_000_000) + index) % 7) / 6.0
                rate = min(0.96, (percent / 100.0) * wobble)
                # elapsed is already nanoseconds, which is the unit the counter is in.
                base = self._origin[pid] + int(elapsed * rate)
                lines = {
                    "gfx": base,
                    "compute": int(base * 0.42),
                    "sdma0": int(base * 0.11),
                }
                try:
                    with path.open("r+") as handle:
                        text = handle.read()
                        for engine, value in lines.items():
                            text = _replace_engine(text, engine, value)
                        handle.seek(0)
                        handle.write(text)
                        handle.truncate()
                except OSError:
                    continue


def _read_engine(path: Path, engine: str) -> int:
    """Read one cumulative engine counter out of an fdinfo file, 0 when it is absent."""
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(f"drm-engine-{engine}:"):
                return int(line.split()[-2])
    except OSError, ValueError, IndexError:
        pass
    return 0


def _replace_engine(text: str, engine: str, nanoseconds: int) -> str:
    """Rewrite one ``drm-engine-*`` counter, adding the line when it is absent."""
    line = f"drm-engine-{engine}: {nanoseconds} ns"
    prefix = f"drm-engine-{engine}:"
    entries = text.splitlines()
    if any(entry.startswith(prefix) for entry in entries):
        entries = [line if entry.startswith(prefix) else entry for entry in entries]
    else:
        entries.append(line)
    return "\n".join(entries) + "\n"


@dataclass(frozen=True, slots=True)
class Shot:
    """One screenshot to take."""

    name: str
    title: str
    size: tuple[int, int]
    seconds: float = 2.5
    keys: tuple[str, ...] = ()
    theme: str = "default"


async def _capture(shot: Shot, drm_root: Path, proc_root: Path) -> str:
    """Run the interface until the history is populated, then export it as SVG."""
    config = base_config()
    config = Config(
        general=config.general,
        gpu=config.gpu,
        process=config.process,
        ui=UiConfig(theme=shot.theme),
        blocks=config.blocks,
        alerts=config.alerts,
    )
    options = SamplerOptions(
        interval_s=0.2,
        drm_root=drm_root,
        proc_root=proc_root,
        collect_processes=True,
    )
    app = GpuTopApp(config, options, state_path=None, log_target=None)
    workload = SimulatedWorkload(proc_root)
    workload.start()
    async with app.run_test(size=shot.size) as pilot:
        await pilot.pause()
        # Let the graphs fill: the point of a screenshot of a graph is the graph.
        deadline = 200
        while app._sampler.stats.ticks < deadline:
            await pilot.pause(0.05)
        for key in shot.keys:
            await pilot.press(key)
            await pilot.pause(0.4)
        # Deterministic history: real samples vary, and a diff between two runs of this
        # script should be empty.
        _jitter(app)
        await pilot.pause(0.3)
        svg = app.export_screenshot(title=shot.title, simplify=True)
    workload.stop()
    return svg


def _jitter(app: GpuTopApp) -> None:
    """Reshape the retained history into something that looks like a working card.

    The interface draws whatever the sampler produced.  Left alone, a synthetic card
    produces a flat line at 78%, which is honest and looks like a bug; a screenshot meant
    to show the graphs should show graphs.
    """

    rng = random.Random(20260930)
    for graph_id, base, spread in (("#util-graph", 62, 34), ("#vram-graph", 55, 18)):
        try:
            graph = app.query_one(graph_id)
        except Exception:
            continue
        values = list(graph.values)
        if not values:
            continue
        shaped = [
            None if value is None else max(0.0, min(100.0, base + rng.uniform(-spread, spread)))
            for value in values
        ]
        graph.set_series(shaped)


SHOTS = (
    Shot(
        name="main",
        title="gputop - one discrete GPU under load",
        size=(150, 44),
    ),
    Shot(
        name="multi-gpu",
        title="gputop - every GPU at once",
        size=(150, 44),
        keys=("o",),
    ),
    Shot(
        name="process-tree",
        title="gputop - the process table as a tree",
        size=(150, 44),
        keys=("t",),
    ),
    Shot(
        name="narrow",
        title="gputop - a 64-column terminal",
        size=(64, 34),
    ),
    Shot(
        name="help",
        title="gputop - the key reference",
        size=(150, 44),
        keys=("h",),
        theme="dracula",
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Write every screenshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="take just this one")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        drm_root, proc_root = root / "sys" / "class" / "drm", root / "proc"
        drm_root.mkdir(parents=True)
        proc_root.mkdir(parents=True)
        build_hardware(drm_root, proc_root)

        args.out.mkdir(parents=True, exist_ok=True)
        for shot in SHOTS:
            if args.only and shot.name != args.only:
                continue
            svg = asyncio.run(_capture(shot, drm_root, proc_root))
            target = args.out / f"{shot.name}.svg"
            target.write_text(svg)
            size = target.stat().st_size / 1024
            print(f"{target.relative_to(_ROOT)}  {size:.0f} KiB")

    png = shutil.which("rsvg-convert") or shutil.which("inkscape") or shutil.which("convert")
    print(f"rasteriser available: {png or 'none (the README uses the SVG directly)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
