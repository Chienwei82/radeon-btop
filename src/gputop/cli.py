"""Command line interface.

``gputop`` with no arguments starts the Textual interface.  ``--dump`` writes a single
JSON snapshot to stdout and exits, which is what makes the tool scriptable and what the
test-suite uses to exercise the whole data path headlessly.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from gputop.config import Config, load_config
from gputop.model.device import DeviceKind
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.process import GpuProcess
from gputop.model.snapshot import GpuSnapshot
from gputop.sampler import Sampler, SamplerOptions
from gputop.ui.theme import theme_names

PROG = "gputop"
VERSION = "0.1.0"

#: Exit status used when no AMD GPU is present, so scripts can distinguish "no hardware"
#: from "gputop is broken".
EXIT_NO_DEVICE = 3


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="A btop-style terminal monitor for AMD Radeon GPUs.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {VERSION}")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="PATH",
        help="configuration file to load instead of searching the default locations",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        metavar="SECONDS",
        help="sampling interval, overriding general.interval_ms",
    )
    parser.add_argument(
        "--drm-root",
        type=Path,
        metavar="PATH",
        help="sysfs DRM class directory (default /sys/class/drm)",
    )
    parser.add_argument(
        "--proc-root",
        type=Path,
        metavar="PATH",
        help="procfs mount point (default /proc)",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="write one JSON snapshot to stdout and exit",
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="list discovered GPUs as JSON and exit",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "igpu", "dgpu"),
        help="force every device to be classified as integrated or discrete",
    )
    parser.add_argument(
        "--theme",
        choices=theme_names(),
        metavar="NAME",
        help="colour theme, one of: " + ", ".join(theme_names()),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable 24-bit colour and fall back to terminal ANSI colours",
    )
    parser.add_argument(
        "--no-processes",
        action="store_true",
        help="skip the /proc scan, which is the most expensive part of a sample",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent --dump output for human reading",
    )
    return parser


def _apply_ui_flags(config: Config, args: argparse.Namespace) -> Config:
    """Overlay the display flags onto the loaded configuration.

    The UI reads appearance from :class:`Config` rather than from the argument vector, so
    that a flag and its config-file equivalent take exactly the same path.
    """
    if args.theme is None and not args.no_color:
        return config
    return replace(
        config,
        ui=replace(
            config.ui,
            theme=args.theme or config.ui.theme,
            no_color=args.no_color or config.ui.no_color,
        ),
    )


def _kind(value: str | None) -> DeviceKind | None:
    """Translate the ``--kind`` flag into a device-kind override."""
    # ``argparse`` restricts this to the three known choices, so the narrowing below is
    # a type-level restatement of what the parser already guarantees.
    match value:
        case "igpu" | "dgpu":
            return value
        case _:
            return None


def sampler_options(args: argparse.Namespace, config: Config) -> SamplerOptions:
    """Merge config defaults with command line overrides."""
    return SamplerOptions(
        interval_s=args.interval if args.interval else config.interval_s,
        history_length=config.history_length,
        collect_processes=config.process.show and not args.no_processes,
        drm_root=args.drm_root or Path("/sys/class/drm"),
        proc_root=args.proc_root or Path("/proc"),
        # The flag wins over the config; both go through the same narrowing so a
        # misspelled config value degrades to "auto" rather than reaching the sampler.
        kind_override=_kind(args.kind) or _kind(config.gpu.kind),
        name_overrides=config.gpu.names,
    )


def _memory(payload: Any) -> dict[str, Any] | None:
    """Serialise a memory pool, or ``None`` when both readings are missing."""
    if payload is None:
        return None
    return {"used": payload.used, "total": payload.total, "percent": payload.percent}


def _device_to_dict(metrics: AmdgpuMetrics) -> dict[str, Any]:
    """Serialise one device's metrics.

    Optional readings are emitted as ``null`` rather than omitted, so a consumer can tell
    "not available on this hardware" apart from "not collected".
    """
    return {
        "index": metrics.device.index,
        "card": metrics.device.card,
        "render": metrics.device.render,
        "bdf": metrics.device.bdf,
        "name": metrics.device.name,
        "kind": metrics.device.kind,
        "integrated": metrics.device.is_integrated,
        "device_id": f"0x{metrics.device.device_id:04x}",
        "boot_vga": metrics.device.boot_vga,
        "metrics_abi": metrics.metrics_abi,
        "gpu_busy_percent": metrics.gpu_busy_percent,
        "mem_busy_percent": metrics.mem_busy_percent,
        "vram": _memory(metrics.vram),
        "gtt": _memory(metrics.gtt),
        "sclk": {
            "current": metrics.sclk.current,
            "maximum": metrics.sclk.maximum,
            "source": metrics.sclk.source,
        },
        "mclk": {
            "current": metrics.mclk.current,
            "maximum": metrics.mclk.maximum,
            "source": metrics.mclk.source,
        },
        "socclk": {
            "current": metrics.socclk.current,
            "maximum": metrics.socclk.maximum,
            "source": metrics.socclk.source,
        },
        "temperatures": [
            {"label": t.label, "celsius": t.celsius, "source": t.source}
            for t in metrics.temperatures
        ],
        "voltages_mv": dict(metrics.voltages_mv),
        "power": {
            "draw_w": metrics.power.draw_w,
            "cap_w": metrics.power.cap_w,
            "energy_mj": metrics.power.energy_mj,
            "source": metrics.power.source,
        },
        "fan": {
            "rpm": metrics.fan.rpm,
            "max_rpm": metrics.fan.max_rpm,
            "pwm_percent": metrics.fan.percent,
            "stopped": metrics.fan.stopped,
        },
        "performance_level": metrics.performance_level,
        "power_profile": metrics.power_profile,
        "throttle": (
            None
            if metrics.throttle is None
            else {
                "raw": metrics.throttle.raw,
                "active": sorted(metrics.throttle.active),
                "is_throttling": metrics.throttle.is_throttling,
            }
        ),
        "pcie": (
            None
            if metrics.pcie is None
            else {
                "width": metrics.pcie.width,
                "speed": metrics.pcie.speed,
                "generation": metrics.pcie.generation,
                "describe": metrics.pcie.describe(),
            }
        ),
    }


def _process_to_dict(process: GpuProcess) -> dict[str, Any]:
    """Serialise one DRM client."""
    return {
        "pid": process.pid,
        "name": process.name,
        "user": process.user,
        "bdf": process.bdf,
        "client_id": process.client_id,
        "engine_percent": process.engine_percent,
        "engines": [
            {
                "engine": usage.engine,
                "total_ns": usage.total_ns,
                "delta_ns": usage.delta_ns,
                "window_ns": usage.window_ns,
                "percent": usage.percent,
            }
            for usage in process.engines
        ],
        "vram_used": process.vram_used,
        "gtt_used": process.gtt_used,
        "cpu_used": process.cpu_used,
        "memory_used": process.memory_used,
    }


def snapshot_to_dict(snapshot: GpuSnapshot) -> dict[str, Any]:
    """Serialise a whole snapshot."""
    return {
        "sequence": snapshot.sequence,
        "timestamp_ns": snapshot.timestamp_ns,
        "metrics_abi": snapshot.source.metrics_abi,
        "using_binary_metrics": snapshot.source.using_binary_metrics,
        "devices": [_device_to_dict(d) for d in snapshot.devices],
        "processes": [_process_to_dict(p) for p in snapshot.processes],
        "process_visibility": {
            "visible": snapshot.visible_process_count,
            "total": snapshot.total_process_count,
            "partial": snapshot.is_partial_process_view,
        },
        "stats": {
            "ticks": snapshot.stats.ticks,
            "tick_errors": snapshot.stats.tick_errors,
            "last_duration_ms": snapshot.stats.last_duration_ms,
            "avg_duration_ms": snapshot.stats.avg_duration_ms,
        },
        "warnings": list(snapshot.warnings),
    }


def _emit(payload: dict[str, Any], stream: TextIO, pretty: bool) -> None:
    """Write JSON, appending a trailing newline so shell pipelines stay tidy."""
    json.dump(payload, stream, indent=2 if pretty else None, sort_keys=False)
    stream.write("\n")


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Argument vector without the program name, defaulting to ``sys.argv[1:]``.
        stdout: Stream for JSON output, defaulting to ``sys.stdout``.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args(argv)
    out = stdout if stdout is not None else sys.stdout
    config = load_config(args.config)
    config = _apply_ui_flags(config, args)

    if args.devices:
        options = sampler_options(args, config)
        devices = discover_for_dump(options)
        _emit(
            {
                "count": len(devices),
                "devices": [
                    {
                        "index": d.index,
                        "card": d.card,
                        "render": d.render,
                        "bdf": d.bdf,
                        "name": d.name,
                        "kind": d.kind,
                        "device_id": f"0x{d.device_id:04x}",
                        "boot_vga": d.boot_vga,
                    }
                    for d in devices
                ],
                "warnings": list(config.warnings),
            },
            out,
            args.pretty,
        )
        return 0 if devices else EXIT_NO_DEVICE

    if args.dump:
        options = sampler_options(args, config)
        sampler = Sampler(options)
        devices = sampler.discover()
        if not devices:
            _emit(
                {
                    "devices": [],
                    "processes": [],
                    "warnings": [
                        *config.warnings,
                        f"no amdgpu device found under {options.drm_root}",
                    ],
                },
                out,
                args.pretty,
            )
            return EXIT_NO_DEVICE
        # Engine utilisation comes from a delta between two samples, so a dump needs a
        # real window: sample once to establish the baseline, wait one interval, then
        # sample again.  Without the wait every engine would read 0%.
        sampler.sample_once()
        time.sleep(min(options.interval_s, 2.0))
        snapshot = sampler.sample_once()
        _emit(snapshot_to_dict(snapshot), out, args.pretty)
        return 0

    from gputop.ui.app import run_app  # imported lazily: the TUI is not needed for --dump

    return run_app(config, sampler_options(args, config))


def discover_for_dump(options: SamplerOptions) -> tuple[Any, ...]:
    """Discover devices for the ``--devices`` listing without starting a thread."""
    from gputop.readers.discovery import discover_devices

    return discover_devices(
        options.drm_root,
        kind_override=options.kind_override,
        name_overrides=dict(options.name_overrides),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
