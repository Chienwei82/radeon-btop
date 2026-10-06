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
from types import MappingProxyType
from typing import Any, TextIO

from gputop.config import Config, load_config
from gputop.model.alerts import AlertLevel, evaluate, worst
from gputop.model.blocks import GpuBlocks
from gputop.model.device import DeviceKind
from gputop.model.metrics import AmdgpuMetrics
from gputop.model.process import GpuProcess
from gputop.model.snapshot import GpuSnapshot, SourceReport
from gputop.readers.radeontop import RadeontopOptions
from gputop.sampler import Sampler, SamplerOptions
from gputop.sessionlog import SessionLog, open_log
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
    blocks = parser.add_mutually_exclusive_group()
    blocks.add_argument(
        "--blocks",
        dest="blocks",
        action="store_true",
        default=None,
        help="force the per-block panel on, overriding blocks.enabled",
    )
    blocks.add_argument(
        "--no-blocks",
        dest="blocks",
        action="store_false",
        help="never start radeontop, even when it is installed",
    )
    parser.add_argument(
        "--log",
        type=Path,
        metavar="PATH",
        help="record the session to PATH (.csv or .json; add .zst to compress)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report which metrics this machine can provide, and why any are missing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit --check as JSON instead of text (ignored without --check)",
    )
    return parser


def _apply_ui_flags(config: Config, args: argparse.Namespace) -> Config:
    """Overlay the display flags onto the loaded configuration.

    The UI reads appearance from :class:`Config` rather than from the argument vector, so
    that a flag and its config-file equivalent take exactly the same path.
    """
    changed = args.theme is not None or args.no_color or args.blocks is not None
    if not changed:
        return config
    blocks = args.blocks if args.blocks is not None else config.blocks.enabled
    return replace(
        config,
        ui=replace(
            config.ui,
            theme=args.theme or config.ui.theme,
            no_color=args.no_color or config.ui.no_color,
        ),
        blocks=replace(config.blocks, enabled=blocks),
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
        # ``is not None`` rather than a truth test: 0 and a negative number are both
        # something the user typed and both deserve the same clamping the README promises,
        # where a bare truth test silently substituted the config value for ``-i 0`` and let
        # ``-i -1`` through unclamped.
        interval_s=args.interval if args.interval is not None else config.interval_s,
        history_length=config.history_length,
        collect_processes=config.process.show and not args.no_processes,
        drm_root=args.drm_root or Path("/sys/class/drm"),
        proc_root=args.proc_root or Path("/proc"),
        # The flag wins over the config; both go through the same narrowing so a
        # misspelled config value degrades to "auto" rather than reaching the sampler.
        kind_override=_kind(args.kind) or _kind(config.gpu.kind),
        name_overrides=config.gpu.names,
        only_devices=config.gpu.devices,
        radeontop=RadeontopOptions(
            enabled=config.blocks.enabled,
            binary=config.blocks.binary,
            ticks=config.blocks.ticks,
            interval_s=config.blocks.interval_s,
            drm_root=config.blocks.drm_root,
            restart=config.blocks.restart,
        ),
    )


def _memory(payload: Any) -> dict[str, Any] | None:
    """Serialise a memory pool, or ``None`` when both readings are missing."""
    if payload is None:
        return None
    return {"used": payload.used, "total": payload.total, "percent": payload.percent}


def _blocks_to_dict(blocks: GpuBlocks | None) -> dict[str, Any] | None:
    """Serialise per-block utilisation, or ``None`` when there is none."""
    if blocks is None:
        return None
    return {
        "percent": dict(blocks.by_key()),
        "busiest": blocks.busiest_key(),
        "clocks": {c.key: {"percent": c.percent, "mhz": c.mhz} for c in blocks.clocks},
        "timestamp_ns": blocks.timestamp_ns,
    }


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
                # ``speed`` is the kernel's own unit, tenths of a GT/s.  A consumer that
                # reads it as a generation -- or as GT/s -- is off by a factor, so the
                # decoded rate is carried alongside it.
                "gt_per_second": metrics.pcie.gt_per_second,
                "generation": metrics.pcie.generation,
                "describe": metrics.pcie.describe(),
            }
        ),
        "blocks": _blocks_to_dict(metrics.blocks),
        "odc": {
            "present": metrics.odc.present,
            "vddgfx_offset_mv": metrics.odc.vddgfx_offset_mv,
            "domains": [
                {
                    "name": d.name,
                    "minimum": d.minimum,
                    "maximum": d.maximum,
                    "ceiling": d.ceiling,
                    "unit": d.unit.value,
                    "percent": d.percent,
                    "entries": [[e.state, e.value, e.unit.value] for e in d.entries],
                }
                for d in metrics.odc.domains
            ],
        },
        "power_profiles": {
            "present": metrics.profiles.present,
            "active": metrics.profiles.active_name,
            "active_index": metrics.profiles.active_index,
            "available": list(metrics.profiles.names()),
        },
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


def snapshot_to_dict(snapshot: GpuSnapshot, config: Config | None = None) -> dict[str, Any]:
    """Serialise a whole snapshot.

    Args:
        snapshot: The sample to serialise.
        config: The loaded configuration, needed for the alert thresholds.  Omitting it
            drops the alert section rather than inventing thresholds, so a caller without
            a config gets no alert claim rather than a wrong one.
    """
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
        "blocks": {
            "status": str(snapshot.blocks_status),
            "hint": snapshot.blocks_hint,
        },
        "alerts": _alerts_to_dict(snapshot, config) if config is not None else None,
    }


def _alerts_to_dict(snapshot: GpuSnapshot, config: Config) -> dict[str, Any]:
    """Serialise the threshold alert state.

    Included in the dump for the same reason it is on screen: a threshold that only exists
    in the interface is one a script cannot check.  The values are the same
    :func:`~gputop.model.alerts.evaluate` produces, so ``--dump`` and the panel can never
    disagree about whether something is over its limit.
    """
    if not config.alerts.enabled:
        return {"enabled": False, "level": AlertLevel.OK.label, "breaches": []}
    thresholds = config.alerts.thresholds()
    state = worst(evaluate(metrics, thresholds) for metrics in snapshot.devices)
    return {
        "enabled": True,
        "level": state.level.label,
        "breaches": [
            {"metric": b.metric, "reading": b.reading, "limit": b.limit} for b in state.breaches
        ],
        "thresholds": {
            "temp_c": thresholds.temp_c,
            "power_percent": thresholds.power_percent,
            "vram_percent": thresholds.vram_percent,
        },
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

    if args.check:
        return _run_check(args, config, out)

    if args.devices:
        options = sampler_options(args, config)
        # A Sampler rather than a bare discovery call, because the ``gpu.devices`` filter
        # lives there: listing every card the machine has while the interface would watch
        # one of them answers a different question than the one that was asked.
        sampler = Sampler(options)
        devices = sampler.discover()
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
                "warnings": _startup_warnings(config, sampler),
            },
            out,
            args.pretty,
        )
        return 0 if devices else EXIT_NO_DEVICE

    if args.dump:
        options = sampler_options(args, config)
        sampler = Sampler(options)
        devices = sampler.discover()
        warnings = _startup_warnings(config, sampler)
        if not devices:
            _emit(
                {
                    "devices": [],
                    "processes": [],
                    "warnings": [*warnings, f"no amdgpu device found under {options.drm_root}"],
                },
                out,
                args.pretty,
            )
            return EXIT_NO_DEVICE
        try:
            # Engine utilisation comes from a delta between two samples, so a dump needs a
            # real window: sample once to establish the baseline, wait one interval, then
            # sample again.  Without the wait every engine would read 0%.
            if options.radeontop.enabled:
                # Only spawn radeontop when the panel was asked for: an unconditional wait
                # here would add seconds to every --dump on a machine that has radeontop
                # installed and does not use this feature.
                sampler.start_blocks()
                sampler.await_blocks()
            sampler.sample_once()
            time.sleep(min(sampler.options.interval_s, 2.0))
            snapshot = sampler.sample_once()
        finally:
            # Stops any radeontop children and closes their pipes.  Without this the
            # process exits with the pipes open, which is a ResourceWarning under pytest and
            # a descriptor leak anywhere that runs this in-process.
            sampler.stop()
        log = _open_session_log(args, config)
        if log is not None:
            log.write(snapshot)
            log.close()
        _emit(snapshot_to_dict(snapshot, config), out, args.pretty)
        return 0

    from gputop.config import state_path
    from gputop.ui.app import run_app  # imported lazily: the TUI is not needed for --dump

    return run_app(
        config,
        sampler_options(args, config),
        state_path=state_path(),
        log_target=args.log,
    )


def _run_check(args: argparse.Namespace, config: Config, out: TextIO) -> int:
    """Answer "what can this machine actually show me, and what is missing?".

    The report is built from a real two-sample run rather than from a list of expected
    filenames, so it describes what the interface would draw right now.  The wait between
    the samples is what per-engine utilisation is derived from, and is therefore kept as
    short as the sampler allows.

    ``--check`` never starts radeontop: the blocks panel is optional and off by default,
    and asking a diagnostic question must not start a process that wants privileges.  Its
    state is reported as it stands in the configuration instead.
    """
    from gputop.check import gather

    options = sampler_options(args, config)
    sampler = Sampler(options)
    devices = sampler.discover()
    warnings = _startup_warnings(config, sampler)
    if not devices:
        report = gather(
            _empty_snapshot(),
            drm_root=options.drm_root,
            proc_root=options.proc_root,
            config_warnings=warnings,
            blocks_enabled=config.blocks.enabled,
            blocks_binary=config.blocks.binary,
            processes_scanned=options.collect_processes,
        )
        _emit_check(report, out, as_json=args.json)
        return EXIT_NO_DEVICE

    try:
        sampler.sample_once()
        time.sleep(min(sampler.options.interval_s, 0.5))
        snapshot = sampler.sample_once()
    finally:
        sampler.stop()
    report = gather(
        snapshot,
        drm_root=options.drm_root,
        proc_root=options.proc_root,
        config_warnings=warnings,
        # None, not the pool's status: this function never starts radeontop, so anything the
        # pool says here is its unstarted default rather than an observation.  gather()
        # resolves the binary itself and reports "not exercised" for a tool it did not run.
        blocks_status=None,
        blocks_enabled=config.blocks.enabled,
        blocks_binary=config.blocks.binary,
        processes_scanned=options.collect_processes,
    )
    _emit_check(report, out, as_json=args.json)
    return 0


def _emit_check(report: Any, out: TextIO, *, as_json: bool) -> None:
    """Write the report in whichever shape was asked for."""
    if as_json:
        _emit(report.to_dict(), out, True)
    else:
        out.write(report.render())
        out.write("\n")


def _empty_snapshot() -> GpuSnapshot:
    """An empty sample, so the no-device report still has a shape."""
    return GpuSnapshot(
        sequence=0,
        timestamp_ns=0,
        devices=(),
        processes=(),
        visible_process_count=0,
        total_process_count=0,
        source=SourceReport(metrics_abi=None, per_metric=MappingProxyType({})),
    )


def _open_session_log(args: argparse.Namespace, config: Config) -> SessionLog | None:
    """Open the ``--log`` target if one was asked for.

    A target that cannot be opened is reported as a warning alongside whatever the command
    produced rather than raised: recording a session is not worth refusing to dump one.
    """
    if args.log is None:
        return None
    log = open_log(args.log, zstd_level=config.log.zstd_level)
    if log.problem is not None:
        print(f"gputop: cannot record to {args.log}: {log.problem.value}", file=sys.stderr)
        return None
    return log


def _startup_warnings(config: Config, sampler: Sampler) -> list[str]:
    """Every warning a one-shot command should carry in its output.

    The device filter's warning is a start-up condition rather than a per-sample fault,
    so it lives on the sampler instead of in the snapshot's warning list and is collected
    here, once, by each of the three commands that can report it.
    """
    warnings = list(config.warnings)
    if sampler.filter_warning:
        warnings.append(sampler.filter_warning)
    return warnings


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
