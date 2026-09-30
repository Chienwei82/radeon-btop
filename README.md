# gputop

A btop-style monitor for AMD Radeon GPUs driven by the `amdgpu` kernel driver.

Currently this repository contains the **data layer only** (see [SPEC.md](SPEC.md) and
[DESIGN.md](DESIGN.md)): device discovery, metric collection, the binary `gpu_metrics`
parser, per-process accounting and the background sampler. The TUI is not implemented yet.

## Quick start

```bash
uv sync
uv run pytest
uv run python -m gputop --dump          # one JSON snapshot of the real machine
uv run python -m gputop --devices       # list detected GPUs
uv run python -m gputop --dump-metrics  # hex-dump an unknown gpu_metrics ABI
```

## What it reads (all unprivileged)

* `/sys/class/drm/card*/device/` — VRAM/GTT, busy %, DPM clocks, power profile
* `/sys/class/drm/card*/device/gpu_metrics` — versioned binary metric table (v1.x/v2.x)
* `/sys/class/hwmon/hwmon*/` — temperatures, power, fan
* `/proc/<pid>/fdinfo/` — per-process VRAM/GTT and per-engine utilisation

The application never writes to sysfs.

## Requirements

Linux with `CONFIG_DRM_AMDGPU`, kernel 5.19+, Python 3.14 (standard or free-threaded).
