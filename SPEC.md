# gputop — Specification

**Working name:** `gputop` (package `gputop`, CLI `gputop`)
**Status:** Draft for review — no code written yet
**Reference hardware used for verification:** AMD Radeon Navi 44 (`0x73bf`, RDNA 3), `amdgpu` 6.19.4, kernel `7.0.0-34-generic`, Python 3.14.4

---

## 1. Purpose and goals

`gputop` is a btop-style terminal UI for monitoring **AMD Radeon GPUs driven by the `amdgpu`
kernel driver**. It is intended as a better replacement for `radeontop`.

### 1.1 Goals

| # | Goal | Measurable criterion |
|---|------|----------------------|
| G1 | **Comprehensive** metrics in one screen | VRAM, GTT, busy %, clocks, 3 temperatures, power vs cap, fan, perf level, power profile, throttle status, per-process engine usage — all on one screen, no scrolling for the default view |
| G2 | **Per-process visibility** | Per-PID VRAM/GTT and per-engine (gfx/compute/dma/enc/dec/VCN) utilisation, deduplicated by `drm-client-id` |
| G3 | **Unprivileged** operation | **Every core metric** works as an ordinary user; no metric *requires* root (see §7 for the one documented partial exception) |
| G4 | **Robustness** | Missing files, missing permissions, kernel version drift and truncated binaries degrade to `N/A`; the app never crashes and never hangs |
| G5 | **Multi-GPU** | Any number of `amdgpu` devices, including dGPU + iGPU, with runtime switching |
| G6 | **APU-aware** | When an integrated GPU is present, the memory panel emphasises GTT over VRAM |
| G7 | **Read-only** | The app **never writes** to sysfs. No clock/power/profile changes exist in v1 |
| G8 | **Modern Python** | Python 3.14 only, PEP 649 lazily-evaluated annotations, no `from __future__ import annotations`, `uv`-managed |

### 1.2 Non-goals

- **No NVIDIA or Intel support.** No vendor abstraction layer. A single concrete `AmdgpuDevice` class.
- **No AMDGPU/ADLX/ROCm SMI library** (`amdsmi` is not a dependency; it requires root/daemon).
- **No GPU control** (no clock locking, no power-profile switching, no fan control) in v1.
- **No cross-platform support.** Linux only.
- **No historical persistence** (no on-disk time series) in v1 — history is an in-memory ring buffer.

### 1.3 What "better than radeontop" means concretely

1. **Engine-level per-process detail.** `radeontop` reports process-level memory; `gputop` reports per-engine
   utilisation for gfx / compute / DMA / video-encode / video-decode / VCN.
2. **Many GPUs at once**, not just one.
3. **Sensible btop-style presentation**: sparkline graphs, dual memory bars, a sensor table.
4. **Clean `N/A` semantics** instead of `0` for metrics the hardware does not implement.
5. **Modern dependencies**: `uv`, Textual, `tomllib`, free-threaded-Python-safe sampling.

---

## 2. Platform and requirements

| Item | Requirement |
|------|-------------|
| OS | Linux, `CONFIG_DRM_AMDGPU=m` or built-in |
| **Minimum kernel** | **5.19** (see §5) |
| Architecture | any (`x86_64`, `aarch64`, …); no arch-specific code |
| Python | **>= 3.14**, standard build **or** free-threaded build (`3.14t`) |
| Tooling | [`uv`](https://docs.astral.sh/uv/) for environment and scripts (`uv run pytest`, `uv run ruff`) |
| UI | [`textual`](https://textual.textualize.io/) |
| Python deps | `textual` only. **No** `ctypes`/`libdrm` bindings, no `psutil`, no numpy |

### 2.1 Non-negotiable engineering constraints

- **PEP 649 / 750**: rely on Python 3.14's lazily-evaluated annotations. **Do not** use
  `from __future__ import annotations` anywhere. All annotations must be evaluable at runtime
  lazily by 3.14 semantics.
- **No `from __future__ import annotations`** — enforced by a test.
- **Free-threaded safety**: the sampler and the UI communicate only by passing **immutable,
  deeply-frozen snapshots** through a `queue.Queue`. No shared mutable state, no shared caches,
  no reference cycles that pin memory. This must hold under `3.14t` with `sys._is_gil_enabled() == False`.
- **`tomllib`** (stdlib) for configuration. `compression.zstd` (stdlib, new in 3.14) for optional
  compressed diagnostic dumps.
- **Read-only**: the code must never open any sysfs path for writing. This is enforced by a
  static test that greps the source for `open(..., "w")` / `write_text` on resolved sysfs paths.

---

## 3. Metrics specification

Every metric is optional. `None` renders as `N/A` in the UI. "Never" means the value is not
available on this hardware/kernel at all.

### 3.1 Core metrics

| Metric | Unit | Meaning | Notes |
|--------|------|---------|-------|
| `gpu_busy_percent` | % | Percent of time the GPU was busy | Not present on legacy GCN (≤ GFX8) |
| `mem_busy_percent` | % | VRAM controller utilisation | Not present on legacy GCN |
| `vram_used` / `vram_total` | bytes | Dedicated GPU memory | `mem_info_vram_*`, fallback `mem_info_vis_vram_*` |
| `gtt_used` / `gtt_total` | bytes | GPU page-table / system-RAM memory | Absent on some older kernels |
| `sclk` / `sclk_max` | MHz | Graphics clock, current / ceiling | |
| `mclk` / `mclk_max` | MHz | Memory clock, current / ceiling | |
| `temp_edge` | °C | Edge temperature | |
| `temp_junction` | °C | Hotspot / junction temperature | Second hwmon sensor when its label is `junction` or `hotspot` |
| `temp_mem` | °C | Memory temperature | **Often exposed by `gpu_metrics` even when hwmon has no `temp3_input`** |
| `power_draw` | W | Instantaneous board power | |
| `power_cap` | W | Enforced power limit | Used to draw the power bar |
| `fan_rpm` | RPM | Fan speed | `N/A` on passively-cooled cards and APUs |
| `fan_percent` | % | Fan duty, derived | RPM ÷ `fan1_max` |
| `performance_level` | str | Active DPM state | `auto`, `manual`, `balanced`, `high`, `low` |
| `power_profile` | str | Active SCPP profile | `BOOTUP_DEFAULT`, `3D_FULL_SCREEN`, `VIDEO`, `COMPUTE`, … |
| `throttle_status` | flags | Active throttling reasons | Decoded to human-readable set |
| `pcie_link` | — | Negotiated link width × speed | Optional extra |
| `voltage_soc` / `voltage_gfx` / `voltage_mem` | mV | Rail voltages | `gpu_metrics` only; often `0`/sentinel → treat as `N/A` |

### 3.2 Per-process metrics

Sourced exclusively from `/proc/<pid>/fdinfo/<fd>` of DRM/render-node file descriptors.

| Metric | Key in fdinfo | Notes |
|--------|---------------|-------|
| Engine time | `drm-engine-<name>: <ns>` | **Cumulative nanoseconds.** This kernel emits only this form — see §6.3 |
| VRAM | `drm-total-vram`, `drm-resident-vram`, `drm-shared-vram`, `drm-purgeable-vram` | |
| GTT | `drm-total-gtt`, `drm-resident-gtt`, `drm-shared-gtt`, `drm-purgeable-gtt` | |
| amdgpu extensions | `amd-requested-vram`, `amd-requested-gtt`, `amd-evicted-vram` | Verified present on `amdgpu` 6.19 |
| Identity | `drm-client-id`, `drm-pdev`, `drm-driver` | `drm-pdev` maps a client to a **specific GPU** |

Process identity (PID, name, user) comes from `/proc/<pid>/{comm,status}`.

**Deduplication rule.** Two or more file descriptors may refer to the same DRM client. All
fdinfo records sharing the same `drm-client-id` **on the same `drm-pdev`** describe one client and
must be merged into exactly one row, taking the maximum per-field, not the sum. Summing would
double-count, because both the compositor and the X server, or a Vulkan app and its compute helper
children, can hold descriptors onto the same client ID.

**Child-process note (verified).** Vulkan drivers (Mesa) fork helper processes
(`AMD family 25 [RADV]`/compute helpers) that open their *own* render-node descriptors and report
*their own* `drm-client-id` and memory. They are genuine, separately-accounted clients and are
shown as separate rows — they must **not** be merged into the parent.

### 3.3 Non-metric information

| Item | Source |
|------|--------|
| Card name / model | `device/device` id → name map, plus optional override in config |
| PCI BDF | `/sys/bus/pci/devices/<BDF>` |
| VRAM vendor | `mem_info_vram_vendor` (`samsung`, `hynix`, `micron`, `N/A` on APUs) |
| `amdgpu` driver version | `/sys/module/amdgpu/version` |
| Boot VGA | `<BDF>/boot_vga` |

---

## 4. Functional requirements

| ID | Requirement |
|----|-------------|
| FR1 | Sampling runs in a **background thread**, never blocks the UI, and enforces the configured interval (default **1000 ms**). |
| FR2 | The UI reads only the **latest** snapshot from a bounded queue (depth 1-2); stale snapshots are discarded, never queued up. |
| FR3 | Configurable interval via TOML config and CLI flag. |
| FR4 | Per-process table sorted by a user-selectable column; default = total engine usage. |
| FR5 | GPU switching without restarting the process, both by key and by a GPU picker overlay. |
| FR6 | APU awareness: iGPU detected → memory panel leads with GTT, VRAM de-emphasised. |
| FR7 | Any missing/inaccessible metric renders `N/A` and never raises. |
| FR8 | A headless/diagnostic mode prints a text report and exits (CI- and bug-report-friendly). |
| FR9 | Graphs show a rolling window of history with the configured retention. |
| FR10 | Read-only: no sysfs write path exists in the codebase. |

---

## 5. Kernel support and metric availability

### 5.1 Minimum kernel: **5.19**

Rationale for each dependency:

| Feature | Since | Why gputop needs it |
|---------|-------|--------------------|
| `mem_info_gtt_used` / `mem_info_gtt_total` | 5.19 | Primary memory metric for **APUs**, where GTT *is* the working set |
| `drm-resident-vram` / `drm-resident-gtt` in fdinfo | 5.19 | Precise per-process resident memory |

On older kernels the app still runs and still reports VRAM, temperatures, power, fan and
per-process memory — it simply shows `N/A` for GTT and for resident-memory columns. Nothing crashes.

### 5.2 `gpu_metrics` availability

The versioned binary `gpu_metrics` sysfs file is **not present on every kernel**, so it is strictly
an acceleration path, never a requirement. When absent or unparseable, the sysfs/hwmon fallback in
§6.4 is used.

### 5.3 Availability by metric

| Metric | Legacy GCN (≤GFX8) | Vega / GFX9 | Navi (RDNA1) | RDNA2 | RDNA3 | RDNA4 | Zen APU |
|--------|--------------------|-------------|--------------|-------|-------|-------|---------|
| `gpu_busy_percent` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mem_busy_percent` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VRAM used/total | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (carve-out) |
| GTT used/total | ✓ (5.19+) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (primary) |
| `current_sclk` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `current_mclk` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gpu_metrics` | ✗ | partial | ✓ (v1.x) | ✓ (v1.x) | ✓ (v1.x) | ✓ (v2.x+) | ✓ (v2.x) |
| `temp_edge` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `temp_junction` / hotspot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `temp_mem` | rarely | rarely | ✓ | ✓ | ✓ | ✓ | varies |
| `power_draw` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| `power_cap` | varies | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fan RPM | dGPU only | dGPU only | dGPU only | dGPU only | dGPU only | dGPU only | ✗ |
| `pp_power_profile_mode` | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `throttle_status` | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VCN engine in fdinfo | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**`gpu_metrics` version ↔ GPU family (verified on RDNA3 = v1.3):**

| `format_revision.content_revision` | Struct | Typical hardware |
|------------------------------------|--------|------------------|
| `1.0` | `gpu_metrics_v1_0` | Vega (GFX9), early Navi |
| `1.1` | `gpu_metrics_v1_1` | Navi 10/12/14, Vega20, Renoir |
| `1.2` | `gpu_metrics_v1_2` | adds firmware timestamp |
| `1.3` | `gpu_metrics_v1_3` | **RDNA1–RDNA3** ← verified here |
| `2.0` | `gpu_metrics_v2_0` | APU / Vega / older APU-style layout (soc/core/L3 temps) |
| `2.1` | `gpu_metrics_v2_1` | APU, reordered |
| `2.2` | `gpu_metrics_v2_2` | APU, adds independent throttle status |
| `3.x` | forward | Reserved; see §6.2 |

### 5.4 APU-specific notes

- **VRAM is a carve-out.** An APU's `mem_info_vram_total` is typically 0-512 MiB taken from system
  RAM, while `mem_info_gtt_total` is a large fraction of physical RAM. The UI must therefore lead
  with GTT on an iGPU.
- **`mem_info_vram_vendor` reads `N/A`** on APUs — a cheap iGPU hint.
- **APUs use the `format_revision == 2` metrics layout**, whose field set is APU-specific
  (`temperature_gfx`/`soc`/`core`/`l3`, `average_cpu_power`, `average_soc_power`) and which lacks
  the `uclk` field used for `mclk`. The parser maps these onto the same normalised metric names.
- **No fan.** Always `N/A`.
- Detection of an integrated GPU is specified in DESIGN.md §5.4 (multi-signal heuristic + config override).

---

## 6. `gpu_metrics` binary format

### 6.1 Layout — **verified empirically** against a live Navi 44

The file begins with a 4-byte common header:

```c
struct metrics_table_header {   /* offset 0, 4 bytes */
    __u16 structure_size;       /* total buffer size, INCLUDING this 4-byte header */
    __u8  format_revision;      /* 1 = v1.x, 2 = v2.x, 3 = v3.x            */
    __u8  content_revision;     /* minor revision within the format        */
};
```

The body follows immediately at **offset 4** and is composed of little-endian fixed-width fields.

**Verification transcript (this machine):**

```
bytes[0:4] = 78 00 01 03   ->  structure_size = 120
                              format_revision  = 1
                              content_revision = 3        =>  v1.3
file length = 120 bytes     ->  body length = 116 = structure_size - 4

  offset  4  temperature_edge           = 51     hwmon temp1_input = 51000  (51.0 C)   MATCH
  offset  6  temperature_hotspot        = 58     hwmon temp2_input = 58000  (58.0 C)   MATCH
  offset  8  temperature_mem            = 56     (no hwmon equivalent exists)           EXTRA
  offset 22  average_socket_power       = 33     hwmon power1_average = 33000000 (33 W) MATCH
  offset 58  current_uclk               = 1000   pp_dpm_mclk current state = 1000 MHz    MATCH
  offset 72  current_fan_speed          = 0      hwmon fan1_input = 0                   MATCH
  offset 74  pcie_link_width            = 16     PCIe Gen5 x16                           MATCH
  offset 76  pcie_link_speed            = 80     encoded as 16 * 5                       MATCH
```

**Units confirmed:** temperatures are whole **degrees Celsius** (`u16`); power is whole **watts**
(`u16`); clocks are whole **MHz** (`u16`).

This yields a key finding that shapes the design: on this card `gpu_metrics` reports **four**
temperatures while hwmon exposes only **two**. `gpu_metrics` is therefore the *preferred* source
for temperature and clock data, and hwmon is the fallback — not the other way round.

### 6.2 Sentinel values

Unsupported fields are **not** zero. Observed sentinels in the live file:

| Sentinel | Fields showing it | Interpretation |
|----------|-------------------|----------------|
| `0xFFFF` (65535) | `average_socclk_frequency`, `temperature_hbm` | Not supported by this ASIC |
| `0xFFFFFFFF` | `gfx_activity_acc`, `mem_activity_acc` | Accumulator not populated |
| `0xFFFFFFFFFFFFFFFF` | `firmware_timestamp` | Not populated |

A parser that renders these as real values will display "65535 MHz" and "65.535 °C".
**Every field must therefore be sentinel-checked and mapped to `N/A`.**

Additionally, some populated fields are semantically invalid on a given ASIC — e.g.
`voltage_gfx` decoded as `6` mV on an RDNA 3 card that has no discrete GFX voltage rail in that
field. Values below a plausible per-field floor are also treated as `N/A` (see DESIGN.md §6.4).

### 6.3 Forward compatibility

- `structure_size` bounds the body: parse `min(len(body), structure_size - 4)`.
- If the driver reports a **larger** struct than this build knows, truncate and use what is known.
- If the driver reports a **smaller** struct, decode what is present and mark the rest `N/A`.
- If `(format_revision, content_revision)` is **unknown** — including any `3.x` — do **not** guess.
  Fall back to sysfs/hwmon and surface `metrics ABI v{fmt}.{rev} unrecognised` in the footer.
  Adding a version is a one-line registry entry plus one `struct.Struct` (DESIGN.md §6.2).

> **Open point.** `format_revision == 3` could not be verified on the available hardware, and
> upstream has an in-flight series changing `gpu_metrics` encoding. The design deliberately treats
> unknown versions as unsupported rather than mis-parsing them. See §9 Q1.

### 6.4 Source-precedence order

Chosen from the evidence above, highest priority first:

1. **`gpu_metrics`** — richest (extra temps, instantaneous power, throttle flags, sockets).
2. **Device sysfs** — VRAM/GTT totals and used, `gpu_busy_percent`, `mem_busy_percent`, link.
3. **hwmon** — temperatures, power average/cap, fan, pwm.
4. **DPM tables** — clocks (`current_sclk`/`current_mclk`, else the `*`-marked `pp_dpm_*` row).
5. **`/proc/*/fdinfo`** — per-process.

---

## 7. Privilege requirements

| Capability | Needs root? | Evidence |
|------------|-------------|----------|
| All device metrics | **No** | `gpu_metrics`, `gpu_busy_percent`, `mem_busy_percent`, `mem_info_*` are mode `0444` root-owned → world-readable |
| All hwmon metrics | **No** | `temp*_input`, `power1_average`, `power1_cap`, `fan1_input` are `0444` |
| Clocks, DPM, power profile | **No** | `pp_dpm_*`, `power_dpm_force_performance_level`, `pp_power_profile_mode` are `0444` |
| Own-user processes | **No** | Same-UID `/proc/<pid>/fdinfo` is readable |
| **Other users' processes** | **Yes** | Verified: reading `/proc/1/fdinfo/0` as uid 1000 returns `Permission denied` |

**Documented partial exception.** The per-process table can only enumerate processes the calling
user may inspect. Without root this is the caller's own UID, so **GPU work by root daemons or other
users is invisible**. The UI states this explicitly rather than silently showing a partial list.
Per-process visibility is therefore *not* covered by the "every core metric works without root"
guarantee; all *device* metrics are.

The render node itself is `root:render` mode `0660`, so a user outside the `render`/`video` groups
cannot open `/dev/dri/renderD*`. This does not block `gputop`: reading `fdinfo` does not require
opening the node, only read access to `/proc`. `gputop` never opens the DRM node.

---

## 8. User interface

### 8.1 Required panels

1. **Header** — GPU identity, model, PCI BDF, driver version, active GPU index.
2. **Utilisation** — `gpu_busy_percent` and `mem_busy_percent` sparkline graphs.
3. **Memory** — VRAM used/total bar and GTT used/total bar; GTT leads when the device is an iGPU.
4. **Clocks** — sclk and mclk, current vs maximum, as bars.
5. **Sensors** — edge / junction / mem temperatures, power vs cap bar, fan RPM, plus perf level,
   power profile and throttle status.
6. **Process table** — per-process engine usage and memory.
7. **Footer** — sampling interval, resolved source per metric family, key hints, warnings.

### 8.2 Keybindings

| Key | Action |
|-----|--------|
| `q`, `Ctrl-C` | Quit |
| `g`, `Tab`, `→`, `]` | Next GPU |
| `G`, `Shift-Tab`, `←`, `[` | Previous GPU |
| `p` | GPU picker overlay |
| `m` | Toggle memory emphasis (GTT-first ↔ VRAM-first) |
| `s` | Cycle sort column (process table) |
| `f` | Toggle filter: all → GPU-heavy → VRAM-heavy |
| `r` | Force a rescan for new GPUs / devices |
| `+` / `-` | Increase / decrease sampling interval |
| `i` | Interval picker overlay |
| `P` | Toggle process table |
| `h`, `?` | Help overlay |
| `1`…`9` | Jump to GPU *n* |
| `c` | Cycle colour theme |
| `d` | Dump a diagnostic snapshot (optionally `.zst`-compressed) |

---

## 9. Configuration

TOML, parsed with `tomllib`. Precedence: CLI flag → environment (`GPUTOP_CONFIG`) → user config →
packaged defaults. Hot-reloadable (re-read on `SIGHUP` and on `r`).

```toml
# gputop.toml

[general]
interval_ms = 1000        # 100–10000
history_points = 180      # sparkline window
theme = "dark"            # dark | light | amdr | btop

[gpu]
# empty => auto-detect every amdgpu device
devices = ["0000:0c:00.0"]
# force classification when auto-detection is wrong
#   "auto" (default) | "igpu" | "dgpu"
kind = "auto"
default_index = 0
# override the marketing name shown in the header
# [gpu.names]
# "0000:0c:00.0" = "Navi 44 XTX"

[process]
show = true
max_rows = 20
sort = "engine"           # engine | vram | gtt | pid | name | user
# hide processes whose total GPU usage stays below this percent
min_usage_percent = 0.0
# hide kernel/system threads from the list
hide_kernel_threads = true
# include the root/session-wide view (requires sufficient privileges to see other users)
include_all_users = false

[display]
units = "auto"            # auto | metric | imperial
temperature_unit = "c"    # c | f
bars = true
graphs = true
unicode = true
color = true

[diagnostics]
# snapshot format: json | toml | zstd-json
dump_format = "zstd-json"
dump_dir = "~/.local/state/gputop"
log_level = "info"
```

---

## 10. Performance targets

| Target | Requirement |
|--------|-------------|
| Sampler thread | Must not hold the GIL for long stretches; must not sleep longer than the interval |
| `/proc` scan cost | Enumerate PID dirs once per tick, then `readlink` each fd; stay well under 20 ms for ~800 processes on a typical machine |
| UI redraw | Textual default refresh, decoupled from the sampler; no redraw work on the sampler thread |
| Memory | Fixed-size ring buffers; no unbounded growth over multi-day runs |
| Startup | First rendered frame < 500 ms |

---

## 11. Acceptance criteria

1. `uv run pytest` green, including fixtures for `gpu_metrics` **v1.0, v1.1, v1.2, v1.3, v2.0,
   v2.1, v2.2**, a truncated buffer, an unknown `3.x` header, and the **real 120-byte Navi 44 blob**
   captured from the reference machine.
2. All sentinel values (`0xFFFF`, `0xFFFFFFFF`, `0xFFFF_FFFF_FFFF_FFFF`) decode to `None`.
3. The real blob decodes to `edge=51 °C`, `hotspot=58 °C`, `socket_power=33 W`, `uclk=1000 MHz`,
   `pcie_link_width=16`, matching the sysfs/hwmon values observed simultaneously.
4. Removing any file from the fake sysfs tree, or chmod-ing it `0000`, produces `N/A` and no
   exception.
5. Two records with the same `drm-client-id` + `drm-pdev` collapse to one row; two records with
   different `drm-client-id` do not.
6. `tests/` pass under both `python3.14` and `python3.14t` (`uv run --python 3.14t pytest`).
7. A static test asserts no sysfs write path exists.
8. `uv run ruff check` clean.
9. `--dump` reproduces every value shown in the TUI, proving there is no hidden UI-only state.

---

## 12. Questions for the reviewer

1. **`gpu_metrics` `format_revision == 3`.** I can verify v1.3 on hardware here and v1.0–v2.2 from a
   reference decoder, but I have **no v3.x sample and no upstream-published struct** for it, and
   upstream has an active series rewriting the metrics encoding. Should I (a) ship v3.x as
   *unrecognised → fall back to sysfs*, or (b) block v1 until a v3 sample is available?
2. **`/dev/dri/renderD*` access.** I have assumed `gputop` never opens the DRM node (it is
   `0660 root:render`). Confirm that is acceptable, or do you want an optional "use my own client"
   mode that requires `render` group membership?
3. **Cross-user process table.** Should non-root gputop *default* to showing only the current user
   (with a clear banner), or should the process panel be hidden entirely without root?
4. **Naming.** `gputop` is already a well-known PyPI project (an unrelated NVIDIA-style tool). The
   code distribution name can stay `gputop`, but should I pick a distinct distribution name now
   (e.g. `radon-tui`) while keeping `gputop` as the CLI?
5. **Config location.** Default to XDG (`~/.config/gputop/gputop.toml`) with a legacy fallback to
   `~/.gputoprc`? And should a packaged `gputop.toml` ship as a default, or should defaults be
   dataclass values with the TOML purely optional?
