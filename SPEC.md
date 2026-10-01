# gputop — Specification

**Working name:** `gputop` (package `gputop`, CLI `gputop`)
**Status:** Draft for review — no code written yet
**Reference hardware used for verification:** AMD Radeon Navi 21 / RX 6800 (`0x73bf`, RDNA2; Sapphire subsystem `148c:2407`, 16 GiB GDDR6), `amdgpu` 6.19.4, kernel `7.0.0-34-generic`, Python 3.14.4

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
- **No GPU control** (no clock locking, no power-profile switching, no fan control) in v1. The
  power-profile and overdrive attributes are read, never written — see §3.4.
- **No cross-platform support.** Linux only.
- **No implicit persistence.** History is an in-memory ring buffer and is gone when gputop exits.
  The only on-disk time series is the one the user names explicitly, with `--log PATH` (§9).
  Apart from the `[state]` section gputop writes back into its own config file (§9), nothing is
  ever written unless the user asked for it by path.

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
| Optional external binary | `radeontop` on `$PATH` (or an absolute path in `blocks.binary`), used **only** by the optional per-block panel. Absent ⇒ that panel hides itself and states why; every other panel is unaffected (§3.4, §7). Nothing else is added to the process tree. |

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
  static test that greps the source for `open(..., "w")` / `write_text` on resolved sysfs paths,
  and — for the two SCPP attributes where a write would reprogram hardware rather than merely
  reconfigure a knob — by a test that scans `readers/pp.py` itself for `write_text`,
  `write_bytes`, `open("w")`, `os.open` and `truncate`, so a later edit that adds a write path
  fails the suite instead of quietly breaking the one rule the rest of the application rests on.

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
| `power_profile_table` | — | Every profile the driver offers, with the active one flagged | The whole `pp_power_profile_mode`, not just the `*`-marked name: the active name answers "what is it now", the table answers "what could it be" |
| `odv_<domain>` | mV **or** MHz | Per-DPM-state overdrive ceiling for one clock domain | `pp_od_clk_voltage`. The unit is carried per entry, never assumed — see §3.4 |
| `vddgfx_offset` | mV | Signed graphics-rail voltage offset | `OD_VDDGFX_OFFSET`, when the driver reports one at all |
| `blocks.<key>` | % | Utilisation of one hardware block | Optional and off by default; sourced from a `radeontop` child, not from sysfs — see §3.4 |
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

### 3.4 Optional metrics: per-block utilisation and overdrive

Two metric families are optional in a stronger sense than the rest of §3.1 — not "absent on this
kernel" but "absent unless the user asks" — and both degrade the same way: the panel is hidden and
the reason is stated rather than drawn as an empty box.

**Per-block utilisation.** The amdgpu driver exposes its GRBM and SRBM busy counters only through
the radeon ioctl on a handle that needs elevated privileges, so gputop — whose whole design is
that it never requires root — cannot take that dependency for the whole application. The counters
are read the one way that does not make *gputop* privileged: by running `radeontop` as a child
process in dump mode and parsing its output (§7). The blocks are Graphics pipe, Event Engine,
Vertex Grouper, Texture Addr, Texture Cache, Shader Export, Seq Inst Cache, Shader Interp, Shader
MemExch, Scan Converter, Prim Assembly, Depth Block, Clip Rect, Color Block, UVD and VCE, plus the
memory and shader clocks, which carry a percentage *and* an absolute frequency. The names are
radeontop's own, abbreviated only enough to fit a label column, and the underlying dump keys travel
with every reading. A block the card has no unit for is omitted by the driver and simply absent
here — never rendered as zero, because a zero utilisation and a block that does not exist are
different facts.

**`pp_od_clk_voltage` is two formats behind one filename.** On older parts it is a voltage table —
`0:300 1:350 2:400`, bare millivolts, no sections. On RDNA2 and later the driver repurposed the
same attribute into an overdrive *clock* table and gave it sections, `OD_SCLK` / `OD_MCLK` /
`OD_RANGE` / `OD_VDDGFX_OFFSET`, with MHz units:

```
OD_SCLK:
0: 500Mhz
1: 2104Mhz
OD_RANGE:
SCLK:     500Mhz       2600Mhz
OD_VDDGFX_OFFSET:
-50mV
```

The unit is therefore carried **per entry** rather than assumed. Presenting a 2104 MHz ceiling as
a voltage would be wrong on every modern card, and a parser that guessed would be confidently
wrong rather than absent — which is worse than an `N/A`. Both eras decode through one entry point,
and a file with no `OD_*` header at all is read as the single legacy voltage table it is. The
per-domain bar is a ratio of the driver's own stated `OD_RANGE`, not of a hard-coded maximum, so it
stays correct on a part whose ceiling is not the one gputop was written against.

Neither attribute is written. `pp_power_profile_mode` and `pp_od_clk_voltage` are writable sysfs
files whose contents change hardware; a monitor that quietly reprograms somebody's GPU is not a
monitor. gputop opens them for reading and nothing else, and the rule is enforced by tests rather
than by convention (§2.1).

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
| FR11 | An **optional** per-block utilisation panel, off by default, sourced from a supervised `radeontop` child. It hides itself and reports the reason rather than rendering empty bars (§8.1). |
| FR12 | **Optional threshold alerts** for temperature, power and VRAM; a crossed threshold flashes the panel border and names the breach. A missing reading never alerts, and every device is judged, not only the focused one (§8.1). |
| FR13 | **Optional session recording** to CSV or JSON Lines via `--log PATH`. A target that cannot be opened disables recording with a warning and never stops the monitor (§9). |

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

**Two optional metrics are not generational claims.** `pp_od_clk_voltage` is present on
SCPP-capable parts but in two different formats (SPEC §3.4), and presence varies inside a family,
so it is an ordinary `N/A`-on-missing metric rather than a per-generation row. Per-block
utilisation is not a sysfs attribute at all and does not belong in this table: it is available or
not according to whether `radeontop` can read the card, which is a property of the *installation*,
not of the silicon.

**`gpu_metrics` version ↔ GPU family (v1.3 observed on RDNA2):**

| `format_revision.content_revision` | Struct | Typical hardware |
|------------------------------------|--------|------------------|
| `1.0` | `gpu_metrics_v1_0` | Vega (GFX9), early Navi |
| `1.1` | `gpu_metrics_v1_1` | Navi 10/12/14, Vega20, Renoir |
| `1.2` | `gpu_metrics_v1_2` | adds firmware timestamp |
| `1.3` | `gpu_metrics_v1_3` | **RDNA1–RDNA3** ← observed on RDNA2 here |
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

### 6.1 Layout — **verified empirically** against a live Navi 21 / RX 6800

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

Additionally, some *populated* fields are semantically invalid on a given ASIC: a field can
be present in the struct and still hold a physically meaningless value. Values below a
plausible per-field floor are therefore treated as `N/A` (see DESIGN.md §6.4).

> **Correction.** An earlier draft of this section cited `voltage_gfx` decoding as `6` mV on a
> card "with no discrete GFX voltage rail". That was not a hardware property — it was our own
> misparse, caused by a two-byte offset error in the voltage block. The same card reports a
> perfectly normal `768` mV on that rail. The floor check is kept as cheap insurance against
> genuinely absent rails, but this example is withdrawn.

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
| Overdrive table | **No** | `pp_od_clk_voltage` is `0444` and is only ever opened for reading; a write would reprogram clocks and voltage (§3.4) |
| Per-block utilisation | **No** — for gputop | Read through a `radeontop` child, which needs whatever *radeontop* needs. gputop itself takes no privilege and opens no DRM node (§3.4) |
| Own-user processes | **No** | Same-UID `/proc/<pid>/fdinfo` is readable |
| **Other users' processes** | **Yes** | Verified: reading `/proc/1/fdinfo/0` as uid 1000 returns `Permission denied` |

**Documented partial exception.** The per-process table can only enumerate processes the calling
user may inspect. Without root this is the caller's own UID, so **GPU work by root daemons or other
users is invisible**. The UI states this explicitly rather than silently showing a partial list.
Per-process visibility is therefore *not* covered by the "every core metric works without root"
guarantee; all *device* metrics are.

The render node itself is `root:render` mode `0660`, so a user outside the `render`/`video` groups
cannot open `/dev/dri/renderD*`. This does not block `gputop`: reading `fdinfo` does not require
opening the node, only read access to `/proc`. `gputop` never opens the DRM node. *(With
`--blocks` it passes the card's node path to `radeontop`, which does open it — but that is the
child's privilege, not gputop's; gputop's own code only `stat`s the path. The default session is
exactly as unprivileged as before.)*

**The optional per-block panel does not weaken any of this.** gputop never requires root, and it
does not require root with `--blocks` either. The amdgpu driver exposes the GRBM/SRBM busy
counters only through the radeon ioctl on a privileged handle, so those counters — and only those
— are obtained by running `radeontop` as a child process. radeontop drops to the invoking user
during its own startup, so running gputop as root makes the child root too and buys nothing
gputop did not already have. Where a user cannot read the registers, radeontop says so — usually
"are you root?" — and gputop reports that as `NOT_PERMITTED`, hides the panel, and carries on
unprivileged with every other panel working.

This is also why the panel is **off by default**: a default-on panel would put a permanent
complaint in the status line of every user who cannot use it, and opt-in is what stops gputop from
launching a process that reaches for the GPU the moment it starts, which is not something a
monitor should do unasked (§8.1).

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

**Two further panels are optional**, in the strong sense that their absence is not a defect and is
never drawn as one. Each appears only when it has something truthful to put in it, and each labels
itself as read-only where that is the fact.

8. **Power profile** *(optional)* — the active SCPP profile and its index, the profile names the
   driver offers, and the overdrive table per clock domain (`sclk`, `mclk`, `socclk`) as a bar
   plus the `vddgfx` offset. Every value in it is one the driver published and that gputop read;
   gputop never writes these files, and a monitor that could change the setting from a keystroke
   would be a different and much more dangerous program (§3.4). Attributes the card does not
   report render `N/A` rather than disappearing, so the panel's presence never implies a
   capability the card lacks.

   It shares a row with the Blocks panel but is **decided independently of it**, because it is
   read from sysfs and works on any card that publishes SCPP. Gating it on the Blocks panel
   meant that on every machine without a privileged `radeontop` — which is every machine
   running gputop unprivileged — a user who had every right to see the power profile was
   silently shown nothing. Both panels share one height threshold, set by what the sensor
   panel can afford to lose rather than by what either optional panel needs: see §8.1.1.
9. **Blocks** *(optional)* — per-block utilisation from a supervised `radeontop` child (§3.4):
   Graphics pipe, Event Engine, Vertex Grouper, Texture Addr, Texture Cache, Shader Export, Seq
   Inst Cache, Shader Interp, Shader MemExch, Scan Converter, Prim Assembly, Depth Block, Clip
   Rect, Color Block, UVD, VCE, plus the memory and shader clocks. Blocks are grouped in pipeline
   order and the title names the busiest, because "which block is busy" is the question this panel
   exists to answer and the bars are the evidence; two adjacent busy blocks mean a stall between
   them. radeontop's own spelling is kept rather than renamed, because a user comparing the two
   tools side by side should not have to learn that the same block is written differently in each.

   It is **off by default** (`blocks.enabled = false`; `--blocks` turns it on) for the reason
   given in §7, and it is **hidden rather than empty** whenever it has no data. There are five
   states, because a missing binary, an unsupported card, a refused ioctl and a crash are four
   different problems with four different remedies — telling a user to install a package when the
   real problem is a missing capability bit wastes their time and teaches them to ignore the
   message:

   | State | Meaning | Remedy |
   |-------|---------|--------|
   | `DISABLED` | off in the configuration | `--blocks` |
   | `MISSING` | the binary is not on `$PATH` or not executable | install it, or set `blocks.binary` |
   | `UNSUPPORTED` | radeontop ran and cannot read this card | none — the driver does not expose it |
   | `NOT_PERMITTED` | radeontop ran and was refused the GPU registers | run radeontop with the privilege it needs |
   | `FAILED` | the process died or produced nothing parseable | radeontop's own stderr, quoted |

   The reason goes to the status line and to the help overlay. The status line is **silent** for
   `OK` and for `DISABLED` — a user who turned it off chose that, and a message every second
   would train them to ignore the part of the line that matters. The other four are all cases
   where the user asked for data and is not getting it, which is why they speak.

**Alert border.** Crossing any configured threshold flashes every panel border between the alert
colour and the track colour, at `alerts.flash_hz` full cycles per second. The thresholds are
`[alerts] temp_c`, `power_percent` and `vram_percent`; a threshold of `0` disables that check and
means "never tell me about this", not "tell me the reading is above zero", and a **missing
reading never alerts** — a card whose junction sensor is not exposed must not flash forever
because it cannot prove it is cool. The **hottest** sensor wins, not the average or the first: an
average over edge, junction and memory hides precisely the case that matters, a hotspot well above
its neighbours while the mean sits below the threshold. State is recomputed from scratch every
sample, so a condition that clears stops being reported the moment it clears and nothing can
latch. And **every** device is judged, not only the focused one: a card the user is not looking at
can be the one that is overheating, and a border that watched only the focused card would report
the machine as fine while it burns.

There are exactly two levels, `OK` and `ALERT`. A `WARN` level between them was considered and
deliberately **not** added: it needs a second threshold per metric, and therefore a second knob
for a user to get wrong, and an unreachable state carrying a docstring that explains what it would
have meant is worse than its absence. If a warning band is ever wanted it should arrive as two
thresholds per metric, not as a level no code sets.

### 8.1.1 When the optional row appears

Both optional panels share one row, and that row is gated by a single measured height rather than
by what either panel needs on its own. The reason is that the cost is not the row's height but
what it takes from the **sensor panel** beside it: that panel scrolls, so a row added below the
main one does not lose its content, it pushes the trailing lines — PCIe link, throttle reasons —
out of view. The primary readings, the clocks, temperatures, power and cap, must survive.

Measured against the reference card, the sensor panel is complete at 38 rows and the row costs 8
more, so **46 rows** is the threshold. Below it neither optional panel is drawn, and the help
overlay says why. A test asserts the invariant at and above the threshold, so raising the row
further cannot silently cost the sensor panel something.

This is the same shape as the other responsive rules in this app — the memory graphs yield below
28, the process table below 20 — and for the same reason: an optional panel must never be the
reason a required one needs scrolling.

### 8.2 Keybindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Tab` / `Shift-Tab` | Next / previous GPU (overrides focus traversal) |
| `1`…`9` | Jump to GPU *n* |
| `o` | Overview: every GPU at once, compact |
| click a device tab | Switch GPU |
| click a panel | Move focus to it |
| `p` | Toggle the process table |
| `s` / `S` | Next / previous sort column (process table) |
| `r` | Reverse the sort |
| `/` | Filter the process table; `Esc` clears it |
| `a` | Show or hide idle processes |
| `t` | Process tree on / off |
| `c` | Full command line instead of the `comm` name |
| `↑` / `↓`, click a row | Move the selection |
| `k` | Send SIGTERM or SIGKILL to the selection — **disabled unless `process.allow_kill`**, and always behind a confirmation naming the target |
| `R` | Sample now |
| `+` / `-` | Increase / decrease sampling interval |
| `m` | Cycle colour theme |
| `h`, `?` | Help overlay |

Sortable columns: `pid`, `user`, `command`, `gfx`, `compute`, `dma`, `encdc`, `vram`,
`gtt`, and `total` (the sum across engines, the default — sortable but not displayed, since
it is the sum of the four engine columns). Clicking a heading sorts by that column;
clicking the active one reverses it.

The process table **hides clients that are doing nothing measurable** by default: a
client that merely holds a descriptor answers none of the questions the table exists for.
`a` shows them. The client using the most GPU is marked with `▶` and named in the panel
heading.

---

## 9. Configuration

TOML, parsed with `tomllib`. Precedence: CLI flag → saved session state → user config →
environment (`GPUTOP_CONFIG`) → packaged defaults.

gputop also **writes** the file, adding a `[state]` section to `~/.config/gputop/config.toml`
that records the theme, sort column and direction, filter text, interval, tree/command/idle
toggles, overview mode and focused GPU from the last session. It is the only section written;
every other key in the file is left exactly as the user wrote it, and the write is atomic
(a temporary file renamed over the old one) so a crash cannot truncate the config. Reading
it back uses `tomllib`; writing uses `tomli-w`, since the standard library has no writer.

Because `[state]` is a *session* rather than a *preference*, it wins over the hand-written
values — which is what "how did I have this last time" means. Deleting the section restores
the file's own values. A saved theme or sort column from another release is ignored with a
warning rather than being allowed to stop gputop starting. An explicit `--config` bypasses
the state file entirely, so a scripted run never depends on the last interactive session.
`$GPUTOP_STATE` overrides where the state is read and written.

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
# "0000:0c:00.0" = "AMD Radeon RX 6800"

[process]
show = true
max_rows = 20
# total | pid | user | command | gfx | compute | dma | encdc | vram | gtt
sort = "total"
# hide processes whose total GPU usage stays below this percent
min_usage_percent = 0.0
# hide kernel/system threads from the list
hide_kernel_threads = true
# include the root/session-wide view (requires sufficient privileges to see other users)
include_all_users = false
# nest the table by PPid
process_tree = false
# show the full /proc/<pid>/cmdline instead of the 15-character comm name
full_command = false
# show clients that hold a descriptor but use no measurable GPU time
show_idle = false
# allow "k" to signal the selected process.  Off by default: terminating another process
# from a monitor is one keystroke away from terminating the wrong one.
allow_kill = false

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

[blocks]
# Off by default.  radeontop needs a privilege most users do not have, so a default-on panel
# would put a permanent complaint in the status line of everyone who cannot use it -- and
# opt-in is what stops gputop spawning a process that reaches for the GPU unasked.
# --blocks / --no-blocks override this either way.
enabled = false
binary = "radeontop"       # program name looked up on $PATH, or an absolute path
ticks = 120                # samples per second asked of radeontop
interval_s = 1             # seconds between dumps.  Whole seconds only: radeontop parses -i
                            # with atoi and floors it at 1, and gputop clamps to match rather
                            # than silently sampling slower than asked.
drm_root = "/dev/dri"      # so -p (DRM node) can be preferred over an ambiguous -b (PCI bus)
restart = true             # bounded retries when a child dies mid-session

[alerts]
enabled = true
temp_c = 90.0              # hottest sensor in Celsius
power_percent = 95.0       # draw as a percentage of the enforced cap
vram_percent = 92.0        # VRAM in use as a percentage of the pool
flash_hz = 1.0             # full border flash cycles per second; 0 holds a steady border
# A threshold of 0 disables that check -- "never tell me about this", not "above zero".
# A missing reading never alerts, and every device is judged, not only the focused one.

[log]
# Only --log PATH turns recording on.  This section says how, so a wrapper script that records
# every session need not repeat the compression level.
zstd_level = 3             # 1-19, or 0 for the library default
interval_s = 0.0           # seconds between records; 0 records every sample
```

Recording is opt-in through `--log PATH`, and the format comes from the suffix: `.csv`, or
`.json`/`.jsonl`/`.ndjson`, with an optional trailing `.zst`/`.zstd` meaning zstd compression
through the standard library's `compression.zstd`. The base name chooses the format and the suffix
chooses the transport, so the two are independent and neither has to be declared twice:
`--log session.csv.zst` is a compressed CSV.

Three decisions are worth stating, because the obvious alternatives are worse:

- **JSON output is JSON *Lines*, not a JSON array** — one object per line, per device. A session
  that is interrupted still leaves a valid, readable file, which an array does not: the cost of a
  crash becomes the last line rather than the whole recording.
- **Files are opened for append, and the CSV header is seeded from whether the file already had
  content.** Recording twice to one path is an ordinary thing to do, and a second header lands in
  the middle of the data, where every strict CSV reader absorbs it as a record. A file that exists
  but is empty still gets a header, because there is nothing to conflict with.
- **An absent reading is an empty cell** — never `0`, never `N/A`. Empty is what CSV readers treat
  as "no value", whereas a literal `N/A` in a numeric column is a parse error the consumer has to
  special-case.

Records are flushed per record and the stream is closed on every exit path, because a `.zst`
stream left unclosed is a truncated archive rather than a readable file — the one outcome worse
than not having recorded at all. A target that cannot be opened, for whatever reason, is reported
as a warning and disables recording: `--log /root/nope.csv` degrades to "recording disabled, and
here is why", never to a monitor that refuses to start. `log.zstd_level` is what the target is
opened with. `log.interval_s` thins the stream: `0` records every sample, and any positive value
records at most one row per that many seconds. The default is `0` because a thinned recording is
a recording with gaps in it, and choosing to lose samples is the user's call rather than gputop's.

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
   v2.1, v2.2**, a truncated buffer, an unknown `3.x` header, and the **real 120-byte Navi 21 blob**
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
10. With no `radeontop` installed, with `radeontop` present but unprivileged, and with a
    card radeontop cannot read, gputop starts, every other panel works, the blocks panel is
    absent rather than empty, and the status line names the state that applies.
11. A dump line from a card that omits `tc`/`smx`/`cr`/`uvd` parses with those blocks missing —
    never with another block's value shifted into their place.
12. A `pp_od_clk_voltage` file in each of its two eras parses, and a modern card's 2104 MHz
    ceiling is never displayed as a voltage.
13. A `--log` session interrupted mid-write leaves a readable file: CSV with exactly one header,
    JSON with one complete object per line and no truncated tail beyond the last flush.
14. Crossing and then clearing a threshold flashes the border and then stops, with nothing left
    over from the alarmed state.

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
