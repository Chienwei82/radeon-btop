# gputop — Design Document

**Companion to:** [SPEC.md](SPEC.md)
**Status:** Draft for review — no implementation code written yet

All kernel-interface claims in this document were verified on live hardware (Navi 21 / RX 6800, RDNA2,
`amdgpu` 6.19.4, kernel `7.0.0-34-generic`). Claims that could **not** be verified are marked
`UNVERIFIED` and are treated as risks, not facts.

---

## 1. Architecture

### 1.1 Component diagram

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                    main thread                          │
                          │                                                          │
                          │   ┌────────────────────────────────────────────────┐     │
                          │   │              Textual App (gputop.ui.app)         │     │
                          │   │                                                │     │
                          │   │   Header   Graphs   Memory   Sensors   Table    │     │
                          │   │      ┌───────────────────────────────────┐        │     │
                          │   │      │  on_mount → SamplerControl        │        │     │
                          │   │      │  on_interval → _pump()           │        │     │
                          │   │      └────────────────┬──────────────────┘        │     │
                          │   └───────────────────────┼──────────────────────────┘     │
                          └────────────────────────────────┼────────────────────────┘
                                                           │ queue.SimpleQueue
                                                           │ (get_nowait / put_nowait)
                    immutable GpuSnapshot                  ▼
                    (frozen dataclasses)        ┌────────────────────────────┐
                                                │   Sampler thread           │
                                                │   (only mutable owner)     │
                                                │                            │
                                                │   ┌────────────────────┐   │
                                                │   │ prev_engine_ns map │   │ ← sampler-private
                                                │   │ prev_fdinfo cache  │   │   delta state
                                                │   └────────────────────┘   │
                                                └─────────────┬──────────────┘
                                                              │ read-only sysfs/procfs
     ┌────────────────────────────────────────────────────────┼───────────────────────┐
     │                                                        │                       │
     ▼                                                        ▼                       ▼
┌──────────────┐                                      ┌─────────────────┐      ┌────────────────┐
│ gpu_metrics  │                                      │ hwmon + device  │      │ /proc/*/fdinfo │
│ (binary,     │──── AmdgpuBinaryMetrics ──┐           │ sysfs readers   │      │ scanner       │
│  struct)     │                            │           └────────┬────────┘      └───────┬────────┘
└──────────────┘                            │                    │                       │
                              ┌─────────────┴────────────────────┴─────────┐             │
                              │            readers.py (merge)              │             │
                              │  precedence: gpu_metrics ▸ device ▸ hwmon   │             │
                              └──────────────────────┬───────────────────────┘             │
                                                     │                                     │
                                                     ▼                                     ▼
                                           ┌──────────────────┐                    ┌──────────────────┐
                                           │  AmdgpuMetrics   │                    │  GpuProcess list │
                                           │  (per device)    │                    │  (dedup by       │
                                           └────────┬─────────┘                    │   client_id)    │
                                                    │                              └────────┬─────────┘
                                                    └──────────────┬───────────────────────────┘
                                                                   ▼
                                                        ┌──────────────────┐
                                                        │   GpuSnapshot    │
                                                        │  (frozen, total) │
                                                        └──────────────────┘
```

### 1.2 Threading model

There are exactly **two** execution contexts on the snapshot path — the UI's event loop and the
sampler thread — and **zero** shared mutable objects *between them*. The optional radeontop
supervision described below adds per-child drain threads and the project's only lock; both are
confined to that one feature and neither participates in the snapshot handoff.

| Context | Owns | Touches |
|---------|------|---------|
| Main thread (Textual event loop) | Textual widget state | Only the queue (`get_nowait`) and its own copies |
| Sampler thread | Previous-sample state for delta math | Only the queue (`put_nowait`) and sysfs/procfs |

The **only** channel between them is a `queue.SimpleQueue[GpuSnapshot]`. A snapshot is a tree of
`@dataclass(frozen=True, slots=True)` objects holding only scalars, tuples, `frozenset`, `str`,
`int`, `float`, `Path`, and nested frozen dataclasses — nothing is mutated after construction.

**Back-pressure:** the queue is drained with `get_nowait()` and the *newest* snapshot wins; older
ones are discarded. The sampler never blocks and never builds a backlog, so a slow UI degrades into
staleness rather than memory growth or latency.

**The optional per-block panel adds two threads per GPU, and one lock.** The radeontop supervision
in §7.5 is the only thing in the program that owns a subprocess handle, and a subprocess handle is
not an immutable value that can be passed through a queue. Each child gets:

| Thread | Started by | Drains | Publishes |
|--------|-----------|--------|-----------|
| `gputop-radeontop-out` | `RadeontopSource.start()` | the child's **stdout** | `GpuBlocks`, by rebinding one attribute to a frozen value |
| `gputop-radeontop-err` | `RadeontopSource.start()` | the child's **stderr** | a bounded deque of the last 20 lines, for classifying a failure |

Both drains are required, not belt-and-braces: an undrained pipe fills, the child blocks on its
own write, and its sampling freezes *silently* — the worst failure mode available, because every
observable still looks healthy. Neither thread is ever joined while the child is alive; joining is
a shutdown step, bounded by a timeout, so a child that dies holding its stderr pipe open cannot
hang the sampler.

**The one lock in the codebase, and why it is allowed.** `RadeontopSource` guards process creation
and teardown with a `threading.Lock`. It is the only lock in the project, and the rule above says
no shared mutable state exists to protect — but that reasoning is about *snapshot* state, and a
`Popen` handle is not snapshot state. The sampler thread starts and polls the child; the UI thread
may stop the pool during unmount. Those two genuinely race on creation and teardown, and unlike a
rebinding of a frozen dataclass there is no ownership argument that makes them safe.

The lock is held only across `Popen` and `terminate`/`wait` — a handful of syscalls per session
and per restart — and **never while reading a sample**. The values the hot path reads (`_status`,
`_blocks`) are still published by single rebinds of frozen objects, so the property the rest of the
program depends on is untouched: the lock protects the *handle*, not the *data*. A lock held across
the object's whole lifetime, or taken on every read, would have bought nothing and reintroduced
exactly the contention the rest of the design avoids.

**Free-threaded safety (`3.14t`):**

- No `threading.Lock` is required on the snapshot path, because there is no shared mutable state
  to protect. The one lock that exists is confined to the radeontop child's lifecycle and is held
  for a bounded number of syscalls.
- The sampler thread's delta state is *thread-confined*, not shared — the UI never reads it.
- The radeontop drain threads publish by rebinding frozen objects, the same rule
  `sampler/engine.py` follows for its history, so they need no lock of their own either.
- No C extension beyond Textual's own dependency tree is assumed to be `3.14t`-clean in v1; the CI
  matrix runs the full suite on `3.14t` and a subset is marked `xfail(strict=False)` if a
  dependency proves non-free-threaded. gputop's own modules must pass fully.
- `sys._is_gil_enabled()` is logged once at startup for diagnostics.

### 1.3 Error-handling policy

Two distinct policies, applied consistently:

| Layer | Policy |
|-------|--------|
| **Read layer** (`readers.py`, `metrics_bin.py`, `proc.py`) | *Total*. Every read is wrapped; any failure yields `None` / an empty result / a sentinel-free default. It never propagates. A failed file read is recorded in `SnapshotSource.unavailable` so the UI can show a subtle marker. |
| **Sampler loop** | *Total*, with a guard: if an entire tick raises unexpectedly, the exception is caught, counted in `SamplerStats.tick_errors`, and the next tick proceeds. The sampler thread must not die. |
| **UI** | *Total* for data rendering (`N/A` everywhere). A genuinely unexpected exception is logged and surfaced in the footer rather than killing the app. |

---

## 2. UI framework: Textual vs Rich `Live`

**Decision: Textual.** Rich `Live` was evaluated and rejected for v1.

| Criterion | Textual | Rich `Live` | Winner |
|-----------|---------|-------------|--------|
| Rich widgets (`Progress`, `Table`, `Tree`) | `Static` + own CSS, or custom `Widget` | Native | Rich |
| Sparkline graphs | `Static` with computed block/braille strings | Native via `Live` + `render_lines` | Tie |
| Focus, key bindings, overlays (`q` quit, `?` help, GPU picker) | **First-class**: `App.BINDINGS`, `Screen`, `ModalScreen` | **Must be hand-rolled** over `Live` | **Textual** |
| Scrolling / viewport for a 20-row process table | **First-class** | Hand-roll with a pager | **Textual** |
| Responsive layout on terminal resize | CSS (`1fr`, `min-width`, media queries) | Manual recomputation | **Textual** |
| CPU cost of an idle frame | Low (event-driven, no diff when nothing changed) | Must call `refresh()` manually | **Textual** |
| Dependency weight | Heavier | Lighter | Rich |
| Testability | Built-in `run_test()` pilot for TUI tests | Snapshot-string assertions | **Textual** |

Rationale: the value-add over `radeontop` is a *btop-like* screen — graphs, bars, a sortable
process table, help and picker overlays. Those interaction primitives are exactly what Rich `Live`
does **not** provide; it is a rendering primitive, not an application framework. Rebuilding focus
management, a modal layer, scrolling and resize handling on top of `Live` would cost more than the
dependency weight saves.

`Live` remains a reasonable choice for a much simpler `--plain` fallback mode if one is ever wanted,
which is one more argument for keeping the rendering isolated behind a `format.py` presentation
layer (see §3.2) rather than formatting inside widgets.

---

## 3. Module layout

### 3.1 Package tree

```
radeon-btop/
├── pyproject.toml              # requires-python >=3.14; uv-managed; ruff config
├── uv.lock
├── README.md
├── SPEC.md
├── DESIGN.md
├── gputop/
│   ├── __init__.py             # __version__, no heavy imports
│   ├── __main__.py             # python -m gputop
│   ├── cli.py                  # argparse surface, --dump, --config, --once, --blocks, --log, --version
│   ├── config.py               # frozen Config dataclasses + tomllib loader + validation
│   ├── sessionlog.py           # --log: CSV / JSON Lines writer, zstd by suffix (same level as config.py)
│   ├── errors.py               # GpuTopError hierarchy (rarely raised; reads never raise)
│   ├── logsetup.py             # logging; optional compression.zstd ring handler
│   ├── model/                  # ── pure data, no I/O ──
│   │   ├── __init__.py
│   │   ├── aliases.py          # type aliases: Mhz, Bytes, Celsius, Watts, Percent, Bdf…
│   │   ├── device.py           # AmdgpuDevice, DeviceKind
│   │   ├── metrics.py          # AmdgpuMetrics + Clock/MemoryPool/Power/Fan/ThrottleInfo/…
│   │   ├── blocks.py           # GpuBlocks, BlockReading, ClockReading, BlocksStatus, block order
│   │   ├── power.py            # PowerProfileTable, OdcVoltage, OdvDomain, OdvUnit (§7.4)
│   │   ├── alerts.py           # Thresholds, AlertState, AlertBreach, evaluate/worst (§10)
│   │   ├── process.py          # GpuProcess, EngineUsage, ProcessScanResult
│   │   ├── snapshot.py         # GpuSnapshot, SourceReport, SamplerStats
│   │   └── history.py          # fixed-size RingBuffer[T] for graphs
│   ├── readers/                # ── total I/O layer ──
│   │   ├── __init__.py
│   │   ├── fsutil.py           # read_text_safe/read_int_safe/read_bytes_safe + caching
│   │   ├── discovery.py        # enumerate /sys/class/drm/card* bound to amdgpu
│   │   ├── hwmon.py            # hwmon device resolution + label-aware temperature mapping
│   │   ├── dpm.py              # pp_dpm_* parsing, current_sclk/mclk, pp_power_profile_mode
│   │   ├── pp.py               # pp_power_profile_mode + pp_od_clk_voltage, **read-only**
│   │   ├── radeontop.py        # radeontop subprocess supervision + dump parsing (§7.5)
│   │   ├── throttle.py         # ThrottleStatus IntFlag + human labels
│   │   ├── metrics_bin.py      # gpu_metrics ABI registry + struct parsing (§6)
│   │   ├── procfs.py           # /proc/<pid>/fdinfo scanning, client dedup, engine deltas
│   │   └── merge.py            # source precedence (§4.3) → AmdgpuMetrics
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── engine.py           # Sampler thread, tick loop, stats, queue handoff
│   ├── ui/                     # ── Textual only; no I/O ──
│   │   ├── __init__.py
│   │   ├── app.py              # GpuTopApp, BINDINGS, SamplerControl widget
│   │   ├── header.py
│   │   ├── graphs.py           # braille/block sparklines from RingBuffer
│   │   ├── memory.py           # VRAM/GTT bars, GTT-first when iGPU
│   │   ├── sensors.py          # temps, power, fan, perf level, profile, throttle
│   │   ├── processes.py        # process table widget
│   │   ├── footer.py           # status + key hints + warnings
│   │   ├── overlays.py         # help, gpu picker, interval picker
│   │   ├── theme.py            # CSS / palettes
│   │   └── format.py           # byte/rate/temperature/unit formatting (no I/O, pure)
│   └── diagnostics/
│       ├── __init__.py
│       └── dump.py             # --dump → json / toml / zstd-json via compression.zstd
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── sysfs/              # synthetic fake sysfs trees (see §8)
    │   ├── proc/               # synthetic /proc trees
    │   └── gpu_metrics/        # binaries, one per ABI version + golden
    ├── test_config.py
    ├── test_fsutil.py
    ├── test_discovery.py
    ├── test_hwmon.py
    ├── test_dpm.py
    ├── test_throttle.py
    ├── test_power_tables.py     # pp_* parsing in both eras + the module's read-only proof
    ├── test_radeontop.py        # dump parsing by field name, argv, degradation states
    ├── test_alerts.py           # thresholds, hottest-sensor rule, no-latch, all-devices
    ├── test_sessionlog.py       # suffix rules, append/header seeding, empty cells, zstd
    ├── test_metrics_bin_v1.py
    ├── test_metrics_bin_v2.py
    ├── test_metrics_bin_edge.py
    ├── test_procfs.py
    ├── test_merge_precedence.py
    ├── test_sampler.py
    ├── test_deltas.py
    ├── test_format.py
    ├── test_ui_blocks_panels.py # blocks/profile panel text, heading, absent-vs-zero
    ├── test_ui_extra_panels.py  # app-level: hidden panels, alert border, recorder lifetime
    ├── test_no_writes.py       # static: read-only guarantee
    └── test_python_compat.py   # PEP 649: no __future__ imports
```

### 3.2 Layering rules (enforced by import-lint)

```
cli → ui → sampler → readers → model
              ↘        ↘
                    config, diagnostics, sessionlog
```

- `model/` imports **nothing** from the package except `aliases`. No I/O, no Textual, no logging.
- `ui/` performs **no** I/O **against the system**: it opens no sysfs path, starts no process and
  reads no `/proc` entry. It renders whatever `GpuSnapshot` it is handed.
- `readers/` never imports `ui` or `sampler`.
- `config.py` is importable from anywhere.
- `sessionlog.py` is importable from anywhere, and is used by both `cli` and `ui`, for the same
  reason `config.py` is: it is neither a snapshot-layer reader nor a presentation concern. It sits
  at the **same level as `config.py`** — shared by callers on both sides of the diagram rather than
  below either of them. It is opened by the app because `--log` is a property of the whole session,
  not of a sample, so the object that owns the session's lifetime owns the file. This is the one
  place `ui/` writes anything, and it writes only where the user named (§7.5).

This is what makes the TUI testable without a GPU and the readers testable without a terminal.

---

## 4. Data model

All dataclasses are `frozen=True, slots=True, kw_only=True`. Annotations use PEP 649 lazily-evaluated
forms (`X | None`, type aliases) with **no** `from __future__ import annotations`.

### 4.1 Type aliases (`model/aliases.py`)

```python
type Mhz = int
type Bytes = int
type Celsius = int
type Watts = int
type Rpm = int
type Percent = float  # 0.0–100.0, already clamped
type Bdf = str  # "0000:0c:00.0"
type ClientId = int
type Nanoseconds = int
```

### 4.2 Identity

```python
type DeviceKind = Literal["igpu", "dgpu"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AmdgpuDevice:
    index: int  # 0-based position in the sorted device list
    card: str  # "card1"
    render: str | None  # "renderD128"
    bdf: Bdf  # "0000:0c:00.0"
    vendor_id: int  # 0x1002
    device_id: int  # 0x73bf
    device_dir: Path  # /sys/class/drm/card1/device
    hwmon_dir: Path | None  # /sys/class/hwmon/hwmon1
    kind: DeviceKind
    name: str  # resolved marketing name
    vram_vendor: str | None  # "samsung" | "hynix" | "micron" | "N/A" | None
    boot_vga: bool
```

### 4.3 Per-metric snapshots

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPool:
    used: Bytes | None
    total: Bytes | None
    source: str  # provenance, e.g. "gpu_metrics" | "sysfs" | "none"


@dataclass(frozen=True, slots=True)
class Clock:
    current: Mhz | None
    maximum: Mhz | None
    source: str = "none"


@dataclass(frozen=True, slots=True)
class Power:
    draw_w: Watts | None
    cap_w: Watts | None
    energy_mj: int | None  # cumulative energy accumulator (gpu_metrics)
    source: str = "none"


@dataclass(frozen=True, slots=True)
class Fan:
    rpm: Rpm | None
    max_rpm: Rpm | None
    pwm: Percent | None

    @property
    def percent(self) -> Percent | None: ...  # rpm / max_rpm * 100, or pwm
    @property
    def stopped(self) -> bool: ...  # pwm == 0 → "0% (stopped)"


@dataclass(frozen=True, slots=True)
class Temperature:
    label: str  # "edge" | "junction" | "mem" | …
    celsius: Celsius | None
    source: str = "none"


@dataclass(frozen=True, slots=True)
class ThrottleInfo:
    raw: int
    active: frozenset[str]  # e.g. {"PPT0", "TDC_GFX", "TEMP_EDGE"}


@dataclass(frozen=True, slots=True)
class PcieLink:
    width: int  # e.g. 16
    speed: int  # tenths of a GT/s, e.g. 80 → 8.0 GT/s (Gen 3)

    @property
    def gt_per_second(self) -> float: ...  # speed / 10

    @property
    def generation(self) -> int: ...  # nearest of 2.5/5/8/16/32/64 GT/s, else 0
```

### 4.4 The per-device sample

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class AmdgpuMetrics:
    device: AmdgpuDevice
    sequence: int  # monotonic tick counter
    timestamp_ns: Nanoseconds  # time.monotonic_ns() at read start
    interval_s: float  # elapsed since the previous sample (for delta math)
    gpu_busy_percent: Percent | None
    mem_busy_percent: Percent | None
    vram: MemoryPool
    gtt: MemoryPool
    sclk: Clock
    mclk: Clock
    socclk: Clock
    voltages_mv: tuple[tuple[str, int], ...]
    temperatures: tuple[Temperature, ...]  # ordered edge, junction, mem, soc…
    power: Power
    fan: Fan
    performance_level: str | None
    power_profile: str | None
    throttle: ThrottleInfo | None
    pcie: PcieLink | None
    metrics_abi: str | None  # "v1.3" when gpu_metrics parsed, else None
    sources: frozenset[str]  # provenance across the whole sample
    unavailable: frozenset[str]  # attributes that were absent/denied this tick
```

### 4.5 Per-process

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class EngineUsage:
    engine: str  # "gfx" | "compute" | "sdma0" | "vcn0" | "vangogh" …
    percent: Percent  # clamped 0–100
    total_ns: Nanoseconds  # cumulative from fdinfo
    delta_ns: Nanoseconds  # this tick's delta


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuProcess:
    pid: int
    name: str
    user: str
    bdf: Bdf  # from drm-pdev — which GPU this client uses
    client_id: ClientId
    engines: tuple[EngineUsage, ...]
    vram_used: Bytes | None
    vram_shared: Bytes | None
    gtt_used: Bytes | None
    gtt_shared: Bytes | None

    @property
    def engine_percent(self) -> Percent: ...  # sum of engine %, clamped to 100
    @property
    def memory_used(self) -> Bytes: ...  # vram_used + gtt_used
```

`engine_percent` is the **sum of per-engine percentages clamped to 100**, not the maximum: a
process genuinely running on both gfx and compute is using both. The clamp prevents a multi-engine
client from reading as ">100 %".

### 4.6 The snapshot handed to the UI

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class SourceReport:
    metrics_abi: str | None
    per_metric: Mapping[str, str]  # MappingProxyType over a dict, built once, never mutated


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuSnapshot:
    sequence: int
    timestamp_ns: Nanoseconds
    devices: tuple[AmdgpuMetrics, ...]
    processes: tuple[GpuProcess, ...]
    visible_process_count: int
    total_process_count: int  # to detect /proc filtering by permissions
    stats: SamplerStats
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerStats:
    ticks: int
    tick_errors: int
    read_errors: int
    last_duration_ms: float
    avg_duration_ms: float
```

`visible_process_count != total_process_count` is what drives the
`showing 34 of 812 — other users hidden (need root)` footer note (SPEC §7).

### 4.7 Configuration (`config.py`)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    general: GeneralConfig
    gpu: GpuConfig
    process: ProcessConfig
    display: DisplayConfig
    diagnostics: DiagnosticsConfig

    @classmethod
    def load(cls, path: Path | None) -> Config: ...  # tomllib + validation + defaults
```

Defaults are dataclass field defaults; TOML is purely an override. Unknown keys produce a warning
list rather than an exception (forward compatibility with future config keys).

---

## 5. Discovery and device classification

### 5.1 Enumeration

`/sys/class/drm/card[0-9]*` — note the glob must exclude connector nodes (`card1-DP-1`,
`card1-HDMI-A-1`), hence the numeric-suffix pattern.

For each candidate, accept it only if **all** hold:

1. `<device>/vendor` reads `0x1002` (AMD).
2. `os.readlink(<device>/driver)` ends with `/amdgpu`. This is the definitive driver check — it
   rejects the `radeon` driver entirely, as required (no vendor abstraction, `amdgpu` only).
3. `<device>/drm/renderD*` exists, or `<device>/drm/card*` matches (render node preferred: primary
   nodes are `0660 root:video` and irrelevant since we never open them).

`bdf` is derived by resolving the device symlink and matching the last four path components
(`0000/c1/00.0`). Verified working on the reference machine (`card1 → 0000:0c:00.0`).

`hwmon_dir` is resolved **through the device**, not by scanning `/sys/class/hwmon` and matching the
`device` symlink — a naive scan breaks when a machine has many hwmon devices. Verified: this
machine has 7 hwmon devices and only `hwmon1` is `amdgpu`.

Devices are sorted `(kind is igpu, index) ` → discrete GPUs first, iGPUs last, each group ordered by
BDF, so the default view is the interesting one.

### 5.2 Device naming

`device_id` → marketing name requires a lookup table. v1 ships a **curated table** of common
consumer parts (Navi 10/12/14/20/21/22/23/24/31/32/33/44, RX 5000–9000, Vega, Radeon 780M/890M and
other APUs), overridable via `[gpu.names]` in the config. Unknown IDs render as
`AMD Radeon [0x73bf]`. No network lookup, no runtime download.

### 5.3 `gpu_metrics` discovery

`<device_dir>/gpu_metrics`. **Verified present** on kernel 7.0 with `amdgpu` 6.19; absent on many
older kernels, which is exactly why it is an acceleration path and not a requirement.

### 5.4 iGPU vs dGPU classification

No single reliable signal exists, so several cheap ones are combined. All are world-readable.

| Signal | iGPU | dGPU | Observed |
|--------|------|------|----------|
| `mem_info_vram_vendor` | `N/A` | `samsung`/`hynix`/`micron` | `samsung` → dGPU |
| `mem_info_vram_total` ≤ 512 MiB | likely | unlikely | 17 GiB → dGPU |
| `mem_info_gtt_total` ≫ `mem_info_vram_total` | likely | unlikely | ~equal here |
| PCI BDF bus `== 00` and device `!= 00:00.0` | common | rare | `0c` bus → dGPU |
| PCI device ID in the curated APU table | yes | no | — |
| `gpu_metrics` `format_revision == 2` | yes | no | `1` → dGPU |
| Boot VGA **and** a system-wide `mem_info_vram_total` ≪ `MemTotal` | likely | — | — |

Scoring: iGPU requires ≥ 2 independent iGPU-indicating signals, or 1 strong signal (the APU device
ID table). `kind = "auto"` uses this; `kind = "igpu"`/`"dgpu"` in the config overrides it. The
resolved kind is shown in the header so a mis-detection is visible and correctable.

---

## 6. `gpu_metrics` parsing

### 6.1 Common header (verified)

```c
struct metrics_table_header {      /* file offset 0, 4 bytes */
    __u16 structure_size;          /* total buffer size INCLUDING these 4 bytes */
    __u8  format_revision;         /* 1 = v1.x, 2 = v2.x, 3 = v3.x */
    __u8  content_revision;        /* minor revision */
};
```

Verified on the reference machine: first four bytes `78 00 01 03` → `structure_size=120`,
`format_revision=1`, `content_revision=3`, **v1.3**; file length exactly 120, so the body is
`structure_size - 4 = 116` bytes.

### 6.2 ABI registry

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class MetricsAbi:
    fmt_revision: int
    content_revision: int
    label: str  # "v1.3"
    body: struct.Struct  # explicit format, leading header included
    field_map: Mapping[str, int]  # canonical name → byte offset
    sentinels: frozenset[int]  # values meaning "not supported"


ABI_REGISTRY: Mapping[tuple[int, int], MetricsAbi] = {...}
```

**Design rule — explicit padding, never native alignment.** Do *not* rely on `struct`'s native
alignment (`"="`), because the C struct's padding depends on the compiler's ABI. Each registry entry
carries a fully explicit format string with literal `x` pad bytes, and a test asserts
`struct.calcsize(fmt) == structure_size` for every captured fixture. Getting this wrong would
silently misplace every `u64` field.

The verified v1.3 layout, offsets in **file coordinates** (the C struct starts at offset 0, so
`u64` fields land on multiples of 8 — confirmed at offsets 24, 32 and 96):

| Offset | Type | Field | Value in the captured blob |
|-------:|------|-------|---------------------------:|
| 0  | `H` | `structure_size` | 120 |
| 2  | `B` | `format_revision` | 1 |
| 3  | `B` | `content_revision` | 3 |
| 4  | `H` | `temperature_edge` | 52 °C |
| 6  | `H` | `temperature_hotspot` | 59 °C |
| 8  | `H` | `temperature_mem` | 54 °C |
| 10 | `H` | `temperature_vrgfx` | 46 mV·10 |
| 12 | `H` | `temperature_vrsoc` | 0 |
| 14 | `H` | `temperature_vrmem` | 0 |
| 16 | `H` | `average_gfx_activity` | 3 % |
| 18 | `H` | `average_umc_activity` | 0 % |
| 20 | `H` | `average_mm_activity` | 0 % |
| 22 | `H` | `average_socket_power` | 33 W |
| 24 | `Q` | `energy_accumulator` | 56 653 004 mJ |
| 32 | `Q` | `system_clock_counter` | 1 340 758 475 714 ns |
| 40–52 | `H`×7 | `average_{gfx,soc,u,vclk0,dclk0,vclk1,dclk1}clk_frequency` | avg uclk 1000 MHz |
| 54 | `H` | `current_gfxclk` | 500 MHz |
| 56 | `H` | `current_socclk` | 993 |
| 58 | `H` | `current_uclk` | 1000 MHz |
| 60–66 | `H`×6 | `current_{vclk0,dclk0,vclk1,dclk1}` + spare | 0 |
| 68 | `I` | `throttle_status` | 0 |
| 72 | `H` | `current_fan_speed` | 0 RPM |
| 74 | `H` | `pcie_link_width` | 16 |
| 76 | `H` | `pcie_link_speed` | 80 (8.0 GT/s → Gen 3) |
| 78 | `H` | `padding` | — |
| 80 | `I` | `gfx_activity_acc` | `0xFFFFFFFF` sentinel |
| 84 | `I` | `mem_activity_acc` | `0xFFFFFFFF` sentinel |
| 88 | `H` | `temperature_hbm` | `0xFFFF` sentinel |
| 90 | `6x`| pad to 8-alignment | — |
| 96 | `Q` | `firmware_timestamp` | `0xFFFFFFFFFFFFFFFF` sentinel |
| 104 | `H` | `voltage_soc` | 918 mV |
| 106 | `H` | `voltage_gfx` | 6 → below floor ⇒ `N/A` |
| 108 | `H` | `voltage_mem` | 900 mV |
| 110 | `B` | `padding1` | — |

Field lists for the other registered ABIs (identical layout rules; only the field set differs):

- **v1.0** — `system_clock_counter` (u64, **preceded by 4 bytes of padding** because the `u64` must
  land on a multiple of 8), 6 temps, 3 activities, `average_socket_power`, `energy_accumulator`
  (**u32**), 7 avg clocks, 7 current clocks, `throttle_status`, `current_fan_speed`,
  `pcie_link_width`/`pcie_link_speed` (**u8** here, not u16).
- **v1.1** — as v1.0 with `energy_accumulator` widened to `u64`, `system_clock_counter` reordered
  after it, PCIe fields widened to `u16`, plus `gfx_activity_acc`, `mem_activity_acc`,
  `temperature_hbm`.
- **v1.2** — v1.1 plus trailing `firmware_timestamp` (u64).
- **v1.3** — v1.2 plus `voltage_soc`, `voltage_gfx`, `voltage_mem` and `padding1`. ← verified
- **v2.0** — APU layout: `system_clock_counter`, `temperature_{gfx,soc,core,l3}`,
  `average_{gfx,mm}_activity`, `average_{socket,cpu,soc,gfx,core}_power`, avg clocks including
  `fclk`, current clocks including `coreclk`/`l3clk`, `throttle_status`, `fan_pwm`.
- **v2.1** — v2.0 with `system_clock_counter` moved after the activities.
- **v2.2** — v2.1 plus trailing `indep_throttle_status` (u64).
- **v3.x** — **not registered.** See §6.5.

The v1.x → v2.x differences are *semantic*, not just positional: v2.x replaces `uclk` with `fclk`,
splits power into five domains, and uses APU-specific temperature names. The parser therefore maps
each ABI onto canonical metric names at decode time (`current_uclk` → `mclk`,
`temperature_gfx` → `edge`, `average_socket_power` → `power.draw`), so downstream code never sees
ABI differences.

### 6.3 Parsing algorithm

```python
def parse_gpu_metrics(blob: bytes) -> AmdgpuBinaryMetrics | None:
    if len(blob) < HEADER_SIZE:
        return None  # too short
    structure_size, fmt_rev, content_rev = HEADER.unpack_from(blob)
    if not (HEADER_SIZE < structure_size <= len(blob)):
        return None  # implausible
    abi = ABI_REGISTRY.get((fmt_rev, content_rev))
    if abi is None:
        return Unrecognised(fmt_rev, content_rev)
    body = blob[HEADER_SIZE:structure_size]  # bound by the header, not by len()
    fields = abi.body.unpack_from(blob)  # may raise → caught, treated as truncated
    ...
```

Key behaviours:

- **`structure_size` bounds the body.** A driver reporting a *larger* struct than this build knows is
  truncated safely; a *smaller* one yields fewer fields, each absent one becoming `None`.
- **Truncation is tolerated.** `struct.error` from a short buffer is caught per-field (fields whose
  bytes are present are kept), not by discarding the whole sample.
- **Unknown ABI is never guessed.** It returns `Unrecognised`, the caller falls back to sysfs/hwmon,
  and the footer shows `gpu_metrics ABI v{fmt}.{rev} unrecognised — using sysfs`.

### 6.4 Sentinel and plausibility filtering

Applied per field, converting to `None`:

| Rule | Rationale |
|------|-----------|
| `u16` field `== 0xFFFF` | Verified unsupported-field sentinel (`temperature_hbm`, `average_socclk_frequency`) |
| `u32` field `== 0xFFFFFFFF` | Verified unsupported accumulator |
| `u64` field `== 0xFFFFFFFFFFFFFFFF` | Verified unsupported timestamp |
| Clock `< 0 or > 10000` MHz | Physically impossible; catches misaligned reads |
| Temperature `< 0 or > 150` °C | Physically impossible |
| `voltage_gfx/vrmem/vrsoc` below floor | No such rail on some ASICs; the floor check is insurance, not an observed case |
| Power `== 0xFFFF` | Would otherwise render 65.535 kW |

The plausibility bounds are **per field family**, not global — a single generic "sanity clamp"
would wrongly discard legitimate readings (e.g. `temperature_mem == 0` on an unpopulated sensor is
also meaningless, but `sclk == 0` when the GPU is idle is legitimate and must be preserved). This is
the distinction that separates a credible monitor from one that shows nonsense.

### 6.5 Forward compatibility with unrecognised versions

Adding a version is: one `struct.Struct`, one `field_map`, one registry line, one fixture. No
parser changes. For any version not in the registry — **including all of `3.x`** — gputop falls back
to sysfs/hwmon and says so in the footer. `gputop --dump-metrics <path>` prints the 4-byte header and
a hex dump so a user on a newer kernel can submit a one-line bug report that makes the next release
support their card. This converts an unknowable-in-advance ABI question into a cheap support ticket.

---

## 7. sysfs / procfs path table

Root: `/sys/class/drm/<card>/device` unless stated. **All reads below were verified present or
absent on the reference machine**, which is why the availability column says so.

### 7.1 Core metrics

| Metric | Primary path | Fallback 1 | Fallback 2 | Ref. machine |
|--------|--------------|-----------|-----------|--------------|
| `gpu_busy_percent` | `gpu_busy_percent` | `gpu_metrics` `average_gfx_activity` | — | present, `0` |
| `mem_busy_percent` | `mem_busy_percent` | `gpu_metrics` `average_umc_activity` | — | present, `0` |
| VRAM used | `mem_info_vis_vram_used` | `mem_info_vram_used` | `gpu_metrics` (none — no VRAM field) | `648 192 000` |
| VRAM total | `mem_info_vis_vram_total` | `mem_info_vram_total` | — | `17 163 091 968` |
| GTT used | `mem_info_gtt_used` | — | — | `78 880 768` |
| GTT total | `mem_info_gtt_total` | — | — | `16 770 666 496` |
| `sclk` | `current_sclk` | `gpu_metrics` `current_gfxclk` | `pp_dpm_sclk` `*`-marked row | **`current_sclk` absent**; metrics gave 500 MHz |
| `sclk` max | `current_sclk_max` | `pp_dpm_sclk` highest row | `gpu_metrics` avg | absent → metrics |
| `mclk` | `current_mclk` | `gpu_metrics` `current_uclk` | `pp_dpm_mclk` `*`-marked row | **`current_mclk` absent**; metrics gave 1000 MHz |
| `mclk` max | `current_mclk_max` | `pp_dpm_mclk` highest row | — | absent → 1000 MHz |
| Edge temp | `hwmon/<h>/temp1_input` | `gpu_metrics` `temperature_edge` | `temp2_input` | `51000` / metrics 51 °C |
| Junction temp | `hwmon/<h>/temp2_input` *(label `junction`/`hotspot`)* | `gpu_metrics` `temperature_hotspot` | — | `58000` label `junction` |
| Mem temp | `gpu_metrics` `temperature_mem` | `hwmon/<h>/temp3_input` *(label `mem`)* | — | **metrics-only: 54 °C, no hwmon equivalent** |
| SOC / core temp | `gpu_metrics` `temperature_soc` | `hwmon/<h>` label match | — | `0` ⇒ filtered to `N/A` |
| Power draw | `hwmon/<h>/power1_average` | `gpu_metrics` `average_socket_power` | `power1_input` | `33 000 000 µW` ⇒ 33 W |
| Power cap | `hwmon/<h>/power1_cap` | `power1_cap_max` | `gpu_metrics` | `211 000 000 µW` ⇒ 211 W |
| Fan RPM | `hwmon/<h>/fan1_input` | `gpu_metrics` `current_fan_speed` | `pwm1` (→ %) | `0` |
| Fan max RPM | `hwmon/<h>/fan1_max` | — | — | `3000` |
| Fan PWM | `hwmon/<h>/pwm1` | — | — | `0` |
| Perf level | `power_dpm_force_performance_level` | `pp_dpm_performance_level` | — | `auto` |
| Power profile | `pp_power_profile_mode` (`*`-marked profile name) | — | — | `BOOTUP_DEFAULT` |
| Throttle status | `gpu_metrics` `throttle_status` / `indep_throttle_status` | — | — | `0` |
| PCIe link | `gpu_metrics` `pcie_link_width`/`pcie_link_speed` (0.1 GT/s) | `current_link_width`/`current_link_speed` (`"16.0 GT/s PCIe"`) | `max_link_width`/`max_link_speed` | metrics 16 / 80 |
| Voltages | `gpu_metrics` `voltage_{soc,gfx,mem}` | — | — | 918 / `N/A` / 900 mV |
| VRAM vendor | `mem_info_vram_vendor` | — | — | `samsung` |
| Device ID | `device` | — | — | `0x73bf` |
| PCI BDF | `readlink(device/)` | — | — | `0000:0c:00.0` |
| Boot VGA | `/sys/bus/pci/devices/<bdf>/boot_vga` | — | — | `1` |
| Driver version | `/sys/module/amdgpu/version` | — | — | `6.19.4` |

> **Design note.** `current_sclk` and `current_mclk` are **absent** on kernel 7.0 in this driver,
> despite being widely documented. This is precisely why the `pp_dpm_*` `*`-row fallback and the
> `gpu_metrics` fallback both exist and are tested. A parser written only against the commonly
> cited paths would show `N/A` for every clock on a modern system.

### 7.2 hwmon temperature label mapping

Labels are read from `tempN_label`; anything unrecognised is surfaced with its raw label rather than
dropped.

| `tempN_label` | Canonical |
|---------------|-----------|
| `edge` | `edge` |
| `junction` / `hotspot` | `junction` |
| `mem` / `memoria` | `mem` |
| `soc` | `soc` |
| *(missing file)* | `tempN` by index |

### 7.3 Per-process paths

| Item | Path / key | Ref. machine |
|------|------------|--------------|
| PID list | `/proc/<pid>/` | — |
| Process name | `/proc/<pid>/comm` (truncated to 15 chars — use `/proc/<pid>/status` `Name:` as a fallback, then `/proc/<pid>/cmdline` argv[0]) | — |
| User | `pwd.getpwuid(os.stat("/proc/<pid>").st_uid).pw_name` | — |
| FD → node | `os.readlink("/proc/<pid>/fd/<fd>")` → `…/dri/renderD128` | verified |
| DRM accounting | `/proc/<pid>/fdinfo/<fd>` | verified |
| Client identity | `drm-client-id`, `drm-pdev`, `drm-driver` | `71`, `0000:0c:00.0`, `amdgpu` |
| Engine time | `drm-engine-<name>: <ns>` | `drm-engine-gfx: 268416580 ns` |
| VRAM | `drm-total-vram`, `drm-resident-vram`, `drm-shared-vram`, `drm-purgeable-vram` | `162860 KiB` |
| GTT | `drm-total-gtt`, `drm-resident-gtt`, `drm-shared-gtt`, `drm-purgeable-gtt` | `16444 KiB` |
| amdgpu extras | `amd-requested-vram`, `amd-requested-gtt`, `amd-evicted-vram` | verified |
| Newer engine ABI | `drm-engine-active-<n>`, `drm-engine-period-<n>`, `drm-engine-capacity-<n>` | **not present** on this kernel |

**Engine-name normalisation.** `drm-engine-<name>` names are driver-defined. Observed on RDNA2:
`gfx`, `compute`. Expect, depending on ASIC: `sdma0`/`sdma1`/`dma`, `vangogh`/`vangogh_vcn*` (VCN
encode/decode on RDNA2+), `jpeg`, `vcn0`/`vcn1`, `video`, `audio`. gputop maps known names into the
display buckets **`gfx`, `compute`, `dma`, `enc`, `dec`, `other`** and passes anything unrecognised
through as `other` with its raw name retained — so a new engine name still shows up rather than
vanishing.

**Cross-user limitation.** `/proc/1/fdinfo/0` returns `Permission denied` for uid 1000 (verified).
The scanner therefore enumerates what it can see and reports both counts so the UI can be honest
about the truncation (SPEC §7, §4.6).

### 7.4 Read-only SCPP and overdrive attributes

Both of these are **writable** sysfs files whose contents reprogram hardware. They are read and
never written, which is a structural property of `readers/pp.py` rather than a convention: the
module exposes no function accepting a mode or a writable handle, a test scans the module's own
source for `write_text`, `write_bytes`, `open("w")`, `os.open` and `truncate`, and a second test
reads a table and asserts the file is byte-identical afterwards — a reader that accidentally
opened for writing would truncate its own input.

| Attribute | Contents | Read as | Notes |
|-----------|----------|---------|-------|
| `pp_power_profile_mode` | `<id> <NAME>` rows, `*` marks the active profile | The **whole** table, plus the active entry | The per-profile detail rows (`0(  GFXCLK)  0  5  1 …`) are skipped without a special case: they open with a digit immediately followed by `(`, which no profile row has, so one expression separates the two |
| `pp_od_clk_voltage` | Either a legacy voltage table or a sectioned overdrive **clock** table | Per-domain overdrive ceilings plus the `OD_RANGE` floor/ceiling and the `OD_VDDGFX_OFFSET` | See below |

**`pp_od_clk_voltage` is two formats behind one filename.** Older parts wrote a voltage table —
`0:300 1:350 2:400`, bare millivolts, no sections. On RDNA2 and later the driver repurposed the
same attribute for overdrive *clock* ceilings and added sections:

```
OD_SCLK:
0: 500Mhz
1: 2104Mhz
OD_MCLK:
0: 97Mhz
1: 1000MHz
OD_RANGE:
SCLK:     500Mhz       2600Mhz
MCLK:     674Mhz       1075Mhz
OD_VDDGFX_OFFSET:
-50mV
```

The parser is therefore **section-aware and unit-carrying**: the unit travels with each entry
instead of being assumed, because presenting a 2104 MHz ceiling as a voltage would be wrong on
every modern card, and a parser that guessed would be confidently wrong rather than absent. Whether
a file declares `OD_*` sections decides the default unit for a bare entry; a file with no section
header at all is read as the single legacy voltage table it is, so cards from both eras decode
through one entry point. Entries are matched with `finditer`, not `match`, because some kernels
write one pair per line and others write a whole run of pairs on one line — an anchored match
would read the first pair and silently discard the rest of the table.

The per-domain bar is a ratio of the driver's own stated `OD_RANGE`, not of a hard-coded maximum,
so it stays correct on a part whose ceiling is not the one this code was written against.

### 7.5 The two paths that are not sysfs

Per-block utilisation is the one **source** that is not a file, and the session log is the only
**destination** that is not a config file or the terminal.

**Source — a supervised `radeontop` child.** The amdgpu driver exposes GRBM and SRBM busy only
through the radeon ioctl on a privileged handle, which is exactly the dependency this program's
unprivileged design refuses to take on globally. The counters are therefore read the one way that
leaves *gputop* unprivileged: by supervising `radeontop` and parsing its dump output.

```
radeontop -d - -i <interval> -l 0 -t <ticks> [-p /dev/dri/cardN | -b <bus>]
```

| Flag | Why |
|------|-----|
| `-d -` | dump to stdout, so the output is a stream to consume rather than a file to re-read |
| `-i <interval>` | seconds between dumps. **Whole seconds only**: radeontop parses it with `atoi` and floors it at 1, so `RadeontopOptions.validated()` clamps to the same floor rather than silently sampling slower than the user asked for |
| `-l 0` | radeontop's spelling of "until terminated" |
| `-t <ticks>` | samples per second the child computes internally. The default of 120 costs CPU and buys accuracy nobody can see on a bar that repaints once a second |
| `-p <node>` | the DRM node, **preferred** — a PCI bus alone is ambiguous on a machine with two cards behind one bridge, and the wrong card's numbers are worse than no numbers |
| `-b <bus>` | the two-digit hexadecimal bus from the BDF, used only when no node path exists |

One child per GPU. The dump line is parsed **by field name, never by position** — `dump.c` emits
`tc`, `smx`, `cr`, `uvd`, `vce0`, `vram`, `gtt` and the clocks only when the card reports those
bits, so a positional parser reads `vgt`'s value as `ta`'s on every modern card and reports
confidently wrong numbers. Naming the fields makes a missing block a missing block. Block *keys*
travel with every reading (they are what the JSON dump and the CSV header use); the *labels* are
radeontop's own display names, abbreviated only enough to fit the panel's label column.

**Degradation is five states, not one.** A missing binary, a card the driver cannot read, a
refused ioctl and a crash are four different problems with four different remedies, so they are
four values of `BlocksStatus` plus `DISABLED`. The panel is hidden and the reason is carried on
the snapshot (`blocks_hint`) — as a hint rather than a warning, because it is a permanent
condition that would otherwise occupy a warning slot for the whole session and push out a real
error that happened once.

Failure classification reads **stderr**, and the stderr drain is joined *before* classifying. That
ordering is load-bearing rather than tidy: radeontop reports every problem it has, most usefully
last — "Failed to find DRM devices", then "Failed to open DRM node", then "Cannot access GPU
registers, are you root?". stdout closes as soon as the child dies while stderr is still being
delivered, so classifying on stdout EOF alone reads a partial buffer and reports the least
actionable of the three, telling a user their card is unsupported when the real answer is that
they need privileges radeontop wants.

A child that dies mid-session (a driver reload, a hot-unplugged eGPU) is restarted a bounded
number of times. `MISSING`, `UNSUPPORTED` and `NOT_PERMITTED` are never retried: they will fail
again just as promptly, and retrying would be a busy loop. The child's pipes are closed
explicitly rather than left to the garbage collector — a session that starts and stops the pool
repeatedly leaks two descriptors each time and eventually runs out of them.

**Destination — the session log.** `--log PATH` is the only way gputop writes anything derived from
the samples; the one other thing it writes is the `[state]` section of its own config file
(SPEC §9).
The format comes from the filename — `.csv`, or `.json`/`.jsonl`/`.ndjson` — with an optional
trailing `.zst`/`.zstd` meaning zstd through the standard library's `compression.zstd`. The base
name chooses the format and the suffix chooses the transport, so the two are independent and
neither is declared twice: `session.csv.zst` is a compressed CSV.

| Decision | The obvious alternative | What the alternative costs |
|----------|-------------------------|---------------------------|
| JSON **Lines**: one object per line, per device | a single JSON array | an interrupted session loses everything; a truncated array is not readable at all |
| append mode, header seeded from whether the file already had content | truncate on open | recording twice to one path writes a second header into the middle of the data, where a strict CSV reader absorbs it as a record |
| an absent reading is an **empty cell** | `0`, or `N/A` | `0` is a lie; `N/A` in a numeric column is a parse error every consumer must special-case |
| first failure wins, then silence | warn per record | a full disk buries the status line at one warning per sample |
| flush per record, close on every exit path | buffer until exit | a `.zst` stream left unclosed is a truncated archive, not a readable file |

The block columns are present whether or not `radeontop` is running. A stable column set is the
entire point of a CSV header: a consumer must be able to read row 4000 with the code that read
row 1, including on a session where `radeontop` was installed halfway through. A card with no
Texture Cache unit leaves that cell empty rather than shifting every later column.

---

## 8. Process-scan algorithm

```
for each /proc/<pid> that is a numeric directory:            # one scandir per tick
    for fd in /proc/<pid>/fd:
        target = readlink(fd)                                 # no open()
        if not target.endswith(("renderD*", "card*")): continue
        parse /proc/<pid>/fdinfo/<fd> line by line            # only for DRM nodes
        if drm-driver != "amdgpu": continue
        key = (drm_pdev, drm_client_id)                       # ← dedup key
        merge into accumulator[key] taking MAX per field
for each accumulated client:
    engines = [{engine, (t_now - t_prev) / wall_dt * 100} for each engine]
    keep the previous cumulative counters (sampler-thread-private) for the next tick
```

Costs, and how they are bounded:

- `scandir("/proc")` is cheap; `readlink` per fd is the dominant cost. **Mitigation:** the set of
  candidate PIDs is filtered *before* fd iteration by checking `/proc/<pid>/fd` only for numeric
  dirs, and the per-process result is cached keyed by `(pid, fd ino)` so an unchanged process (no new
  memory counters, no engine advance) re-parses only its `fdinfo` header, not every file.
- **Full engine deltas need the previous tick's counters**, which live only in the sampler thread.
  A client first seen this tick has no baseline and reports `N/A` for utilisation (not `0`).
- Utilisation is `delta_ns / (wall_delta_ns) * 100`, clamped to 100 — because `drm-engine-*` is
  cumulative time *on the engine*, a delta can exceed wall time on engines that count concurrently
  scheduled work. Clamping is mandatory, not cosmetic.
- **Multi-GPU:** `drm-pdev` routes each client to its own device's table, so a dGPU and an iGPU are
  accounted separately. This is verified-format data on this machine and is the load-bearing
  mechanism for FR5.

---

## 9. Sampler

```python
class Sampler:
    def __init__(self, config: Config, queue: queue.SimpleQueue[GpuSnapshot]) -> None: ...
    def run(self) -> None:                     # body of the thread
    def stop(self) -> None: ...                # threading.Event, wakes the sleep immediately
```

Tick loop:

1. `t0 = time.monotonic_ns()`
2. Enumerate devices (cheap re-check for hotplug; cached unless `r` or an interval elapses).
3. For each device: parse `gpu_metrics` → read device sysfs → read hwmon → merge by precedence.
4. Scan `/proc` → dedup → compute engine deltas against this thread's previous counters.
5. Build `GpuSnapshot`, `queue.put_nowait(snapshot)`.
6. `stop.wait(timeout=interval_s)` — interruptible sleep, so quitting is immediate.

When the per-block panel is enabled the tick also calls `RadeontopPool.poll()`: a liveness check and
a bounded restart per child, with no I/O of its own — the drain threads own the pipes. The pool's
status and hint ride along on the snapshot, so the UI never has to ask the subprocess anything.

Behavioural guarantees:

- **Never misses the deadline**: work is bounded; if a tick overruns, the next sleep is shortened
  rather than skipped, and `SamplerStats.tick_errors` records anomalies.
- **Never raises out**: a broad `except Exception` around the tick body logs and continues.
- **Read amplification is bounded**: `gpu_busy_percent` and friends are read exactly once per tick
  per device — the cost of the process scan, not of the sysfs reads, dominates. This is what keeps
  a 100 ms interval viable.
- **No sysfs writes**, ever.

---

## 10. UI wireframe

Target: 100×34 minimum, 120×40 comfortable. Shown for a **discrete** RDNA2 GPU.

```
┌─ gputop ─────────────────────────────────────────────────────────────────────────────────────────────────┐
│  AMD Radeon RX 9060 XT  ·  0000:0c:00.0  ·  dGPU  ·  amdgpu 6.19.4  ·  VRAM: samsung  ·  [1/2]        │
├──────────────────────────────────────────────────────┬──────────────────────────────────────────────────┤
│  GPU  3 ▁▂▂▁▂▃▂▁▁▁▂▃▄▅▅▆▇▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇  38%      │  ┌ CLOCKS ──────────────────────────────────┐ │
│  MEM  1 ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▃▄▅▆▇  7%       │  │ sclk  ▐█████████████▌              500   │ │
│                                                      │  │       ▐                            MHz   │ │
│  ┌ MEMORY ────────────────────────────┐              │  │       max                           MHz   │ │
│  │ VRAM ▐██████████████████░░░░░░░  648 MB / 16.0 GB │  │ mclk  ▐██████████████████████████ 1000   │ │
│  │ GTT  ▐░░░░░░░░░░░░░░░░░░░░░░░░░░░    75 MB / 15.6 GB│  │       max                          MHz   │ │
│  └────────────────────────────────────┘              │  └──────────────────────────────────────────┘ │
│                                                      │  ┌ SENSORS ────────────────────────────────┐ │
│                                                      │  │ edge       ▐████████▌            52 °C    │ │
│                                                      │  │ junction   ▐██████████▌          59 °C    │ │
│                                                      │  │ mem        ▐████████▌            54 °C    │ │
│                                                      │  │ power      ▐██▌             33 W / 211 W │ │
│                                                      │  │ fan        ░░░░░░░░░░░   0 RPM  (stopped) │ │
│                                                      │  │ perf level  auto                       │ │
│                                                      │  │ profile     BOOTUP_DEFAULT             │ │
│                                                      │  │ throttle    none                       │ │
│                                                      │  │ pcie        Gen3 x16                   │ │
│                                                      │  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┴──────────────────────────────────────────────────┤
│  PROCESSES  (engine ▾)                             showing 9 of 214 — other users hidden (need root)   │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│    PID  USER       NAME                GFX  COMPUTE  DMA   ENC   DEC    VRAM       GTT               │
│  12345  david      AMD family 25       ███  42.1%      1.2%    -     -      159 MB     16 MB           │
│  22871  david      firefox            ██▌  18.7%      0.8%    2.1%   -       84 MB      2 MB           │
│   1042  root      Xorg                 ▌    -          -      1.4%   -       38 MB      0 B            │
│  30114  david      python3              -    6.3%     0.1%    -     -      12 MB     41 MB           │
│   2071  david      plasmashell          ▏    -          -      -     -       4 MB      4 MB           │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ gpu_metrics v1.3 · interval 1000 ms · tick 4.1 ms · q quit  g/G switch GPU  s sort  f filter  m memory │
│ i interval  p picker  r rescan  h help  d dump                                              ⚠ x unknown │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**On an iGPU** the memory panel swaps emphasis and the header drops the VRAM line:

```
│  ┌ MEMORY (iGPU — GTT is the working set) ──────────────────────┐   │
│  │ GTT  ▐██████████████████████████▌  3.42 GB / 15.6 GB  (22%) │   │
│  │ VRAM ▐▌                        512 MB / 512 MB   (carve-out) │   │
│  └─────────────────────────────────────────────────────────────┘   │
```

**With `--blocks`**, an extra row carries the two optional panels side by side, each using the
wider label column the block names need so the pair shares one left edge:

```
│  ┌ BLOCKS · Texture Addr 99% ─────────────┐  ┌ POWER PROFILE ─────────────────────────────┐  │
│  │                                        │  │                 profile  BOOTUP_DEFAULT    │  │
│  │                                        │  │                 index    0 of 7            │  │
│  │                                        │  │                 available  3D_FULL_SCREEN …│  │
│  │ Graphics pipe    ▐██████████▌ 82%      │                                                  │
│  │ Event Engine     ▐██▌ 24%              │                                                  │
│  │ Vertex Grouper   ▐▌ 9%                 │                                                  │
│  │ Texture Addr     ▐█████████████ 99%    │  │ OD Shader clock  ▐█████████▌ 2104/2600     │  │
│  │ Texture Cache    ▐█████████████ 98%    │  │ OD Memory clock  ▐██████████▌ 1000/1075    │  │
│  │ Shader Export    ▐██▌ 18%              │  │ OD SoC clock     ▐████████▌ 900/1200       │  │
│  │ Shader Interp    ▐▌ 6%                 │                                                  │
│  │ Scan Converter   ▐              0%     │  │                 vddgfx      +25 mV         │  │
│  │ …                                      │  │                 read-only                  │  │
│  │ UVD              ▐              0%     │                                                  │
│  │ Memory Clock    ▐████████▌ 840 MHz     │                                                  │
│  └────────────────────────────────────────┘  └────────────────────────────────────────────┘  │
```

Wireframe rules:

- The footer always shows the **resolved source** (`gpu_metrics v1.3` vs `sysfs`) — users can
  immediately tell whether they are getting the rich path or the fallback.
- `⚠` markers appear for: partial process visibility, unrecognised metrics ABI, degraded GPU clock
  source.
- Every bar has a text value; bars are decorative, never the only carrier of information
  (accessibility, and narrow terminals).
- At width < 100 the graphs and sensors collapse to a single column; below 70 columns a
  "terminal too narrow" message is shown rather than a mangled layout.
- **The two optional panels are decided independently**, even though they share one row. The blocks
  panel needs a privileged child process and the power-profile panel is read from sysfs, so gating
  the second on the first withheld a table that works everywhere from every unprivileged user.
  Each is hidden when it has nothing to say; a panel of empty bars with a "waiting" caption is a
  worse answer than no panel plus a sentence, because it implies the data is coming. When the
  blocks panel is hidden the reason is in the status line and in the help overlay.
- **One height threshold gates the row**, measured rather than estimated. The cost is not the row's
  own height but what it takes from the sensor panel beside it: that panel scrolls, so a row added
  below the main one pushes its trailing lines (PCIe link, throttle reasons) out of view. The
  primary readings — clocks, temperatures, power, cap — must survive, and
  `MIN_HEIGHT_FOR_OPTIONAL_ROW` is the measured height at which they do. A test asserts the
  invariant at and above the threshold, so raising the row further cannot silently cost the
  sensors something.
- The blocks panel's **title** names the busiest block, so "which block" — the question the panel
  is opened with — is answered before the bars are read. Labels are radeontop's own names,
  abbreviated (`Texture Addr`, `Shader Interp`) only enough to fit the label column; that column's
  width is *derived* from the label table rather than hard-coded, so a label whose length no longer
  matches the column is caught by a test rather than discovered as a wrapped row. The power-profile
  panel indents its plain text rows to the same column, so both panels' bars start at the same
  offset.
- A block at exactly zero still draws an empty bar with a `0%` label: radeontop omits a block the
  hardware has no unit for, so every row shown describes something that exists.
- The power-profile panel states `read-only` in its own body. It is a claim the code makes
  checkable (§7.4), and printing it costs one line of a panel that is usually taller than that.
  It shares a row with the blocks panel and is shown only when that panel is — one
  "SCPP is unavailable" row is worth more than two panels that can disagree about it.
- The **alert border** replaces the accent on every panel at once while a threshold is crossed,
  alternating between the alert colour and the track colour — the dark phase is the *track* rather
  than the accent, so the flash reads as "this border went away" rather than as a second thing to
  look at. One border that changes says "this screen"; three borders with independent phases
  would say nothing. The colour is written to each panel's own `widget.styles.border` and **not**
  through a CSS variable: `Stylesheet.set_variables` *replaces* the variable map rather than
  merging into it, so driving a once-per-second flash through it deletes Textual's own design
  tokens — including the `$background` its default `App` rule references — and the whole
  stylesheet then fails to resolve. That was a real failure, found and fixed; it is recorded here
  because the mistake is invisible until the sheet stops loading.
- With nothing over a limit the border is the theme accent, and must leave no trace of having been
  alarmed: the alert state is recomputed from scratch every sample, so it cannot latch.

---

## 11. Keybindings

| Key | Action |
|-----|--------|
| `q`, `Ctrl-C` | Quit |
| `g`, `Tab`, `→`, `]` | Next GPU |
| `G`, `Shift-Tab`, `←`, `[` | Previous GPU |
| `1`–`9` | Jump to GPU *n* |
| `p` | GPU picker overlay |
| `m` | Toggle memory emphasis (GTT-first ↔ VRAM-first) |
| `s` | Cycle process sort column |
| `f` | Cycle process filter: all → GPU-heavy → VRAM-heavy |
| `P` | Toggle the process table |
| `r` | Rescan devices / reload config |
| `+` / `-` | Sampling interval ±100 ms (clamped 100–10000) |
| `i` | Interval picker overlay |
| `c` | Cycle colour theme |
| `d` | Dump a diagnostic snapshot |
| `h`, `?` | Help overlay |
| `Esc` | Close the top overlay |

### 11.1 Flags that are deliberately not keys

| Flag | Effect | Why a flag, not a binding |
|------|--------|-------------------------|
| `--blocks` / `--no-blocks` | Force the per-block panel on or off, overriding `blocks.enabled` | Starting and stopping a supervised child process from a keystroke would let a stray keypress spawn a process that reaches for the GPU, and would turn a whole-session resource decision into a toggle whose state nobody remembers |
| `--log PATH` | Record the session to `PATH` | A recording is a property of a session, not a view. A binding would have to open and close a file mid-run, interleaving two encoders into one path and leaving a truncated `.zst` archive behind |

`--blocks` and `--no-blocks` are a mutually exclusive group with `default=None`, so "flag absent"
is distinguishable from "flag says false": absent leaves `blocks.enabled` alone, which is what
lets a config file stay authoritative and a flag override it in either direction.

---

## 12. Configuration format

Full reference in [SPEC.md §9](SPEC.md). Loader semantics:

1. Built-in dataclass defaults.
2. Packaged default `gputop.toml`, if present.
3. `$GPUTOP_CONFIG`, else `~/.config/gputop/gputop.toml`, else `~/.gputoprc`.
4. CLI flags (highest precedence).

Implemented with `tomllib.loads`, then a `validate()` that coerces types, clamps ranges, and
collects (rather than raises) warnings for unknown keys. Hot-reload on `SIGHUP` and on `r`.

---

## 13. Test strategy

### 13.1 Principles

- **No test touches the real `/sys` or `/proc`.** Every reader takes an explicit root path, so the
  whole read layer is testable against a synthetic tree.
- **Fixtures are generated from verified ground truth**, not invented — the golden `gpu_metrics`
  blob in §13.3 was captured from real hardware alongside the sysfs values it must match.
- **Property-based checks** for the delta/clamp math, where the invariants matter more than cases.
- **Determinism**: `timestamp_ns` and the wall-clock delta are injected, never read from the clock
  inside logic under test.

### 13.2 Fake sysfs tree

Built by a `conftest.py` factory that materialises a directory tree and writes files:

```
fixtures/sysfs/
├── class/drm/card0/device/        → 0000:c1:00.0  (Zen APU / iGPU, format_revision 2)
│   ├── vendor                     0x1002
│   ├── device                     0x164e
│   ├── gpu_metrics                (binary: v2.1 APU layout)
│   ├── mem_info_vis_vram_used     268435456
│   ├── mem_info_vis_vram_total    536870912
│   ├── mem_info_gtt_used          4294967296
│   ├── mem_info_gtt_total         12884901888
│   ├── mem_info_vram_vendor       N/A
│   ├── gpu_busy_percent           17
│   ├── mem_busy_percent           4
│   ├── pp_dpm_sclk / pp_dpm_mclk
│   ├── pp_power_profile_mode
│   ├── power_dpm_force_performance_level   auto
│   └── driver -> ../../../../bus/pci/drivers/amdgpu
├── class/drm/card1/device/        → 0000:0c:00.0  (Navi 21, v1.3, WITH gpu_metrics)
├── class/drm/card1/device/        → same, but a "no gpu_metrics" variant for fallback tests
├── class/drm/card1-HDMI-A-1/      → connector node that enumeration MUST skip
├── class/hwmon/hwmon1/{name,temp1_input,...}
├── class/hwmon/hwmon7/...         (a different hwmon device, to prove device-scoped resolution)
└── bus/pci/devices/0000:0c:00.0/{vendor,device,boot_vga,class}
```

Named variants drive the behaviour matrix:

| Fixture variant | Asserts |
|------------------|---------|
| `full` | Every field populated; no `N/A` |
| `no_gpu_metrics` | Falls back to sysfs/hwmon; `metrics_abi is None` |
| `gpu_metrics_v3` (header `03 00`, junk body) | Unrecognised → fallback + footer warning; **never** a bogus parse |
| `truncated` (`structure_size` > file length) | Truncation tolerated, remaining fields `None` |
| `missing_files` | Whole attribute set deleted → all `N/A`, zero exceptions |
| `unreadable` (`chmod 000`) | `PermissionError` swallowed → `N/A` |
| `corrupt` (`busy` = `"banana"`, `mem_info` = `""`) | Value parse failure → `N/A` |
| `sentinels` (all-`0xFFFF` blob) | Every field filtered to `None` |
| `apuu` | iGPU classification → GTT-first memory panel |

The reference machine's hwmon tree has **7** hwmon devices (`nvme`, `amdgpu`, `k10temp`,
`iwlwifi_1`, `asus`, `asus_wmi_sensors`, `hidpp_battery_0`) — the fixture reproduces that shape so
the "resolve hwmon through the device, not by name" rule is genuinely tested.

### 13.3 `gpu_metrics` binary fixtures

One binary per ABI, generated by a fixture builder that packs explicit `struct` format strings
(which is also what keeps the registry honest), **plus one captured from real hardware**:

| Fixture | Size | Header | Notes |
|---------|-----:|--------|-------|
| `v1_0.bin` | 76 | `4c 00 01 00` | u32 energy acc, u8 PCIe fields, 4-byte pad before the u64 |
| `v1_1.bin` | 106 | `6a 00 01 01` | u64 energy acc, u16 PCIe fields |
| `v1_2.bin` | 116 | `74 00 01 02` | + firmware timestamp |
| `v1_3.bin` | 120 | `78 00 01 03` | + voltages ← matches the captured blob's size |
| `v2_0.bin` | 64 | `40 00 02 00` | APU layout, `system_clock_counter` first |
| `v2_1.bin` | 64 | `40 00 02 01` | reordered |
| `v2_2.bin` | 72 | `48 00 02 02` | + independent throttle status |
| `v3_0.bin` | — | `00 00 03 00` | Unknown ABI — body intentionally nonsense |
| `golden_navi44_v1_3.bin` | **120** | `78 00 01 03` | **Captured from the reference machine** |
| `truncated.bin` | 40 | `78 00 01 03` | `structure_size` 120, file only 40 bytes |
| `sentinels.bin` | 120 | `78 00 01 03` | every field `0xFF…` |
| `short.bin` | 3 | — | below header size |

**The golden fixture**, captured live, with the sysfs values read in the same instant:

```python
# tests/fixtures/gpu_metrics/golden_navi21_v1_3.bin — Navi 21, amdgpu 6.19.4, kernel 7.0.0-34
#    0x78, 0x00, 0x01, 0x03, 0x34, 0x00, 0x3b, 0x00, 0x36, 0x00, 0x2e, 0x00,
#    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00,
#    0x4c, 0x3a, 0x41, 0x03, 0x00, 0x00, 0x00, 0x00, 0xaf, 0xe9, 0x96, 0xd7,
#    0x86, 0x01, 0x00, 0x00, 0x17, 0x00, 0xff, 0xff, 0xe1, 0x03, 0x21, 0x00,
#    0x21, 0x00, 0x21, 0x00, 0x21, 0x00, 0xf4, 0x01, 0x20, 0x03, 0xe8, 0x03,
#    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
#    0x00, 0x00, 0x10, 0x00, 0x50, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
#    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
#    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x96, 0x03,
#    0x00, 0x03, 0x84, 0x03, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
# Expected decode (cross-checked against sysfs/hwmon read simultaneously):
#   structure_size=120  format_revision=1  content_revision=3   -> v1.3
#   temperature_edge       @4  = 52    (hwmon temp1_input     = 52000  -> 52.0 C)
#   temperature_hotspot    @6  = 59    (hwmon temp2_input     = 59000  -> 59.0 C, label "junction")
#   temperature_mem        @8  = 54    (NO hwmon equivalent on this card)
#   temperature_vrgfx      @10 = 46
#   average_gfx_activity   @16 = 3
#   average_socket_power   @22 = 33    (hwmon power1_average  = 33000000 -> 33 W)
#   current_gfxclk         @54 = 500
#   current_uclk           @58 = 1000  (pp_dpm_mclk current state "3" = 1000 MHz)
#   throttle_status        @68 = 0
#   current_fan_speed      @72 = 0     (hwmon fan1_input      = 0)
#   pcie_link_width        @74 = 16
#   pcie_link_speed        @76 = 80    (0.1 GT/s units -> 8.0 GT/s -> Gen 3)
#   gfx_activity_acc       @80 = 0xFFFFFFFF   SENTINEL -> None
#   temperature_hbm        @88 = 0xFFFF        SENTINEL -> None
#   firmware_timestamp     @96 = 0xFFFFFFFFFFFFFFFF  SENTINEL -> None
#   voltage_soc            @104 = 918
#   voltage_gfx            @106 = 6    -> below floor -> None
#   voltage_mem            @108 = 900
```

### 13.4 Fake `/proc` tree

```
fixtures/proc/
├── 1042/status   → Name:\tXorg
├── 1042/fdinfo/14
├── 12345/status  → Name:\tAMD family 25
├── 12345/fdinfo/{9,10,11}       ← SAME drm-client-id: exercises dedup
├── 22871/status  → Name:\tfirefox
├── 22871/fdinfo/7               ← different client-id, same pdev
├── 30114/fdinfo/3               ← drm-pdev = 0000:c1:00.0: routes to the iGPU
└── 9999/fdinfo/5                ← drm-driver: i915: must be ignored entirely
```

Each `fdinfo` fixture reproduces the **real format captured from the reference machine**
(`pos:`, `flags:`, `mnt_id:`, `ino:`, `drm-driver:`, `drm-client-id:`, `drm-pdev:`, `pasid:`,
the `drm-{total,shared,resident,purgeable}-{cpu,gtt,vram,gds,gws,oa,doorbell,mmioremap}` family,
`drm-memory-{vram,gtt,cpu}`, and the amdgpu-specific `amd-{requested,evicted}-{vram,gtt}`).

Dedup test cases:

| Input | Expected |
|-------|----------|
| Two fds, same `(pdev, client-id)`, 100 MiB + 40 MiB VRAM | One row, `100 MiB` (**max**, not `140 MiB`) |
| Two fds, same `client-id`, **different** `pdev` | Two rows, one per GPU |
| Two fds, different `client-id`, same `pdev` | Two rows (verified real: Mesa's forked compute helpers) |
| `drm-engine-gfx` delta > wall delta | Clamped to `100%` |
| First sighting, no baseline | `N/A`, not `0%` |

### 13.5 Other tests

- `test_python_compat.py` — asserts **no** `from __future__ import annotations` anywhere in the
  source tree; asserts every module imports cleanly under 3.14 with lazy annotations; asserts
  `sys.version_info >= (3, 14)`.
- `test_no_writes.py` — static: no `open(..., "w"/"a"/"+")`, no `write_text`, no `Path.write_*`,
  no `os.truncate` targeting a resolved path under `/sys`; plus a runtime guard test that installs a
  fake sysfs tree and asserts the tree's file mtimes and contents are byte-identical after a full
  sample cycle.
- `test_sampler.py` — runs a real `Sampler` against fake trees with an injected clock; asserts the
  queue receives frozen snapshots, that the snapshot is *not* mutated by the sampler afterwards
  (identity check on the tuple), and that `stop()` returns promptly.
- `test_format.py` — byte/KiB/MiB/GiB scaling, `N/A` rendering, temperature °F, bar widths at
  widths 60/80/100/120, and the `999.9+` saturation convention.
- `test_ui_smoke.py` — Textual `run_test()` pilot: mount with a synthetic snapshot, assert key
  bindings switch GPU, cycle sort, toggle panels, and that no widget raises at any terminal size.
- `test_config.py` — TOML parsing, precedence, clamping, unknown-key warnings, hot reload.
- `test_power_tables.py` — `pp_od_clk_voltage` in **both** eras through one entry point, the
  `OD_RANGE`/`OD_VDDGFX_OFFSET` shapes, bar ratios, and the module's own read-only proof: the
  no-write-call source scan, plus a read that leaves the fixture byte-identical.
- `test_radeontop.py` — dump lines from cards that omit `tc`/`smx`/`cr` parse by field name with
  those blocks *absent* rather than shifted; argv construction for both `-p` and `-b`; the
  interval clamp; stderr classification ordering (a permission failure must not be reported as
  "unsupported"); restart bounds.
- `test_alerts.py` — hottest-sensor-wins, `None` never alerts, a threshold of `0` disables only
  its own check, and the same condition crossing and clearing leaves no residue.
- `test_sessionlog.py` — suffix rules including `.zst`, append/header seeding against a
  pre-existing file, empty cells for absent readings, round-trip through `read_log`.
- **CI matrix:** `uv run pytest` on CPython 3.14; `uv run --python 3.14t pytest`; `uv run ruff check`;
  a `python3.14 -X importtime` budget check; and `mypy --strict` once type stubs settle.

---

## 14. Phased implementation plan

> **Delivery status is tracked in [PROGRESS.md](PROGRESS.md)**, which is updated at the end
> of each milestone. It records what was verified against real hardware versus what is
> still an assumption — see its Verification ledger.

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0. Foundations** | `pyproject.toml` (`requires-python = ">=3.14"`), `uv` lock, ruff + pytest config, package skeleton, `model/aliases.py`, `model/` dataclasses, CI matrix | `uv run pytest` runs; static tests enforce PEP 649 and no-`__future__` |
| **1. Safe I/O + discovery** | `fsutil` (total read primitives), `discovery`, `hwmon`, `dpm`, `throttle`, fake sysfs fixtures | Enumeration skips connector nodes, resolves hwmon through the device, classifies iGPU/dGPU; every fixture variant yields `N/A` without raising |
| **2. `gpu_metrics`** | ABI registry with explicit padding, sentinels, plausibility filters, per-version binary fixtures + the golden blob | All 12 fixtures parse as specified; golden blob matches the captured cross-check table; unknown `3.x` falls back safely |
| **3. Per-process** | `/proc` scanner, `(pdev, client-id)` dedup, engine deltas, engine-name normalisation, fake `/proc` fixtures | Dedup matrix passes; clamp and first-sighting `N/A` pass |
| **4. Sampler** | Thread, queue handoff, stats, device hotplug re-scan, stop responsiveness | No shared mutable state (asserted); sampler survives an injected exception; free-threaded suite green |
| **5. UI v1** | Textual app, header, graphs, memory bars (GTT-first on APU), sensors, process table, footer, overlays, themes, formatting | Wireframe reproduced; TUI tests pass; renders sanely from 60 to 200 columns |
| **6. Merge + diagnostics** | Source-precedence merge, `--dump`, `--dump-metrics`, `compression.zstd` dumps, `SIGHUP` reload | `--dump` reproduces every TUI value; SPEC §11 acceptance criteria all pass |
| **7. Polish + docs** | README, device-name table, `--once` headless mode, performance tuning, release packaging | README quick-start; 24-hour soak with flat memory; first frame < 500 ms |

**Suggested sequencing rationale:** phases 1–3 are the technically risky part (kernel ABI reverse
engineering), so they come first and are fully testable without a UI. The UI is deliberately last
because the data model's shape is only settled once the readers are real.

---

## 15. Risks and open items

| Risk | Impact | Mitigation |
|------|--------|------------|
| `gpu_metrics` `3.x` layout unknown (`UNVERIFIED`) | Misses richest metrics on newest kernels | Safe fallback + `--dump-metrics` to collect samples; registry is one line to extend |
| Upstream is actively rewriting the metrics encoding | Possible future ABI break | Version registry keyed on `(format, content)`, never on struct assumptions; unknown ⇒ fallback |
| Cross-user process visibility needs root | Partial process table | Explicit footer disclosure; rootless operation remains fully useful for device metrics |
| `current_sclk`/`current_mclk` absent on newer drivers | Clock `N/A` if `gpu_metrics` is also absent | Three-level fallback (§7.1), including `pp_dpm_*` `*`-row parsing |
| `radeon` driver (legacy, non-`amdgpu`) | Not supported | Explicit driver check; the UI states the requirement rather than failing obscurely |
| APU `v2.x` layouts verified only from a reference decoder, not live hardware (`UNVERIFIED`) | Possible APU field mis-mapping | Fixtures pin the layout; the `--dump` cross-check makes mismatches reportable |
| Textual's free-threaded readiness | May block `3.14t` CI | gputop's own modules tested strictly; dependency-specific xfails isolated |
| Device-name table ages | Cosmetic "unknown device" names | Config override `[gpu.names]` |
| `radeontop` absent, or present but unprivileged | Per-block panel unavailable on many machines | Off by default (§7.5); five distinct states so the reason is always specific; every other panel is unaffected and gputop itself never gains a privilege |
| A third format appears in `pp_od_clk_voltage` | Overdrive panel misreads one generation | Unit carried per entry and sections detected by shape, so an unrecognised section degrades to `N/A` rather than to a wrong number |
| An interrupted session loses its recording | Partial CSV/JSONL file | Append-only, JSON Lines rather than an array, flush per record — the cost of a crash is one row, and an unopenable target degrades to a notice rather than a failed start (§7.5) |
