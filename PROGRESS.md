# gputop — progress log

**Convention:** this file is updated at the end of every milestone. Each entry records what
was built, what was *verified* (and on what), what was assumed, and what was left undone.
The Verification ledger below is the part that matters: it separates properties we have
actually measured from properties we merely designed for.

Phase numbers refer to the plan in [DESIGN.md §14](DESIGN.md).

---

## Milestone 1 — data layer

**Scope:** DESIGN phases 0–4. `gpu_metrics` ABI reverse engineering, sysfs/procfs readers,
device discovery, per-process accounting, sampler thread, `--dump`.

**Delivered**

- `model/` — frozen dataclasses, PEP 695 type aliases, no I/O, no Textual.
- `readers/` — `fsutil` (total read primitives), `discovery`, `hwmon`, `dpm`, `throttle`,
  `metrics_bin` (v1.0–v2.2 ABI registry), `procfs` (per-client accounting).
- `sampler/` — background thread, `queue.SimpleQueue` handoff, ring-buffer history.
- `cli.py` — `--dump`, `--devices`, `--kind`, `--pretty`.
- `tests/` — synthetic sysfs and procfs trees built in `tmp_path`.

**Verified**

- `gpu_metrics` v1.3 decoded against a **live Navi 21 / RX 6800** (`amdgpu` 6.19.4, kernel 7.0.0-34).
  Byte offsets and the 120-byte `structure_size` were captured from hardware and pinned as
  literal expectations, so the aligner cannot rewrite its own test.
- Per-process accounting against live `/proc`: engine deltas, `(pdev, client-id)`
  deduplication, IEC (`KiB`) memory units.
- Kernel quirks handled after being hit on the real machine: `current_sclk` absent on
  recent drivers, DPM units written `Mhz` with a lowercase `h`, `drm-engine-*` counters
  carrying an `ns` suffix, `drm-pdev` matching the discovered BDF.

**Bugs found and fixed during this milestone** (all by tests or by comparing against the
real machine, not by inspection):

| Symptom | Cause |
|---|---|
| `calcsize` 116, not 120 | Aligner started at offset 4 but never emitted the header bytes |
| `sclk` max always `None` | DPM regex expected `MHz`; kernel writes `Mhz` |
| Zero GPU clients found | DRM node matched with `endswith` against a prefix |
| Whole sample raised | Engine counters carry a trailing ` ns` unit |
| Per-process memory always `None` | Unit table had `kb`; kernel writes `KiB` |
| Mutilated BDF `03.1:.0:0000:0b:00.0` | String slicing instead of the resolved path name |

---

## Milestone 2 — Textual UI

**Scope:** DESIGN phase 5, plus the sampling concurrency fix the graphs required.

**Delivered**

- `BrailleGraph` — reusable 2×4-dot rasterised graph, shared by the utilisation, VRAM/GTT
  and sensor plots. Gaps are preserved rather than interpolated.
- `theme.py` — `default` / `dracula` / `gruvbox`, runtime-switchable and config-driven.
  Truecolor uses a 255-stop ramp; the 256-colour path snaps to the xterm 6×6×6 cube so the
  gradient does not band.
- `panels.py` — header, utilisation, memory, sensors, footer. Every row sizes its bar from
  the measured value text, so nothing wraps out of its panel.
- Responsive layout: panels stack below 72 columns, the process table yields below 20 rows,
  the memory graphs are hidden below 28 rows. The utilisation graph is never dropped.
- `CpuMeter` — the app's own CPU usage at 1 Hz, shown in the status line.
- `--theme`, `--no-color`; `[ui]` config section; `gputop.example.toml`.

**Data-layer change (one):** `Sampler.history()` iterated the ring buffer's `deque` while
the sampler thread appended to it. Readers now receive an immutable tuple rebound each tick.

**Verified**

- 348 tests, `ruff` clean, `mypy --strict` clean.
- Live on the Navi 21 / RX 6800: real braille plots, per-client table, theme cycling, `--no-color`.
- **Free-threaded (3.14t), genuinely no-GIL** (`sys._is_gil_enabled() is False`):
  - full 348-test suite passes;
  - 3.4 M concurrent `history()` reads across 4 threads while the sampler mutates it,
    with 0 tick errors.
- UI thread performs no filesystem I/O — asserted by a test that wraps the reader functions
  and fails on any main-thread call while deliberately ignoring the sampler thread's.

**Bugs caught by tests rather than by looking at the screen**

| Symptom | Cause |
|---|---|
| Graphs looked plausible but were wrong | Rasteriser emitted no row separators; one long strip was being *wrapped*, smearing the series instead of plotting it |
| `500 MHzMemory` — values ran into the next row | A Rich `Group` is not a layout container; rows bled instead of stacking |
| Resizes silently ignored | `on_resize` declared an argument Textual never passes |
| `cpu 139%` while idle | First sample measured a microsecond-wide window |
| Narrow terminal lost the graphs | Sensors panel sized to content and pushed them off-screen |

---

## Verification ledger

| Property | Status | Evidence |
|---|---|---|
| `gpu_metrics` v1.3 layout | **Verified** | Live Navi 21; offsets pinned as literals in tests |
| `gpu_metrics` v1.0–v1.2 | **Assumed** | Reconstructed from upstream decoders; no hardware reports them |
| Silicon attribution for a given device id | **Was wrong, now fixed** | `0x73BF` is Navi 21 (RDNA2); the table had claimed Navi 44 (RDNA3) |
| `gpu_metrics` v2.x (APU) | **Assumed** | Reconstructed from a reference decoder. **No APU available to test against** |
| `gpu_metrics` v3.x | **Unsupported by design** | Unknown ABIs return `None` and fall back to sysfs/hwmon |
| `indep_throttle_status` @112 | **Inferred** | Only way to reach the declared 120-byte size; reads 0, unused |
| Engine deltas, dedup, `/proc` accounting | **Verified** | Live machine |
| Free-threaded correctness | **Verified** | 3.14t suite + concurrency stress |
| Graceful degradation to `N/A` | **Verified** | Fixture matrix; every reader is total |
| Responsive layout 40–140 cols | **Verified** | Headless screen tests |
| Rendering >200 cols | **Untested** | No terminal that wide available |
| Multi-GPU ordering, dGPU+APU | **Fixture-verified only** | One physical GPU on this machine |
| First-frame latency <500 ms, 24 h soak | **Not measured** | DESIGN phase 7 exit criteria |

---

## Known gaps

Deliberately not built yet, in rough priority order:

1. **No APU validation.** The v2.x layout is the largest single risk: on an APU, a field
   mis-mapping would report plausible but wrong numbers rather than failing.
2. **No device hotplug re-scan.** `Sampler` does not notice a GPU appearing or vanishing;
   the device list is fixed at `start()`.
3. **No diagnostics beyond `--dump`.** `--dump-metrics` (to collect an unrecognised ABI
   from a new kernel), `zstd` dumps and `SIGHUP` reload are all in DESIGN phase 6.
4. **No logging subsystem.** DESIGN §152 sketches a `logsetup.py` with an optional
   `compression.zstd` ring handler; it does not exist.
5. **No `--once`**, and no overlays (DESIGN phase 7).
6. **Cross-user process visibility needs root.** Disclosed in the footer; no root path
   exists, by design.
7. **No packaging or release.** No wheel build has been tested.

## Notes for future milestones

- `BRAILLE_BITS` is the single source of truth for the dot layout; the usual braille
  off-by-one is dots 1–3 and 7 forming the *left* column, not a linear grid.
- Panel rows take `available` columns, not a bar width. Passing a fixed width is what caused
  the overflow; `fit_bar_width` derives it from the measured value string.
- Tests await the app's own tick count and a stable layout rather than sleeping; `pilot.pause()`
  alone does not mean "mounted", and that produced intermittent `NoMatches` failures.
- The rasteriser is a pure function (`braille_text`) precisely so it can be tested at
  explicit dimensions without mounting a widget.
