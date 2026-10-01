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

## Milestone 3 — per-process panel and interactivity

**Scope:** SPEC FR4/FR5, the interactive half of DESIGN phase 5, and session persistence.
`r` changed meaning; `R` is the new force-sample key.

**Delivered**

- `ui/procview.py` — the table's entire decision-making as pure functions: the column
  set, `SortState`, `FilterState`, the parent/child tree, and the row/cell builders. It
  imports no Textual, which is what makes the interesting behaviour testable without an
  event loop.
- `ui/proctable.py` — the widget. Sortable headings, mouse selection, the `▶` marker on
  the busiest client, and a selection tracked by `(drm-pdev, drm-client-id)` so it stays
  on the same *process* as the table reorders.
- `ui/devices.py` — a clickable device tab bar and the compact all-GPUs overview.
- `ui/overlays.py` — the filter prompt, the signal confirmation, the key reference.
- `control.py` — the one place that can end a process, and the pure rules about when not to.
- `state.py` — session state in `~/.config/gputop/config.toml`: read with `tomllib`,
  written with `tomli-w`, atomically, touching nothing but the `[state]` section.
- Reader: `PPid` and `cmdline` per client, which is what makes the tree, the full command
  line and the kernel-thread rule possible.
- Tests: 86 for the view logic alone, plus 37 for persistence, 19 for signalling, 6
  toolchain canaries and 83 driving the real app.

**Verified**

- 579 tests, `ruff` clean, `mypy --strict` clean, three consecutive clean full runs.
- The whole suite also passes on **3.14t, genuinely no-GIL** (`sys._is_gil_enabled()` is
  `False`), which is the build the sampler/UI split exists for.
- Live-render checks against a synthetic dGPU+APU tree: GPU switching scopes the table by
  `drm-pdev`, the memory panel leads with GTT on the APU, the overview shows both cards,
  the tree nests, the filter narrows, and the signal dialog names its target.

**Bugs caught by the new tests rather than by looking at the screen**

| Symptom | Cause |
|---|---|
| A click on a process row was swallowed about one time in ten | The table was rebuilt from scratch on every sample; a click landing mid-rebuild hit an empty grid. The rebuild is now skipped when nothing changed |
| A key press that changed the view appeared to do nothing | Switching GPU or view called `_refresh`, which only repaints if a *new* sample has arrived — up to a second later. Those actions now repaint from the snapshot in hand |
| `NoMatches: '#status'` while the app was closing | Textual closes the screens *before* it dispatches `Unmount` and only then stops the timers, so a 1 Hz timer could fire into a torn-down screen. `_drawable()` brackets the window in which the widgets exist |
| The signal dialog's key hints were not visible | A `Horizontal` between two Statics filled the height it was given and pushed them off the bottom of a box that had already wrapped |
| The first run left no state file | State was only written when something changed, so "the user never chose a theme" and "the file was never created" were indistinguishable |

**Toolchain: a CPython codegen hazard, found and characterised**

`CPython 3.14.6` was observed dropping `BUILD_TUPLE 1` for a single-element tuple written
as a call and passed straight to a keyword argument inside a list display:
`proc(..., engines=(engine(...),))` arrived as `engines=engine(...)`. It is silent, it is
data-dependent, and neighbouring calls in the same list compiled correctly. It is
reproduced in a standalone file and avoided by building such tuples through a function
call, which cannot be flattened; `tests/test_toolchain.py` pins the behaviour so a future
interpreter change is reported as a canary failure rather than as corrupt data somewhere
else. **Nothing in `src/` uses the affected shape**, and the whole suite exercises the code
paths either way.

---

## Milestone 4 — per-block panel, power profile, alerts and session logging

**Scope:** the four features that are *opt-in*: the radeontop-backed blocks panel, the
read-only power tables, threshold alerting, and session recording. None of them changes
what the program does for a user who has asked for nothing, which is the property they
share and the reason they landed together.

**Delivered**

- `readers/radeontop.py` — one supervised `radeontop -d - -i N -l 0 -t T` child per GPU,
  started only when `[blocks]` asks for it. Dump lines are parsed **by field name, not by
  position**: `dump.c` emits `tc`, `smx`, `cr`, `uvd`, `vce0`, `vram`, `gtt` and the clocks
  only when the card reports those bits, so a positional parser reads `vgt`'s value as
  `ta`'s on every modern card and reports confidently wrong numbers. stdout and stderr each
  get their own drain thread (an undrained pipe fills and freezes the child), SIGTERM is
  tried before SIGKILL, pipes are closed explicitly rather than left to the collector, and
  restarts are bounded at two and only for transient failures.
- `model/blocks.py` — the fourteen pipeline blocks in the order a triangle travels through
  the pipe, UVD and VCE as a separate group, the memory and shader clocks, and
  `BlocksStatus` with six members rather than one `unavailable`: "install a package" and
  "you need root" are different problems with different remedies.
- `readers/pp.py`, `model/power.py` — `pp_power_profile_mode` and `pp_od_clk_voltage`,
  section-aware and unit-carrying, because `pp_od_clk_voltage` held a voltage table on old
  parts and per-domain overdrive ceilings on RDNA2 and later, in the same attribute.
  `assert_read_only` makes the read-only promise checkable rather than a comment.
- `ui/panels.py` — `blocks_panel`, `blocks_heading` (the busiest block goes in the panel's
  title, because "which block" is the question and the bars are the evidence),
  `power_profile_panel`, and the blocks hint in the status line.
- `model/alerts.py`, `ui/app.py` — thresholds recomputed from scratch every sample, so a
  condition that clears stops being reported the moment it clears and the border can only
  flash because something is wrong right now. One timer drives one shared border colour
  for every panel and runs whether or not anything is breached.
- `sessionlog.py` — CSV, JSON Lines, or either under zstd, all three chosen from the
  filename. Append-only and flushed per record, because an interrupted session leaving a
  readable file is worth more than a tidier one that loses everything.
- Config `[blocks]`, `[alerts]`, `[log]`; CLI `--blocks`, `--no-blocks`, `--log`. `--dump`
  now carries a top-level `blocks` status and `alerts` section, and per device `blocks`,
  `odc` and `power_profiles`.

**Verified**

- 918 tests pass, `ruff` clean over `src/` and `tests/`, `mypy --strict` clean over 39
  source files. The count moved during this milestone (891 → 918) as the last of the
  feature tests landed; the figure here is the one at the final full run.
- 322 of those tests exercise this milestone's code: 113 alerts, 69 session log, 55
  radeontop, 34 blocks panel, 28 power tables, and 23 for the extra row and the alert
  borders. The rest — the whole of Milestone 3's suite — pass unchanged.
- Supervision is exercised against **real subprocesses**: a stand-in binary is launched,
  polled, restarted and terminated, and the tests assert nothing is left behind and that
  the pipes are closed.
- The failure classifier is checked against every message radeontop prints, in the order it
  prints them — radeontop reports "Failed to find DRM devices", then "Failed to open DRM
  node", then finally "Cannot access GPU registers, are you root?", and classifying on
  stdout EOF alone would report the least actionable of the three.
- `assert_read_only` plus a test that scans `readers/pp.py` for write operations.
- The flag contract is checked against the *installed* radeontop rather than assumed:
  `test_the_parser_accepts_the_system_radeontop_version_if_present` runs `--help` and
  requires `-d`, `-i` and `-l` to exist. It is skipped, not assumed, where radeontop is
  absent.
- Live on the RX 6800 development machine: `gputop --blocks --dump` started the child,
  classified the failure and exited cleanly with
  `"blocks": {"status": "unsupported", "hint": "radeontop cannot read this card"}` — no
  exception, no crash, and the unprivileged run's three-line failure is exactly the case
  the degradation states exist for. (Why this machine says `unsupported` rather than
  `not permitted` is a real finding; see Known gaps.)
- `--dump` on the same machine reports a live alert (`vram 94% > 92%`), so the thresholds
  in `[alerts]` are doing something to real readings and not only to fixtures.


---

## Milestone 5 — finishing: diagnostics, packaging, CI, and a measured cost

**Scope:** everything between "the feature works" and "someone else can install it and
trust it". `gputop --check`, the man page, the shell completions, the Arch `PKGBUILD`, the
CI workflow, pinned dependencies, and the profiler whose numbers are now in the README.

### Delivered

- **`check.py` and `gputop --check`** — a report of which metrics this machine can provide
  and why any are missing, built from a *real sample* rather than a list of expected
  filenames, so what it says is what the interface would draw. `--json` for the same report
  as data. Eight distinct reasons (`not published`, `needs a newer kernel`, `not on this GPU
  generation`, `permission denied`, `unknown gpu_metrics version`, `not installed`, `turned
  off`, `no GPU clients`), each derived from something observed — an `errno`, a kernel
  version compared against a documented minimum, or the shape of the device. Never starts
  `radeontop`; exits 3 when there is no AMD GPU, like `--dump` and `--devices`.
- **`packaging/`** — `man/gputop.1`, completions for bash, zsh and fish, and an Arch
  `PKGBUILD` that builds the wheel and installs the command, the man page and all three
  completions.
- **`.github/workflows/ci.yml`** — ruff (check *and* format), mypy with an assertion that
  strict mode is still on, pytest on 3.14, a smoke test of `--check`/`--dump` exit codes,
  a wheel job that installs the built wheel and runs the installed command, and a
  non-blocking free-threaded (3.14t) job.
- **Pinned dependencies**, each with the reason recorded next to it in `pyproject.toml`.
- **`tools/profile_gputop.py`** and **`tools/make_screenshots.py`**, and the CPU/memory
  section of the README that is their output.

### Bugs this milestone found, by measuring rather than by looking

| Symptom | Cause |
|---|---|
| A 1000-process `/proc` scan took 25 ms and was the single largest cost in the program | The scan built a `pathlib.Path` per open descriptor and called `.readlink().as_posix()` on it. Rewritten against the `os` string APIs: **25.3 ms → 11.5 ms**, and 105 ms → 47 ms at 4000 processes |
| A second `scandir` per GPU client, every tick | The scan listed `fdinfo/` to check a descriptor had accounting data. The read itself answers the same question — `ENOENT` means the same thing — so the listing is gone |
| Each clock cost four sysfs reads where two suffice | `current_sclk`/`current_sclk_max` are absent on current drivers, so `read_clock` and `read_clock_max` each fell through to the DPM table. One `read_clock_pair` reads it once |
| The braille rasteriser spent ~1.8 ms per graph on a full panel | It painted every sample's fill separately. Folding samples into their dot column first is the same output for half the work: 1.76 ms → 1.05 ms at 58×10, 3.44 ms → 1.91 ms at 60×20 |
| `gputop -i 0.5` sampled twice as often and still redrew once a second | The app read the interval from the config instead of from the sampler options the CLI had already merged into |
| A reply to a key press ("SIGTERM → 4211 blender") was gone before it could be read | The status line cleared it on the *next sample*, which is a second at the default interval and a tenth of one at the 100 ms the `-` key reaches. It now has a fixed 1.5 s lifetime — and two tests that raced the sample that erased the text stopped failing on a fast machine |
| Every panel was repainted on every sample, to arrive at the layout that was already there | `_apply_alert_borders` and `_apply_responsive_layout` re-asserted inline styles unconditionally. Both are now keyed on whether the value actually changed |

The last two were found by profiling, and neither was visible on the screen.

### Verified

- **1026 tests**, `ruff check` and `ruff format --check` clean, `mypy --strict` clean over
  40 source files. The suite passes on **3.14t** (no-GIL, `sys._is_gil_enabled()` is
  `False`) and on the **standard** 3.14 build — both are CI jobs.
- **The wheel builds, installs and runs.** `uv build` → `uv tool install` → `gputop
  --version`, in a clean standard-build 3.14 environment, with `textual 8.2.8`,
  `rich 15.0.0` and `tomli-w 1.2.0` resolved from the pins.
- **The `PKGBUILD` was executed, not read.** Its `build()` and `package()` steps were run
  against a stand-in release tree under `set -euo pipefail`: the wheel built, the package
  was staged, the generated console script printed `gputop 0.1.0`, and the staging step
  imports the copy in `$pkgdir` rather than the installed one.
- **The man page and the completions are checked against the parser.** `tests/test_packaging.py`
  derives the option list from `build_parser()` and fails if the man page, any completion
  file, the `PKGBUILD` or the README has not caught up — including a check that no
  completion offers an option the program rejects, and that every declared key binding is
  documented.
- **`--check` on the live Navi 21**: 16 of the 17 metrics it reports available (the
  seventeenth is the blocks panel, which is off), `gpu_metrics` v1.3 decoded, and the
  system-wide rows correct for a container with no `/dev/dri` and a five-process `/proc`.
  A synthetic APU with nothing but identity files reports 2 of 18, with every absence
  explained.
- **Python 3.14 support of every dependency, from PyPI metadata** rather than assumed:
  textual 8.2.8 (3.14 support landed in 6.3.0, PR #6121; issue #5797 was the real
  `App.run()` breakage), rich 15.0.0 (3.14 fix in 14.2.0, PR #3861), tomli-w 1.2.0,
  pytest 9.1.1, mypy 2.3.1 (first release with compiled `cp314t` wheels), ruff 0.16.9,
  hatchling 1.32.4. Textual and Rich ship only `py3-none-any` wheels, so there is no
  `cp314t` wheel to wait for and no C extension whose GIL-free status has to be trusted.
- **Cost, measured** (24-core x86-64, Navi 21, kernel 7.0, 1 s interval): sampler tick
  2.7 ms, of which one device is 1.8 ms and the `/proc` scan 0.2 ms at five processes and
  ~12 µs/process beyond; the running interface at 120×40 costs ≈4 % of one core, about a
  quarter of it gputop's own code; peak RSS ≈78 MiB with the interface, ≈35 MiB for the
  sampler alone. The numbers and the commands that reproduce them are in the README.
- **Screenshots are generated, not drawn.** `tools/make_screenshots.py` runs the real
  interface headlessly over a deterministic synthetic two-GPU machine with a simulated
  workload (the engine counters are advanced from the clock, because a fixture whose
  counters never move renders a table of 0.0% that is true of the fixture and a lie about
  the program) and exports Textual's SVG.

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
| Process table sorting, filtering, tree, GPU switching | **Verified** | 86 logic tests + 83 against the running app, on a synthetic dGPU+APU tree |
| Session state round trip | **Verified** | 37 tests: merge, atomicity, permissions, failure reporting, stale values |
| Signal confirmation | **Verified** | 19 tests; the refusals are pure, delivery goes through a spy |
| Signal *delivery* to a real client | **Untested** | No GPU client was killed during development; only the refusal rules are exercised |
| Per-block parsing from radeontop dump lines | **Fixture-verified only** | 55 tests over literal dump lines; no run on real hardware has produced a block sample |
| radeontop child supervision | **Verified** | Real subprocesses against a stand-in binary: start, bounded restart, SIGTERM then kill, pipes closed, nothing left behind |
| radeontop's `-d`/`-i`/`-l` flags | **Verified** | Asked of the installed radeontop; skipped when it is absent |
| The four blocks degradation states | **Partial** | All four asserted on synthetic stderr; on this machine the not-permitted case classifies as *unsupported* — see Known gaps |
| `pp_power_profile_mode` / `pp_od_clk_voltage` parsing | **Fixture-verified only** | 28 tests over inlined captures of real files; both attributes are absent from this container's card |
| Read-only guarantee for the power tables | **Verified** | `assert_read_only` plus a source scan for write operations |
| Alert evaluation: hottest sensor, missing readings, zero thresholds, every GPU | **Verified** | 113 tests |
| Alert border flash, including `flash_hz = 0` | **Verified** | Headless app tests: the border alternates, holds steady at 0, and stays on the theme accent otherwise |
| Session log round trip and interruption safety | **Verified** | 69 tests: CSV/JSONL/zstd, header written once, flush per record, unknown suffix, failure mid-file |
| `log.interval_s` | **Not implemented** | Parsed and range-checked; nothing reads it. See Known gaps |
| `--blocks` / `--no-blocks` / `--log` | **Verified** | CLI tests, including flag-over-config precedence and `--dump` carrying the status |
| CPython 3.14.6 tuple-flattening codegen | **Observed** | See Milestone 3; avoided in code, pinned by a canary |
| First-frame latency <500 ms, 24 h soak | **Not measured** | DESIGN phase 7 exit criteria |
| `gputop --check` reasons | **Verified** | Live Navi 21 (16/16 metrics available) and a synthetic APU with nothing published (2/18, every absence explained); each reason asserted in tests |
| `/proc` scan cost | **Verified** | 12 µs/process, measured 5→4000 processes against synthetic trees |
| Interface CPU and RSS | **Verified** | `tools/profile_gputop.py render`/`memory`, 20 s runs, repeated |
| Rasteriser output after the per-column rewrite | **Verified** | Byte-identical to the per-sample reference across 6 sizes × 2 scales × 34 series shapes |
| Python 3.14 support of every dependency | **Verified** | PyPI metadata and upstream release notes; both builds' suites pass |
| Wheel builds, installs and runs | **Verified** | `uv build` → `uv tool install` → `gputop --version` in a clean standard-build 3.14 venv |
| `PKGBUILD` correctness | **Verified by execution** | `build()` and `package()` run under `set -euo pipefail` against a stand-in tree; five artifacts staged, console script runs |
| Man page / completions / PKGBUILD match the parser | **Verified** | `tests/test_packaging.py` derives the option list from `build_parser()`; 50 tests |
| Free-threaded (3.14t) support of Textual/Rich upstream | **Not claimed** | Neither project runs a free-threaded CI job. gputop's own suite passes on 3.14t; that is this project's evidence, not theirs |
| Keyboard shortcuts and mouse behaviour on a real terminal | **Untested** | Exercised headlessly through Textual's pilot; no human has pressed a key |
| `gputop --check` on hardware other than Navi 21 | **Untested** | No GCN, RDNA1/3/4 or APU has been available |
| Kernel-version table in `ATTRIBUTE_MIN_KERNEL` | **Verified where listed** | Each entry established by diffing the driver's attribute tables across release tags; see the corrections below |
| `pp_power_profile_mode` / `pp_od_clk_voltage` = Linux 4.17 | **Verified** | Rex Zhu, "drm/amdgpu: add custom power policy support in sysfs", 2018-02-19 |
| `gpu_metrics` = Linux 5.10 | **Verified** | Absent in `drivers/gpu/drm/amd/pm/amdgpu_pm.c` at v5.9, present at v5.10, continuously through v7.2 |
| DRM fdinfo `drm-client-id` / `drm-engine-*` / `drm-memory-*` = Linux 5.19 | **Verified** | Absent in `drm_file.c` at v5.18, present at v5.19; `drm-usage-stats.rst` landed the same release |
| ROCm visibility through DRM fdinfo rather than `/sys/class/kfd` | **Verified from source** | amdgpu's engine table includes `AMDGPU_HW_IP_COMPUTE -> "compute"`; KFD publishes static topology and no per-process utilisation |

---

## Known gaps

Deliberately not built yet, in rough priority order:

1. **No APU validation.** The v2.x layout is the largest single risk: on an APU, a field
   mis-mapping would report plausible but wrong numbers rather than failing. The iGPU in
   the multi-GPU fixtures is synthetic for the same reason.
2. **No device hotplug re-scan.** `Sampler` does not notice a GPU appearing or vanishing;
   the device list is fixed at `start()`.
3. **No diagnostics beyond `--dump`.** `--dump-metrics` (to collect an unrecognised ABI
   from a new kernel), `zstd` dumps and `SIGHUP` reload are all in DESIGN phase 6.
4. **No log rotation, and no diagnostics for gputop itself.** `--log` records a session to
   CSV or JSON Lines, optionally zstd-compressed, but DESIGN §152's `logsetup.py` with an
   optional `compression.zstd` ring handler does not exist. A failure *inside* gputop is
   still reported only on the status line and in `--dump`'s warnings; there is no file for
   it, and nothing rotates one.
5. **No `--once`**, and no GPU/interval picker overlays. The filter prompt, the signal
   confirmation and the help overlay exist; the dedicated pickers in DESIGN §11 do not.
6. **Cross-user process visibility needs root.** Disclosed in the footer; no root path
   exists, by design.
7. ~~**No packaging or release.**~~ Closed in Milestone 5: the wheel builds, installs and
   runs, and the `PKGBUILD` was executed. There is still no tagged release, no changelog
   and no publish step; the Arch recipe points at a placeholder URL and a `SKIP` checksum
   until a release exists to point at.
8. **No overlays for GPU or interval picking.** `o` shows every GPU and `+`/`-` change the
   interval; the dedicated pickers in DESIGN §11 are not built, and neither is `SIGHUP`
   reload.

Added by Milestone 5, in the order they will be felt:

12. **`--check` explains absences from a partly established version table.** Every entry in
    `ATTRIBUTE_MIN_KERNEL` has now been tied to a release by diffing the driver's attribute
    tables, but the list is not exhaustive: an attribute absent from it always reports "not
    published" rather than "needs a newer kernel". That direction is safe -- it never claims
    a metric is available when it is not -- but it is less helpful than the text could be.
    Four version claims were checked and found *unverifiable* rather than wrong, and are
    deliberately absent from the README: RDNA3 (commonly 6.2), MI300 (6.8), RX 9000 (6.14),
    MI200 full support (5.15/5.16).
13. **The kernel-version and generation claims in the README are from documentation, not
    from hardware.** GCN 1-4, RDNA 1-4 and the APUs of each era are supported in the
    sense that gputop reads generation-neutral paths and degrades per metric; only RDNA2
    (Navi 21) has actually been run against.
14. **A composite-APU or UMA carve-out quirk would not be noticed.** The APU classification
    votes on device id, VRAM vendor, carve-out size and bus number; a machine that gets
    all four wrong is reported as a discrete GPU with a small VRAM pool rather than as an
    APU leading on GTT.

Added by Milestone 4, in the order they will be felt:

9. **Blocks is off by default, and needs privileges gputop itself does not.** Most users
   run unprivileged, where radeontop answers "are you root?", so a default-on panel would
   put a permanent complaint in the status bar of everyone who cannot use it. Opting in
   (`--blocks`, or `blocks.enabled = true`) is the user asserting that radeontop works for
   them; gputop itself still never requires root.
10. **No live per-block numbers yet.** The parser, the supervisor and the panel are
    verified against literal dump lines and a stand-in child; nothing has run on hardware
    that actually produced a block sample, so no block label, ordering or percentage has
    been compared against radeontop's own display. Everything downstream of the parse is
    exercised end to end — the Navi 21 in the reference machine drives the overdrive and
    profile panels with real values — but the block percentages themselves are still
    unconfirmed against the tool they are copied from.
11. **radeontop's dump interval is whole seconds only.** `-i` is parsed with `atoi` and
    floored at 1 s, so a sub-second panel cannot be asked for; `blocks.interval_s` clamps
    with a warning rather than letting gputop honour a request the child ignores.

## Closed during Milestone 5

Four claims this project had been carrying turned out to be wrong, and were corrected after
checking the driver's own attribute tables across release tags rather than the summaries
that produced them:

| Was | Is | Consequence |
|---|---|---|
| `gpu_metrics` needs Linux 6.10 | It was registered in **5.10** | `--check` was blaming a six-year-old gap where there was none |
| `current_sclk` / `current_mclk` / `current_link_*` are documented sysfs attributes that recent drivers dropped | They are fields of the internal `struct amdgpu_dpm` and appear in **no** in-tree release (checked v5.4 and v7.2) | The probe stays (one failed `open()` is cheap), but the comment no longer tells a reader to expect them back |
| `gpu_metrics` v2.x is the APU layout | v2_1 is returned by Navi 23/24 dGPUs; only v2_2+ are APU-shaped, and RDNA3 reports v3_0 | The README describes the revision as what it is -- a header field -- rather than a hardware class |
| `drm-engine-active-*` / `drm-engine-period-*` are what newer kernels report | They do not exist in any kernel; amdgpu emits only `drm-engine-<name>` | The parser still accepts them, now as defensive handling of a shape no in-tree driver produces, and says so |


- **`--check` was going to be a list of expected filenames.** The first draft checked for
  the presence of files and printed a table of what it expected to find. A report that
  describes the design rather than the machine is wrong in exactly the case it is for: a
  driver that renames an attribute, or a card that publishes an attribute gputop has no
  reader for. It is built from a real sample, and only the *absences* are explained.
- **A `permission denied` reported as `not published`.** `_explain` probed each attribute
  lazily and re-opened the same file to ask a second question, so an unreadable file was
  classified as an absent one. Every candidate is now probed once and the answers are
  compared in the order of what the user can act on.
- **`--check` never printed a reason for a note that was not a failure.** The standing
  caveats ("APUs have no fan", "GCN parts publish a voltage table here") were attached to
  rows that said `ok`, which reads as doubt about a number the machine had just produced.
  They are now only printed under a row that is missing something.
- **`-` and a unit with its own leading space produced `77  C`** in the report. Caught by
  reading the rendered output rather than the code that produced it.

## Closed during Milestone 4

- **`not-permitted` was reported as `unsupported`.** The packaged radeontop writes *"Cannot
  access GPU registers, are you root?"* to **stdout** and only *"Failed to find DRM devices"*
  to stderr, so a classifier reading stderr alone told a user their card was unsupported
  when the real answer was a privilege problem. Both streams are now read, and a test pins
  the exact split captured from the real binary.
- **The power-profile panel was gated on the blocks panel.** It is read from sysfs and works
  on any card publishing SCPP, so gating it on a privileged child process that is off by
  default silently withheld it from every unprivileged user. The two are now decided
  independently.
- **Two descriptor leaks.** The child's pipes were never closed on `stop()`, and `start()`
  replaced a process without closing the one it dropped — so every restart leaked two.
- **`log.interval_s` was accepted and ignored.** It now thins the recording.
- **`alerts.temp_c = 0`** alerted on every sample rather than disabling the check, despite
  the config file documenting `0` as "off".
- **A `temp_c = "hot"` in a config file** raised `TypeError` out of `load_config`, breaking
  the module's contract that a malformed file yields warnings and a working program.

## Notes for future milestones

- `BRAILLE_BITS` is the single source of truth for the dot layout; the usual braille
  off-by-one is dots 1–3 and 7 forming the *left* column, not a linear grid.
- A fixture whose `drm-engine-*` counters never advance has *no* utilisation: the counters
  are cumulative and utilisation is a delta. Tests that need a busy GPU must advance them
  and keep advancing them, because the sampler's tick and the interface's repaint run at
  the same period and a delta that exists for a single tick is regularly discarded before
  anything is painted.
- `tests/snapshot.wait_for_stable_layout` is what a test needs between an action and a
  capture. A bare `pilot.pause()` yields the loop once, which is not the same as "the
  frame is up", and a test that skips it fails only on a busy machine.
- Panel rows take `available` columns, not a bar width. Passing a fixed width is what caused
  the overflow; `fit_bar_width` derives it from the measured value string.
- Tests await the app's own tick count and a stable layout rather than sleeping; `pilot.pause()`
  alone does not mean "mounted", and that produced intermittent `NoMatches` failures.
- The rasteriser is a pure function (`braille_text`) precisely so it can be tested at
  explicit dimensions without mounting a widget.
- radeontop splits its diagnostics across **both** stdout and stderr, and the actionable
  message is on stdout. Any future reader of that child's output has to drain both pipes
  and classify from the pair; a test pins the exact split captured from the packaged binary.
- `Stylesheet.set_variables` **replaces** the variable map rather than merging into it, so
  it must not be used for anything that changes per sample. The alert flash writes to
  `widget.styles.border` for that reason, and using the variable path deletes Textual's own
  design tokens and fails the whole stylesheet.
