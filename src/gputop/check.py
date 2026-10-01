"""``gputop --check``: what this machine can report, and why not.

A monitor that shows ``N/A`` in a panel is easy to read and hard to act on: the same
``N/A`` means "this GPU has no fan", "your kernel predates the attribute", "the driver
publishes it under a different name", "you need root" and "a newer gputop knows how to
read it".  This module separates those.

The report is built from a *real sample*, not from a list of expected filenames, so what
it says is what the interface would actually draw on this machine right now.  Only the
missing metrics are then probed on disk, to explain themselves.

Design notes:

* **No reason is invented.**  Every :class:`Reason` is derived from something observed:
  an ``errno``, a kernel version compared against a documented minimum, or the shape of
  the device.  When none of those explains it, the reason is :attr:`Reason.NOT_PUBLISHED`
  and the text says so rather than guessing.
* **The text report is plain ASCII.**  It is meant to be pasted into a bug report, so it
  carries no colour and no cursor control, and the same content is available as JSON via
  ``--json`` for anything that wants to consume it.
"""

import errno
import os
import platform
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from gputop.model.blocks import BlocksStatus
from gputop.model.device import AmdgpuDevice
from gputop.model.metrics import AmdgpuMetrics, Fan, MemoryPool, Power
from gputop.model.snapshot import GpuSnapshot
from gputop.readers.radeontop import resolve_binary

#: Widest the text report will draw itself before wrapping a column.
REPORT_WIDTH = 96


class Availability(StrEnum):
    """Whether a metric can be shown, and if not, which category it falls into."""

    OK = "ok"
    MISSING = "missing"
    #: Present in the data model, deliberately not collected in this run.
    DISABLED = "disabled"


class Reason(StrEnum):
    """Why a metric is not available.

    Each value carries the information a user needs to *act*, which is the only reason the
    distinctions exist: "install radeontop" and "you need root" are different problems.
    """

    NOT_PUBLISHED = "not published"
    NEEDS_NEWER_KERNEL = "needs a newer kernel"
    NEEDS_NEWER_DRIVER = "needs a newer amdgpu driver"
    UNSUPPORTED_GENERATION = "not on this GPU generation"
    PERMISSION = "permission denied"
    UNKNOWN_ABI = "unknown gpu_metrics version"
    NOT_INSTALLED = "not installed"
    #: The tool is present but was deliberately not run, so its behaviour is untested.
    NOT_EXERCISED = "not exercised by this check"
    DISABLED = "turned off"
    NO_CLIENTS = "no GPU clients to account for"
    HIDDEN = "other users' processes are not visible"


#: The Linux kernel release that introduced DRM fdinfo accounting (``drm-client-id``,
#: ``drm-engine-*``, ``drm-memory-*``).  Everything per-process depends on it.
MIN_KERNEL_FDINFO = (5, 19)

#: The release that introduced the versioned ``gpu_metrics`` binary table
#: (``AMDGPU_DEVICE_ATTR_RO(gpu_metrics, ...)`` in ``drivers/gpu/drm/amd/pm/amdgpu_pm.c``,
#: absent at v5.9 and present at v5.10).  Before it, every reading came from scattered
#: sysfs attributes, which is why the interface still works without it -- degraded, one
#: attribute at a time.
#:
#: Note that ``gpu_metrics`` carries no version *string*: the first four bytes are
#: ``structure_size | format_revision | content_revision``, and the kernel's
#: ``gpu_metrics_vX_Y`` source structs are exactly ``format=X, content=Y``.  The ABI
#: label gputop shows is built from those two bytes, never from a name.
MIN_KERNEL_GPU_METRICS = (5, 10)

#: Minimum kernel per sysfs attribute, used only to explain an absence.  Each entry is the
#: release in which the attribute was registered, established by diffing the driver's
#: attribute tables across release tags; when one could not be established it is left out
#: rather than guessed, and the report says "not published" instead.  The amdgpu hwmon
#: device itself is older than any of these (4.2) -- 4.17 added power and voltage.
ATTRIBUTE_MIN_KERNEL: dict[str, tuple[int, int]] = {
    "gpu_metrics": MIN_KERNEL_GPU_METRICS,
    "gpu_busy_percent": (4, 19),
    "mem_busy_percent": (5, 3),
    "pp_power_profile_mode": (4, 17),
    "pp_od_clk_voltage": (4, 17),
    "mem_info_vis_vram_used": (5, 2),
    "mem_info_vis_vram_total": (5, 2),
    "mem_info_gtt_used": (5, 2),
    "mem_info_gtt_total": (5, 2),
    # ``drm-client-id``/``drm-engine-*``/``drm-memory-*`` arrived together in 5.19, when
    # the amdgpu fdinfo file was converted to the common drm-* spelling.  amdgpu printed
    # fdinfo from 5.14, but under different key names, so a parser looking for drm-* finds
    # nothing before 5.19 -- which makes 5.19 the right floor to quote.
    "drm-engine-gfx": MIN_KERNEL_FDINFO,
}

_RELEASE = re.compile(r"^(\d+)\.(\d+)")

#: What :func:`kernel_version` returns for a release string it could not parse.  It is not
#: a kernel version and must never be printed as one; see :func:`kernel_known`.
UNKNOWN_KERNEL = (0, 0)


def kernel_version(release: str | None = None) -> tuple[int, int]:
    """Parse ``major.minor`` out of a kernel release string.

    Anything unparseable -- a vendor suffix, a container's odd string -- yields
    :data:`UNKNOWN_KERNEL` so that "is this kernel new enough" is answered False and the
    explanation falls back to the weaker, still-true "not published".
    """
    # ``None`` means "ask the platform"; an empty string is an empty string, and treating
    # the two alike would make a caller that passes a blank value believe it had parsed a
    # release and go on to conclude the kernel is new enough.
    match = _RELEASE.match(platform.release() if release is None else release)
    if match is None:
        return UNKNOWN_KERNEL
    return (int(match.group(1)), int(match.group(2)))


def kernel_known(version: tuple[int, int]) -> bool:
    """Whether *version* came from a real release rather than a parse failure.

    "Older than 5.10" and "could not tell" are different answers and only the first is
    worth telling a user to act on, so a caller comparing against a minimum must ask this
    first.  :func:`at_least` deliberately answers False for an unknown version, which is
    the right answer to "can I rely on this" and exactly the wrong answer to "is the
    kernel too old" -- without this guard an unparseable release string produced "needs
    Linux 5.10 or newer, this is 0.0", inventing a version nobody is running.
    """
    return version != UNKNOWN_KERNEL


def at_least(version: tuple[int, int], minimum: tuple[int, int]) -> bool:
    """Whether ``version`` is at least ``minimum``."""
    return version >= minimum if version != (0, 0) else False


def describe_version(version: tuple[int, int]) -> str:
    """Render a version tuple for display."""
    return f"{version[0]}.{version[1]}"


@dataclass(frozen=True, slots=True)
class Metric:
    """One metric the report can say something about.

    Attributes:
        key: Stable identifier, also the JSON field name.
        label: What a human calls this metric.
        source: The interface it is read through, e.g. ``gpu_metrics v1.3``.
        value: The reading this sample produced, or ``None`` when there is none.
        attributes: Files consulted, relative to the device directory; used to explain a
            missing reading by probing for them.
        min_kernel: Kernel release that introduced the *first* of ``attributes``, used to
            turn "absent" into "your kernel is too old".
        discrete_only: Whether an integrated GPU is expected not to publish this at all.
        remedy: What the user can do about it, when there is something.
        extra: A short note appended to the report, e.g. a fallback source.
    """

    key: str
    label: str
    source: str
    value: str | None
    attributes: tuple[str, ...] = ()
    min_kernel: tuple[int, int] | None = None
    discrete_only: bool = False
    remedy: str = ""
    extra: str = ""


@dataclass(frozen=True, slots=True)
class Finding:
    """A metric plus the verdict on it."""

    metric: Metric
    availability: Availability
    reason: Reason | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "metric": self.metric.key,
            "label": self.metric.label,
            "status": str(self.availability),
            "source": self.metric.source,
            "reading": self.metric.value,
            "reason": None if self.reason is None else str(self.reason),
            "detail": self.detail,
            "remedy": self.metric.remedy,
        }


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """Everything the report says about one GPU."""

    device: AmdgpuDevice
    metrics_abi: str | None
    findings: tuple[Finding, ...] = ()

    @property
    def missing(self) -> tuple[Finding, ...]:
        """Findings that are not ``ok``."""
        return tuple(f for f in self.findings if f.availability is not Availability.OK)

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "index": self.device.index,
            "card": self.device.card,
            "bdf": self.device.bdf,
            "name": self.device.name,
            "kind": self.device.kind,
            "device_id": f"0x{self.device.device_id:04x}",
            "metrics_abi": self.metrics_abi,
            "available": len(self.findings) - len(self.missing),
            "metrics": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True, slots=True)
class Environment:
    """The machine facts that decide what can be read at all."""

    kernel: str
    kernel_version: tuple[int, int]
    amdgpu_version: str | None
    python: str
    free_threaded: bool
    drm_root: Path
    proc_root: Path
    render_nodes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "kernel": self.kernel,
            "amdgpu": self.amdgpu_version,
            "python": self.python,
            "free_threaded": self.free_threaded,
            "drm_root": str(self.drm_root),
            "proc_root": str(self.proc_root),
            "render_nodes": list(self.render_nodes),
        }


@dataclass(frozen=True, slots=True)
class Report:
    """The whole ``--check`` answer."""

    environment: Environment
    devices: tuple[DeviceReport, ...] = ()
    findings: tuple[Finding, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the run found anything to report on."""
        return bool(self.devices)

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "ok": self.ok,
            "environment": self.environment.to_dict(),
            "devices": [d.to_dict() for d in self.devices],
            "system": [f.to_dict() for f in self.findings],
            "warnings": list(self.warnings),
        }

    # -- text ------------------------------------------------------------

    def render(self) -> str:
        """The human report: plain ASCII, safe to paste into a bug report."""
        lines: list[str] = ["gputop check", "=" * REPORT_WIDTH, ""]
        env = self.environment
        amdgpu = env.amdgpu_version or "not loaded"
        lines.append(f"kernel           {env.kernel}")
        lines.append(f"amdgpu           {amdgpu}")
        lines.append(f"python           {env.python}")
        lines.append(
            f"build            {'free-threaded' if env.free_threaded else 'standard (GIL)'}"
        )
        lines.append(f"sysfs            {env.drm_root}")
        lines.append(f"procfs           {env.proc_root}")
        if env.render_nodes:
            lines.append(f"render nodes     {', '.join(env.render_nodes)}")
        else:
            lines.append("render nodes     none found (no /dev/dri entry for these cards)")
        lines.append("")

        if not self.devices:
            lines.append("No amdgpu device was found under the sysfs DRM class directory.")
            lines.append("")
            lines.append("Things that produce exactly this:")
            lines.append("  - the kernel has no CONFIG_DRM_AMDGPU, or no module built")
            lines.append("  - the card is bound to the older 'radeon' driver, not 'amdgpu'")
            lines.append("  - the container this runs in does not pass /sys through")
            lines.append("")

        for device in self.devices:
            d = device.device
            lines.append(
                f"{d.card}  {d.name}  ({d.bdf}, {'integrated' if d.is_integrated else 'discrete'}, "
                f"0x{d.device_id:04x})"
            )
            lines.append(f"{'metrics ABI':<18}{device.metrics_abi or 'none (sysfs only)'}")
            lines.append("")
            lines.extend(_table(device.findings))
            lines.append("")

        # Everything below applies with or without a device.  A run that found no card is
        # exactly the run where a broken config is most worth reporting, so returning early
        # above this point printed a clean-looking report that dropped every warning and
        # every system finding -- and disagreed with the same run's own ``--json``.
        if self.findings:
            lines.append("System-wide")
            lines.append("-" * REPORT_WIDTH)
            lines.extend(_table(self.findings))
            lines.append("")

        for warning in self.warnings:
            lines.append(f"warning: {warning}")

        missing = sum(len(d.missing) for d in self.devices) + sum(
            1 for f in self.findings if f.availability is not Availability.OK
        )
        total = sum(len(d.findings) for d in self.devices) + len(self.findings)
        lines.append(
            f"{total - missing}/{total} metrics available"
            + ("" if missing == 0 else f"  ({missing} not available, each explained above)")
        )
        return "\n".join(lines)


def _table(findings: Sequence[Finding]) -> list[str]:
    """Render findings as an aligned table, with an explanation under anything missing.

    A note is only attached to a metric that is *not* available.  Attaching the standing
    caveat ("APUs have no fan") to a reading that is present makes the table say "ok" and
    then immediately qualify it, which reads as doubt about a number the machine just
    produced.
    """
    key_width = min(max((len(f.metric.label) for f in findings), default=8), 26)
    state_width = 10
    source_width = 26
    lines = [
        f"{'METRIC'.ljust(key_width)}  {'STATUS'.ljust(state_width)}  "
        f"{'SOURCE'.ljust(source_width)}  VALUE",
        f"{'-' * key_width}  {'-' * state_width}  {'-' * source_width}  -----",
    ]
    for finding in findings:
        metric = finding.metric
        value = metric.value if finding.availability is Availability.OK else "-"
        lines.append(
            f"{metric.label[:key_width].ljust(key_width)}  "
            f"{str(finding.availability).ljust(state_width)}  "
            f"{metric.source[:source_width].ljust(source_width)}  {value}"
        )
        if finding.availability is Availability.OK:
            continue
        reason = finding.reason
        note = str(reason) if reason is not None else ""
        if finding.detail:
            note = f"{note}: {finding.detail}" if note else finding.detail
        if not note and metric.extra:
            note = metric.extra
        if note:
            lines.append(f"{'':<{key_width}}  {'':<{state_width}}  -> {note}")
        # ``extra`` is appended as its own line rather than folded into ``note``.  It is the
        # standing caveat that says what to *do* ("add --blocks", "APUs have no fan"), and
        # every finding that carries one also has a reason -- so a note that is only shown
        # when there is nothing else to say rendered none of them.
        if metric.extra:
            lines.append(f"{'':<{key_width}}  {'':<{state_width}}  -> {metric.extra}")
        if metric.remedy:
            lines.append(f"{'':<{key_width}}  {'':<{state_width}}  fix: {metric.remedy}")
    return lines


# -- metric construction -------------------------------------------------


def _probe(path: Path) -> str | None:
    """Classify one path: ``"ok"``, a reason string, or ``None`` when it does not exist."""
    try:
        with path.open("rb"):
            return "ok"
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return "permission"
        if exc.errno == errno.ENOENT:
            return None
        if exc.errno == errno.EISDIR:
            # hwmon and friends are directories, and a directory that exists is exactly
            # as informative as a file that exists: this source is published here.
            return "ok"
        return "unreadable"
    except ValueError:  # embedded NUL, or a path that vanished mid-construction
        return None


def _explain(
    device_dir: Path,
    attributes: Sequence[str],
    kernel: tuple[int, int],
    *,
    discrete_only: bool = False,
    integrated: bool = False,
    min_kernel: tuple[int, int] | None = None,
    metrics_abi: str | None = None,
) -> tuple[Reason, str]:
    """Work out why a metric is missing, from what is on disk.

    The order is the order of what the user can act on.  Hardware that cannot have the
    sensor at all is settled first, because no amount of kernel or permissions will make
    an APU grow a fan; then a source that is present but empty, which is the driver or the
    silicon; then a source that exists but cannot be opened, which is the user; and only
    then the kernel version, which is the one explanation that cannot be wrong but is also
    the least likely.

    Every candidate is probed once.  Probing lazily and re-opening the same files to ask a
    second question is how "permission denied" turns into "not published".
    """
    if not attributes:
        return Reason.NOT_PUBLISHED, "this card exposes no source for it"
    if discrete_only and integrated:
        return (
            Reason.UNSUPPORTED_GENERATION,
            "an integrated GPU has no fan to report",
        )

    states = {name: _probe(device_dir / name) for name in attributes}
    readable = [name for name, state in states.items() if state == "ok"]

    if readable:
        if "gpu_metrics" in readable and metrics_abi is None:
            # The file is there and openable but did not decode, which is a different
            # problem from "the driver does not publish it" and a different one again from
            # a missing file: a newer driver rolled out a layout this build has no decoder
            # for.  The two used to be reported the same way, and the module's own summary
            # promises to separate exactly this from "a newer gputop knows how to read it".
            return (
                Reason.UNKNOWN_ABI,
                "this kernel publishes a gpu_metrics layout this build does not decode, "
                "so the numbers are being read from sysfs one attribute at a time instead",
            )
        return (
            Reason.NOT_PUBLISHED,
            f"{readable[0]} exists but holds no value this build can use",
        )

    denied = [name for name, state in states.items() if state == "permission"]
    if denied:
        return Reason.PERMISSION, f"{denied[0]} is not readable by this user"

    unreadable = [name for name, state in states.items() if state == "unreadable"]
    if unreadable:
        return Reason.NOT_PUBLISHED, f"{unreadable[0]} could not be opened"

    required = min_kernel or next(
        (ATTRIBUTE_MIN_KERNEL[name] for name in attributes if name in ATTRIBUTE_MIN_KERNEL),
        None,
    )
    if required is not None and kernel_known(kernel) and not at_least(kernel, required):
        return (
            Reason.NEEDS_NEWER_KERNEL,
            f"needs Linux {describe_version(required)} or newer, "
            f"this is {describe_version(kernel)}",
        )
    return (
        Reason.NOT_PUBLISHED,
        f"the driver does not create {attributes[0]} on this card",
    )


@dataclass(frozen=True, slots=True)
class _Spec:
    """Declarative description of one metric in the report."""

    key: str
    label: str
    source: str
    read: Callable[[AmdgpuMetrics], str | None]
    attributes: tuple[str, ...] = ()
    min_kernel: tuple[int, int] | None = None
    discrete_only: bool = False
    remedy: str = ""
    extra: str = ""


def _fmt(value: float | None, unit: str, digits: int = 1) -> str | None:
    """Format a reading with its unit, or ``None`` when there is none.

    The unit never carries its own leading space: composing ``f"{value} {unit}"`` from a
    unit that already has one yields ``77  C``, and a report pasted into a bug report
    should not look careless.
    """
    if value is None:
        return None
    return f"{value:.{digits}f} {unit}"


def _part(value: float | None, unit: str, digits: int = 1) -> str:
    """Format a *sub*-reading that may be absent while the whole reading is not.

    Distinct from the ``None`` that :func:`_fmt` returns, which means "this whole metric
    is unavailable" and is what drives the ``ok``/``missing`` verdict.  Interpolating that
    into an f-string writes the literal word ``None`` into a report people paste into bug
    reports, so a part that is missing renders as ``N/A`` instead -- and never as a zero,
    which would be a measurement rather than an absence.
    """
    return "N/A" if value is None else f"{value:.{digits}f} {unit}"


def _mib(value: int | None) -> str:
    """Format a byte count as MiB, or a dash when there is no reading."""
    return "-" if value is None else f"{value / (1024 * 1024):.0f} MiB"


def _pool(pool: MemoryPool) -> str | None:
    """Format a memory pool as ``used / total``, or ``None`` when neither is known."""
    if pool.used is None and pool.total is None:
        return None
    return f"{_mib(pool.used)} / {_mib(pool.total)}"


def _power(power: Power) -> str | None:
    """Format the draw, with the cap when the card publishes one.

    The cap and the draw come from different hwmon files and either may be absent, so the
    clause is conditional rather than interpolated: ``power1_cap`` is not published on many
    boards, and a fixed ``(cap ...)`` group printed ``(cap None)`` there.
    """
    if power.draw_w is None:
        return None
    if power.cap_w is None:
        return f"{power.draw_w:.0f} W"
    return f"{power.draw_w:.0f} W (cap {power.cap_w:.0f} W)"


def _fan(fan: Fan) -> str | None:
    """Format whichever of the tach and the duty cycle the card actually publishes.

    ``fan1_input`` and ``pwm1`` are independent files, so a card can have either without the
    other.  A missing tach is rendered as nothing rather than as ``0 RPM``: the fan model
    treats a zero as *stopped* (:attr:`Fan.stopped`), so reporting an absent sensor as a
    stopped fan turns a gap in the data into a claim about the hardware.
    """
    parts = [
        text
        for text in (
            f"{fan.rpm} RPM" if fan.rpm is not None else "",
            f"{_part(fan.pwm, '%', 0)} duty" if fan.pwm is not None else "",
        )
        if text
    ]
    return ", ".join(parts) or None


def _specs() -> tuple[_Spec, ...]:
    """Every metric the report covers, in the order they appear in the interface."""
    return (
        _Spec(
            "gpu_utilisation",
            "GPU utilisation",
            "gpu_metrics+sysfs",
            lambda m: _fmt(m.gpu_busy_percent, "%"),
            ("gpu_metrics", "gpu_busy_percent"),
        ),
        _Spec(
            "mem_utilisation",
            "Memory utilisation",
            "gpu_metrics+sysfs",
            lambda m: _fmt(m.mem_busy_percent, "%"),
            ("gpu_metrics", "mem_busy_percent"),
        ),
        _Spec(
            "vram",
            "VRAM",
            "mem_info_vis_vram_*",
            lambda m: _pool(m.vram),
            ("mem_info_vis_vram_used", "mem_info_vram_used"),
        ),
        _Spec(
            "gtt",
            "GTT",
            "mem_info_gtt_*",
            lambda m: _pool(m.gtt),
            ("mem_info_gtt_used", "mem_info_gtt_total"),
        ),
        _Spec(
            "sclk",
            "Graphics clock",
            "gpu_metrics+pp_dpm_sclk",
            lambda m: (
                None
                if m.sclk.current is None
                else f"{m.sclk.current} MHz (max {m.sclk.maximum or '?'})"
            ),
            ("gpu_metrics", "pp_dpm_sclk", "current_sclk"),
        ),
        _Spec(
            "mclk",
            "Memory clock",
            "gpu_metrics+pp_dpm_mclk",
            lambda m: (
                None
                if m.mclk.current is None
                else f"{m.mclk.current} MHz (max {m.mclk.maximum or '?'})"
            ),
            ("gpu_metrics", "pp_dpm_mclk", "current_mclk"),
        ),
        _Spec(
            "socclk",
            "SoC clock",
            "gpu_metrics+pp_dpm_socclk",
            lambda m: (
                None
                if m.socclk.current is None
                else f"{m.socclk.current} MHz (max {m.socclk.maximum or '?'})"
            ),
            ("gpu_metrics", "pp_dpm_socclk", "current_socclk"),
            extra="only present on parts with a separate SoC clock domain",
        ),
        _Spec(
            "temperatures",
            "Temperatures",
            "hwmon+gpu_metrics",
            lambda m: (
                ", ".join(f"{t.label} {_part(t.celsius, 'C', 0)}" for t in m.temperatures)
                or None
            ),
            ("gpu_metrics", "hwmon"),
        ),
        _Spec(
            "power",
            "Power draw",
            "hwmon+gpu_metrics",
            lambda m: _power(m.power),
            ("hwmon", "gpu_metrics"),
        ),
        _Spec(
            "fan",
            "Fan",
            "hwmon+gpu_metrics",
            lambda m: _fan(m.fan),
            ("hwmon", "gpu_metrics"),
            discrete_only=True,
            extra="APUs are passively cooled and have no fan to report",
        ),
        _Spec(
            "throttle",
            "Throttle status",
            "gpu_metrics",
            lambda m: (
                None if m.throttle is None else ", ".join(sorted(m.throttle.active)) or "idle"
            ),
            ("gpu_metrics",),
        ),
        _Spec(
            "pcie",
            "PCIe link",
            "gpu_metrics+current_link_*",
            lambda m: None if m.pcie is None else m.pcie.describe(),
            ("gpu_metrics", "current_link_width"),
        ),
        _Spec(
            "voltages",
            "Voltages",
            "gpu_metrics",
            lambda m: ", ".join(f"{name} {mv} mV" for name, mv in m.voltages_mv) or None,
            ("gpu_metrics",),
        ),
        _Spec(
            "performance_level",
            "Performance level",
            "power_dpm_*",
            lambda m: m.performance_level,
            ("power_dpm_force_performance_level", "pp_dpm_performance_level"),
        ),
        _Spec(
            "power_profile",
            "Power profile",
            "pp_power_profile_mode",
            lambda m: m.power_profile,
            ("pp_power_profile_mode",),
        ),
        _Spec(
            "overdrive",
            "Overdrive table",
            "pp_od_clk_voltage",
            lambda m: None if not m.odc.present else f"{len(m.odc.domains)} clock domains",
            ("pp_od_clk_voltage",),
            extra="GCN parts publish a voltage table here instead of overdrive ceilings",
        ),
    )


def device_findings(
    device: AmdgpuDevice, metrics: AmdgpuMetrics, kernel: tuple[int, int]
) -> tuple[Finding, ...]:
    """Evaluate every metric for one device against a real sample."""
    directory = device.device_dir
    findings: list[Finding] = []

    clocks = {"sclk": metrics.sclk, "mclk": metrics.mclk, "socclk": metrics.socclk}
    for spec in _specs():
        value = spec.read(metrics)
        # A clock can come from three places, and which one answered is the difference
        # between "2475 MHz" and "2475 MHz taken from the DPM table" -- so the source
        # column names the winner rather than the whole fallback chain.
        source = spec.source
        clock = clocks.get(spec.key)
        if value is not None and clock is not None and clock.source != "none":
            source = {
                "gpu_metrics": "gpu_metrics",
                "sysfs": "sysfs current_*",
                "dpm": f"pp_dpm_{spec.key}",
            }.get(clock.source, clock.source)
        metric = Metric(
            key=spec.key,
            label=spec.label,
            source=source,
            value=value,
            attributes=spec.attributes,
            min_kernel=spec.min_kernel,
            discrete_only=spec.discrete_only,
            remedy=spec.remedy,
            extra=spec.extra,
        )
        if value is not None:
            findings.append(Finding(metric, Availability.OK))
            continue
        reason, detail = _explain(
            directory,
            spec.attributes,
            kernel,
            discrete_only=spec.discrete_only,
            integrated=device.is_integrated,
            min_kernel=spec.min_kernel,
            metrics_abi=metrics.metrics_abi,
        )
        findings.append(Finding(metric, Availability.MISSING, reason, detail))

    return tuple(findings)


def process_ownership(proc_root: Path) -> tuple[int, int]:
    """Count ``(processes, hidden)`` under *proc_root*.

    ``hidden`` is the number of process directories this user cannot inspect, which is the
    honest version of "you may not see every client's utilisation".  It costs one ``stat``
    per process, which is why it is computed here rather than in the sampler: the
    sampler's job is to be cheap on every tick, and this is a question asked once.
    """
    total = hidden = 0
    uid = os.geteuid()
    try:
        entries = os.scandir(proc_root)
    except OSError:
        return (0, 0)
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            total += 1
            try:
                owner = entry.stat(follow_symlinks=False).st_uid
            except FileNotFoundError:
                # The process exited between the scan and the stat, which on a busy machine
                # is routine and says nothing about who owns what.  Counting it as hidden
                # reported "N of M processes belong to another user" for processes that no
                # longer exist, and counting it in M at all inflated the denominator of a
                # permissions statement that should be about live processes only.
                total -= 1
                continue
            except OSError:
                hidden += 1
                continue
            if owner != uid:
                hidden += 1
    return (total, hidden)


def _process_finding(
    snapshot: GpuSnapshot,
    *,
    kernel: tuple[int, int],
    proc_root: Path,
    scanned: bool = True,
) -> Finding:
    """Describe the per-process table from a real scan.

    The interesting question when the table is empty is "is nothing running, or can I not
    see it?", and those need different answers, so the ownership scan is what decides
    between them rather than the client count alone.

    ``scanned`` is False when the caller turned the /proc scan off, and an empty table then
    means "nobody looked" rather than "the GPU is idle".  Without this the report answered
    the one question a user most often brings to ``--check`` with a guess: an empty process
    table is the symptom that sends people here, and "the GPU is idle" is exactly the wrong
    conclusion when the scan was switched off.
    """
    clients = len(snapshot.processes)
    total, hidden = process_ownership(proc_root)
    value = (
        f"{clients} GPU client{'' if clients == 1 else 's'} of "
        f"{total} process{'' if total == 1 else 'es'}"
        if total
        else "no processes visible"
    )
    metric = Metric(
        key="processes",
        label="Process table",
        source="/proc/<pid>/fdinfo",
        value=value,
        attributes=(),
        remedy="",
    )
    if not scanned:
        # Availability.DISABLED rather than MISSING: nothing is wrong, the data was not
        # collected, and the reason says which of the two switches did it so the user can
        # turn it back on.  A MISSING here would read as a broken machine.
        return Finding(
            metric,
            Availability.DISABLED,
            Reason.DISABLED,
            "the /proc scan was switched off, so this table was not read and says nothing "
            "about whether the GPU is busy",
        )
    if kernel_known(kernel) and not at_least(kernel, MIN_KERNEL_FDINFO):
        return Finding(
            metric,
            Availability.MISSING,
            Reason.NEEDS_NEWER_KERNEL,
            f"per-process accounting needs Linux {describe_version(MIN_KERNEL_FDINFO)} "
            f"or newer, this is {describe_version(kernel)}",
        )
    if clients:
        return Finding(metric, Availability.OK)
    if not total:
        return Finding(
            metric,
            Availability.MISSING,
            Reason.NO_CLIENTS,
            f"{proc_root} lists no processes at all",
        )
    if hidden:
        return Finding(
            metric,
            Availability.MISSING,
            Reason.HIDDEN,
            f"{hidden} of {total} processes belong to another user and cannot be "
            "inspected; gputop never asks for privileges to read them",
        )
    return Finding(
        metric,
        Availability.OK,
        Reason.NO_CLIENTS,
        f"none of the {total} visible processes holds a /dev/dri descriptor, which is "
        "what an idle GPU looks like; ROCm, Vulkan and OpenGL workloads appear here as "
        "soon as they run",
    )


def system_findings(
    snapshot: GpuSnapshot,
    *,
    kernel: tuple[int, int],
    proc_root: Path,
    blocks_status: BlocksStatus | None = None,
    blocks_hint: str = "",
    blocks_enabled: bool = False,
    blocks_binary: str = "radeontop",
    processes_scanned: bool = True,
) -> tuple[Finding, ...]:
    """Evaluate the findings that are about the system rather than one GPU.

    ``blocks_status`` is None when radeontop was never run, which is the normal case for
    ``--check``: the panel is optional and off by default, and a diagnostic question must
    not start a process that wants privileges.  None means "not observed", and is reported
    as such rather than as whatever status a pool that was never started happens to hold.
    """
    findings: list[Finding] = []
    findings.append(
        _process_finding(
            snapshot, kernel=kernel, proc_root=proc_root, scanned=processes_scanned
        )
    )

    blocks_metric = Metric(
        key="blocks",
        label="Per-block panel",
        source=blocks_binary,
        value=None,
        attributes=(),
        extra="add --blocks or blocks.enabled = true to use it",
    )
    if not blocks_enabled:
        findings.append(Finding(blocks_metric, Availability.DISABLED, Reason.DISABLED, ""))
    elif blocks_status is None:
        # ``--check`` never spawns radeontop, so there is no status to report and the pool's
        # own unstarted value is not evidence: reading it as MISSING told every user to
        # install a binary they already had.  Resolving the name is a ``$PATH`` lookup and a
        # stat -- no process, no privileges -- so installation can still be answered
        # honestly, and what remains unknown is said to be unknown.
        if resolve_binary(blocks_binary) is None:
            findings.append(
                Finding(
                    blocks_metric,
                    Availability.MISSING,
                    Reason.NOT_INSTALLED,
                    f"no {blocks_binary} binary on this system (install it, or set "
                    "blocks.binary)",
                )
            )
        else:
            # Availability.DISABLED: nothing is wrong and nothing was collected, which is
            # what that member means.  Whether radeontop can read *this* card is only
            # knowable by running it, and saying so beats guessing "unsupported".
            findings.append(
                Finding(
                    Metric(
                        key="blocks",
                        label="Per-block panel",
                        source=str(resolve_binary(blocks_binary)),
                        value="installed",
                        attributes=(),
                    ),
                    Availability.DISABLED,
                    Reason.NOT_EXERCISED,
                    "radeontop is installed but --check never runs it, so whether it can "
                    "read this card is untested; enable the blocks panel to find out",
                )
            )
    elif blocks_status.ok:
        blocks_metric = Metric(
            key="blocks",
            label="Per-block panel",
            source=blocks_binary,
            value="available",
            attributes=(),
        )
        findings.append(Finding(blocks_metric, Availability.OK))
    else:
        reason = {
            BlocksStatus.MISSING: Reason.NOT_INSTALLED,
            BlocksStatus.NOT_PERMITTED: Reason.PERMISSION,
            BlocksStatus.UNSUPPORTED: Reason.UNSUPPORTED_GENERATION,
            BlocksStatus.FAILED: Reason.NEEDS_NEWER_DRIVER,
        }.get(blocks_status, Reason.NOT_PUBLISHED)
        detail = blocks_hint or blocks_status.hint
        findings.append(Finding(blocks_metric, Availability.MISSING, reason, detail))

    return tuple(findings)


def amdgpu_version(drm_root: Path) -> str | None:
    """The loaded ``amdgpu`` module version, when it can be read.

    Read from ``/sys/module/amdgpu/version`` rather than by asking the module: this is
    unprivileged, costs no subprocess, and works inside a container that has sysfs but not
    ``modinfo``.
    """
    candidates = (
        Path("/sys/module/amdgpu/version"),
        drm_root.parent.parent / "module" / "amdgpu" / "version",
    )
    for candidate in candidates:
        try:
            return candidate.read_text(errors="replace").strip() or None
        except OSError, ValueError:
            continue
    return None


def render_nodes(
    devices: Iterable[AmdgpuDevice], dev_root: Path = Path("/dev/dri")
) -> tuple[str, ...]:
    """The DRM device nodes that exist on this machine.

    A card with no node under ``/dev/dri`` can still be monitored -- sysfs does not need
    one -- but it is worth saying, because its absence is also why the process table is
    empty.
    """
    try:
        entries = sorted(entry.name for entry in dev_root.iterdir())
    except OSError, ValueError:
        return ()
    names = {device.render for device in devices if device.render}
    return tuple(name for name in entries if name in names)


def gather(
    snapshot: GpuSnapshot,
    *,
    drm_root: Path,
    proc_root: Path,
    config_warnings: Sequence[str] = (),
    blocks_status: BlocksStatus | None = None,
    blocks_hint: str = "",
    blocks_enabled: bool = False,
    blocks_binary: str = "radeontop",
    processes_scanned: bool = True,
) -> Report:
    """Build the report from a sample.

    Args:
        snapshot: A real sample, so the report reflects this machine and this moment.
        drm_root: The sysfs DRM class directory in use.
        proc_root: The procfs mount point in use.
        config_warnings: Configuration problems to surface alongside the metrics.
        blocks_status: What the radeontop pool reported, or None if it was never run.
        blocks_hint: The human explanation for that status.
        blocks_enabled: Whether the blocks panel was asked for.
        blocks_binary: The radeontop binary that would be run.
        processes_scanned: Whether the /proc scan actually ran, so an empty process table
            is not mistaken for an idle GPU.
    """
    kernel = platform.release()
    version = kernel_version(kernel)
    environment = Environment(
        kernel=kernel,
        kernel_version=version,
        amdgpu_version=amdgpu_version(drm_root),
        python=platform.python_version(),
        free_threaded=not sys._is_gil_enabled(),
        drm_root=drm_root,
        proc_root=proc_root,
        render_nodes=render_nodes(
            (m.device for m in snapshot.devices),
            dev_root=Path(str(drm_root).replace("/sys/class/drm", "/dev/dri"))
            if str(drm_root).startswith("/sys/class/drm")
            else Path("/dev/dri"),
        ),
    )

    reports = tuple(
        DeviceReport(
            device=m.device,
            metrics_abi=m.metrics_abi,
            findings=device_findings(m.device, m, version),
        )
        for m in snapshot.devices
    )
    findings = system_findings(
        snapshot,
        kernel=version,
        proc_root=proc_root,
        blocks_status=blocks_status,
        blocks_hint=blocks_hint,
        blocks_enabled=blocks_enabled,
        blocks_binary=blocks_binary,
        processes_scanned=processes_scanned,
    )
    return Report(
        environment=environment,
        devices=reports,
        findings=findings,
        warnings=tuple(config_warnings),
    )


__all__ = [
    "Availability",
    "Reason",
    "Report",
    "gather",
    "kernel_version",
    "render_nodes",
]
