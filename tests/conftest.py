"""Shared fixtures: synthetic sysfs and procfs trees built in ``tmp_path``.

The fixtures mirror the *real* layout observed on the reference machine, including the
awkward parts that bit during development: the ``current_sclk`` attribute is absent on
recent drivers, the DPM unit is written ``Mhz`` with a lowercase ``h``, and engine
counters carry a trailing ``ns`` unit.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from gputop.readers.metrics_bin import ABI_REGISTRY, HEADER, MetricsAbi

#: Byte offsets verified against a real Navi 21 metric table.  Tests assert against these
#: literal numbers rather than re-deriving them from the registry, so a bug in the aligner
#: cannot silently rewrite the expectation along with the code.
V1_3_OFFSETS = {
    "temperature_edge": 4,
    "temperature_hotspot": 6,
    "temperature_mem": 8,
    "average_gfx_activity": 16,
    "average_socket_power": 22,
    "energy_accumulator": 24,
    "system_clock_counter": 32,
    "current_gfxclk": 54,
    "current_uclk": 58,
    "throttle_status": 68,
    "current_fan_speed": 72,
    "pcie_link_width": 74,
    "pcie_link_speed": 76,
    "temperature_hbm": 88,
    "firmware_timestamp": 96,
    "voltage_soc": 104,
    "voltage_gfx": 106,
    "voltage_mem": 108,
    "indep_throttle_status": 112,
}

#: The declared ``structure_size`` of a v1.3 table, read off the reference hardware.
V1_3_STRUCTURE_SIZE = 120


def write(path: Path, content: str | bytes) -> Path:
    """Write a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def build_metric_table(
    abi: MetricsAbi,
    values: dict[str, int] | None = None,
    *,
    structure_size: int | None = None,
    trailing: int = 0,
) -> bytes:
    """Pack a metric table for one ABI.

    Args:
        abi: The layout to build against.
        values: Field overrides, keyed by field name.  Everything else is zero.
        structure_size: Override the declared size, for testing mismatched headers.
        trailing: Extra bytes appended after the struct, for forward-compatibility tests.

    Returns:
        The raw buffer, including the 4-byte header.
    """
    fields = values or {}
    raw = bytearray(abi.body.size + trailing)
    for spec in abi.fields:
        value = fields.get(spec.name, 0)
        offset = abi.field_map[spec.name]
        size = {"B": 1, "H": 2, "I": 4, "Q": 8}[spec.code]
        raw[offset : offset + size] = int(value).to_bytes(size, "little")

    declared = structure_size if structure_size is not None else abi.body.size + trailing
    raw[0:4] = HEADER.pack(declared, abi.fmt_revision, abi.content_revision)
    return bytes(raw)


def pci_root_for(drm_root: Path) -> Path:
    """Return the fake ``pci_root`` that :func:`make_gpu` writes device attributes into.

    Mirrors the real layout ``/sys/bus/pci/devices/<bdf>`` so ``boot_vga`` is discoverable
    without a second fixture parameter.
    """
    return drm_root.parent.parent.parent / "devices" / "pci0000:00" / "0000:00:00.0"


def make_gpu(
    drm_root: Path,
    *,
    card: int = 0,
    bdf: str = "0000:0c:00.0",
    device_id: int = 0x73BF,
    vendor_id: int = 0x1002,
    driver: str | None = "amdgpu",
    metrics: dict[str, int] | None = None,
    metrics_abi: MetricsAbi | None = None,
    structure_size: int | None = None,
    metrics_raw: bytes | None = None,
    hwmon: dict[str, str] | None = None,
    extra_sysfs: dict[str, str] | None = None,
    hwmon_index: int = 0,
) -> Path:
    """Create one fake GPU under ``drm_root`` and return its device directory.

    Args:
        drm_root: The fake ``/sys/class/drm`` directory.
        card: The card number, producing ``cardN``.
        bdf: PCI address; also used as the resolved device directory name.
        device_id: PCI device id, written hex as the kernel does.
        driver: Bound driver name, or ``None`` to leave the driver link absent.
        metrics: Field values for the binary metric table.
        metrics_abi: Which ABI to emit; defaults to v1.3.
        structure_size: Override the declared structure size.
        metrics_raw: Write these exact bytes instead of building a table.
        hwmon: hwmon attribute name to contents, e.g. ``{"temp1_input": "49000"}``.
        extra_sysfs: Additional device attributes, name to contents.
        hwmon_index: hwmon directory number, so multi-GPU trees stay distinct.

    Returns:
        The path to the created ``cardN/device`` directory.
    """
    card_name = f"card{card}"
    # Real sysfs exposes cardN/device as a symlink into the PCI device tree, and the BDF
    # is derived from where that link resolves to.  A plain directory would hide the
    # address entirely, so the fixture reproduces the symlink.
    pci_dir = drm_root.parent.parent.parent / "devices" / "pci0000:00" / "0000:00:00.0" / bdf
    pci_dir.mkdir(parents=True, exist_ok=True)
    write(pci_dir / "boot_vga", "1\n")

    device_dir = drm_root / card_name / "device"
    device_dir.parent.mkdir(parents=True, exist_ok=True)
    if device_dir.is_symlink():
        # Idempotent: a test may build the same tree more than once.
        device_dir.unlink()
    device_dir.symlink_to(pci_dir)
    device_dir = pci_dir

    write(device_dir / "vendor", f"0x{vendor_id:04x}\n")
    write(device_dir / "device", f"0x{device_id:04x}\n")

    if driver is not None:
        driver_link = device_dir / "driver"
        driver_link.unlink(missing_ok=True)
        driver_link.symlink_to(f"../../../../bus/pci/drivers/{driver}")

    if metrics_raw is not None:
        write(device_dir / "gpu_metrics", metrics_raw)
    elif metrics_abi is not None or metrics is not None:
        abi = metrics_abi or ABI_REGISTRY[(1, 3)]
        write(
            device_dir / "gpu_metrics",
            build_metric_table(abi, metrics, structure_size=structure_size),
        )

    if hwmon is not None or extra_sysfs is not None:
        hwmon_dir = device_dir / "hwmon" / f"hwmon{hwmon_index}"
        hwmon_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (hwmon or {}).items():
            write(hwmon_dir / name, value)
        write(hwmon_dir / "name", "amdgpu\n")

    for name, value in (extra_sysfs or {}).items():
        write(device_dir / name, value)

    return device_dir


def make_process(
    proc_root: Path,
    pid: int,
    *,
    name: str = "testproc",
    drm_node: str = "/dev/dri/renderD128",
    fdinfo: Sequence[str] | None = None,
    extra_fds: Sequence[str] = (),
    ppid: int = 1,
    cmdline: Sequence[str] | None = None,
    client_id: int = 42,
    pdev: str = "0000:0c:00.0",
) -> Path:
    """Create a fake process holding a DRM descriptor.

    Args:
        proc_root: The fake ``/proc`` directory.
        pid: Process id, which is also the directory name.
        name: Contents of ``comm``.
        drm_node: The symlink target for the GPU descriptor.
        fdinfo: Extra ``fdinfo`` lines appended after the standard DRM header.  Later
            lines win, so a test can override a default such as ``drm-resident-vram``.
        extra_fds: Additional non-DRM descriptors, which must be ignored.
        ppid: Written to ``status`` as ``PPid``; drives the process tree.
        cmdline: The argument vector.  Defaults to ``[name]``; an empty sequence
            produces a kernel thread, which is how the kernel presents one.
        client_id: The ``drm-client-id`` the process reports.
        pdev: The ``drm-pdev`` the process reports, which routes it to one GPU.

    Returns:
        The path to the created process directory.
    """
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    write(pid_dir / "comm", f"{name}\n")
    # ``status`` is what the parent PID comes from; the real file has a lot more, and
    # only these two lines are ever parsed.
    write(pid_dir / "status", f"Name:\t{name}\nPPid:\t{ppid}\n")
    argv = (name,) if cmdline is None else tuple(cmdline)
    write(pid_dir / "cmdline", "".join(f"{arg}\0" for arg in argv))

    fd_dir = pid_dir / "fd"
    fdinfo_dir = pid_dir / "fdinfo"
    fd_dir.mkdir(exist_ok=True)
    fdinfo_dir.mkdir(exist_ok=True)

    (fd_dir / "9").unlink(missing_ok=True)
    (fd_dir / "9").symlink_to(drm_node)
    lines = [
        "pos:\t0",
        "flags:\t02100002",
        "drm-driver:\tamdgpu",
        f"drm-client-id:\t{client_id}",
        f"drm-pdev:\t{pdev}",
        "drm-total-vram:\t0 B",
        "drm-resident-vram:\t0 B",
    ]
    if fdinfo:
        lines.extend(fdinfo)
    write(fdinfo_dir / "9", "\n".join(lines) + "\n")

    for index, target in enumerate(extra_fds, start=20):
        (fd_dir / str(index)).unlink(missing_ok=True)
        (fd_dir / str(index)).symlink_to(target)
    return pid_dir


def engine_line(engine: str, nanoseconds: int) -> str:
    """Build a ``drm-engine-<name>`` line in the kernel's exact format."""
    return f"drm-engine-{engine}: {nanoseconds} ns"


@pytest.fixture
def drm_root(tmp_path: Path) -> Path:
    """An empty fake ``/sys/class/drm``."""
    root = tmp_path / "sys" / "class" / "drm"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def proc_root(tmp_path: Path) -> Path:
    """An empty fake ``/proc``."""
    root = tmp_path / "proc"
    root.mkdir(parents=True)
    return root
