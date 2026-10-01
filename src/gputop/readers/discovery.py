"""Discovery of ``amdgpu`` devices under ``/sys/class/drm``.

Only the ``amdgpu`` driver is accepted.  The ``radeon`` driver, and any non-AMD vendor,
is skipped silently: the specification deliberately has no vendor abstraction layer.
"""

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from gputop.model.aliases import Bdf
from gputop.model.device import APU_DEVICE_IDS, AmdgpuDevice, DeviceKind, describe_device_id
from gputop.readers.fsutil import link_target, read_hex_int, read_int, read_text, resolve

#: Matches ``card0``, ``card12`` but never connector nodes such as ``card1-DP-1``.
CARD_PATTERN = re.compile(r"^card\d+$")

#: A PCI domain:bus:slot.function address, as used by ``drm-pdev`` in fdinfo.
BDF_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")

AMD_VENDOR_ID = 0x1002

#: APU carve-outs are small; anything above this is treated as a discrete board.
IGPU_VRAM_CEILING = 512 * 1024 * 1024

#: GTT is the APU working set, so a GTT pool much larger than VRAM implies integration.
GTT_TO_VRAM_RATIO = 4


def _bdf_from_device_dir(device_dir: Path) -> Bdf:
    """Derive ``0000:0c:00.0`` from the resolved sysfs device path.

    The resolved path ends in ``.../pci0000:00/0000:00:0c.0``, so the PCI address is the
    final component verbatim -- no slicing is needed, and slicing is what previously
    produced a mangled ``03.1:.0:0000:0b:00.0``.
    """
    name = device_dir.resolve(strict=False).name
    return name if BDF_PATTERN.fullmatch(name) else ""


def _is_amdgpu(device_dir: Path) -> bool:
    """Check the bound kernel driver is ``amdgpu`` (not ``radeon``)."""
    driver = link_target(device_dir / "driver")
    return driver is not None and driver.rsplit("/", 1)[-1] == "amdgpu"


def _resolve_hwmon(device_dir: Path) -> Path | None:
    """Resolve the hwmon directory that belongs to *this* device.

    Resolving through ``<device>/hwmon`` is essential: scanning ``/sys/class/hwmon`` and
    matching by name breaks on any machine with many sensors, because only ``hwmon/name``
    identifies the class, not the owning device.
    """
    hwmon_link = device_dir / "hwmon"
    try:
        entries = sorted(
            entry for entry in hwmon_link.iterdir() if entry.name.startswith("hwmon")
        )
    except OSError:
        return None
    return entries[0] if entries else None


def _find_render_node(device_dir: Path) -> str | None:
    """Find the render node belonging to a device, e.g. ``renderD128``."""
    drm_dir = device_dir / "drm"
    try:
        entries = sorted(entry.name for entry in drm_dir.iterdir())
    except OSError:
        return None
    renders = [name for name in entries if re.match(r"^renderD\d+$", name)]
    return renders[0] if renders else None


def _classify(
    device_dir: Path,
    device_id: int,
    bdf: Bdf,
    vram_vendor: str | None,
    forced: DeviceKind | None,
) -> DeviceKind:
    """Decide whether a device is integrated or discrete.

    No single signal is reliable, so several independent ones are combined.  A confident
    ``forced`` value short-circuits everything.
    """
    if forced is not None:
        return forced

    votes_igpu = 0
    if device_id in APU_DEVICE_IDS:
        votes_igpu += 1
    if vram_vendor is not None and vram_vendor.strip().upper() == "N/A":
        votes_igpu += 1

    vram_total = read_int(device_dir / "mem_info_vis_vram_total")
    if vram_total is None:
        vram_total = read_int(device_dir / "mem_info_vram_total")
    gtt_total = read_int(device_dir / "mem_info_gtt_total")

    if vram_total is not None and vram_total <= IGPU_VRAM_CEILING:
        votes_igpu += 1
    if (
        gtt_total is not None
        and vram_total is not None
        and vram_total > 0
        and gtt_total > vram_total * GTT_TO_VRAM_RATIO
    ):
        votes_igpu += 1

    # The CPU's own bus (typically 0000:00) is where an APU's graphics device sits.
    if bdf.startswith("0000:00:") and ":00.0" not in bdf:
        votes_igpu += 1

    return "igpu" if votes_igpu >= 2 else "dgpu"


def device_tokens(device: AmdgpuDevice) -> frozenset[str]:
    """Every lower-cased spelling of this device that ``gpu.devices`` may name.

    Three, because a user has three plausible things to write and only one of them is the
    address the ``--dump`` output shows them:

    * the full PCI address, ``0000:0c:00.0`` -- what ``[gpu.names]`` is keyed by and what
      every diagnostic prints;
    * the same address without the domain, ``0c:00.0`` -- which is how the header renders
      it, and how it is written in a bug report;
    * the sysfs node, ``card1`` -- which is what a user reading ``ls /sys/class/drm``
      actually sees.

    The domain is the only thing dropped, matching :func:`gputop.ui.app.short_bdf`, which
    renders that same shortened form.  A device with no resolvable address contributes
    just its card name.
    """
    tokens = {device.card.lower(), device.bdf.lower()}
    parts = device.bdf.split(":")
    if len(parts) == 3:
        tokens.add(":".join(parts[1:]).lower())
    return frozenset(token for token in tokens if token)


def select_devices(
    devices: Sequence[AmdgpuDevice], requested: Sequence[str]
) -> tuple[tuple[AmdgpuDevice, ...], tuple[str, ...]]:
    """Narrow *devices* to those named by the ``gpu.devices`` config key.

    Args:
        devices: Everything discovery found, in its own order.
        requested: The configured names.  Empty means "no filter".

    Returns:
        The selected devices in *discovery* order, and the requested names that matched
        none of them.

    A requested name that matches nothing is reported rather than honoured, and a filter
    that matches nothing at all falls back to every device.  Both halves matter: silently
    dropping an unmatched entry would turn a typo in one element of a two-card filter into
    a monitor watching one card with nothing to say so, and honouring the filter strictly
    would replace the monitor with "no amdgpu device found", which is a different and
    wrong answer to a question the user did not ask.  The caller turns the unmatched names
    into a warning naming them, so the configuration error is still visible.

    Order is preserved rather than reordered to the config's, because a device's ``index``
    is its position in exactly this sequence -- it is what the focus state, the per-device
    radeontop arguments and the dump output all address the card by.
    """
    wanted = {name.strip().lower() for name in requested if name.strip()}
    if not wanted:
        return tuple(devices), ()
    selected = tuple(device for device in devices if device_tokens(device) & wanted)
    if selected:
        return selected, tuple(sorted(name for name in wanted if not _any_match(devices, name)))
    return tuple(devices), tuple(sorted(wanted))


def _any_match(devices: Sequence[AmdgpuDevice], name: str) -> bool:
    """Whether any device answers to *name*."""
    return any(name in device_tokens(device) for device in devices)


def discover_devices(
    drm_root: Path = Path("/sys/class/drm"),
    *,
    kind_override: DeviceKind | None = None,
    name_overrides: dict[str, str] | None = None,
    pci_root: Path = Path("/sys/bus/pci/devices"),
) -> tuple[AmdgpuDevice, ...]:
    """Enumerate every ``amdgpu`` GPU exposed under ``drm_root``.

    Args:
        drm_root: The DRM sysfs class directory.  Overridable so tests can supply a fake
            tree.
        kind_override: Force every device to be classified as ``"igpu"`` or ``"dgpu"``.
        name_overrides: Map of BDF to marketing name, from the ``[gpu.names]`` config.
        pci_root: Where per-device PCI attributes such as ``boot_vga`` live.

    Returns:
        Devices sorted discrete-first, then by BDF, each carrying its resolved index.
        A machine with no AMD GPUs yields an empty tuple rather than raising.
    """
    overrides = name_overrides or {}
    found: list[tuple[bool, Bdf, AmdgpuDevice]] = []

    try:
        entries = sorted(drm_root.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return ()

    for entry in entries:
        if not CARD_PATTERN.match(entry.name):
            continue
        device_dir = entry / "device"
        vendor_id = read_hex_int(device_dir / "vendor")
        if vendor_id != AMD_VENDOR_ID:
            continue
        if not _is_amdgpu(device_dir):
            continue
        device_id = read_hex_int(device_dir / "device")
        if device_id is None:
            continue

        bdf = _bdf_from_device_dir(device_dir)
        vram_vendor = read_text(device_dir / "mem_info_vram_vendor")
        kind = _classify(device_dir, device_id, bdf, vram_vendor, kind_override)
        pci_dir = pci_root / bdf if bdf else None
        boot_vga = pci_dir is not None and read_text(pci_dir / "boot_vga") == "1"
        resolved = resolve(device_dir) or device_dir
        hwmon_dir = _resolve_hwmon(device_dir)
        resolved_hwmon = resolve(hwmon_dir) if hwmon_dir is not None else None

        device = AmdgpuDevice(
            index=0,
            card=entry.name,
            render=_find_render_node(device_dir),
            bdf=bdf,
            vendor_id=vendor_id,
            device_id=device_id,
            device_dir=resolved,
            hwmon_dir=resolved_hwmon,
            kind=kind,
            name=overrides.get(bdf) or describe_device_id(device_id),
            vram_vendor=vram_vendor,
            boot_vga=boot_vga,
        )
        found.append((kind == "igpu", bdf, device))

    found.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        replace(device, index=position) for position, (_, _, device) in enumerate(found)
    )
