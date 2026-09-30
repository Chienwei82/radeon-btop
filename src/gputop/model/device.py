"""GPU identity: a single concrete ``AmdgpuDevice`` record.

Per the specification there is deliberately **no** vendor abstraction layer.  This is a
plain frozen dataclass describing one ``amdgpu`` device discovered under
``/sys/class/drm``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gputop.model.aliases import Bdf

type DeviceKind = Literal["igpu", "dgpu"]

#: Human readable names for a *small, verified* set of AMD parts.
#:
#: Mapping PCI device ids to marketing names is inherently a data-maintenance problem, and
#: shipping a guessed or stale table would print confidently wrong names.  We therefore only
#: include ids that have been verified against real hardware or are unambiguous parts, and
#: render everything else as ``AMD Radeon [0x....]``.  Users can extend the mapping through
#: the ``[gpu.names]`` table in the TOML config without touching the code.
DEVICE_NAMES: dict[int, str] = {
    # Verified on the reference machine used to develop gputop.
    0x73BF: "AMD Radeon Navi 44",
    # Vega / GFX9 (consumer Vega and Frontier Edition).
    0x6860: "AMD Radeon RX Vega 64",
    0x6861: "AMD Radeon RX Vega 56",
    0x687F: "AMD Radeon Vega Frontier Edition",
    # Renoir / Cezanne APUs (Zen 2 / Zen 3).
    0x1638: "AMD Radeon Graphics (Cezanne)",
    0x164E: "AMD Radeon Graphics (Renoir)",
}

#: Coarse device ids that are known to be integrated GPUs (APUs).
APU_DEVICE_IDS: frozenset[int] = frozenset(
    {
        0x1550,
        0x1551,
        0x1552,
        0x1553,
        0x1554,
        0x1555,
        0x1556,
        0x1557,
        0x1558,
        0x1559,
        0x155A,
        0x155B,
        0x155C,
        0x155D,
        0x155E,
        0x155F,
        0x150C,
        0x150E,
        0x151C,
        0x1586,
        0x1638,
        0x163E,
        0x1640,
        0x164E,
        0x15BF,
        0x15C1,
    }
)


def describe_device_id(device_id: int) -> str:
    """Return a human readable name for a PCI device id.

    Args:
        device_id: The 16-bit PCI device id read from ``device/device``.

    Returns:
        A marketing name when known, otherwise ``AMD Radeon [0x....]`` so the UI always
        has something sensible to show.
    """
    return DEVICE_NAMES.get(device_id, f"AMD Radeon [0x{device_id:04x}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class AmdgpuDevice:
    """An immutable description of one ``amdgpu`` device.

    Attributes:
        index: Zero based position in the sorted device list.
        card: The DRM card node name, e.g. ``card1``.
        render: The render node name, e.g. ``renderD128``, or ``None`` if absent.
        bdf: PCI address in domain:bus:device.function form, e.g. ``0000:0c:00.0``.
        vendor_id: PCI vendor id; always ``0x1002`` for an accepted device.
        device_id: PCI device id.
        device_dir: Resolved path to the device's sysfs directory.
        hwmon_dir: Resolved path to the matching hwmon device, or ``None``.
        kind: ``"igpu"`` for integrated GPUs, ``"dgpu"`` for discrete ones.
        name: Human readable marketing name.
        vram_vendor: The value of ``mem_info_vram_vendor`` (e.g. ``samsung``), or ``None``.
        boot_vga: Whether this device is the boot display adapter.
    """

    index: int
    card: str
    render: str | None
    bdf: Bdf
    vendor_id: int
    device_id: int
    device_dir: Path
    hwmon_dir: Path | None
    kind: DeviceKind
    name: str
    vram_vendor: str | None
    boot_vga: bool

    @property
    def is_integrated(self) -> bool:
        """Whether this device is an APU/integrated GPU."""
        return self.kind == "igpu"

    @property
    def supports_fan(self) -> bool:
        """Integrated GPUs never have a controllable fan, so the panel can be hidden."""
        return not self.is_integrated

    def label(self) -> str:
        """Short label used in compact UI locations."""
        return f"{self.name} ({self.bdf})"
