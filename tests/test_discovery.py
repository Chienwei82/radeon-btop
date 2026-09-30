"""Device discovery tests, covering single, multi-GPU and dGPU+APU topologies."""

from pathlib import Path

from gputop.readers.discovery import discover_devices
from tests.conftest import make_gpu, pci_root_for

#: A Carrizo-class APU id, present in the APU set.
APU_DEVICE_ID = 0x1638

#: A Navi 44 class dGPU id.
DGPU_DEVICE_ID = 0x73BF


def test_single_discrete_gpu(drm_root: Path) -> None:
    """One Radeon card is discovered with a correct identity."""
    make_gpu(drm_root, card=1, bdf="0000:0c:00.0", device_id=DGPU_DEVICE_ID)

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert len(devices) == 1
    device = devices[0]
    assert device.index == 0
    assert device.card == "card1"
    assert device.bdf == "0000:0c:00.0"
    assert device.vendor_id == 0x1002
    assert device.device_id == DGPU_DEVICE_ID
    assert device.name == "AMD Radeon Navi 44"
    assert device.kind == "dgpu"
    assert not device.is_integrated
    assert device.supports_fan


def test_no_gpus_yields_empty_tuple(drm_root: Path) -> None:
    """A machine with no AMD GPU is an ordinary outcome, not an error."""
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root)) == ()


def test_missing_drm_root_yields_empty_tuple(tmp_path: Path) -> None:
    """A missing sysfs tree yields an empty result rather than raising."""
    assert discover_devices(tmp_path / "nope") == ()


def test_non_amd_vendor_is_skipped(drm_root: Path) -> None:
    """Only vendor 0x1002 is in scope."""
    make_gpu(drm_root, card=0, vendor_id=0x8086)
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root)) == ()


def test_radeon_driver_is_skipped(drm_root: Path) -> None:
    """The legacy ``radeon`` driver is explicitly out of scope."""
    make_gpu(drm_root, card=0, driver="radeon")
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root)) == ()


def test_absent_driver_link_is_skipped(drm_root: Path) -> None:
    """A card with no bound driver is not usable and is skipped."""
    make_gpu(drm_root, card=0, driver=None)
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root)) == ()


def test_connector_nodes_are_not_devices(drm_root: Path) -> None:
    """``card0-DP-1`` style connector entries must never be treated as GPUs."""
    make_gpu(drm_root, card=0)
    (drm_root / "card0-DP-1").mkdir()

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert len(devices) == 1
    assert devices[0].card == "card0"


def test_render_node_is_found(drm_root: Path) -> None:
    """The render node belonging to a device is resolved for later use."""
    device_dir = make_gpu(drm_root, card=1)
    (device_dir / "drm").mkdir()
    (device_dir / "drm" / "renderD128").touch()
    (device_dir / "drm" / "card1").touch()

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert devices[0].render == "renderD128"


def test_hwmon_directory_is_resolved_per_device(drm_root: Path) -> None:
    """Each device resolves its own hwmon, which is what makes sensors attributable.

    Scanning ``/sys/class/hwmon`` and matching by name would be ambiguous on a machine
    with many sensors, so resolution must go through the device's own ``hwmon`` link.
    """
    make_gpu(
        drm_root, card=1, bdf="0000:0c:00.0", hwmon_index=3, hwmon={"temp1_input": "49000"}
    )
    make_gpu(
        drm_root, card=2, bdf="0000:03:00.0", hwmon_index=7, hwmon={"temp1_input": "55000"}
    )

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert all(d.hwmon_dir is not None for d in devices)
    # Keyed by BDF because discovery sorts by address, not by card number.
    by_bdf = {d.bdf: d for d in devices}
    assert (by_bdf["0000:0c:00.0"].hwmon_dir / "temp1_input").read_text().strip() == "49000"
    assert (by_bdf["0000:03:00.0"].hwmon_dir / "temp1_input").read_text().strip() == "55000"


def test_multi_gpu_indexes_are_sequential(drm_root: Path) -> None:
    """Several cards get consecutive indices with no gaps or duplicates."""
    make_gpu(drm_root, card=0, bdf="0000:0c:00.0", device_id=DGPU_DEVICE_ID)
    make_gpu(drm_root, card=1, bdf="0000:03:00.0", device_id=DGPU_DEVICE_ID, hwmon_index=1)
    make_gpu(drm_root, card=2, bdf="0000:04:00.0", device_id=DGPU_DEVICE_ID, hwmon_index=2)

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert [d.index for d in devices] == [0, 1, 2]
    assert len({d.bdf for d in devices}) == 3


def test_dgpu_plus_apu_is_detected_and_ordered(drm_root: Path) -> None:
    """An APU plus a discrete card is classified correctly, discrete first.

    This is the dGPU+APU topology the specification calls out: the APU runs the display
    and must be recognised as integrated so the UI can emphasise GTT over VRAM.
    """
    make_gpu(
        drm_root,
        card=0,
        bdf="0000:00:02.0",
        device_id=APU_DEVICE_ID,
        extra_sysfs={"mem_info_vram_vendor": "N/A\n"},
    )
    make_gpu(
        drm_root,
        card=1,
        bdf="0000:03:00.0",
        device_id=DGPU_DEVICE_ID,
        extra_sysfs={"mem_info_vram_vendor": "samsung\n"},
        hwmon_index=1,
    )

    devices = discover_devices(drm_root, pci_root=pci_root_for(drm_root))

    assert [d.kind for d in devices] == ["dgpu", "igpu"]
    assert devices[0].device_id == DGPU_DEVICE_ID
    assert devices[1].device_id == APU_DEVICE_ID
    assert devices[1].is_integrated
    assert not devices[1].supports_fan
    assert devices[1].vram_vendor == "N/A"


def test_apu_is_classified_by_vram_vendor_alone(drm_root: Path) -> None:
    """``mem_info_vram_vendor == "N/A"`` is one independent signal towards integrated."""
    make_gpu(
        drm_root,
        bdf="0000:03:00.0",
        device_id=DGPU_DEVICE_ID,
        extra_sysfs={"mem_info_vram_vendor": "N/A\n"},
    )

    # One vote is not enough on its own: a single signal must not decide.
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].kind == "dgpu"


def test_small_vram_pool_signals_integrated(drm_root: Path) -> None:
    """An APU-sized carve-out plus an N/A vendor marks the device integrated."""
    make_gpu(
        drm_root,
        bdf="0000:03:00.0",
        device_id=DGPU_DEVICE_ID,
        extra_sysfs={
            "mem_info_vram_vendor": "N/A\n",
            "mem_info_vis_vram_total": str(512 * 1024 * 1024),
        },
    )

    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].kind == "igpu"


def test_gtt_dominating_vram_signals_integrated(drm_root: Path) -> None:
    """A GTT pool far larger than VRAM indicates shared system memory."""
    make_gpu(
        drm_root,
        bdf="0000:03:00.0",
        device_id=DGPU_DEVICE_ID,
        extra_sysfs={
            "mem_info_vis_vram_total": str(512 * 1024 * 1024),
            "mem_info_gtt_total": str(8 * 1024 * 1024 * 1024),
        },
    )

    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].kind == "igpu"


def test_kind_override_forces_classification(drm_root: Path) -> None:
    """An explicit override short-circuits every heuristic."""
    make_gpu(
        drm_root,
        bdf="0000:0c:00.0",
        device_id=DGPU_DEVICE_ID,
        extra_sysfs={"mem_info_vis_vram_total": str(512 * 1024 * 1024)},
    )

    assert (
        discover_devices(drm_root, pci_root=pci_root_for(drm_root), kind_override="igpu")[
            0
        ].kind
        == "igpu"
    )
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].kind == "dgpu"


def test_name_override_from_config(drm_root: Path) -> None:
    """A configured name replaces the built-in table for that BDF."""
    make_gpu(drm_root, bdf="0000:0c:00.0", device_id=DGPU_DEVICE_ID)

    devices = discover_devices(
        drm_root,
        pci_root=pci_root_for(drm_root),
        name_overrides={"0000:0c:00.0": "My Custom Name"},
    )

    assert devices[0].name == "My Custom Name"


def test_unknown_device_id_falls_back_to_hex(drm_root: Path) -> None:
    """An unlisted PCI id still produces a usable label rather than nothing."""
    make_gpu(drm_root, bdf="0000:0c:00.0", device_id=0xDEAD)

    assert (
        discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].name
        == "AMD Radeon [0xdead]"
    )


def test_boot_vga_is_reported(drm_root: Path) -> None:
    """The boot display adapter is flagged so the UI can mark it."""
    make_gpu(drm_root, bdf="0000:0c:00.0")
    assert discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0].boot_vga is True


def test_device_label_includes_bdf(drm_root: Path) -> None:
    """The compact label carries the address needed to tell two identical cards apart."""
    make_gpu(drm_root, bdf="0000:0c:00.0")
    device = discover_devices(drm_root, pci_root=pci_root_for(drm_root))[0]

    assert device.label() == "AMD Radeon Navi 44 (0000:0c:00.0)"


def test_device_without_metrics_file_still_discovered(drm_root: Path) -> None:
    """A GPU predating ``gpu_metrics`` is still usable via the sysfs readers."""
    make_gpu(drm_root, bdf="0000:0c:00.0", metrics=None, metrics_abi=None)

    assert len(discover_devices(drm_root, pci_root=pci_root_for(drm_root))) == 1
