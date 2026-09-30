"""Throttle status bit decoding.

The layout mirrors ``drivers/gpu/drm/amd/pm/inc/amdgpu_smu.h``.  Bits are sparse (0-7,
16-23, 32-47, 56-57) so a plain dict of ``bit -> label`` is clearer and more robust than
assuming contiguous ranges.
"""

from enum import IntFlag


class ThrottleStatus(IntFlag):
    """Power and thermal throttle reasons reported by ``gpu_metrics``."""

    PPT0 = 1 << 0
    PPT1 = 1 << 1
    PPT2 = 1 << 2
    PPT3 = 1 << 3
    SPL = 1 << 4
    FPPT = 1 << 5
    SPPT = 1 << 6
    SPPT_APU = 1 << 7
    TDC_GFX = 1 << 16
    TDC_SOC = 1 << 17
    TDC_MEM = 1 << 18
    TDC_VDD = 1 << 19
    TDC_CVIP = 1 << 20
    EDC_CPU = 1 << 21
    EDC_GFX = 1 << 22
    APCC = 1 << 23
    TEMP_GPU = 1 << 32
    TEMP_CORE = 1 << 33
    TEMP_MEM = 1 << 34
    TEMP_EDGE = 1 << 35
    TEMP_HOTSPOT = 1 << 36
    TEMP_SOC = 1 << 37
    TEMP_VR_GFX = 1 << 38
    TEMP_VR_SOC = 1 << 39
    TEMP_VR_MEM0 = 1 << 40
    TEMP_VR_MEM1 = 1 << 41
    TEMP_LIQUID0 = 1 << 42
    TEMP_LIQUID1 = 1 << 43
    VRHOT0 = 1 << 44
    VRHOT1 = 1 << 45
    PROCHOT_CPU = 1 << 46
    PROCHOT_GFX = 1 << 47
    PPM = 1 << 56
    FIT = 1 << 57


def decode_throttle(raw: int) -> frozenset[str]:
    """Return the names of every throttle bit set in ``raw``.

    Unknown bits are ignored rather than raising, so a newer kernel that adds a reason
    still decodes every reason we do understand.
    """
    if raw <= 0:
        return frozenset()
    # ``IntFlag.name`` is typed ``str | None`` because a flag may have no name; every
    # member declared above does, so the ``None`` case is filtered rather than asserted.
    return frozenset(
        name
        for name in (flag.name for flag in ThrottleStatus if raw & int(flag))
        if name is not None
    )
