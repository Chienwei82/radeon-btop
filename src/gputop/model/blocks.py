"""Per-block GPU utilisation, sourced from the ``radeontop`` subprocess.

This is the one part of the data layer that is *not* read from sysfs.  The amdgpu driver
exposes GRBM/SRBM busy counters only through the radeon ioctl on a privileged handle, which
is exactly the kind of access the rest of gputop refuses to require.  Rather than take that
dependency for the whole application, the block counters are optional: when the
``radeontop`` binary is present *and* usable they are shown, and when it is not the panel
disappears and the reason is stated.  Every other panel keeps working unprivileged.

The labels below are radeontop's own display names, taken from its ``ui.c``.  They are used
verbatim rather than renamed, because a user comparing gputop against radeontop side by side
should not have to learn that the same hardware block is spelled differently in each.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from gputop.model.aliases import Mhz, Nanoseconds, Percent

#: radeontop dump key -> display label.
#:
#: The keys are the names ``dump.c`` writes, and they are what identifies a block
#: everywhere else in gputop -- in the JSON dump, the CSV header and the log.  The *labels*
#: are radeontop's own display names from its ``ui.c``, lightly abbreviated to fit the
#: blocks panel's label column.
#:
#: The abbreviation is a width decision, not a correction.  radeontop's own panel is
#: full-screen; here it shares a row with the power-profile panel, and "Shader Interpolator"
#: alone would eat a sixth of the width before the bar started.  Abbreviating the label costs
#: nothing because the full radeontop key travels with every reading.
#:
#: Note that several keys do *not* match the AMD hardware block they report: radeontop labels
#: ``sx`` "Shader Export" where AMD calls that block the Shader Export/Interpolator unit, and
#: ``sh`` "Seq Inst Cache" where AMD calls it the Shader block.  radeontop's naming is kept
#: so that a user comparing the two tools side by side is not asked to learn that the same
#: hardware block is spelled differently in each.
BLOCK_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "gpu": "Graphics pipe",
        "ee": "Event Engine",
        "vgt": "Vertex Grouper",
        "ta": "Texture Addr",
        "tc": "Texture Cache",
        "sx": "Shader Export",
        "sh": "Seq Inst Cache",
        "spi": "Shader Interp",
        "smx": "Shader MemExch",
        "sc": "Scan Converter",
        "pa": "Prim Assembly",
        "db": "Depth Block",
        "cr": "Clip Rect",
        "cb": "Color Block",
        "uvd": "UVD",
        "vce0": "VCE",
    }
)

#: Width of the label column the blocks and profile panels reserve.
#:
#: Wider than :data:`~gputop.ui.panels.LABEL_WIDTH` because the block names need it, plus one
#: column of gutter so the bar never starts flush against the longest label.  Derived from
#: the labels above rather than hard-coded, so a label that outgrows it fails a test instead
#: of being discovered as an unreadable row on someone's terminal.
BLOCK_LABEL_WIDTH = max(len(label) for label in BLOCK_LABELS.values()) + 1

#: The order blocks are shown in, which is the order a triangle travels through the
#: graphics pipeline.  Grouping by stage is what makes the panel readable: two adjacent
#: busy blocks mean a stall between them, which is the single most useful thing this panel
#: can say and the reason it exists.
BLOCK_ORDER: tuple[str, ...] = (
    "gpu",
    "ee",
    "vgt",
    "ta",
    "tc",
    "sx",
    "sh",
    "spi",
    "smx",
    "sc",
    "pa",
    "db",
    "cr",
    "cb",
)

#: Blocks that are not part of the 3D pipeline.  They are shown after the pipeline, in a
#: separate group, so that an idle video decoder does not push the triangle path apart.
AUX_BLOCK_ORDER: tuple[str, ...] = ("uvd", "vce0")

#: Clock labels.  Kept apart from :data:`BLOCK_LABELS` because the dump parser dispatches on
#: ``key in BLOCK_LABELS`` to decide a field is a plain percentage, and a clock key in that
#: table would be consumed before its unit-bearing payload was read.
CLOCK_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "mclk": "Memory Clock",
        "sclk": "Shader Clock",
    }
)

#: The clock keys, which carry a percentage *and* an absolute frequency.
CLOCK_KEYS: tuple[str, ...] = tuple(CLOCK_LABELS)

#: Memory keys: a percentage and an absolute amount in megabytes.
MEMORY_KEYS: tuple[str, ...] = ("vram", "gtt")


def block_label(key: str) -> str:
    """Return the display label for a radeontop key.

    An unrecognised key is returned unchanged.  radeontop can grow a block before this
    table is updated, and showing a raw key beats hiding a real measurement.
    """
    label = BLOCK_LABELS.get(key)
    if label is not None:
        return label
    return CLOCK_LABELS.get(key, key)


class BlocksStatus(StrEnum):
    """Why the blocks panel is or is not showing data.

    Every member except :attr:`OK` is a way of saying "no", and each carries a different
    remedy, which is why they are distinct rather than one ``UNAVAILABLE``: telling a user
    to install a package when the real problem is a missing capability bit wastes their
    time and teaches them to ignore the message.
    """

    #: Data is flowing.
    OK = "ok"
    #: Switched off in the configuration.
    DISABLED = "disabled"
    #: The binary is not installed, or not on ``$PATH``.
    MISSING = "missing"
    #: radeontop ran but cannot use this card (no VRAM path, unknown family).
    UNSUPPORTED = "unsupported"
    #: radeontop ran and was refused access to the GPU registers.
    NOT_PERMITTED = "not permitted"
    #: The process died or produced nothing parseable.
    FAILED = "failed"

    @property
    def ok(self) -> bool:
        """Whether this status carries data."""
        return self is BlocksStatus.OK

    @property
    def hint(self) -> str:
        """A one-line explanation and remedy, or ``""`` when there is nothing to explain."""
        return _STATUS_HINTS[self]


_STATUS_HINTS: Mapping[BlocksStatus, str] = MappingProxyType(
    {
        BlocksStatus.OK: "",
        BlocksStatus.DISABLED: "blocks off (--blocks to enable)",
        BlocksStatus.MISSING: "no radeontop binary (install it, or set blocks.binary)",
        BlocksStatus.UNSUPPORTED: "radeontop cannot read this card",
        BlocksStatus.NOT_PERMITTED: "radeontop needs root to read the GPU registers",
        BlocksStatus.FAILED: "radeontop failed to produce data",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockReading:
    """One block's utilisation over the last sampling window."""

    key: str
    label: str
    percent: Percent


@dataclass(frozen=True, slots=True, kw_only=True)
class ClockReading:
    """One clock's utilisation, with the absolute frequency it corresponds to.

    radeontop reports both a percentage of the maximum and a frequency.  The percentage is
    what the bar needs; the frequency is what a user actually wants to read, so both are
    kept rather than deriving one from the other.
    """

    key: str
    label: str
    percent: Percent | None = None
    mhz: Mhz | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuBlocks:
    """A complete per-block sample for one device.

    Attributes:
        blocks: Pipeline blocks, ordered by :data:`BLOCK_ORDER`.
        aux_blocks: Video and compute blocks, ordered by :data:`AUX_BLOCK_ORDER`.
        clocks: Memory and shader clock utilisation.
        timestamp_ns: When radeontop produced the line this came from, in nanoseconds
            since the epoch.  It is radeontop's own stamp, not ours, so it records when
            the hardware was actually sampled -- but radeontop takes it from
            ``gettimeofday()``, so it is a **wall clock**, and it is not comparable with
            the monotonic ``timestamp_ns`` on a :class:`GpuSnapshot`.  Only :attr:`age_ns`
            expresses an interval, and it is only meaningful when the caller passes a
            wall-clock ``now_ns`` to match.
        age_ns: How stale the reading is relative to a caller-supplied wall-clock now.
    """

    blocks: tuple[BlockReading, ...] = ()
    aux_blocks: tuple[BlockReading, ...] = ()
    clocks: tuple[ClockReading, ...] = ()
    timestamp_ns: Nanoseconds = 0
    age_ns: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether this sample carries no blocks at all."""
        return not self.blocks and not self.aux_blocks

    def by_key(self) -> Mapping[str, Percent]:
        """Return the blocks keyed by radeontop key, for callers that look one up."""
        merged: dict[str, Percent] = {b.key: b.percent for b in self.blocks}
        merged.update({b.key: b.percent for b in self.aux_blocks})
        return MappingProxyType(merged)

    def busiest(self) -> BlockReading | None:
        """The busiest block, or ``None`` when there are none.

        This is the number the panel puts in its heading: "which block" is the question a
        user opens this panel with, and answering it in the title means the bars are detail
        rather than the thing that must be read.
        """
        candidates = (*self.blocks, *self.aux_blocks)
        if not candidates:
            return None
        return max(candidates, key=lambda block: block.percent)

    def busiest_key(self) -> str | None:
        """The busiest block's radeontop key, or ``None`` when there are no blocks.

        The serialisers need the key rather than the reading, and three separate copies of
        "unpack the optional reading, then take its key" is two more places to keep in step.
        """
        reading = self.busiest()
        return None if reading is None else reading.key

    def display_rows(self) -> tuple[tuple[BlockReading | ClockReading, ...], ...]:
        """Group the readings into pipeline blocks, auxiliary blocks and clocks.

        Returns:
            Non-empty groups in presentation order, so a caller can render them in order
            without checking which are empty.
        """
        groups: list[tuple[BlockReading | ClockReading, ...]] = []
        if self.blocks:
            groups.append(self.blocks)
        if self.aux_blocks:
            groups.append(self.aux_blocks)
        if self.clocks:
            groups.append(self.clocks)
        return tuple(groups)


__all__ = [
    "AUX_BLOCK_ORDER",
    "BLOCK_LABELS",
    "BLOCK_ORDER",
    "CLOCK_KEYS",
    "CLOCK_LABELS",
    "MEMORY_KEYS",
    "BlockReading",
    "BlocksStatus",
    "ClockReading",
    "GpuBlocks",
    "block_label",
]
