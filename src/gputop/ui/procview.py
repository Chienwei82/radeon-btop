"""The per-process table's logic, with no Textual in sight.

Everything the process table *decides* lives here -- which column is sorted, which rows
survive the filter, where a child sits under its parent, what each cell says -- and none
of it is a widget.  The widget in :mod:`gputop.ui.proctable` only turns the rows it is
handed into a :class:`~textual.widgets.DataTable`.

That split is what makes the interesting behaviour testable at all.  Sorting stability,
"where do children of a filtered-out parent go", "does a ppid cycle loop forever" are
questions with crisp answers, and none of them need an event loop, a terminal or a GPU to
answer them.  It is also why the module imports :mod:`gputop.ui.format` but not Textual:
formatting is a pure function of a value, and a widget is not.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from gputop.model.process import SORT_COLUMNS, GpuProcess, SortColumn
from gputop.ui.format import fmt_bytes, fmt_percent

#: The displayed columns, in order.  This is the table's shape and the contract the
#: process panel's width budget is built against.  ``total`` is sortable but not shown;
#: see :data:`gputop.model.process.SORT_COLUMNS`.
COLUMN_ORDER: tuple[SortColumn, ...] = (
    "pid",
    "user",
    "command",
    "gfx",
    "compute",
    "dma",
    "encdc",
    "vram",
    "gtt",
)

COLUMN_TITLES: Mapping[SortColumn, str] = {
    "total": "Σ",
    "pid": "PID",
    "user": "USER",
    "command": "COMMAND",
    "gfx": "GFX %",
    "compute": "COMPUTE %",
    "dma": "DMA %",
    "encdc": "ENC/DEC %",
    "vram": "VRAM",
    "gtt": "GTT",
}

#: Column widths, in cells.  ``command`` takes the slack because it is the only column
#: whose content is unbounded, and a process name is what a user actually scans for.
COLUMN_WIDTHS: Mapping[SortColumn, int] = {
    "pid": 7,
    "user": 10,
    "command": 26,
    "gfx": 6,
    "compute": 9,
    "dma": 6,
    "encdc": 9,
    "vram": 9,
    "gtt": 9,
}

#: Columns whose value is an ordered magnitude, so a new selection starts descending --
#: largest first, which is what every "top processes" view is read as.
_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {"total", "gfx", "compute", "dma", "encdc", "vram", "gtt", "pid"}
)

#: Tree drawing glyphs, matching the block style used by the bars so the panel reads as
#: one system rather than two.
TREE_BRANCH = "├─ "
TREE_LAST = "└─ "
TREE_BAR = "│  "
TREE_GAP = "   "


def _describe(column: SortColumn) -> str:
    """Human label for a column, used in the panel header and in errors."""
    return COLUMN_TITLES.get(column, str(column))


def _check(column: str) -> SortColumn | None:
    """Narrow an arbitrary string to a known column, or ``None``."""
    return column if column in SORT_COLUMNS else None


@dataclass(frozen=True, slots=True, kw_only=True)
class SortState:
    """Which column the table is ordered by, and in which direction.

    Attributes:
        column: The sort key.  See :data:`SORT_COLUMNS`.
        reverse: ``True`` for largest or last first.
    """

    column: SortColumn = "total"
    reverse: bool = True

    @classmethod
    def parse(cls, value: str, *, reverse: bool | None = None) -> SortState:
        """Build a state from a config string, falling back to the default.

        A config file and a persisted state file are both things a user can edit by
        hand, so an unrecognised column has to degrade rather than refuse to start.

        ``reverse=None`` means "no direction was recorded" and takes each column's
        natural one -- largest first for a magnitude, A-Z for text.  That is what makes
        a first run put the busiest process at the top while still letting a saved
        ascending order survive the next start.
        """
        column = _check(value.strip().lower()) or "total"
        if reverse is None:
            return cls(column=column, reverse=column in _NUMERIC_COLUMNS)
        return cls(column=column, reverse=reverse)

    def with_column(self, column: SortColumn) -> SortState:
        """Select a column, or flip the direction when it is already selected.

        Re-selecting the active column to reverse it is the convention every sortable
        table shares, and it is the only way to reach the opposite direction without
        reaching for the dedicated key.
        """
        if column == self.column:
            return self.toggled()
        return replace(self, column=column, reverse=column in _NUMERIC_COLUMNS)

    def toggled(self) -> SortState:
        """Return the same ordering with the direction reversed."""
        return replace(self, reverse=not self.reverse)

    def cycled(self, step: int = 1) -> SortState:
        """Advance to the next column, wrapping at either end.

        Each new column starts in its natural direction rather than inheriting the
        previous one: carrying "reverse" across a change of key makes the next column
        read backwards for a reason the user cannot see.
        """
        index = SORT_COLUMNS.index(self.column)
        column = SORT_COLUMNS[(index + step) % len(SORT_COLUMNS)]
        return replace(self, column=column, reverse=column in _NUMERIC_COLUMNS)

    @property
    def arrow(self) -> str:
        """The direction glyph shown next to the active column."""
        return "▾" if self.reverse else "▴"

    def describe(self) -> str:
        """One-line summary, e.g. ``total ▾``.

        The *config key* is used rather than the column heading, so what the panel says
        and what ``process.sort`` holds are visibly the same string.
        """
        return f"{self.column} {self.arrow}"

    def title_for(self, column: SortColumn) -> str:
        """Column heading carrying the sort marker, for the table header."""
        label = _describe(column)
        return f"{label} {self.arrow}" if column == self.column else label


#: Default sort: total engine load, busiest first.  Matches SPEC FR4.
DEFAULT_SORT = SortState()


@dataclass(frozen=True, slots=True, kw_only=True)
class FilterState:
    """Which rows the table shows at all.

    Attributes:
        text: Case-insensitive substring; empty means "match everything".
        show_idle: When false, clients doing nothing measurable are hidden.  This is the
            default: a client that merely holds a descriptor is not using the GPU.
        min_engine_percent: Drop anything below this total engine utilisation.
        hide_kernel_threads: Drop clients with no argument vector.
        bdf: Restrict to one GPU, or ``None`` for every GPU at once.  The multi-GPU
            switcher scopes the table to the focused device; the overview mode does not.
    """

    text: str = ""
    show_idle: bool = False
    min_engine_percent: float = 0.0
    hide_kernel_threads: bool = True
    bdf: str | None = None

    @classmethod
    def parse(cls, value: str) -> FilterState:
        """Build a filter from a persisted/config string."""
        return cls(text=value)

    def with_text(self, text: str) -> FilterState:
        """Return a copy filtering on ``text``."""
        return replace(self, text=text)

    def toggled_idle(self) -> FilterState:
        """Return a copy that also shows, or hides, idle clients."""
        return replace(self, show_idle=not self.show_idle)

    def for_device(self, bdf: str | None) -> FilterState:
        """Return a copy scoped to one GPU, or to all of them when ``bdf`` is ``None``."""
        return replace(self, bdf=bdf)

    @property
    def needle(self) -> str:
        """The filter text, lowercased, with surrounding space removed.

        Stripping here rather than at match time means an accidental trailing space in
        the prompt does not silently hide every row.
        """
        return self.text.strip().lower()

    def matches(self, process: GpuProcess) -> bool:
        """Whether one client survives this filter."""
        if self.bdf is not None and process.bdf != self.bdf:
            return False
        if self.hide_kernel_threads and process.is_kernel_thread:
            return False
        if not self.show_idle and process.is_idle:
            return False
        if process.engine_percent < self.min_engine_percent:
            return False
        needle = self.needle
        return not needle or _matches_text(process, needle)

    def describe(self) -> str:
        """One-line summary for the panel header.

        Every clause has to be here, not just the two the user can type.  :meth:`matches`
        also drops rows below ``min_engine_percent`` and, by default, every kernel thread,
        and the count the header shows is the number that *survived* the filter -- so a
        title reading "all" above a table missing half its rows is a table quietly lying
        about how complete it is.  "Anything that is not on screen has to be on this line"
        is the rule, and the only way to keep it is to enumerate the filters rather than
        the keys.
        """
        parts = [f"filter {self.text!r}"] if self.text.strip() else ["all"]
        parts.append("idle shown" if self.show_idle else "idle hidden")
        if self.min_engine_percent > 0:
            parts.append(f"under {self.min_engine_percent:g}% engine hidden")
        if self.hide_kernel_threads:
            parts.append("kernel threads hidden")
        return "  ·  ".join(parts)


def _matches_text(process: GpuProcess, needle: str) -> bool:
    """Match the filter against every field a user might plausibly type.

    The command line is searched as well as the name, because ``comm`` is truncated to
    15 characters and matching only that would make ``blender`` find ``blender`` but not
    ``blender-launcher`` or a versioned binary path.  The PID is searched as a substring
    so a number can be pasted straight from another terminal.
    """
    return (
        needle in process.name.lower()
        or needle in process.command.lower()
        or needle in process.user.lower()
        or needle in str(process.pid)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessRow:
    """One rendered line of the table.

    Attributes:
        process: The client this row shows.
        depth: Tree depth; 0 for a root.
        has_children: Whether the row has nested clients.  Purely descriptive: the
            connector comes from :attr:`is_last`, and the children's guides from their
            parent's ``last``.
        is_last: Whether the row is the final entry of its group, which decides whether
            its own connector is ``└`` or ``├``.
        guides: One flag per ancestor level: ``True`` means an ancestor still has
            following siblings, so a vertical line is drawn at that depth.
    """

    process: GpuProcess
    depth: int = 0
    has_children: bool = False
    is_last: bool = True
    guides: tuple[bool, ...] = ()

    @property
    def prefix(self) -> str:
        """The tree guides for this row, empty at depth 0."""
        if self.depth == 0:
            return ""
        drawn = "".join(TREE_BAR if guide else TREE_GAP for guide in self.guides)
        return drawn + (TREE_LAST if self.is_last else TREE_BRANCH)


def column_value(process: GpuProcess, column: SortColumn) -> float | int | str | None:
    """The value a column sorts and filters by, or ``None`` when it is unknown.

    ``None`` is meaningful and is *not* zero: a client seen for the first time has no
    engine baseline, and ``0%`` would claim it is idle when the truth is "not measured
    yet".
    """
    match column:
        case "total":
            return process.engine_percent
        case "pid":
            return process.pid
        case "user":
            return process.user.lower()
        case "command":
            return process.command.lower()
        case "gfx":
            return process.engine_percent_for("gfx")
        case "compute":
            return process.engine_percent_for("compute")
        case "dma":
            return process.engine_percent_for("dma")
        case "encdc":
            return process.enc_dec_percent
        case "vram":
            return process.vram_used
        case "gtt":
            return process.gtt_used


def sort_processes(
    processes: Iterable[GpuProcess], state: SortState = DEFAULT_SORT
) -> tuple[GpuProcess, ...]:
    """Order clients by ``state``.

    Two rules make the result predictable enough to be worth stating:

    * **Unknown values sort last, in both directions.**  Flipping a column must not
      shuffle a column full of ``N/A`` up above the rows that do have a reading; the
      absence of a measurement is a statement about the hardware, not a small number.
    * **Ties break on ``(pid, bdf, client_id)``.**  Two clients can report the same
      percentage, and a table whose rows swap places between one-second samples is
      unreadable.  The tie-break is the same identity the kernel gave us, so a row stays
      where it was.
    """
    keyed: list[tuple[float | int | str, GpuProcess]] = []
    unmeasured: list[GpuProcess] = []
    for process in processes:
        value = column_value(process, state.column)
        if value is None:
            unmeasured.append(process)
        else:
            keyed.append((value, process))

    # Two passes, both stable.  Reversing a single composite key would also reverse the
    # tie-break, so equal rows would swap places the moment the direction changed; this
    # way the ordering by value flips and the ordering *within* a tie does not.
    keyed.sort(key=lambda item: (item[1].pid, item[1].bdf, item[1].client_id))
    keyed.sort(key=lambda item: item[0], reverse=state.reverse)
    return tuple(process for _value, process in keyed) + tuple(unmeasured)


def filter_processes(
    processes: Iterable[GpuProcess], state: FilterState | None = None
) -> tuple[GpuProcess, ...]:
    """Keep only the clients that satisfy ``state``."""
    active = state or FilterState()
    return tuple(process for process in processes if active.matches(process))


def flat_rows(processes: Iterable[GpuProcess]) -> tuple[ProcessRow, ...]:
    """Wrap each client as a depth-0 row, with no sorting of its own."""
    return tuple(
        ProcessRow(process=process, depth=0, has_children=False, is_last=True)
        for process in processes
    )


def tree_rows(
    processes: Iterable[GpuProcess], state: SortState = DEFAULT_SORT
) -> tuple[ProcessRow, ...]:
    """Arrange clients into a parent/child tree, ordered by ``state``.

    The tree is built from ``PPid``, which means it inherits everything ``PPid`` is:
    a parent that is not itself a GPU client is simply absent, and its children become
    roots.  That is the common case -- a compositor holding the GPU is the child of
    whatever session manager started it -- and showing those as roots is correct rather
    than a loss.

    Three properties are load-bearing and are asserted by the tests:

    * **Every client appears exactly once.**  A ``PPid`` cycle, a process that is its own
      parent, and a client whose parent is not a client all have to terminate somewhere.
    * **The guides line up.**  A row at depth *d* draws *d - 1* vertical guides, one per
      ancestor that still has siblings below it, and a root draws none.
    * **One process with several clients is one node.**  A Vulkan app can hold two
      ``drm-client-id``s; those are separate rows but the same process, so they sit
      beside each other at the same depth instead of nesting under one another.
    * **The ordering applies at every level.**  Children are ordered by the same rule as
      roots, so sorting by VRAM does not leave one branch sorted and another not.
    """
    ordered = sort_processes(processes, state)
    groups: dict[int, list[GpuProcess]] = {}
    for process in ordered:
        groups.setdefault(process.pid, []).append(process)

    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for pid, members in groups.items():
        parent = members[0].ppid
        if parent is None or parent == pid or parent not in groups:
            roots.append(pid)
        else:
            children.setdefault(parent, []).append(pid)

    rows: list[ProcessRow] = []
    emitted: set[int] = set()

    def emit(
        pid: int,
        depth: int,
        guides: tuple[bool, ...],
        seen: frozenset[int],
        last: bool,
    ) -> None:
        """Walk one node's subtree, appending rows depth first.

        ``last`` is this node's position among its siblings, and decides whether its own
        connector is a tee or an elbow.  A root is always last: nothing follows it.
        """
        if pid in seen:
            # A cycle: this node is already on the current path, so descending again
            # would recurse forever.  It is left where it is.
            return
        members = groups[pid]
        emitted.add(pid)
        kids = tuple(
            child for child in children.get(pid, ()) if child not in seen and child != pid
        )
        for index, member in enumerate(members):
            rows.append(
                ProcessRow(
                    process=member,
                    depth=depth,
                    has_children=bool(kids),
                    # The final entry of the group draws the elbow whether or not the
                    # group has children below it.  An earlier ``and not kids`` here made
                    # every parent draw a tee even as the last sibling -- over a guide
                    # column that was empty beneath it, because the children's guides
                    # derive from ``last``, not from this.
                    is_last=last and index == len(members) - 1,
                    guides=guides,
                )
            )
        for position, child in enumerate(kids):
            # A guide says "this ancestor still has following siblings", and the ancestor
            # a child inherits is its *parent* -- which is why the flag is derived from
            # ``last`` (this node's own position) and not from the child's.  A root has no
            # line of its own, so its children inherit an empty guide list.
            child_guides = guides if depth == 0 else (*guides, not last)
            emit(
                child,
                depth + 1,
                child_guides,
                seen | {pid},
                last=position == len(kids) - 1,
            )

    for root in roots:
        emit(root, 0, (), frozenset(), last=True)
    # Anything left is part of a cycle no root could reach.  It is still real GPU work,
    # so it is shown at the top level rather than dropped.
    for pid in groups:
        if pid not in emitted:
            emit(pid, 0, (), frozenset(), last=True)
    return tuple(rows)


def build_rows(
    processes: Iterable[GpuProcess],
    *,
    sort: SortState = DEFAULT_SORT,
    filters: FilterState | None = None,
    tree: bool = False,
) -> tuple[ProcessRow, ...]:
    """Filter, order and shape the rows in one call.

    This is the single entry point the widget uses, so "what the user sees" is one pure
    function of "what the sampler saw" plus the view state.
    """
    visible = filter_processes(processes, filters)
    return tree_rows(visible, sort) if tree else flat_rows(sort_processes(visible, sort))


def busiest_processes(
    processes: Sequence[GpuProcess], *, sort: SortState = DEFAULT_SORT
) -> tuple[GpuProcess, ...]:
    """The clients tied for the most GPU work, or nothing when nothing is busy.

    A tie returns every contender, not just the first: two processes at 40% are both
    the answer, and highlighting one arbitrarily would misreport what the GPU is doing.
    An idle table highlights nothing, because "the busiest process" is then a statement
    about nothing at all.

    Ranked by magnitude, not by table position: ``sort`` selects the *column*, and its
    direction is deliberately ignored.  Taking the first sorted row marked the *least*
    busy client whenever the user reversed the sort, and claiming that row was the top
    consumer of the GPU.  A non-measurement column -- ``pid``, ``user``, ``command`` --
    has no "most" to find and yields nothing; see :data:`NUMERIC_SORT_COLUMNS`.
    """
    if not processes or sort.column not in NUMERIC_SORT_COLUMNS:
        return ()
    values = [
        value
        for process in processes
        if isinstance(value := column_value(process, sort.column), (int, float))
    ]
    if not values:
        return ()
    best = max(values)
    if best <= 0:
        return ()
    return tuple(process for process in processes if column_value(process, sort.column) == best)


def cell_value(process: GpuProcess, column: SortColumn, *, full_command: bool = False) -> str:
    """The rendered text of one cell.

    Percentages keep one decimal because the interesting range is 0-5%: on an otherwise
    idle desktop the difference between "a video is playing" and "nothing is happening"
    is a fraction of a percent, and ``0%`` would erase it.  Memory uses the shared byte
    formatter so the table and the memory panel never disagree about units.

    The command cell shows the 15-character ``comm`` name by default and the full
    ``/proc/<pid>/cmdline`` on request, because the full line is what identifies a
    process unambiguously and is far too wide to show unasked.
    """
    match column:
        case "pid":
            return str(process.pid)
        case "user":
            return process.user
        case "command":
            return process.command if full_command else process.name
        case "gfx":
            return fmt_percent(process.engine_percent_for("gfx"), 1)
        case "compute":
            return fmt_percent(process.engine_percent_for("compute"), 1)
        case "dma":
            return fmt_percent(process.engine_percent_for("dma"), 1)
        case "encdc":
            return fmt_percent(process.enc_dec_percent, 1)
        case "vram":
            return fmt_bytes(process.vram_used)
        case "gtt":
            return fmt_bytes(process.gtt_used)
        case "total":
            return fmt_percent(process.engine_percent, 1)


def row_cells(process: GpuProcess, *, full_command: bool = False) -> tuple[str, ...]:
    """The rendered cells of one row, in :data:`COLUMN_ORDER`."""
    return tuple(
        cell_value(process, column, full_command=full_command) for column in COLUMN_ORDER
    )


#: Sort columns that measure *how hard a client is working*, so ranking by one produces a
#: meaningful "top" and a meaningful "idle".  ``user`` and ``command`` rank by name, where
#: neither word means anything, and the panel has to stay silent rather than claim
#: otherwise.  ``pid`` is a number but not a measurement: it is an identifier, so ordering
#: by it makes the highest-numbered process "top" while it may be the *least* busy one on
#: screen, and marking that row with the leader glyph is a claim about the GPU.
NUMERIC_SORT_COLUMNS: frozenset[SortColumn] = frozenset(
    {"total", "gfx", "compute", "dma", "encdc", "vram", "gtt"}
)


def panel_title(
    sort: SortState,
    filters: FilterState,
    *,
    shown: int,
    total: int,
    top: Sequence[GpuProcess] = (),
    tree: bool = False,
) -> str:
    """The one-line summary above the table.

    It answers, in order, the four questions a user has when they look at this panel:
    how is it sorted, what is hidden, how much is hidden, and what is the GPU actually
    doing.  Anything that is not on screen has to be on this line, or the table is
    quietly lying about how complete it is.
    """
    parts = [f"sort {sort.describe()}", filters.describe()]
    if tree:
        parts.append("tree")
    if shown != total:
        parts.append(f"{shown} of {total}")
    if not top:
        # Only say "idle" when the ranking column is one that could have said otherwise.
        # Sorting by user or command ranks by name, so an empty ``top`` there means the
        # ranking says nothing about activity -- and announcing a GPU at 80% as idle is
        # worse than saying nothing.
        if sort.column in NUMERIC_SORT_COLUMNS:
            parts.append("idle")
    else:
        # The value printed is the one the ranking used, so a table ordered by VRAM does
        # not introduce its leader with a usage figure that has nothing to do with it.
        names = ", ".join(f"{p.name} {p.pid} {cell_value(p, sort.column)}" for p in top[:2])
        parts.append(f"top {names}")
    return "  ·  ".join(parts)


def table_width() -> int:
    """Total width of the table's columns, excluding the border and padding."""
    return sum(COLUMN_WIDTHS[column] for column in COLUMN_ORDER)


__all__ = [
    "COLUMN_ORDER",
    "COLUMN_TITLES",
    "COLUMN_WIDTHS",
    "DEFAULT_SORT",
    "NUMERIC_SORT_COLUMNS",
    "SORT_COLUMNS",
    "FilterState",
    "ProcessRow",
    "SortColumn",
    "SortState",
    "build_rows",
    "busiest_processes",
    "cell_value",
    "column_value",
    "filter_processes",
    "flat_rows",
    "panel_title",
    "row_cells",
    "sort_processes",
    "table_width",
    "tree_rows",
]
