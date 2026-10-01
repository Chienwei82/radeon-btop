"""The per-process table widget.

The decisions -- which rows, in which order, saying what -- are made by
:mod:`gputop.ui.procview`.  This module is the part that needs a terminal: the
:class:`~textual.widgets.DataTable`, the header that turns a click into a sort request,
and the machinery that keeps a selection attached to a process across samples.

That last part is the only non-obvious thing here.  A new sample arrives every second
and the table is rebuilt from scratch, so a cursor stored as a row index would drift onto
a different process every time a row appeared or disappeared.  The cursor is therefore
tracked by :attr:`GpuProcess.identity` -- the ``(drm-pdev, drm-client-id)`` pair the
kernel itself reported -- and re-resolved to a row index after every rebuild.  A process
that has exited simply stops being selected, which is the truth.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Static
from textual.widgets._data_table import ColumnKey

from gputop.model.process import GpuProcess
from gputop.ui.procview import (
    COLUMN_ORDER,
    COLUMN_TITLES,
    COLUMN_WIDTHS,
    FilterState,
    ProcessRow,
    SortColumn,
    SortState,
    cell_value,
    panel_title,
)
from gputop.ui.theme import Theme

#: Marker placed on the row using the most GPU.  A glyph rather than a colour change
#: alone, because the highlight has to survive ``--no-color`` and stay findable by
#: anyone who cannot tell the accent colour from the text colour.
TOP_MARKER = "▶"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Shown:
    """The arguments of the last :meth:`ProcessTable.update_rows` call.

    Kept so a theme change can repaint without the app having to remember, and so
    :meth:`ProcessTable.repaint` is a one-liner rather than a second code path that
    could disagree with the first about what a row looks like.
    """

    rows: tuple[ProcessRow, ...]
    sort: SortState
    filters: FilterState
    top: tuple[GpuProcess, ...]
    total: int
    tree: bool


#: The change-detection key for one grid: the palette it was rendered with, then the
#: identity and rendered text of each row, in order.
type _Signature = tuple[Theme, tuple[tuple[tuple[str, int], tuple[str, ...]], ...]]


class ProcessTable(Vertical):
    """The per-client table, its heading line, and the messages its header sends."""

    DEFAULT_CSS: ClassVar[str] = """
    ProcessTable { height: 1fr; }
    #process-title { height: 1; }
    #process-table { height: 1fr; }
    """

    class SortRequested(Message):
        """Posted when a column heading is clicked.

        Args:
            column: The column that was clicked.
        """

        def __init__(self, column: SortColumn) -> None:
            super().__init__()
            self.column = column

    class RowSelected(Message):
        """Posted when a row is chosen with the mouse or the keyboard.

        Args:
            process: The client in that row.
        """

        def __init__(self, process: GpuProcess) -> None:
            super().__init__()
            self.process = process

    def __init__(
        self,
        theme: Theme,
        *,
        sort: SortState | None = None,
        full_command: bool = False,
        name: str | None = None,
        id: str | None = None,  # noqa: A002 - matches Textual's Widget signature
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._theme = theme
        self._sort = sort or SortState()
        self._full_command = full_command
        self._keys: tuple[tuple[str, int], ...] = ()
        self._shown: _Shown | None = None
        # What the headings currently say, so an unchanged sample does not rewrite them.
        self._headings: dict[SortColumn, str] = {}
        # What the grid currently holds, so an unchanged sample costs nothing.  The theme
        # is part of the key because it reaches the render through the marker's style
        # rather than through the cell text: keying on text alone made ``set_theme`` a
        # no-op until some cell's characters happened to change.  Seeded with the theme it
        # was built with, so a table that has never been given rows is not "unchanged"
        # merely because it is empty.
        self._signature: _Signature = (self._theme, ())
        # Filled in by on_mount; the keys are what DataTable indexes its columns by, and
        # a plain string is not the same thing to it.
        self._column_keys: dict[SortColumn, ColumnKey] = {}

    # -- composition --------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="process-title")
        yield DataTable(id="process-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        """Add the columns once, in the order :data:`COLUMN_ORDER` fixes.

        The key *is* the column name, so a header click needs no lookup table to find out
        which column it means.
        """
        table = self.query_one("#process-table", DataTable)
        for column in COLUMN_ORDER:
            self._column_keys[column] = table.add_column(
                self._heading(column), width=COLUMN_WIDTHS[column], key=column
            )
        self._headings = {column: self._heading(column) for column in COLUMN_ORDER}

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Turn a click on a heading into a sort request.

        Clicking the column already sorted by reverses it, which is what a click on a
        sortable heading means everywhere else; the app owns the resulting state.
        """
        column = str(event.column_key)
        if column in COLUMN_ORDER:
            event.stop()
            self.post_message(self.SortRequested(column))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Report the chosen process, whether by click or by pressing enter."""
        process = self.process_at(event.cursor_row)
        if process is not None:
            event.stop()
            self.post_message(self.RowSelected(process))

    # -- data ---------------------------------------------------------------

    def update_rows(
        self,
        rows: Iterable[ProcessRow],
        *,
        sort: SortState,
        filters: FilterState,
        top: Iterable[GpuProcess] = (),
        total: int | None = None,
        tree: bool = False,
    ) -> None:
        """Rebuild the table from ``rows``.

        Args:
            rows: The rows to show, already filtered and ordered.
            sort: The active sort, for the headings and the title line.
            filters: The active filter, for the title line.
            top: The clients using the most GPU, which are marked.
            total: How many clients exist before filtering, so the title can say how
                many are hidden instead of showing a short list with no explanation.
            tree: Whether the tree view is on, for the title line.
        """
        shown = tuple(rows)
        table = self.query_one("#process-table", DataTable)
        # The client the cursor is on has to be resolved *before* ``_shown`` is replaced,
        # because it is looked up in the rows the cursor was sitting in.  Reading it from
        # the grid rather than from a cached identity is what makes this correct for a
        # cursor that moved without gputop hearing about it -- see :meth:`_restore_cursor`.
        selected = self.process_at(table.cursor_row)

        self._sort = sort
        self._shown = _Shown(
            rows=shown,
            sort=sort,
            filters=filters,
            top=tuple(top),
            total=total if total is not None else len(shown),
            tree=tree,
        )
        marked = {process.identity for process in self._shown.top}
        self.query_one("#process-title", Static).update(self._title())

        cells = [self._cells(row, marked=row.process.identity in marked) for row in shown]
        signature = (
            self._theme,
            tuple(
                (row.process.identity, tuple(str(cell) for cell in cell_row))
                for row, cell_row in zip(shown, cells, strict=True)
            ),
        )
        # Above the early return, not below: the headings depend on the sort, which can
        # change while the rows do not -- a one-row table re-sorted left the arrow on the
        # column it had just left.
        self._sync_headings(table)
        if signature == self._signature:
            # Nothing moved.  Rebuilding anyway would empty the grid and swallow any click
            # that happened to land mid-refresh -- on a one-second sampler that is a coin
            # toss every time the user reaches for a row -- and would drop the cursor
            # back to the top for a frame on every sample.
            return
        self._signature = signature

        table.clear()
        for row, cell_row in zip(shown, cells, strict=True):
            table.add_row(*cell_row, key=self._row_key(row.process.identity))
        self._keys = tuple(row.process.identity for row in shown)
        self._restore_cursor(table, selected)

    def repaint(self) -> None:
        """Redraw the last known rows, e.g. after a theme change."""
        if self._shown is not None:
            self.update_rows(
                self._shown.rows,
                sort=self._shown.sort,
                filters=self._shown.filters,
                top=self._shown.top,
                total=self._shown.total,
                tree=self._shown.tree,
            )

    def set_theme(self, theme: Theme) -> None:
        """Switch palette and repaint; the marker is the only styled part."""
        self._theme = theme
        self.repaint()

    def set_full_command(self, full: bool) -> None:
        """Switch between the truncated ``comm`` name and the full command line."""
        if full == self._full_command:
            # The app calls this on every sample.  Repainting unconditionally would, on
            # the one sample where the flag actually changed, rebuild the grid twice: once
            # from the previous sample's rows and once from the current ones.
            return
        self._full_command = full
        self.repaint()

    def _cells(self, row: ProcessRow, *, marked: bool) -> tuple[object, ...]:
        """Render one row, indenting it for the tree and marking the busiest client."""
        style = Style(color=self._theme.accent, bold=True) if marked else None
        cells: list[object] = []
        for column in COLUMN_ORDER:
            text = cell_value(row.process, column, full_command=self._full_command)
            if column == "command":
                text = f"{row.prefix}{text}"
            elif column == "pid" and marked:
                text = f"{TOP_MARKER}{text}"
            cells.append(Text(text, style=style) if style is not None else text)
        return tuple(cells)

    def _sync_headings(self, table: DataTable[object]) -> None:
        """Repaint the headings so the active column carries the direction marker.

        Only the active column's label changes, but all of them are rewritten because the
        sort marker has to *leave* the previous column, and a label set once at mount
        would keep pointing at it.
        """
        labels = {column: self._heading(column) for column in COLUMN_ORDER}
        if labels == self._headings:
            return
        self._headings = labels
        for column, key in self._column_keys.items():
            # The label is stored as a ``Text`` but left unstyled: DataTable's own header
            # CSS owns the colour, and a per-cell style here would fight the theme
            # rather than follow it.
            table.columns[key].label = Text(labels[column])

    def _heading(self, column: SortColumn) -> str:
        """The label for one heading, including the sort marker."""
        return self._sort.title_for(column)

    def _title(self) -> Text:
        """Build the heading line above the table."""
        if self._shown is None:  # pragma: no cover - only before the first update
            return Text("")
        return Text(
            panel_title(
                self._shown.sort,
                self._shown.filters,
                shown=len(self._shown.rows),
                total=self._shown.total,
                # Deliberately every ranked client, not only the ones on screen: the
                # marker is drawn per row, but a leader trimmed off by ``max_rows`` is
                # still the leader, and reporting the table as idle because its busiest
                # row happens to be the one that did not fit is a lie.
                top=self._shown.top,
                tree=self._shown.tree,
            ),
            style=Style(color=self._theme.muted),
        )

    # -- selection ----------------------------------------------------------

    @staticmethod
    def _row_key(identity: tuple[str, int]) -> str:
        """A ``DataTable`` row key that encodes the client's identity."""
        return f"{identity[0]}:{identity[1]}"

    def _restore_cursor(self, table: DataTable[object], previous: GpuProcess | None) -> None:
        """Put the cursor back on the same client after a rebuild.

        A row index would be wrong the moment a process exits; the identity is the only
        handle that means the same thing in two consecutive samples.

        The client is passed in rather than read from a cached identity because the cursor
        moves in ways gputop never hears about: a click on a row that is not already
        highlighted posts ``RowHighlighted``, not ``RowSelected``, and once the grid has
        focus its own arrow-key bindings take over from the app's.  A cache fed only by
        gputop's handlers stayed ``None`` after either, and every rebuild then dropped the
        cursor onto row 0 -- onto whichever process happened to sort first.
        """
        if previous is None:
            return
        for index, identity in enumerate(self._keys):
            if identity == previous.identity:
                table.move_cursor(row=index, animate=False)
                return

    def process_at(self, row: int) -> GpuProcess | None:
        """The client in a row, or ``None`` when the row no longer exists."""
        if self._shown is None or not 0 <= row < len(self._shown.rows):
            return None
        return self._shown.rows[row].process

    @property
    def grid(self) -> DataTable[object]:
        """The underlying grid.

        Exposed for hit-testing.  A click offset is only meaningful against the grid's
        own header height and scroll position, and recomputing that anywhere else is a
        second, subtly different copy of the same arithmetic.
        """
        return self.query_one("#process-table", DataTable)

    @property
    def rows(self) -> tuple[ProcessRow, ...]:
        """The rows currently shown, in the order they are shown."""
        return self._shown.rows if self._shown is not None else ()

    @property
    def selected(self) -> GpuProcess | None:
        """The client the cursor is on, or ``None`` when the table is empty."""
        return self.process_at(self.cursor_row)

    def move_cursor(self, row: int) -> None:
        """Place the cursor on an absolute row, clamped to the table."""
        if self._shown is None or not self._shown.rows:
            return
        table = self.query_one("#process-table", DataTable)
        target = max(0, min(len(self._shown.rows) - 1, row))
        table.move_cursor(row=target, animate=False)

    def step_cursor(self, step: int) -> GpuProcess | None:
        """Move the cursor by ``step`` rows and return the client it lands on."""
        if self._shown is None or not self._shown.rows:
            return None
        table = self.query_one("#process-table", DataTable)
        target = max(0, min(len(self._shown.rows) - 1, table.cursor_row + step))
        table.move_cursor(row=target, animate=False)
        return self._shown.rows[target].process

    @property
    def cursor_row(self) -> int:
        """Which row the cursor is on."""
        return self.query_one("#process-table", DataTable).cursor_row

    @property
    def headings(self) -> dict[SortColumn, str]:
        """Heading text per column, for tests and for the help overlay."""
        return {column: self._heading(column) for column in COLUMN_ORDER}


__all__ = ["COLUMN_TITLES", "TOP_MARKER", "ProcessTable"]
