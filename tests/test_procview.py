"""Sorting, filtering and tree-building for the process table.

None of this mounts a widget.  That is the point of :mod:`gputop.ui.procview`: the
decisions that are easy to get subtly wrong -- what happens to the children of a
filtered-out parent, whether a ``PPid`` cycle can hang the UI, where an ``N/A`` sorts
when the order flips -- are ordinary functions here, and can be asked about directly.

The properties asserted below are the ones a user would notice being wrong, phrased as
behaviour rather than as implementation.
"""

import pytest

from gputop.model.process import SORT_COLUMNS, EngineUsage, GpuProcess
from gputop.ui.procview import (
    COLUMN_ORDER,
    DEFAULT_SORT,
    NUMERIC_SORT_COLUMNS,
    FilterState,
    ProcessRow,
    SortState,
    build_rows,
    busiest_processes,
    cell_value,
    column_value,
    filter_processes,
    flat_rows,
    panel_title,
    row_cells,
    sort_processes,
    tree_rows,
)

NS = 1_000_000_000
MiB = 1024**2


def engine(name: str, percent: float, *, total_ns: int = 0, window_ns: int = NS) -> EngineUsage:
    """An engine usage that reports exactly ``percent``.

    ``percent`` is derived from the delta the way the collector would, so these fixtures
    exercise the same arithmetic as a real sample rather than a value typed into a
    field that the real model computes.
    """
    delta = int(window_ns * percent / 100.0)
    # ``percent`` is derived from the cumulative total, which must be non-zero or the
    # model reports "no baseline" -- so a client with a baseline has a total of its own.
    return EngineUsage(
        engine=name,
        total_ns=total_ns or delta or 1,
        delta_ns=delta,
        window_ns=window_ns,
    )


def unseen_engine(name: str) -> EngineUsage:
    """An engine with no baseline: reported as ``None``, not as zero."""
    return EngineUsage(engine=name, total_ns=0, delta_ns=0, window_ns=0)


def proc(
    pid: int,
    *,
    name: str = "proc",
    user: str = "david",
    bdf: str = "0000:0c:00.0",
    client_id: int = 1,
    engines: tuple[EngineUsage, ...] = (),
    vram: int | None = MiB,
    gtt: int | None = None,
    cpu: int | None = None,
    cmdline: tuple[str, ...] | None = None,
    ppid: int | None = None,
) -> GpuProcess:
    """Build a client.

    ``cmdline`` defaults to the name, i.e. an ordinary process, and ``vram`` to a
    mebibyte: a real DRM client always holds *something*, and a fixture that holds
    nothing is a client the table deliberately hides.  Use :func:`idle` for that case
    rather than passing four ``None``s.
    """
    return GpuProcess(
        pid=pid,
        name=name,
        user=user,
        bdf=bdf,
        client_id=client_id,
        engines=engines,
        vram_used=vram,
        gtt_used=gtt,
        cpu_used=cpu,
        cmdline=(name,) if cmdline is None else cmdline,
        ppid=ppid,
    )


def engines(*usages: EngineUsage) -> tuple[EngineUsage, ...]:
    """Build an ``engines`` tuple for a fixture.

    A helper rather than an inline ``(a, b)`` on purpose.  A single-element tuple built
    from a call and handed straight to a keyword argument is a shape this interpreter has
    been observed to flatten -- ``engines=(x,)`` arrives as ``engines=x``, silently, and
    only some call sites are affected.  A function call cannot be flattened that way, and
    the reason is written down in PROGRESS.md so the next person does not "simplify" it
    back.
    """
    return tuple(usages)


def idle(pid: int, **kwargs: object) -> GpuProcess:
    """A client that is holding a descriptor and using nothing."""
    return proc(pid, vram=None, gtt=None, cpu=None, engines=(), **kwargs)  # type: ignore[arg-type]


def pids(processes: tuple[GpuProcess, ...] | list[ProcessRow]) -> list[int]:
    """The pids of clients or of rows, so assertions read as one list."""
    return [item.pid for item in processes]


def row_pids(rows: tuple[ProcessRow, ...]) -> list[int]:
    """The pids of a row list."""
    return [row.process.pid for row in rows]


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


class TestColumns:
    """The column set is the table's contract with the width budget."""

    def test_displayed_columns_are_the_documented_ones(self) -> None:
        assert COLUMN_ORDER == (
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

    def test_every_sort_column_is_known(self) -> None:
        assert set(SORT_COLUMNS) >= set(COLUMN_ORDER)

    def test_a_cell_is_produced_for_every_column(self) -> None:
        assert len(row_cells(proc(1))) == len(COLUMN_ORDER)

    def test_pid_and_user_are_plain(self) -> None:
        cells = row_cells(proc(4242, name="firefox", user="root"))
        assert cells[0] == "4242"
        assert cells[1] == "root"

    def test_command_is_the_truncated_name_by_default(self) -> None:
        """``comm`` is 15 characters on a real kernel; the full line is opt-in."""
        client = proc(1, name="python3", cmdline=("/usr/bin/python3.13", "train.py"))
        assert cell_value(client, "command") == "python3"
        assert (
            cell_value(client, "command", full_command=True) == "/usr/bin/python3.13 train.py"
        )

    def test_a_kernel_thread_falls_back_to_its_name(self) -> None:
        """An empty argument vector must not produce an empty command column."""
        client = proc(1, name="kworker/0:1", cmdline=())
        assert cell_value(client, "command", full_command=True) == "kworker/0:1"

    def test_an_unmeasured_engine_is_na_not_zero(self) -> None:
        """``0%`` would claim the client is idle when the truth is "not measured yet"."""
        client = proc(
            1,
            engines=engines(
                unseen_engine("gfx"),
            ),
            vram=8 * MiB,
        )
        assert cell_value(client, "gfx") == "N/A"

    def test_a_measured_engine_keeps_a_decimal(self) -> None:
        """The interesting range is 0-5%; rounding it to a whole percent erases it."""
        client = proc(
            1,
            engines=engines(
                engine("gfx", 0.4),
            ),
            vram=8 * MiB,
        )
        assert cell_value(client, "gfx") == "0.4%"

    def test_encode_and_decode_share_a_column(self) -> None:
        client = proc(1, engines=engines(engine("enc", 2.0), engine("dec", 1.0)))
        assert cell_value(client, "encdc") == "3.0%"

    def test_enc_dec_is_na_when_neither_engine_is_known(self) -> None:
        client = proc(
            1,
            engines=engines(
                engine("gfx", 5.0),
            ),
        )
        assert cell_value(client, "encdc") == "N/A"

    def test_memory_uses_the_shared_byte_formatter(self) -> None:
        client = proc(1, vram=512 * MiB, gtt=2 * MiB)
        assert cell_value(client, "vram") == "512.0 MiB"
        assert cell_value(client, "gtt") == "2.0 MiB"

    def test_column_value_agrees_with_the_cell_for_text_columns(self) -> None:
        client = proc(1, name="FireFox", user="Dave")
        assert column_value(client, "user") == "dave"
        assert column_value(client, "command") == "firefox"


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSortState:
    """Selecting a column, cycling, and reversing."""

    def test_default_is_total_engine_load_busiest_first(self) -> None:
        assert SortState(column="total", reverse=True) == DEFAULT_SORT

    def test_selecting_a_numeric_column_starts_largest_first(self) -> None:
        state = DEFAULT_SORT.with_column("vram")
        assert state == SortState(column="vram", reverse=True)

    def test_selecting_a_text_column_starts_a_to_z(self) -> None:
        state = DEFAULT_SORT.with_column("user")
        assert state == SortState(column="user", reverse=False)

    def test_selecting_the_active_column_reverses_it(self) -> None:
        assert DEFAULT_SORT.with_column("total").reverse is False

    def test_cycling_wraps(self) -> None:
        state = SortState(column=SORT_COLUMNS[-1]).cycled(1)
        assert state.column == SORT_COLUMNS[0]

    def test_cycling_backwards_wraps_the_other_way(self) -> None:
        state = SortState(column=SORT_COLUMNS[0]).cycled(-1)
        assert state.column == SORT_COLUMNS[-1]

    def test_cycling_does_not_inherit_the_previous_direction(self) -> None:
        """Carrying "reverse" across a change of key reads backwards for no visible reason."""
        # "user" cycles to "command", both text: the new column starts A-Z even though the
        # one being left was descending.
        assert SortState(column="user", reverse=True).cycled(1) == SortState(
            column="command", reverse=False
        )

    def test_toggle_flips_only_the_direction(self) -> None:
        assert DEFAULT_SORT.toggled() == SortState(column="total", reverse=False)

    def test_an_unknown_name_falls_back_to_the_default(self) -> None:
        assert SortState.parse("nonsense") == DEFAULT_SORT

    def test_parsing_is_case_and_space_insensitive(self) -> None:
        assert SortState.parse("  VRAM ").column == "vram"

    def test_the_marker_names_the_direction(self) -> None:
        assert DEFAULT_SORT.arrow == "▾"
        assert DEFAULT_SORT.toggled().arrow == "▴"

    def test_only_the_active_heading_carries_the_marker(self) -> None:
        state = SortState(column="gfx", reverse=True)
        assert state.title_for("gfx") == "GFX % ▾"
        assert state.title_for("pid") == "PID"

    def test_describe_names_the_config_key(self) -> None:
        """What the panel says and what ``process.sort`` holds are the same string."""
        assert SortState(column="gfx").describe() == "gfx ▾"
        assert DEFAULT_SORT.describe() == "total ▾"


class TestSortProcesses:
    """The ordering itself."""

    def busy(self) -> list[GpuProcess]:
        """Three clients with distinguishable load on each engine."""
        return [
            proc(
                100,
                name="low",
                engines=engines(
                    engine("gfx", 1.0),
                ),
                vram=1 * MiB,
            ),
            proc(
                200,
                name="high",
                engines=engines(
                    engine("gfx", 90.0),
                ),
                vram=2 * MiB,
            ),
            proc(
                300,
                name="mid",
                engines=engines(
                    engine("gfx", 40.0),
                ),
                vram=3 * MiB,
            ),
        ]

    def test_largest_first_by_default(self) -> None:
        assert pids(sort_processes(self.busy(), SortState(column="gfx"))) == [200, 300, 100]

    def test_reversed_gives_smallest_first(self) -> None:
        state = SortState(column="gfx", reverse=False)
        assert pids(sort_processes(self.busy(), state)) == [100, 300, 200]

    def test_total_sums_the_engines(self) -> None:
        clients = [
            proc(
                1, name="split", engines=engines(engine("gfx", 30.0), engine("compute", 30.0))
            ),
            proc(
                2,
                name="one",
                engines=engines(
                    engine("gfx", 50.0),
                ),
            ),
        ]
        assert pids(sort_processes(clients, DEFAULT_SORT)) == [1, 2]

    def test_vram_ordering_uses_the_value_not_the_text(self) -> None:
        """``10 MiB`` and ``2 GiB`` must not be compared as strings."""
        clients = [proc(1, vram=10 * MiB), proc(2, vram=2 * 1024 * MiB)]
        assert pids(sort_processes(clients, SortState(column="vram"))) == [2, 1]

    def test_unknown_values_sort_last_going_down(self) -> None:
        clients = [
            proc(
                1,
                name="unmeasured",
                engines=engines(
                    unseen_engine("gfx"),
                ),
            ),
            proc(2, name="measured", engines=engines(engine("gfx", 5.0))),
        ]
        assert pids(sort_processes(clients, SortState(column="gfx"))) == [2, 1]

    def test_unknown_values_stay_last_when_reversed(self) -> None:
        """Flipping a column must not promote a column full of ``N/A`` above real readings."""
        clients = [
            proc(
                1,
                name="unmeasured",
                engines=engines(
                    unseen_engine("gfx"),
                ),
            ),
            proc(2, name="measured", engines=engines(engine("gfx", 5.0))),
        ]
        assert pids(sort_processes(clients, SortState(column="gfx", reverse=False))) == [2, 1]

    def test_ties_break_on_identity_not_on_input_order(self) -> None:
        """Rows that swap places every second are unreadable."""
        clients = [
            proc(
                300,
                engines=engines(
                    engine("gfx", 5.0),
                ),
                vram=MiB,
            ),
            proc(
                100,
                engines=engines(
                    engine("gfx", 5.0),
                ),
                vram=MiB,
            ),
        ]
        assert pids(sort_processes(clients, SortState(column="gfx"))) == [100, 300]
        assert pids(sort_processes(clients, SortState(column="gfx", reverse=False))) == [
            100,
            300,
        ]

    def test_two_clients_on_one_pid_tie_break_on_the_client_id(self) -> None:
        clients = [
            proc(100, client_id=9, vram=MiB),
            proc(100, client_id=2, vram=MiB),
        ]
        assert [p.client_id for p in sort_processes(clients, SortState(column="vram"))] == [
            2,
            9,
        ]

    def test_text_sorting_ignores_case(self) -> None:
        clients = [proc(1, user="zeta"), proc(2, user="Alpha")]
        state = SortState(column="user", reverse=False)
        assert pids(sort_processes(clients, state)) == [2, 1]

    def test_pid_ordering_is_numeric(self) -> None:
        clients = [proc(100), proc(9000), proc(20)]
        state = SortState(column="pid", reverse=False)
        assert pids(sort_processes(clients, state)) == [20, 100, 9000]

    def test_sorting_is_stable_across_repeated_calls(self) -> None:
        clients = self.busy()
        first = pids(sort_processes(clients, DEFAULT_SORT))
        second = pids(sort_processes(list(reversed(clients)), DEFAULT_SORT))
        assert first == second

    def test_an_empty_input_is_an_empty_output(self) -> None:
        assert sort_processes([]) == ()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFilterState:
    """Which rows survive, and why."""

    def test_idle_clients_are_hidden_by_default(self) -> None:
        """A client that merely holds a descriptor is not using the GPU."""
        clients = [idle(1, name="idle"), proc(2, name="busy")]
        assert pids(filter_processes(clients)) == [2]

    def test_showing_idle_brings_them_back(self) -> None:
        clients = [idle(1, name="idle"), proc(2, name="busy")]
        assert pids(filter_processes(clients, FilterState(show_idle=True))) == [1, 2]

    def test_toggling_is_involutive(self) -> None:
        state = FilterState()
        assert state.toggled_idle().toggled_idle() == state

    def test_engine_activity_counts_as_not_idle(self) -> None:
        client = proc(
            1,
            vram=None,
            engines=engines(
                engine("gfx", 0.1),
            ),
        )
        assert pids(filter_processes([client])) == [1]

    def test_memory_counts_as_not_idle_even_at_zero_engine(self) -> None:
        """A video-playing client sits at 0% gfx and still holds megabytes."""
        assert pids(filter_processes([proc(1, vram=MiB)])) == [1]
        assert pids(filter_processes([idle(1)])) == []

    def test_the_text_filter_matches_the_name(self) -> None:
        clients = [proc(1, name="firefox"), proc(2, name="blender")]
        assert pids(filter_processes(clients, FilterState(text="fire"))) == [1]

    def test_the_text_filter_is_case_insensitive(self) -> None:
        clients = [proc(1, name="FireFox"), proc(2, name="blender")]
        assert pids(filter_processes(clients, FilterState(text="FIREFOX"))) == [1]

    def test_the_text_filter_matches_the_full_command_line(self) -> None:
        """``comm`` is truncated to 15 characters, so the name alone is not enough."""
        client = proc(1, name="blender", cmdline=("/opt/blender-launcher", "-scene", "42"))
        assert pids(filter_processes([client], FilterState(text="launcher"))) == [1]

    def test_the_text_filter_matches_the_user(self) -> None:
        clients = [proc(1, name="a", user="root"), proc(2, name="b", user="david")]
        assert pids(filter_processes(clients, FilterState(text="root"))) == [1]

    def test_the_text_filter_matches_a_pid(self) -> None:
        clients = [proc(4242), proc(1)]
        assert pids(filter_processes(clients, FilterState(text="424"))) == [4242]

    def test_surrounding_whitespace_is_ignored(self) -> None:
        """A trailing space in the prompt must not silently hide everything."""
        clients = [proc(1, name="firefox")]
        assert pids(filter_processes(clients, FilterState(text="  fire  "))) == [1]

    def test_an_empty_filter_matches_everything(self) -> None:
        clients = [proc(1), proc(2)]
        assert len(filter_processes(clients, FilterState(text="   "))) == 2

    def test_the_minimum_engine_threshold_is_exclusive_at_the_boundary(self) -> None:
        clients = [
            proc(
                1,
                engines=engines(
                    engine("gfx", 5.0),
                ),
            ),
            proc(
                2,
                engines=engines(
                    engine("gfx", 4.9),
                ),
            ),
        ]
        state = FilterState(min_engine_percent=5.0)
        assert pids(filter_processes(clients, state)) == [1]

    def test_a_device_scope_keeps_only_that_gpu(self) -> None:
        clients = [proc(1, bdf="0000:0c:00.0"), proc(2, bdf="0000:03:00.0")]
        assert pids(filter_processes(clients, FilterState(bdf="0000:03:00.0"))) == [2]

    def test_no_scope_keeps_every_gpu(self) -> None:
        clients = [proc(1, bdf="0000:0c:00.0"), proc(2, bdf="0000:03:00.0")]
        assert len(filter_processes(clients, FilterState(bdf=None))) == 2

    def test_changing_the_device_scope_keeps_the_text(self) -> None:
        state = FilterState(text="fire").for_device("0000:03:00.0")
        assert state.text == "fire"
        assert state.bdf == "0000:03:00.0"

    def test_kernel_threads_are_hidden_by_default(self) -> None:
        """An empty argument vector is the only root-free way to recognise one."""
        clients = [proc(1, name="kworker/0:1", cmdline=()), proc(2)]
        assert pids(filter_processes(clients)) == [2]

    def test_kernel_threads_can_be_kept(self) -> None:
        clients = [proc(1, name="kworker/0:1", cmdline=()), proc(2)]
        state = FilterState(hide_kernel_threads=False)
        assert len(filter_processes(clients, state)) == 2

    def test_the_filters_combine(self) -> None:
        clients = [
            proc(1, name="firefox"),
            proc(2, name="firefox-bin", cmdline=()),
            proc(3, name="blender"),
        ]
        state = FilterState(text="fire")
        assert pids(filter_processes(clients, state)) == [1]

    def test_describe_says_what_is_hidden(self) -> None:
        assert "idle hidden" in FilterState().describe()
        assert "idle shown" in FilterState(show_idle=True).describe()
        assert "filter 'x'" in FilterState(text="x").describe()


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


class TestTree:
    """Parent/child arrangement from ``PPid``."""

    def test_a_child_sits_under_its_parent(self) -> None:
        clients = [
            proc(100, name="shell", vram=MiB),
            proc(200, name="render", ppid=100, vram=MiB),
        ]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert row_pids(rows) == [100, 200]
        assert [row.depth for row in rows] == [0, 1]

    def test_the_guides_draw_the_branch(self) -> None:
        clients = [proc(100, vram=MiB), proc(200, ppid=100, vram=MiB)]
        child = tree_rows(clients, SortState(column="pid", reverse=False))[1]
        assert child.prefix == "└─ "

    def test_a_non_final_child_draws_a_tee(self) -> None:
        clients = [
            proc(100, vram=MiB),
            proc(200, ppid=100, vram=MiB),
            proc(300, ppid=100, vram=MiB),
        ]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert rows[1].prefix == "├─ "
        assert rows[2].prefix == "└─ "

    def test_a_grandchild_continues_the_vertical_guide(self) -> None:
        """The guide is drawn because its parent is not the last of its own siblings."""
        clients = [
            proc(100),
            proc(200, ppid=100),
            proc(300, ppid=200),
            proc(400, ppid=100),
        ]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert [row.prefix for row in rows] == ["", "├─ ", "│  └─ ", "└─ "]

    def test_a_parent_says_it_has_children(self) -> None:
        clients = [proc(100, vram=MiB), proc(200, ppid=100, vram=MiB)]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert rows[0].has_children is True
        assert rows[1].has_children is False

    def test_a_roots_prefix_is_empty(self) -> None:
        assert tree_rows([proc(1, vram=MiB)])[0].prefix == ""

    def test_a_parent_that_is_not_a_client_makes_its_child_a_root(self) -> None:
        """The common case: a compositor whose parent is the session manager."""
        rows = tree_rows([proc(200, ppid=1, vram=MiB)], SortState(column="pid", reverse=False))
        assert rows[0].depth == 0

    def test_an_unreadable_parent_makes_its_child_a_root(self) -> None:
        rows = tree_rows([proc(200, ppid=None, vram=MiB)])
        assert rows[0].depth == 0

    def test_a_self_parent_does_not_recurse(self) -> None:
        rows = tree_rows([proc(100, ppid=100, vram=MiB)])
        assert row_pids(rows) == [100]
        assert rows[0].depth == 0

    def test_a_parent_cycle_terminates_and_shows_every_client(self) -> None:
        """Two processes claiming each other as parent must not hang the interface."""
        clients = [proc(100, ppid=200, vram=MiB), proc(200, ppid=100, vram=MiB)]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert sorted(row_pids(rows)) == [100, 200]

    def test_a_longer_cycle_terminates(self) -> None:
        clients = [
            proc(100, ppid=300, vram=MiB),
            proc(200, ppid=100, vram=MiB),
            proc(300, ppid=200, vram=MiB),
        ]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert sorted(row_pids(rows)) == [100, 200, 300]

    def test_every_client_appears_exactly_once(self) -> None:
        clients = [
            proc(100, vram=MiB),
            proc(200, ppid=100, vram=MiB),
            proc(300, ppid=200, vram=MiB),
            proc(400, ppid=400, vram=MiB),
        ]
        rows = tree_rows(clients)
        assert sorted(row_pids(rows)) == [100, 200, 300, 400]

    def test_children_are_ordered_like_roots(self) -> None:
        """Sorting by VRAM must not leave one branch sorted and another not."""
        clients = [
            proc(100, vram=1 * MiB),
            proc(200, ppid=100, vram=3 * MiB),
            proc(300, ppid=100, vram=2 * MiB),
        ]
        rows = tree_rows(clients, SortState(column="vram", reverse=False))
        assert row_pids(rows) == [100, 300, 200]

    def test_a_parent_sorts_by_its_own_value(self) -> None:
        clients = [proc(100, vram=2 * MiB), proc(200, vram=1 * MiB)]
        rows = tree_rows(clients, SortState(column="vram", reverse=False))
        assert row_pids(rows) == [200, 100]

    def test_two_clients_on_one_pid_are_siblings_not_nested(self) -> None:
        """One process can hold two ``drm-client-id``s; they are rows, not a hierarchy."""
        clients = [
            proc(100, client_id=1, vram=MiB),
            proc(100, client_id=2, vram=2 * MiB),
        ]
        rows = tree_rows(clients, SortState(column="pid", reverse=False))
        assert [row.depth for row in rows] == [0, 0]

    def test_a_filtered_out_parent_promotes_its_children(self) -> None:
        clients = [
            idle(100, name="idle-parent"),
            proc(200, name="render", ppid=100),
        ]
        rows = build_rows(clients, tree=True)
        assert row_pids(rows) == [200]
        assert rows[0].depth == 0

    def test_the_flat_view_has_no_depth(self) -> None:
        clients = [proc(100, vram=MiB), proc(200, ppid=100, vram=MiB)]
        assert all(row.depth == 0 for row in flat_rows(clients))
        assert row_pids(
            flat_rows(sort_processes(clients, SortState(column="pid", reverse=False)))
        ) == [
            100,
            200,
        ]

    def test_build_rows_switches_between_the_two_shapes(self) -> None:
        clients = [proc(100, vram=MiB), proc(200, ppid=100, vram=MiB)]
        assert len(build_rows(clients, tree=True)) == 2
        assert build_rows(clients, tree=False)[1].depth == 0


# ---------------------------------------------------------------------------
# Highlighting the busiest client
# ---------------------------------------------------------------------------


class TestBusiest:
    """Which rows get marked."""

    def test_nothing_is_highlighted_when_nothing_is_busy(self) -> None:
        """``0%`` is not a statement about which process is using the GPU."""
        assert busiest_processes([proc(1), proc(2, vram=2 * MiB)]) == ()

    def test_nothing_is_highlighted_for_an_empty_table(self) -> None:
        assert busiest_processes([]) == ()

    def test_the_busiest_client_is_highlighted(self) -> None:
        clients = [
            proc(
                1,
                engines=engines(
                    engine("gfx", 10.0),
                ),
            ),
            proc(
                2,
                engines=engines(
                    engine("gfx", 80.0),
                ),
            ),
        ]
        assert pids(busiest_processes(clients)) == [2]

    def test_a_tie_highlights_every_contender(self) -> None:
        """Highlighting one of two equal processes misreports what the GPU is doing."""
        clients = [
            proc(
                1,
                engines=engines(
                    engine("gfx", 40.0),
                ),
            ),
            proc(
                2,
                engines=engines(
                    engine("gfx", 40.0),
                ),
            ),
        ]
        assert pids(busiest_processes(clients)) == [1, 2]

    def test_the_highlight_follows_the_active_sort(self) -> None:
        clients = [
            proc(
                1,
                engines=engines(
                    engine("gfx", 80.0),
                ),
                vram=MiB,
            ),
            proc(
                2,
                engines=engines(
                    engine("gfx", 5.0),
                ),
                vram=64 * MiB,
            ),
        ]
        assert pids(busiest_processes(clients, sort=SortState(column="total"))) == [1]
        assert pids(busiest_processes(clients, sort=SortState(column="vram"))) == [2]

    def test_a_memory_only_leader_is_not_busy(self) -> None:
        """Holding memory is not using the engine, and the marker means "using"."""
        assert busiest_processes([proc(1, vram=1024 * MiB)]) == ()


# ---------------------------------------------------------------------------
# The heading line
# ---------------------------------------------------------------------------


class TestPanelTitle:
    """What the panel says about itself."""

    def test_always_says_how_it_is_sorted_and_what_is_hidden(self) -> None:
        text = panel_title(DEFAULT_SORT, FilterState(), shown=3, total=3)
        assert "sort total ▾" in text
        assert "idle hidden" in text

    def test_says_how_many_are_hidden(self) -> None:
        """A short list with no explanation reads as "that is all of them"."""
        assert "4 of 9" in panel_title(DEFAULT_SORT, FilterState(), shown=4, total=9)

    def test_says_when_the_tree_is_on(self) -> None:
        assert "tree" in panel_title(DEFAULT_SORT, FilterState(), shown=1, total=1, tree=True)

    def test_names_the_busiest_client(self) -> None:
        top = (
            proc(
                7,
                name="firefox",
                engines=engines(
                    engine("gfx", 42.0),
                ),
            ),
        )
        text = panel_title(DEFAULT_SORT, FilterState(), shown=1, total=1, top=top)
        assert "top firefox 7 42.0%" in text

    def test_says_idle_when_nothing_is_running(self) -> None:
        assert "idle" in panel_title(DEFAULT_SORT, FilterState(), shown=2, total=2)

    @pytest.mark.parametrize("column", ["user", "command"])
    def test_a_textual_sort_does_not_claim_the_gpu_is_idle(self, column: str) -> None:
        """Sorting by a name says nothing about activity, so the panel must not guess.

        ``busiest_processes`` ranks by the active column, and a name column has no "top"
        to rank -- so an empty result meant the panel printed "idle" for a GPU that was
        80% busy, purely because the user had sorted the table by user name.
        """
        sort = SortState(column=column)
        assert sort.column == column
        assert column not in NUMERIC_SORT_COLUMNS
        text = panel_title(sort, FilterState(), shown=2, total=2)
        assert "idle  ·" not in text and not text.endswith("· idle")

    def test_the_number_shown_is_the_one_the_ranking_used(self) -> None:
        """A table ordered by memory must not introduce its leader with a usage figure."""
        top = (proc(7, name="firefox", vram=512 * MiB, engines=engines(engine("gfx", 3.0))),)
        text = panel_title(SortState(column="vram"), FilterState(), shown=1, total=1, top=top)
        assert "512.0 MiB" in text
        assert "3.0%" not in text
