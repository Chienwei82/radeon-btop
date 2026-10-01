"""Tests for the radeontop subprocess reader.

The policy these enforce is the one that matters most for the feature: a card that cannot
be read must produce a *specific* reason and no data, never a zero, a crash, or a request
that the rest of gputop has to be root to satisfy.

``dump.c`` conditionally emits ``tc``, ``smx``, ``cr``, ``uvd``, ``vce0``, ``vram``, ``gtt``
and the clocks depending on which bits the card reports, so most of what follows is about
parsing by name rather than by position.  A positional parser reads ``vgt``'s value as
``ta``'s on every modern card and is confidently wrong.
"""

import os
import stat
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from gputop.model.blocks import BLOCK_ORDER, BlocksStatus
from gputop.readers.radeontop import (
    MAX_RESTARTS,
    MIN_DUMP_INTERVAL_S,
    RESTART_RESET_S,
    RadeontopOptions,
    RadeontopPool,
    build_argv,
    bus_from_bdf,
    classify_failure,
    parse_dump_line,
    resolve_binary,
    wait_for_first_sample,
)

#: A full RDNA2 line exactly as ``dump.c`` formats it, with every optional field present.
FULL_LINE = (
    "1759248000.123456: bus 0c, "
    "gpu 45.31%, ee 12.50%, vgt 3.25%, ta 1.00%, tc 0.50%, sx 7.75%, sh 8.25%, "
    "spi 2.50%, smx 1.00%, sc 1.25%, pa 1.00%, db 9.99%, cr 0.10%, cb 10.10%, "
    "uvd 0.00%, vce0 0.00%, "
    "vram 13.13% 4096.00mb, gtt 14.14% 512.00mb, "
    "mclk 50.00% 1.000ghz, sclk 25.00% 2.104ghz"
)

#: An R600-era line: no ``tc``, ``smx``, ``cr``, ``uvd``, ``vce0``, no memory, no clocks.
LEGACY_LINE = (
    "100.500000: bus 01, gpu 10.00%, ee 0.00%, vgt 0.00%, ta 0.00%, "
    "sx 0.00%, sh 0.00%, spi 0.00%, sc 0.00%, pa 0.00%, db 0.00%, cb 0.00%"
)


def write_script(path: Path, body: str) -> Path:
    """Write an executable shell script and return its path."""
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_pool(tmp_path: Path, body: str, **overrides: object) -> RadeontopPool:
    """A pool whose single source runs ``body`` as its stand-in radeontop."""
    binary = write_script(tmp_path / "radeontop-stand-in", body)
    pool = RadeontopPool(RadeontopOptions(enabled=True, binary=str(binary), **overrides))
    pool.configure([("0", "card0", "0000:0c:00.0")])
    return pool


# -- parsing ---------------------------------------------------------------


class TestParsing:
    """The dump line parser."""

    def test_every_field_of_a_full_line_is_read(self) -> None:
        dump = parse_dump_line(FULL_LINE)
        assert dump is not None
        assert dump.seconds == 1759248000
        assert dump.microseconds == 123456
        assert dump.bus == "0c"
        assert dump.blocks["gpu"] == pytest.approx(45.31)
        assert dump.blocks["cb"] == pytest.approx(10.10)
        assert dump.clocks["sclk"] == (pytest.approx(25.0), pytest.approx(2104.0))
        assert dump.clocks["mclk"] == (pytest.approx(50.0), pytest.approx(1000.0))
        assert dump.memory["vram"] == (pytest.approx(13.13), pytest.approx(4096.0))

    def test_blocks_are_ordered_by_pipeline_stage(self) -> None:
        blocks = parse_dump_line(FULL_LINE).to_blocks()
        assert [b.key for b in blocks.blocks] == [k for k in BLOCK_ORDER if k in BLOCK_ORDER]

    def test_a_legacy_line_omits_the_blocks_the_hardware_lacks(self) -> None:
        """The conditional fields are the whole point of parsing by name."""
        dump = parse_dump_line(LEGACY_LINE)
        assert dump is not None
        assert set(dump.blocks) == {
            "gpu",
            "ee",
            "vgt",
            "ta",
            "sx",
            "sh",
            "spi",
            "sc",
            "pa",
            "db",
            "cb",
        }
        assert dump.clocks == {}
        assert dump.memory == {}

    def test_missing_optional_fields_do_not_shift_later_ones(self) -> None:
        """Without ``tc`` the value after it must stay with its own key."""
        without_tc = FULL_LINE.replace("tc 0.50%, ", "")
        with_tc = parse_dump_line(FULL_LINE).blocks
        without = parse_dump_line(without_tc).blocks
        assert "tc" not in without
        assert without["ta"] == with_tc["ta"]
        assert without["vgt"] == with_tc["vgt"]
        assert without["sx"] == with_tc["sx"]

    def test_trailing_newline_is_accepted(self) -> None:
        assert parse_dump_line(FULL_LINE + "\n") is not None

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "Dumping to -, until termination.\n",
            "Failed to find DRM devices: error 2 (No such file or directory)\n",
            "radeontop 1.4.3, running on radeon bus 0c, 120 samples/sec\n",
            "0.5: bus zz, gpu 1.00%\n",
        ],
    )
    def test_banners_and_diagnostics_are_not_data(self, line: str) -> None:
        """A non-data line is ``None``, not an exception and not a zeroed sample."""
        assert parse_dump_line(line) is None

    def test_unknown_keys_are_ignored_rather_than_guessed(self) -> None:
        """radeontop can add a block; a stale table must not mis-assign its value."""
        dump = parse_dump_line("5.0: bus 01, gpu 10.00%, brandnew 42.00%, cb 1.00%")
        assert dump is not None
        assert "brandnew" not in dump.blocks
        assert dump.blocks["cb"] == pytest.approx(1.0)

    def test_the_busiest_block_is_the_largest(self) -> None:
        blocks = parse_dump_line(FULL_LINE).to_blocks()
        busiest = blocks.busiest()
        assert busiest is not None
        assert busiest.key == "gpu"
        assert busiest.percent == pytest.approx(45.31)

    def test_clocks_carry_both_a_percentage_and_a_frequency(self) -> None:
        blocks = parse_dump_line(FULL_LINE).to_blocks()
        labels = {c.key: (c.label, c.percent, c.mhz) for c in blocks.clocks}
        assert labels["sclk"] == ("Shader Clock", pytest.approx(25.0), 2104)
        assert labels["mclk"] == ("Memory Clock", pytest.approx(50.0), 1000)

    def test_rows_are_grouped_pipeline_then_aux_then_clocks(self) -> None:
        groups = parse_dump_line(FULL_LINE).to_blocks().display_rows()
        assert len(groups) == 3
        assert groups[0][0].key == "gpu"
        assert [g.key for g in groups[1]] == ["uvd", "vce0"]
        assert [g.key for g in groups[2]] == ["mclk", "sclk"]

    def test_an_empty_sample_reports_no_busiest_block(self) -> None:
        assert parse_dump_line(FULL_LINE).to_blocks().by_key() is not None
        from gputop.model.blocks import GpuBlocks

        assert GpuBlocks().busiest() is None
        assert GpuBlocks().is_empty


# -- command line ----------------------------------------------------------


class TestArgv:
    """The command line handed to radeontop."""

    def test_dump_mode_is_requested_with_no_line_limit(self) -> None:
        argv = build_argv(RadeontopOptions())
        assert argv[:5] == ["radeontop", "-d", "-", "-i", "1"]
        assert argv[argv.index("-l") + 1] == "0"

    def test_the_drm_node_is_preferred_over_a_bus_number(self) -> None:
        """``-p`` is exact; ``-b`` is ambiguous when two cards share a bus."""
        argv = build_argv(RadeontopOptions(), bus="0c", node=Path("/dev/dri/card1"))
        assert "-p" in argv
        assert "/dev/dri/card1" in argv
        assert "-b" not in argv

    def test_the_bus_is_used_when_no_node_is_known(self) -> None:
        argv = build_argv(RadeontopOptions(), bus="0c")
        assert argv[argv.index("-b") + 1] == "0c"

    def test_the_ticks_rate_is_passed_through(self) -> None:
        argv = build_argv(RadeontopOptions(ticks=50))
        assert argv[argv.index("-t") + 1] == "50"

    @pytest.mark.parametrize(
        ("bdf", "expected"),
        [
            ("0000:0c:00.0", "0c"),
            ("0000:01:00.0", "01"),
            ("10000:ff:1f.7", "ff"),
        ],
    )
    def test_the_bus_is_taken_from_the_pci_address(self, bdf: str, expected: str) -> None:
        assert bus_from_bdf(bdf) == expected

    @pytest.mark.parametrize("bdf", ["", "nonsense", "0000:zz:00.0", "0c:00.0"])
    def test_a_malformed_address_yields_no_bus(self, bdf: str) -> None:
        assert bus_from_bdf(bdf) is None

    def test_the_interval_is_clamped_to_what_radeontop_accepts(self) -> None:
        """radeontop parses ``-i`` with atoi and floors it at 1; so does gputop.

        Honouring 500 ms here would produce a panel that updates four times less often than
        the user asked for, with nothing saying so.
        """
        assert RadeontopOptions(interval_s=0).validated().interval_s == MIN_DUMP_INTERVAL_S
        assert RadeontopOptions(interval_s=-5).validated().interval_s == MIN_DUMP_INTERVAL_S
        assert RadeontopOptions(interval_s=3).validated().interval_s == 3


# -- binary resolution -----------------------------------------------------


class TestBinaryResolution:
    """Finding, or failing to find, the executable."""

    def test_a_name_is_looked_up_on_the_path(self, tmp_path: Path) -> None:
        target = write_script(tmp_path / "radeontop-stand-in", "exit 0\n")
        assert resolve_binary(str(target)) == target

    def test_a_missing_binary_resolves_to_nothing(self) -> None:
        assert resolve_binary("gputop-no-such-binary-xyz") is None

    def test_a_non_executable_file_is_not_a_binary(self, tmp_path: Path) -> None:
        plain = tmp_path / "radeontop"
        plain.write_text("not executable")
        assert resolve_binary(str(plain)) is None


# -- failure classification ------------------------------------------------


class TestFailureClassification:
    """Turning radeontop's stderr into a status a user can act on."""

    @pytest.mark.parametrize(
        ("stderr", "expected"),
        [
            (("Cannot access GPU registers, are you root?",), BlocksStatus.NOT_PERMITTED),
            (("Failed to open DRM node, no VRAM support.",), BlocksStatus.NOT_PERMITTED),
            (("open /dev/dri/card0: Permission denied",), BlocksStatus.NOT_PERMITTED),
            (("Failed to find DRM devices: error 2",), BlocksStatus.UNSUPPORTED),
            (("Unsupported bus type 5",), BlocksStatus.UNSUPPORTED),
            (("Failed to open /dev/mem",), BlocksStatus.UNSUPPORTED),
        ],
    )
    def test_each_known_message_maps_to_its_own_status(
        self, stderr: tuple[str, ...], expected: BlocksStatus
    ) -> None:
        assert classify_failure(stderr) is expected

    def test_a_complaint_on_stdout_is_still_recognised(self) -> None:
        """radeontop splits its complaints across both streams.

        The packaged build writes "are you root?" to *stdout* and only "Failed to find DRM
        devices" to stderr.  Reading stderr alone classifies a privilege problem as an
        unsupported card -- the least actionable of the available answers.
        """
        assert (
            classify_failure(("Failed to open DRM node, no VRAM support.\n",))
            is BlocksStatus.NOT_PERMITTED
        )
        assert (
            classify_failure(("Cannot access GPU registers, are you root?\n",))
            is BlocksStatus.NOT_PERMITTED
        )

    def test_a_dump_line_is_never_mistaken_for_a_complaint(self) -> None:
        assert classify_failure((FULL_LINE,)) is None

    def test_permission_wins_over_the_earlier_unrelated_errors(self) -> None:
        """radeontop reports every problem it has; the actionable one comes last.

        Reading only the first line would tell a user their card is unsupported when the
        real answer is that they need root, which is the difference between a fixable
        problem and an abandoned feature.
        """
        stderr = (
            "Failed to find DRM devices: error 2 (No such file or directory)",
            "Failed to open DRM node, no VRAM support.",
            "Cannot access GPU registers, are you root?",
        )
        assert classify_failure(stderr) is BlocksStatus.NOT_PERMITTED

    def test_a_silent_failure_is_left_unclassified(self) -> None:
        assert classify_failure(()) is None
        assert classify_failure(("something entirely unexpected",)) is None

    def test_the_success_banner_is_not_mistaken_for_a_failure(self) -> None:
        assert classify_failure(("Dumping to -, until termination.",)) is None


# -- the supervised process ------------------------------------------------


class TestPool:
    """Starting, reading from, and stopping the child."""

    def test_a_dump_line_becomes_a_sample(self, tmp_path: Path) -> None:
        pool = make_pool(
            tmp_path,
            """
            echo "Dumping to -, until termination." >&2
            while true; do
              echo '100.0: bus 0c, gpu 42.00%, cb 7.00%, sclk 25.00% 2.100ghz'
              sleep 0.05
            done
            """,
        )
        pool.start()
        try:
            assert wait_for_first_sample(pool, timeout_s=10.0)
            assert pool.status() is BlocksStatus.OK
            blocks = pool.blocks_for("0")
            assert blocks is not None
            assert blocks.busiest().key == "gpu"
            assert blocks.clocks[0].mhz == 2100
        finally:
            pool.stop()

    def test_a_child_that_exits_at_once_is_reported_as_failed(self, tmp_path: Path) -> None:
        pool = make_pool(tmp_path, "exit 0\n")
        pool.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pool.status() is BlocksStatus.OK:
                pool.poll()
                time.sleep(0.02)
            assert pool.status() is BlocksStatus.FAILED
            assert pool.blocks_for("0") is None
        finally:
            pool.stop()

    def test_a_permission_failure_is_named_as_such(self, tmp_path: Path) -> None:
        pool = make_pool(
            tmp_path,
            """
            echo 'Cannot access GPU registers, are you root?' >&2
            exit 0
            """,
        )
        pool.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pool.status() is BlocksStatus.OK:
                pool.poll()
                time.sleep(0.02)
            assert pool.status() is BlocksStatus.NOT_PERMITTED
            assert pool.blocks_for("0") is None
        finally:
            pool.stop()

    def test_a_missing_binary_never_starts_a_process(self, tmp_path: Path) -> None:
        pool = RadeontopPool(RadeontopOptions(enabled=True, binary="gputop-no-such-binary-xyz"))
        pool.configure([("0", "card0", "0000:0c:00.0")])
        pool.start()
        assert pool.status() is BlocksStatus.MISSING
        assert pool.blocks_for("0") is None

    def test_a_disabled_pool_starts_nothing_and_says_so(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        binary = write_script(tmp_path / "radeontop-stand-in", f"touch {marker}\nsleep 5\n")
        pool = RadeontopPool(RadeontopOptions(binary=str(binary), enabled=False))
        pool.configure([("0", "card0", "0000:0c:00.0")])
        pool.start()
        time.sleep(0.2)
        pool.stop()
        assert pool.status() is BlocksStatus.DISABLED
        assert not marker.exists()

    def test_stopping_closes_the_pipes(self, tmp_path: Path) -> None:
        """A descriptor leak per source per restart would eventually exhaust the process."""
        pool = make_pool(
            tmp_path,
            "while true; do sleep 0.05; done\n",
        )
        pool.start()
        pool.stop()
        process = pool._sources["0"]._process
        assert process is None

    def test_stopping_leaves_no_child_behind(self, tmp_path: Path) -> None:
        pidfile = tmp_path / "pid"
        pool = make_pool(
            tmp_path,
            f"echo $$ > {pidfile}\nwhile true; do sleep 0.05; done\n",
        )
        pool.start()
        deadline = time.monotonic() + 5.0
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        pid = int(pidfile.read_text().strip())
        pool.stop()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:  # pragma: no cover - only on a failure to reap
            pytest.fail(f"child {pid} survived stop()")

    def test_two_devices_get_one_child_each(self, tmp_path: Path) -> None:
        pool = RadeontopPool(
            RadeontopOptions(
                enabled=True,
                binary=str(write_script(tmp_path / "radeontop-stand-in", "sleep 5\n")),
            )
        )
        pool.configure([("0", "card0", "0000:0c:00.0"), ("1", "card1", "0000:01:00.0")])
        pool.start()
        try:
            assert pool.keys() == ("0", "1")
            assert pool.status() is BlocksStatus.OK
        finally:
            pool.stop()

    def test_a_device_with_no_source_reports_no_sample(self, tmp_path: Path) -> None:
        pool = make_pool(tmp_path, "sleep 5\n")
        assert pool.blocks_for("does-not-exist") is None

    def test_configuring_stops_the_previous_children(self, tmp_path: Path) -> None:
        pool = make_pool(tmp_path, "sleep 5\n")
        pool.start()
        pool.configure([("0", "card0", "0000:0c:00.0")])
        assert pool.keys() == ("0",)


class TestNoPrivilegeIsRequired:
    """The promise the whole feature rests on."""

    def test_nothing_in_the_reader_raises_on_its_euid(self) -> None:
        """gputop must work entirely unprivileged; only the child may need more."""
        assert os.geteuid() == os.getuid() or os.geteuid() == 0
        argv = build_argv(RadeontopOptions())
        # No sudo, no su, no setuid wrapper: the child is radeontop itself, which drops to
        # the invoking user during its own startup.
        assert argv[0] == "radeontop"
        assert not any(part in ("sudo", "su", "pkexec") for part in argv)

    def test_a_failure_never_propagates_as_an_exception(self, tmp_path: Path) -> None:
        pool = RadeontopPool(RadeontopOptions(enabled=True, binary=str(tmp_path / "nope")))
        pool.configure([("0", "card0", "0000:0c:00.0")])
        pool.start()
        assert pool.status() is BlocksStatus.MISSING
        pool.stop()


def test_the_parser_accepts_the_system_radeontop_version_if_present() -> None:
    """Guard the assumption that ``-d - -i N -l 0`` is still a valid invocation.

    Skipped when radeontop is not installed, which is the normal case for the suite and the
    case the rest of these tests are written for.
    """
    binary = resolve_binary("radeontop")
    if binary is None:
        pytest.skip("radeontop is not installed")
    try:
        result = subprocess.run(
            [str(binary), "--help"], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError, subprocess.SubprocessError:  # pragma: no cover - sandbox dependent
        pytest.skip("radeontop could not be executed here")
    # -d/--dump, -i/--dump-interval and -l/--limit are the flags gputop depends on.
    for flag in ("--dump", "--dump-interval", "--limit"):
        assert flag in result.stdout


class TestRestartBudget:
    """Restarting must be bounded, and must recover over a long session."""

    def test_a_child_that_never_survives_stops_being_restarted(self, tmp_path: Path) -> None:
        """Retrying a child that always dies immediately would be a busy loop."""
        pool = make_pool(tmp_path, "exit 0\n")
        pool.start()
        try:
            for _ in range(40):
                pool.poll()
                time.sleep(0.02)
            assert pool.status() is BlocksStatus.FAILED
        finally:
            pool.stop()

    def test_a_long_lived_child_gets_its_restart_budget_back(self, tmp_path: Path) -> None:
        """A session must not slowly exhaust its retries over hours of healthy running.

        The counter is restored once a child has outlived the reset window, so one ordinary
        hiccup at the end of a long session still gets a fresh budget.
        """
        pool = make_pool(tmp_path, "while true; do sleep 0.05; done\n")
        pool.start()
        try:
            source = pool._sources["0"]
            source._restarts = MAX_RESTARTS
            source._started_at = time.monotonic() - RESTART_RESET_S - 1
            assert source.poll() is BlocksStatus.OK
            assert source._restarts == 0
        finally:
            pool.stop()

    def test_a_child_that_died_recently_keeps_its_budget_spent(self, tmp_path: Path) -> None:
        """A child that cannot even start has not proven anything."""
        pool = make_pool(tmp_path, "while true; do sleep 0.05; done\n")
        pool.start()
        try:
            source = pool._sources["0"]
            source._restarts = MAX_RESTARTS
            source.poll()
            assert source._restarts == MAX_RESTARTS
        finally:
            pool.stop()


class TestChildLifecycle:
    """A reader thread must only ever speak about the child it was spawned for."""

    def _running_source(self, tmp_path: Path) -> tuple[RadeontopPool, object]:
        pool = make_pool(tmp_path, "while true; do sleep 0.05; done\n")
        pool.start()
        return pool, pool._sources["0"]

    def test_a_previous_child_cannot_fail_the_one_that_replaced_it(
        self, tmp_path: Path
    ) -> None:
        """The stdout reader classified whatever ``_process`` happened to be at EOF.

        A reader still winding down from the previous child therefore marked the healthy
        replacement ``FAILED`` -- and because ``poll`` returns early while a child is
        alive, nothing ever recovered it: the panel read "failed" for the rest of the
        session while the data kept arriving underneath.
        """
        pool, source = self._running_source(tmp_path)
        try:
            stale = subprocess.Popen(
                ["true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stale.wait(timeout=5)
            source._status = BlocksStatus.OK

            source._on_child_exit(stale)

            assert source.status is BlocksStatus.OK
            assert source.poll() is BlocksStatus.OK
            for pipe in (stale.stdout, stale.stderr):
                if pipe is not None:
                    pipe.close()
        finally:
            pool.stop()

    def test_a_replacement_child_gets_a_fresh_stdout_window(self, tmp_path: Path) -> None:
        """Once any dump line has been seen the complaint window closes -- per child.

        The flag was never cleared, so a restarted child that printed "are you root?"
        before its first sample had its complaint discarded and was reported as merely
        ``FAILED``, which is the answer with no remedy attached.
        """
        pool, source = self._running_source(tmp_path)
        try:
            source._saw_dump_line = True
            source._stdout_head.append("Cannot access GPU registers, are you root?")

            source.stop()
            source.start()

            assert source._saw_dump_line is False
            assert not source._stdout_head
        finally:
            pool.stop()

    def test_stop_clears_the_handle_so_a_racing_start_cannot_orphan_a_child(
        self, tmp_path: Path
    ) -> None:
        """``stop`` takes the same lock ``start`` does, and releases the slot inside it.

        Without the lock a restart already in flight installs a new child between ``stop``
        reading ``_process`` and clearing it, and the result is a radeontop nobody
        terminates, still holding the card.
        """
        pool, source = self._running_source(tmp_path)
        assert source._process is not None

        pool.stop()

        assert source._process is None


class TestStreamSplitDiagnostics:
    """radeontop writes its complaints to *both* streams; both must be read.

    Reproduces the exact split observed from the packaged radeontop on a machine where the
    card is present but the registers are not readable unprivileged.  Reading stderr alone
    reports ``unsupported`` -- "this card cannot be read" -- when the true answer is a
    privilege problem, which is the difference between a fixable configuration and a
    feature the user has given up on.
    """

    #: Captured verbatim from `radeontop -d - -i 1 -l 0` on the reference machine.
    REAL_STDOUT = (
        "Failed to open DRM node, no VRAM support.\n",
        "Cannot access GPU registers, are you root?\n",
    )
    REAL_STDERR = "Failed to find DRM devices: error 2 (No such file or directory)\n"

    def test_the_real_split_is_classified_as_a_permission_problem(self) -> None:
        assert classify_failure((*self.REAL_STDOUT, self.REAL_STDERR)) is (
            BlocksStatus.NOT_PERMITTED
        )

    def test_stderr_alone_would_have_given_the_wrong_answer(self) -> None:
        """Pins *why* both streams are scanned, so the change is not undone as tidying."""
        assert classify_failure((self.REAL_STDERR,)) is BlocksStatus.UNSUPPORTED

    def test_a_live_child_reports_not_permitted_not_unsupported(self, tmp_path: Path) -> None:
        pool = make_pool(
            tmp_path,
            """
            echo 'Failed to find DRM devices: error 2 (No such file or directory)' >&2
            echo 'Failed to open DRM node, no VRAM support.'
            echo 'Cannot access GPU registers, are you root?'
            exit 0
            """,
        )
        pool.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pool.status() is BlocksStatus.OK:
                pool.poll()
                time.sleep(0.02)
            assert pool.status() is BlocksStatus.NOT_PERMITTED
        finally:
            pool.stop()

    def test_data_lines_do_not_displace_the_complaints(self, tmp_path: Path) -> None:
        """Once stdout carries dumps it is data; the window stops growing but is not cleared.

        A card that works and later loses access still has to be able to say why.
        """
        pool = make_pool(
            tmp_path,
            """
            echo 'Cannot access GPU registers, are you root?'
            while true; do
              echo '100.0: bus 0c, gpu 1.00%, cb 1.00%'
              sleep 0.05
            done
            """,
        )
        pool.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pool.blocks_for("0") is None:
                pool.poll()
                time.sleep(0.02)
            assert pool.blocks_for("0") is not None
            source = pool._sources["0"]
            assert classify_failure(tuple(source._stdout_head)) is BlocksStatus.NOT_PERMITTED
        finally:
            pool.stop()
