"""Tests for the model layer, config loading and the sampler/CLI plumbing."""

import argparse
import io
import json
import threading
import time
from pathlib import Path

import pytest

from gputop.config import Config, load_config
from gputop.model.history import RingBuffer
from gputop.model.metrics import Clock, Fan, MemoryPool, PcieLink, Power
from gputop.model.process import EngineUsage, GpuProcess
from gputop.model.snapshot import GpuSnapshot, SamplerStats, SourceReport
from gputop.sampler import Sampler, SamplerOptions
from tests.conftest import make_gpu, make_process, write


class TestModel:
    """Derived values must guard against the degenerate cases, not divide by zero."""

    def test_memory_pool_percent(self) -> None:
        assert MemoryPool(used=50, total=200).percent == 25.0

    def test_memory_pool_percent_is_none_without_a_total(self) -> None:
        assert MemoryPool(used=50, total=None).percent is None

    def test_memory_pool_percent_is_none_when_total_is_zero(self) -> None:
        """An APU can report a zero-sized pool; that must not divide by zero."""
        assert MemoryPool(used=0, total=0).percent is None

    def test_memory_pool_percent_clamps(self) -> None:
        assert MemoryPool(used=300, total=200).percent == 100.0

    def test_clock_percent(self) -> None:
        assert Clock(current=500, maximum=1000).percent == 50.0

    def test_clock_percent_needs_a_maximum(self) -> None:
        assert Clock(current=500, maximum=None).percent is None

    def test_power_percent(self) -> None:
        assert Power(draw_w=50, cap_w=200).percent == 25.0

    def test_power_percent_needs_a_cap(self) -> None:
        assert Power(draw_w=50, cap_w=None).percent is None

    def test_fan_percent_prefers_rpm(self) -> None:
        assert Fan(rpm=1500, max_rpm=3000).percent == 50.0

    def test_fan_percent_falls_back_to_pwm(self) -> None:
        assert Fan(rpm=None, max_rpm=None, pwm=40.0).percent == 40.0

    def test_fan_stopped_detects_a_stalled_fan(self) -> None:
        assert Fan(rpm=0).stopped
        assert not Fan(rpm=1200, max_rpm=3000).stopped

    def test_pcie_generation(self) -> None:
        """``pcie_link_speed`` is 0.1 GT/s, so 80 is 8.0 GT/s -- Gen3, not Gen5.

        ``kgd_pp_interface.h`` documents the field as "in 0.1 GT/s" in every
        ``gpu_metrics`` revision.  Treating it as a 16x-multiple of the generation
        reports a Gen3 link as Gen5, which is what the reference Navi 21 did.
        """
        link = PcieLink(width=16, speed=80)
        assert link.gt_per_second == 8.0
        assert link.generation == 3
        assert link.describe() == "Gen3 x16"

    def test_pcie_generation_table(self) -> None:
        assert [
            (
                PcieLink(width=16, speed=rate).generation,
                PcieLink(width=16, speed=rate).describe(),
            )
            for rate in (25, 50, 80, 160, 320, 640)
        ] == [
            (1, "Gen1 x16"),
            (2, "Gen2 x16"),
            (3, "Gen3 x16"),
            (4, "Gen4 x16"),
            (5, "Gen5 x16"),
            (6, "Gen6 x16"),
        ]

    def test_pcie_without_a_generation(self) -> None:
        """A rate matching no generation is shown as the speed it is.

        Snapping it to the closest one would be a claim about the hardware, and this
        figure exists to report what the link is actually doing.
        """
        link = PcieLink(width=4, speed=90)
        assert link.gt_per_second == 9.0
        assert link.generation == 0
        assert link.describe() == "9 GT/s x4"

    def test_pcie_unmeasured(self) -> None:
        assert PcieLink(width=16, speed=0).generation == 0


class TestEngineUsage:
    """Delta maths, including the clamp that keeps the figure honest."""

    def test_percent_of_a_known_delta(self) -> None:
        usage = EngineUsage(engine="gfx", total_ns=500, delta_ns=250, window_ns=1000)
        assert usage.percent == 25.0

    def test_percent_is_none_without_a_window(self) -> None:
        assert EngineUsage(engine="gfx", total_ns=5, delta_ns=0, window_ns=0).percent is None

    def test_percent_clamps_at_one_hundred(self) -> None:
        usage = EngineUsage(engine="gfx", total_ns=5000, delta_ns=5000, window_ns=1000)
        assert usage.percent == 100.0

    def test_process_percent_sums_engines(self) -> None:
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="0000:0c:00.0",
            client_id=1,
            engines=(
                EngineUsage("gfx", 400, 400, 1000),
                EngineUsage("compute", 300, 300, 1000),
            ),
        )
        assert process.engine_percent == 70.0

    def test_process_percent_is_zero_without_engines(self) -> None:
        process = GpuProcess(pid=1, name="x", user="u", bdf="b", client_id=1)
        assert process.engine_percent == 0.0

    def test_memory_used_includes_the_cpu_pool(self) -> None:
        """On an APU the CPU-visible pool dominates, so excluding it would mislead."""
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="b",
            client_id=1,
            vram_used=10,
            gtt_used=20,
            cpu_used=70,
        )
        assert process.memory_used == 100

    def test_engine_percent_for_a_bucket(self) -> None:
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="b",
            client_id=1,
            engines=(EngineUsage("gfx", 250, 250, 1000),),
        )
        assert process.engine_percent_for("gfx") == 25.0
        assert process.engine_percent_for("enc") is None

    def test_a_bucket_sums_the_engines_that_make_it_up(self) -> None:
        """``sdma0`` and ``sdma1`` are both ``dma``, and a client using both uses both.

        The bucket used to report the busier engine instead of the sum, so its column
        no longer added up to the total the rows are ordered by.
        """
        process = GpuProcess(
            pid=1,
            name="x",
            user="u",
            bdf="b",
            client_id=1,
            engines=(EngineUsage("dma", 300, 300, 1000), EngineUsage("dma", 200, 200, 1000)),
        )
        assert process.engine_percent_for("dma") == 50.0
        assert process.engine_percent == 50.0


class TestRingBuffer:
    """The history buffer is bounded and oldest-first."""

    def test_bounded_capacity(self) -> None:
        buffer = RingBuffer[int](capacity=3)
        for value in range(1, 6):
            buffer.append(value)
        assert buffer.items() == (3, 4, 5)
        assert len(buffer) == 3

    def test_capacity_is_at_least_one(self) -> None:
        assert RingBuffer[int](capacity=0).capacity == 1

    def test_last(self) -> None:
        buffer = RingBuffer[int](capacity=2)
        assert buffer.last() is None
        buffer.append(7)
        assert buffer.last() == 7

    def test_items_are_a_snapshot(self) -> None:
        """Returning a tuple means a reader cannot be affected by later appends."""
        buffer = RingBuffer[int](capacity=2)
        buffer.append(1)
        snapshot = buffer.items()
        buffer.append(2)
        assert snapshot == (1,)


class TestConfig:
    """Config loading is total and never rejects an unknown key by crashing."""

    def test_missing_file_yields_defaults_with_a_warning(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.toml")
        assert config.general.interval_ms == 1000
        assert any("not found" in w for w in config.warnings)

    def test_valid_file_is_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(
            path,
            "[general]\ninterval_ms = 250\nhistory_points = 60\n"
            "[process]\nmax_rows = 5\nsort = 'memory'\n",
        )
        config = load_config(path)

        assert config.general.interval_ms == 250
        assert config.history_length == 60
        assert config.process.max_rows == 5
        assert config.warnings == ()

    def test_unknown_key_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        """A config written for a future release must still work today."""
        path = tmp_path / "gputop.toml"
        write(path, "[general]\ninterval_ms = 500\nfrom_the_future = 1\n")

        config = load_config(path)

        assert config.general.interval_ms == 500
        assert any("from_the_future" in w for w in config.warnings)

    def test_malformed_toml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "this is not = = toml")
        config = load_config(path)
        assert config.warnings

    def test_out_of_range_interval_is_clamped(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[general]\ninterval_ms = 5\n")
        config = load_config(path)
        assert config.general.interval_ms == 100
        assert any("out of range" in w for w in config.warnings)

    def test_invalid_kind_falls_back_to_auto(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[gpu]\nkind = 'nonsense'\n")
        assert load_config(path).gpu.kind == "auto"

    def test_gpu_names_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[gpu.names]\n'0000:0c:00.0' = 'My GPU'\n")
        assert load_config(path).gpu.names == {"0000:0c:00.0": "My GPU"}

    def test_defaults_have_no_warnings(self) -> None:
        assert Config().warnings == ()

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("[general]\ninterval_ms = 'fast'\n", 1000),
            ("[general]\nhistory_points = 'many'\n", 300),
            ("[process]\nmax_rows = 'lots'\n", 20),
            ("[ui]\ngraph_height = 'tall'\n", 0),
        ],
    )
    def test_a_number_that_is_not_a_number_is_a_warning_not_a_crash(
        self, tmp_path: Path, body: str, expected: int
    ) -> None:
        """A frozen dataclass does not validate types, so the comparison used to raise.

        ``100 <= "fast"`` is a ``TypeError`` straight out of ``load_config``, which breaks
        the module's own promise that a malformed file produces warnings and a working
        configuration rather than a refusal to start.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)

        config = load_config(path)

        assert config.warnings
        numbers = [
            config.general.interval_ms,
            config.general.history_points,
            config.process.max_rows,
            config.ui.graph_height,
        ]
        assert expected in numbers

    def test_a_boolean_is_not_a_number(self, tmp_path: Path) -> None:
        """``interval_ms = true`` is 1 ms, and TOML's boolean is an int subclass."""
        path = tmp_path / "gputop.toml"
        write(path, "[general]\ninterval_ms = true\n")
        config = load_config(path)
        assert config.general.interval_ms == 1000
        assert any("boolean" in w for w in config.warnings)

    @pytest.mark.parametrize(
        "body",
        [
            "[ui]\ntheme = ['a']\n",
            "[state]\ntheme = {a = 1}\n",
            "[state]\ninterval_ms = 'fast'\n",
            "[state]\ngpu_index = 'first'\n",
        ],
    )
    def test_a_state_section_that_is_the_wrong_shape_still_starts(
        self, tmp_path: Path, body: str
    ) -> None:
        """Unhashable values reached the ``in`` checks; comparing one to a number raises."""
        path = tmp_path / "gputop.toml"
        write(path, body)
        assert isinstance(load_config(path), Config)

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("[gpu]\nnames = 5\n", {}),
            ('[gpu]\nnames = "card0"\n', {}),
            ("[gpu]\nnames = {0 = 'igpu'}\n", {"0": "igpu"}),
        ],
    )
    def test_a_gpu_names_table_of_the_wrong_shape_still_starts(
        self, tmp_path: Path, body: str, expected: dict[str, str]
    ) -> None:
        """``names`` went to ``dict(...)`` unvalidated, so a typo raised from the Sampler.

        The failure was a traceback from ``Sampler.__init__`` after start-up, with no
        warning pointing at the file, so a config typo produced a stack trace where the
        monitor should have been.  The sampler is constructed here on purpose: loading
        alone never touched the value.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)
        config = load_config(path)
        assert config.gpu.names == expected
        sampler = Sampler(SamplerOptions(name_overrides=config.gpu.names))
        assert sampler.options.name_overrides == expected

    def test_a_mistyped_gpu_names_table_warns(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        write(path, "[gpu]\nnames = 5\n")
        assert any("gpu.names" in w for w in load_config(path).warnings)

    @pytest.mark.parametrize(
        "body",
        [
            "[state]\nprocess_filter = ['a']\n",
            "[process]\nmin_usage_percent = 'abc'\n",
        ],
    )
    def test_a_filter_value_of_the_wrong_type_is_coerced(
        self, tmp_path: Path, body: str
    ) -> None:
        """Both reached the running interface, long after loading, and raised there.

        ``process_filter`` went to ``str.strip`` and ``min_usage_percent`` to a comparison
        against a float, so the monitor died on the first repaint after the first filter
        evaluation rather than at start-up.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)
        config = load_config(path)
        assert isinstance(config.state.filter, str)
        assert isinstance(config.process.min_usage_percent, float)

    def test_a_numeric_filter_is_kept_as_text(self, tmp_path: Path) -> None:
        """``process_filter = 42`` is a plausible typo with an unambiguous intent."""
        path = tmp_path / "gputop.toml"
        write(path, "[state]\nprocess_filter = 42\n")
        assert load_config(path).state.filter == "42"

    @pytest.mark.parametrize(
        "body",
        [
            "[ui]\ngraph_history = 300\n",
            "[ui]\nhide_on_zero = true\n",
            "[gpu]\ndefault_index = 1\n",
            "[process]\ninclude_all_users = true\n",
            "[diagnostics]\ndump_dir = '/tmp/x'\n",
            "[diagnostics]\ndump_format = 'json'\n",
        ],
    )
    def test_a_key_that_no_longer_exists_is_reported_not_silently_dropped(
        self, tmp_path: Path, body: str
    ) -> None:
        """Every one of these was documented and read by nothing, so it was removed.

        The warning is the whole point of removing rather than leaving them: a config
        written against the old README now says so at start-up instead of accepting a
        setting that has never done anything.  The last two report the *section* rather
        than the key, because with the section gone there is no key left to name.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)

        config = load_config(path)

        assert any("unknown" in w and "ignored" in w for w in config.warnings), config.warnings

    @pytest.mark.parametrize(
        "body",
        [
            "[diagnostics]\ndump_dir = '/tmp/x'\n",
            "[generl]\ninterval_ms = 500\n",  # a misspelled section, not a removed one
            "stray = 1\n",  # not a table at all
        ],
    )
    def test_an_unknown_section_is_reported_too(self, tmp_path: Path, body: str) -> None:
        """A whole table nothing reads used to vanish without a word.

        The loader's promise is that a config written for a different version still works
        *and says so*.  That was kept for keys inside a section and quietly broken for
        sections: ``[diagnostics]`` could sit in the README, in the example file and in a
        user's config while being parsed cleanly and read by nothing.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)

        config = load_config(path)

        assert any("unknown section ignored" in w for w in config.warnings), config.warnings

    def test_a_graph_height_of_zero_means_fill_rather_than_nothing(
        self, tmp_path: Path
    ) -> None:
        """``0`` is the documented "let the layout decide" value, not a zero-height graph.

        It has to survive validation untouched: the loader clamps numbers it cannot use,
        and clamping this one to a minimum would silently opt every user into a pinned
        graph the moment they set it.
        """
        path = tmp_path / "gputop.toml"
        write(path, "[ui]\ngraph_height = 0\n")

        assert load_config(path).ui.graph_height == 0

    @pytest.mark.parametrize("value", [-1, -50])
    def test_a_negative_graph_height_falls_back_to_fill(
        self, tmp_path: Path, value: int
    ) -> None:
        """There is no such thing as a negative number of rows."""
        path = tmp_path / "gputop.toml"
        write(path, f"[ui]\ngraph_height = {value}\n")

        config = load_config(path)

        assert config.ui.graph_height == 0
        assert any("graph_height" in w for w in config.warnings)

    def test_a_devices_list_survives_a_shape_it_was_not_written_for(
        self, tmp_path: Path
    ) -> None:
        """``devices`` reaches the matcher element by element, so a nested shape matters.

        A TOML array holding a table or a nested array used to reach the device matcher
        intact; it now degrades to a warning naming the offending position, because the
        alternative is a ``TypeError`` from inside discovery at start-up.
        """
        path = tmp_path / "gputop.toml"
        write(path, '[gpu]\ndevices = ["card0", {a = 1}, "card1"]\n')

        config = load_config(path)

        assert config.gpu.devices == ("card0", "card1")
        assert any("gpu.devices[1]" in w for w in config.warnings), config.warnings

    @pytest.mark.parametrize(
        ("body", "expected", "warns"),
        [
            ("[gpu]\ndevices = 'card0'\n", (), True),
            ("[gpu]\ndevices = 5\n", (), True),
            ("[gpu]\ndevices = ['card0', '', ' ']\n", ("card0",), True),
            # A number is accepted silently on purpose: `devices = [0, 1]` cannot mean
            # anything else, and a warning on an unambiguous value is noise.
            ("[gpu]\ndevices = [0, 1]\n", ("0", "1"), False),
        ],
    )
    def test_a_devices_value_of_the_wrong_type_degrades_to_a_warning(
        self, tmp_path: Path, body: str, expected: tuple[str, ...], warns: bool
    ) -> None:
        """A bare string is not a one-element list.

        It is the obvious thing to type, and reading it as ``["card0"]`` would be kind --
        except that it would be the only list-shaped key in the file that accepts a bare
        scalar, so the next one written would be rejected for no visible reason.  The
        numbers *are* accepted, because ``devices = [0, 1]`` cannot mean anything else.
        """
        path = tmp_path / "gputop.toml"
        write(path, body)

        config = load_config(path)

        assert config.gpu.devices == expected
        assert bool(config.warnings) is warns, config.warnings


class TestDeviceFilter:
    """``gpu.devices`` narrows what is watched, and says so when it cannot."""

    def _two_cards(self, drm_root: Path) -> None:
        make_gpu(drm_root, card=0, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        make_gpu(drm_root, card=1, bdf="0000:03:00.0", metrics={"temperature_edge": 40})

    def _sampler(self, drm_root: Path, proc_root: Path, only: tuple[str, ...]) -> Sampler:
        return Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=False,
                interval_s=0.05,
                only_devices=only,
            )
        )

    @pytest.mark.parametrize(
        "name",
        [
            "0000:0c:00.0",  # the full PCI address, as --dump prints it
            "0c:00.0",  # the same address without the domain, as the header shows it
            "card0",  # the sysfs node, as ls /sys/class/drm shows it
            "CARD0",  # case is not a distinction anyone means to make
            "0000:0C:00.0",
        ],
    )
    def test_a_card_can_be_named_three_ways(
        self, drm_root: Path, proc_root: Path, name: str
    ) -> None:
        """Only one of the three spellings is what the tool prints, but all three are typed.

        A user reading ``ls /sys/class/drm`` writes ``card1``; a user reading a bug report
        writes ``0c:00.0``; only a script writes the full address.  Accepting one and
        warning about the other two would make the key look broken to two thirds of them.
        """
        self._two_cards(drm_root)

        sampler = self._sampler(drm_root, proc_root, (name,))
        devices = sampler.discover()

        assert [d.card for d in devices] == ["card0"]
        assert sampler.filter_warning == ""

    def test_the_filter_narrows_to_exactly_what_was_named(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The point of the key: watch one of two cards, not both."""
        self._two_cards(drm_root)

        devices = self._sampler(drm_root, proc_root, ("card1",)).discover()

        assert [d.bdf for d in devices] == ["0000:03:00.0"]

    def test_an_empty_filter_watches_every_card(self, drm_root: Path, proc_root: Path) -> None:
        """The default, and what a single-GPU machine wants."""
        self._two_cards(drm_root)

        sampler = self._sampler(drm_root, proc_root, ())
        devices = sampler.discover()

        assert len(devices) == 2
        assert sampler.filter_warning == ""

    def test_a_name_that_matches_nothing_is_reported_and_the_others_are_kept(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A typo in one element must not silently shorten the list.

        Dropping it quietly would turn ``["card0", "card9"]`` into a one-card monitor that
        looks exactly like a deliberate choice, with nothing on screen to say otherwise.
        """
        self._two_cards(drm_root)

        sampler = self._sampler(drm_root, proc_root, ("card0", "card9"))
        devices = sampler.discover()

        assert [d.card for d in devices] == ["card0"]
        assert "card9" in sampler.filter_warning
        assert "0000:0c:00.0" in sampler.filter_warning

    def test_a_filter_matching_nothing_at_all_watches_every_card(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A blank monitor is a worse answer to a typo than one card too many.

        Honouring the filter strictly would replace the display with "no amdgpu device
        found", which is a different and wrong answer to a question nobody asked.  The
        warning still names what was not found, so the mistake is not hidden.
        """
        self._two_cards(drm_root)

        sampler = self._sampler(drm_root, proc_root, ("0000:ff:00.0",))
        devices = sampler.discover()

        assert len(devices) == 2
        assert "0000:ff:00.0" in sampler.filter_warning

    def test_the_filter_applies_to_the_sampling_thread_too(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``start()`` and ``discover()`` share one path, so they cannot disagree.

        A filter honoured on the one-shot ``--dump`` path and ignored by the sampling
        thread would make the dump describe a machine the interface is not watching.
        """
        self._two_cards(drm_root)
        sampler = self._sampler(drm_root, proc_root, ("card1",))
        try:
            sampler.start()
            snapshot = sampler.wait(timeout=5.0)
        finally:
            sampler.stop()
        assert snapshot is not None
        assert [m.device.card for m in snapshot.devices] == ["card1"]

    def test_the_selected_devices_keep_their_indices(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The filtered list is the discovered list, in the discovered order.

        A device's index is its position in this sequence, and it is what the focus
        state, the radeontop arguments and the dump output all address the card by.
        Reordering to match the config's order would renumber the cards underneath all
        three, so the comparison is against what discovery produced rather than against
        an ordering written out by hand.
        """
        self._two_cards(drm_root)

        everything = self._sampler(drm_root, proc_root, ()).discover()
        filtered = self._sampler(drm_root, proc_root, ("card1", "card0")).discover()

        assert [d.bdf for d in filtered] == [d.bdf for d in everything]
        assert [(d.card, d.index) for d in filtered] == [(d.card, d.index) for d in everything]


class TestDeviceFilterOnTheCommandLine:
    """``--devices`` and ``--dump`` both go through the sampler, so both honour the key.

    The failure this guards against is a script reading a dump that describes a different
    card set from the one the interface is watching, with nothing to say so.
    """

    def _run(self, argv: list[str], drm_root: Path, proc_root: Path) -> dict:
        from gputop.cli import main

        buffer = io.StringIO()
        status = main(
            [*argv, "--drm-root", str(drm_root), "--proc-root", str(proc_root)],
            buffer,
        )
        assert status == 0, f"exit {status}"
        return json.loads(buffer.getvalue())

    def test_the_devices_listing_is_filtered_and_says_so(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        make_gpu(drm_root, card=0, bdf="0000:0c:00.0")
        make_gpu(drm_root, card=1, bdf="0000:03:00.0")
        config = tmp_path / "gputop.toml"
        write(config, '[gpu]\ndevices = ["0000:0c:00.0", "card9"]\n')

        payload = self._run(["--devices", "-c", str(config)], drm_root, proc_root)

        assert [d["card"] for d in payload["devices"]] == ["card0"]
        assert any("card9" in w for w in payload["warnings"]), payload["warnings"]

    def test_the_dump_is_filtered_too(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        make_gpu(drm_root, card=0, bdf="0000:0c:00.0")
        make_gpu(drm_root, card=1, bdf="0000:03:00.0")
        config = tmp_path / "gputop.toml"
        write(config, '[gpu]\ndevices = ["card1"]\n')

        payload = self._run(
            ["--dump", "-c", str(config), "--no-processes"], drm_root, proc_root
        )

        assert [d["card"] for d in payload["devices"]] == ["card1"]


class TestSampler:
    """The sampler ties the readers together and must survive hostile input."""

    def _sampler(self, drm_root: Path, proc_root: Path) -> Sampler:
        return Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=True,
                interval_s=0.05,
            )
        )

    def test_empty_tree_produces_an_empty_snapshot(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        sampler = self._sampler(drm_root, proc_root)
        assert sampler.discover() == ()

        snapshot = sampler.sample_once()

        assert snapshot.devices == ()
        assert snapshot.processes == ()
        assert snapshot.sequence == 1

    def test_snapshot_combines_devices_and_processes(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        make_process(proc_root, 100)
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        snapshot = sampler.sample_once()

        assert len(snapshot.devices) == 1
        assert len(snapshot.processes) == 1
        assert snapshot.total_process_count == 1
        assert snapshot.stats.ticks == 1
        assert not snapshot.is_partial_process_view

    def test_partial_process_visibility_is_reported(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """Hiding other users' processes must be visible, not silently misleading."""
        make_gpu(drm_root, bdf="0000:0c:00.0")
        make_process(proc_root, 100)
        (proc_root / "200").mkdir()
        (proc_root / "200" / "fd").symlink_to(proc_root / "999" / "fd")
        (proc_root / "200" / "fdinfo").symlink_to(proc_root / "999" / "fdinfo")

        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()
        snapshot = sampler.sample_once()

        assert snapshot.total_process_count == 2
        assert snapshot.visible_process_count == 1
        assert snapshot.is_partial_process_view

    def test_a_device_without_binary_metrics_still_produces_a_sample(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A pre-5.19 kernel has no ``gpu_metrics``; the sysfs path must carry it."""
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            metrics=None,
            metrics_abi=None,
            extra_sysfs={
                "pp_dpm_sclk": "0: 500Mhz *\n1: 2400Mhz \n",
                "mem_info_vis_vram_used": "1048576",
                "mem_info_vis_vram_total": "8589934592",
                "gpu_busy_percent": "42",
            },
        )
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        metrics = sampler.sample_once().devices[0]

        assert metrics.metrics_abi is None
        assert metrics.sclk.current == 500
        assert metrics.sclk.maximum == 2400
        assert metrics.sclk.source == "dpm"
        assert metrics.gpu_busy_percent == 42.0
        assert metrics.vram.used == 1048576
        # 1 MiB of 8 GiB is 0.0122%, not 12.2%.
        assert metrics.vram.percent == pytest.approx(0.0122, abs=0.0001)

    def test_one_broken_device_does_not_hide_the_others(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A failing device is isolated so the healthy one is still reported."""
        make_gpu(drm_root, card=0, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        make_gpu(drm_root, card=1, bdf="0000:03:00.0", metrics={"temperature_edge": 40})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        # Make one device's directory unusable without touching the other.
        broken = sampler.devices[0].device_dir / "gpu_metrics"
        broken.unlink()
        broken.mkdir()

        snapshot = sampler.sample_once()

        # Both devices still produce a sample; the broken one simply has no binary data.
        assert len(snapshot.devices) == 2
        assert sorted(str(m.metrics_abi) for m in snapshot.devices) == ["None", "v1.3"]
        broken_metrics = next(m for m in snapshot.devices if m.metrics_abi is None)
        assert broken_metrics.sclk.current is None
        healthy = next(m for m in snapshot.devices if m.metrics_abi == "v1.3")
        assert healthy.temperatures[0].celsius == 49

    def test_options_are_validated(self) -> None:
        options = SamplerOptions(interval_s=0.001, history_length=0).validated()
        assert options.interval_s == 0.05
        assert options.history_length == 1

    def test_sample_once_is_repeatable(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        first = sampler.sample_once()
        second = sampler.sample_once()

        assert first.sequence == 1
        assert second.sequence == 2
        assert first.timestamp_ns <= second.timestamp_ns

    def test_thread_publishes_snapshots(self, drm_root: Path, proc_root: Path) -> None:
        """The background thread hands immutable snapshots over the queue."""
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.start()
        try:
            snapshot = sampler.wait(timeout=5.0)
        finally:
            sampler.stop()

        assert snapshot is not None
        assert len(snapshot.devices) == 1
        assert sampler.history()

    def test_set_interval_changes_the_rate_the_thread_actually_samples_at(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The tick loop must not cache the period it started with.

        ``set_interval`` reported the new value back to every caller -- including the
        ``+``/``-`` keys, which also reset their own refresh timer -- so the only thing
        that failed to notice was the loop itself, and the interval keys silently did
        nothing to the rate the GPU was sampled at until the next start.
        """
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=False,
                interval_s=30.0,
            )
        )
        sampler.start()
        try:
            # Let the first tick through, then ask for a much faster rate.
            assert sampler.wait(timeout=5.0) is not None
            assert sampler.set_interval(0.05) == 0.05

            deadline = time.monotonic() + 5.0
            ticks = 0
            while time.monotonic() < deadline and ticks < 5:
                if sampler.wait(timeout=2.0) is not None:
                    ticks += 1
        finally:
            sampler.stop()

        assert ticks >= 5, "the loop kept the 30 s period it was started with"

    def test_latest_returns_the_newest_queued_snapshot(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``latest`` drains, so a slow repaint cannot hand the UI a stale sample."""
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()
        for _ in range(3):
            sampler.sample_once()

        newest = sampler.latest()

        assert newest is not None
        assert newest.sequence == 3
        assert sampler.latest() is None

    def test_source_report_tracks_the_binary_abi(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = self._sampler(drm_root, proc_root)
        sampler.discover()

        snapshot = sampler.sample_once()

        assert snapshot.source.metrics_abi == "v1.3"
        assert snapshot.source.using_binary_metrics

    def _await_snapshot(self, sampler: Sampler, timeout: float = 5.0) -> GpuSnapshot | None:
        """Poll for one snapshot, keeping it: ``latest`` empties the queue when it returns."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = sampler.latest()
            if snapshot is not None:
                return snapshot
            time.sleep(0.01)
        return None

    def test_request_tick_wakes_a_sleeping_sampler(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The UI's refresh key asks for a sample rather than taking one on its own thread.

        ``sample_once`` runs a whole tick, and the UI called it from the Textual thread
        while the sampler's own thread was ticking too -- two ticks sharing the sequence
        counter, the history ring and the collector's delta baseline, which is the one
        thing the module's design says must never happen.  ``request_tick`` pokes the
        loop's wake-up event instead, so the sampler thread stays the only owner.
        """
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=False,
                # Long enough that only the nudge can produce the second sample.
                interval_s=30.0,
            )
        )
        sampler.start()
        try:
            first = self._await_snapshot(sampler)
            assert first is not None, "the first tick did not happen"
            sampler.request_tick()
            second = self._await_snapshot(sampler)
            assert second is not None, "request_tick did not produce a sample"
            assert second.sequence > first.sequence
        finally:
            sampler.stop()

    def test_a_tick_runs_on_the_sampler_thread_and_nowhere_else(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """The tick body must have a single owner.

        This is the invariant ``sample_once`` violated when the UI called it: a second
        ``_tick`` on the UI thread is two writers to ``_sequence``, ``_history`` and
        ``_stats`` at once, on a build that is explicitly free-threaded.
        """
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        sampler = Sampler(
            SamplerOptions(
                drm_root=drm_root,
                proc_root=proc_root,
                collect_processes=False,
                interval_s=0.05,
            )
        )
        sampler.start()
        try:
            assert self._await_snapshot(sampler) is not None
            assert "gputop-sampler" in {t.name for t in threading.enumerate()}
        finally:
            sampler.stop()


class TestSnapshot:
    """Snapshot lookups used by the UI."""

    def test_device_by_index(self) -> None:
        snapshot = GpuSnapshot(sequence=1, timestamp_ns=0)
        assert snapshot.device_by_index(0) is None

    def test_partial_visibility_flags(self) -> None:
        partial = GpuSnapshot(
            sequence=1, timestamp_ns=0, visible_process_count=1, total_process_count=5
        )
        assert partial.is_partial_process_view

    def test_default_stats_are_zeroed(self) -> None:
        assert SamplerStats().ticks == 0

    def test_source_report_defaults(self) -> None:
        assert not SourceReport().using_binary_metrics


class TestCli:
    """The JSON dump is the tool's scriptable contract."""

    def _run(self, argv: list[str]) -> dict:
        from gputop.cli import main

        buffer = io.StringIO()
        status = main(
            [*argv, "--drm-root", str(self._drm), "--proc-root", str(self._proc)],
            buffer,
        )
        assert status == 0, f"exit {status}"
        return json.loads(buffer.getvalue())

    def test_devices_listing(self, drm_root: Path, proc_root: Path) -> None:
        self._drm, self._proc = drm_root, proc_root
        make_gpu(drm_root, bdf="0000:0c:00.0", device_id=0x73BF)

        payload = self._run(["--devices"])

        assert payload["count"] == 1
        assert payload["devices"][0]["bdf"] == "0000:0c:00.0"
        assert payload["devices"][0]["name"] == "AMD Radeon RX 6800"

    def test_dump_emits_json(self, drm_root: Path, proc_root: Path) -> None:
        self._drm, self._proc = drm_root, proc_root
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            metrics={"temperature_edge": 49, "average_gfx_activity": 12},
        )
        make_process(proc_root, 100)

        payload = self._run(["--dump", "--interval", "0.05", "--no-processes"])

        assert payload["metrics_abi"] == "v1.3"
        assert payload["using_binary_metrics"] is True
        device = payload["devices"][0]
        assert device["temperatures"][0] == {
            "label": "edge",
            "celsius": 49,
            "source": "gpu_metrics",
        }
        assert device["gpu_busy_percent"] == 12.0

    def test_dump_reports_the_link_rate_not_a_generation(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``speed`` is the kernel's 0.1 GT/s unit, so the decoded rate has to travel with it.

        ``--dump`` consumers get ``speed: 80`` verbatim; without ``gt_per_second`` in the
        payload the only way to read it is to guess, and guessing 16x-per-generation is
        what turned a Gen3 link into a reported Gen5.
        """
        self._drm, self._proc = drm_root, proc_root
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            metrics={"pcie_link_width": 16, "pcie_link_speed": 80},
        )

        payload = self._run(["--dump", "--interval", "0.05", "--no-processes"])

        assert payload["devices"][0]["pcie"] == {
            "width": 16,
            "speed": 80,
            "gt_per_second": 8.0,
            "generation": 3,
            "describe": "Gen3 x16",
        }

    def test_dump_without_a_device_reports_and_exits_nonzero(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        from gputop.cli import main

        buffer = io.StringIO()
        status = main(
            ["--dump", "--drm-root", str(drm_root), "--proc-root", str(proc_root)], buffer
        )

        assert status == 3
        payload = json.loads(buffer.getvalue())
        assert payload["devices"] == []
        assert any("no amdgpu device" in w for w in payload["warnings"])

    def test_pretty_output_is_indented(self, drm_root: Path, proc_root: Path) -> None:
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0")
        buffer = io.StringIO()
        main(
            [
                "--devices",
                "--pretty",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
            ],
            buffer,
        )
        assert "\n  " in buffer.getvalue()

    def test_unavailable_metrics_are_null_not_zero(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """A consumer must be able to tell "not available" from "measured zero"."""
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        buffer = io.StringIO()
        main(
            [
                "--dump",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        device = json.loads(buffer.getvalue())["devices"][0]

        # No VRAM attributes in the fake tree, so these must be null rather than 0.
        assert device["vram"] == {"used": None, "total": None, "percent": None}
        # Absent entirely: this layout has no PCIe or voltage fields at all.
        assert device["pcie"] is None
        assert device["voltages_mv"] == {}
        # throttle_status exists in v1.3 and reads zero, which means "not throttling" --
        # a real reading, so it must be reported rather than nulled out.
        assert device["throttle"] == {"raw": 0, "active": [], "is_throttling": False}

    @pytest.mark.parametrize("flag", ["-1", "-0.5", "0", "0.001"])
    def test_an_out_of_range_interval_does_not_crash_the_one_shot_paths(
        self, drm_root: Path, proc_root: Path, flag: str
    ) -> None:
        """``-i`` went into an unvalidated ``SamplerOptions`` that the sleeps read directly.

        ``--check -i -1`` died with ``ValueError: sleep length must be un-negative``, and
        ``-i 0`` was silently replaced by the config value because the guard was a truth
        test.  The README promises the interval is floored; the CLI was the one place it
        was not.
        """
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        buffer = io.StringIO()
        code = main(
            [
                "--check",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
                "-i",
                flag,
            ],
            buffer,
        )
        assert code in (0, 1)
        assert "Traceback" not in buffer.getvalue()

    def test_the_sampler_applies_the_floor_to_the_flag(self) -> None:
        from gputop.cli import sampler_options
        from gputop.config import load_config
        from gputop.sampler import Sampler

        args = argparse.Namespace(
            interval=0.0, drm_root=None, proc_root=None, no_processes=True, kind=None
        )
        options = sampler_options(args, load_config(Path("/nonexistent.toml")))
        # The validated options are what the one-shot paths sleep on.
        assert Sampler(options).options.interval_s >= 0.05


class TestBlocksCliFlags:
    """``--blocks`` / ``--no-blocks`` and what the dump reports about the panel."""

    def _dump(self, argv: list[str], drm_root: Path, proc_root: Path) -> dict:
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        buffer = io.StringIO()
        status = main(
            [
                *argv,
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        assert status == 0
        return json.loads(buffer.getvalue())

    def test_blocks_is_off_unless_asked_for(self, drm_root: Path, proc_root: Path) -> None:
        """Default-off keeps every user from getting a status line they cannot fix."""
        payload = self._dump(["--dump"], drm_root, proc_root)
        assert payload["blocks"]["status"] == "disabled"

    def test_the_flag_overrides_the_config_file(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        config = tmp_path / "gputop.toml"
        config.write_text("[blocks]\nenabled = true\nbinary = 'no-such-binary-xyz'\n")
        payload = self._dump(
            ["--dump", "--config", str(config), "--no-blocks"], drm_root, proc_root
        )
        assert payload["blocks"]["status"] == "disabled"

    def test_enabling_a_missing_binary_is_reported_as_missing_not_as_an_error(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        config = tmp_path / "gputop.toml"
        config.write_text("[blocks]\nenabled = true\nbinary = 'no-such-binary-xyz'\n")
        payload = self._dump(["--dump", "--config", str(config)], drm_root, proc_root)
        assert payload["blocks"]["status"] == "missing"
        assert "radeontop" in payload["blocks"]["hint"]

    def test_no_block_data_is_null_rather_than_zeroed(
        self, drm_root: Path, proc_root: Path
    ) -> None:
        """``None`` and an empty sample are different answers; zero would be a lie."""
        payload = self._dump(["--dump"], drm_root, proc_root)
        assert payload["devices"][0]["blocks"] is None

    def test_the_power_tables_appear_in_the_dump(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(
            drm_root,
            bdf="0000:0c:00.0",
            extra_sysfs={"pp_power_profile_mode": " 0 BOOTUP_DEFAULT*:\n 1 VIDEO :\n"},
        )
        buffer = io.StringIO()
        from gputop.cli import main

        main(
            [
                "--dump",
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        device = json.loads(buffer.getvalue())["devices"][0]
        assert device["power_profiles"]["active"] == "BOOTUP_DEFAULT"
        assert device["power_profiles"]["available"] == ["BOOTUP_DEFAULT", "VIDEO"]
        # No pp_od_clk_voltage in the fixture: absent, not a table of zeroes.
        assert device["odc"]["present"] is False
        assert device["odc"]["domains"] == []


class TestAlertDump:
    """The threshold state is scriptable, not only visible."""

    def _dump(self, argv: list[str], drm_root: Path, proc_root: Path) -> dict:
        from gputop.cli import main

        buffer = io.StringIO()
        main(
            [
                *argv,
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        return json.loads(buffer.getvalue())

    def test_a_quiet_card_reports_no_breach(self, drm_root: Path, proc_root: Path) -> None:
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_hotspot": 56})
        payload = self._dump(["--dump"], drm_root, proc_root)
        assert payload["alerts"]["enabled"] is True
        assert payload["alerts"]["level"] == "ok"
        assert payload["alerts"]["breaches"] == []

    def test_a_hot_card_names_the_metric_reading_and_limit(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        config = tmp_path / "gputop.toml"
        config.write_text("[alerts]\ntemp_c = 50\n")
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_hotspot": 56})
        payload = self._dump(["--dump", "--config", str(config)], drm_root, proc_root)
        assert payload["alerts"]["level"] == "alert"
        assert payload["alerts"]["breaches"][0]["metric"] == "temp junction"
        assert payload["alerts"]["thresholds"]["temp_c"] == 50.0

    def test_disabling_alerts_reports_themself_off(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        config = tmp_path / "gputop.toml"
        config.write_text("[alerts]\nenabled = false\n")
        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_hotspot": 200})
        payload = self._dump(["--dump", "--config", str(config)], drm_root, proc_root)
        assert payload["alerts"] == {"enabled": False, "level": "ok", "breaches": []}


class TestLogFlag:
    """``--log`` from the command line."""

    def test_a_dump_is_recorded_to_the_named_file(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        from gputop.cli import main
        from gputop.sessionlog import read_log

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        target = tmp_path / "session.csv"
        buffer = io.StringIO()
        main(
            [
                "--dump",
                "--log",
                str(target),
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        lines = list(read_log(target))
        assert lines[0].startswith("time,sequence")
        assert len(lines) == 2

    def test_an_unwritable_target_warns_but_still_dumps(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        """Recording a session is not worth refusing to produce one."""
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        blocker = tmp_path / "a-file"
        blocker.write_text("x")
        buffer = io.StringIO()
        status = main(
            [
                "--dump",
                "--log",
                str(blocker / "nested" / "s.csv"),
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        assert status == 0
        assert json.loads(buffer.getvalue())["devices"]

    def test_an_unknown_suffix_warns_but_still_dumps(
        self, drm_root: Path, proc_root: Path, tmp_path: Path
    ) -> None:
        from gputop.cli import main

        make_gpu(drm_root, bdf="0000:0c:00.0", metrics={"temperature_edge": 49})
        buffer = io.StringIO()
        status = main(
            [
                "--dump",
                "--log",
                str(tmp_path / "session.txt"),
                "--drm-root",
                str(drm_root),
                "--proc-root",
                str(proc_root),
                "--no-processes",
            ],
            buffer,
        )
        assert status == 0
        assert json.loads(buffer.getvalue())["devices"]
        assert not (tmp_path / "session.txt").exists()


class TestNewConfigSections:
    """The three new config sections load, clamp and warn."""

    def test_a_file_with_the_new_sections_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        path.write_text(
            "[blocks]\n"
            "enabled = true\n"
            "binary = '/usr/bin/radeontop'\n"
            "ticks = 60\n"
            "interval_s = 2\n"
            "[alerts]\n"
            "enabled = true\n"
            "temp_c = 85\n"
            "power_percent = 90\n"
            "vram_percent = 80\n"
            "flash_hz = 2\n"
            "[log]\n"
            "zstd_level = 9\n"
            "interval_s = 0\n"
        )
        config = load_config(path)
        assert config.blocks.enabled is True
        assert config.blocks.ticks == 60
        assert config.blocks.interval_s == 2
        assert config.alerts.temp_c == 85
        assert config.alerts.flash_hz == 2
        assert config.log.zstd_level == 9
        assert config.warnings == ()

    def test_a_sub_second_radeontop_interval_is_clamped_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """radeontop floors ``-i`` at 1 s; honouring 500 ms would update 4x slower."""
        path = tmp_path / "gputop.toml"
        path.write_text("[blocks]\ninterval_s = 0\n")
        config = load_config(path)
        assert config.blocks.interval_s == 1
        assert any("interval_s" in w for w in config.warnings)

    @pytest.mark.parametrize(
        "line",
        [
            'temp_c = "hot"',
            "temp_c = true",
            "power_percent = 'lots'",
        ],
    )
    def test_a_non_numeric_threshold_warns_instead_of_crashing(
        self, tmp_path: Path, line: str
    ) -> None:
        """A malformed file yields warnings and a working config, never a failed start."""
        path = tmp_path / "gputop.toml"
        path.write_text(f"[alerts]\n{line}\n")
        config = load_config(path)
        assert isinstance(config.alerts.temp_c, float)
        assert config.warnings

    def test_an_unknown_key_in_a_new_section_is_a_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "gputop.toml"
        path.write_text("[blocks]\nenabled = true\nnonsense = 1\n")
        config = load_config(path)
        assert config.blocks.enabled is True
        assert any("nonsense" in w for w in config.warnings)
