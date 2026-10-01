"""Persisting the interface state into the config file.

The contract being tested is narrow and mostly about *not* breaking things: the hand
written parts of the file survive, the write is atomic, a file that cannot be written is
reported rather than raised, and a saved value from a future or past release is ignored
with a warning instead of stopping gputop from starting.
"""

import os
import stat
import tomllib
from pathlib import Path

import pytest

from gputop.config import (
    STATE_ENV,
    Config,
    GeneralConfig,
    StateConfig,
    load_config,
    state_path,
)
from gputop.state import (
    KEY_ORDER,
    STATE_SECTION,
    load_state,
    read_document,
    save_state,
    state_table,
)

FULL = StateConfig(
    theme="gruvbox",
    interval_ms=250,
    sort="compute",
    sort_reverse=True,
    filter="firefox",
    show_idle=True,
    process_tree=True,
    full_command=True,
    overview=True,
    gpu_index=1,
)


class TestRoundTrip:
    """What goes in comes out."""

    def test_every_field_survives(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        assert save_state(FULL, path) is None
        assert load_state(path) == FULL

    def test_the_section_is_named_state(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        assert STATE_SECTION in read_document(path)

    def test_an_absent_file_loads_as_defaults(self, tmp_path: Path) -> None:
        assert load_state(tmp_path / "absent.toml") == StateConfig()

    def test_a_file_without_the_section_loads_as_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[ui]\ntheme = 'dracula'\n")
        assert load_state(path) == StateConfig()

    def test_a_saved_session_survives_load_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole round trip, through the path a real start-up takes.

        ``[state]`` is keyed differently from :class:`StateConfig` on purpose -- the file
        says ``process_sort`` where the field is ``sort`` -- so the translation lives in
        one place.  ``load_config`` used to bypass it on the default path, reusing the
        document it had already parsed, and every key gputop itself writes came back as
        "unknown key ignored" with the session reset to its defaults.  On the default
        configuration, so it was not a rare setup.
        """
        path = tmp_path / "config.toml"
        monkeypatch.setenv(STATE_ENV, str(path))
        monkeypatch.setenv("GPUTOP_CONFIG", str(path))
        assert save_state(FULL, path) is None
        config = load_config()
        assert config.state == FULL
        # The two values the rest of the program reads through the config are applied.
        assert config.ui.theme == FULL.theme
        assert config.general.interval_ms == FULL.interval_ms
        assert not [w for w in config.warnings if w.startswith("state.")], config.warnings

    def test_saving_twice_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        first = path.read_text()
        save_state(FULL, path)
        assert path.read_text() == first

    def test_a_later_save_replaces_rather_than_accumulates(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        save_state(StateConfig(theme="dracula"), path)
        assert load_state(path) == StateConfig(theme="dracula")

    def test_an_unrecorded_field_is_not_written(self, tmp_path: Path) -> None:
        """``None`` means "never chosen" and must not become a value on the next read."""
        path = tmp_path / "config.toml"
        save_state(StateConfig(theme="dracula"), path)
        section = read_document(path)[STATE_SECTION]
        assert "theme" in section
        # The two other optional fields were never set; writing a placeholder for them
        # would turn "the user never chose an interval" into "the user chose none".
        assert "interval_ms" not in section
        assert "process_sort" not in section

    def test_keys_are_written_in_a_readable_order(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        keys = list(read_document(path)[STATE_SECTION])
        assert keys == [key for key in KEY_ORDER if key in keys]

    def test_every_written_key_has_a_place_in_the_order(self) -> None:
        """``state_table`` projects through ``KEY_ORDER``, so a key missing from it is dropped.

        Nothing else catches this: the setting simply stops being written, no warning is
        emitted, and it looks exactly like gputop forgetting it.  The failure this guards
        against is silent by construction, so the invariant has to be asserted.
        """
        from gputop.state import _KEYS

        assert set(_KEYS.values()) == set(KEY_ORDER)

    def test_the_file_is_valid_toml(self, tmp_path: Path) -> None:
        """It is read back with tomllib in :func:`load_state`, so it must parse."""
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        with path.open("rb") as handle:
            assert STATE_SECTION in tomllib.load(handle)


class TestPreservation:
    """The rest of the user's file is not ours to destroy."""

    def test_other_sections_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[gpu]\nkind = 'igpu'\n\n[ui]\ntheme = 'gruvbox'\n")
        save_state(StateConfig(theme="dracula"), path)
        document = read_document(path)
        assert document["gpu"] == {"kind": "igpu"}
        assert document["ui"] == {"theme": "gruvbox"}

    def test_a_new_state_file_keeps_the_others_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[gpu.names]\n'0000:0c:00.0' = 'My GPU'\n")
        save_state(StateConfig(theme="dracula"), path)
        assert read_document(path)["gpu"]["names"] == {"0000:0c:00.0": "My GPU"}

    def test_an_existing_state_is_replaced_not_merged(self, tmp_path: Path) -> None:
        """A later session drops the keys it no longer uses, rather than leaving them."""
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        save_state(StateConfig(theme="dracula"), path)
        assert read_document(path)[STATE_SECTION] == state_table(StateConfig(theme="dracula"))


class TestFailures:
    """Nothing here may raise: the monitor has been running all along."""

    def test_a_malformed_file_is_never_overwritten(self, tmp_path: Path) -> None:
        """An unparseable file is left exactly as it was, and the refusal is reported.

        This used to assert the opposite -- that the file is *replaced* so the user is not
        left with one that never parses.  That trade was made without noticing which files
        actually fail to parse: in practice it is a hand-written config with one typo in it,
        not ``this is = = not toml``.  A single trailing comma used to cost the user every
        other key in the file, silently, because the unreadable document was replaced with
        a bare ``[state]`` and the function reported success.  The self-healing is worth
        far less than the settings, and the message tells the user exactly what to do, so
        the file is left alone and gputop saves again once it parses.
        """
        path = tmp_path / "config.toml"
        original = "[general]\ninterval_ms = 1000,\n"
        path.write_text(original)
        problem = save_state(StateConfig(theme="dracula"), path)
        assert problem is not None
        assert "not valid TOML" in problem
        assert path.read_text() == original

    def test_a_typo_in_a_real_config_does_not_cost_the_rest_of_it(self, tmp_path: Path) -> None:
        """The case the refusal exists for: one mistake in an otherwise good file."""
        path = tmp_path / "config.toml"
        original = (
            "[general]\ninterval_ms = 1000,\n"  # the trailing comma\n"
            '[gpu.names]\n"0000:0c:00.0" = "My GPU"\n'
        )
        path.write_text(original)
        assert save_state(StateConfig(theme="dracula"), path) is not None
        assert path.read_text() == original

    def test_saving_resumes_once_the_file_is_fixed(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("this is = = not toml")
        assert save_state(StateConfig(theme="dracula"), path) is not None
        path.write_text("[general]\ninterval_ms = 1000\n")
        assert save_state(StateConfig(theme="dracula"), path) is None
        assert load_state(path) == StateConfig(theme="dracula")
        # The hand-written section survived the save, which is the whole point.
        assert read_document(path)["general"] == {"interval_ms": 1000}

    def test_a_malformed_file_reads_as_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("this is = = not toml")
        assert load_state(path) == StateConfig()

    def test_an_unreadable_file_reads_as_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[state]\ntheme = 'dracula'\n")
        path.chmod(0o000)
        try:
            if os.getuid() == 0:  # pragma: no cover - root can read anything
                pytest.skip("running as root: the mode does not deny access")
            assert load_state(path) == StateConfig()
        finally:
            path.chmod(0o644)

    def test_a_write_into_a_missing_directory_creates_it(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "config.toml"
        assert save_state(StateConfig(theme="dracula"), path) is None
        assert load_state(path) == StateConfig(theme="dracula")

    def test_a_write_failure_is_reported_not_raised(self, tmp_path: Path) -> None:
        """A full disk or a read-only home must not crash the monitor on the way out."""
        directory = tmp_path / "locked"
        directory.mkdir()
        directory.chmod(0o500)
        try:
            if os.getuid() == 0:  # pragma: no cover - root ignores the mode
                pytest.skip("running as root: the mode does not deny access")
            message = save_state(StateConfig(theme="dracula"), directory / "config.toml")
            assert message is not None
            assert "config.toml" in message
        finally:
            directory.chmod(0o700)

    def test_a_failed_write_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        directory = tmp_path / "locked"
        directory.mkdir()
        directory.chmod(0o500)
        try:
            if os.getuid() == 0:  # pragma: no cover - root ignores the mode
                pytest.skip("running as root: the mode does not deny access")
            save_state(StateConfig(theme="dracula"), directory / "config.toml")
            assert list(directory.iterdir()) == []
        finally:
            directory.chmod(0o700)

    def test_an_unknown_key_in_the_section_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[state]\ntheme = 'dracula'\nfrom_the_future = 1\n")
        assert load_state(path) == StateConfig(theme="dracula")


class TestAtomicity:
    """The file being rewritten is the file the next start reads."""

    def test_the_write_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        assert [entry.name for entry in tmp_path.iterdir()] == ["config.toml"]

    def test_the_previous_content_is_intact_if_the_write_does_not_happen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        before = path.read_text()

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("gputop.state.os.replace", boom)
        assert save_state(StateConfig(theme="dracula"), path) is not None
        assert path.read_text() == before
        assert [entry.name for entry in tmp_path.iterdir()] == ["config.toml"]

    def test_an_existing_file_keeps_its_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[ui]\n")
        path.chmod(0o640)
        save_state(FULL, path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    def test_a_new_file_is_not_world_readable(self, tmp_path: Path) -> None:
        """It is about to hold the user's own settings."""
        path = tmp_path / "config.toml"
        save_state(FULL, path)
        assert not stat.S_IMODE(path.stat().st_mode) & stat.S_IROTH


class TestStatePath:
    """Where the file is, and why the environment variable exists."""

    def test_defaults_under_the_config_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv(STATE_ENV, raising=False)
        assert state_path() == tmp_path / ".config" / "gputop" / "config.toml"

    def test_the_environment_variable_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "elsewhere.toml"))
        assert state_path() == tmp_path / "elsewhere.toml"

    def test_an_explicit_path_wins_over_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "elsewhere.toml"))
        assert state_path(tmp_path / "explicit.toml") == tmp_path / "explicit.toml"

    def test_home_is_read_at_call_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A test that changes ``$HOME`` after import must still be honoured."""
        monkeypatch.delenv(STATE_ENV, raising=False)
        first = tmp_path / "one"
        second = tmp_path / "two"
        monkeypatch.setenv("HOME", str(first))
        assert state_path().parents[2] == first
        monkeypatch.setenv("HOME", str(second))
        assert state_path().parents[2] == second


class TestConfigIntegration:
    """How the saved state meets the hand-written configuration."""

    def test_a_saved_theme_overrides_the_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Last session wins: that is what "how did I have this" means."""
        config_file = tmp_path / "gputop.toml"
        config_file.write_text("[ui]\ntheme = 'gruvbox'\n")
        monkeypatch.setenv("GPUTOP_CONFIG", str(config_file))
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        save_state(StateConfig(theme="dracula"), tmp_path / "config.toml")
        assert load_config().ui.theme == "dracula"

    def test_without_a_saved_theme_the_config_file_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "gputop.toml"
        config_file.write_text("[ui]\ntheme = 'gruvbox'\n")
        monkeypatch.setenv("GPUTOP_CONFIG", str(config_file))
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        assert load_config().ui.theme == "gruvbox"

    def test_a_saved_interval_reaches_the_sampler(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        save_state(StateConfig(interval_ms=250), tmp_path / "config.toml")
        assert load_config().interval_s == pytest.approx(0.25)

    def test_an_explicit_config_file_ignores_the_saved_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming a file means naming the whole configuration.

        Without this, running the test suite -- or any scripted invocation -- would depend
        on whatever the last interactive session happened to leave behind.
        """
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        save_state(StateConfig(theme="dracula", interval_ms=250), tmp_path / "config.toml")
        config_file = tmp_path / "gputop.toml"
        config_file.write_text("[ui]\ntheme = 'gruvbox'\n")
        config = load_config(config_file)
        assert config.ui.theme == "gruvbox"
        assert config.general.interval_ms == 1000

    def test_the_state_can_still_be_asked_for_explicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        save_state(StateConfig(theme="dracula"), tmp_path / "config.toml")
        assert load_config().state.theme == "dracula"

    def test_an_unknown_saved_theme_is_ignored_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        path = tmp_path / "config.toml"
        path.write_text("[state]\ntheme = 'from-the-future'\n")
        config = load_config()
        assert config.ui.theme == "default"
        assert any("from-the-future" in warning for warning in config.warnings)

    def test_an_unknown_saved_sort_column_is_ignored_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        path = tmp_path / "config.toml"
        path.write_text("[state]\nprocess_sort = 'quantum'\n")
        config = load_config()
        assert config.state.sort is None
        assert any("quantum" in warning for warning in config.warnings)

    def test_an_out_of_range_saved_interval_is_ignored_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        path = tmp_path / "config.toml"
        path.write_text("[state]\ninterval_ms = 5\n")
        config = load_config()
        assert config.state.interval_ms is None
        assert any("out of range" in warning for warning in config.warnings)

    def test_a_malformed_state_file_does_not_stop_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        (tmp_path / "config.toml").write_text("= = =")
        assert isinstance(load_config(), Config)

    def test_the_general_section_still_clamps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A saved interval goes through the same validation, not around it."""
        monkeypatch.setenv(STATE_ENV, str(tmp_path / "config.toml"))
        path = tmp_path / "config.toml"
        path.write_text("[state]\ninterval_ms = 10_000_000\n")
        assert load_config().general.interval_ms == 1000
        assert isinstance(load_config().general, GeneralConfig)
