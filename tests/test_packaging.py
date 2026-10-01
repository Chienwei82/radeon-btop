"""The shipped documentation has to match the shipped program.

A man page that documents a flag which no longer exists, or a completion file that has
never heard of one, is worse than no completion file: it completes to nothing, or offers
a flag that errors.  These tests derive the option list from the parser itself rather than
from a copy of it, so adding an option without documenting it fails here first.
"""

import argparse
import re
import subprocess
from pathlib import Path

import pytest

from gputop.cli import PROG, build_parser

REPO = Path(__file__).resolve().parent.parent
MAN = REPO / "packaging" / "man" / "gputop.1"
COMPLETIONS = REPO / "packaging" / "completions"
PKGBUILD = REPO / "packaging" / "arch" / "PKGBUILD"

#: Options that take a value, per shell completion file, so a missing one is a failure
#: rather than an omission nobody notices until they press TAB.
COMPLETION_FILES = {
    "bash": COMPLETIONS / "gputop.bash",
    "zsh": COMPLETIONS / "gputop.zsh",
    "fish": COMPLETIONS / "gputop.fish",
}


def long_actions() -> tuple[argparse.Action, ...]:
    """Actions the parser registers for an option, excluding ``--help``.

    Read from the parser rather than from a copy of it: a duplicated list is exactly the
    thing that stops being true.
    """
    parser = build_parser()
    return tuple(
        action
        for action in parser._actions
        if any(name.startswith("--") for name in action.option_strings)
        and "--help" not in action.option_strings
    )


#: Textual spells a few keys by name; the man page prints what the user actually presses.
PRINTABLE_KEYS = {"slash": "/", "question_mark": "?", "space": " ", "plus": "+", "minus": "-"}


def long_options() -> tuple[str, ...]:
    """Every long option the parser registers, sorted.

    Both spellings of a short/long pair are collected: ``-c`` has a long form, and a
    completion file is wrong to omit ``--config`` because the parser lists ``-c`` first.
    """
    options: set[str] = set()
    for action in long_actions():
        options.update(name for name in action.option_strings if name.startswith("--"))
    return tuple(sorted(options))


def man_text() -> str:
    """The man page, with roff escaping undone so the options can be matched as written."""
    text = MAN.read_text()
    return text.replace("\\-", "-")


class TestPackagedFilesExist:
    """Everything the PKGBUILD and the README promise is actually in the tree."""

    @pytest.mark.parametrize("path", [*COMPLETION_FILES.values(), MAN, PKGBUILD])
    def test_present_and_not_empty(self, path: Path) -> None:
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 0, f"{path} is empty"


class TestManPage:
    """The man page, checked against the parser."""

    @pytest.mark.parametrize("option", long_options())
    def test_every_option_is_documented(self, option: str) -> None:
        assert option in man_text(), f"{option} is not in the man page"

    @pytest.mark.parametrize(
        "option",
        ["--check", "--json", "--dump", "--devices", "--blocks", "--no-blocks", "--log"],
    )
    def test_the_ones_that_shape_a_script_are_documented(self, option: str) -> None:
        assert option in man_text()

    def test_it_documents_the_exit_status(self) -> None:
        text = man_text()
        assert "EXIT STATUS" in text
        # 3 is the one scripts branch on, so it has to be spelled out.
        assert "3" in text.split("EXIT STATUS")[1]

    def test_it_documents_both_configuration_locations_and_env_vars(self) -> None:
        text = man_text()
        assert "GPUTOP_CONFIG" in text
        assert "GPUTOP_STATE" in text
        assert ".gputoprc" in text

    def test_it_names_every_binding_the_app_declares(self) -> None:
        """A key that is not documented is a key nobody finds.

        The bindings list is split on commas because Textual allows one action to answer to
        several keys ("h,question_mark"), and a man page that documented only the first
        would be silently incomplete.
        """
        from gputop.ui.app import GpuTopApp

        text = man_text()
        keys: set[str] = set()
        for binding in GpuTopApp.BINDINGS:
            first = binding[0] if isinstance(binding, tuple) else binding.key
            for key in str(first).split(","):
                key = PRINTABLE_KEYS.get(key.strip(), key.strip())
                if key:
                    keys.add(key)
        # The nine GPU bindings are a range on screen ("1"-"9"), not nine separate claims,
        # and a page that lists them individually would be longer without being clearer.
        if all(str(digit) in keys for digit in range(1, 10)):
            assert re.search(r"\b1\b.{0,40}\b9\b", text, re.DOTALL), (
                "the GPU number keys are missing"
            )
            keys -= {str(digit) for digit in range(1, 10)}
        for key in sorted(keys):
            assert key in text, f"key {key} is not in the man page"

    def test_it_renders_without_warnings(self) -> None:
        """A man page that emits groff diagnostics on every `man gputop` is not finished."""
        result = subprocess.run(
            ["man", "--warnings=all", "-l", str(MAN)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and not result.stdout:
            pytest.skip("man is not installed")
        assert "warning" not in result.stderr.lower(), result.stderr


class TestCompletions:
    """Each completion file must know about each option.

    fish spells an option ``-l name`` rather than ``--name``, so the search string is
    chosen per shell rather than assuming one convention for all three.
    """

    @staticmethod
    def mentions(text: str, option: str, shell: str) -> bool:
        """Whether a completion file offers this option, in that shell's spelling."""
        return f"-l {option[2:]}" in text if shell == "fish" else option in text

    @pytest.mark.parametrize("shell,path", COMPLETION_FILES.items(), ids=list(COMPLETION_FILES))
    def test_every_option_appears(self, shell: str, path: Path) -> None:
        text = path.read_text()
        missing = [
            option for option in long_options() if not self.mentions(text, option, shell)
        ]
        assert not missing, f"{shell} completion is missing {missing}"

    @pytest.mark.parametrize("shell,path", COMPLETION_FILES.items(), ids=list(COMPLETION_FILES))
    def test_no_invented_options(self, shell: str, path: Path) -> None:
        """A completion offering a flag the program rejects is worse than none.

        ``--help`` is exempt: argparse adds it, and a completion without it would be the
        odd one out.
        """
        text = path.read_text()
        pattern = r"-l\s+([a-z][a-z-]+)" if shell == "fish" else r"(--[a-z][a-z-]+)"
        offered = {name.lstrip("-") for name in re.findall(pattern, text)}
        known = {option.lstrip("-") for option in long_options()} | {"help"}
        invented = offered - known
        assert not invented, f"{shell} completion offers {sorted(invented)}"

    def test_the_themes_offered_are_the_themes_that_exist(self) -> None:
        from gputop.ui.theme import theme_names

        text = COMPLETION_FILES["bash"].read_text() + COMPLETION_FILES["zsh"].read_text()
        for theme in theme_names():
            assert theme in text, f"{theme} is not offered"

    def test_bash_completion_is_syntactically_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(COMPLETION_FILES["bash"])],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_the_shells_are_named_correctly(self) -> None:
        """`complete -F _gputop gputop` binds to the program name; the rest must match."""
        assert PROG in COMPLETION_FILES["bash"].read_text()
        assert COMPLETION_FILES["zsh"].read_text().startswith("#compdef gputop")
        assert "complete -c gputop" in COMPLETION_FILES["fish"].read_text()


class TestPkgbuild:
    """The Arch recipe, checked against the project's own metadata."""

    @pytest.mark.parametrize(
        "needle",
        [
            "pkgname=gputop",
            "python>=3.14",
            "uv build",
            "gputop.1",
            "radeontop",
        ],
    )
    def test_it_declares_what_it_must(self, needle: str) -> None:
        assert needle in PKGBUILD.read_text()

    def test_the_version_matches_pyproject(self) -> None:
        import tomllib

        with (REPO / "pyproject.toml").open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
        assert f"pkgver={version}" in PKGBUILD.read_text()


class TestReadme:
    """The README documents the same flags and the same requirements."""

    def test_every_option_appears(self) -> None:
        text = (REPO / "README.md").read_text()
        missing = [option for option in long_options() if option not in text]
        assert not missing, f"README is missing {missing}"

    def test_the_requirements_are_stated(self) -> None:
        text = (REPO / "README.md").read_text()
        assert "3.14" in text
        assert "5.19" in text
        assert "amdgpu" in text.lower() or "CONFIG_DRM_AMDGPU" in text
