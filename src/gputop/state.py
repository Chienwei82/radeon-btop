"""Reading and writing the session state in ``~/.config/gputop/config.toml``.

The interface remembers what the user was looking at: the theme, the sort column and
its direction, the filter text, the sampling interval, which GPU was focused.  That is
state, not configuration -- it is the answer to "how did I have this last time", not to
"how do I want it" -- and it is stored in the same file, under ``[state]``, because a
second file is one more thing to discover and one more thing to keep in sync.

Reading is :mod:`tomllib`, the stdlib reader already used for the rest of the config.
Writing needs the reverse, and the stdlib has no writer, so :mod:`tomli_w` does it.

Two properties matter more than the format:

* **The rest of the file is preserved.**  Writing state must not clobber a hand-edited
  ``[gpu.names]`` table or drop a comment-worthy file to a bare ``[state]``, so the
  document is read, one key replaced, and written back.
* **The write is atomic.**  A crash or a full disk in the middle of it must not leave a
  truncated config behind -- the file being rewritten is the file the next start reads.
  The new content goes to a temporary file in the same directory and is renamed over the
  old one, which POSIX makes atomic.
"""

import os
import tempfile
import tomllib
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import tomli_w

from gputop.config import StateConfig, build_section, state_path

#: The section this module owns.  Nothing else in the file is ever written.
STATE_SECTION = "state"

#: Key order in the written section.  Explicit rather than dataclass order so the file
#: reads the way the interface does: what it looks like, then what it shows.
KEY_ORDER: tuple[str, ...] = (
    "theme",
    "interval_ms",
    "process_sort",
    "process_sort_reverse",
    "process_filter",
    "show_idle_processes",
    "show_process_tree",
    "full_command",
    "overview",
    "gpu_index",
)

#: ``StateConfig`` field -> key in the file.  Two names rather than one because the
#: section is read by hand as often as it is written, and ``show_idle_processes`` says
#: what it means inside a file that also contains ``[process]``.
_KEYS: dict[str, str] = {
    "theme": "theme",
    "interval_ms": "interval_ms",
    "sort": "process_sort",
    "sort_reverse": "process_sort_reverse",
    "filter": "process_filter",
    "show_idle": "show_idle_processes",
    "process_tree": "show_process_tree",
    "full_command": "full_command",
    "overview": "overview",
    "gpu_index": "gpu_index",
}

#: The same mapping in reverse, so a file written by any build is read by this one.
#: Getting this wrong is silent: a round trip that quietly dropped the sort column would
#: look exactly like gputop forgetting the setting.
_FIELDS: dict[str, str] = {key: field for field, key in _KEYS.items()}


def state_table(state: StateConfig) -> dict[str, Any]:
    """Render a :class:`StateConfig` as a TOML table.

    ``None`` means "not recorded" and is dropped rather than written: TOML has no null,
    and emitting something like an empty string in its place would turn "the user never
    chose a theme" into "the user chose no theme" on the next read.
    """
    table: dict[str, Any] = {}
    for field, key in _KEYS.items():
        value = getattr(state, field)
        if value is not None:
            table[key] = value
    return {key: table[key] for key in KEY_ORDER if key in table}


def table_to_fields(table: Mapping[str, Any]) -> dict[str, Any]:
    """Map a file's ``[state]`` table back onto :class:`StateConfig` field names.

    Keys this build does not know are dropped here; :func:`gputop.config.build_section`
    then reports anything it recognises but cannot use.
    """
    return {field: value for key, value in table.items() if (field := _FIELDS.get(key))}


def _read(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read a TOML document, reporting *why* it could not be read.

    Split out from :func:`read_document` because a reader and a writer need opposite
    treatment of the two failure modes.  "There is no file" is normal -- it is a first
    run.  "There is a file and I could not understand it" is somebody's hand-written
    configuration, and the caller's response has to be different: a reader falls back to
    defaults, and a writer must decline to write.
    """
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError:
        return {}, None
    except tomllib.TOMLDecodeError as exc:
        return {}, (
            f"{path}: state not saved because the file is not valid TOML ({exc}); "
            "gputop read it as empty and will save again once it parses"
        )
    except OSError as exc:
        return {}, f"{path}: state could not be read ({exc.strerror or exc})"
    return dict(document), None


def read_document(path: Path) -> dict[str, Any]:
    """Read a TOML document, returning ``{}`` when it is absent or unusable.

    State is never worth refusing to start over, so every failure here is a silent empty
    document and the caller decides what that means.  A *writer* must not use this: an
    empty document is indistinguishable from an unreadable one, and merging into it would
    replace the user's configuration with the one key being saved.  Use :func:`_read`.
    """
    document, _ = _read(path)
    return document


def save_state(state: StateConfig, path: Path | None = None) -> str | None:
    """Write ``state`` into the file's ``[state]`` section.

    Args:
        state: The session state to remember.
        path: The file to write.  Defaults to :func:`gputop.config.state_path`.

    Returns:
        ``None`` on success, or a message describing why the state could not be saved.
        A monitor that has been running for hours is not the place to raise: the data is
        still on screen, and the caller turns the message into a footer warning.
    """
    target = state_path(path)
    document, problem = _read(target)
    if problem is not None:
        # Declining is the only safe answer.  The document could not be parsed, so the
        # other sections could not be carried across, and writing anyway would replace a
        # hand-edited config -- every ``[general]`` and ``[gpu.names]`` key in it -- with a
        # bare ``[state]``.  That happened silently, because this function reported
        # success: a single trailing comma anywhere in the file was enough to trigger it.
        return problem
    document[STATE_SECTION] = state_table(state)
    try:
        payload = tomli_w.dumps(document)
    except (TypeError, ValueError) as exc:  # pragma: no cover - only a broken value
        return f"{target}: state could not be encoded ({exc})"
    return _atomic_write(target, payload)


def _atomic_write(target: Path, payload: str) -> str | None:
    """Replace ``target`` with ``payload`` without ever leaving it half-written."""
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Keep the permissions the file already had: a config may have been tightened
        # deliberately, and a rename must not loosen it.  A file that does not exist yet
        # is created private, because it is about to hold the user's own settings.
        try:
            mode = target.stat().st_mode & 0o777
        except OSError:
            mode = 0o600
        temporary.chmod(mode)
        temporary.replace(target)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return f"{target}: state could not be saved ({exc})"
    # The rename is only durable once the *directory* is flushed too.  Without this the
    # data is on disk and the name that points at it is not, so a power cut can leave the
    # file absent or holding the previous version -- the one outcome the fsync above is
    # there to prevent, and one that no amount of correctness in the write path prevents.
    # Best effort: a filesystem that cannot fsync a directory (some network mounts) is not
    # a reason to report a failure for a write that plainly succeeded.
    with suppress(OSError):
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return None


def load_state(path: Path | None = None) -> StateConfig:
    """Read the ``[state]`` section, or defaults when there is none.

    Validation is deliberately *not* repeated here.  :func:`gputop.config.load_config`
    already rejects a value this build does not understand and reports it as a warning,
    and doing it twice would mean two places to keep in step.
    """
    target = state_path(path)
    section = read_document(target).get(STATE_SECTION)
    if not isinstance(section, dict):
        return StateConfig()
    return cast(
        StateConfig, build_section(StateConfig, table_to_fields(section), STATE_SECTION, [])
    )


__all__ = [
    "KEY_ORDER",
    "STATE_SECTION",
    "load_state",
    "read_document",
    "save_state",
    "state_table",
]
