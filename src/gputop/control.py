"""Sending a signal to a process, and the rules about when not to.

This is the only part of gputop that can end something other than itself, so the
restrictions live in one place, in a module with no Textual in it, and are pure: deciding
whether a signal is allowed is a function of the target and can be tested exhaustively
without ever sending anything.

The default is to refuse.  A monitor that can kill a process is one keystroke away from
terminating the wrong one, and the cost of that mistake is measured in someone's
unsaved work, not in a rendering glitch.  ``process.allow_kill`` in the config is what
turns the capability on at all, and even then every target is checked.
"""

import os
import signal
from typing import Final

from gputop.model.process import GpuProcess

#: The signals offered, in the order the confirmation lists them.  ``SIGTERM`` first: a
#: process asked politely gets to clean up, and the dialog says so.
SIGNALS: Final[tuple[tuple[str, int], ...]] = (
    ("SIGTERM", signal.SIGTERM),
    ("SIGKILL", signal.SIGKILL),
)

#: What each signal means, shown in the confirmation dialog.  A user who does not know
#: what ``SIGKILL`` is cannot meaningfully consent to it.
SIGNAL_HELP: Final[dict[str, str]] = {
    "SIGTERM": "ask the process to exit",
    "SIGKILL": "terminate immediately; no cleanup, no chance to save",
}


def signal_number(name: str) -> int:
    """Look up a signal number by name, defaulting to ``SIGTERM``.

    An unknown name falls back to the *reversible* signal rather than the other way
    round: if anything about the request is not understood, the answer should be the
    option that can still be undone.
    """
    for known, number in SIGNALS:
        if known == name:
            return number
    return signal.SIGTERM


def refusal(process: GpuProcess | None, *, own_pid: int | None = None) -> str | None:
    """Why this process must not be signalled, or ``None`` when it may be.

    The checks are ordered cheapest first and, more importantly, from "can this even be
    signalled" to "should it be".  A kernel thread has no userspace task to signal; PID 1
    is the machine's init, and killing it is not a thing a monitor should be able to do
    even by accident; and gputop signalling itself would exit without saving the state
    the user just configured.
    """
    if process is None:
        return "no process is selected"
    if process.pid <= 0:
        return f"pid {process.pid} is not a signalable target"
    if process.pid == 1:
        return "refusing to signal init (pid 1)"
    if process.is_kernel_thread:
        return f"{process.pid} is a kernel thread and has no signalable task"
    if own_pid is not None and process.pid == own_pid:
        return "refusing to signal gputop itself"
    return None


def terminate(
    process: GpuProcess | None, signal_name: str, *, own_pid: int | None = None
) -> str | None:
    """Send ``signal_name`` to ``process``.

    Args:
        process: The target; taken from the process table, never from user input.
        signal_name: ``"SIGTERM"`` or ``"SIGKILL"``.
        own_pid: gputop's own pid, used by :func:`refusal`.

    Returns:
        ``None`` on success, otherwise a message explaining the refusal or the failure.

    The one syscall here cannot block -- the kernel delivers a signal and returns -- so
    it runs on the UI thread rather than in a worker.  Handing it to a thread would add a
    message round trip and a failure mode (the worker dying silently) to avoid waiting for
    nothing.
    """
    reason = refusal(process, own_pid=own_pid)
    if reason is not None:
        return reason
    assert process is not None  # guaranteed by ``refusal``
    try:
        os.kill(process.pid, signal_number(signal_name))
    except ProcessLookupError:
        return f"{process.pid} ({process.name}) exited before the signal was sent"
    except PermissionError:
        return f"not permitted to signal {process.pid} ({process.name}); try again as root"
    except OSError as exc:
        return f"{process.pid} ({process.name}): {exc.strerror or exc}"
    return None


__all__ = ["SIGNALS", "SIGNAL_HELP", "refusal", "signal_number", "terminate"]
