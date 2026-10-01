"""The rules about which process may be signalled, and what happens when it is.

:mod:`gputop.control` is the only part of gputop that can end something other than
itself, so the refusals are tested exhaustively and the delivery path is tested through
a spy rather than against the real ``/proc``.

Every test that reaches :func:`terminate` sends a real signal to a real pid, so they
either target this process (harmless: the signal is intercepted) or a pid that is
guaranteed not to exist.
"""

import os
import signal

import pytest

from gputop import control
from gputop.model.process import EngineUsage, GpuProcess

NS = 1_000_000_000
MiB = 1024**2


def proc(
    pid: int,
    *,
    name: str = "proc",
    cmdline: tuple[str, ...] | None = None,
    vram: int | None = 8 * MiB,
) -> GpuProcess:
    """Build a client.  ``cmdline=()`` makes it look like a kernel thread."""
    return GpuProcess(
        pid=pid,
        name=name,
        user="david",
        bdf="0000:0c:00.0",
        client_id=1,
        cmdline=("gputop-test",) if cmdline is None else cmdline,
        engines=(EngineUsage(engine="gfx", total_ns=10**8, delta_ns=10**8, window_ns=NS),),
        vram_used=vram,
    )


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record every signal instead of sending it."""
    calls: list[tuple[int, int]] = []

    def record(pid: int, number: int) -> None:
        calls.append((pid, number))

    monkeypatch.setattr(control.os, "kill", record)
    return calls


class TestSignals:
    """The two offered signals, and the fallback."""

    def test_both_are_offered(self) -> None:
        assert [name for name, _ in control.SIGNALS] == ["SIGTERM", "SIGKILL"]

    def test_sigterm_comes_first(self) -> None:
        """A process asked politely gets to clean up; the dialog says so."""
        assert control.SIGNALS[0][0] == "SIGTERM"

    def test_each_is_described(self) -> None:
        for name, _ in control.SIGNALS:
            assert control.SIGNAL_HELP[name]

    def test_lookup_returns_the_real_number(self) -> None:
        assert control.signal_number("SIGKILL") == signal.SIGKILL
        assert control.signal_number("SIGTERM") == signal.SIGTERM

    def test_an_unknown_name_falls_back_to_the_reversible_one(self) -> None:
        """If the request is not understood, the answer should be the undoable one."""
        assert control.signal_number("SIGNOPE") == signal.SIGTERM


class TestRefusal:
    """Who may not be signalled.  All of this is pure and sends nothing."""

    def test_an_ordinary_process_is_allowed(self) -> None:
        assert control.refusal(proc(4242), own_pid=1) is None

    def test_nothing_selected_is_refused(self) -> None:
        assert control.refusal(None) == "no process is selected"

    def test_init_is_refused(self) -> None:
        assert "init" in (control.refusal(proc(1)) or "")

    def test_a_zero_or_negative_pid_is_refused(self) -> None:
        assert control.refusal(proc(0)) is not None
        assert control.refusal(proc(-5)) is not None

    def test_a_kernel_thread_is_refused(self) -> None:
        """It has no userspace task, so there is nothing to signal."""
        assert "kernel thread" in (control.refusal(proc(9, cmdline=())) or "")

    def test_gputop_itself_is_refused(self) -> None:
        assert "itself" in (control.refusal(proc(4242), own_pid=4242) or "")

    def test_the_kernel_thread_check_comes_before_our_own(self) -> None:
        """A worker with no argv that also happens to be us is still not signalable."""
        reason = control.refusal(proc(4242, cmdline=()), own_pid=4242)
        assert reason is not None and "kernel thread" in reason


class TestTerminate:
    """Delivery, and what is reported when it does not happen."""

    def test_a_confirmed_signal_is_sent(self, sent: list[tuple[int, int]]) -> None:
        assert control.terminate(proc(4242), "SIGKILL", own_pid=1) is None
        assert sent == [(4242, signal.SIGKILL)]

    def test_sigterm_is_delivered_as_sigterm(self, sent: list[tuple[int, int]]) -> None:
        control.terminate(proc(4242), "SIGTERM", own_pid=1)
        assert sent == [(4242, signal.SIGTERM)]

    def test_a_refusal_sends_nothing(self, sent: list[tuple[int, int]]) -> None:
        reason = control.terminate(proc(1), "SIGKILL")
        assert reason is not None
        assert sent == []

    def test_the_own_pid_check_sends_nothing(self, sent: list[tuple[int, int]]) -> None:
        assert control.terminate(proc(os.getpid()), "SIGKILL", own_pid=os.getpid()) is not None
        assert sent == []

    def test_a_vanished_process_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def gone(_pid: int, _number: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(control.os, "kill", gone)
        reason = control.terminate(proc(4242, name="firefox"), "SIGTERM", own_pid=1)
        assert reason is not None
        assert "4242" in reason
        assert "firefox" in reason

    def test_running_as_another_user_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def denied(_pid: int, _number: int) -> None:
            raise PermissionError

        monkeypatch.setattr(control.os, "kill", denied)
        reason = control.terminate(proc(4242), "SIGTERM", own_pid=1)
        assert reason is not None
        assert "not permitted" in reason

    def test_anything_else_is_reported_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def other(_pid: int, _number: int) -> None:
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr(control.os, "kill", other)
        reason = control.terminate(proc(4242), "SIGTERM", own_pid=1)
        assert reason == "4242 (proc): Invalid argument"
