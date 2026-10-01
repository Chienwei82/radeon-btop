"""Canaries for the interpreter the suite runs on.

Nothing here tests gputop.  These are tripwires: properties of CPython that the code in
this repository *relies on being true*, asserted so that a toolchain change shows up as
a clear failure here rather than as a baffling one somewhere else.

The tuple case was added after this interpreter was observed flattening a single-element
tuple into its element -- ``engines=(engine(...),)`` arriving as ``engines=engine(...)``
-- in a call inside a list display, while neighbouring and apparently identical calls
compiled correctly.  It is silent, it is data-dependent, and it would turn a one-engine
client into something the model cannot even iterate.  Fixtures now build such tuples
through a function call, which cannot be flattened; if a future interpreter restores the
correct behaviour, or breaks more shapes than this one, the assertions below say so.
"""

import sys


def takes(pid: int, *, engines: tuple[int, ...] = ()) -> tuple[int, ...]:
    """A callee whose ``engines`` argument is only usable as a tuple."""
    return (pid, engines)


def one(value: int) -> int:
    """A factory for a single tuple element."""
    return value


def two(a: int, b: int) -> int:
    """A factory taking two arguments, which is where the flattening appeared."""
    return a + b


def _tuple_of(value: int) -> tuple[int, ...]:
    """Wrap a value in a tuple from inside a function."""
    return (value,)


class TestInterpreter:
    def test_python_version(self) -> None:
        assert sys.version_info >= (3, 14), "gputop targets 3.14 and uses PEP 649 annotations"

    def test_a_one_element_tuple_of_a_call_survives_a_keyword_argument(self) -> None:
        rows = [
            takes(1, engines=(one(1),)),
            takes(2, engines=(two(1, 2),)),
        ]
        assert [row[1] for row in rows] == [(1,), (3,)]

    def test_a_one_element_tuple_of_a_name_survives(self) -> None:
        usage = two(1, 2)
        assert takes(1, engines=(usage,))[1] == (3,)

    def test_a_two_element_tuple_of_calls_survives(self) -> None:
        assert takes(1, engines=(one(1), two(2, 3)))[1] == (1, 5)

    def test_an_unpacked_collection_survives(self) -> None:
        assert takes(1, engines=(*[two(1, 2)],))[1] == (3,)

    def test_a_tuple_built_by_a_function_survives(self) -> None:
        assert takes(1, engines=_tuple_of(two(1, 2)))[1] == (3,)
