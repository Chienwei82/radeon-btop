"""A fixed-capacity ring buffer used for graph history.

The buffer is owned exclusively by the sampler thread.  Readers never mutate it, which
keeps the sampler free of locking and therefore free-threading friendly.  Values are
frozen dataclasses (numbers), so publishing a shallow copy of the tuples is safe.
"""

from collections import deque
from collections.abc import Iterator

DEFAULT_HISTORY_LENGTH = 300


class RingBuffer[T]:
    """A bounded FIFO of at most ``capacity`` items.

    Args:
        capacity: Maximum number of retained items.  Values below one are coerced to one.
    """

    __slots__ = ("_capacity", "_items")

    def __init__(self, capacity: int = DEFAULT_HISTORY_LENGTH) -> None:
        self._capacity = max(1, capacity)
        self._items: deque[T] = deque(maxlen=self._capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of items retained."""
        return self._capacity

    def append(self, item: T) -> None:
        """Append an item, discarding the oldest one when full."""
        self._items.append(item)

    def items(self) -> tuple[T, ...]:
        """Return the retained items oldest-first as an immutable tuple."""
        return tuple(self._items)

    def last(self) -> T | None:
        """Return the most recent item, or ``None`` when empty."""
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(tuple(self._items))

    def clear(self) -> None:
        """Drop all retained items."""
        self._items.clear()
