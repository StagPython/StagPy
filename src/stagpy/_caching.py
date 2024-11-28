from __future__ import annotations

import typing
from collections import deque
from dataclasses import dataclass
from functools import cached_property

if typing.TYPE_CHECKING:
    from .datatypes import Field


@dataclass(frozen=True)
class FieldCache:
    """FIFO cache of [Field][]s.

    If `maxsize` is None, entries are never evicted from the cache.
    """

    maxsize: int | None

    @cached_property
    def _stack(self) -> deque[tuple[int, str]]:
        return deque()

    @cached_property
    def _data(self) -> dict[tuple[int, str], Field]:
        return {}

    def _prune(self) -> None:
        if self.maxsize is None:
            return
        while len(self._stack) > self.maxsize:
            elt = self._stack.popleft()
            del self._data[elt]

    def resize(self, new_size: int | None) -> None:
        object.__setattr__(self, "maxsize", new_size)
        self._prune()

    def insert(self, istep: int, name: str, field: Field) -> None:
        key = (istep, name)
        if key not in self._data:
            self._stack.append(key)
        self._data[key] = field
        self._prune()

    def get(self, istep: int, name: str) -> Field | None:
        return self._data.get((istep, name))

    def evict_istep(self, istep: int) -> None:
        to_keep = []
        for key in self._stack:
            if key[0] == istep:
                del self._data[key]
            else:
                to_keep.append(key)
        self._stack.clear()
        self._stack.extend(to_keep)
        assert len(self._stack) == len(self._data)
