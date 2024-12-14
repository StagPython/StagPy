from __future__ import annotations

import re
import typing
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from functools import cached_property

from . import phyvars, stagyyparsers
from .error import InvalidSnapshotError

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from .datatypes import Field
    from .stagyydata import StagyyData


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


class StepSnap(ABC):
    """Keep track of the step/snap correspondence."""

    @abstractmethod
    def istep(self, *, isnap: int) -> int | None: ...

    @abstractmethod
    def isnap(self, *, istep: int) -> int | None: ...

    @abstractmethod
    def len_snap(self) -> int: ...


@dataclass(frozen=True)
class StepSnapInfo:
    step_to_snap: Mapping[int, int]
    snap_to_step: Mapping[int, int]
    isnap_max: int


@dataclass(frozen=True)
class StepSnapH5(StepSnap):
    sdat: StagyyData

    @cached_property
    def _info(self) -> StepSnapInfo:
        assert self.sdat.hdf5 is not None
        isnap = -1
        step_to_snap = {}
        snap_to_step = {}
        for isnap, istep in stagyyparsers.read_time_h5(self.sdat.hdf5):
            step_to_snap[istep] = isnap
            snap_to_step[isnap] = istep
        return StepSnapInfo(
            step_to_snap=step_to_snap,
            snap_to_step=snap_to_step,
            isnap_max=isnap,
        )

    def istep(self, *, isnap: int) -> int | None:
        return self._info.snap_to_step.get(isnap)

    def isnap(self, *, istep: int) -> int | None:
        return self._info.step_to_snap.get(istep)

    def len_snap(self) -> int:
        return self._info.isnap_max + 1


@dataclass(frozen=True)
class StepSnapLegacy(StepSnap):
    sdat: StagyyData

    @cached_property
    def _step_to_snap(self) -> dict[int, int | None]:
        return {}

    @cached_property
    def _snap_to_step(self) -> dict[int, int | None]:
        return {}

    @cached_property
    def isnap_max(self) -> int:
        imax = -1
        out_stem = re.escape(self.sdat.par.legacy_output("_").name[:-1])
        rgx = re.compile(f"^{out_stem}_([a-zA-Z]+)([0-9]{{5}})$")
        fstems = set(fstem for fstem in phyvars.FIELD_FILES)
        for fname in self.sdat._files:
            match = rgx.match(fname.name)
            if match is not None and match.group(1) in fstems:
                imax = max(int(match.group(2)), imax)
        return imax

    def len_snap(self) -> int:
        return self.isnap_max + 1

    def istep(self, *, isnap: int) -> int | None:
        if isnap < 0 or isnap > self.isnap_max:
            return None
        istep = self._snap_to_step.get(isnap, -1)
        if istep == -1:
            binfiles = self.sdat._binfiles_set(isnap)
            if binfiles:
                istep = stagyyparsers.field_istep(binfiles.pop())
            else:
                istep = None
            self._snap_to_step[isnap] = istep
            if istep is not None:
                self._step_to_snap[istep] = isnap
        return istep

    def isnap(self, *, istep: int) -> int | None:
        if istep < 0:
            return None
        isnap = self._step_to_snap.get(istep, -1)
        if isnap == -1:
            istep_try = None
            # might be more efficient to do 0 and -1 then bisection, even if
            # that means losing intermediate information
            while (istep_try is None or istep_try < istep) and isnap < 99999:
                isnap += 1
                try:
                    istep_try = self.sdat.snaps[isnap].istep
                except InvalidSnapshotError:
                    pass
                # all intermediate istep could have their isnap to None
                self._snap_to_step[isnap] = istep_try
                if istep_try is not None:
                    self._step_to_snap[istep_try] = isnap

            if istep_try != istep:
                self._step_to_snap[istep] = None

        return self._step_to_snap[istep]
