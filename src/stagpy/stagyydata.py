"""Define high level structure StagyyData and helper classes.

Note:
    The helper classes are not designed to be instantiated on their own, but
    only as attributes of `StagyyData` instances. Users of this module should
    only instantiate [`StagyyData`][stagpy.stagyydata.StagyyData].
"""

from __future__ import annotations

import re
import typing
from collections import abc
from dataclasses import dataclass, field
from functools import cached_property
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import _helpers, error, phyvars, stagyyparsers, step
from . import datatypes as dt
from ._caching import FieldCache
from .parfile import StagyyPar
from .stagyyparsers import FieldXmf, TracersXmf
from .step import Step

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Any, Callable, Iterable, Iterator, Sequence, TypeAlias

    from numpy.typing import NDArray
    from pandas import DataFrame, Series

    from .config import Core

    StepIndex: TypeAlias = int | slice


@typing.overload
def _as_view_item(obj: Sequence[StepIndex]) -> Sequence[StepIndex]: ...


@typing.overload
def _as_view_item(obj: slice) -> Sequence[slice]: ...


@typing.overload
def _as_view_item(obj: int) -> None: ...


def _as_view_item(
    obj: Sequence[StepIndex] | slice | int,
) -> Sequence[StepIndex] | Sequence[slice] | None:
    """Return None or a suitable iterable to build a StepsView."""
    try:
        iter(obj)  # type: ignore
        return obj  # type: ignore
    except TypeError:
        pass
    if isinstance(obj, slice):
        return (obj,)
    return None


class Refstate:
    """Reference state profiles.

    The `StagyyData.refstate` attribute is an instance of this class.
    Reference state profiles are accessed through the attributes of this
    object.
    """

    def __init__(self, sdat: StagyyData):
        self._sdat = sdat

    @cached_property
    def _data(self) -> tuple[list[list[DataFrame]], list[DataFrame]]:
        """Read reference state profile."""
        reffile = self._sdat.filename("refstat.dat")
        if self._sdat.hdf5 and not reffile.is_file():
            # check legacy folder as well
            reffile = self._sdat.filename("refstat.dat", force_legacy=True)
        data = stagyyparsers.refstate(reffile)
        if data is None:
            raise error.NoRefstateError(self._sdat)
        return data

    @property
    def systems(self) -> list[list[DataFrame]]:
        """Reference state profiles of phases.

        It is a list of list of :class:`pandas.DataFrame` containing
        the reference profiles associated with each phase in each system.

        Example:
            The temperature profile of the 3rd phase in the 1st
            system is

            ```py
            sdat.refstate.systems[0][2]["T"]
            ```
        """
        return self._data[0]

    @property
    def adiabats(self) -> list[DataFrame]:
        """Adiabatic reference state profiles.

        It is a list of :class:`pandas.DataFrame` containing the reference
        profiles associated with each system.  The last item is the combined
        adiabat.

        Example:
            The adiabatic temperature profile of the 2nd system is

            ```py
            sdat.refstate.adiabats[1]["T"]
            ```

            The combined density profile is

            ```py
            sdat.refstate.adiabats[-1]["rho"]
            ```
        """
        return self._data[1]


class Tseries:
    """Time series.

    The `StagyyData.tseries` attribute is an instance of this class.

    `Tseries` implements the getitem mechanism.  Keys are series names
    defined in `stagpy.phyvars.TIME[_EXTRA]`.  Items are
    [stagpy.datatypes.Tseries][] instances.
    """

    def __init__(self, sdat: StagyyData):
        self.sdat = sdat
        self._cached_extra: dict[str, dt.Tseries] = {}

    @cached_property
    def _data(self) -> DataFrame | None:
        timefile = self.sdat.filename("TimeSeries.h5")
        data = stagyyparsers.time_series_h5(timefile, list(phyvars.TIME.keys()))
        if data is not None:
            return data
        timefile = self.sdat.filename("time.dat")
        if self.sdat.hdf5 and not timefile.is_file():
            # check legacy folder as well
            timefile = self.sdat.filename("time.dat", force_legacy=True)
        data = stagyyparsers.time_series(timefile, list(phyvars.TIME.keys()))
        return data

    @property
    def _tseries(self) -> DataFrame:
        if self._data is None:
            raise error.MissingDataError(f"No tseries data in {self.sdat}")
        return self._data

    def __getitem__(self, name: str) -> dt.Tseries:
        if name in self._tseries.columns:
            series = self._tseries[name].to_numpy()
            time = self.time
            if name in phyvars.TIME:
                meta = phyvars.TIME[name]
            else:
                meta = dt.Vart(name, "", "1")
        elif name in self._cached_extra:
            tseries = self._cached_extra[name]
            series = tseries.values
            time = tseries.time
            meta = tseries.meta
        elif name in phyvars.TIME_EXTRA:
            self._cached_extra[name] = phyvars.TIME_EXTRA[name](self.sdat)
            tseries = self._cached_extra[name]
            series = tseries.values
            time = tseries.time
            meta = tseries.meta
        else:
            raise error.UnknownTimeVarError(name)
        return dt.Tseries(series, time, meta)

    def tslice(
        self, name: str, tstart: float | None = None, tend: float | None = None
    ) -> dt.Tseries:
        """Return a [`Tseries`][stagpy.datatypes.Tseries] between specified times.

        Args:
            name: time variable.
            tstart: starting time. Set to None to start at the beginning of
                available data.
            tend: ending time. Set to None to stop at the end of available
                data.
        """
        series = self[name]
        istart = 0
        iend = len(series.time)
        if tstart is not None:
            istart = _helpers.find_in_sorted_arr(tstart, series.time)
        if tend is not None:
            iend = _helpers.find_in_sorted_arr(tend, series.time, True) + 1
        return dt.Tseries(
            series.values[istart:iend],
            series.time[istart:iend],
            series.meta,
        )

    @property
    def time(self) -> NDArray:
        """Time vector."""
        return self._tseries["t"].to_numpy()

    @property
    def isteps(self) -> NDArray:
        """Step indices.

        This is such that `time[istep]` is at step `isteps[istep]`.
        """
        return self._tseries.index.values

    def at_step(self, istep: int) -> Series:
        """Time series output for a given step."""
        return self._tseries.loc[istep]  # type: ignore


class RprofsAveraged(step.Rprofs):
    """Radial profiles time-averaged over a [`StepsView`][stagpy.stagyydata.StepsView].

    The [`StepsView.rprofs_averaged`][stagpy.stagyydata.StepsView.rprofs_averaged]
    attribute is an instance of this class.

    It implements the same interface as [`Rprofs`][stagpy.step.Rprofs] but
    returns time-averaged profiles instead.
    """

    def __init__(self, steps: StepsView):
        self.steps = steps.filter(rprofs=True)
        self._cached_data: dict[str, dt.Rprof] = {}
        super().__init__(next(iter(self.steps)))

    def __getitem__(self, name: str) -> dt.Rprof:
        # the averaging method has two shortcomings:
        # - does not take into account time changing geometry;
        # - does not take into account time changing timestep.
        if name in self._cached_data:
            return self._cached_data[name]
        steps_iter = iter(self.steps)
        rpf = next(steps_iter).rprofs[name]
        rprof = np.copy(rpf.values)
        nprofs = 1
        for step_ in steps_iter:
            nprofs += 1
            rprof += step_.rprofs[name].values
        rprof /= nprofs
        self._cached_data[name] = dt.Rprof(rprof, rpf.rad, rpf.meta)
        return self._cached_data[name]

    @property
    def stepstr(self) -> str:
        """String representation of steps indices."""
        return self.steps.stepstr


class Steps:
    """Collections of time steps.

    The `StagyyData.steps` attribute is an instance of this class.
    Time steps (which are [`Step`][stagpy.step.Step] instances) can be
    accessed with the item accessor:

    ```py
    sdat = StagyyData(Path("path/to/run"))
    sdat.steps[istep]  # Step object of the istep-th time step
    ```

    Slices or tuple of istep and slices of `Steps` object are
    `StepsView` instances that you can iterate and filter:

    ```py
    for step in steps[500:]:
        # iterate through all time steps from the 500-th one
        do_something(step)

    for step in steps[-100:].filter(snap=True):
        # iterate through all snapshots present in the last 100 time steps
        do_something(step)

    for step in steps[0,3,5,-2:]:
        # iterate through steps 0, 3, 5 and the last two
        do_something(step)
    ```
    """

    def __init__(self, sdat: StagyyData):
        self.sdat = sdat

    def __repr__(self) -> str:
        return f"{self.sdat!r}.steps"

    @cached_property
    def _data(self) -> dict[int, Step]:
        return {}

    @typing.overload
    def __getitem__(self, istep: int) -> Step: ...

    @typing.overload
    def __getitem__(self, istep: slice | Sequence[StepIndex]) -> StepsView: ...

    def __getitem__(self, istep: int | slice | Sequence[StepIndex]) -> Step | StepsView:
        keys = _as_view_item(istep)
        if keys is not None:
            return StepsView(self, keys)
        try:
            istep = int(istep)  # type: ignore
        except ValueError:
            raise error.InvalidTimestepError(
                self.sdat,
                istep,  # type: ignore
                "Time step should be an integer value",
            )
        if istep < 0:
            istep += len(self)
            if istep < 0:
                istep -= len(self)
                raise error.InvalidTimestepError(
                    self.sdat, istep, f"Last istep is {len(self) - 1}"
                )
        if istep not in self._data:
            self._data[istep] = Step(istep, self.sdat)
        return self._data[istep]

    def __delitem__(self, istep: int | None) -> None:
        if istep is not None and istep in self._data:
            self.sdat._field_cache.evict_istep(istep)
            del self._data[istep]

    @cached_property
    def _len(self) -> int:
        return self.sdat.tseries.isteps[-1] + 1

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Step]:
        return iter(self[:])

    def at_time(self, time: float, after: bool = False) -> Step:
        """Return step corresponding to a given physical time.

        Args:
            time: the physical time requested.
            after: when False (the default), the returned step is such that its
                time is immediately before the requested physical time. When
                True, the returned step is the next one instead (if it exists,
                otherwise the same step is returned).

        Returns:
            the relevant step.
        """
        itime = _helpers.find_in_sorted_arr(time, self.sdat.tseries.time, after)
        return self[self.sdat.tseries.isteps[itime]]

    def filter(
        self,
        snap: bool = False,
        rprofs: bool = False,
        fields: Iterable[str] | None = None,
        func: Callable[[Step], bool] | None = None,
    ) -> StepsView:
        """Build a `StepsView` with requested filters."""
        return self[:].filter(snap, rprofs, fields, func)


class Snaps:
    """Collection of snapshots.

    The `StagyyData.snaps` attribute is an instance of this class.
    Snapshots (which are [`Step`][stagpy.step.Step] instances) can be accessed
    with the item accessor:

    ```py
    sdat = StagyyData(Path("path/to/run"))
    sdat.snaps[isnap]  # Step object of the isnap-th snapshot
    ```
    """

    def __init__(self, sdat: StagyyData):
        self.sdat = sdat
        self._isteps: dict[int, int | None] = {}
        self._all_isteps_known = False

    def __repr__(self) -> str:
        return f"{self.sdat!r}.snaps"

    @typing.overload
    def __getitem__(self, istep: int) -> Step: ...

    @typing.overload
    def __getitem__(self, istep: slice | Sequence[StepIndex]) -> StepsView: ...

    def __getitem__(self, isnap: Any) -> Step | StepsView:
        keys = _as_view_item(isnap)
        if keys is not None:
            return StepsView(self, keys).filter(snap=True)
        if isnap < 0:
            isnap += len(self)
        if isnap < 0 or isnap >= len(self):
            istep = None
        else:
            istep = self._isteps.get(isnap, None if self._all_isteps_known else -1)
        if istep == -1:
            # isnap not in _isteps but not all isteps known, keep looking
            binfiles = self.sdat._binfiles_set(isnap)
            if binfiles:
                istep = stagyyparsers.field_istep(binfiles.pop())
            else:
                istep = None
            if istep is not None:
                self._bind(isnap, istep)
            else:
                self._isteps[isnap] = None
        if istep is None:
            raise error.InvalidSnapshotError(self.sdat, isnap, "Invalid snapshot index")
        return self.sdat.steps[istep]

    def __delitem__(self, isnap: int | None) -> None:
        if isnap is not None:
            istep = self._isteps.get(isnap)
            del self.sdat.steps[istep]

    @cached_property
    def _len(self) -> int:
        length = -1
        if self.sdat.hdf5:
            isnap = -1
            for isnap, istep in stagyyparsers.read_time_h5(self.sdat.hdf5):
                self._bind(isnap, istep)
            length = isnap
            self._all_isteps_known = True
        if length < 0:
            out_stem = re.escape(self.sdat.par.legacy_output("_").name[:-1])
            rgx = re.compile(f"^{out_stem}_([a-zA-Z]+)([0-9]{{5}})$")
            fstems = set(fstem for fstem in phyvars.FIELD_FILES)
            for fname in self.sdat._files:
                match = rgx.match(fname.name)
                if match is not None and match.group(1) in fstems:
                    length = max(int(match.group(2)), length)
        if length < 0:
            raise error.NoSnapshotError(self.sdat)
        return length + 1

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Step]:
        return iter(self[:])

    def at_time(self, time: float, after: bool = False) -> Step:
        """Return snap corresponding to a given physical time.

        Args:
            time: the physical time requested.
            after: when False (the default), the returned snap is such that its
                time is immediately before the requested physical time. When
                True, the returned snap is the next one instead (if it exists,
                otherwise the same snap is returned).

        Returns:
            the relevant `Step`.
        """
        # in theory, this could be a valid implementation of Steps.at_time
        # but this isn't safe against missing data...
        igm = 0
        igp = len(self) - 1
        while igp - igm > 1:
            istart = igm + (igp - igm) // 2
            if self[istart].time >= time:
                igp = istart
            else:
                igm = istart
        if self[igp].time > time and not after and igp > 0:
            igp -= 1
        return self[igp]

    def _bind(self, isnap: int, istep: int) -> None:
        """Register the isnap / istep correspondence.

        Args:
            isnap: snapshot index.
            istep: time step index.
        """
        self._isteps[isnap] = istep
        self.sdat.steps[istep]._isnap = isnap

    def filter(
        self,
        snap: bool = False,
        rprofs: bool = False,
        fields: Iterable[str] | None = None,
        func: Callable[[Step], bool] | None = None,
    ) -> StepsView:
        """Build a `StepsView` with requested filters."""
        return self[:].filter(snap, rprofs, fields, func)


@dataclass(frozen=True)
class Filters:
    """Filters on a step view."""

    snap: bool = False
    rprofs: bool = False
    fields: set[str] = field(default_factory=set)
    funcs: list[Callable[[Step], bool]] = field(default_factory=list)

    def passes(self, step: Step) -> bool:
        """Whether a given Step passes the filters."""
        if self.snap and step.isnap is None:
            return False
        if self.rprofs:
            try:
                _ = step.rprofs.centers
            except error.MissingDataError:
                return False
        if any(fld not in step.fields for fld in self.fields):
            return False
        return all(func(step) for func in self.funcs)

    def compose_with(self, other: Filters) -> Filters:
        return Filters(
            snap=self.snap or other.snap,
            rprofs=self.rprofs or other.rprofs,
            fields=self.fields | other.fields,
            funcs=self.funcs + other.funcs,
        )


@dataclass(frozen=True)
class StepsView:
    """Filtered iterator over steps or snaps.

    Instances of this class are returned when taking slices of
    `StagyyData.steps` or `StagyyData.snaps` attributes.

    Args:
        over: steps collection, i.e. `StagyyData.steps` or
            `StagyyData.snaps` attributes.
        items: iterable of isteps/isnaps or slices.
    """

    over: Steps | Snaps
    items: Sequence[StepIndex]
    filters: Filters = field(default_factory=Filters)

    @cached_property
    def rprofs_averaged(self) -> RprofsAveraged:
        """Time-averaged radial profiles."""
        return RprofsAveraged(self)

    @cached_property
    def stepstr(self) -> str:
        """String representation of the requested set of steps."""
        items = []
        no_slice = True
        for item in self.items:
            if isinstance(item, slice):
                items.append("{}:{}:{}".format(*item.indices(len(self.over))))
                no_slice = False
            else:
                items.append(repr(item))
        item_str = ",".join(items)
        if no_slice and len(items) == 1:
            item_str += ","
        colstr = repr(self.over).rsplit(".", maxsplit=1)[-1]
        return f"{colstr}[{item_str}]"

    def _pass(self, item: int) -> bool:
        """Check whether an item passes the filters."""
        try:
            step = self.over[item]
        except KeyError:
            return False
        return self.filters.passes(step)

    def filter(
        self,
        snap: bool = False,
        rprofs: bool = False,
        fields: Iterable[str] | None = None,
        func: Callable[[Step], bool] | None = None,
    ) -> StepsView:
        """Add filters to the view.

        Note that filters are only resolved when the view is iterated.
        Successive calls to :meth:`filter` compose.  For example, with this
        code:

        ```py
        view = sdat.steps[500:].filter(rprofs=True, fields=["T"])
        view.filter(fields=["eta"])
        ```

        the produced `view`, when iterated, will generate the steps after the
        500-th that have radial profiles, and both the temperature and
        viscosity fields.

        Args:
            snap: if true, the step must be a snapshot to pass.
            rprofs: if true, the step must have rprofs data to pass.
            fields: list of fields that must be present to pass.
            func: arbitrary function returning whether a step should pass the
                filter.

        Returns:
            self.
        """
        new_filters = Filters(
            snap=snap,
            rprofs=rprofs,
            fields=set() if fields is None else set(fields),
            funcs=[] if func is None else [func],
        )
        return StepsView(
            over=self.over,
            items=self.items,
            filters=self.filters.compose_with(new_filters),
        )

    def __iter__(self) -> Iterator[Step]:
        for item in self.items:
            if isinstance(item, slice):
                idx = item.indices(len(self.over))
                yield from (self.over[i] for i in range(*idx) if self._pass(i))
            elif self._pass(item):
                yield self.over[item]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, abc.Iterable):
            return NotImplemented
        return all(s1 is s2 for s1, s2 in zip_longest(self, other))


def _sdat_from_conf(core: Core) -> StagyyData:
    return StagyyData(core.path, core.read_parameters_dat)


@dataclass(frozen=True)
class StagyyData:
    """Generic lazy interface to StagYY output data.

    Args:
        path_hint: path of the StagYY run. It can either be the path of the
            directory containing the par file, or the path of the par file. If
            the path given is a directory, the path of the par file is assumed
            to be path/par.
        read_parameters_dat: read `parameters.dat` file produced by StagYY. This
            flag can be switched off to ignore this file. This is intended for
            runs of StagYY that predate version 1.2.6 for which the
            `parameters.dat` file contained some values affected by internal
            logic.
    """

    path_hint: PathLike | str
    read_parameters_dat: bool = True

    @property
    def path(self) -> Path:
        """Path of StagYY run directory."""
        return self.parpath.parent

    @cached_property
    def parpath(self) -> Path:
        """Path of par file."""
        parpath = Path(self.path_hint)
        if parpath.is_file():
            return parpath
        return parpath / "par"

    @cached_property
    def hdf5(self) -> Path | None:
        """Path of output hdf5 folder if relevant, None otherwise."""
        h5xmf = self.par.h5_output("Data.xmf")
        return h5xmf.parent if h5xmf.is_file() else None

    @cached_property
    def steps(self) -> Steps:
        """Collection of time steps."""
        return Steps(self)

    @cached_property
    def snaps(self) -> Snaps:
        """Collection of snapshots."""
        return Snaps(self)

    @cached_property
    def tseries(self) -> Tseries:
        """Time series data."""
        return Tseries(self)

    @cached_property
    def refstate(self) -> Refstate:
        """Reference state profiles."""
        return Refstate(self)

    @cached_property
    def _dataxmf(self) -> FieldXmf:
        assert self.hdf5 is not None
        return FieldXmf(
            path=self.hdf5 / "Data.xmf",
        )

    @cached_property
    def _topxmf(self) -> FieldXmf:
        assert self.hdf5 is not None
        return FieldXmf(
            path=self.hdf5 / "DataSurface.xmf",
        )

    @cached_property
    def _botxmf(self) -> FieldXmf:
        assert self.hdf5 is not None
        return FieldXmf(
            path=self.hdf5 / "DataBottom.xmf",
        )

    @cached_property
    def _traxmf(self) -> TracersXmf:
        assert self.hdf5 is not None
        return TracersXmf(
            path=self.hdf5 / "DataTracers.xmf",
        )

    @cached_property
    def par(self) -> StagyyPar:
        """Content of par file."""
        return StagyyPar.from_main_par(self.parpath, self.read_parameters_dat)

    @cached_property
    def _rprof_and_times(self) -> tuple[dict[int, DataFrame], DataFrame | None]:
        rproffile = self.filename("rprof.h5")
        data = stagyyparsers.rprof_h5(rproffile, list(phyvars.RPROF.keys()))
        if data[1] is not None:
            return data
        rproffile = self.filename("rprof.dat")
        if self.hdf5 and not rproffile.is_file():
            # check legacy folder as well
            rproffile = self.filename("rprof.dat", force_legacy=True)
        return stagyyparsers.rprof(rproffile, list(phyvars.RPROF.keys()))

    @property
    def rtimes(self) -> DataFrame | None:
        """Radial profiles times."""
        return self._rprof_and_times[1]

    @cached_property
    def _files(self) -> set[Path]:
        """Set of found binary files output by StagYY."""
        out_dir = self.par.legacy_output("_").parent
        if out_dir.is_dir():
            return set(out_dir.iterdir())
        return set()

    def set_nfields_max(self, nfields: int | None) -> None:
        """Adjust maximum number of scalar fields kept in memory.

        Setting this to a value lower or equal to 5 raises a
        [stagpy.error.InvalidNfieldsError][].  Set this to `None` if
        you do not want any limit on the number of scalar fields kept in
        memory.  Defaults to 50.
        """
        if nfields is not None and nfields <= 5:
            raise error.InvalidNfieldsError(nfields)
        self._field_cache.resize(nfields)

    def filename(
        self,
        fname: str,
        timestep: int | None = None,
        suffix: str = "",
        force_legacy: bool = False,
    ) -> Path:
        """Return name of StagYY output file.

        Args:
            fname: name stem.
            timestep: snapshot number if relevant.
            suffix: optional suffix of file name.
            force_legacy: force returning the legacy output path.

        Returns:
            the path of the output file constructed with the provided segments.
        """
        if timestep is not None:
            fname += f"{timestep:05d}"
        fname += suffix
        if not force_legacy and self.hdf5:
            fpath = self.par.h5_output(fname)
        else:
            fpath = self.par.legacy_output(f"_{fname}")
        return fpath

    def _binfiles_set(self, isnap: int) -> set[Path]:
        """Set of existing binary files at a given snap.

        Args:
            isnap: snapshot index.

        Returns:
            the set of output files available for this snapshot number.
        """
        possible_files = set(
            self.filename(fstem, isnap, force_legacy=True)
            for fstem in phyvars.FIELD_FILES
        )
        return possible_files & self._files

    @cached_property
    def _field_cache(self) -> FieldCache:
        return FieldCache(maxsize=50)
