"""Implementation of Step objects.

Note:
    This module and the classes it defines are internals of StagPy, they
    should not be used in an external script.  Instead, use the
    [`StagyyData`][stagpy.stagyydata.StagyyData] class.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import error, phyvars, stagyyparsers
from .datatypes import Field, Rprof, Varr
from .dimensions import Scales

if typing.TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Callable, NoReturn

    from numpy.typing import NDArray
    from pandas import DataFrame, Series

    from ._caching import FieldCache
    from .datatypes import Varf
    from .stagyydata import StagyyData


@dataclass(frozen=True)
class Geometry:
    """Geometry information.

    It is deduced from the information in the header of binary field files
    output by StagYY.
    """

    step: Step

    @cached_property
    def _maybe_header(self) -> dict[str, Any] | None:
        if self.step.isnap is None:
            return None
        sdat = self.step.sdat
        binfiles = sdat._binfiles_set(self.step.isnap)
        header = None
        if binfiles:
            header = stagyyparsers.field_header(binfiles.pop())
        elif sdat.hdf5:
            header = stagyyparsers.read_geom_h5(sdat._dataxmf, self.step.isnap)
        return header if header else None

    @cached_property
    def _header(self) -> Mapping[str, Any]:
        if self._maybe_header is None:
            raise error.NoGeomError(self.step)
        return self._maybe_header

    def _scale_radius_mo(self, radius: NDArray) -> NDArray:
        """Rescale radius for evolving MO runs."""
        if self.step.sdat.par.get("magma_oceans_in", "evolving_magma_oceans", False):
            return self._header["mo_thick_sol"] * (radius + self._header["mo_lambda"])
        return radius

    @property
    def aspect_ratio(self) -> tuple[float, float]:
        return self.step.sdat.par.nml["geometry"]["aspect_ratio"]

    @cached_property
    def _ntot(self) -> tuple[int, int, int, int]:
        return (*self._header["nts"], self._header["ntb"])

    @cached_property
    def nttot(self) -> int:
        """Number of grid point along the x/theta direction."""
        return self._ntot[0]

    @cached_property
    def nptot(self) -> int:
        """Number of grid point along the y/phi direction."""
        return self._ntot[1]

    @cached_property
    def nrtot(self) -> int:
        """Number of grid point along the z/r direction."""
        return self._ntot[2]

    @cached_property
    def nbtot(self) -> int:
        """Number of blocks."""
        return self._ntot[3]

    @property
    def nxtot(self) -> int:
        """Same as nttot."""
        return self.nttot

    @property
    def nytot(self) -> int:
        """Same as nptot."""
        return self.nptot

    @property
    def nztot(self) -> int:
        """Same as nrtot."""
        return self.nrtot

    @cached_property
    def r_walls(self) -> NDArray:
        """Position of FV walls along the z/r direction."""
        rgeom = self._header.get("rgeom")
        if rgeom is not None:
            walls = rgeom[:, 0] + self.rcmb
        else:
            walls = self._header["e3_coord"] + self.rcmb
            walls = np.append(walls, self.step.rprofs.bounds[1])
        return self._scale_radius_mo(walls)

    @cached_property
    def r_centers(self) -> NDArray:
        """Position of FV centers along the z/r direction."""
        rgeom = self._header.get("rgeom")
        if rgeom is not None:
            walls = rgeom[:-1, 1] + self.rcmb
        else:
            walls = self.step.rprofs.centers
        return self._scale_radius_mo(walls)

    @cached_property
    def t_walls(self) -> NDArray:
        """Position of FV walls along x/theta."""
        if self.threed or self.twod_xz:
            if self.yinyang:
                tmin, tmax = -np.pi / 4, np.pi / 4
            elif self.curvilinear:
                # should take theta_position/theta_center into account
                tmin = 0
                tmax = min(np.pi, self.aspect_ratio[0])
            else:
                tmin = 0
                tmax = self.aspect_ratio[0]
            return np.linspace(tmin, tmax, self.nttot + 1)
        # twoD YZ
        center = np.pi / 2 if self.curvilinear else 0
        d_t = (self.p_walls[1] - self.p_walls[0]) / 2
        return np.array([center - d_t, center + d_t])

    @cached_property
    def t_centers(self) -> NDArray:
        """Position of FV centers along x/theta."""
        return (self.t_walls[:-1] + self.t_walls[1:]) / 2

    @cached_property
    def p_walls(self) -> NDArray:
        """Position of FV walls along y/phi."""
        if self.threed or self.twod_yz:
            if self.yinyang:
                pmin, pmax = -3 * np.pi / 4, 3 * np.pi / 4
            elif self.curvilinear:
                pmin = 0
                pmax = min(2 * np.pi, self.aspect_ratio[1])
            else:
                pmin = 0
                pmax = self.aspect_ratio[1]
            return np.linspace(pmin, pmax, self.nptot + 1)
        # twoD YZ
        d_p = (self.t_walls[1] - self.t_walls[0]) / 2
        return np.array([-d_p, d_p])

    @cached_property
    def p_centers(self) -> NDArray:
        """Position of FV centers along y/phi."""
        return (self.p_walls[:-1] + self.p_walls[1:]) / 2

    @property
    def z_walls(self) -> NDArray:
        """Same as r_walls."""
        return self.r_walls

    @property
    def z_centers(self) -> NDArray:
        """Same as r_centers."""
        return self.r_centers

    @property
    def x_walls(self) -> NDArray:
        """Same as t_walls."""
        return self.t_walls

    @property
    def x_centers(self) -> NDArray:
        """Same as t_centers."""
        return self.t_centers

    @property
    def y_walls(self) -> NDArray:
        """Same as p_walls."""
        return self.p_walls

    @property
    def y_centers(self) -> NDArray:
        """Same as p_centers."""
        return self.p_centers

    @cached_property
    def rcmb(self) -> float:
        """Radius of CMB, 0 in cartesian geometry."""
        return max(self._header["rcmb"], 0)

    @cached_property
    def _shape(self) -> str:
        return self.step.sdat.par.nml["geometry"]["shape"].lower()

    @cached_property
    def curvilinear(self) -> bool:
        """Whether the grid is in curvilinear geometry."""
        return self._header["rcmb"] >= 0

    @property
    def cartesian(self) -> bool:
        """Whether the grid is in cartesian geometry."""
        return not self.curvilinear

    @cached_property
    def cylindrical(self) -> bool:
        """Whether the grid is in cylindrical geometry (2D spherical)."""
        aspect = self._header["aspect"]
        return (
            self.curvilinear
            and self.twod_xz
            and (self._shape == "cylindrical" or aspect[0] >= np.pi)
        )

    @cached_property
    def spherical(self) -> bool:
        """Whether the grid is in spherical geometry."""
        return self.curvilinear and not self.cylindrical

    @cached_property
    def axisymmetric(self) -> bool:
        """Whether the grid is in cartesian axisymmetric geometry."""
        return self.cartesian and self.twod_xz and self._shape == "axisymmetric"

    @property
    def yinyang(self) -> bool:
        """Whether the grid is in Yin-yang geometry (3D spherical)."""
        return self.spherical and self.nbtot == 2

    @property
    def twod_xz(self) -> bool:
        """Whether the grid is in the XZ plane only."""
        return self.nytot == 1

    @property
    def twod_yz(self) -> bool:
        """Whether the grid is in the YZ plane only."""
        return self.nxtot == 1

    @property
    def twod(self) -> bool:
        """Whether the grid is 2 dimensional."""
        return self.twod_xz or self.twod_yz

    @property
    def threed(self) -> bool:
        """Whether the grid is 3 dimensional."""
        return not self.twod

    def at_z(self, zval: float) -> int:
        """Return iz closest to given zval position.

        In spherical geometry, the bottom boundary is considered to be at z=0.
        Use `at_r` to find a cell at a given radial position.
        """
        if self.curvilinear:
            zval += self.rcmb
        return int(np.argmin(np.abs(self.z_centers - zval)))

    def at_r(self, rval: float) -> int:
        """Return ir closest to given rval position.

        If called in cartesian geometry, this is equivalent to `at_z`.
        """
        return int(np.argmin(np.abs(self.r_centers - rval)))


@dataclass(frozen=True)
class Fields:
    """Fields data structure.

    The `Step.fields` attribute is an instance of this class.
    """

    step: Step
    variables: Mapping[str, Varf]
    extravars: Mapping[str, Callable[[Step], Field]]
    files: Mapping[str, list[str]]
    filesh5: Mapping[str, list[str]]

    @cached_property
    def _all_vars(self) -> set[str]:
        return set(self.variables.keys()).union(self.extravars.keys())

    @cached_property
    def _cache(self) -> FieldCache:
        return self.step.sdat._field_cache

    def __getitem__(self, name: str) -> Field:
        if name not in self._all_vars:
            raise error.UnknownFieldVarError(name)

        maybe_fld = self._cache.get(self.step.istep, name)
        if maybe_fld is not None:
            return maybe_fld

        if name in self.extravars:
            fld = self.extravars[name](self.step)
            self._cache.insert(self.step.istep, name, fld)
            return fld

        # requested field is one of self._vars
        fld_names, parsed_data = self._get_raw_data(name)
        if parsed_data is None:
            raise error.MissingDataError(
                f"Missing field {name} in step {self.step.istep}"
            )
        header, fields = parsed_data
        for fld_name, fld_vals in zip(fld_names, fields):
            fld = Field(fld_vals, self.variables[fld_name])
            self._cache.insert(self.step.istep, fld_name, fld)
        return self[name]

    def __contains__(self, item: Any) -> bool:
        try:
            return self[item] is not None
        except error.MissingDataError:
            return False

    def __eq__(self, other: object) -> bool:
        return self is other

    def _get_raw_data(self, name: str) -> tuple[list[str], Any]:
        """Find file holding data and return its content."""
        # try legacy first, then hdf5
        filestem = ""
        for filestem, list_fvar in self.files.items():
            if name in list_fvar:
                break
        parsed_data = None
        if self.step.isnap is None:
            return list_fvar, None
        fieldfile = self.step.sdat.filename(
            filestem, self.step.isnap, force_legacy=True
        )
        if not fieldfile.is_file():
            fieldfile = self.step.sdat.filename(filestem, self.step.isnap)
        if fieldfile.is_file():
            parsed_data = stagyyparsers.fields(fieldfile)
        elif self.step.sdat.hdf5 and self.filesh5:
            # files in which the requested data can be found
            files = [
                (stem, fvars) for stem, fvars in self.filesh5.items() if name in fvars
            ]
            for filestem, list_fvar in files:
                sdat = self.step.sdat
                if filestem in phyvars.SFIELD_FILES_H5:
                    xmff = sdat._botxmf if name.endswith("bot") else sdat._topxmf
                    header = self.step.geom._maybe_header
                    assert header is not None
                else:
                    xmff = sdat._dataxmf
                    header = None
                parsed_data = stagyyparsers.read_field_h5(
                    xmff, filestem, self.step.isnap, header
                )
                if parsed_data is not None:
                    break
        return list_fvar, parsed_data


@dataclass(frozen=True)
class Tracers:
    """Tracers data structure.

    The `Step.tracers` attribute is an instance of this class.

    `Tracers` implements the getitem mechanism. Items are tracervar names such
    as `"Type"` or `"Mass"`.  The position of tracers are the `"x"`, `"y"` and
    `"z"` items.
    """

    step: Step

    @cached_property
    def _data(self) -> dict[str, list[NDArray] | None]:
        return {}

    def __getitem__(self, name: str) -> list[NDArray] | None:
        if name in self._data:
            return self._data[name]
        if self.step.isnap is None:
            return None
        data = stagyyparsers.tracers(
            self.step.sdat.filename("tra", timestep=self.step.isnap, force_legacy=True)
        )
        if data is None and self.step.sdat.hdf5:
            self._data[name] = stagyyparsers.read_tracers_h5(
                self.step.sdat._traxmf,
                name,
                self.step.isnap,
            )
        elif data is not None:
            self._data.update(data)
        if name not in self._data:
            self._data[name] = None
        return self._data[name]

    def __iter__(self) -> NoReturn:
        raise TypeError("tracers collection is not iterable")


class Rprofs(ABC):
    """Radial profiles.

    `Rprofs` implements the getitem mechanism.  Keys are profile names
    defined in `stagpy.phyvars.RPROF[_EXTRA]`.  Items are
    [`Rprof`][stagpy.datatypes.Rprof] instances.
    """

    @abstractmethod
    def __getitem__(self, name: str) -> Rprof: ...

    @property
    @abstractmethod
    def stepstr(self) -> str:
        """String representation of steps indices."""

    @property
    @abstractmethod
    def centers(self) -> NDArray:
        """Radial position of cell centers."""

    @property
    @abstractmethod
    def walls(self) -> NDArray:
        """Radial position of cell walls."""

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Radial or vertical position of boundaries.

        Radial/vertical positions of boundaries of the domain.
        """


@dataclass(frozen=True)
class RprofsInstant(Rprofs):
    """Radial profiles at a given step.

    The `Step.rprofs` attribute is an instance of this class.
    """

    step: Step

    @cached_property
    def _cached_extra(self) -> dict[str, Rprof]:
        return {}

    @cached_property
    def _data(self) -> DataFrame | None:
        step = self.step
        return step.sdat._rprof_and_times[0].get(step.istep)

    @property
    def _rprofs(self) -> DataFrame:
        if self._data is None:
            step = self.step
            raise error.MissingDataError(
                f"No rprof data in step {step.istep} of {step.sdat}"
            )
        return self._data

    def __getitem__(self, name: str) -> Rprof:
        step = self.step
        if name in self._rprofs.columns:
            rprof = self._rprofs[name].to_numpy()
            rad = self.centers
            if name in phyvars.RPROF:
                meta = phyvars.RPROF[name]
            else:
                meta = Varr(name, "", "1")
        elif name in self._cached_extra:
            rpf = self._cached_extra[name]
            rprof = rpf.values
            rad = rpf.rad
            meta = rpf.meta
        elif name in phyvars.RPROF_EXTRA:
            self._cached_extra[name] = phyvars.RPROF_EXTRA[name](step)
            rpf = self._cached_extra[name]
            rprof = rpf.values
            rad = rpf.rad
            meta = rpf.meta
        else:
            raise error.UnknownRprofVarError(name)

        return Rprof(rprof, rad, meta)

    @property
    def stepstr(self) -> str:
        """String representation of the parent :class:`Step`."""
        return str(self.step.istep)

    @cached_property
    def centers(self) -> NDArray:
        """Radial position of cell centers."""
        return self._rprofs["r"].to_numpy() + self.bounds[0]

    @cached_property
    def walls(self) -> NDArray:
        """Radial position of cell walls."""
        rbot, rtop = self.bounds
        try:
            walls = self.step.geom.r_walls
        except error.StagpyError:
            # assume walls are mid-way between T-nodes
            # could be T-nodes at center between walls
            centers = self.centers
            walls = (centers[:-1] + centers[1:]) / 2
            walls = np.insert(walls, 0, rbot)
            walls = np.append(walls, rtop)
        return walls

    @cached_property
    def bounds(self) -> tuple[float, float]:
        """Radial or vertical position of boundaries.

        Radial/vertical positions of boundaries of the domain.
        """
        step = self.step
        try:
            rcmb = step.geom.rcmb
        except error.NoGeomError:
            rcmb = step.sdat.par.get("geometry", "r_cmb", 3480e3)
            if step.sdat.par.nml["geometry"]["shape"].lower() == "cartesian":
                rcmb = 0
        rbot = max(rcmb, 0)
        thickness = (
            Scales(step.sdat).length
            if step.sdat.par.get("switches", "dimensional_units", True)
            else 1
        )
        return rbot, rbot + thickness


@dataclass(frozen=True)
class Step:
    """Time step data structure.

    Elements of [`Steps`][stagpy.stagyydata.Steps] and
    [`Snaps`][stagpy.stagyydata.Snaps] instances are all `Step`
    instances. Note that `Step` objects are not duplicated.

    Examples:
        Here are a few examples illustrating some properties of `Step`
        instances.

        ```py
        sdat = StagyyData(Path("path/to/run"))
        istep_last_snap = sdat.snaps[-1].istep
        assert(sdat.steps[istep_last_snap] is sdat.snaps[-1])
        n = 0  # or any valid time step / snapshot index
        assert(sdat.steps[n].sdat is sdat)
        assert(sdat.steps[n].istep == n)
        assert(sdat.snaps[n].isnap == n)
        assert(sdat.snaps[n] is sdat.snaps[n].fields.step)
        ```

    Attributes:
        istep: the index of the time step that the instance represents.
        sdat: the owner of the `Step` instance.
    """

    istep: int
    sdat: StagyyData

    @cached_property
    def fields(self) -> Fields:
        """Fields available at this time step."""
        return Fields(
            self,
            phyvars.FIELD,
            phyvars.FIELD_EXTRA,
            phyvars.FIELD_FILES,
            phyvars.FIELD_FILES_H5,
        )

    @cached_property
    def sfields(self) -> Fields:
        """Surface fields available at this time step."""
        return Fields(
            self,
            phyvars.SFIELD,
            {},
            phyvars.SFIELD_FILES,
            phyvars.SFIELD_FILES_H5,
        )

    @cached_property
    def tracers(self) -> Tracers:
        """Tracer information available at this time step."""
        return Tracers(self)

    @cached_property
    def rprofs(self) -> RprofsInstant:
        """Radial profiles available at this time step."""
        return RprofsInstant(self)

    def __repr__(self) -> str:
        if self.isnap is not None:
            return f"{self.sdat!r}.snaps[{self.isnap}]"
        else:
            return f"{self.sdat!r}.steps[{self.istep}]"

    @cached_property
    def geom(self) -> Geometry:
        """Geometry information.

        [`Geometry`][stagpy.step.Geometry] instance holding geometry information.
        It is issued from binary files holding field information.
        """
        return Geometry(self)

    @property
    def timeinfo(self) -> Series:
        """Time series data of the time step."""
        try:
            info = self.sdat.tseries.at_step(self.istep)
        except KeyError:
            raise error.MissingDataError(f"No time series for {self!r}")
        return info

    @property
    def time(self) -> float:
        """Time of this time step."""
        steptime = None
        try:
            steptime = self.timeinfo["t"]
        except error.MissingDataError:
            if self.isnap is not None:
                steptime = self.geom._header.get("ti_ad")
        if steptime is None:
            raise error.NoTimeError(self)
        return steptime

    @property
    def isnap(self) -> int | None:
        """Snapshot index corresponding to time step.

        It is None if no snapshot exists for the time step.
        """
        return self.sdat._step_snap.isnap(istep=self.istep)
