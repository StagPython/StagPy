"""Implementation of Step objects.

Note:
    This module and the classes it defines are internals of StagPy, they
    should not be used in an external script.  Instead, use the
    :class:`~stagpy.stagyydata.StagyyData` class.
"""

from __future__ import annotations
from collections import abc
from itertools import chain
import typing

import numpy as np

from . import error, phyvars, stagyyparsers
from ._helpers import CachedReadOnlyProperty as crop
from .datatypes import Field, Rprof, Varr

if typing.TYPE_CHECKING:
    from typing import (Dict, Any, Mapping, List, Iterator, Tuple, Optional,
                        Callable, NoReturn)
    from numpy import ndarray
    from pandas import DataFrame, Series
    from .datatypes import Varf
    from .stagyydata import StagyyData


class _Geometry:
    """Geometry information.

    It is deduced from the information in the header of binary field files
    output by StagYY.
    """

    def __init__(self, header: Dict[str, Any], step: Step):
        self._header = header
        self._step = step
        self._shape: Dict[str, Any] = {
            'sph': False, 'cyl': False, 'axi': False,
            'ntot': list(header['nts']) + [header['ntb']]}
        self._init_shape()

    def _scale_radius_mo(self, radius: ndarray) -> ndarray:
        """Rescale radius for MO runs."""
        if self._step.sdat.par['magma_oceans_in']['magma_oceans_mode']:
            return self._header['mo_thick_sol'] * (
                radius + self._header['mo_lambda'])
        return radius

    @crop
    def nttot(self) -> int:
        """Number of grid point along the x/theta direction."""
        return self._shape['ntot'][0]

    @crop
    def nptot(self) -> int:
        """Number of grid point along the y/phi direction."""
        return self._shape['ntot'][1]

    @crop
    def nrtot(self) -> int:
        """Number of grid point along the z/r direction."""
        return self._shape['ntot'][2]

    @crop
    def nbtot(self) -> int:
        """Number of blocks."""
        return self._shape['ntot'][3]

    nxtot = nttot
    nytot = nptot
    nztot = nrtot

    @crop
    def r_walls(self) -> ndarray:
        """Position of FV walls along the z/r direction."""
        rgeom = self._header.get("rgeom")
        if rgeom is not None:
            walls = rgeom[:, 0] + self.rcmb
        else:
            walls = self._header["e3_coord"] + self.rcmb
            walls = np.append(walls, self._step.rprofs.bounds[1])
        return self._scale_radius_mo(walls)

    @crop
    def r_centers(self) -> ndarray:
        """Position of FV centers along the z/r direction."""
        rgeom = self._header.get("rgeom")
        if rgeom is not None:
            walls = rgeom[:-1, 1] + self.rcmb
        else:
            walls = self._step.rprofs.centers
        return self._scale_radius_mo(walls)

    @crop
    def t_walls(self) -> ndarray:
        """Position of FV walls along x/theta."""
        if self.threed or self.twod_xz:
            if self.yinyang:
                tmin, tmax = -np.pi / 4, np.pi / 4
            elif self.curvilinear:
                # should take theta_position/theta_center into account
                tmin = 0
                tmax = min(np.pi,
                           self._step.sdat.par['geometry']['aspect_ratio'][0])
            else:
                tmin = 0
                tmax = self._step.sdat.par['geometry']['aspect_ratio'][0]
            return np.linspace(tmin, tmax, self.nttot + 1)
        # twoD YZ
        center = np.pi / 2 if self.curvilinear else 0
        d_t = (self.p_walls[1] - self.p_walls[0]) / 2
        return np.array([center - d_t, center + d_t])

    @crop
    def t_centers(self) -> ndarray:
        """Position of FV centers along x/theta."""
        return (self.t_walls[:-1] + self.t_walls[1:]) / 2

    @crop
    def p_walls(self) -> ndarray:
        """Position of FV walls along y/phi."""
        if self.threed or self.twod_yz:
            if self.yinyang:
                pmin, pmax = -3 * np.pi / 4, 3 * np.pi / 4
            elif self.curvilinear:
                pmin = 0
                pmax = min(2 * np.pi,
                           self._step.sdat.par['geometry']['aspect_ratio'][1])
            else:
                pmin = 0
                pmax = self._step.sdat.par['geometry']['aspect_ratio'][1]
            return np.linspace(pmin, pmax, self.nptot + 1)
        # twoD YZ
        d_p = (self.t_walls[1] - self.t_walls[0]) / 2
        return np.array([-d_p, d_p])

    @crop
    def p_centers(self) -> ndarray:
        """Position of FV centers along y/phi."""
        return (self.p_walls[:-1] + self.p_walls[1:]) / 2

    z_walls = r_walls
    z_centers = r_centers
    x_walls = t_walls
    x_centers = t_centers
    y_walls = p_walls
    y_centers = p_centers

    def _init_shape(self) -> None:
        """Determine shape of geometry."""
        shape = self._step.sdat.par['geometry']['shape'].lower()
        aspect = self._header['aspect']
        if self._header['rcmb'] >= 0:
            # curvilinear
            self._shape['cyl'] = self.twod_xz and (shape == 'cylindrical' or
                                                   aspect[0] >= np.pi)
            self._shape['sph'] = not self._shape['cyl']
        self._shape['axi'] = self.cartesian and self.twod_xz and \
            shape == 'axisymmetric'

    @crop
    def rcmb(self) -> float:
        """Radius of CMB, 0 in cartesian geometry."""
        return max(self._header["rcmb"], 0)

    @property
    def cartesian(self) -> bool:
        """Whether the grid is in cartesian geometry."""
        return not self.curvilinear

    @property
    def curvilinear(self) -> bool:
        """Whether the grid is in curvilinear geometry."""
        return self.spherical or self.cylindrical

    @property
    def cylindrical(self) -> bool:
        """Whether the grid is in cylindrical geometry (2D spherical)."""
        return self._shape['cyl']

    @property
    def spherical(self) -> bool:
        """Whether the grid is in spherical geometry."""
        return self._shape['sph']

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
        Use :meth:`at_r` to find a cell at a given radial position.
        """
        if self.curvilinear:
            zval += self.rcmb
        return int(np.argmin(np.abs(self.z_centers - zval)))

    def at_r(self, rval: float) -> int:
        """Return ir closest to given rval position.

        If called in cartesian geometry, this is equivalent to :meth:`at_z`.
        """
        return int(np.argmin(np.abs(self.r_centers - rval)))


class _Fields(abc.Mapping):
    """Fields data structure.

    The :attr:`Step.fields` attribute is an instance of this class.

    :class:`_Fields` inherits from :class:`collections.abc.Mapping`. Keys are
    fields names defined in :data:`stagpy.phyvars.[S]FIELD[_EXTRA]`.  Each item
    is a :class:`stagpy.datatypes.Field` instance.

    Attributes:
        step: the step object owning the :class:`_Fields` instance.
    """

    def __init__(self, step: Step, variables: Mapping[str, Varf],
                 extravars: Mapping[str, Callable[[Step], Field]],
                 files: Mapping[str, List[str]],
                 filesh5: Mapping[str, List[str]]):
        self.step = step
        self._vars = variables
        self._extra = extravars
        self._files = files
        self._filesh5 = filesh5
        self._data: Dict[str, Field] = {}
        super().__init__()

    def __getitem__(self, name: str) -> Field:
        if name in self._data:
            return self._data[name]
        if name in self._vars:
            fld_names, parsed_data = self._get_raw_data(name)
        elif name in self._extra:
            self._data[name] = self._extra[name](self.step)
            return self._data[name]
        else:
            raise error.UnknownFieldVarError(name)
        if parsed_data is None:
            raise error.MissingDataError(
                f'Missing field {name} in step {self.step.istep}')
        header, fields = parsed_data
        self._cropped__header = header
        for fld_name, fld in zip(fld_names, fields):
            self._set(fld_name, fld)
        return self._data[name]

    @crop
    def _present_fields(self) -> List[str]:
        return [fld for fld in chain(self._vars, self._extra)
                if fld in self]

    def __iter__(self) -> Iterator[str]:
        return iter(self._present_fields)

    def __contains__(self, item: Any) -> bool:
        try:
            return self[item] is not None
        except error.MissingDataError:
            return False

    def __len__(self) -> int:
        return len(self._present_fields)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _get_raw_data(self, name: str) -> Tuple[List[str], Any]:
        """Find file holding data and return its content."""
        # try legacy first, then hdf5
        filestem = ''
        for filestem, list_fvar in self._files.items():
            if name in list_fvar:
                break
        parsed_data = None
        if self.step.isnap is None:
            return list_fvar, None
        fieldfile = self.step.sdat.filename(filestem, self.step.isnap,
                                            force_legacy=True)
        if not fieldfile.is_file():
            fieldfile = self.step.sdat.filename(filestem, self.step.isnap)
        if fieldfile.is_file():
            parsed_data = stagyyparsers.fields(fieldfile)
        elif self.step.sdat.hdf5 and self._filesh5:
            # files in which the requested data can be found
            files = [(stem, fvars) for stem, fvars in self._filesh5.items()
                     if name in fvars]
            for filestem, list_fvar in files:
                if filestem in phyvars.SFIELD_FILES_H5:
                    xmff = 'Data{}.xmf'.format(
                        'Bottom' if name.endswith('bot') else 'Surface')
                    header = self._header
                else:
                    xmff = 'Data.xmf'
                    header = None
                parsed_data = stagyyparsers.read_field_h5(
                    self.step.sdat.hdf5 / xmff, filestem,
                    self.step.isnap, header)
                if parsed_data is not None:
                    break
        return list_fvar, parsed_data

    def _set(self, name: str, fld: ndarray) -> None:
        sdat = self.step.sdat
        col_fld = sdat._collected_fields
        col_fld.append((self.step.istep, name))
        if sdat.nfields_max is not None:
            while len(col_fld) > sdat.nfields_max:
                istep, fld_name = col_fld.pop(0)
                del sdat.steps[istep].fields[fld_name]
        self._data[name] = Field(fld, self._vars[name])

    def __delitem__(self, name: str) -> None:
        if name in self._data:
            del self._data[name]

    @crop
    def _header(self) -> Optional[Dict[str, Any]]:
        if self.step.isnap is None:
            return None
        binfiles = self.step.sdat._binfiles_set(self.step.isnap)
        header = None
        if binfiles:
            header = stagyyparsers.field_header(binfiles.pop())
        elif self.step.sdat.hdf5:
            xmf = self.step.sdat.hdf5 / 'Data.xmf'
            header = stagyyparsers.read_geom_h5(xmf, self.step.isnap)[0]
        return header if header else None

    @crop
    def geom(self) -> _Geometry:
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information.
        """
        if self._header is None:
            raise error.NoGeomError(self.step)
        return _Geometry(self._header, self.step)


class _Tracers:
    """Tracers data structure.

    The :attr:`Step.tracers` attribute is an instance of this class.

    :class:`_Tracers` implements the getitem mechanism. Items are tracervar
    names such as 'Type' or 'Mass'.  The position of tracers are the 'x', 'y'
    and 'z' items.

    Attributes:
        step: the :class:`Step` owning the :class:`_Tracers` instance.
    """

    def __init__(self, step: Step):
        self.step = step
        self._data: Dict[str, Optional[List[ndarray]]] = {}

    def __getitem__(self, name: str) -> Optional[List[ndarray]]:
        if name in self._data:
            return self._data[name]
        if self.step.isnap is None:
            return None
        data = stagyyparsers.tracers(
            self.step.sdat.filename('tra', timestep=self.step.isnap,
                                    force_legacy=True))
        if data is None and self.step.sdat.hdf5:
            position = any(axis not in self._data for axis in 'xyz')
            self._data.update(
                stagyyparsers.read_tracers_h5(
                    self.step.sdat.hdf5 / 'DataTracers.xmf', name,
                    self.step.isnap, position))
        elif data is not None:
            self._data.update(data)
        if name not in self._data:
            self._data[name] = None
        return self._data[name]

    def __iter__(self) -> NoReturn:
        raise TypeError('tracers collection is not iterable')


class _Rprofs:
    """Radial profiles data structure.

    The :attr:`Step.rprofs` attribute is an instance of this class.

    :class:`_Rprofs` implements the getitem mechanism.  Keys are profile names
    defined in :data:`stagpy.phyvars.RPROF[_EXTRA]`.  Items are
    :class:`stagpy.datatypes.Rprof` instances.  Note that
    profiles are automatically scaled if conf.scaling.dimensional is True.

    Attributes:
        step: the :class:`Step` owning the :class:`_Rprofs` instance
    """

    def __init__(self, step: Step):
        self.step = step
        self._cached_extra: Dict[str, Rprof] = {}

    @crop
    def _data(self) -> Optional[DataFrame]:
        step = self.step
        return step.sdat._rprof_and_times[0].get(step.istep)

    @property
    def _rprofs(self) -> DataFrame:
        if self._data is None:
            step = self.step
            raise error.MissingDataError(
                f'No rprof data in step {step.istep} of {step.sdat}')
        return self._data

    def __getitem__(self, name: str) -> Rprof:
        step = self.step
        if name in self._rprofs.columns:
            rprof = self._rprofs[name].values
            rad = self.centers
            if name in phyvars.RPROF:
                meta = phyvars.RPROF[name]
            else:
                meta = Varr(name, '', '1')
        elif name in self._cached_extra:
            rprof, rad, meta = self._cached_extra[name]
        elif name in phyvars.RPROF_EXTRA:
            self._cached_extra[name] = phyvars.RPROF_EXTRA[name](step)
            rprof, rad, meta = self._cached_extra[name]
        else:
            raise error.UnknownRprofVarError(name)
        rprof, _ = step.sdat.scale(rprof, meta.dim)
        rad, _ = step.sdat.scale(rad, 'm')

        return Rprof(rprof, rad, meta)

    @property
    def stepstr(self) -> str:
        """String representation of the parent :class:`Step`."""
        return str(self.step.istep)

    @crop
    def centers(self) -> ndarray:
        """Radial position of cell centers."""
        return self._rprofs['r'].values + self.bounds[0]

    @crop
    def walls(self) -> ndarray:
        """Radial position of cell walls."""
        rbot, rtop = self.bounds
        try:
            walls = self.step.fields.geom.r_walls
        except error.StagpyError:
            # assume walls are mid-way between T-nodes
            # could be T-nodes at center between walls
            centers = self.centers
            walls = (centers[:-1] + centers[1:]) / 2
            walls = np.insert(walls, 0, rbot)
            walls = np.append(walls, rtop)
        return walls

    @crop
    def bounds(self) -> Tuple[float, float]:
        """Radial or vertical position of boundaries.

        Radial/vertical positions of boundaries of the domain.
        """
        step = self.step
        try:
            rcmb = step.geom.rcmb
        except error.NoGeomError:
            rcmb = step.sdat.par['geometry']['r_cmb']
            if step.sdat.par['geometry']['shape'].lower() == 'cartesian':
                rcmb = 0
        rbot = max(rcmb, 0)
        thickness = (step.sdat.scales.length
                     if step.sdat.par['switches']['dimensional_units'] else 1)
        return rbot, rbot + thickness


class Step:
    """Time step data structure.

    Elements of :class:`~stagpy.stagyydata._Steps` and
    :class:`~stagpy.stagyydata._Snaps` instances are all :class:`Step`
    instances. Note that :class:`Step` objects are not duplicated.

    Examples:
        Here are a few examples illustrating some properties of :class:`Step`
        instances.

        >>> sdat = StagyyData('path/to/run')
        >>> istep_last_snap = sdat.snaps[-1].istep
        >>> assert(sdat.steps[istep_last_snap] is sdat.snaps[-1])
        >>> n = 0  # or any valid time step / snapshot index
        >>> assert(sdat.steps[n].sdat is sdat)
        >>> assert(sdat.steps[n].istep == n)
        >>> assert(sdat.snaps[n].isnap == n)
        >>> assert(sdat.steps[n].geom is sdat.steps[n].fields.geom)
        >>> assert(sdat.snaps[n] is sdat.snaps[n].fields.step)

    Args:
        istep: the index of the time step that the instance represents.
        sdat: the :class:`~stagpy.stagyydata.StagyyData` instance owning the
            :class:`Step` instance.

    Attributes:
        istep: the index of the time step that the instance represents.
        sdat: the :class:`~stagpy.stagyydata.StagyyData` StagyyData instance
            owning the :class:`Step` instance.
        fields (:class:`_Fields`): fields available at this time step.
        sfields (:class:`_Fields`): surface fields available at this time
            step.
        tracers (:class:`_Tracers`): tracers available at this time step.
    """

    def __init__(self, istep: int, sdat: StagyyData):
        self.istep = istep
        self.sdat = sdat
        self.fields = _Fields(self, phyvars.FIELD, phyvars.FIELD_EXTRA,
                              phyvars.FIELD_FILES, phyvars.FIELD_FILES_H5)
        self.sfields = _Fields(self, phyvars.SFIELD, {},
                               phyvars.SFIELD_FILES, phyvars.SFIELD_FILES_H5)
        self.tracers = _Tracers(self)
        self.rprofs = _Rprofs(self)
        self._isnap: Optional[int] = -1

    def __repr__(self) -> str:
        if self.isnap is not None:
            return f'{self.sdat!r}.snaps[{self.isnap}]'
        else:
            return f'{self.sdat!r}.steps[{self.istep}]'

    @property
    def geom(self) -> _Geometry:
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information. It is set to
        None if not available for this time step.
        """
        return self.fields.geom

    @property
    def timeinfo(self) -> Series:
        """Time series data of the time step."""
        try:
            info = self.sdat.tseries.at_step(self.istep)
        except KeyError:
            raise error.MissingDataError(f'No time series for {self!r}')
        return info

    @property
    def time(self) -> float:
        """Time of this time step."""
        steptime = None
        try:
            steptime = self.timeinfo['t']
        except error.MissingDataError:
            if self.isnap is not None:
                steptime = self.geom._header.get('ti_ad')
        if steptime is None:
            raise error.NoTimeError(self)
        return steptime

    @property
    def isnap(self) -> Optional[int]:
        """Snapshot index corresponding to time step.

        It is None if no snapshot exists for the time step.
        """
        if self._isnap == -1:
            istep = None
            isnap = -1
            # could be more efficient if do 0 and -1 then bisection
            # (but loose intermediate <- would probably use too much
            # memory for what it's worth if search algo is efficient)
            while (istep is None or istep < self.istep) and isnap < 99999:
                isnap += 1
                try:
                    istep = self.sdat.snaps[isnap].istep
                except KeyError:
                    pass
                # all intermediate istep could have their ._isnap to None
            if istep != self.istep:
                self._isnap = None
        return self._isnap
