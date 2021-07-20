"""Implementation of Step objects.

Note:
    This module and the classes it defines are internals of StagPy, they
    should not be used in an external script.  Instead, use the
    :class:`~stagpy.stagyydata.StagyyData` class.
"""

from collections.abc import Mapping
from collections import namedtuple
from itertools import chain
import re

import numpy as np

from . import error, misc, phyvars, stagyyparsers
from .misc import CachedReadOnlyProperty as crop


class _Geometry:
    """Geometry information.

    It is deduced from the information in the header of binary field files
    output by StagYY.

    Attributes:
        nxtot, nytot, nztot, nttot, nptot, nrtot, nbtot (int): number of grid
            point in the various directions. Note that nxtot==nttot,
            nytot==nptot, nztot==nrtot.
        x_coord, y_coord, z_coord, t_coord, p_coord, r_coord (numpy.array):
            positions of cell centers in the various directions. Note that
            x_coord==t_coord, y_coord==p_coord, z_coord==r_coord.
        x_mesh, y_mesh, z_mesh, t_mesh, p_mesh, r_mesh (numpy.array):
            mesh in cartesian and curvilinear frames. The last three are
            not defined if the geometry is cartesian.
    """

    _regexes = (re.compile(r'^n([xyztprb])tot$'),  # ntot
                re.compile(r'^([xyztpr])_coord$'),  # coord
                re.compile(r'^([xyz])_mesh$'),  # cartesian mesh
                re.compile(r'^([tpr])_mesh$'))  # curvilinear mesh

    def __init__(self, header, par):
        self._header = header
        self._par = par
        self._coords = None
        self._cart_meshes = None
        self._curv_meshes = None
        self._shape = {'sph': False, 'cyl': False, 'axi': False,
                       'ntot': list(header['nts']) + [header['ntb']]}
        self._init_shape()

        self._coords = [header['e1_coord'],
                        header['e2_coord'],
                        header['e3_coord']]

        # instead of adding horizontal rows, should construct two grids:
        # - center of cells coordinates (the current one);
        # - vertices coordinates on which vector fields are determined,
        #   which geometrically contains one more row.

        # add theta, phi / x, y row to have a continuous field
        if not self.twod_yz:
            self._coords[0] = np.append(
                self.t_coord,
                self.t_coord[-1] + self.t_coord[1] - self.t_coord[0])
        if not self.twod_xz:
            self._coords[1] = np.append(
                self.p_coord,
                self.p_coord[-1] + self.p_coord[1] - self.p_coord[0])

        if self.cartesian:
            self._cart_meshes = np.meshgrid(self.x_coord, self.y_coord,
                                            self.z_coord, indexing='ij')
            self._curv_meshes = (None, None, None)
        else:
            if self.twod_yz:
                self._coords[0] = np.array(np.pi / 2)
            elif self.twod_xz:
                self._coords[1] = np.array(0)
            self._coords[2] += self.rcmb
            if par['magma_oceans_in']['magma_oceans_mode']:
                self._coords[2] += header['mo_lambda']
                self._coords[2] *= header['mo_thick_sol']
            t_mesh, p_mesh, r_mesh = np.meshgrid(
                self.t_coord, self.p_coord, self.r_coord, indexing='ij')
            # compute cartesian coordinates
            # z along rotation axis at theta=0
            # x at th=90, phi=0
            # y at th=90, phi=90
            x_mesh = r_mesh * np.cos(p_mesh) * np.sin(t_mesh)
            y_mesh = r_mesh * np.sin(p_mesh) * np.sin(t_mesh)
            z_mesh = r_mesh * np.cos(t_mesh)
            self._cart_meshes = (x_mesh, y_mesh, z_mesh)
            self._curv_meshes = (t_mesh, p_mesh, r_mesh)

    def _init_shape(self):
        """Determine shape of geometry."""
        shape = self._par['geometry']['shape'].lower()
        aspect = self._header['aspect']
        if self.rcmb is not None and self.rcmb >= 0:
            # curvilinear
            self._shape['cyl'] = self.twod_xz and (shape == 'cylindrical' or
                                                   aspect[0] >= np.pi)
            self._shape['sph'] = not self._shape['cyl']
        elif self.rcmb is None:
            self._header['rcmb'] = self._par['geometry']['r_cmb']
            if self.rcmb >= 0:
                if self.twod_xz and shape == 'cylindrical':
                    self._shape['cyl'] = True
                elif shape == 'spherical':
                    self._shape['sph'] = True
        self._shape['axi'] = self.cartesian and self.twod_xz and \
            shape == 'axisymmetric'

    @property
    def cartesian(self):
        """Whether the grid is in cartesian geometry."""
        return not self.curvilinear

    @property
    def curvilinear(self):
        """Whether the grid is in curvilinear geometry."""
        return self.spherical or self.cylindrical

    @property
    def cylindrical(self):
        """Whether the grid is in cylindrical geometry (2D spherical)."""
        return self._shape['cyl']

    @property
    def spherical(self):
        """Whether the grid is in spherical geometry."""
        return self._shape['sph']

    @property
    def yinyang(self):
        """Whether the grid is in Yin-yang geometry (3D spherical)."""
        return self.spherical and self.nbtot == 2

    @property
    def twod_xz(self):
        """Whether the grid is in the XZ plane only."""
        return self.nytot == 1

    @property
    def twod_yz(self):
        """Whether the grid is in the YZ plane only."""
        return self.nxtot == 1

    @property
    def twod(self):
        """Whether the grid is 2 dimensional."""
        return self.twod_xz or self.twod_yz

    @property
    def threed(self):
        """Whether the grid is 3 dimensional."""
        return not self.twod

    def at_z(self, zval):
        """Return iz closest to given zval position.

        In spherical geometry, the bottom boundary is considered to be at z=0.
        Use :meth:`at_r` to find a cell at a given radial position.
        """
        if self.curvilinear:
            zval += self.rcmb
        return np.argmin(np.abs(self.z_coord - zval))

    def at_r(self, rval):
        """Return ir closest to given rval position.

        If called in cartesian geometry, this is equivalent to :meth:`at_z`.
        """
        return np.argmin(np.abs(self.r_coord - rval))

    def __getattr__(self, attr):
        # provide nDtot, D_coord, D_mesh and nbtot
        # with D = x, y, z or t, p, r
        for reg, dat in zip(self._regexes, (self._shape['ntot'],
                                            self._coords,
                                            self._cart_meshes,
                                            self._curv_meshes)):
            match = reg.match(attr)
            if match is not None:
                return dat['xtypzrb'.index(match.group(1)) // 2]
        return self._header[attr]


class _Fields(Mapping):
    """Fields data structure.

    The :attr:`Step.fields` attribute is an instance of this class.

    :class:`_Fields` inherits from :class:`collections.abc.Mapping`. Keys are
    fields names defined in :data:`stagpy.phyvars.[S]FIELD[_EXTRA]`.

    Attributes:
        step (:class:`Step`): the step object owning the :class:`_Fields`
            instance.
    """

    def __init__(self, step, variables, extravars, files, filesh5):
        self.step = step
        self._vars = variables
        self._extra = extravars
        self._files = files
        self._filesh5 = filesh5
        self._data = {}
        super().__init__()

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
        if name in self._vars:
            fld_names, parsed_data = self._get_raw_data(name)
        elif name in self._extra:
            self._data[name] = self._extra[name].description(self.step)
            return self._data[name]
        else:
            raise error.UnknownFieldVarError(name)
        if parsed_data is None:
            raise error.MissingDataError(
                f'Missing field {name} in step {self.step.istep}')
        header, fields = parsed_data
        self._cropped__header = header
        for fld_name, fld in zip(fld_names, fields):
            if self._header['xyp'] == 0:
                if not self.geom.twod_yz:
                    newline = (fld[:1, ...] + fld[-1:, ...]) / 2
                    fld = np.concatenate((fld, newline), axis=0)
                if not self.geom.twod_xz:
                    newline = (fld[:, :1, ...] + fld[:, -1:, ...]) / 2
                    fld = np.concatenate((fld, newline), axis=1)
            self._set(fld_name, fld)
        return self._data[name]

    def __iter__(self):
        return (fld for fld in chain(self._vars, self._extra)
                if fld in self)

    def __contains__(self, item):
        try:
            return self[item] is not None
        except error.StagpyError:
            return False

    def __len__(self):
        return len(iter(self))

    def __eq__(self, other):
        return self is other

    def _get_raw_data(self, name):
        """Find file holding data and return its content."""
        # try legacy first, then hdf5
        filestem = ''
        for filestem, list_fvar in self._files.items():
            if name in list_fvar:
                break
        fieldfile = self.step.sdat.filename(filestem, self.step.isnap,
                                            force_legacy=True)
        if not fieldfile.is_file():
            fieldfile = self.step.sdat.filename(filestem, self.step.isnap)
        parsed_data = None
        if fieldfile.is_file():
            parsed_data = stagyyparsers.fields(fieldfile)
        elif self.step.sdat.hdf5 and self._filesh5:
            for filestem, list_fvar in self._filesh5.items():
                if name in list_fvar:
                    break
            if filestem in phyvars.SFIELD_FILES_H5:
                xmff = 'Data{}.xmf'.format(
                    'Bottom' if name.endswith('bot') else 'Surface')
                header = self._header
            else:
                xmff = 'Data.xmf'
                header = None
            parsed_data = stagyyparsers.read_field_h5(
                self.step.sdat.hdf5 / xmff, filestem, self.step.isnap, header)
        return list_fvar, parsed_data

    def _set(self, name, fld):
        sdat = self.step.sdat
        col_fld = sdat._collected_fields
        col_fld.append((self.step.istep, name))
        if sdat.nfields_max is not None:
            while len(col_fld) > sdat.nfields_max:
                istep, fld_name = col_fld.pop(0)
                del sdat.steps[istep].fields[fld_name]
        self._data[name] = fld

    def __delitem__(self, name):
        if name in self._data:
            del self._data[name]

    @crop
    def _header(self):
        binfiles = self.step.sdat._binfiles_set(self.step.isnap)
        if binfiles:
            return stagyyparsers.fields(binfiles.pop(), only_header=True)
        elif self.step.sdat.hdf5:
            xmf = self.step.sdat.hdf5 / 'Data.xmf'
            return stagyyparsers.read_geom_h5(xmf, self.step.isnap)[0]

    @crop
    def geom(self):
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information. It is set to
        None if not available for this time step.
        """
        if self._header is not None:
            return _Geometry(self._header, self.step.sdat.par)


class _Tracers:
    """Tracers data structure.

    The :attr:`Step.tracers` attribute is an instance of this class.

    :class:`_Tracers` implements the getitem mechanism. Items are tracervar
    names such as 'Type' or 'Mass'.  The position of tracers are the 'x', 'y'
    and 'z' items.

    Attributes:
        step (:class:`Step`): the step object owning the :class:`_Tracers`
            instance.
    """

    def __init__(self, step):
        self.step = step
        self._data = {}

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
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

    def __iter__(self):
        raise TypeError('tracers collection is not iterable')


Rprof = namedtuple('Rprof', ['values', 'rad', 'meta'])


class _Rprofs:
    """Radial profiles data structure.

    The :attr:`Step.rprofs` attribute is an instance of this class.

    :class:`_Rprofs` implements the getitem mechanism.  Keys are profile names
    defined in :data:`stagpy.phyvars.RPROF[_EXTRA]`.  An item is a named tuple
    ('values', 'rad', 'meta'), respectively the profile itself, the radial
    position at which it is evaluated, and meta is a
    :class:`stagpy.phyvars.Varr` instance with relevant metadata.  Note that
    profiles are automatically scaled if conf.scaling.dimensional is True.

    Attributes:
        step (:class:`Step`): the step object owning the :class:`_Rprofs`
            instance
    """

    def __init__(self, step):
        self.step = step
        self._cached_extra = {}

    @crop
    def _data(self):
        step = self.step
        return step.sdat._rprof_and_times[0].get(step.istep)

    @property
    def _rprofs(self):
        if self._data is None:
            step = self.step
            raise error.MissingDataError(
                f'No rprof data in step {step.istep} of {step.sdat}')
        return self._data

    def __str__(self):
        """ MD: String Representation of rprof object.
            Would be great to have this for all classes 
        """
        return ' | '.join(self._rprofs.columns.to_list())

    def __getitem__(self, name):
        step = self.step
        if name in self._rprofs.columns:
            rprof = self._rprofs[name].values
            rad = self.centers
            if name in phyvars.RPROF:
                meta = phyvars.RPROF[name]
            else:
                meta = phyvars.Varr(name, None, '1')
        elif name in self._cached_extra:
            rprof, rad, meta = self._cached_extra[name]
        elif name in phyvars.RPROF_EXTRA:
            meta = phyvars.RPROF_EXTRA[name]
            rprof, rad = meta.description(step)
            meta = phyvars.Varr(misc.baredoc(meta.description),
                                meta.kind, meta.dim)
            self._cached_extra[name] = rprof, rad, meta
        else:
            raise error.UnknownRprofVarError(name)
        rprof, _ = step.sdat.scale(rprof, meta.dim)
        rad, _ = step.sdat.scale(rad, 'm')

        return Rprof(rprof, rad, meta)

    @crop
    def centers(self):
        """Radial position of cell centers."""
        return self._rprofs['r'].values + self.bounds[0]

    @crop
    def walls(self):
        """Radial position of cell walls."""
        rbot, rtop = self.bounds
        centers = self.centers
        # assume walls are mid-way between T-nodes
        # could be T-nodes at center between walls
        walls = (centers[:-1] + centers[1:]) / 2
        walls = np.insert(walls, 0, rbot)
        walls = np.append(walls, rtop)
        return walls

    @crop
    def bounds(self):
        """Radial or vertical position of boundaries.

        Radial/vertical positions of boundaries of the domain.
        """
        step = self.step
        if step.geom is not None:
            rcmb = step.geom.rcmb
        else:
            rcmb = step.sdat.par['geometry']['r_cmb']
            if step.sdat.par['geometry']['shape'].lower() == 'cartesian':
                rcmb = 0
        rcmb = max(rcmb, 0)
        return rcmb, rcmb + 1


class Step:
    """Time step data structure.

    Elements of :class:`_Steps` and :class:`_Snaps` instances are all
    :class:`Step` instances. Note that :class:`Step` objects are not
    duplicated.

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
        istep (int): the index of the time step that the instance
            represents.
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance owning the :class:`Step` instance.

    Attributes:
        istep (int): the index of the time step that the instance
            represents.
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance owning the :class:`Step` instance.
        fields (:class:`_Fields`): fields available at this time step.
        sfields (:class:`_Fields`): surface fields available at this time
            step.
        tracers (:class:`_Tracers`): tracers available at this time step.
    """

    def __init__(self, istep, sdat):
        self.istep = istep
        self.sdat = sdat
        self.fields = _Fields(self, phyvars.FIELD, phyvars.FIELD_EXTRA,
                              phyvars.FIELD_FILES, phyvars.FIELD_FILES_H5)
        self.sfields = _Fields(self, phyvars.SFIELD, [],
                               phyvars.SFIELD_FILES, phyvars.SFIELD_FILES_H5)
        self.tracers = _Tracers(self)
        self.rprofs = _Rprofs(self)
        self._isnap = -1

    def __repr__(self):
        if self.isnap is not None:
            return f'{self.sdat!r}.snaps[{self.isnap}]'
        else:
            return f'{self.sdat!r}.steps[{self.istep}]'

    @property
    def geom(self):
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information. It is set to
        None if not available for this time step.
        """
        return self.fields.geom

    @property
    def timeinfo(self):
        """Time series data of the time step."""
        try:
            info = self.sdat.tseries.at_step(self.istep)
        except KeyError:
            raise error.MissingDataError(f'No time series for {self!r}')
        return info

    @property
    def time(self):
        """Time of this time step."""
        steptime = None
        try:
            steptime = self.timeinfo['t']
        except error.MissingDataError:
            if self.isnap is not None:
                steptime = self.geom.ti_ad
        return steptime

    @property
    def isnap(self):
        """Snapshot index corresponding to time step.

        It is set to None if no snapshot exists for the time step.
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
