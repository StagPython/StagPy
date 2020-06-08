"""Implementation of Step objects.

Note:
    This module and the classes it defines are internals of StagPy, they
    should not be used in an external script.  Instead, use the
    :class:`~stagpy.stagyydata.StagyyData` class.
"""

from collections.abc import Mapping
from itertools import chain
import re

import numpy as np

from . import error, phyvars, stagyyparsers


UNDETERMINED = object()
# dummy object with a unique identifier,
# useful to mark stuff as yet undetermined,
# as opposed to either some value or None if
# non existent


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
        self._header = UNDETERMINED
        self._geom = UNDETERMINED
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
            raise error.MissingDataError('Missing field {} in step {}'
                                         .format(name, self.step.istep))
        header, fields = parsed_data
        self._header = header
        for fld_name, fld in zip(fld_names, fields):
            if self._header['xyp'] == 0:
                if not self.geom.twod_yz:
                    newline = (fld[:1, :, :, :] + fld[-1:, :, :, :]) / 2
                    fld = np.concatenate((fld, newline), axis=0)
                if not self.geom.twod_xz:
                    newline = (fld[:, :1, :, :] + fld[:, -1:, :, :]) / 2
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
            parsed_data = stagyyparsers.read_field_h5(
                self.step.sdat.hdf5 / 'Data.xmf', filestem, self.step.isnap)
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

    @property
    def geom(self):
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information. It is set to
        None if not available for this time step.
        """
        if self._header is UNDETERMINED:
            binfiles = self.step.sdat._binfiles_set(self.step.isnap)
            if binfiles:
                self._header = stagyyparsers.fields(binfiles.pop(),
                                                    only_header=True)
            elif self.step.sdat.hdf5:
                xmf = self.step.sdat.hdf5 / 'Data.xmf'
                self._header, _ = stagyyparsers.read_geom_h5(xmf,
                                                             self.step.isnap)
            else:
                self._header = None
        if self._geom is UNDETERMINED:
            if self._header is None:
                self._geom = None
            else:
                self._geom = _Geometry(self._header, self.step.sdat.par)
        return self._geom


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
                               phyvars.SFIELD_FILES, [])
        self.tracers = _Tracers(self)
        self._isnap = UNDETERMINED

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
        """Time series data of the time step.

        Set to None if no time series data is available for this time step.
        """
        if self.istep not in self.sdat.tseries.index:
            return None
        return self.sdat.tseries.loc[self.istep]

    @property
    def rprof(self):
        """Radial profiles data of the time step.

        Set to None if no radial profiles data is available for this time step.
        """
        if self.istep not in self.sdat.rprof.index.levels[0]:
            return None
        return self.sdat.rprof.loc[self.istep]

    @property
    def isnap(self):
        """Snapshot index corresponding to time step.

        It is set to None if no snapshot exists for the time step.
        """
        if self._isnap is UNDETERMINED:
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
