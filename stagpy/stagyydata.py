"""Define high level structure StagyyData"""

import re
import pathlib
from os.path import expanduser, expandvars
import numpy as np
from . import constants, parfile, stagyyparsers


UNDETERMINED = object()
# dummy object with a unique identifier,
# useful to mark stuff as yet undetermined,
# as opposed to either some value or None if
# non existent


class Error(Exception):

    """Base class for exceptions raised in this module"""

    pass


class NoSnapshotError(Error):

    """Raised when last snapshot is required but none exists"""

    pass


class _Geometry:

    """Geometry information"""

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
            t_mesh, p_mesh, r_mesh = np.meshgrid(
                self.x_coord, self.y_coord, self.z_coord + self.rcmb,
                indexing='ij')
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
        """Determine shape of geometry"""
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
        """Cartesian geometry"""
        return not self.curvilinear

    @property
    def curvilinear(self):
        """Spherical or cylindrical geometry"""
        return self.spherical or self.cylindrical

    @property
    def cylindrical(self):
        """Cylindrical geometry (2D spherical)"""
        return self._shape['cyl']

    @property
    def spherical(self):
        """Spherical geometry"""
        return self._shape['sph']

    @property
    def yinyang(self):
        """Yin-yang geometry (3D spherical)"""
        return self.spherical and self.nbtot == 2

    @property
    def twod_xz(self):
        """XZ plane only"""
        return self.nytot == 1

    @property
    def twod_yz(self):
        """YZ plane only"""
        return self.nxtot == 1

    @property
    def twod(self):
        """2D geometry"""
        return self.twod_xz or self.twod_yz

    @property
    def threed(self):
        """3D geometry"""
        return not self.twod

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


class _Fields(dict):

    """Wrap fields of a step"""

    def __init__(self, step):
        self.step = step
        self._header = UNDETERMINED
        self._geom = UNDETERMINED
        super().__init__()

    def __missing__(self, name):
        if name not in constants.FIELD_VAR_LIST:
            raise ValueError("Unknown field variable: '{}'".format(name))
        par_type = constants.FIELD_VAR_LIST[name].par
        fieldfile = self.step.sdat.filename(par_type, self.step.isnap)
        parsed_data = stagyyparsers.fields(fieldfile)
        if parsed_data is None:
            return None
        header, fields = parsed_data
        self._header = header
        if par_type == 'vp':
            fld_names = ['u', 'v', 'w', 'p']
        elif par_type == 'sx':
            fld_names = ['sxi', 'sxj', 'sxk', 'x']
        else:
            fld_names = [name]  # wrong for some stuff like stream func
        if name not in fld_names:
            # could use a function for fields not in a file (such as stream)
            # if can't call it, assume this is the name of the field file
            print("'{}' field computation not available".format(name))
            return None
        for fld_name, fld in zip(fld_names, fields):
            if self._header['xyp'] == 0:
                if not self.geom.twod_yz:
                    newline = (fld[:1, :, :, :] + fld[-1:, :, :, :]) / 2
                    fld = np.concatenate((fld, newline), axis=0)
                if not self.geom.twod_xz:
                    newline = (fld[:, :1, :, :] + fld[:, -1:, :, :]) / 2
                    fld = np.concatenate((fld, newline), axis=1)
            self[fld_name] = fld
        return self[name]

    @property
    def geom(self):
        """Header info from bin file"""
        if self._header is UNDETERMINED:
            binfiles = self.step.sdat.binfiles_set(self.step.isnap)
            if binfiles:
                self._header = stagyyparsers.fields(binfiles.pop(),
                                                    only_header=True)
            else:
                self._header = None
        if self._geom is UNDETERMINED:
            if self._header is None:
                self._geom = None
            else:
                self._geom = _Geometry(self._header, self.step.sdat.par)
        return self._geom


class _Step:

    """Time step data structure"""

    def __init__(self, istep, sdat):
        self.istep = istep
        self.sdat = sdat
        self.fields = _Fields(self)
        self._isnap = UNDETERMINED

    @property
    def geom(self):
        """Geometry object"""
        return self.fields.geom

    @property
    def timeinfo(self):
        """Relevant time series data"""
        if self.istep in self.sdat.tseries.index:
            return self.sdat.tseries.loc[self.istep]
        else:
            return None

    @property
    def rprof(self):
        """Relevant radial profiles data"""
        if self.istep in self.sdat.rprof.index.levels[0]:
            return self.sdat.rprof.loc[self.istep]
        else:
            return None

    @property
    def isnap(self):
        """Fields snap corresponding to time step"""
        if self._isnap is UNDETERMINED:
            istep = None
            isnap = -1
            # could be more efficient if do 0 and -1 then bisection
            # (but loose intermediate <- would probably use too much
            # memory for what it's worth if search algo is efficient)
            while (istep is None or istep < self.istep) and isnap < 99999:
                isnap += 1
                istep = self.sdat.snaps[isnap].istep
                self.sdat.snaps.bind(isnap, istep)
                # all intermediate istep could have their ._isnap to None
            if istep != self.istep:
                self._isnap = None
        return self._isnap

    @isnap.setter
    def isnap(self, isnap):
        """Fields snap corresponding to time step"""
        try:
            self._isnap = int(isnap)
        except ValueError:
            pass


class _EmptyStep(_Step):

    """Dummy step object for nonexistent snaps"""

    def __init__(self):
        super().__init__(None, None)

    def __getattribute__(self, name):
        return None


class _Steps(dict):

    """Implement the .steps[istep] accessor"""

    def __init__(self, sdat):
        self.sdat = sdat
        self._last = UNDETERMINED
        super().__init__()

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        try:
            # slice
            idxs = key.indices(self.last.istep + 1)
            return (super(self.__class__, self).__getitem__(k)
                    for k in range(*idxs))
        except AttributeError:
            return super().__getitem__(key)

    def __missing__(self, istep):
        if istep is None:  # if called for nonexistent snap
            return _EmptyStep()
        try:
            istep = int(istep)
        except ValueError:
            raise ValueError('Time step should be an integer value')
        if istep < 0:
            istep += self.last.istep + 1
            if istep < 0:
                raise ValueError('Time step should be positive')
        if not self.__contains__(istep):
            super().__setitem__(istep, _Step(istep, self.sdat))
        return super().__getitem__(istep)

    @property
    def last(self):
        """Last timestep available"""
        if self._last is UNDETERMINED:
            # not necessarily the last one...
            self._last = self.sdat.tseries.index[-1]
        return self[self._last]


class _Snaps(_Steps):

    """Implement the .snaps[isnap] accessor"""

    def __init__(self, sdat):
        self._isteps = {}
        super().__init__(sdat)

    def __getitem__(self, key):
        try:
            # slice
            idxs = key.indices(self.last.isnap + 1)
            return (self.__missing__(k) for k in range(*idxs))
        except AttributeError:
            return self.__missing__(key)

    def __missing__(self, isnap):
        if isnap < 0:
            isnap += self.last.isnap + 1
        istep = self._isteps.get(isnap, UNDETERMINED)
        if istep is UNDETERMINED:
            binfiles = self.sdat.binfiles_set(isnap)
            if binfiles:
                istep = stagyyparsers.fields(binfiles.pop(), only_istep=True)
            else:
                istep = None
            if istep is not None:
                self.bind(isnap, istep)
            else:
                self._isteps[isnap] = None
        return self.sdat.steps[istep]

    def bind(self, isnap, istep):
        """Make the isnap <-> istep link"""
        self._isteps[isnap] = istep
        self.sdat.steps[istep].isnap = isnap

    @property
    def last(self):
        """Last snapshot available"""
        if self._last is UNDETERMINED:
            self._last = -1
            out_stem = re.escape(pathlib.Path(
                self.sdat.par['ioin']['output_file_stem'] + '_').name[:-1])
            rgx = re.compile('^{}_([a-zA-Z]+)([0-9]{{5}})$'.format(out_stem))
            pars = set(item.par for item in constants.FIELD_VAR_LIST.values())
            for fname in self.sdat.files:
                match = rgx.match(fname.name)
                if match is not None and match.group(1) in pars:
                    self._last = max(int(match.group(2)), self._last)
            if self._last < 0:
                raise NoSnapshotError
        return self[self._last]


class StagyyData:

    """Offer a generic interface to StagYY output data"""

    def __init__(self, path):
        """Generic lazy StagYY output data accessors"""
        self.path = pathlib.Path(expanduser(expandvars(path)))
        self.par = parfile.readpar(self.path)
        self.steps = _Steps(self)
        self.snaps = _Snaps(self)
        self._files = UNDETERMINED
        self._tseries = UNDETERMINED
        self._rprof = UNDETERMINED

    @property
    def tseries(self):
        """Time series data"""
        if self._tseries is UNDETERMINED:
            timefile = self.filename('time.dat')
            self._tseries = stagyyparsers.time_series(
                timefile, list(constants.TIME_VARS.keys()))
        return self._tseries

    @property
    def _rprof_and_times(self):
        if self._rprof is UNDETERMINED:
            rproffile = self.filename('rprof.dat')
            self._rprof = stagyyparsers.rprof(
                rproffile, list(constants.RPROF_VARS.keys()))
        return self._rprof

    @property
    def rprof(self):
        """Radial profiles data"""
        return self._rprof_and_times[0]

    @property
    def rtimes(self):
        """Radial profiles data"""
        return self._rprof_and_times[1]

    @property
    def files(self):
        """Set of output binary files"""
        if self._files is UNDETERMINED:
            out_stem = pathlib.Path(self.par['ioin']['output_file_stem'] + '_')
            out_dir = self.path / out_stem.parent
            self._files = set(out_dir.iterdir())
        return self._files

    def tseries_between(self, tstart=0., tend=None):
        """Time series data between requested times"""
        if self.tseries is None:
            return None

        ndat = self.tseries.shape[0]

        if tstart <= 0.:
            istart = 0
        else:
            igm = 0
            igp = ndat - 1
            while igp - igm > 1:
                istart = igm + (igp - igm) // 2
                if self.tseries.iloc[istart]['t'] >= tstart:
                    igp = istart
                else:
                    igm = istart
            istart = igp

        if tend is None:
            iend = None
        else:
            igm = 0
            igp = ndat - 1
            while igp - igm > 1:
                iend = igm + (igp - igm) // 2
                if self.tseries.iloc[iend]['t'] > tend:
                    igp = iend
                else:
                    igm = iend
            iend = igm + 1

        return self.tseries.iloc[istart:iend]

    def filename(self, fname, timestep=None, suffix=''):
        """return name of StagYY out file"""
        if timestep is not None:
            fname += '{:05d}'.format(timestep)
        fname = self.par['ioin']['output_file_stem'] + '_' + fname + suffix
        return self.path / fname

    def binfiles_set(self, isnap):
        """Set of existing binary files at a given snap"""
        possible_files = set(self.filename(item.par, isnap)
                             for item in constants.FIELD_VAR_LIST.values())
        return possible_files & self.files
