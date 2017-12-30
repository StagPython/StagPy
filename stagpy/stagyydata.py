"""Define high level structure StagyyData and helper classes.

Note:
    The helper classes are not designed to be instantiated on their own, but
    only as attributes of StagyyData instances. Users of this module should
    only instantiate :class:`StagyyData`.
"""

import re
import pathlib
import numpy as np
from . import error, parfile, phyvars, stagyyparsers


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

    """Fields data structure.

    The :attr:`_Step.fields` attribute is an instance of this class.

    :class:`_Fields` inherits from :class:`dict`. Keys are fields names defined
    in :data:`stagpy.phyvars.FIELD`.

    Attributes:
        step (:class:`_Step`): the step object owning the :class:`_Fields`
            instance.
    """

    def __init__(self, step):
        self.step = step
        self._header = UNDETERMINED
        self._geom = UNDETERMINED
        super().__init__()

    def __missing__(self, name):
        if name in phyvars.FIELD:
            filestem = ''
            for filestem, list_fvar in phyvars.FIELD_FILES.items():
                if name in list_fvar:
                    break
            fieldfile = self.step.sdat.filename(filestem, self.step.isnap)
        elif name in phyvars.FIELD_EXTRA:
            self[name] = phyvars.FIELD_EXTRA[name].description(self.step)
            return self[name]
        else:
            raise error.UnknownFieldVarError(name)
        parsed_data = stagyyparsers.fields(fieldfile)
        if parsed_data is None:
            return None
        header, fields = parsed_data
        self._header = header
        fld_names = phyvars.FIELD_FILES[filestem]
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

    def __setitem__(self, name, fld):
        sdat = self.step.sdat
        col_fld = sdat.collected_fields
        col_fld.append((self.step.istep, name))
        while len(col_fld) > sdat.nfields_max > 5:
            istep, fld_name = col_fld.pop(0)
            del sdat.steps[istep].fields[fld_name]
        super().__setitem__(name, fld)

    @property
    def geom(self):
        """Geometry information.

        :class:`_Geometry` instance holding geometry information. It is
        issued from binary files holding field information. It is set to
        None if not available for this time step.
        """
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

    """Time step data structure.

    Elements of :class:`_Steps` and :class:`_Snaps` instances are all
    :class:`_Step` instances. Note that :class:`_Step` objects are not
    duplicated.

    Examples:
        Here are a few examples illustrating some properties of :class:`_Step`
        instances.

        >>> sdat = StagyyData('path/to/run')
        >>> istep_last_snap = sdat.snaps.last.istep
        >>> assert(sdat.steps[istep_last_snap] is sdat.snaps.last)
        >>> n = 0  # or any valid time step / snapshot index
        >>> assert(sdat.steps[n].sdat is sdat)
        >>> assert(sdat.steps[n].istep == n)
        >>> assert(sdat.snaps[n].isnap == n)
        >>> assert(sdat.steps[n].geom is sdat.steps[n].fields.geom)
        >>> assert(sdat.snaps[n] is sdat.snaps[n].fields.step)
    """

    def __init__(self, istep, sdat):
        """Initialization of instances:

        This class should not be instantiated by the user, the instantiation
        is handled by :class:`StagyyData`.

        Args:
            istep (int): the index of the time step that the instance
                represents.
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Step` instance.

        Attributes:
            istep (int): the index of the time step that the instance
                represents.
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Step` instance.
            fields (:class:`_Fields`): fields available at this time step.
        """
        self.istep = istep
        self.sdat = sdat
        self.fields = _Fields(self)
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
        if self.istep in self.sdat.tseries.index:
            return self.sdat.tseries.loc[self.istep]
        else:
            return None

    @property
    def rprof(self):
        """Radial profiles data of the time step.

        Set to None if no radial profiles data is available for this time step.
        """
        if self.istep in self.sdat.rprof.index.levels[0]:
            return self.sdat.rprof.loc[self.istep]
        else:
            return None

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

    """Dummy step object for nonexistent snaps.

    This class inherits from :class:`_Step`, but its :meth:`__getattribute__`
    method always return :obj:`None`.
    """

    def __init__(self):
        super().__init__(None, None)

    def __getattribute__(self, name):
        return None


class _Steps(dict):

    """Collections of time steps.

    The :attr:`StagyyData.steps` attribute is an instance of this class.
    Time steps (which are :class:`_Step` instances) can be accessed with the
    item accessor::

        sdat = StagyyData('path/to/run')
        sdat.steps[istep]  # _Step object of the istep-th time step
    """

    def __init__(self, sdat):
        """Initialization of instances:

        Args:
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Steps` instance.
        Attributes:
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Steps` instance.
        """
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
            raise error.InvalidTimestepError(
                self.sdat, istep, 'Time step should be an integer value')
        if istep < 0:
            istep += self.last.istep + 1
            if istep < 0:
                istep -= self.last.istep + 1
                raise error.InvalidTimestepError(
                    self.sdat, istep,
                    'Last istep is {}'.format(self.last.istep))
        if not self.__contains__(istep):
            super().__setitem__(istep, _Step(istep, self.sdat))
        return super().__getitem__(istep)

    @property
    def last(self):
        """Last time step available.

            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.steps.last is sdat.steps[-1])
        """
        if self._last is UNDETERMINED:
            # not necessarily the last one...
            self._last = self.sdat.tseries.index[-1]
        return self[self._last]


class _Snaps(_Steps):

    """Collections of snapshots.

    The :attr:`StagyyData.snaps` attribute is an instance of this class.
    Snapshots (which are :class:`_Step` instances) can be accessed with the
    item accessor::

        sdat = StagyyData('path/to/run')
        sdat.snaps[isnap]  # _Step object of the isnap-th snapshot

    This class inherits from :class:`_Steps`.
    """

    def __init__(self, sdat):
        """Initialization of instances:

        Args:
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Snaps` instance.
        Attributes:
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Snaps` instance.
        """
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
        """Register the isnap / istep correspondence.

        Users of :class:`StagyyData` should not use this method.

        Args:
            isnap (int): snapshot index.
            istep (int): time step index.
        """
        self._isteps[isnap] = istep
        self.sdat.steps[istep].isnap = isnap

    @property
    def last(self):
        """Last snapshot available.

            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.snaps.last is sdat.snaps[-1])
        """
        if self._last is UNDETERMINED:
            self._last = -1
            out_stem = re.escape(pathlib.Path(
                self.sdat.par['ioin']['output_file_stem'] + '_').name[:-1])
            rgx = re.compile('^{}_([a-zA-Z]+)([0-9]{{5}})$'.format(out_stem))
            fstems = set(fstem for fstem in phyvars.FIELD_FILES)
            for fname in self.sdat.files:
                match = rgx.match(fname.name)
                if match is not None and match.group(1) in fstems:
                    self._last = max(int(match.group(2)), self._last)
            if self._last < 0:
                raise error.NoSnapshotError(self.sdat)
        return self[self._last]


class StagyyData:

    """Generic lazy interface to StagYY output data."""

    def __init__(self, path, nfields_max=50):
        """Initialization of instances:

        Args:
            path (pathlike): path of the StagYY run.
            nfields_max (int): the maximum number of scalar fields that should
                be kept in memory. Set to a value smaller than 6 if you want no
                limit.

        Attributes:
            steps (:class:`_Steps`): collection of time steps.
            snaps (:class:`_Snaps`): collection of snapshots.
            nfields_max (int): the maximum number of scalar fields that should
                be kept in memory. Set to a value smaller than 6 if you want no
                limit.
            collected_fields (list of (int, str)): list of fields currently in
                memory, described by istep and field name.
        """
        self._rundir = {'path': pathlib.Path(path),
                        'ls': UNDETERMINED}
        self._stagdat = {'par': parfile.readpar(self.path),
                         'tseries': UNDETERMINED,
                         'rprof': UNDETERMINED}
        self.steps = _Steps(self)
        self.snaps = _Snaps(self)
        self.nfields_max = nfields_max
        self.collected_fields = []

    def __repr__(self):
        return 'StagyyData({}, nfields_max={})'.format(
            repr(self.path), self.nfields_max)

    def __str__(self):
        return 'StagyyData in {}'.format(self.path)

    @property
    def path(self):
        """Path of StagYY run directory.

        :class:`pathlib.Path` instance.
        """
        return self._rundir['path']

    @property
    def par(self):
        """Content of par file.

        This is a dictionary of dictionaries, the first key being namelists and
        the second key the parameter name.
        """
        return self._stagdat['par']

    @property
    def tseries(self):
        """Time series data.

        This is a :class:`pandas.DataFrame` with istep as index and variable
        names as columns.
        """
        if self._stagdat['tseries'] is UNDETERMINED:
            timefile = self.filename('time.dat')
            self._stagdat['tseries'] = stagyyparsers.time_series(
                timefile, list(phyvars.TIME.keys()))
        return self._stagdat['tseries']

    @property
    def _rprof_and_times(self):
        if self._stagdat['rprof'] is UNDETERMINED:
            rproffile = self.filename('rprof.dat')
            self._stagdat['rprof'] = stagyyparsers.rprof(
                rproffile, list(phyvars.RPROF.keys()))
        return self._stagdat['rprof']

    @property
    def rprof(self):
        """Radial profiles data.

        This is a :class:`pandas.DataFrame` with a 2-level index (istep and iz)
        and variable names as columns.
        """
        return self._rprof_and_times[0]

    @property
    def rtimes(self):
        """Radial profiles times.

        This is a :class:`pandas.DataFrame` with istep as index.
        """
        return self._rprof_and_times[1]

    @property
    def files(self):
        """Set of found binary files output by StagYY."""
        if self._rundir['ls'] is UNDETERMINED:
            out_stem = pathlib.Path(self.par['ioin']['output_file_stem'] + '_')
            out_dir = self.path / out_stem.parent
            self._rundir['ls'] = set(out_dir.iterdir())
        return self._rundir['ls']

    def tseries_between(self, tstart=None, tend=None):
        """Return time series data between requested times.

        Args:
            tstart (float): starting time. Set to None to start at the
                beginning of available data.
            tend (float): ending time. Set to None to stop at the end of
                available data.
        Returns:
            :class:`pandas.DataFrame`: slice of :attr:`tseries`.
        """
        if self.tseries is None:
            return None

        ndat = self.tseries.shape[0]

        if tstart is None:
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
        """Return name of StagYY output file.

        Args:
            fname (str): name stem.
            timestep (int): snapshot number, set to None if this is not
                relevant.
            suffix (str): optional suffix of file name.
        Returns:
            :class:`pathlib.Path`: the path of the output file constructed
            with the provided segments.
        """
        if timestep is not None:
            fname += '{:05d}'.format(timestep)
        fname = self.par['ioin']['output_file_stem'] + '_' + fname + suffix
        return self.path / fname

    def binfiles_set(self, isnap):
        """Set of existing binary files at a given snap.

        Args:
            isnap (int): snapshot index.
        Returns:
            set of pathlib.Path: the set of output files available for this
            snapshot number.
        """
        possible_files = set(self.filename(fstem, isnap)
                             for fstem in phyvars.FIELD_FILES)
        return possible_files & self.files
