"""Define high level structure StagyyData and helper classes.

Note:
    The helper classes are not designed to be instantiated on their own, but
    only as attributes of StagyyData instances. Users of this module should
    only instantiate :class:`StagyyData`.
"""

import re
import pathlib

import numpy as np

from . import conf, error, parfile, phyvars, stagyyparsers


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
            fld_names, parsed_data = self._get_raw_data(name)
        elif name in phyvars.FIELD_EXTRA:
            self[name] = phyvars.FIELD_EXTRA[name].description(self.step)
            return self[name]
        else:
            raise error.UnknownFieldVarError(name)
        if parsed_data is None:
            return None
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
            self[fld_name] = fld
        return self[name]

    def _get_raw_data(self, name):
        """Find file holding data and return its content."""
        # try legacy first, then hdf5
        filestem = ''
        for filestem, list_fvar in phyvars.FIELD_FILES.items():
            if name in list_fvar:
                break
        fieldfile = self.step.sdat.filename(filestem, self.step.isnap,
                                            force_legacy=True)
        parsed_data = None
        if fieldfile.is_file():
            parsed_data = stagyyparsers.fields(fieldfile)
        elif self.step.sdat.hdf5:
            for filestem, list_fvar in phyvars.FIELD_FILES_H5.items():
                if name in list_fvar:
                    break
            parsed_data = stagyyparsers.read_field_h5(
                self.step.sdat.hdf5 / 'Data.xmf', filestem, self.step.isnap)
        return list_fvar, parsed_data

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
    method always return :obj:`None`. Its instances are falsy values.
    """

    def __init__(self):
        super().__init__(None, None)

    def __getattribute__(self, name):
        return None

    def __bool__(self):
        return False


class _Steps(dict):

    """Collections of time steps.

    The :attr:`StagyyData.steps` attribute is an instance of this class.
    Time steps (which are :class:`_Step` instances) can be accessed with the
    item accessor::

        sdat = StagyyData('path/to/run')
        sdat.steps[istep]  # _Step object of the istep-th time step

    Slices of :class:`_Steps` object are :class:`_StepsView` instances that
    you can iterate and filter::

        for step in steps[500:]:
            # iterate through all time steps from the 500-th one
            do_something(step)

        for step in steps[-100:].filter(snap=True):
            # iterate through all snapshots present in the last 100 time steps
            do_something(step)
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
        super().__setitem__(None, _EmptyStep())  # for non existent snaps

    def __repr__(self):
        return '{}.steps'.format(repr(self.sdat))

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        try:
            return _StepsView(self, key)
        except AttributeError:
            return super().__getitem__(key)

    def __missing__(self, istep):
        try:
            istep = int(istep)
        except ValueError:
            raise error.InvalidTimestepError(
                self.sdat, istep, 'Time step should be an integer value')
        if istep < 0:
            istep += len(self)
            if istep < 0:
                istep -= len(self)
                raise error.InvalidTimestepError(
                    self.sdat, istep,
                    'Last istep is {}'.format(self.last.istep))
        if not self.__contains__(istep):
            super().__setitem__(istep, _Step(istep, self.sdat))
        return super().__getitem__(istep)

    def __len__(self):
        return self.last.istep + 1

    def __iter__(self):
        return iter(self[:])

    def filter(self, **filters):
        """Build a _StepsView with requested filters."""
        return self[:].filter(**filters)

    @property
    def last(self):
        """Last time step available.

        Example:
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
        self._all_isteps_known = False
        super().__init__(sdat)

    def __repr__(self):
        return '{}.snaps'.format(repr(self.sdat))

    def __getitem__(self, key):
        try:
            return _StepsView(self, key)
        except AttributeError:
            return self.__missing__(key)

    def __missing__(self, isnap):
        if isnap < 0:
            isnap += len(self)
        istep = self._isteps.get(
            isnap, None if self._all_isteps_known else UNDETERMINED)
        if istep is UNDETERMINED:
            binfiles = self.sdat.binfiles_set(isnap)
            if binfiles:
                istep = stagyyparsers.fields(binfiles.pop(), only_istep=True)
            else:
                istep = None
            if istep is not None:
                self.bind(isnap, istep)
            elif self.sdat.hdf5:
                for isn, ist in stagyyparsers.read_time_h5(self.sdat.hdf5):
                    self.bind(isn, ist)
                self._all_isteps_known = True
            else:
                self._isteps[isnap] = None
        return self.sdat.steps[istep]

    def __len__(self):
        return self.last.isnap + 1

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

        Example:
            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.snaps.last is sdat.snaps[-1])
        """
        if self._last is UNDETERMINED:
            self._last = -1
            if self.sdat.hdf5:
                isnap = -1
                for isnap, istep in stagyyparsers.read_time_h5(self.sdat.hdf5):
                    self.bind(isnap, istep)
                self._last = isnap
                self._all_isteps_known = True
            if self._last < 0:
                out_stem = re.escape(pathlib.Path(
                    self.sdat.par['ioin']['output_file_stem'] + '_').name[:-1])
                rgx = re.compile(
                    '^{}_([a-zA-Z]+)([0-9]{{5}})$'.format(out_stem))
                fstems = set(fstem for fstem in phyvars.FIELD_FILES)
                for fname in self.sdat.files:
                    match = rgx.match(fname.name)
                    if match is not None and match.group(1) in fstems:
                        self._last = max(int(match.group(2)), self._last)
            if self._last < 0:
                raise error.NoSnapshotError(self.sdat)
        return self[self._last]


class _StepsView:

    """Filtered iterator over steps or snaps.

    Instances of this class are returned when taking slices of
    :attr:`StagyyData.steps` or :attr:`StagyyData.snaps` attributes.
    """

    def __init__(self, steps_col, slc):
        """Initialization of instances:

        Args:
            steps_col (:class:`_Steps` or :class:`_Snaps`): steps collection,
                i.e. :attr:`StagyyData.steps` or :attr:`StagyyData.snaps`
                attributes.
            slc (slice): slice of desired isteps or isnap.
        """
        self._col = steps_col
        self._idx = slc.indices(len(self._col))
        self._flt = {
            'snap': False,
            'rprof': False,
            'fields': [],
            'func': lambda _: True,
        }
        self._dflt_func = self._flt['func']

    def __repr__(self):
        rep = repr(self._col)
        rep += '[{}:{}:{}]'.format(*self._idx)
        flts = []
        for flt in ('snap', 'rprof', 'fields'):
            if self._flt[flt]:
                flts.append('{}={}'.format(flt, repr(self._flt[flt])))
        if self._flt['func'] is not self._dflt_func:
            flts.append('func={}'.format(repr(self._flt['func'])))
        if flts:
            rep += '.filter({})'.format(', '.join(flts))
        return rep

    def _pass(self, step):
        """Check whether a :class:`_Step` passes the filters."""
        okf = True
        okf = okf and (not self._flt['snap'] or step.isnap is not None)
        okf = okf and (not self._flt['rprof'] or step.rprof is not None)
        okf = okf and all(
            step.fields[f] is not None for f in self._flt['fields'])
        okf = okf and bool(self._flt['func'](step))
        return okf

    def filter(self, **filters):
        """Update filters with provided arguments.

        Note that filters are only resolved when the view is iterated, and
        hence they do not compose. Each call to filter merely updates the
        relevant filters. For example, with this code::

            view = sdat.steps[500:].filter(rprof=True, fields=['T'])
            view.filter(fields=[])

        the produced ``view``, when iterated, will generate the steps after the
        500-th that have radial profiles. The ``fields`` filter set in the
        first line is emptied in the second line.

        Args:
            snap (bool): the step must be a snapshot to pass.
            rprof (bool): the step must have rprof data to pass.
            fields (list): list of fields that must be present to pass.
            func (function): arbitrary function taking a :class:`_Step` as
                argument and returning a True value if the step should pass
                the filter.

        Returns:
            self.
        """
        for flt, val in self._flt.items():
            self._flt[flt] = filters.pop(flt, val)
        if filters:
            raise error.UnknownFiltersError(filters.keys())
        return self

    def __iter__(self):
        return (self._col[i] for i in range(*self._idx)
                if self._pass(self._col[i]))


class StagyyData:

    """Generic lazy interface to StagYY output data."""

    def __init__(self, path, nfields_max=50):
        """Initialization of instances:

        Args:
            path (pathlike): path of the StagYY run. It can either be the path
                of the directory containing the par file, or the path of the
                par file. If the path given is a directory, the path of the par
                file is assumed to be path/par.
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
        runpath = pathlib.Path(path)
        if runpath.is_file():
            parname = runpath.name
            runpath = runpath.parent
        else:
            parname = 'par'
        self._rundir = {'path': runpath,
                        'par': parname,
                        'hdf5': UNDETERMINED,
                        'ls': UNDETERMINED}
        self._stagdat = {'par': parfile.readpar(self.parpath),
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
    def parpath(self):
        """Path of par file.

        :class:`pathlib.Path` instance.
        """
        return self.path / self._rundir['par']

    @property
    def hdf5(self):
        """Path of output hdf5 folder if relevant, None otherwise."""
        if self._rundir['hdf5'] is UNDETERMINED:
            h5_folder = self.path / self.par['ioin']['hdf5_output_folder']
            if (h5_folder / 'Data.xmf').is_file():
                self._rundir['hdf5'] = h5_folder
            else:
                self._rundir['hdf5'] = None
        return self._rundir['hdf5']

    @property
    def par(self):
        """Content of par file.

        This is a :class:`f90nml.namelist.Namelist`, the first key being
        namelists and the second key the parameter name.
        """
        return self._stagdat['par']

    @property
    def tseries(self):
        """Time series data.

        This is a :class:`pandas.DataFrame` with istep as index and variable
        names as columns.
        """
        if self._stagdat['tseries'] is UNDETERMINED:
            timefile = self.filename('TimeSeries.h5')
            self._stagdat['tseries'] = stagyyparsers.time_series_h5(
                timefile, list(phyvars.TIME.keys()))
            if self._stagdat['tseries'] is not None:
                return self._stagdat['tseries']
            timefile = self.filename('time.dat')
            if self.hdf5 and not timefile.is_file():
                # check legacy folder as well
                timefile = self.filename('time.dat', force_legacy=True)
            self._stagdat['tseries'] = stagyyparsers.time_series(
                timefile, list(phyvars.TIME.keys()))
        return self._stagdat['tseries']

    @property
    def _rprof_and_times(self):
        if self._stagdat['rprof'] is UNDETERMINED:
            rproffile = self.filename('rprof.h5')
            self._stagdat['rprof'] = stagyyparsers.rprof_h5(
                rproffile, list(phyvars.RPROF.keys()))
            if self._stagdat['rprof'][0] is not None:
                return self._stagdat['rprof']
            rproffile = self.filename('rprof.dat')
            if self.hdf5 and not rproffile.is_file():
                # check legacy folder as well
                rproffile = self.filename('time.dat', force_legacy=True)
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
            if out_dir.is_dir():
                self._rundir['ls'] = set(out_dir.iterdir())
            else:
                self._rundir['ls'] = set()
        return self._rundir['ls']

    @property
    def walk(self):
        """Return view on configured steps slice.

        Other Parameters:
            conf.core.snapshots: the slice of snapshots.
            conf.core.timesteps: the slice of timesteps.
        """
        if conf.core.snapshots is not None:
            return self.snaps[conf.core.snapshots]
        elif conf.core.timesteps is not None:
            return self.steps[conf.core.timesteps]
        return self.snaps[-1:]

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

    def filename(self, fname, timestep=None, suffix='', force_legacy=False):
        """Return name of StagYY output file.

        Args:
            fname (str): name stem.
            timestep (int): snapshot number, set to None if this is not
                relevant.
            suffix (str): optional suffix of file name.
            force_legacy (bool): force returning the legacy output path.
        Returns:
            :class:`pathlib.Path`: the path of the output file constructed
            with the provided segments.
        """
        if timestep is not None:
            fname += '{:05d}'.format(timestep)
        fname += suffix
        if not force_legacy and self.hdf5:
            fpath = self.hdf5 / fname
        else:
            fpath = self.par['ioin']['output_file_stem'] + '_' + fname
            fpath = self.path / fpath
        return fpath

    def binfiles_set(self, isnap):
        """Set of existing binary files at a given snap.

        Args:
            isnap (int): snapshot index.
        Returns:
            set of pathlib.Path: the set of output files available for this
            snapshot number.
        """
        possible_files = set(self.filename(fstem, isnap, force_legacy=True)
                             for fstem in phyvars.FIELD_FILES)
        return possible_files & self.files
