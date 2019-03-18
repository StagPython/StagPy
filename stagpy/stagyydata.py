"""Define high level structure StagyyData and helper classes.

Note:
    The helper classes are not designed to be instantiated on their own, but
    only as attributes of StagyyData instances. Users of this module should
    only instantiate :class:`StagyyData`.
"""

import re
import pathlib
from itertools import zip_longest

from . import conf, error, parfile, phyvars, stagyyparsers, _step
from ._step import UNDETERMINED


class _Scales:

    """Dimensionful scales."""

    def __init__(self, sdat):
        """Initialization of instances:

        Args:
            sdat (:class:`StagyyData`): the StagyyData instance owning the
                :class:`_Scales` instance.
        """
        self._sdat = sdat

    @property
    def length(self):
        """Length in m."""
        return self._sdat.par['geometry']['d_dimensional']

    @property
    def temperature(self):
        """Temperature in K."""
        return self._sdat.par['refstate']['deltaT_dimensional']

    @property
    def density(self):
        """Density in kg/m3."""
        return self._sdat.par['refstate']['dens_dimensional']

    @property
    def th_cond(self):
        """Thermal conductivity in W/(m.K)."""
        return self._sdat.par['refstate']['tcond_dimensional']

    @property
    def sp_heat(self):
        """Specific heat capacity in J/(kg.K)."""
        return self._sdat.par['refstate']['Cp_dimensional']

    @property
    def dyn_visc(self):
        """Dynamic viscosity in Pa.s."""
        return self._sdat.par['viscosity']['eta0']

    @property
    def th_diff(self):
        """Thermal diffusivity in m2/s."""
        return self.th_cond / (self.density * self.sp_heat)

    @property
    def time(self):
        """Time in s."""
        return self.length**2 / self.th_diff

    @property
    def power(self):
        """Power in W."""
        return self.th_cond * self.temperature * self.length

    @property
    def heat_flux(self):
        """Local heat flux in W/m2."""
        return self.power / self.length**2

    @property
    def stress(self):
        """Stress in Pa."""
        return self.dyn_visc / self.time


class _Steps:

    """Collections of time steps.

    The :attr:`StagyyData.steps` attribute is an instance of this class.
    Time steps (which are :class:`~stagpy._step.Step` instances) can be
    accessed with the item accessor::

        sdat = StagyyData('path/to/run')
        sdat.steps[istep]  # Step object of the istep-th time step

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
        self._data = {None: _step.EmptyStep()}  # for non existent snaps

    def __repr__(self):
        return '{}.steps'.format(repr(self.sdat))

    def __getitem__(self, istep):
        if istep is None:
            return self._data[None]
        try:
            return _StepsView(self, istep)
        except AttributeError:
            pass
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
        if istep not in self._data:
            self._data[istep] = _step.Step(istep, self.sdat)
        return self._data[istep]

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
    Snapshots (which are :class:`~stagpy._step.Step` instances) can be accessed
    with the item accessor::

        sdat = StagyyData('path/to/run')
        sdat.snaps[isnap]  # Step object of the isnap-th snapshot

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

    def __getitem__(self, isnap):
        try:
            return _StepsView(self, isnap)
        except AttributeError:
            pass
        if isnap < 0:
            isnap += len(self)
        if isnap < 0 or isnap >= len(self):
            istep = None
        else:
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
            else:
                self._isteps[isnap] = None
        return self.sdat.steps[istep]

    def __len__(self):
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
        return self._last + 1

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
        return self[len(self) - 1]


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
        """Check whether a :class:`~stagpy._step.Step` passes the filters."""
        okf = True
        okf = okf and (not self._flt['snap'] or step.isnap is not None)
        okf = okf and (not self._flt['rprof'] or step.rprof is not None)
        okf = okf and all(f in step.fields for f in self._flt['fields'])
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
            func (function): arbitrary function taking a
                :class:`~stagpy._step.Step` as argument and returning a True
                value if the step should pass the filter.

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

    def __eq__(self, other):
        return all(s1 is s2 for s1, s2 in zip_longest(self, other))


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
            scales (:class:`_Scales`): dimensionful scaling factors.
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
        self._stagdat = {'par': parfile.readpar(self.parpath, self.path),
                         'tseries': UNDETERMINED,
                         'rprof': UNDETERMINED}
        self.scales = _Scales(self)
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
                rproffile = self.filename('rprof.dat', force_legacy=True)
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

    def scale(self, data, unit):
        """Scales quantity to obtain dimensionful quantity.

        Args:
            data (numpy.array): the quantity that should be scaled.
            dim (str): the dimension of data as defined in phyvars.
        Return:
            (float, str): scaling factor and unit string.
        Other Parameters:
            conf.scaling.dimensional: if set to False (default), the factor is
                always 1.
        """
        if self.par['switches']['dimensional_units'] or \
           not conf.scaling.dimensional or \
           unit == '1':
            return data, ''
        scaling = phyvars.SCALES[unit](self.scales)
        factor = conf.scaling.factors.get(unit, ' ')
        if conf.scaling.time_in_y and unit == 's':
            scaling /= conf.scaling.yearins
            unit = 'yr'
        elif conf.scaling.vel_in_cmpy and unit == 'm/s':
            scaling *= 100 * conf.scaling.yearins
            unit = 'cm/y'
        if factor in phyvars.PREFIXES:
            scaling *= 10**(-3 * (phyvars.PREFIXES.index(factor) + 1))
            unit = factor + unit
        return data * scaling, unit

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
