"""Define high level structure StagyyData and helper classes.

Note:
    The helper classes are not designed to be instantiated on their own, but
    only as attributes of StagyyData instances. Users of this module should
    only instantiate :class:`StagyyData`.

"""

import re
import pathlib
from collections import namedtuple
from itertools import zip_longest

import numpy as np

from . import conf, error, misc, parfile, phyvars, stagyyparsers, _step
from .misc import CachedReadOnlyProperty as crop


def _as_view_item(obj):
    """Return None or a suitable iterable to build a _StepsView."""
    try:
        iter(obj)
        return obj
    except TypeError:
        pass
    if isinstance(obj, slice):
        return (obj,)


class _Scales:
    """Dimensionful scales.

    Args:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Scales` instance.
    """

    def __init__(self, sdat):
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


class _Refstate:
    """Reference state profiles.

    The :attr:`StagyyData.refstate` attribute is an instance of this class.
    Reference state profiles are accessed through the attributes of this
    object.

    Args:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Steps` instance.
    """

    def __init__(self, sdat):
        self._sdat = sdat

    @crop
    def _data(self):
        """Read reference state profile."""
        reffile = self._sdat.filename('refstat.dat')
        if self._sdat.hdf5 and not reffile.is_file():
            # check legacy folder as well
            reffile = self._sdat.filename('refstat.dat', force_legacy=True)
        return stagyyparsers.refstate(reffile)

    @property
    def systems(self):
        """Reference state profiles of phases.

        It is a list of list of :class:`pandas.DataFrame` containing
        the reference profiles associated with each phase in each system.

        Example:
            The temperature profile of the 3rd phase in the 1st
            system is

            >>> sdat.refstate.systems[0][2]['T']
        """
        return self._data[0]

    @property
    def adiabats(self):
        """Adiabatic reference state profiles.

        It is a list of :class:`pandas.DataFrame` containing the reference
        profiles associated with each system.  The last item is the combined
        adiabat.

        Example:
            The adiabatic temperature profile of the 2nd system is

            >>> sdat.refstate.adiabats[1]['T']

            The combined density profile is

            >>> sdat.refstate.adiabats[-1]['rho']
        """
        return self._data[1]


Tseries = namedtuple('Tseries', ['values', 'time', 'meta'])


class _Tseries:
    """Time series.

    The :attr:`StagyyData.tseries` attribute is an instance of this class.

    :class:`_Tseries` implements the getitem mechanism.  Keys are series names
    defined in :data:`stagpy.phyvars.TIME[_EXTRA]`.  An item is a named tuple
    ('values', 'time', 'meta'), respectively the series itself, the time at
    which it is evaluated, and meta is a :class:`stagpy.phyvars.Vart` instance
    with relevant metadata.  Note that series are automatically scaled if
    conf.scaling.dimensional is True.

    Attributes:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Tseries` instance.
    """

    def __init__(self, sdat):
        self.sdat = sdat
        self._cached_extra = {}

    @crop
    def _data(self):
        timefile = self.sdat.filename('TimeSeries.h5')
        data = stagyyparsers.time_series_h5(
            timefile, list(phyvars.TIME.keys()))
        if data is not None:
            return data
        timefile = self.sdat.filename('time.dat')
        if self.sdat.hdf5 and not timefile.is_file():
            # check legacy folder as well
            timefile = self.sdat.filename('time.dat', force_legacy=True)
        data = stagyyparsers.time_series(timefile, list(phyvars.TIME.keys()))
        return data

    @property
    def _tseries(self):
        if self._data is None:
            raise error.MissingDataError(f'No tseries data in {self.sdat}')
        return self._data

    def __getitem__(self, name):
        if name in self._tseries.columns:
            series = self._tseries[name].values
            time = self.time
            if name in phyvars.TIME:
                meta = phyvars.TIME[name]
            else:
                meta = phyvars.Vart(name, None, '1')
        elif name in self._cached_extra:
            series, time, meta = self._cached_extra[name]
        elif name in phyvars.TIME_EXTRA:
            meta = phyvars.TIME_EXTRA[name]
            series, time = meta.description(self.sdat)
            meta = phyvars.Vart(misc.baredoc(meta.description),
                                meta.kind, meta.dim)
            self._cached_extra[name] = series, time, meta
        else:
            raise error.UnknownTimeVarError(name)
        series, _ = self.sdat.scale(series, meta.dim)
        time, _ = self.sdat.scale(time, 's')
        return Tseries(series, time, meta)

    def tslice(self, name, tstart=None, tend=None):
        """Return a Tseries between specified times.

        Args:
            name (str): time variable.
            tstart (float): starting time. Set to None to start at the
                beginning of available data.
            tend (float): ending time. Set to None to stop at the end of
                available data.
        """
        data, time, meta = self[name]
        istart = 0
        iend = len(time)
        if tstart is not None:
            istart = misc.find_in_sorted_arr(tstart, time)
        if tend is not None:
            iend = misc.find_in_sorted_arr(tend, time, True) + 1
        return Tseries(data[istart:iend], time[istart:iend], meta)

    @property
    def time(self):
        """Time vector."""
        return self._tseries['t'].values

    @property
    def isteps(self):
        """Step indices.

        This is such that time[istep] is at step isteps[istep].
        """
        return self._tseries.index.values

    def at_step(self, istep):
        """Time series output for a given step."""
        return self._tseries.loc[istep]


class _RprofsAveraged(_step._Rprofs):
    """Radial profiles time-averaged over a :class:`_StepsView`.

    The :attr:`_StepsView.rprofs_averaged` attribute is an instance of this
    class.

    It implements the same interface as :class:`~stagpy._step._Rprofs` but
    returns time-averaged profiles instead.

    Attributes:
        steps (:class:`_StepsView`): the object owning the
            :class:`_RprofsAveraged` instance
    """

    def __init__(self, steps):
        self.steps = steps.filter(rprofs=True)
        self._cached_data = {}
        super().__init__(next(iter(self.steps)))

    def __getitem__(self, name):
        # the averaging method has two shortcomings:
        # - does not take into account time changing geometry;
        # - does not take into account time changing timestep.
        if name in self._cached_data:
            return self._cached_data[name]
        steps_iter = iter(self.steps)
        rprof, rad, meta = next(steps_iter).rprofs[name]
        rprof = np.copy(rprof)
        nprofs = 1
        for step in steps_iter:
            nprofs += 1
            rprof += step.rprofs[name].values
        rprof /= nprofs
        self._cached_data[name] = _step.Rprof(rprof, rad, meta)
        return self._cached_data[name]


class _Steps:
    """Collections of time steps.

    The :attr:`StagyyData.steps` attribute is an instance of this class.
    Time steps (which are :class:`~stagpy._step.Step` instances) can be
    accessed with the item accessor::

        sdat = StagyyData('path/to/run')
        sdat.steps[istep]  # Step object of the istep-th time step

    Slices or tuple of istep and slices of :class:`_Steps` object are
    :class:`_StepsView` instances that you can iterate and filter::

        for step in steps[500:]:
            # iterate through all time steps from the 500-th one
            do_something(step)

        for step in steps[-100:].filter(snap=True):
            # iterate through all snapshots present in the last 100 time steps
            do_something(step)

        for step in steps[0,3,5,-2:]:
            # iterate through steps 0, 3, 5 and the last two
            do_something(step)

    Args:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Steps` instance.
    Attributes:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Steps` instance.
    """

    def __init__(self, sdat):
        self.sdat = sdat
        self._data = {}

    def __repr__(self):
        return f'{self.sdat!r}.steps'

    def __getitem__(self, istep):
        keys = _as_view_item(istep)
        if keys is not None:
            return _StepsView(self, keys)
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
                    f'Last istep is {len(self) - 1}')
        if istep not in self._data:
            self._data[istep] = _step.Step(istep, self.sdat)
        return self._data[istep]

    def __delitem__(self, istep):
        if istep is not None and istep in self._data:
            self.sdat._collected_fields = [
                (i, f) for i, f in self.sdat._collected_fields if i != istep]
            del self._data[istep]

    @crop
    def _len(self):
        # not necessarily the last one...
        return self.sdat.tseries.isteps[-1] + 1

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self[:])

    def at_time(self, time, after=False):
        """Return step corresponding to a given physical time.

        Args:
            time (float): the physical time requested.
            after (bool): when False (the default), the returned step is such
                that its time is immediately before the requested physical
                time. When True, the returned step is the next one instead (if
                it exists, otherwise the same step is returned).

        Returns:
            :class:`~stagpy._step.Step`: the relevant step.
        """
        itime = misc.find_in_sorted_arr(time, self.sdat.tseries.time, after)
        return self[self.sdat.tseries.isteps[itime]]

    def filter(self, **filters):
        """Build a _StepsView with requested filters."""
        return self[:].filter(**filters)


class _Snaps(_Steps):
    """Collections of snapshots.

    The :attr:`StagyyData.snaps` attribute is an instance of this class.
    Snapshots (which are :class:`~stagpy._step.Step` instances) can be accessed
    with the item accessor::

        sdat = StagyyData('path/to/run')
        sdat.snaps[isnap]  # Step object of the isnap-th snapshot

    This class inherits from :class:`_Steps`.

    Args:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Snaps` instance.
    Attributes:
        sdat (:class:`StagyyData`): the StagyyData instance owning the
            :class:`_Snaps` instance.
    """

    def __init__(self, sdat):
        self._isteps = {}
        self._all_isteps_known = False
        super().__init__(sdat)

    def __repr__(self):
        return f'{self.sdat!r}.snaps'

    def __getitem__(self, isnap):
        keys = _as_view_item(isnap)
        if keys is not None:
            return _StepsView(self, keys).filter(snap=True)
        if isnap < 0:
            isnap += len(self)
        if isnap < 0 or isnap >= len(self):
            istep = None
        else:
            istep = self._isteps.get(
                isnap, None if self._all_isteps_known else -1)
        if istep == -1:
            # isnap not in _isteps but not all isteps known, keep looking
            binfiles = self.sdat._binfiles_set(isnap)
            if binfiles:
                istep = stagyyparsers.fields(binfiles.pop(), only_istep=True)
            else:
                istep = None
            if istep is not None:
                self._bind(isnap, istep)
            else:
                self._isteps[isnap] = None
        if istep is None:
            raise error.InvalidSnapshotError(
                self.sdat, isnap, 'Invalid snapshot index')
        return self.sdat.steps[istep]

    def __delitem__(self, isnap):
        istep = self._isteps.get(isnap)
        del self.sdat.steps[istep]

    @crop
    def _len(self):
        length = -1
        if self.sdat.hdf5:
            isnap = -1
            for isnap, istep in stagyyparsers.read_time_h5(self.sdat.hdf5):
                self._bind(isnap, istep)
            length = isnap
            self._all_isteps_known = True
        if length < 0:
            out_stem = re.escape(pathlib.Path(
                self.sdat.par['ioin']['output_file_stem'] + '_').name[:-1])
            rgx = re.compile(f'^{out_stem}_([a-zA-Z]+)([0-9]{{5}})$')
            fstems = set(fstem for fstem in phyvars.FIELD_FILES)
            for fname in self.sdat._files:
                match = rgx.match(fname.name)
                if match is not None and match.group(1) in fstems:
                    length = max(int(match.group(2)), length)
        if length < 0:
            raise error.NoSnapshotError(self.sdat)
        return length + 1

    def at_time(self, time, after=False):
        """Return snap corresponding to a given physical time.

        Args:
            time (float): the physical time requested.
            after (bool): when False (the default), the returned snap is such
                that its time is immediately before the requested physical
                time. When True, the returned snap is the next one instead (if
                it exists, otherwise the same snap is returned).

        Returns:
            :class:`~stagpy._step.Step`: the relevant snap.
        """
        # in theory, this could be a valid implementation of _Steps.at_time
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

    def _bind(self, isnap, istep):
        """Register the isnap / istep correspondence.

        Args:
            isnap (int): snapshot index.
            istep (int): time step index.
        """
        self._isteps[isnap] = istep
        self.sdat.steps[istep]._isnap = isnap


class _StepsView:
    """Filtered iterator over steps or snaps.

    Instances of this class are returned when taking slices of
    :attr:`StagyyData.steps` or :attr:`StagyyData.snaps` attributes.

    Args:
        steps_col (:class:`_Steps` or :class:`_Snaps`): steps collection,
            i.e. :attr:`StagyyData.steps` or :attr:`StagyyData.snaps`
            attributes.
        items (iterable): iterable of isteps/isnaps or slices.
    """

    def __init__(self, steps_col, items):
        self._col = steps_col
        self._items = items
        self._rprofs_averaged = None
        self._flt = {
            'snap': False,
            'rprofs': False,
            'fields': [],
            'func': lambda _: True,
        }
        self._dflt_func = self._flt['func']

    @property
    def rprofs_averaged(self):
        """Time-averaged radial profiles."""
        if self._rprofs_averaged is None:
            self._rprofs_averaged = _RprofsAveraged(self)
        return self._rprofs_averaged

    @crop
    def stepstr(self):
        """String representation of the requested set of steps."""
        items = []
        no_slice = True
        for item in self._items:
            if isinstance(item, slice):
                items.append('{}:{}:{}'.format(*item.indices(len(self._col))))
                no_slice = False
            else:
                items.append(repr(item))
        item_str = ','.join(items)
        if no_slice and len(items) == 1:
            item_str += ','
        colstr = repr(self._col).rsplit('.', maxsplit=1)[-1]
        return f'{colstr}[{item_str}]'

    def __repr__(self):
        rep = f'{self._col.sdat!r}.{self.stepstr}'
        flts = []
        for flt in ('snap', 'rprofs', 'fields'):
            if self._flt[flt]:
                flts.append(f'{flt}={self._flt[flt]!r}')
        if self._flt['func'] is not self._dflt_func:
            flts.append(f"func={self._flt['func']!r}")
        if flts:
            rep += '.filter({})'.format(', '.join(flts))
        return rep

    def _pass(self, item):
        """Check whether an item passes the filters."""
        try:
            step = self._col[item]
        except KeyError:
            return False
        okf = True
        okf = okf and (not self._flt['snap'] or step.isnap is not None)
        if self._flt['rprofs']:
            try:
                _ = step.rprofs.centers
            except error.MissingDataError:
                return False
        okf = okf and all(f in step.fields for f in self._flt['fields'])
        okf = okf and bool(self._flt['func'](step))
        return okf

    def filter(self, **filters):
        """Update filters with provided arguments.

        Note that filters are only resolved when the view is iterated, and
        hence they do not compose. Each call to filter merely updates the
        relevant filters. For example, with this code::

            view = sdat.steps[500:].filter(rprofs=True, fields=['T'])
            view.filter(fields=[])

        the produced ``view``, when iterated, will generate the steps after the
        500-th that have radial profiles. The ``fields`` filter set in the
        first line is emptied in the second line.

        Args:
            snap (bool): the step must be a snapshot to pass.
            rprofs (bool): the step must have rprofs data to pass.
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
        for item in self._items:
            if isinstance(item, slice):
                idx = item.indices(len(self._col))
                yield from (self._col[i] for i in range(*idx)
                            if self._pass(i))
            elif self._pass(item):
                yield self._col[item]

    def __eq__(self, other):
        return all(s1 is s2 for s1, s2 in zip_longest(self, other))


class StagyyData:
    """Generic lazy interface to StagYY output data.

    Args:
        path (pathlike): path of the StagYY run. It can either be the path
            of the directory containing the par file, or the path of the
            par file. If the path given is a directory, the path of the par
            file is assumed to be path/par.  If no path is given (or None)
            it is set to ``conf.core.path``.

    Other Parameters:
        conf.core.path: the default path.

    Attributes:
        steps (:class:`_Steps`): collection of time steps.
        snaps (:class:`_Snaps`): collection of snapshots.
        scales (:class:`_Scales`): dimensionful scaling factors.
        refstate (:class:`_Refstate`): reference state profiles.
    """

    def __init__(self, path=None):
        if path is None:
            path = conf.core.path
        runpath = pathlib.Path(path)
        if runpath.is_file():
            parname = runpath.name
            runpath = runpath.parent
        else:
            parname = 'par'
        self._rundir = {'path': runpath,
                        'par': parname}
        self._par = parfile.readpar(self.parpath, self.path)
        self.scales = _Scales(self)
        self.refstate = _Refstate(self)
        self.tseries = _Tseries(self)
        self.steps = _Steps(self)
        self.snaps = _Snaps(self)
        self._nfields_max = 50
        # list of (istep, field_name) in memory
        self._collected_fields = []

    def __repr__(self):
        return f'StagyyData({self.path!r})'

    def __str__(self):
        return f'StagyyData in {self.path}'

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

    @crop
    def hdf5(self):
        """Path of output hdf5 folder if relevant, None otherwise."""
        h5_folder = self.path / self.par['ioin']['hdf5_output_folder']
        if (h5_folder / 'Data.xmf').is_file():
            return h5_folder

    @property
    def par(self):
        """Content of par file.

        This is a :class:`f90nml.namelist.Namelist`, the first key being
        namelists and the second key the parameter name.
        """
        return self._par

    @crop
    def _rprof_and_times(self):
        rproffile = self.filename('rprof.h5')
        data = stagyyparsers.rprof_h5(rproffile, list(phyvars.RPROF.keys()))
        if data[1] is not None:
            return data
        rproffile = self.filename('rprof.dat')
        if self.hdf5 and not rproffile.is_file():
            # check legacy folder as well
            rproffile = self.filename('rprof.dat', force_legacy=True)
        return stagyyparsers.rprof(rproffile, list(phyvars.RPROF.keys()))

    @property
    def rtimes(self):
        """Radial profiles times.

        This is a :class:`pandas.DataFrame` with istep as index.
        """
        return self._rprof_and_times[1]

    @crop
    def _files(self):
        """Set of found binary files output by StagYY."""
        out_stem = pathlib.Path(self.par['ioin']['output_file_stem'] + '_')
        out_dir = self.path / out_stem.parent
        if out_dir.is_dir():
            return set(out_dir.iterdir())
        return set()

    @property
    def walk(self):
        """Return view on configured steps slice.

        Other Parameters:
            conf.core.snapshots: the slice of snapshots.
            conf.core.timesteps: the slice of timesteps.
        """
        if conf.core.snapshots:
            return self.snaps[conf.core.snapshots]
        elif conf.core.timesteps:
            return self.steps[conf.core.timesteps]
        return self.snaps[-1, ]

    @property
    def nfields_max(self):
        """Maximum number of scalar fields kept in memory.

        Setting this to a value lower or equal to 5 raises a
        :class:`~stagpy.error.InvalidNfieldsError`.  Set this to ``None`` if
        you do not want any limit on the number of scalar fields kept in
        memory.  Defaults to 50.
        """
        return self._nfields_max

    @nfields_max.setter
    def nfields_max(self, nfields):
        """Check nfields > 5 or None."""
        if nfields is not None and nfields <= 5:
            raise error.InvalidNfieldsError(nfields)
        self._nfields_max = nfields

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
            fname += f'{timestep:05d}'
        fname += suffix
        if not force_legacy and self.hdf5:
            fpath = self.hdf5 / fname
        else:
            fpath = self.par['ioin']['output_file_stem'] + '_' + fname
            fpath = self.path / fpath
        return fpath

    def _binfiles_set(self, isnap):
        """Set of existing binary files at a given snap.

        Args:
            isnap (int): snapshot index.
        Returns:
            set of pathlib.Path: the set of output files available for this
            snapshot number.
        """
        possible_files = set(self.filename(fstem, isnap, force_legacy=True)
                             for fstem in phyvars.FIELD_FILES)
        return possible_files & self._files
