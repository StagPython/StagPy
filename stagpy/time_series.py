"""Plots time series."""

import numpy as np
import matplotlib.pyplot as plt

from . import conf, misc, phyvars
from .error import UnknownTimeVarError, InvalidTimeFractionError
from .stagyydata import StagyyData


def _plot_time_list(lovs, tseries, metas, times=None):
    """Plot requested profiles"""
    if times is None:
        times = {}
    for vfig in lovs:
        fig, axes = plt.subplots(nrows=len(vfig), sharex=True,
                                 figsize=(30, 5 * len(vfig)))
        axes = [axes] if len(vfig) == 1 else axes
        fname = ['time']
        for iplt, vplt in enumerate(vfig):
            ylabel = None
            for tvar in vplt:
                fname.append(tvar)
                time = times[tvar] if tvar in times else tseries['t']
                axes[iplt].plot(time, tseries[tvar],
                                conf.time.style,
                                label=metas[tvar].description,
                                linewidth=conf.plot.linewidth)
                lbl = metas[tvar].shortname
                if ylabel is None:
                    ylabel = lbl
                elif ylabel != lbl:
                    ylabel = ''
            if ylabel:
                axes[iplt].set_ylabel(r'${}$'.format(ylabel),
                                      fontsize=conf.plot.fontsize)
            if vplt[0][:3] == 'eta':  # list of log variables
                axes[iplt].set_yscale('log')
            axes[iplt].legend(fontsize=conf.plot.fontsize)
            axes[iplt].tick_params(labelsize=conf.plot.fontsize)
        axes[-1].set_xlabel(r'$t$', fontsize=conf.plot.fontsize)
        axes[-1].set_xlim((tseries['t'].iloc[0], tseries['t'].iloc[-1]))
        axes[-1].tick_params(labelsize=conf.plot.fontsize)
        misc.saveplot(fig, '_'.join(fname))


def get_time_series(sdat, var, tstart, tend):
    """Extract or compute a time series.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        var (str): time series name, a key of :data:`stagpy.phyvars.TIME`
            or :data:`stagpy.phyvars.TIME_EXTRA`.
        tstart (float): starting time of desired series. Set to None to start
            at the beginning of available data.
        tend (float): ending time of desired series. Set to None to stop at the
            end of available data.
    Returns:
        tuple of :class:`numpy.array` and :class:`stagpy.phyvars.Vart`:
        series, time, meta
            series is the requested time series, time the time at which it
            is evaluated (set to None if it is the one of time series output
            by StagYY), and meta is a :class:`stagpy.phyvars.Vart` instance
            holding metadata of the requested variable.
    """
    tseries = sdat.tseries_between(tstart, tend)
    if var in tseries.columns:
        series = tseries[var]
        time = None
        if var in phyvars.TIME:
            meta = phyvars.TIME[var]
        else:
            meta = phyvars.Varr(None, None)
    elif var in phyvars.TIME_EXTRA:
        meta = phyvars.TIME_EXTRA[var]
        series, time = meta.description(sdat, tstart, tend)
        meta = phyvars.Vart(misc.baredoc(meta.description), meta.shortname)
    else:
        raise UnknownTimeVarError(var)

    return series, time, meta


def plot_time_series(sdat, lovs):
    """Plot requested time series.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        lovs (nested list of str): nested list of series names such as
            the one produced by :func:`stagpy.misc.list_of_vars`.

    Other Parameters:
        conf.time.tstart: the starting time.
        conf.time.tend: the ending time.
    """
    sovs = misc.set_of_vars(lovs)
    tseries = {}
    times = {}
    metas = {}
    for tvar in sovs:
        series, time, meta = get_time_series(
            sdat, tvar, conf.time.tstart, conf.time.tend)
        tseries[tvar] = series
        metas[tvar] = meta
        if time is not None:
            times[tvar] = time
    tseries['t'] = get_time_series(
        sdat, 't', conf.time.tstart, conf.time.tend)[0]

    _plot_time_list(lovs, tseries, metas, times)


def compstat(sdat, tstart=None, tend=None):
    """Compute statistics from series output by StagYY.

    Create a file 'statistics.dat' containing the mean and standard deviation
    of each series on the requested time span.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        tstart (float): starting time. Set to None to start at the beginning of
            available data.
        tend (float): ending time. Set to None to stop at the end of available
            data.
    """
    data = sdat.tseries_between(tstart, tend)
    time = data['t'].values
    delta_time = time[-1] - time[0]
    data = data.iloc[:, 1:].values  # assume t is first column

    mean = np.trapz(data, x=time, axis=0) / delta_time
    rms = np.sqrt(np.trapz((data - mean)**2, x=time, axis=0) / delta_time)

    with open(misc.out_name('statistics.dat'), 'w') as out_file:
        mean.tofile(out_file, sep=' ', format="%10.5e")
        out_file.write('\n')
        rms.tofile(out_file, sep=' ', format="%10.5e")
        out_file.write('\n')


def cmd():
    """Implementation of time subcommand.

    Other Parameters:
        conf.time
        conf.core
    """
    sdat = StagyyData(conf.core.path)
    if sdat.tseries is None:
        return

    if conf.time.fraction is not None:
        if not 0 < conf.time.fraction <= 1:
            raise InvalidTimeFractionError(conf.time.fraction)
        conf.time.tend = None
        t_0 = sdat.tseries.iloc[0].loc['t']
        t_f = sdat.tseries.iloc[-1].loc['t']
        conf.time.tstart = (t_0 * conf.time.fraction
                            + t_f * (1 - conf.time.fraction))

    lovs = misc.list_of_vars(conf.time.plot)
    if lovs:
        plot_time_series(sdat, lovs)

    if conf.time.compstat:
        compstat(sdat, conf.time.tstart, conf.time.tend)
