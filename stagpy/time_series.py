"""Plots time series."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import conf, misc
from .error import InvalidTimeFractionError
from .stagyydata import StagyyData


def _collect_marks(sdat):
    """Concatenate mark* config variable."""
    times = set(conf.time.marktimes.replace(',', ' ').split())
    times = list(map(float, times))
    times.extend(step.timeinfo['t']
                 for step in sdat.snaps[conf.time.marksnaps])
    times.extend(step.timeinfo['t']
                 for step in sdat.steps[conf.time.marksteps])
    return times


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
    time_marks = _collect_marks(sdat)
    for vfig in lovs:
        tstart = conf.time.tstart
        tend = conf.time.tend
        fig, axes = plt.subplots(nrows=len(vfig), sharex=True,
                                 figsize=(12, 2 * len(vfig)))
        axes = [axes] if len(vfig) == 1 else axes
        fname = ['time']
        for iplt, vplt in enumerate(vfig):
            ylabel = None
            series_on_plt = [
                sdat.tseries.tslice(tvar, conf.time.tstart, conf.time.tend)
                for tvar in vplt]
            ptstart = min(time[0] for _, time, _ in series_on_plt)
            ptend = max(time[-1] for _, time, _ in series_on_plt)
            tstart = ptstart if tstart is None else min(ptstart, tstart)
            tend = ptend if tend is None else max(ptend, tend)
            fname.extend(vplt)
            for ivar, (series, time, meta) in enumerate(series_on_plt):
                axes[iplt].plot(time, series, conf.time.style,
                                label=meta.description)
                lbl = meta.kind
                if ylabel is None:
                    ylabel = lbl
                elif ylabel != lbl:
                    ylabel = ''
            if ivar == 0:
                ylabel = meta.description
            if ylabel:
                _, unit = sdat.scale(1, meta.dim)
                if unit:
                    ylabel += f' ({unit})'
                axes[iplt].set_ylabel(ylabel)
            if vplt[0][:3] == 'eta':  # list of log variables
                axes[iplt].set_yscale('log')
            axes[iplt].set_ylim(bottom=conf.plot.vmin, top=conf.plot.vmax)
            if ivar:
                axes[iplt].legend()
            axes[iplt].tick_params()
            for time_mark in time_marks:
                axes[iplt].axvline(time_mark, color='black', linestyle='--')
        _, unit = sdat.scale(1, 's')
        if unit:
            unit = f' ({unit})'
        axes[-1].set_xlabel('Time' + unit)
        axes[-1].set_xlim(tstart, tend)
        axes[-1].tick_params()
        misc.saveplot(fig, '_'.join(fname))


def compstat(sdat, *names, tstart=None, tend=None):
    """Compute statistics from series output by StagYY.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        names (str): variables whose statistics should be computed.
        tstart (float): starting time. Set to None to start at the beginning of
            available data.
        tend (float): ending time. Set to None to stop at the end of available
            data.
    Returns:
        :class:`pandas.DataFrame`: computed statistics with 'mean' and 'rms' as
            index and variable names as columns.
    """
    stats = pd.DataFrame(columns=names, index=['mean', 'rms'])
    for name in names:
        data, time, _ = sdat.tseries.tslice(name, tstart, tend)
        delta_time = time[-1] - time[0]
        mean = np.trapz(data, x=time) / delta_time
        stats.loc['mean', name] = mean
        stats.loc['rms', name] = np.sqrt(np.trapz((data - mean)**2, x=time) /
                                         delta_time)
    return stats


def cmd():
    """Implementation of time subcommand.

    Other Parameters:
        conf.time
        conf.core
    """
    sdat = StagyyData()
    if sdat.tseries is None:
        return

    if conf.time.fraction is not None:
        if not 0 < conf.time.fraction <= 1:
            raise InvalidTimeFractionError(conf.time.fraction)
        conf.time.tend = None
        t_0 = sdat.tseries.time[0]
        t_f = sdat.tseries.time[-1]
        conf.time.tstart = (t_0 * conf.time.fraction +
                            t_f * (1 - conf.time.fraction))

    lovs = misc.list_of_vars(conf.time.plot)
    if lovs:
        plot_time_series(sdat, lovs)

    if conf.time.compstat:
        names = conf.time.compstat.replace(',', ' ').split()
        stats = compstat(sdat, *names, tstart=conf.time.tstart,
                         tend=conf.time.tend)
        stats.to_csv('statistics.dat')
