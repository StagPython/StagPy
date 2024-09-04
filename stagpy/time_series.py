"""Plots time series."""

from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from . import _helpers
from .config import Config
from .error import InvalidTimeFractionError
from .stagyydata import StagyyData, _sdat_from_conf

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence

    from pandas import DataFrame


def _collect_marks(sdat: StagyyData, conf: Config) -> list[float]:
    """Concatenate `mark*` config variable."""
    times = list(conf.time.marktimes)
    times.extend(step.timeinfo["t"] for step in sdat.snaps[conf.time.marksnaps])
    times.extend(step.timeinfo["t"] for step in sdat.steps[conf.time.marksteps])
    return times


def plot_time_series(
    sdat: StagyyData,
    names: Sequence[Sequence[Sequence[str]]],
    conf: Optional[Config] = None,
) -> None:
    """Plot requested time series.

    Args:
        sdat: a `StagyyData` instance.
        names: time series names organized by figures, plots and subplots.
        conf: configuration.
    """
    if conf is None:
        conf = Config.default_()
    time_marks = _collect_marks(sdat, conf)
    for vfig in names:
        tstart = conf.time.tstart
        tend = conf.time.tend
        fig, axes = plt.subplots(
            nrows=len(vfig), sharex=True, figsize=(12, 2 * len(vfig))
        )
        axes = [axes] if len(vfig) == 1 else axes
        fname = ["time"]
        for iplt, vplt in enumerate(vfig):
            ylabel = None
            series_on_plt = [
                sdat.tseries.tslice(tvar, conf.time.tstart, conf.time.tend)
                for tvar in vplt
            ]
            ptstart = min(series.time[0] for series in series_on_plt)
            ptend = max(series.time[-1] for series in series_on_plt)
            tstart = ptstart if tstart is None else min(ptstart, tstart)
            tend = ptend if tend is None else max(ptend, tend)
            fname.extend(vplt)
            for ivar, tseries in enumerate(series_on_plt):
                axes[iplt].plot(
                    tseries.time,
                    tseries.values,
                    conf.time.style,
                    label=tseries.meta.description,
                )
                lbl = tseries.meta.kind
                if ylabel is None:
                    ylabel = lbl
                elif ylabel != lbl:
                    ylabel = ""
            if ivar == 0:
                ylabel = tseries.meta.description
            if ylabel:
                axes[iplt].set_ylabel(ylabel)
            if vplt[0][:3] == "eta":  # list of log variables
                axes[iplt].set_yscale("log")
            axes[iplt].set_ylim(bottom=conf.plot.vmin, top=conf.plot.vmax)
            if ivar:
                axes[iplt].legend()
            axes[iplt].tick_params()
            for time_mark in time_marks:
                axes[iplt].axvline(time_mark, color="black", linestyle="--")
        axes[-1].set_xlabel("Time")
        axes[-1].set_xlim(tstart, tend)
        axes[-1].tick_params()
        _helpers.saveplot(conf, fig, "_".join(fname))


def compstat(
    sdat: StagyyData,
    *names: str,
    tstart: Optional[float] = None,
    tend: Optional[float] = None,
) -> DataFrame:
    """Compute statistics from series output by StagYY.

    Args:
        sdat: a `StagyyData` instance.
        names: variables whose statistics should be computed.
        tstart: starting time. Set to None to start at the beginning of
            available data.
        tend: ending time. Set to None to stop at the end of available data.

    Returns:
        statistics. `"mean"` and `"rms"` as index, variable names as columns.
    """
    stats = pd.DataFrame(columns=names, index=["mean", "rms"])
    for name in names:
        series = sdat.tseries.tslice(name, tstart, tend)
        delta_time = series.time[-1] - series.time[0]
        mean = trapezoid(series.values, x=series.time) / delta_time
        stats.loc["mean", name] = mean
        stats.loc["rms", name] = np.sqrt(
            trapezoid((series.values - mean) ** 2, x=series.time) / delta_time
        )
    return stats


def cmd(conf: Config) -> None:
    """Implementation of time subcommand."""
    sdat = _sdat_from_conf(conf.core)
    if sdat.tseries is None:
        return

    if conf.time.fraction is not None:
        if not 0 < conf.time.fraction <= 1:
            raise InvalidTimeFractionError(conf.time.fraction)
        conf.time.tend = None
        t_0 = sdat.tseries.time[0]
        t_f = sdat.tseries.time[-1]
        conf.time.tstart = t_0 * conf.time.fraction + t_f * (1 - conf.time.fraction)

    plot_time_series(sdat, conf.time.plot, conf)

    if conf.time.compstat:
        stats = compstat(
            sdat, *conf.time.compstat, tstart=conf.time.tstart, tend=conf.time.tend
        )
        stats.to_csv("statistics.dat")
