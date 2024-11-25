"""Plot radial profiles."""

from __future__ import annotations

import typing

import matplotlib.pyplot as plt

from . import _helpers
from .config import Config
from .stagyydata import _sdat_from_conf

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence

    from .step import Rprofs, Step


def plot_rprofs(
    rprofs: Rprofs,
    names: Sequence[Sequence[Sequence[str]]],
    conf: Optional[Config] = None,
) -> None:
    """Plot requested radial profiles.

    Args:
        rprofs: a radial profile collection, such as `Step.rprofs` or
            [`StepsView.rprofs_averaged`][stagpy.stagyydata.StepsView.rprofs_averaged].
        names: profile names organized by figures, plots and subplots.
        conf: configuration.
    """
    if conf is None:
        conf = Config.default_()
    stepstr = rprofs.stepstr

    for vfig in names:
        fig, axes = plt.subplots(
            ncols=len(vfig), sharey=True, figsize=(4 * len(vfig), 6)
        )
        axes = [axes] if len(vfig) == 1 else axes
        fname = "rprof_"
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            profs_on_plt = (rprofs[rvar] for rvar in vplt)
            fname += "_".join(vplt) + "_"
            for ivar, rpf in enumerate(profs_on_plt):
                rprof = rpf.values
                rad = rpf.rad
                meta = rpf.meta
                if conf.rprof.depth:
                    rad = rprofs.bounds[1] - rad
                axes[iplt].plot(rprof, rad, conf.rprof.style, label=meta.description)
                if xlabel is None:
                    xlabel = meta.kind
                elif xlabel != meta.kind:
                    xlabel = ""
            if ivar == 0:
                xlabel = meta.description
            if xlabel:
                axes[iplt].set_xlabel(xlabel)
            if vplt[0][:3] == "eta":  # list of log variables
                axes[iplt].set_xscale("log")
            axes[iplt].set_xlim(left=conf.plot.vmin, right=conf.plot.vmax)
            if ivar:
                axes[iplt].legend()
        if conf.rprof.depth:
            axes[0].invert_yaxis()
        ylabel = "Depth" if conf.rprof.depth else "Radius"
        axes[0].set_ylabel(ylabel)
        _helpers.saveplot(conf, fig, fname + stepstr)


def plot_grid(step: Step, conf: Optional[Config] = None) -> None:
    """Plot cell position and thickness.

    The figure is call grid_N.pdf where N is replace by the step index.

    Args:
        step: a `Step` of a `StagyyData` instance.
        conf: configuration.
    """
    if conf is None:
        conf = Config.default_()
    drprof = step.rprofs["dr"]
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(drprof.rad, "-ko")
    ax1.set_ylabel("$r$")
    ax2.plot(drprof.values, "-ko")
    ax2.set_ylabel("$dr$")
    ax2.set_xlim([-0.5, len(drprof.rad) - 0.5])
    ax2.set_xlabel("Cell number")
    _helpers.saveplot(conf, fig, "grid", step.istep)


def cmd(conf: Config) -> None:
    """Implementation of rprof subcommand."""
    sdat = _sdat_from_conf(conf.core)
    view = _helpers.walk(sdat, conf)

    if conf.rprof.grid:
        for step in view.filter(rprofs=True):
            plot_grid(step, conf)

    if conf.rprof.average:
        plot_rprofs(view.rprofs_averaged, conf.rprof.plot, conf)
    else:
        for step in view.filter(rprofs=True):
            plot_rprofs(step.rprofs, conf.rprof.plot, conf)
