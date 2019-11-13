"""Plot radial profiles."""

import matplotlib.pyplot as plt

from . import conf, misc, phyvars
from .error import UnknownRprofVarError
from .stagyydata import StagyyData


def _plot_rprof_list(sdat, lovs, rprofs, metas, stepstr, rads=None):
    """Plot requested profiles."""
    if rads is None:
        rads = {}
    for vfig in lovs:
        fig, axes = plt.subplots(ncols=len(vfig), sharey=True,
                                 figsize=(4 * len(vfig), 6))
        axes = [axes] if len(vfig) == 1 else axes
        fname = 'rprof_'
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            for ivar, rvar in enumerate(vplt):
                fname += rvar + '_'
                rad = rads[rvar] if rvar in rads else rprofs['r']
                if conf.rprof.depth:
                    rad = rprofs['bounds'][1] - rad
                axes[iplt].plot(rprofs[rvar], rad,
                                conf.rprof.style,
                                label=metas[rvar].description)
                if conf.rprof.depth:
                    axes[iplt].invert_yaxis()
                if xlabel is None:
                    xlabel = metas[rvar].kind
                elif xlabel != metas[rvar].kind:
                    xlabel = ''
            if ivar == 0:
                xlabel = metas[rvar].description
            if xlabel:
                _, unit = sdat.scale(1, metas[rvar].dim)
                if unit:
                    xlabel += ' ({})'.format(unit)
                axes[iplt].set_xlabel(xlabel)
            if vplt[0][:3] == 'eta':  # list of log variables
                axes[iplt].set_xscale('log')
            axes[iplt].set_xlim(left=conf.plot.vmin, right=conf.plot.vmax)
            if ivar:
                axes[iplt].legend()
        ylabel = 'Depth' if conf.rprof.depth else 'Radius'
        _, unit = sdat.scale(1, 'm')
        if unit:
            ylabel += ' ({})'.format(unit)
        axes[0].set_ylabel(ylabel)
        misc.saveplot(fig, fname + stepstr)


def get_rprof(step, var):
    """Extract or compute and rescale requested radial profile.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): radial profile name, a key of :data:`stagpy.phyvars.RPROF`
            or :data:`stagpy.phyvars.RPROF_EXTRA`.
    Returns:
        tuple of :class:`numpy.array` and :class:`stagpy.phyvars.Varr`:
        rprof, rad, meta
            rprof is the requested profile, rad the radial position at which it
            is evaluated (set to None if it is the position of profiles output
            by StagYY), and meta is a :class:`stagpy.phyvars.Varr` instance
            holding metadata of the requested variable.
    """
    if var in step.rprof.columns:
        rprof = step.rprof[var]
        rad = None
        if var in phyvars.RPROF:
            meta = phyvars.RPROF[var]
        else:
            meta = phyvars.Varr(var, None, '1')
    elif var in phyvars.RPROF_EXTRA:
        meta = phyvars.RPROF_EXTRA[var]
        rprof, rad = meta.description(step)
        meta = phyvars.Varr(misc.baredoc(meta.description),
                            meta.kind, meta.dim)
    else:
        raise UnknownRprofVarError(var)
    rprof, _ = step.sdat.scale(rprof, meta.dim)
    if rad is not None:
        rad, _ = step.sdat.scale(rad, 'm')

    return rprof, rad, meta


def plot_grid(step):
    """Plot cell position and thickness.

    The figure is call grid_N.pdf where N is replace by the step index.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    """
    rad = get_rprof(step, 'r')[0]
    drad = get_rprof(step, 'dr')[0]
    _, unit = step.sdat.scale(1, 'm')
    if unit:
        unit = ' ({})'.format(unit)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(rad, '-ko')
    ax1.set_ylabel('$r$' + unit)
    ax2.plot(drad, '-ko')
    ax2.set_ylabel('$dr$' + unit)
    ax2.set_xlim([-0.5, len(rad) - 0.5])
    ax2.set_xlabel('Cell number')
    misc.saveplot(fig, 'grid', step.istep)


def plot_average(sdat, lovs):
    """Plot time averaged profiles.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        lovs (nested list of str): nested list of profile names such as
            the one produced by :func:`stagpy.misc.list_of_vars`.

    Other Parameters:
        conf.core.snapshots: the slice of snapshots.
        conf.conf.timesteps: the slice of timesteps.
    """
    steps_iter = iter(sdat.walk.filter(rprof=True))
    try:
        step = next(steps_iter)
    except StopIteration:
        return

    sovs = misc.set_of_vars(lovs)

    istart = step.istep
    nprofs = 1
    rprof_averaged = {}
    rads = {}
    metas = {}

    # assume constant z spacing for the moment
    for rvar in sovs:
        rprof_averaged[rvar], rad, metas[rvar] = get_rprof(step, rvar)
        if rad is not None:
            rads[rvar] = rad

    for step in steps_iter:
        nprofs += 1
        for rvar in sovs:
            rprof_averaged[rvar] += get_rprof(step, rvar)[0]

    ilast = step.istep
    for rvar in sovs:
        rprof_averaged[rvar] /= nprofs
    rcmb, rsurf = misc.get_rbounds(step)
    rprof_averaged['bounds'] = (step.sdat.scale(rcmb, 'm')[0],
                                step.sdat.scale(rsurf, 'm')[0])
    rprof_averaged['r'] = get_rprof(step, 'r')[0] + rprof_averaged['bounds'][0]

    stepstr = '{}_{}'.format(istart, ilast)

    _plot_rprof_list(sdat, lovs, rprof_averaged, metas, stepstr, rads)


def plot_every_step(sdat, lovs):
    """Plot profiles at each time step.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        lovs (nested list of str): nested list of profile names such as
            the one produced by :func:`stagpy.misc.list_of_vars`.

    Other Parameters:
        conf.core.snapshots: the slice of snapshots.
        conf.conf.timesteps: the slice of timesteps.
    """
    sovs = misc.set_of_vars(lovs)

    for step in sdat.walk.filter(rprof=True):
        rprofs = {}
        rads = {}
        metas = {}
        for rvar in sovs:
            rprof, rad, meta = get_rprof(step, rvar)
            rprofs[rvar] = rprof
            metas[rvar] = meta
            if rad is not None:
                rads[rvar] = rad
        rprofs['bounds'] = misc.get_rbounds(step)
        rcmb, rsurf = misc.get_rbounds(step)
        rprofs['bounds'] = (step.sdat.scale(rcmb, 'm')[0],
                            step.sdat.scale(rsurf, 'm')[0])
        rprofs['r'] = get_rprof(step, 'r')[0] + rprofs['bounds'][0]
        stepstr = str(step.istep)

        _plot_rprof_list(sdat, lovs, rprofs, metas, stepstr, rads)


def cmd():
    """Implementation of rprof subcommand.

    Other Parameters:
        conf.rprof
        conf.core
    """
    sdat = StagyyData()
    if sdat.rprof is None:
        return

    if conf.rprof.grid:
        for step in sdat.walk.filter(rprof=True):
            plot_grid(step)

    lovs = misc.list_of_vars(conf.rprof.plot)
    if not lovs:
        return

    if conf.rprof.average:
        plot_average(sdat, lovs)
    else:
        plot_every_step(sdat, lovs)
