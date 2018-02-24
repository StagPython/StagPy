"""Plot radial profiles."""

import matplotlib.pyplot as plt

from . import conf, misc, phyvars
from .error import UnknownRprofVarError
from .stagyydata import StagyyData


def _plot_rprof_list(lovs, rprofs, metas, stepstr, rads=None):
    """Plot requested profiles"""
    if rads is None:
        rads = {}
    for vfig in lovs:
        fig, axes = plt.subplots(ncols=len(vfig), sharey=True)
        axes = [axes] if len(vfig) == 1 else axes
        fname = 'rprof_'
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            for rvar in vplt:
                fname += rvar + '_'
                rad = rads[rvar] if rvar in rads else rprofs['r']
                axes[iplt].plot(rprofs[rvar], rad,
                                conf.rprof.style,
                                label=metas[rvar].description)
                if xlabel is None:
                    xlabel = metas[rvar].shortname
                elif xlabel != metas[rvar].shortname:
                    xlabel = ''
            if xlabel:
                axes[iplt].set_xlabel(r'${}$'.format(xlabel))
            if vplt[0][:3] == 'eta':  # list of log variables
                axes[iplt].set_xscale('log')
            axes[iplt].legend()
        axes[0].set_ylabel(r'$r$')
        misc.saveplot(fig, fname + stepstr)


def get_rprof(step, var):
    """Extract or compute a radial profile.

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
            meta = phyvars.Varr(None, None)
    elif var in phyvars.RPROF_EXTRA:
        meta = phyvars.RPROF_EXTRA[var]
        rprof, rad = meta.description(step)
        meta = phyvars.Varr(misc.baredoc(meta.description), meta.shortname)
    else:
        raise UnknownRprofVarError(var)

    return rprof, rad, meta


def plot_grid(step):
    """Plot cell position and thickness.

    The figure is call grid_N.pdf where N is replace by the step index.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    """
    rad = step.rprof['r']
    drad, _, _ = get_rprof(step, 'dr')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(rad, '-ko')
    ax1.set_ylabel('$r$')
    ax2.plot(drad, '-ko')
    ax2.set_ylabel('$dr$')
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
    rprof_averaged['r'] = step.rprof.loc[:, 'r'] + \
        misc.get_rbounds(step)[0]

    stepstr = '{}_{}'.format(istart, ilast)

    _plot_rprof_list(lovs, rprof_averaged, metas, stepstr, rads)


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
        rprofs['r'] = step.rprof.loc[:, 'r'] + misc.get_rbounds(step)[0]
        stepstr = str(step.istep)

        _plot_rprof_list(lovs, rprofs, metas, stepstr, rads)


def cmd():
    """Implementation of rprof subcommand.

    Other Parameters:
        conf.rprof
        conf.core
    """
    sdat = StagyyData(conf.core.path)
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
