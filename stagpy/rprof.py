"""Plots radial profiles coming out of stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/09/11
"""
from inspect import getdoc
from . import constants, misc
from .stagyydata import StagyyData


def _plot_rprof_list(lovs, rprofs, metas, args, stepstr, rads=None):
    """Plot requested profiles"""
    if rads is None:
        rads = {}
    for vfig in lovs:
        fig, axes = args.plt.subplots(ncols=len(vfig), sharey=True)
        axes = [axes] if len(vfig) == 1 else axes
        fname = ''
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            for rvar in vplt:
                fname += rvar + '_'
                rad = rads[rvar] if rvar in rads else rprofs['r']
                axes[iplt].plot(rprofs[rvar], rad,
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
        fig.savefig('{}{}.pdf'.format(fname, stepstr),
                    format='PDF', bbox_inches='tight')


def get_rprof(step, var):
    """Return read or computed rprof along with metadata"""
    if var in step.rprof.columns:
        rprof = step.rprof[var]
        rad = None
        if var in constants.RPROF_VARS:
            meta = constants.RPROF_VARS[var]
        else:
            meta = constants.Varr(None, None)
    elif var in constants.RPROF_VARS_EXTRA:
        meta = constants.RPROF_VARS_EXTRA[var]
        rprof, rad = meta.description(step)
        meta = constants.Varr(getdoc(meta.description), meta.shortname)
    else:
        raise ValueError('Unknown rprof variable {}.'.format(var))

    return rprof, rad, meta


def plot_grid(step, args):
    """Plot cell position and thickness"""
    rad = step.rprof['r']
    drad, _, _ = get_rprof(step, 'dr')
    fig, (ax1, ax2) = args.plt.subplots(2, sharex=True)
    ax1.plot(rad, '-ko')
    ax1.set_ylabel('$r$')
    ax2.plot(drad, '-ko')
    ax2.set_ylabel('$dr$')
    ax2.set_xlim([-0.5, len(rad) - 0.5])
    ax2.set_xlabel('Cell number')
    fig.savefig('grid_{}.pdf'.format(step.istep))


def plot_average(sdat, lovs, args):
    """Plot time averaged profiles"""
    sovs = misc.set_of_vars(lovs)
    istart = None
    # assume constant z spacing for the moment
    ilast = sdat.rprof.index.levels[0][-1]
    rlast = sdat.rprof.loc[ilast]
    rprof_averaged = rlast.loc[:, sovs] * 0
    nprofs = 0
    rads = {}
    metas = {}
    for step in misc.steps_gen(sdat, args):
        if step.rprof is None:
            continue
        if istart is None:
            istart = step.istep
        ilast = step.istep
        nprofs += 1
        for rvar in sovs:
            rprof, rad, meta = get_rprof(step, rvar)
            rprof_averaged[rvar] += rprof
            metas[rvar] = meta
            if rad is not None:
                rads[rvar] = rad

    rprof_averaged /= nprofs
    rprof_averaged['r'] = rlast.loc[:, 'r'] + \
        misc.get_rbounds(sdat.steps[ilast])[0]

    stepstr = '{}_{}'.format(istart, ilast)

    _plot_rprof_list(lovs, rprof_averaged, metas, args, stepstr, rads)


def plot_every_step(sdat, lovs, args):
    """One plot per time step"""
    sovs = misc.set_of_vars(lovs)

    for step in misc.steps_gen(sdat, args):
        if step.rprof is None:
            continue
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

        _plot_rprof_list(lovs, rprofs, metas, args, stepstr, rads)


def rprof_cmd(args):
    """Plot radial profiles"""
    sdat = StagyyData(args.path)
    if sdat.rprof is None:
        return

    if args.grid:
        for step in misc.steps_gen(sdat, args):
            plot_grid(step, args)

    lovs = misc.list_of_vars(args.plot)
    if not lovs:
        return

    if args.average:
        plot_average(sdat, lovs, args)
    else:
        plot_every_step(sdat, lovs, args)
