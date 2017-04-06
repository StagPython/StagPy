"""Plots radial profiles coming out of stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/09/11
"""
from inspect import getdoc
import math
import numpy as np
from scipy import integrate as itg
# from cycler import cycler
from . import constants, misc
from .stagyydata import StagyyData, NoSnapshotError


def _normprof(rrr, func):  # for args.plot_difference
    """Volumetric norm of a profile

    Two arrays: rrr is the radius position and f the function.
    """
    norm = 3. / (rrr[-1]**3 - rrr[0]**3) * itg.trapz(func**2 * rrr**2, rrr)
    return norm


def _extrap(xpos, xpoints, ypoints):  # for args.plot_difference
    """np.interp function with linear extrapolation.

    Would be best to use degree 3 extrapolation
    """
    ypos = np.interp(xpos, xpoints, ypoints)
    ypos[xpos < xpoints[0]] = ypoints[0]\
        + (xpos[xpos < xpoints[0]] - xpoints[0])\
        * (ypoints[0] - ypoints[1]) / (xpoints[0] - xpoints[1])
    ypos[xpos > xpoints[-1]] = ypoints[-1]\
        + (xpos[xpos > xpoints[-1]] - xpoints[-1])\
        * (ypoints[-1] - ypoints[-2]) / (xpoints[-1] - xpoints[-2])
    return ypos


def _calc_energy(rprof):  # for args.plot_energy
    """Compute energy balance(r)"""
    zgrid = rprof[0]
    zgrid = np.append(zgrid, 1.)
    dzg = rprof[0, 1:] - rprof[0, :-1]
    qadv = rprof[66]
    qadv = np.append(qadv, 0.)
    qcond = (rprof[1, :-1] - rprof[1, 1:]) / dzg
    qcond0 = (1. - rprof[1, 0]) / rprof[0, 0]
    qtop = rprof[1, -1] / (1. - rprof[0, -1])
    qcond = np.insert(qcond, 0, qcond0)
    qcond = np.append(qcond, qtop)
    qtot = qadv + qcond
    return qtot, qadv, qcond, zgrid


def fmttime(tin):
    """Time formatting for labels"""
    aaa, bbb = '{:.2e}'.format(tin).split('e')
    bbb = int(bbb)
    return r'$t={} \times 10^{{{}}}$'.format(aaa, bbb)


def plotprofiles(sdat, quant, vartuple, rbounds, args,
                 ctheoarg, integrate=False):
    """Plot the chosen profiles for the chosen timesteps

    quant holds the strings for the x axis annotation and
    the legends for the additional profiles

    vartuple contains the numbers of the column to be plotted
    """
    plt = args.plt
    lwdth = args.linewidth
    ftsz = args.fontsize
    rmin, rmax, rcmb = rbounds
    axax, initprof = ctheoarg
    linestyles = ('-', '--', '-.', ':')

    if integrate:
        def integ(fct, rad):
            """(theta, phi) surface scaling factor"""
            return fct * (rad / rmax)**2

    if quant[0] == 'Grid' or quant[0] == 'Grid km':
        fig, axe = plt.subplots(2, sharex=True)
    else:
        fig, axe = plt.subplots()

    if args.plot_difference:
        concdif = []
        tempdif = []
        wmax = []

    # this is from http://stackoverflow.com/questions/4805048/
    # how-to-get-different-colored-lines-for-different-plots-in-a-single-figure
    # colormap = plt.cm.winter_r
    # plt.gca().set_prop_cycle(cycler('color', [colormap(i)
    #                          for i in np.linspace(0, 0.9, num_plots)]))
    #
    # There is no way to know in advance how many plots will be made. Need to
    # collect the line objects and give them a color at the end.

    istart, ilast = None, None
    adv_times = []
    for step in misc.steps_gen(sdat, args):
        rprof = step.rprof
        if rprof is None:
            continue
        if istart is None:
            istart = step.istep
        ilast = step.istep
        # advective time should be step.rprof.time
        adv_times.append(step.rprof.times[step.irsnap])
        # Plot the profiles
        if quant[0] == 'Grid' or quant[0] == 'Grid km':
            axe[0].plot(rprof[0], '-ko', label='z')
            axe[0].set_ylabel('z', fontsize=ftsz)
            axe[0].set_xlim([0, len(rprof[0])])

            dzgrid = rprof[0, 1:] - rprof[0, :-1]
            if quant[0] == 'Grid km':
                ddim = step.sdat.par['geometry']['d_dimensional'] / 1000.
                axe[1].plot(dzgrid * ddim, '-ko', label='dz')
                axe[1].set_ylabel('dz [km]', fontsize=ftsz)
            else:
                axe[1].plot(dzgrid, '-ko', label='dz')
                axe[1].set_ylabel('dz', fontsize=ftsz)
            axe[1].set_xlabel('Cell number', fontsize=ftsz)
            axe[1].set_xlim([0, len(rprof[0])])
        else:
            if quant[0] == 'Energy':
                energy = _calc_energy(rprof)
                profiles = np.array(energy[0:3])
                radius = np.array(energy[3]) + rcmb
            else:
                profiles = rprof[vartuple]
                radius = rprof[0] + rcmb
            for i in range(profiles.shape[0]):
                if integrate:
                    donnee = np.array(list(map(integ, profiles[i], radius)))
                else:
                    donnee = profiles[i]
                if i == 0:
                    pplot = plt.plot(donnee, radius, linewidth=lwdth,
                                     label=fmttime(adv_times[-1]))

                    # get color and size characteristics
                    col = pplot[0].get_color()

                    # overturned version of the initial profiles
                    if quant[0] in ('Concentration', 'Temperature') and\
                       (args.plot_overturn_init or args.plot_difference) and\
                       step == istart + 1:
                        rfin = (rmax**3 + rmin**3 - radius**3)**(1 / 3)
                        if quant[0] == 'Concentration':
                            conc0 = _extrap(rfin, radius, profiles[0])
                        if quant[0] == 'Temperature':
                            temp0 = _extrap(rfin, radius, profiles[0])
                        plt.plot(donnee, rfin, '--', c=col,
                                 linewidth=lwdth, label='Overturned')

                    if quant[0] == 'Concentration' and args.plot_difference:
                        concd1 = _normprof(radius, profiles[0] - conc0)
                        concdif.append(concd1)
                    if quant[0] == 'Temperature' and args.plot_difference:
                        tempd1 = _normprof(radius, profiles[0] - temp0)
                        tempdif.append(tempd1)
                        wmax.append(max(rprof[7]))

                    # plot the theoretical initial profile and its
                    # overturned version
                    if (quant[0] == 'Concentration' and
                            args.plot_conctheo and step == istart + 1):
                        # plot the full profile between rmin and rmax
                        radius2 = np.linspace(rmin, rmax, 1000)
                        cinit = list(map(initprof, radius2))
                        rfin = (rmax**3 + rmin**3 - radius2**3)**(1 / 3)
                        plt.plot(cinit, radius2, 'r--',
                                 linewidth=lwdth, label='Theoretical')
                        plt.plot(cinit, rfin, 'r-.',
                                 linewidth=lwdth, label='Overturned')
                        # add the begining and end points of the stagyy
                        # profile
                        plt.plot([donnee[0], donnee[-1]],
                                 [radius[0], radius[-1]], "o",
                                 label='StagYY profile ends')
                        plt.xlim([0.9 * donnee[0], 1.2 * donnee[-1]])
                else:
                    # additional plots (e. g. min, max)
                    plt.plot(donnee, radius, c=col, dash_capstyle='round',
                             linestyle=linestyles[i], linewidth=lwdth)
                # change the vertical limits
                plt.ylim([rmin - 0.05, rmax + 0.05])
            if len(vartuple) > 1 and step.istep == ilast\
               and quant[0] != 'Viscosity':  # no way to know if at last step
                # legends for the additionnal profiles
                axes = plt.gca()
                rangex = axes.get_xlim()
                rangey = axes.get_ylim()
                xlgd1 = rangex[1] - 0.12 * (rangex[1] - rangex[0])
                xlgd2 = rangex[1] - 0.05 * (rangex[1] - rangex[0])
                for i in range(profiles.shape[0]):
                    ylgd = rangey[1] - 0.05 * (i + 1) * (rangey[1] - rangey[0])
                    plt.plot([xlgd1, xlgd2], [ylgd, ylgd], c='black',
                             linestyle=linestyles[i], linewidth=lwdth,
                             dash_capstyle='round',)
                    plt.text(xlgd1 - 0.02 * (rangex[1] - rangex[0]), ylgd,
                             quant[i + 1], ha='right')

            if step.istep == ilast:  # no way to know if at last step
                if quant[0] == 'Viscosity':
                    plt.xscale('log')
                plt.xlabel(quant[0], fontsize=ftsz)
                plt.ylabel('z', fontsize=ftsz)
                plt.xticks(fontsize=ftsz)
                plt.yticks(fontsize=ftsz)

    timename = '{}_{}'.format(istart, ilast)
    if quant[0] == 'Grid':
        plt.savefig("Grid" + timename + ".pdf", format='PDF')
    elif quant[0] == 'Grid km':
        plt.savefig("Gridkm" + timename + ".pdf", format='PDF')
    else:
        # legend
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         borderaxespad=0., mode="expand",
                         ncol=3, fontsize=ftsz,
                         columnspacing=1.0, labelspacing=0.0,
                         handletextpad=0.1, handlelength=1.5,
                         fancybox=True, shadow=False)

        plt.savefig(quant[0].replace(' ', '_') + timename + ".pdf",
                    format='PDF',
                    bbox_extra_artists=(lgd, ), bbox_inches='tight')
    plt.close(fig)
    if args.plot_difference:
        # plot time series of difference profiles
        if quant[0] == 'Concentration':
            iminc = concdif.index(min(concdif))
            axax[0].semilogy(adv_times, concdif / concdif[0])
            axax[0].semilogy(adv_times[iminc],
                             concdif[iminc] / concdif[0],
                             'o', label=fmttime(adv_times[iminc]))
            axax[0].set_ylabel('Composition diff.')
            plt.legend(loc='upper right')
            return adv_times, iminc, timename
        if quant[0] == 'Temperature':
            axax[1].semilogy(adv_times, tempdif / tempdif[0])
            imint = tempdif.index(min(tempdif))
            axax[1].semilogy(adv_times[imint],
                             tempdif[imint] / tempdif[0],
                             'o', label=fmttime(adv_times[imint]))
            axax[1].set_ylabel('Temperature diff.')
            plt.legend(loc='lower right')
            # maximum velocity as function of time
            axax[2].semilogy(adv_times, wmax)
            axax[2].set_ylabel('Max. rms vert. velocity')
            axax[2].set_xlabel('Time')
            wma = max(wmax)
            iwm = wmax.index(wma)
            sigma = math.log(wmax[iwm - 3] / wmax[0]) / adv_times[iwm - 3]
            expw = [wmax[0] * math.exp(sigma * t)
                    for t in adv_times[0:iwm + 2]]
            axax[2].semilogy(adv_times[0:iwm + 2], expw,
                             linestyle='--', label=r'$sigma=%.2e$' % sigma)
            plt.legend(loc='upper right')
            return adv_times, imint, sigma, timename


def _list_of_vars(arg_plot):
    """Compute list of variables per plot

    Three nested lists:
    - variables on the same subplots;
    - subplots on the same figure;
    - figures.
    """
    lovs = [[[var for var in svars.split(',') if var]
             for svars in pvars.split('.') if svars]
            for pvars in arg_plot.split('_') if pvars]
    return [lov for lov in lovs if lov]


def _set_of_vars(lovs):
    """Build set of variables from list"""
    return set(var for pvars in lovs for svars in pvars for var in svars)


def _plot_rprof_list(lovs, rprofs, metas, args, stepstr):
    """Plot requested profiles"""
    for vfig in lovs:
        fig, axes = args.plt.subplots(ncols=len(vfig), sharey=True)
        axes = [axes] if len(vfig) == 1 else axes
        fname = ''
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            for rvar in vplt:
                fname += rvar + '_'
                axes[iplt].plot(rprofs[rvar], rprofs['r'],
                                label=metas[rvar].description)
                lbl = metas[rvar].shortname
                if xlabel is None:
                    xlabel = lbl
                elif xlabel != lbl:
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
        if var in constants.RPROF_VARS:
            meta = constants.RPROF_VARS[var]
        else:
            meta = constants.Varr(None, None)
    elif var in constants.RPROF_VARS_EXTRA:
        meta = constants.RPROF_VARS_EXTRA[var]
        rprof = meta.description(step)
        meta = constants.Varr(getdoc(meta.description), meta.shortname)
    else:
        raise ValueError('Unknown rprof variable {}.'.format(var))

    return rprof, meta


def plot_average(sdat, lovs, args):
    """Plot time averaged profiles"""
    sovs = _set_of_vars(lovs)
    istart = None
    # assume constant z spacing for the moment
    ilast = sdat.rprof.index.levels[0][-1]
    rlast = sdat.rprof.loc[ilast]
    rprof_averaged = rlast.loc[:, sovs] * 0
    nprofs = 0
    metas = {}
    for step in misc.steps_gen(sdat, args):
        if step.rprof is None:
            continue
        if istart is None:
            istart = step.istep
        ilast = step.istep
        nprofs += 1
        for rvar in sovs:
            rprof, meta = get_rprof(step, rvar)
            rprof_averaged[rvar] += rprof
            metas[rvar] = meta

    rprof_averaged /= nprofs
    rprof_averaged['r'] = rlast.loc[:, 'r']

    stepstr = '{}_{}'.format(istart, ilast)

    _plot_rprof_list(lovs, rprof_averaged, metas, args, stepstr)


def plot_every_step(sdat, lovs, args):
    """One plot per time step"""
    sovs = _set_of_vars(lovs)

    for step in misc.steps_gen(sdat, args):
        if step.rprof is None:
            continue
        rprofs = {}
        metas = {}
        for rvar in sovs:
            rprof, meta = get_rprof(step, rvar)
            rprofs[rvar] = rprof
            metas[rvar] = meta
        rprofs['r'] = step.rprof.loc[:, 'r']
        stepstr = str(step.istep)

        _plot_rprof_list(lovs, rprofs, metas, args, stepstr)


def rprof_cmd(args):
    """Plot radial profiles"""
    sdat = StagyyData(args.path)
    lovs = _list_of_vars(args.plot)
    if not lovs or sdat.rprof is None:
        return

    if args.average:
        plot_average(sdat, lovs, args)
    else:
        plot_every_step(sdat, lovs, args)

    return

    if args.plot_difference:
        # plot time series of difference profiles
        # initialize the plot here
        figd, axax = args.plt.subplots(3, sharex=True)
        ra0 = sdat.par['refstate']['ra0']
        ctheoarg = axax, None

    # parameters for the theoretical composition profiles
    # could change over time!
    try:
        rcmb = sdat.snaps[-1].geom.rcmb
    except NoSnapshotError:
        rcmb = sdat.par['geometry']['r_cmb']
    rcmb = max(0, rcmb)
    rmin = rcmb  # two times the same info...
    rmax = rcmb + 1.
    rbounds = rmin, rmax, rcmb

    if args.plot_conctheo:
        xieut = sdat.par['tracersin']['fe_eut']
        k_fe = sdat.par['tracersin']['k_fe']
        xi0l = sdat.par['tracersin']['fe_cont']
        xi0s = k_fe * xi0l
        xired = xi0l / xieut
        rsup = (rmax**3 - xired**(1 / (1 - k_fe)) *
                (rmax**3 - rmin**3))**(1 / 3)
        print('rmin, rmax, rsup=', rmin, rmax, rsup)

        def initprof(rpos):
            """Theoretical initial profile."""
            if rpos < rsup:
                return xi0s * ((rmax**3 - rmin**3) /
                               (rmax**3 - rpos**3))**(1 - k_fe)
            else:
                return xieut
        ctheoarg = ctheoarg[0], initprof

    for var in 'tvunc':  # temp, vertical vel, horizontal vel, viscosity, conc
        out = plotprofiles(sdat, labels, cols, rbounds, args, ctheoarg)
        if var == 't' and args.plot_difference:
            adv_times, imint, sigma, timename = out
        if var == 'c' and args.plot_difference:
            adv_times, iminc, timename = out

    if args.plot_difference:
        args.plt.ticklabel_format(style='sci', axis='x')
        args.plt.savefig('Difference_to_overturned{}.pdf'.format(timename),
                         format='PDF')
        args.plt.close(figd)
        with open('statmin.dat', 'w') as fich:
            fmt = '{:12}' * 6 + '\n'
            fich.write(fmt.format('rcmb', 'k_fe', 'ra', 'tminT',
                                  'sigma', 'tminC'))
            fmt = '{:12.5e}' * 6
            fich.write(fmt.format(rcmb, k_fe, ra0, adv_times[imint],
                                  sigma, adv_times[iminc]))

    # Plot grid spacing
    if args.plot_grid:
        plotprofiles(sdat, ['Grid'], None, rbounds, args, ctheoarg)

    # Plot the profiles of vertical advection: total and contributions from up-
    # and down-welling currents
    try:
        spherical = sdat.snaps[-1].geom.spherical
    except NoSnapshotError:
        spherical = sdat.par['geometry']['shape'].lower() == 'spherical'
    if args.plot_advection:
        plotprofiles(sdat,
                     ['Advection per unit surface', 'Total', 'down-welling',
                      'Up-welling'], [66, 64, 65], rbounds, args, ctheoarg)
        if spherical:
            plotprofiles(sdat,
                         ['Total scaled advection', 'Total', 'down-welling',
                          'Up-welling'], [66, 64, 65], rbounds, args,
                         ctheoarg, integrate=True)
    if args.plot_energy:
        plotprofiles(sdat, ['Energy', 'Total', 'Advection', 'conduction'],
                     [66, 64, 65], rbounds, args, ctheoarg, integrate=spherical)
