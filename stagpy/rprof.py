"""Plots radial profiles coming out of stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/09/11
"""
import numpy as np
from scipy import integrate as itg
import math
from . import constants, misc
from .stagdata import RprofData


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


def _calc_energy(data, ir0, ir1):  # for args.plot_energy
    """Compute energy balance(r)"""
    zgrid = np.array(data[ir0:ir1, 63], float)
    zgrid = np.append(zgrid, 1.)
    dzg = np.array(data[ir0 + 1:ir1, 0], float)\
        - np.array(data[ir0:ir1 - 1, 0], float)
    qadv = np.array(data[ir0:ir1 - 1, 60], float)
    qadv = np.insert(qadv, 0, 0.)
    qadv = np.append(qadv, 0.)
    qcond = (np.array(data[ir0:ir1 - 1, 1], float) -
             np.array(data[ir0 + 1:ir1, 1], float)) / dzg
    qcond0 = (1. - float(data[ir0, 1])) / float(data[ir0, 0])
    qtop = float(data[ir1, 1]) / (1. - float(data[ir1, 0]))
    qcond = np.insert(qcond, 0, qcond0)
    qcond = np.append(qcond, qtop)
    qtot = qadv + qcond
    return qtot, qadv, qcond, zgrid


def plotprofiles(quant, vartuple, data, tsteps, nzi, rbounds, args,
                 ctheoarg, integrate=False):
    """Plot the chosen profiles for the chosen timesteps

    quant holds the strings for the x axis annotation and
    the legends for the additional profiles

    vartuple contains the numbers of the column to be plotted
    """
    plt = args.plt
    istart, ilast, istep = args.timestep
    lwdth = args.linewidth
    ftsz = args.fontsize
    rmin, rmax, rcmb = rbounds
    axax, initprof = ctheoarg
    linestyles = ('-', '--', '-.', ':')

    if integrate:
        def integ(fct, rad):
            """(theta, phi) surface scaling factor"""
            return fct * (rad / rmax)**2

    if quant[0] == 'Grid':
        fig, axe = plt.subplots(2, sharex=True)
    else:
        fig, axe = plt.subplots()

    timename = str(istart) + "_" + str(ilast) + "_" + str(istep)
    if args.plot_difference:
        concdif = []
        tempdif = []
        wmax = []

    for step in range(istart + 1, ilast + 1, istep):
        # find the indices
        ann = sorted(np.append(nzi[:, 0], step))
        inn = ann.index(step)
        nnz = np.multiply(nzi[:, 1], nzi[:, 2])

        ir0 = np.sum([nnz[0:inn]]) + (step - nzi[inn - 1, 0] - 1) * nzi[inn, 2]
        ir1 = ir0 + nzi[inn, 2] - 1

        if quant[0] == 'Energy':
            energy = _calc_energy(data, ir0, ir1)

        # Plot the profiles
        if quant[0] == 'Grid':
            axe[0].plot(data[ir0:ir1, 0], '-ko', label='z')
            axe[0].set_ylabel('z', fontsize=ftsz)

            dzgrid = (np.array(data[ir0 + 1:ir1, 0], np.float) -
                      np.array(data[ir0:ir1 - 1, 0], np.float))
            axe[1].plot(dzgrid, '-ko', label='dz')
            axe[1].set_xlabel('cell number', fontsize=ftsz)
            axe[1].set_ylabel('dz', fontsize=ftsz)
        else:
            if quant[0] == 'Energy':
                profiles = np.array(np.transpose(energy)[:, [0, 1, 2]],
                                    float)
                radius = np.array(np.transpose(energy)[:, 3], float) + rcmb
            else:
                profiles = np.array(data[ir0:ir1, vartuple], float)
                radius = np.array(data[ir0:ir1, 0], float) + rcmb
            for i in range(profiles.shape[1]):
                if integrate:
                    donnee = list(map(integ, profiles[:, i], radius))
                else:
                    donnee = profiles[:, i]
                if i == 0:
                    pplot = plt.plot(donnee, radius, linewidth=lwdth,
                                     label=r'$t=%.2e$' %
                                     (tsteps[step - 1, 2]))

                    # get color and size characteristics
                    col = pplot[0].get_color()

                    # overturned version of the initial profiles
                    if quant[0] in ('Concentration', 'Temperature') and\
                       (args.plot_overturn_init or args.plot_difference) and\
                       step == istart + 1:
                        rfin = (rmax**3 + rmin**3 - radius**3)**(1 / 3)
                        if quant[0] == 'Concentration':
                            conc0 = _extrap(rfin, radius, profiles[:, 0])
                        if quant[0] == 'Temperature':
                            temp0 = _extrap(rfin, radius, profiles[:, 0])
                        plt.plot(donnee, rfin, '--', c=col,
                                 linewidth=lwdth, label='Overturned')

                    if quant[0] == 'Concentration' and args.plot_difference:
                        concd1 = _normprof(radius, profiles[:, 0] - conc0)
                        concdif.append(concd1)
                    if quant[0] == 'Temperature' and args.plot_difference:
                        tempd1 = _normprof(radius, profiles[:, 0] - temp0)
                        tempdif.append(tempd1)
                        wmax.append(max(np.array(data[ir0:ir1, 7],
                                                 np.float)))
                    # plot the overturned version of the initial profiles
                    # if ((quant[0] == 'Concentration' or
                    #      quant[0] == 'Temperature') and
                    #         args.plot_overturn_init and step == istart+1):
                    #     rfin = (rmax**3.+rmin**3.-radius**3.)**(1./3.)
                    #     plt.plot(donnee, rfin, '--', c=col,
                    #              linewidth=lwdth, label='Overturned')

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
            if len(vartuple) > 1 and step == ilast:
                    # legends for the additionnal profiles
                axes = plt.gca()
                rangex = axes.get_xlim()
                rangey = axes.get_ylim()
                xlgd1 = rangex[1] - 0.12 * (rangex[1] - rangex[0])
                xlgd2 = rangex[1] - 0.05 * (rangex[1] - rangex[0])
                for i in range(profiles.shape[1]):
                    ylgd = rangey[1] - 0.05 * (i + 1) * (rangey[1] - rangey[0])
                    plt.plot([xlgd1, xlgd2], [ylgd, ylgd], c='black',
                             linestyle=linestyles[i], linewidth=lwdth,
                             dash_capstyle='round',)
                    plt.text(xlgd1 - 0.02 * (rangex[1] - rangex[0]), ylgd,
                             quant[i + 1], ha='right')

                plt.xlabel(quant[0], fontsize=ftsz)
                plt.ylabel('z', fontsize=ftsz)
                plt.xticks(fontsize=ftsz)
                plt.yticks(fontsize=ftsz)
    if quant[0] == 'Grid':
        plt.savefig("Grid" + timename + ".pdf", format='PDF')
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
            axax[0].semilogy(tsteps[0:ilast:istep, 2], concdif / concdif[0])
            axax[0].semilogy(tsteps[iminc * istep, 2],
                             concdif[iminc] / concdif[0],
                             'o', label=r'$t=%.2e$' % (tsteps[iminc, 2]))
            axax[0].set_ylabel('Composition diff.')
            plt.legend(loc='upper right')
            return tsteps[iminc * istep, 2], concdif[iminc] / concdif[0],\
                iminc, timename
        if quant[0] == 'Temperature':
            axax[1].semilogy(tsteps[istart:ilast:istep, 2],
                             tempdif / tempdif[0])
            imint = tempdif.index(min(tempdif))
            axax[1].semilogy(tsteps[imint * istep, 2],
                             tempdif[imint] / tempdif[0],
                             'o', label=r'$t=%.2e$' % (tsteps[imint, 2]))
            axax[1].set_ylabel('Temperature diff.')
            plt.legend(loc='lower right')
            # maximum velocity as function of time
            axax[2].semilogy(tsteps[istart:ilast:istep, 2], wmax)
            axax[2].set_ylabel('Max. rms vert. velocity')
            axax[2].set_xlabel('Time')
            wma = max(wmax)
            iwm = wmax.index(wma)
            sigma = math.log(wmax[iwm - 3] / wmax[0]) / tsteps[iwm - 3, 2]
            expw = [wmax[0] * math.exp(sigma * t)
                    for t in tsteps[0:iwm + 2:istep, 2]]
            axax[2].semilogy(tsteps[0:iwm + 2:istep, 2], expw,
                             linestyle='--', label=r'$sigma=%.2e$' % sigma)
            plt.legend(loc='upper right')
            return tsteps[imint * istep, 2], tempdif[imint] / tempdif[0], iwm,\
                wma, imint, sigma, timename
    return None


def rprof_cmd(args):
    """Plot radial profiles"""
    if not (args.plot_conctheo and args.plot_temperature and
            args.plot_concentration):
        args.plot_difference = False

    ctheoarg = None, None

    if args.plot_difference:
        # plot time series of difference profiles
        # initialize the plot here
        figd, axax = args.plt.subplots(3, sharex=True)
        ra0 = args.par_nml['refstate']['ra0']
        ctheoarg = axax, None

    # parameters for the theoretical composition profiles

    spherical = args.par_nml['geometry']['shape'].lower() == 'spherical'
    if spherical:
        rcmb = args.par_nml['geometry']['r_cmb']
    else:
        rcmb = 0.
    rmin = rcmb
    rmax = rcmb + 1.
    rbounds = rmin, rmax, rcmb

    if args.plot_conctheo:
        xieut = args.par_nml['tracersin']['fe_eut']
        k_fe = args.par_nml['tracersin']['k_fe']
        xi0l = args.par_nml['tracersin']['fe_cont']
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

    rprof_data = RprofData(args)
    data, tsteps, nzi = rprof_data.data, rprof_data.tsteps, rprof_data.nzi

    for var in 'tvnc':  # temperature, velo, visco, concentration
        meta = constants.RPROF_VAR_LIST[var]
        if not misc.get_arg(args, meta.arg):
            continue
        labels = [meta.name]
        cols = [meta.prof_idx]
        if misc.get_arg(args, meta.min_max):
            labels.extend(['Mean', 'Minimum', 'Maximum'])
            cols.extend([meta.prof_idx + 1, meta.prof_idx + 2])
        out = plotprofiles(labels, cols, data, tsteps, nzi, rbounds,
                           args, ctheoarg)
        if var == 't' and args.plot_difference:
            _, _, _, _, imint, sigma, timename = out
        if var == 'c' and args.plot_difference:
            _, _, iminc, timename = out

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
            fich.write(fmt.format(rcmb, k_fe, ra0, tsteps[imint, 2],
                       sigma, tsteps[iminc, 2]))

    # Plot grid spacing
    if args.plot_grid:
        plotprofiles(['Grid'], None, data, tsteps, nzi, rbounds,
                     args, ctheoarg)

    # Plot the profiles of vertical advection: total and contributions from up-
    # and down-welling currents
    if args.plot_advection:
        plotprofiles(['Advection per unit surface', 'Total', 'down-welling',
                      'Up-welling'], (57, 58, 59), data, tsteps, nzi,
                     rbounds, args, ctheoarg)
        if spherical:
            plotprofiles(['Total scaled advection', 'Total', 'down-welling',
                          'Up-welling'], (57, 58, 59), data, tsteps, nzi,
                         rbounds, args, ctheoarg, integrate=True)
    if args.plot_energy:
        plotprofiles(['Energy', 'Total', 'Advection',
                      'conduction'], (57, 58, 59), data, tsteps, nzi,
                     rbounds, args, ctheoarg, integrate=True)
