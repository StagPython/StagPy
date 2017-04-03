"""Plots time series of temperature and heat fluxes outputs from stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/11/27
"""

import numpy as np
from math import sqrt
from .stagyydata import StagyyData


def time_cmd(args):
    """plot temporal series"""
    eps = 1.e-10
    plt = args.plt

    lwdth = args.linewidth
    ftsz = args.fontsize

    sdat = StagyyData(args.path)
    if args.tend < eps:
        args.tend = sdat.tseries.iloc[-1].loc['t']
    data = sdat.tseries[args.tstart <= sdat.tseries['t']]
    data = data[data['t'] <= args.tend]
    args.tstart = data['t'].iloc[0]
    args.tend = data['t'].iloc[-1]
    ntot = len(data)

    rab = sdat.par['refstate']['ra0']
    rah = sdat.par['refstate']['Rh']
    botpphase = sdat.par['boundaries']['BotPphase']

    spherical = sdat.par['geometry']['shape'].lower() == 'spherical'
    if spherical:
        rcmb = sdat.par['geometry']['r_cmb']
        rmin = rcmb
        rmax = rmin + 1
        coefb = 1  # rb**2*4*pi
        coefs = (rmax / rmin)**2  # *4*pi
        volume = rmin * (1 - (rmax / rmin)**3) / 3  # *4*pi/3
    else:
        rcmb = 0.
        coefb = 1.
        coefs = 1.
        volume = 1.

    time = data['t'].values
    mtemp = data['Tmean'].values
    ftop = data['ftop'].values * coefs
    fbot = data['fbot'].values * coefb
    vrms = data['vrms'].values

    dtdt = (mtemp[2:] - mtemp[:-2]) / (time[2:] - time[:-2])
    ebalance = ftop[1:-1] - fbot[1:-1] - volume * dtdt

    # -------- TEMPERATURE and FLOW PLOTS
    fig = plt.figure(figsize=(30, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, ftop, 'b', label='Surface', linewidth=lwdth)
    plt.plot(time, fbot, 'r', label='Bottom', linewidth=lwdth)
    if args.energy:
        plt.plot(time[1:-1], ebalance, 'g', label='Energy balance',
                 linewidth=lwdth)
    plt.ylabel('Heat flow', fontsize=ftsz)
    plt.legend = plt.legend(loc='upper right', shadow=False, fontsize=ftsz)
    plt.legend.get_frame().set_facecolor('white')
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.xlim([args.tstart, args.tend])

    if args.annottmin:
        plt.annotate('tminT', xy=(args.tmint, 0), xytext=(args.tmint, -10),
                     arrowprops={'facecolor': 'black'})
        plt.annotate('tminC', xy=(args.tminc, 0), xytext=(args.tminc, 10),
                     arrowprops={'facecolor': 'black'})

    plt.subplot(2, 1, 2)
    plt.plot(time, mtemp, 'k', linewidth=lwdth)
    plt.xlabel('Time', fontsize=ftsz)
    plt.ylabel('Mean temperature', fontsize=ftsz)
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.xlim([args.tstart, args.tend])

    plt.savefig("fig_fluxtime.pdf", format='PDF')

    # -------- TEMPERATURE and VRMS PLOTS
    fig = plt.figure(figsize=(30, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, vrms, 'g', linewidth=lwdth)
    plt.ylabel(r'$v_{\rm rms}$', fontsize=ftsz)
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.xlim([args.tstart, args.tend])

    plt.subplot(2, 1, 2)
    plt.plot(time, mtemp, 'k', linewidth=lwdth)
    plt.xlabel('Time', fontsize=ftsz)
    plt.ylabel('Mean temperature', fontsize=ftsz)
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.xlim([args.tstart, args.tend])

    plt.savefig("fig_vrmstime.pdf", format='PDF')

    if not args.compstat:
        return None

    coords = []
    moy = []
    rms = []
    ebal = []
    rms_ebal = []
    delta_time = time[-1] - time[0]
    for col in data.columns[1:]:
        moy.append(np.trapz(data[col], x=time) / delta_time)
        rms.append(sqrt(np.trapz((data[col] - moy[-1])**2, x=time) /
                        delta_time))
    delta_time = time[-2] - time[1]
    ebal.append(np.trapz(ebalance, x=time[1:-1]) / delta_time)
    rms_ebal.append(sqrt(np.trapz((ebalance - ebal)**2, x=time[1:-1]) /
                         delta_time))
    print('Energy balance', ebal, 'pm', rms_ebal)
    results = moy + ebal + rms + rms_ebal
    with open('statistics.dat', 'w') as out_file:
        out_file.write("%10.5e %10.5e %10.5e " % (rab, sum(rah), botpphase))
        for item in results:
            out_file.write("%10.5e " % item)
        out_file.write("\n")
