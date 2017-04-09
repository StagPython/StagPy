"""Plots time series of temperature and heat fluxes outputs from stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/11/27
"""

import numpy as np
from math import sqrt
from .stagyydata import StagyyData
from . import constants


def time_cmd(args):
    """plot temporal series"""
    plt = args.plt

    lwdth = args.linewidth
    ftsz = args.fontsize

    sdat = StagyyData(args.path)
    data = sdat.tseries_between(args.tstart, args.tend)

    rab = sdat.par['refstate']['ra0']
    rah = sdat.par['refstate']['Rh']
    botpphase = sdat.par['boundaries']['BotPphase']

    time = data['t'].values
    mtemp = data['Tmean'].values
    ftop = data['Nutop'].values
    fbot = data['Nubot'].values
    vrms = data['vrms'].values

    ebalance_func = constants.TIME_VARS_EXTRA['ebalance'].description
    ebalance, _ = ebalance_func(sdat, args.tstart, args.tend)

    args.tstart = time[0]
    args.tend = time[-1]
    # -------- TEMPERATURE and FLOW PLOTS
    plt.figure(figsize=(30, 10))

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

    plt.subplot(2, 1, 2)
    plt.plot(time, mtemp, 'k', linewidth=lwdth)
    plt.xlabel('Time', fontsize=ftsz)
    plt.ylabel('Mean temperature', fontsize=ftsz)
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.xlim([args.tstart, args.tend])

    plt.savefig("fig_fluxtime.pdf", format='PDF')

    # -------- TEMPERATURE and VRMS PLOTS
    plt.figure(figsize=(30, 10))

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
