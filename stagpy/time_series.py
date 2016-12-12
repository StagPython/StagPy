"""Plots time series of temperature and heat fluxes outputs from stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/11/27
"""

import numpy as np
from math import sqrt
from .stagyydata import StagyyData


def find_nearest(array, value):
    """Find the data point nearest to value"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def time_cmd(args):
    """plot temporal series"""
    eps = 1.e-10
    plt = args.plt

    lwdth = args.linewidth
    ftsz = args.fontsize

    sdat = StagyyData(args.path)
    data = sdat.tseries
    ntot = len(data)

    rab = sdat.par['refstate']['Ra0']
    rah = sdat.par['refstate']['Rh']
    botpphase = sdat.par['boundaries']['BotPphase']

    if args.tstart < 0:
        args.tstart = data[0, 1]

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

    if abs(args.tstart) < eps:
        nstart = 0
    else:
        nstart = np.argmin(abs(args.tstart - data[:, 1]))

    if abs(args.tend) < eps:
        args.tend = np.amax(data[:, 1])

    time = data[nstart:, 1]
    mtemp = data[nstart:, 5]
    ftop = data[nstart:, 2] * coefs
    fbot = data[nstart:, 3] * coefb
    vrms = data[nstart:, 8]

    dtdt = (data[2:ntot - 1, 5] - data[0:ntot - 3, 5]) /\
        (data[2:ntot - 1, 1] - data[0:ntot - 3, 1])
    ebalance = data[1:ntot - 2, 2] * coefs - data[1:ntot - 2, 3] * coefb\
        - volume * dtdt

    # -------- TEMPERATURE and FLOW PLOTS
    fig = plt.figure(figsize=(30, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, ftop, 'b', label='Surface', linewidth=lwdth)
    plt.plot(time, fbot, 'r', label='Bottom', linewidth=lwdth)
    if args.energy:
        plt.plot(time[1:ntot - 2:], ebalance, 'g', label='Energy balance',
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
    print('right click to select starting time of statistics computations')

    def onclick(event):
        """get position and button from mouse click"""
        ixc, iyc = event.xdata, event.ydata
        button = event.button
        # assign global variable to access outside of function
        if button == 3:
            coords.append((ixc, iyc))
            # Disconnect after 1 clicks
        if len(coords) == 1:
            fig.canvas.mpl_disconnect(cid)
            plt.close(1)
        return

    # Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    moy = []
    rms = []
    ebal = []
    rms_ebal = []
    ch1 = np.where(time == (find_nearest(time, coords[0][0])))

    print('Statistics computed from t =' + str(time[ch1[0][0]]))
    for num in range(2, data.shape[1]):
        moy.append(np.trapz(data[ch1[0][0]:ntot - 1, num],
                            x=time[ch1[0][0]:ntot - 1]) /
                   (time[ntot - 1] - time[ch1[0][0]]))
        rms.append(sqrt(np.trapz((data[ch1[0][0]:ntot - 1, num] -
                                  moy[num - 2])**2,
                                 x=time[ch1[0][0]:ntot - 1]) /
                        (time[ntot - 1] - time[ch1[0][0]])))
    with open('statistics.dat') as fid:
        for val, err in np.array([moy, rms]).T:
            fid.write('{:.5e}   {:.5e}\n'.format(val, err))
    ebal.append(np.trapz(ebalance[ch1[0][0] - 1:ntot - 3],
                         x=time[ch1[0][0]:ntot - 2]) /
                (time[ntot - 2] - time[ch1[0][0]]))
    rms_ebal.append(sqrt(np.trapz(
                         (ebalance[ch1[0][0] - 1:ntot - 3] - ebal)**2,
                         x=time[ch1[0][0]:ntot - 2]) /
                         (time[ntot - 2] - time[ch1[0][0]])))
    print('Energy balance', ebal, 'pm', rms_ebal)
    results = moy + ebal + rms + rms_ebal
    with open('Stats.dat', 'w') as out_file:
        out_file.write("%10.5e %10.5e %10.5e " % (rab, rah, botpphase))
        for item in results:
            out_file.write("%10.5e " % item)
        out_file.write("\n")
