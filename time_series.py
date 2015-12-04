"""Plots time series of temperature and heat fluxes outputs from stagyy.

Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
Date: 2015/11/27
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import f90nml
import os

def find_nearest(array, value):
    """Find the data point nearest to value"""
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def time_cmd(args):
    """plot temporal series"""
    test = raw_input('Compute statistics? [Y/n] ')
    compstat = test == 'y' or not test
    if args.xkcd:
        plt.xkcd()


    lwdth = args.linewidth
    ftsz = args.fontsize

    #Read par file in the parent or present directory.
    read_par_file = True
    if os.path.exists('../par'):
        par_file = '../par'
    elif os.path.exists('par'):
        par_file = 'par'
    else:
        print 'No par file found. Input pars by hand'
        read_par_file = False
        rcmb = 1
        geom = str(raw_input('spherical (s) or cartesian (c)? '))
        spherical = geom == 's'

    print 'par file used = ', par_file

    if read_par_file:
        nml = f90nml.read(par_file)
        spherical = (nml['geometry']['shape'] == 'spherical' or
                     nml['geometry']['shape'] == 'Spherical')
        if spherical:
            rcmb = nml['geometry']['r_cmb']
        else:
            rcmb = 0.
        stem = nml['ioin']['output_file_stem']
        ste = stem.split('/')[-1]
        filetype = 'time'
        if os.path.exists('../'+ste+'_'+filetype+'.dat'):
            timefile = '../'+ste+'_'+filetype+'.dat'
        elif os.path.exists(ste+'_'+filetype+'.dat'):
            timefile = ste+'_'+filetype+'.dat'
        elif os.path.exists(stem+'_'+filetype+'.dat'):
            timefile = stem+'_'+filetype+'.dat'
        else:
            print 'No profile file found. stem = ', ste
            sys.exit()
        rab = nml['refstate']['Ra0']
        rah = nml['refstate']['Rh']
        botpphase = nml['boundaries']['BotPphase']

    with open(timefile, 'r') as infile:
        first = infile.readline()

    colnames = first.split()
    # suppress two columns from the header.
    # Only temporary since this has been corrected in stag
    if len(colnames) == 33:
        colnames = colnames[:28]+colnames[30:]

    data = np.loadtxt(timefile, skiprows=1)

    ntot = len(data)

    if spherical:
        rmin = rcmb
        rmax = rmin+1
        coefb = 1 #rb**2*4*pi
        coefs = (rmax/rmin)**2 #*4*pi
        volume = rmin*(1-(rmax/rmin)**3)/3 #*4*pi/3
    else:
        coefb = 1.
        coefs = 1.
        volume = 1.

    time = data[:, 1]

    dtdt = (data[2:ntot-1, 5]-data[0:ntot-3, 5]) / (data[2:ntot-1, 1]-
                                                    data[0:ntot-3, 1])
    ebalance = data[1:ntot-2, 2]*coefs - data[1:ntot-2, 3]*coefb - volume*dtdt

    fig = plt.figure(figsize=(30, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, data[:, 2]*coefs, 'b', label='Surface', linewidth=lwdth)
    plt.plot(time, data[:, 3]*coefb, 'r', label='Bottom', linewidth=lwdth)
    plt.plot(time[1:ntot-2:], ebalance, 'g', label='Energy balance',
             linewidth=lwdth)
    plt.ylabel('Heat flow', fontsize=ftsz)
    plt.legend = plt.legend(loc='upper right', shadow=False, fontsize=ftsz)
    plt.legend.get_frame().set_facecolor('white')
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)

    plt.subplot(2, 1, 2)
    plt.plot(time, data[:, 5], 'k', linewidth=lwdth)
    plt.xlabel('Time', fontsize=ftsz)
    plt.ylabel('Mean temperature', fontsize=ftsz)
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)

    plt.savefig("flux_time.pdf", format='PDF')

    if compstat:
        coords = []
        print 'right click to select starting time of statistics computations'
        # Simple mouse click function to store coordinates
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
    if compstat:
        ch1 = np.where(time == (find_nearest(time, coords[0][0])))

        print 'Statistics computed from t ='+str(time[ch1[0][0]])
        for num in range(2, len(colnames)):
            moy.append(np.trapz(data[ch1[0][0]:ntot-1, num],
                                x=time[ch1[0][0]:ntot-1])/
                       (time[ntot-1]-time[ch1[0][0]]))
            rms.append(sqrt(np.trapz((data[ch1[0][0]:ntot-1, num] -
                                         moy[num-2])**2,
                                        x=time[ch1[0][0]:ntot-1])/
                               (time[ntot-1]-time[ch1[0][0]])))
            print colnames[num]+' = '+str(moy[num-2])+' pm '+str(rms[num-2])
        ebal.append(np.trapz(ebalance[ch1[0][0]-1:ntot-3],
                             x=time[ch1[0][0]:ntot-2])/(time[ntot-2]-
                                                        time[ch1[0][0]]))
        rms_ebal.append(sqrt(np.trapz((ebalance[ch1[0][0]-1:ntot-3]-ebal)**2,
                                         x=time[ch1[0][0]:ntot-2])/
                                (time[ntot-2]-time[ch1[0][0]])))
        print 'Energy balance '+str(ebal)+' pm '+str(rms_ebal)
        results = moy+ebal+rms+rms_ebal
        fich = open('Stats.dat', 'w')
        fich.write("%10.5e %10.5e %10.5e " % (rab, rah, botpphase))
        for item in results:
            fich.write("%10.5e " % item)
        fich.write("\n")
        fich.close()

