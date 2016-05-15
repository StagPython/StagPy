"""Plots position of subduction and ridge at the surface.

Date: 2016/26/01
"""
import numpy as np
import sys
from . import constants, misc
from .stagdata import BinData, RprofData, TimeData
from .field import plot_scalar
from scipy.signal import argrelextrema
from copy import deepcopy


def detect_plates_vzcheck(stagdat_t, stagdat_vp, stagdat_h, rprof_data,
                          args, seuil_memz):
    """detect plates and check with vz and plate size"""
    v_z = stagdat_vp.fields['w']
    v_x = stagdat_vp.fields['v']
    h2o = stagdat_h.fields['h']
    tcell = stagdat_t.fields['t']
    data = rprof_data.data
    n_z = len(v_z)
    nphi = len(v_z[0]) - 1
    radius = list(map(float, data[0:n_z, 0]))
    if args.par_nml['geometry']['shape'].lower() == 'spherical':
        rcmb = args.par_nml['geometry']['r_cmb']
    else:
        rcmb = 0.
    dphi = 1 / nphi

    # calculing radius on the grid
    radiusgrid = len(radius) * [0]
    radiusgrid.append(1)
    for i in range(1, len(radius)):
        radiusgrid[i] = 2 * radius[i - 1] - radiusgrid[i - 1]
    for i in range(len(radiusgrid)):
        radiusgrid[i] += rcmb
    for i in range(len(radius)):
        radius[i] += rcmb

    # water profile
    water_profile = n_z * [0]
    for i_z in range(n_z):
        for phi in range(nphi):
            water_profile[i_z] += h2o[i_z, phi, 0] / nphi
    # calculing tmean
    tmean = 0
    for i_r in range(len(radius)):
        for phi in range(nphi):
            tmean += (radiusgrid[i_r + 1]**2 -
                      radiusgrid[i_r] ** 2) * dphi * tcell[i_r, phi]
    tmean /= (radiusgrid[-1]**2 - rcmb**2)

    # calculing temperature on the grid and vz_mean/v_rms
    v_rms = 0
    vz_mean = 0
    tgrid = np.zeros((n_z + 1, nphi))
    for phi in range(nphi):
        tgrid[0, phi] = 1
    for i_z in range(1, n_z):
        for phi in range(nphi):
            tgrid[i_z, phi] = (
                tcell[i_z - 1, phi] *
                (radiusgrid[i_z] - radius[i_z - 1]) + tcell[i_z, phi] *
                (-radiusgrid[i_z] + radius[i_z])) / (radius[i_z] -
                                                     radius[i_z - 1])
            v_rms += (v_z[i_z, phi, 0]**2 + v_x[i_z, phi, 0]**2) / (nphi * n_z)
            vz_mean += abs(v_z[i_z, phi, 0]) / (nphi * n_z)
    v_rms = v_rms**0.5
    print(v_rms, vz_mean)

    flux_c = n_z * [0]
    for i_z in range(1, n_z - 1):
        for phi in range(nphi):
            flux_c[i_z] += (tgrid[i_z, phi] - tmean) * \
                v_z[i_z, phi, 0] * radiusgrid[i_z] * dphi

    # checking stagnant lid
    stagnant_lid = True
    max_flx = np.max(flux_c)
    for i_z in range(n_z - n_z // 20, n_z):
        if abs(flux_c[i_z]) > max_flx / 50:
            stagnant_lid = False
            break
    if stagnant_lid:
        print('stagnant lid')
        sys.exit()
    else:
        # verifying horizontal plate speed and closeness of plates
        dvphi = nphi * [0]
        dvx_thres = 16 * v_rms

        for phi in range(0, nphi):
            dvphi[phi] = (v_x[n_z - 1, phi, 0] -
                          v_x[n_z - 1, phi - 1, 0]) / ((1 + rcmb) * dphi)
        limits = []
        for phi in range(0, nphi - nphi // 33):
            mark = True
            for i in range(phi - nphi // 33, phi + nphi // 33):
                if abs(dvphi[i]) > abs(dvphi[phi]):
                    mark = False
            if mark and abs(dvphi[phi]) >= dvx_thres:
                limits.append(phi)
        for phi in range(nphi - nphi // 33 + 1, nphi):
            mark = True
            for i in range(phi - nphi // 33 - nphi, phi + nphi // 33 - nphi):
                if abs(dvphi[i]) > abs(dvphi[phi]):
                    mark = False
            if mark and abs(dvphi[phi]) >= dvx_thres:
                limits.append(phi)
        print(limits)

        # verifying vertical speed
        k = 0
        for i in range(len(limits)):
            vzm = 0
            phi = limits[i - k]
            if phi == nphi - 1:
                for i_z in range(1, n_z):
                    vzm += (abs(v_z[i_z, phi, 0]) +
                            abs(v_z[i_z, phi - 1, 0]) +
                            abs(v_z[i_z, 0, 0])) / (n_z * 3)
            else:
                for i_z in range(0, n_z):
                    vzm += (abs(v_z[i_z, phi, 0]) +
                            abs(v_z[i_z, phi - 1, 0]) +
                            abs(v_z[i_z, phi + 1, 0])) / (n_z * 3)

            if seuil_memz != 0:
                vz_thres = vz_mean * 0.1 + seuil_memz / 2
            else:
                vz_thres = vz_mean * 0
            if vzm < vz_thres:
                limits.remove(phi)
                k += 1
        print(limits)

        print('\n')
    return limits, nphi, dvphi, vz_thres, v_x[n_z - 1, :, 0], water_profile


def detect_plates(args, velocity, age, vrms_surface, file_results, timestep, time):
    """detect plates using horizontal velocity"""
    ttransit = 1.78e15  # My
    yearins = 2.16E7

    velocityfld = velocity.fields['v']
    ph_coord = velocity.ph_coord
    agefld = age.fields['a']

    dsa = args.dsa
    # we are a bit below the surface; should check if you are in the
    # mechanical/thermal boundary layer
    indsurf = np.argmin(abs((1 - dsa) - velocity.r_coord)) - 4
    vphi = velocityfld[:, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    # velocity derivation
    dvph2 = (np.diff(vph2[indsurf, :]) / (ph_coord[0] * 2.))

    # prepare stuff to find trenches and ridges
    myorder_trench = 40
    myorder_ridge = 20  # threshold

    # finding trenches
    pom2 = deepcopy(dvph2)
    maskbigdvel = np.amin(dvph2) * 0.25  # putting threshold
    pom2[pom2 > maskbigdvel] = maskbigdvel   # user putting threshold
    argless_dv = argrelextrema(
        pom2, np.less, order=myorder_trench, mode='wrap')[0]
    trench = ph_coord[argless_dv]
    velocity_trench=vph2[indsurf,argless_dv]
    dv_trench=dvph2[argless_dv]

    # finding ridges
    pom2 = deepcopy(dvph2)
    masksmalldvel = np.amax(dvph2) * 0.2  # putting threshold
    pom2[pom2 < masksmalldvel] = masksmalldvel
    arggreat_dv = argrelextrema(
        pom2, np.greater, order=myorder_ridge, mode='wrap')[0]
    ridge = ph_coord[arggreat_dv]

    # elimination of ridges that are too close to trench
    argdel = []
    if len(trench) and len(ridge):
        for i in range(len(ridge)):
            mdistance = np.amin(abs(trench - ridge[i]))
            if mdistance < 0.016:
                argdel.append(i)
        if argdel:
            print('deleting from ridge', trench, ridge[argdel])
            ridge = np.delete(ridge, np.array(argdel))
            arggreat_dv = np.delete(arggreat_dv, np.array(argdel))

    dv_ridge=dvph2[arggreat_dv]
    age_surface = np.ma.masked_where(agefld[indsurf, :] < 0.00001, agefld[indsurf, :])
    age_surface_dim = age_surface * vrms_surface * ttransit / yearins / 1.e6
    agetrench=age_surface_dim[argless_dv] # age at the trench

    # writing the output into a file, all time steps are in one file  
    for ii in np.arange(len(trench)):
      file_results.write("%7.0f %9.2f %11.7f %10.6f %9.2f %9.2f \n" % ( \
                                     timestep,        \
                                     time,          \
                                     velocity.ti_ad,     \
                                     trench[ii],  \
                                     velocity_trench[ii], \
                                     agetrench[ii] \
                                     ))

    return trench, ridge, agetrench, dv_trench, dv_ridge 


def plot_plates(args, velocity, temp, conc, age, timestep, time, vrms_surface, trench, ridge, agetrench,dv_trench, dv_ridge, file_results_subd):
    """handle ploting stuffs"""
    plt = args.plt
    lwd = args.linewidth
    ttransit = 1.78e15  # My
    yearins = 2.16E7
    dsa = args.dsa
    plot_age = True
    velocityfld = velocity.fields['v']
    tempfld = temp.fields['t']
    concfld = conc.fields['c']
    agefld = age.fields['a']

    # if stgdat.par_type == 'vp':
    #     fld = fld[:, :, 0]
    newline = tempfld[:, 0, 0]
    tempfld = np.vstack([tempfld[:, :, 0].T, newline]).T
    newline = concfld[:, 0, 0]
    concfld = np.vstack([concfld[:, :, 0].T, newline]).T
    newline = agefld[:, 0, 0]
    agefld = np.vstack([agefld[:, :, 0].T, newline]).T

    # we are a bit below the surface; delete "-some number" to be just below
    # the surface (that is considered plane here); should check if you are in
    # the mechanical/thermal boundary layer
    indsurf = np.argmin(abs((1 - dsa) - temp.r_coord)) - 4
    # depth to detect the continents
    indcont = np.argmin(abs((1 - dsa) - np.array(velocity.r_coord))) - 10
    continents = np.ma.masked_where(
        np.logical_or(concfld[indcont, :-1] < 3, concfld[indcont, :-1] > 4),
        concfld[indcont, :-1])
    # masked array, only continents are true
    continentsall = continents / continents
    # if(vp.r_coord[indsurf]>1.-dsa):
    #    print 'WARNING lowering index for surface'
    #    indsurf=indsurf-1
    # if verbose_figures:
    # age just below the surface
    if plot_age:
        age_surface = np.ma.masked_where(
            agefld[indsurf, :] < 0.00001, agefld[indsurf, :])
        age_surface_dim = age_surface * vrms_surface * ttransit / yearins / 1.e6

    ph_coord = conc.ph_coord

    # velocity
    vphi = velocityfld[:, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    dvph2 = (np.diff(vph2[indsurf, :]) / (ph_coord[0] * 2.))
    # dvph2=dvph2/amax(abs(dvph2))  # normalization

    # plotting
    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    ax1.plot(ph_coord[:-1], concfld[indsurf, :-1],
             color='g', linewidth=lwd, label='Conc')
    ax2.plot(ph_coord[:-1], tempfld[indsurf, :-1],
             color='m', linewidth=lwd, label='Temp')
    ax3.plot(ph_coord[:-1] + ph_coord[0], dvph2,
             color='c', linewidth=lwd, label='dv')
    ax4.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')

    velocitymin=-5000
    velocitymax=5000
    ax1.fill_between(
        ph_coord[:-1], continents, 1., facecolor='#8B6914', alpha=0.2)
    ax2.fill_between(
        ph_coord[:-1], continentsall, 0., facecolor='#8B6914', alpha=0.2)
    ax2.set_ylim(0, 1)
    ax3.fill_between(
        ph_coord[:-1], continentsall * round(1.5 * np.amax(dvph2), 1),
        round(np.amin(dvph2) * 1.1, 1),  facecolor='#8B6914', alpha=0.2)
    ax3.set_ylim(
        round(np.amin(dvph2) * 1.1, 1), round(1.5 * np.amax(dvph2), 1))
    ax4.fill_between(
        ph_coord[:-1], continentsall * velocitymax,velocitymin, facecolor='#8B6914', alpha=0.2)
    ax4.set_ylim(velocitymin, velocitymax)

    ax1.set_ylabel("Concentration", fontsize=args.fontsize)
    ax2.set_ylabel("Temperature", fontsize=args.fontsize)
    ax3.set_ylabel("dv", fontsize=args.fontsize)
    ax4.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.set_title(timestep, fontsize=args.fontsize)
    ax1.text(0.95,1.07,str(round(time,0))+' My', 
             transform=ax1.transAxes, fontsize=args.fontsize)

    # topography
    fname = misc.stag_file(args, 'sc', timestep=temp.step, suffix='.dat')
    topo = np.genfromtxt(fname)
    # rescaling topography!
    topo[:, 1] = topo[:, 1] / (1. - dsa)
    topomin = -40
    topomax = 100
    # majorLocator = MultipleLocator(20)

    ax31 = ax3.twinx()
    ax31.set_ylabel("Topography [km]", fontsize=args.fontsize)
    ax31.plot(topo[:, 0], topo[:, 1] * args.par_nml['geometry']['d_dimensional']/1000., color='black', alpha=0.4)
    ax31.set_ylim(topomin, topomax)
    ax31.grid()
    ax3.scatter(trench,dv_trench,c='red')
    ax3.scatter(ridge,dv_ridge,c='green')

    for i in range(len(trench)):
        ax4.axvline(
            x=trench[i], ymin=velocitymin, ymax=velocitymax,
            color='red', ls='dashed', alpha=0.4)
    for i in range(len(ridge)):
        ax4.axvline(
            x=ridge[i], ymin=velocitymin, ymax=velocitymax,
            color='green', ls='dashed', alpha=0.4)
    ax1.set_xlim(0, 2 * np.pi)

    figname = misc.out_name(args, 'surf').format(temp.step) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close()
 
    # plotting only velocity and topography
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylim(velocitymin, velocitymax)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time,0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)

    times_subd=[]
    age_subd=[]
    distance_subd=[] 
    ph_trench_subd=[]
    ph_cont_subd=[]
    for i in range(len(trench)):
        ax1.axvline(
            x=trench[i], ymin=topomin, ymax=topomax,
            color='red', ls='dashed', alpha=0.4)
        # detection of the distance in between subduction and continent
        ph_coord_noendpoint = ph_coord[:-1]
        distancecont = min(
            abs(ph_coord_noendpoint[continentsall == 1] - trench[i]))
        argdistancecont = np.argmin(
            abs(ph_coord_noendpoint[continentsall == 1] - trench[i]))
        continentpos = ph_coord_noendpoint[continentsall == 1][argdistancecont]

        ph_trench_subd.append(trench[i])
        age_subd.append(agetrench[i])
        ph_cont_subd.append(continentpos)
        distance_subd.append(distancecont)
        times_subd.append(temp.ti_ad)

        # continent is on the left
        if (continentpos - trench[i]) < 0:
                    ax1.annotate('', xy=(trench[i] - distancecont, 2000),
                                 xycoords='data', xytext=(trench[i], 2000),
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",lw="2",
                                                 shrinkA=0, shrinkB=0))
        else:  # continent is on the right
                    ax1.annotate('', xy=(trench[i] + distancecont, 2000),
                                 xycoords='data', xytext=(trench[i], 2000),
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",lw="2",
                                                 shrinkA=0, shrinkB=0))

    for i in range(len(ridge)):
        ax1.axvline(
            x=ridge[i], ymin=topomin, ymax=topomax,
            color='green', ls='dashed', alpha=0.4)
    ax1.fill_between(
            ph_coord[:-1], continentsall * velocitymin, velocitymax,
            facecolor='#8B6914', alpha=0.2)
    ax2.set_ylabel("Topography [km]", fontsize=args.fontsize)
    ax2.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax2.plot(topo[:, 0], topo[:, 1] * args.par_nml['geometry']['d_dimensional']/1000., color='black')
    ax2.set_xlim(0, 2 * np.pi)
    dtopo = deepcopy(topo[:, 1] * args.par_nml['geometry']['d_dimensional']/1000.)
    mask = dtopo > 0
    water = deepcopy(dtopo)
    water[mask] = 0
    ax2.set_ylim(topomin, topomax)
    ax2.fill_between(
            ph_coord[:-1], continentsall * topomax, topomin,
            facecolor='#8B6914', alpha=0.2)
    for i in range(len(trench)):
        ax2.axvline(
            x=trench[i], ymin=topomin, ymax=topomax,
            color='red', ls='dashed', alpha=0.4)
    for i in range(len(ridge)):
        ax2.axvline(
            x=ridge[i], ymin=topomin, ymax=topomax,
            color='green', ls='dashed', alpha=0.4)
    ax1.set_title(timestep, fontsize=args.fontsize)
    figname = misc.out_name(args, 'surftopo').format(temp.step) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close()

    # plotting only velocity and age at surface
    if plot_age:
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
        ax1.axhline(
            y=0, xmin=0, xmax=2 * np.pi,
            color='black', ls='solid', alpha=0.2)
        ax1.set_ylim(-5000, 5000)
        ax1.set_ylabel("Velocity", fontsize=args.fontsize)
        ax1.text(0.95, 1.07, str(round(time,0)) + ' My',
                 transform=ax1.transAxes, fontsize=args.fontsize)
        agemax = 500
        agemin = -50
        ax1.fill_between(
            ph_coord[:-1], continentsall * velocitymax, velocitymin,
            facecolor='#8B6914', alpha=0.2)
        for i in range(len(trench)):
            ax1.axvline(
                x=trench[i], ymin=agemin, ymax=agemax,
                color='red', ls='dashed', alpha=0.4)

            # continent is on the left
            if (ph_cont_subd[i] - trench[i]) < 0:
                    ax1.annotate('', xy=(trench[i] - distance_subd[i], 2000),
                                 xycoords='data', xytext=(trench[i], 2000),
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",lw="2",
                                                 shrinkA=0, shrinkB=0))
            else:  # continent is on the right
                    ax1.annotate('', xy=(trench[i] + distance_subd[i], 2000),
                                 xycoords='data', xytext=(trench[i], 2000),
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",lw="2",
                                                 shrinkA=0, shrinkB=0))
        for i in range(len(ridge)):
            ax1.axvline(
                x=ridge[i], ymin=agemin, ymax=agemax,
                color='green', ls='dashed', alpha=0.4)

        ax2.set_ylabel("Age [My]",fontsize=args.fontsize)
        # in dimensions
        ax2.plot(ph_coord[:-1], age_surface_dim[:-1], color='black')
        ax2.set_xlim(0, 2 * np.pi)
        ax2.fill_between(
            ph_coord[:-1], continentsall * agemax, agemin,
            facecolor='#8B6914', alpha=0.2)
        ax2.set_ylim(agemin, agemax)
        for i in range(len(trench)):
            ax2.axvline(
                x=trench[i], ymin=agemin, ymax=agemax,
                color='red', ls='dashed', alpha=0.4)
        for i in range(len(ridge)):
            ax2.axvline(
                x=ridge[i], ymin=agemin, ymax=agemax,
                color='green', ls='dashed', alpha=0.4)
        ax1.set_title(timestep, fontsize=args.fontsize)
        figname = misc.out_name(args, 'surfage').format(temp.step) + '.pdf'
        plt.savefig(figname, format='PDF')
        plt.close()


    # writing the output into a file, all time steps are in one file
    for ii in np.arange(len(distance_subd)):
      file_results_subd.write("%6.0f %11.7f %10.6f %10.6f %10.6f %11.3f\n" % ( \
                                     timestep,        \
                                     times_subd[ii],     \
                                     distance_subd[ii], \
                                     ph_trench_subd[ii],  \
                                     ph_cont_subd[ii], \
                                     age_subd[ii], \
                                     ))

    return None


def plates_cmd(args):
    """find positions of trenches and subductions

    uses velocity field (velocity derivation)
    plots the number of plates over a designated lapse of time
    """
    
    ttransit = 1.78e15  # My; Earth transit time
    yearins = 2.16E7

    if args.vzcheck:
        seuil_memz = 0
        nb_plates = []
        timedat = TimeData(args)
        slc = slice(*(i * args.par_nml['ioin']['save_file_framestep']
                      for i in args.timestep))
        time, ch2o = timedat.data[:, 1][slc], timedat.data[:, 27][slc]
    else:
        file_results = open('results_plate_velocity_{}_{}_{}.dat'.format(*args.timestep),'w')
        file_results.write('# it  timestep  time  ph_trench vel_trench age_trench\n')
        file_results_subd = open('results_distance_subd_{}_{}_{}.dat'.format(*args.timestep),'w')
        file_results_subd.write('#  it      time      distance     ph_trench     ph_cont  age_trench \n')

    for timestep in range(*args.timestep):
        velocity = BinData(args, 'v', timestep)
        temp = BinData(args, 't', timestep)
        rprof_data = RprofData(args)
        print('Treating timestep',timestep)
        if args.vzcheck:
            water = BinData(args, 'h', timestep)
            rprof_data = RprofData(args)
            plt = args.plt
            limits, nphi, dvphi, seuil_memz, vphi_surf, water_profile =\
                detect_plates_vzcheck(temp, velocity, water, rprof_data,
                                      args, seuil_memz)
            limits.sort()
            sizeplates = [limits[0] + nphi - limits[-1]]
            for lim in range(1, len(limits)):
                sizeplates.append(limits[lim] - limits[lim - 1])
            lim = len(limits) * [max(dvphi)]
            plt.figure(timestep)
            plt.subplot(221)
            plt.axis([0, len(velocity.fields['w'][0]) - 1,
                      np.min(vphi_surf) * 1.2, np.max(vphi_surf) * 1.2])
            plt.plot(vphi_surf)
            plt.subplot(223)
            plt.axis(
                [0, len(velocity.fields['w'][0]) - 1,
                 np.min(dvphi) * 1.2, np.max(dvphi) * 1.2])
            plt.plot(dvphi)
            plt.scatter(limits, lim, color='red')
            plt.subplot(222)
            plt.hist(sizeplates, 10, (0, nphi / 2))
            plt.subplot(224)
            plt.plot(water_profile)
            plt.savefig('plates' + str(timestep) + '.pdf', format='PDF')

            nb_plates.append(len(limits))
            plt.close(timestep)
        else:
            conc = BinData(args, 'c', timestep)
            viscosity = BinData(args, 'n', timestep)
            age = BinData(args, 'a', timestep)
            rcmb = viscosity.rcmb

            if timestep == args.timestep[0]:
                # calculating averaged horizontal surface velocity
                # needed for redimensionalisation
                # using mean profiles
                data, tsteps = rprof_data.data, rprof_data.tsteps
                meta = constants.RPROF_VAR_LIST['u']
                cols = [meta.prof_idx]
                chunks = lambda mydata, nbz: [mydata[ii:ii+nbz] for ii in range(0, len(mydata), nbz)]
                nztot = int(np.shape(data)[0] / (np.shape(tsteps)[0]))
                radius = np.array(chunks(np.array(data[:, 0], float) + rcmb,nztot))
                donnee = np.array(data[:, cols], float)
                donnee_chunk = chunks(donnee, nztot)
                donnee_averaged = np.mean(donnee_chunk, axis = 0)
                if args.par_nml['boundaries']['air_layer']:
                       dsa = args.par_nml['boundaries']['air_thickness']
                       myarg = np.argmin(abs(radius[0,:] - radius[0,-1] + dsa))
                else:
                       myarg=-1
                vrms_surface = donnee_averaged[myarg,0]
  
            time = temp.ti_ad * vrms_surface * ttransit / yearins / 1.e6
            trenches, ridges, agetrenches, dv_trench, dv_ridge =\
                detect_plates(args, velocity, age, vrms_surface, file_results, 
                              timestep, time)
            plot_plates(args, velocity, temp, conc, age, timestep, time, 
                        vrms_surface,trenches, ridges, agetrenches,
                        dv_trench, dv_ridge, file_results_subd)

            # plot viscosity field with position of trenches and ridges
            fig, axis = plot_scalar(args, viscosity, 'n')
            args.plt.figure(fig.number)
            axis.text(1., 0.9, str(round(time,0)) + ' My',
                      transform=axis.transAxes, fontsize=args.fontsize)
            for ii in np.arange(len(trenches)):
                    xx = (viscosity.rcmb + 1.02) * np.cos(trenches[ii]) 
                    yy = (viscosity.rcmb + 1.02) * np.sin(trenches[ii]) 
                    xxt = (viscosity.rcmb + 1.35) * np.cos(trenches[ii]) 
                    yyt = (viscosity.rcmb + 1.35) * np.sin(trenches[ii]) 
                    axis.annotate('', xy=(xx,yy), xytext=(xxt,yyt),
                                  arrowprops=dict(facecolor='red',shrink=0.05))
            for ii in np.arange(len(ridges)):
                    xx = (viscosity.rcmb + 1.02) * np.cos(ridges[ii]) 
                    yy = (viscosity.rcmb + 1.02) * np.sin(ridges[ii]) 
                    xxt = (viscosity.rcmb + 1.35) * np.cos(ridges[ii]) 
                    yyt = (viscosity.rcmb + 1.35) * np.sin(ridges[ii]) 
                    axis.annotate('', xy=(xx,yy), xytext=(xxt,yyt),
                                  arrowprops=dict(facecolor='green',shrink=0.05))
            args.plt.tight_layout()
            args.plt.savefig(
                misc.out_name(args, 'n').format(viscosity.step) + '.pdf',
                format='PDF')
            args.plt.close(fig)

    if args.timeprofile and args.vzcheck:
        for i in range(2, len(nb_plates) - 3):
            nb_plates[i] = (nb_plates[i - 2] + nb_plates[i - 1] +
                            nb_plates[i] + nb_plates[i + 1] +
                            nb_plates[i + 2]) / 5
        plt.figure(-1)
        plt.subplot(121)
        plt.axis([time[0], time[-1], 0, np.max(nb_plates)])
        plt.plot(time, nb_plates)
        plt.subplot(122)
        plt.plot(time, ch2o)
        plt.savefig('plates_{}_{}_{}.pdf'.format(*args.timestep),
                    format='PDF')
        plt.close(-1)
    else:
        file_results.close()
        file_results_subd.close()
