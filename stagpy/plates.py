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
import os.path


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
        sys.Exit()
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


def detect_plates(args, velocity, age, vrms_surface,
                  fids, timestep, time):
    """detect plates using derivative of horizontal velocity"""

    velocityfld = velocity.fields['v']
    ph_coord = velocity.ph_coord
    if args.plot_age:
        agefld = age.fields['a']
    else:
        agefld = []

    if args.par_nml['boundaries']['air_layer']:
        dsa = args.par_nml['boundaries']['air_thickness']
        # we are a bit below the surface; should check if you are in the
        # thermal boundary layer
        indsurf = np.argmin(abs((1 - dsa) - velocity.r_coord)) - 4
    else:
        dsa = 0.
        indsurf = -1

    vphi = velocityfld[:, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    # velocity derivation
    dvph2 = (np.diff(vph2[indsurf, :]) / (ph_coord[0] * 2.))

    io_surface(timestep, time, fids[6], dvph2)
    io_surface(timestep, time, fids[7], vph2[indsurf, :-1])

    # prepare stuff to find trenches and ridges
    if args.par_nml['boundaries']['air_layer']:
        myorder_trench = 15
    else:
        myorder_trench = 10
    myorder_ridge = 20  # threshold

    # finding trenches
    pom2 = deepcopy(dvph2)
    if args.par_nml['boundaries']['air_layer']:
        maskbigdvel = -30 * vrms_surface  # np.amin(dvph2) * 0.1  #  threshold
    else:
        maskbigdvel = -10 * vrms_surface  # np.amin(dvph2) * 0.1  #  threshold
    pom2[pom2 > maskbigdvel] = maskbigdvel   # putting threshold
    argless_dv = argrelextrema(
        pom2, np.less, order=myorder_trench, mode='wrap')[0]
    trench = ph_coord[argless_dv]
    velocity_trench = vph2[indsurf, argless_dv]
    dv_trench = dvph2[argless_dv]

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

    dv_ridge = dvph2[arggreat_dv]
    if args.plot_age:
        age_surface = np.ma.masked_where(agefld[indsurf, :] < 0.00001,
                                         agefld[indsurf, :])
        age_surface_dim = age_surface * vrms_surface * args.ttransit / args.yearins / 1.e6
        agetrench = age_surface_dim[argless_dv]  # age at the trench
    else:
        agetrench = np.zeros(len(argless_dv))

    # writing the output into a file, all time steps are in one file
    for itrench in np.arange(len(trench)):
        fids[0].write("%7.0f %11.7f %10.6f %9.2f %9.2f \n" % (
            timestep,
            velocity.ti_ad,
            trench[itrench],
            velocity_trench[itrench],
            agetrench[itrench]
        ))

    return trench, ridge, agetrench, dv_trench, dv_ridge


def plot_plate_limits(args, fig, axis, ridge, trench, ymin, ymax):
    """plot lines designating ridges and trenches"""

    args.plt.figure(fig.number)
    for i in range(len(trench)):
        axis.axvline(
            x=trench[i], ymin=ymin, ymax=ymax,
            color='red', ls='dashed', alpha=0.4)
    for i in range(len(ridge)):
        axis.axvline(
            x=ridge[i], ymin=ymin, ymax=ymax,
            color='green', ls='dashed', alpha=0.4)
    axis.set_xlim(0, 2 * np.pi)
    axis.set_ylim(ymin, ymax)


def plot_plate_limits_field(args, fig, ax, ridge, trench):
    """plot arrows designating ridges and trenches in 2D field plots"""

    args.plt.figure(fig.number)
    if args.par_nml['geometry']['shape'].lower() == 'spherical':
        rcmb = args.par_nml['geometry']['r_cmb']
    else:
        rcmb = 0.

    for itr in np.arange(len(trench)):
        xxd = (rcmb + 1.02) * np.cos(trench[itr])  # arrow begin
        yyd = (rcmb + 1.02) * np.sin(trench[itr])  # arrow begin
        xxt = (rcmb + 1.35) * np.cos(trench[itr])  # arrow end
        yyt = (rcmb + 1.35) * np.sin(trench[itr])  # arrow end
        ax.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                    arrowprops=dict(facecolor='red', shrink=0.05))
    for iri in np.arange(len(ridge)):
        xxd = (rcmb + 1.02) * np.cos(ridge[iri])
        yyd = (rcmb + 1.02) * np.sin(ridge[iri])
        xxt = (rcmb + 1.35) * np.cos(ridge[iri])
        yyt = (rcmb + 1.35) * np.sin(ridge[iri])
        ax.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                    arrowprops=dict(facecolor='green', shrink=0.05))


def plot_plates(args, velocity, temp, conc, age, stress, timestep,
                time, vrms_surface, trench, ridge, agetrench, topo,
                dv_trench, dv_ridge, fids):
    """handle ploting stuff"""

    plot_age = args.plot_age
    dimensions = True

    if dimensions:
        l_scale = args.par_nml['geometry']['d_dimensional'] / 1000.  # km
    else:
        l_scale = 1.0

    if args.par_nml['boundaries']['air_layer']:
        dsa = args.par_nml['boundaries']['air_thickness']
    else:
        dsa = 0.

    plt = args.plt
    lwd = args.linewidth
    velocityfld = velocity.fields['v']
    tempfld = temp.fields['t']
    concfld = conc.fields['c']
    if args.plot_age:
        agefld = age.fields['a']
        newline = agefld[:, 0, 0]
        agefld = np.vstack([agefld[:, :, 0].T, newline]).T
    if args.plot_stress:
        stressfld = stress.fields['s']

    # if stgdat.par_type == 'vp':
    #     fld = fld[:, :, 0]
    newline = tempfld[:, 0, 0]
    tempfld = np.vstack([tempfld[:, :, 0].T, newline]).T
    newline = concfld[:, 0, 0]
    concfld = np.vstack([concfld[:, :, 0].T, newline]).T

    if args.par_nml['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        # the surface (that is considered plane here); should check if you are
        # in the thermal boundary layer
        indsurf = np.argmin(abs((1 - dsa) - temp.r_coord)) - 4
        # depth to detect the continents
        indcont = np.argmin(abs((1 - dsa) - np.array(velocity.r_coord))) - 10
    else:
        indsurf = -1
        indcont = -1  # depth to detect continents

    if args.par_nml['boundaries']['air_layer'] and not args.par_nml['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[indcont, :-1] < 3,
                          concfld[indcont, :-1] > 4),
            concfld[indcont, :-1])
    elif args.par_nml['boundaries']['air_layer'] and args.par_nml['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[indcont, :-1] < 3,
                          concfld[indcont, :-1] > 5),
            concfld[indcont, :-1])
    elif args.par_nml['tracersin']['tracers_weakcrust']:
        continents = np.ma.masked_where(
            concfld[indcont, :-1] < 3, concfld[indcont, :-1])
    else:
        continents = np.ma.masked_where(
            concfld[indcont, :-1] < 2, concfld[indcont, :-1])

    # masked array, only continents are true
    continentsall = continents / continents
    # if(vp.r_coord[indsurf]>1.-dsa):
    #    print 'WARNING lowering index for surface'
    #    indsurf=indsurf-1

    if plot_age:
        age_surface = np.ma.masked_where(
            agefld[indsurf, :] < 0.00001, agefld[indsurf, :])
        age_surface_dim =\
            age_surface * vrms_surface * args.ttransit / args.yearins / 1.e6

    ph_coord = conc.ph_coord

    # velocity
    vphi = velocityfld[:, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    dvph2 = (np.diff(vph2[indsurf, :]) / (ph_coord[0] * 2.))
    # dvph2=dvph2/amax(abs(dvph2))  # normalization

    # plotting
    fig0, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], concfld[indsurf, :-1],
             color='g', linewidth=lwd, label='Conc')
    ax2.plot(ph_coord[:-1], tempfld[indsurf, :-1],
             color='k', linewidth=lwd, label='Temp')
    ax3.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')

    ax1.fill_between(
        ph_coord[:-1], continents, 1., facecolor='#8B6914', alpha=0.2)
    ax2.fill_between(
        ph_coord[:-1], continentsall, 0., facecolor='#8B6914', alpha=0.2)

    if args.par_nml['boundaries']['topT_mode'] == 'iso':
        tempmin = args.par_nml['boundaries']['topT_val'] * 0.9
    else:
        tempmin = 0.0
    if args.par_nml['boundaries']['botT_mode'] == 'iso':
        tempmax = args.par_nml['boundaries']['botT_val'] * 0.35
    else:
        tempmax = 0.8

    ax2.set_ylim(tempmin, tempmax)
    ax3.fill_between(
        ph_coord[:-1], continentsall * round(1.5 * np.amax(dvph2), 1),
        round(np.amin(dvph2) * 1.1, 1), facecolor='#8B6914', alpha=0.2)
    ax3.set_ylim(args.velocitymin, args.velocitymax)

    ax1.set_ylabel("Concentration", fontsize=args.fontsize)
    ax2.set_ylabel("Temperature", fontsize=args.fontsize)
    ax3.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.set_title(timestep, fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax1.text(0.01, 1.07, str(round(temp.ti_ad, 8)),
             transform=ax1.transAxes, fontsize=args.fontsize)

    plot_plate_limits(args, fig0, ax3, ridge, trench,
                      args.velocitymin, args.velocitymax)

    figname = misc.out_name(args, 'sveltempconc').format(temp.step) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close(fig0)

    # plotting velocity and velocity derivative
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax1.text(0.01, 1.07, str(round(temp.ti_ad, 8)),
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax2.plot(ph_coord[:-1] + ph_coord[0], dvph2,
             color='k', linewidth=lwd, label='dv')
    ax2.set_ylabel("dv", fontsize=args.fontsize)

    plot_plate_limits(args, fig0, ax1, ridge, trench,
                      args.velocitymin, args.velocitymax)
    plot_plate_limits(args, fig0, ax2, ridge, trench,
                      args.dvelocitymin, args.dvelocitymax)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_title(timestep, fontsize=args.fontsize)

    ax1.fill_between(
        ph_coord[:-1], continentsall * args.velocitymin, args.velocitymax,
        facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(args.velocitymin, args.velocitymax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * args.dvelocitymin, args.dvelocitymax,
        facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(args.dvelocitymin, args.dvelocitymax)

    figname = misc.out_name(args, 'sveldvel').format(temp.step) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close(fig0)

    # plotting velocity and second invariant of stress
    if args.plot_stress:
        fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
        ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                    color='black', ls='solid', alpha=0.2)
        ax1.set_ylabel("Velocity", fontsize=args.fontsize)
        ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax1.transAxes, fontsize=args.fontsize)
        ax1.text(0.01, 1.07, str(round(temp.ti_ad, 8)),
                 transform=ax1.transAxes, fontsize=args.fontsize)
        ax2.plot(ph_coord[:-1], stressfld[indsurf, :],
                 color='k', linewidth=lwd, label='Stress')
        ax2.set_ylim(args.stressmin, args.stressmax)
        ax2.set_ylabel("Stress [MPa]", fontsize=args.fontsize)

        plot_plate_limits(args, fig0, ax1, ridge, trench,
                          args.velocitymin, args.velocitymax)
        plot_plate_limits(args, fig0, ax2, ridge, trench,
                          args.stressmin, args.stressmax)
        ax1.set_xlim(0, 2 * np.pi)
        ax1.set_title(timestep, fontsize=args.fontsize)

        ax1.fill_between(
            ph_coord[:-1], continentsall * args.velocitymin, args.velocitymax,
            facecolor='#8B6914', alpha=0.2)
        ax1.set_ylim(args.velocitymin, args.velocitymax)
        ax2.fill_between(
            ph_coord[:-1], continentsall * args.dvelocitymin, args.dvelocitymax,
            facecolor='#8B6914', alpha=0.2)

        figname = misc.out_name(args, 'svelstress').format(temp.step) + '.pdf'
        plt.savefig(figname, format='PDF')
        plt.close(fig0)

    # plotting velocity and topography
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylim(args.velocitymin, args.velocitymax)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    plot_plate_limits(args, fig1, ax1, ridge, trench,
                      args.velocitymin, args.velocitymax)

    # plotting velocity and age at surface
    if plot_age:
        fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax3.plot(ph_coord[:-1], vph2[indsurf, :-1], linewidth=lwd, label='Vel')
        ax3.axhline(
            y=0, xmin=0, xmax=2 * np.pi,
            color='black', ls='solid', alpha=0.2)
        ax3.set_ylim(args.velocitymin, args.velocitymax)
        ax3.set_ylabel("Velocity", fontsize=args.fontsize)
        ax3.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax3.transAxes, fontsize=args.fontsize)
        ax3.fill_between(
            ph_coord[:-1], continentsall * args.velocitymax, args.velocitymin,
            facecolor='#8B6914', alpha=0.2)
        plot_plate_limits(args, fig2, ax3, ridge, trench,
                          args.velocitymin, args.velocitymax)

    times_subd = []
    age_subd = []
    distance_subd = []
    ph_trench_subd = []
    ph_cont_subd = []
    for i in range(len(trench)):
        # detection of the distance in between subduction and continent
        ph_coord_noendpoint = ph_coord[:-1]
        # angdistance = 2.*np.arcsin(abs(np.sin(
        # 0.5 * (ph_coord_noendpoint[continentsall == 1] - trench[i]))))
        # distancecont = min(angdistance)
        # print(distancecont)
        angdistance1 = abs(ph_coord_noendpoint[continentsall == 1] - trench[i])
        angdistance2 = 2. * np.pi - angdistance1
        angdistance = np.minimum(angdistance1, angdistance2)
        distancecont = min(angdistance)
        # print(distancecont)
        # distancecont = min(
        #    abs(ph_coord_noendpoint[continentsall == 1] - trench[i]))
        # print(distancecont)
        argdistancecont = np.argmin(angdistance)
        # argdistancecont = np.argmin(
        #    abs(ph_coord_noendpoint[continentsall == 1] - trench[i]))
        continentpos = ph_coord_noendpoint[continentsall == 1][argdistancecont]

        ph_trench_subd.append(trench[i])
        age_subd.append(agetrench[i])
        ph_cont_subd.append(continentpos)
        distance_subd.append(distancecont)
        times_subd.append(temp.ti_ad)

        # continent is on the left
        if angdistance1[argdistancecont] < angdistance2[argdistancecont]:
            if continentpos - trench[i] < 0:
                ax1.annotate('', xy=(trench[i] - distancecont, 2000),
                             xycoords='data', xytext=(trench[i], 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))
            else:  # continent is on the right
                ax1.annotate('', xy=(trench[i] + distancecont, 2000),
                             xycoords='data', xytext=(trench[i], 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))
        else:  # distance over boundary
            if continentpos - trench[i] < 0:
                ax1.annotate('', xy=(2. * np.pi, 2000),
                             xycoords='data', xytext=(trench[i], 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="-", lw="2",
                                             shrinkA=0, shrinkB=0))
                ax1.annotate('', xy=(continentpos, 2000),
                             xycoords='data', xytext=(0, 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))
            else:
                ax1.annotate('', xy=(0, 2000),
                             xycoords='data', xytext=(trench[i], 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="-", lw="2",
                                             shrinkA=0, shrinkB=0))
                ax1.annotate('', xy=(continentpos, 2000),
                             xycoords='data', xytext=(2. * np.pi, 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))

    ax1.fill_between(
        ph_coord[:-1], continentsall * args.velocitymin, args.velocitymax,
        facecolor='#8B6914', alpha=0.2)
    ax2.set_ylabel("Topography [km]", fontsize=args.fontsize)
    ax2.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax2.plot(topo[:, 0],
             topo[:, 1] * l_scale,
             color='black')
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(args.topomin, args.topomax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * args.topomax, args.topomin,
        facecolor='#8B6914', alpha=0.2)
    plot_plate_limits(args, fig1, ax2, ridge, trench,
                      args.topomin, args.topomax)
    ax1.set_title(timestep, fontsize=args.fontsize)
    figname = misc.out_name(args, 'sveltopo').format(temp.step) + '.pdf'
    fig1.savefig(figname, format='PDF')
    plt.close(fig1)

    if plot_age:
        ax4.set_ylabel("Seafloor age [My]", fontsize=args.fontsize)
        # in dimensions
        ax4.plot(ph_coord[:-1], age_surface_dim[:-1], color='black')
        ax4.set_xlim(0, 2 * np.pi)
        ax4.fill_between(
            ph_coord[:-1], continentsall * args.agemax, args.agemin,
            facecolor='#8B6914', alpha=0.2)
        ax4.set_ylim(args.agemin, args.agemax)
        plot_plate_limits(args, fig2, ax4, ridge, trench,
                          args.agemin, args.agemax)
        ax3.set_title(timestep, fontsize=args.fontsize)
        figname = misc.out_name(args, 'svelage').format(temp.step) + '.pdf'
        fig2.savefig(figname, format='PDF')
        plt.close(fig2)

    # writing the output into a file, all time steps are in one file
    for isubd in np.arange(len(distance_subd)):
        fids[1].write("%6.0f %11.7f %11.3f %10.6f %10.6f %10.6f %11.3f\n" % (
            timestep,
            times_subd[isubd],
            time,
            distance_subd[isubd],
            ph_trench_subd[isubd],
            ph_cont_subd[isubd],
            age_subd[isubd],
        ))

    return None


def io_surface(timestep, time, fid, fld):
    fid.write("{} {}".format(timestep, time))
    fid.writelines(["%10.2e" % item for item in fld[:]])
    fid.writelines(["\n"])


def lithospheric_stress(args, temp, conc, stress, velocity, trench, ridge, timestep, time):
    """ calculate stress in the lithosphere """
    plt = args.plt
    lwd = args.linewidth
    base_lith = temp.rcmb + 1 - 0.105

    tempfld = temp.fields['t'][:, :, 0].T
    stressfld = stress.fields['s'][:, :, 0].T

    r_mesh, ph_mesh = np.meshgrid(
        temp.r_coord + temp.rcmb, temp.ph_coord,
        indexing='ij')

    temp.fields['t'] = np.ma.masked_where(r_mesh[:, :-1].T < base_lith, tempfld)
    stressfld = np.ma.masked_where(r_mesh[:, :-1].T < base_lith, stressfld)

    # stress integration in the lithosphere
    ddim = args.par_nml['geometry']['d_dimensional']
    dzm = np.array((stress.r_coord[1:]-stress.r_coord[:-1]) * ddim)
    # dzlith=(dzm[stress.r_coord[1:]>(base_lith-stress.rcmb)])
    stress_lith = np.sum((stressfld[:, 1:]*dzm.T) / 1e6, axis=1)
    ph_coord = stress.ph_coord

    # plot stress in the lithosphere
    fig, axis = args.plt.subplots(ncols=1)
    xmesh, ymesh = stress.x_mesh[0, :-1, :], stress.y_mesh[0, :-1, :]
    surf = axis.pcolormesh(xmesh, ymesh, stressfld, cmap='gnuplot2_r',
                           rasterized=not args.pdf,
                           shading='gouraud')
    surf.set_clim(vmin=0, vmax=300)
    cbar = plt.colorbar(surf, shrink=args.shrinkcb)
    cbar.set_label(constants.FIELD_VAR_LIST['s'].name)
    plt.axis('equal')
    plt.axis('off')
    args.plt.figure(fig.number)

    # Annotation with time and step
    axis.text(1., 0.9, str(round(time, 0)) + ' My',
              transform=axis.transAxes, fontsize=args.fontsize)
    axis.text(1., 0.1, str(timestep),
              transform=axis.transAxes, fontsize=args.fontsize)
    args.plt.savefig(
        misc.out_name(args, 'lith').format(temp.step) + '.pdf',
        format='PDF')

    # velocity
    vphi = velocity.fields['v'][:, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi

    # position of continents
    concfld = conc.fields['c']
    newline = concfld[:, 0, 0]
    concfld = np.vstack([concfld[:, :, 0].T, newline]).T

    if args.par_nml['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        dsa = args.par_nml['boundaries']['air_thickness']
        # depth to detect the continents
        indcont = np.argmin(abs((1 - dsa) - np.array(velocity.r_coord))) - 10
    else:
        # depth to detect continents
        indcont = -1
    if args.par_nml['boundaries']['air_layer'] and not args.par_nml['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[indcont, :-1] < 3,
                          concfld[indcont, :-1] > 4),
            concfld[indcont, :-1])
    elif args.par_nml['boundaries']['air_layer'] and args.par_nml['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[indcont, :-1] < 3,
                          concfld[indcont, :-1] > 5),
            concfld[indcont, :-1])
    elif args.par_nml['tracersin']['tracers_weakcrust']:
        continents = np.ma.masked_where(
            concfld[indcont, :-1] < 3, concfld[indcont, :-1])
    else:
        continents = np.ma.masked_where(
            concfld[indcont, :-1] < 2, concfld[indcont, :-1])

    # masked array, only continents are true
    continentsall = continents / continents

    # plot integrated stress in the lithosphere
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[-1, :-1], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax1.text(0.01, 1.07, str(round(temp.ti_ad, 8)),
             transform=ax1.transAxes, fontsize=args.fontsize)

    ax2.plot(ph_coord[:-1], stress_lith,
             color='k', linewidth=lwd, label='Stress')
    ax2.set_ylabel(r"Integrated stress [$TN\,m^{-1}$]", fontsize=args.fontsize)

    plot_plate_limits(args, fig0, ax1, ridge, trench,
                      args.velocitymin, args.velocitymax)
    plot_plate_limits(args, fig0, ax2, ridge, trench,
                      args.stressmin, args.lstressmax)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_title(timestep, fontsize=args.fontsize)

    ax1.fill_between(
        ph_coord[:-1], continentsall * args.velocitymin, args.velocitymax,
        facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(args.velocitymin, args.velocitymax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * args.stressmin, args.lstressmax,
        facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(args.stressmin, args.lstressmax)

    figname = misc.out_name(args, 'svelslith').format(temp.step) + '.pdf'
    fig0.savefig(figname, format='PDF')

    return None


def plates_cmd(args):
    """find positions of trenches and subductions

    uses velocity field (velocity derivation)
    plots the number of plates over a designated lapse of time
    """

    if not args.vzcheck:
        # check if the files with results exist
        if os.path.exists('results_plate_velocity_{}_{}_{}.dat'.format(*args.timestep)):
            print(' *WARNING* ')
            print(' The files with results',
                  'results_distance_subd_{}_{}_{}.dat'.format(*args.timestep),
                  'cannot be overwritten')
            print(' Exiting the code ')
            sys.exit()

        fids = [0 for _ in range(8)]
        with open('results_plate_velocity_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[0],\
                open('results_distance_subd_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[1],\
                open('results_continents_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[2],\
                open('results_flux_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[3],\
                open('results_topography_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[4],\
                open('results_age_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[5],\
                open('results_velderiv_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[6],\
                open('results_velocity_{}_{}_{}.dat'.format(
                *args.timestep), 'w') as fids[7]:

            fids[0].write('#  it  time  ph_trench vel_trench age_trench\n')
            fids[1].write('#  it      time   time [My]   distance     ph_trench     ph_cont  age_trench [My] \n')

            for timestep in range(*args.timestep):
                print('Treating timestep', timestep)

                velocity = BinData(args, 'v', timestep)
                temp = BinData(args, 't', timestep)
                rprof_data = RprofData(args)
                conc = BinData(args, 'c', timestep)
                viscosity = BinData(args, 'n', timestep)
                viscosityfld = viscosity.fields['n']
                newline = viscosityfld[:, 0, 0]
                viscosityfld = np.vstack([viscosityfld[:, :, 0].T, newline])
                rcmb = viscosity.rcmb
                if args.plot_age:
                    age = BinData(args, 'a', timestep)
                else:
                    age = []
                if args.plot_stress:
                    stress = BinData(args, 's', timestep)
                    stressfld = stress.fields['s']
                    newline = stressfld[:, 0, 0]
                    stressfld = np.vstack([stressfld[:, :, 0].T, newline])
                    stressdim = stress
                    stressdim.fields['s'] = stressdim.fields['s'] * \
                        args.kappa * args.viscosity_ref / args.mantle**2 / 1.e6
                else:
                    stress = []

                if timestep == args.timestep[0]:
                    # calculating averaged horizontal surface velocity
                    # needed for redimensionalisation
                    # using mean profiles
                    data, tsteps = rprof_data.data, rprof_data.tsteps
                    meta = constants.RPROF_VAR_LIST['u']
                    cols = [meta.prof_idx]

                    def chunks(mydata, nbz):
                        """Divide vector mydata into an array"""
                        return [mydata[ii:ii + nbz]
                                for ii in range(0, len(mydata), nbz)]
                    nztot = int(np.shape(data)[0] / (np.shape(tsteps)[0]))
                    radius = np.array(chunks(np.array(data[:, 0], float) + rcmb,
                                             nztot))
                    donnee = np.array(data[:, cols], float)
                    donnee_chunk = chunks(donnee, nztot)
                    donnee_averaged = np.mean(donnee_chunk, axis=0)
                    if args.par_nml['boundaries']['air_layer']:
                        dsa = args.par_nml['boundaries']['air_thickness']
                        myarg = np.argmin(abs(radius[0, :] - radius[0, -1] + dsa))
                    else:
                        myarg = -1
                    vrms_surface = donnee_averaged[myarg, 0]

                    if args.par_nml['boundaries']['air_layer']:
                        dsa = args.par_nml['boundaries']['air_thickness']
                        isurf = np.argmin(abs((1 - dsa) - temp.r_coord)) - 4
                    else:
                        isurf = -1

                # topography
                fname = misc.stag_file(args, 'sc', timestep=temp.step, suffix='.dat')
                topo = np.genfromtxt(fname)
                # rescaling topography!
                if args.par_nml['boundaries']['air_layer']:
                    topo[:, 1] = topo[:, 1] / (1. - dsa)

                time = temp.ti_ad * vrms_surface * args.ttransit / args.yearins / 1.e6
                trenches, ridges, agetrenches, dv_trench, dv_ridge =\
                    detect_plates(args, velocity,
                                  age, vrms_surface,
                                  fids, timestep, time)
                plot_plates(args, velocity, temp, conc, age, stress, timestep,
                            time, vrms_surface, trenches, ridges, agetrenches,
                            topo, dv_trench, dv_ridge, fids)

                # prepare for continent plotting
                concfld = conc.fields['c']
                newline = concfld[:, 0, 0]
                concfld = np.vstack([concfld[:, :, 0].T, newline])
                continentsfld = np.ma.masked_where(
                    concfld < 3, concfld)  # plotting continents, to-do
                continentsfld = continentsfld / continentsfld

                io_surface(timestep, time, fids[2], concfld[:-1, isurf])
                tgrad = (temp.fields['t'][isurf-1, :] -
                         temp.fields['t'][isurf, :]) / \
                        (temp.r_coord[isurf] - temp.r_coord[isurf - 1])
                io_surface(timestep, time, fids[3], tgrad[:, 0])
                io_surface(timestep, time, fids[4], topo[:, 1])
                if args.plot_age:
                    io_surface(timestep, time, fids[5], age.fields['a'][isurf, :])

                # plot viscosity field with position of trenches and ridges
                fig, axis, surf = plot_scalar(args, viscosity, 'n')
                etamax = args.par_nml['viscosity']['eta_max']
                surf.set_clim(vmin=1e-2, vmax=etamax)

                args.plt.figure(fig.number)
                plt = args.plt

                # plotting continents
                xmesh, ymesh = conc.x_mesh[0, :, :], conc.y_mesh[0, :, :]
                surf2 = axis.pcolormesh(xmesh, ymesh, continentsfld,
                                        rasterized=not args.pdf, cmap='cool_r',
                                        vmin=0, vmax=0, shading='goaround')
                cmap2 = args.plt.cm.ocean
                cmap2.set_over('m')

                # plotting velocity vectors
                vphi = velocity.fields['v'][:, :, 0]
                velr = velocity.fields['w'][:, :, 0]
                r_mesh, ph_mesh = np.meshgrid(
                    velocity.r_coord + velocity.rcmb, velocity.ph_coord,
                    indexing='ij')
                velx = -vphi * np.sin(ph_mesh) + velr * np.cos(ph_mesh)
                vely = vphi * np.cos(ph_mesh) + velr * np.sin(ph_mesh)
                # xmesh, ymesh = velocity.x_mesh[0, :, :], velocity.y_mesh[0, :, :]
                step = np.int(np.size(ph_mesh[0, :]) / 100.)
                axis.quiver(xmesh[::step, ::step], ymesh[::step, ::step],
                            velx[::step, ::step].T, vely[::step, ::step].T)

                # Annotation with time and step
                axis.text(1., 0.9, str(round(time, 0)) + ' My',
                          transform=axis.transAxes, fontsize=args.fontsize)
                axis.text(1., 0.1, str(timestep),
                          transform=axis.transAxes, fontsize=args.fontsize)

                # Put arrow where ridges and trenches are
                plot_plate_limits_field(args, fig, axis, ridges, trenches)

                # Save figure
                args.plt.tight_layout()
                args.plt.savefig(
                    misc.out_name(args, 'eta').format(viscosity.step) +
                    '.pdf', format='PDF')

                # Zoom
                if args.zoom is not None:
                    if args.zoom > 360 or args.zoom < 0:
                        print(' *WARNING* ')
                        print('Zoom angle should be positive and less than 360 deg')
                        print(' Exiting the code ')
                        sys.exit()
                    if args.zoom > 315. or args.zoom <= 45:
                        ladd, radd, uadd, dadd = 0.1, 0.05, 0.8, 0.8
                    elif args.zoom > 45. and args.zoom <= 135:
                        ladd, radd, uadd, dadd = 0.8, 0.8, 0.05, 0.1
                    elif args.zoom > 135. and args.zoom <= 225:
                        ladd, radd, uadd, dadd = 0.05, 0.1, 0.8, 0.8
                    else:
                        ladd, radd, uadd, dadd = 0.8, 0.8, 0.1, 0.05
                    xzoom = (rcmb + 1) * np.cos(args.zoom / 180. * np.pi)
                    yzoom = (rcmb + 1) * np.sin(args.zoom / 180. * np.pi)
                    axis.set_xlim(xzoom - ladd, xzoom + radd)
                    axis.set_ylim(yzoom - dadd, yzoom + uadd)
                    args.plt.savefig(
                        misc.out_name(args, 'etazoom').format(viscosity.step) +
                        '.pdf', format='PDF')
                args.plt.close(fig)

                # plot stress field with position of trenches and ridges
                if args.plot_stress:
                    constants.FIELD_VAR_LIST['s'] = constants.FIELD_VAR_LIST['s']._replace(name='Stress [MPa]')
                    fig, axis, surf = plot_scalar(args, stressdim, 's')
                    surf.set_clim(vmin=0, vmax=300)
                    args.plt.figure(fig.number)

                    # Annotation with time and step
                    axis.text(1., 0.9, str(round(time, 0)) + ' My',
                              transform=axis.transAxes, fontsize=args.fontsize)
                    axis.text(1., 0.1, str(timestep),
                              transform=axis.transAxes, fontsize=args.fontsize)

                    # Put arrow where ridges and trenches are
                    plot_plate_limits_field(args, fig, axis, ridges, trenches)

                    # Save figure
                    args.plt.tight_layout()
                    args.plt.savefig(
                        misc.out_name(args, 's').format(viscosity.step) +
                        '.pdf', format='PDF')

                    # Zoom
                    if args.zoom is not None:
                        axis.set_xlim(xzoom - ladd, xzoom + radd)
                        axis.set_ylim(yzoom - dadd, yzoom + uadd)
                        args.plt.savefig(
                            misc.out_name(args, 'szoom').format(viscosity.step) + '.pdf',
                            format='PDF')
                    args.plt.close(fig)

                    # calculate stresses in the lithosphere
                    lithospheric_stress(args, temp, conc, stressdim,
                                        velocity, trenches, ridges,
                                        timestep, time)

                # plotting the principal deviatoric stress field
                if args.plot_deviatoric_stress:
                    fig, axis, surf = plot_scalar(args, stressdim, 's')
                    surf.set_clim(vmin=0, vmax=300)
                    args.plt.figure(fig.number)

                    stressvec = BinData(args, 'x', timestep)

                    # plotting continents
                    xmesh, ymesh = conc.x_mesh[0, :, :], conc.y_mesh[0, :, :]
                    surf2 = axis.pcolormesh(xmesh, ymesh, continentsfld,
                                            rasterized=not args.pdf,
                                            cmap='cool_r', vmin=0, vmax=0,
                                            shading='goaround')
                    cmap2 = args.plt.cm.ocean
                    cmap2.set_over('m')

                    # plotting principal deviatoric stress
                    sphi = stressvec.fields['sy'][:, :, 0]
                    sr = stressvec.fields['sz'][:, :, 0]
                    stressx = -sphi * np.sin(ph_mesh) + sr * np.cos(ph_mesh)
                    stressy = sphi * np.cos(ph_mesh) + sr * np.sin(ph_mesh)
                    step = np.int(np.size(ph_mesh[0, :]) / 100.)
                    # xmesh, ymesh = sphi.x_mesh[0, :, :], sr.y_mesh[0, :, :]
                    axis.quiver(xmesh[::step, ::step], ymesh[::step, ::step],
                                stressx[::step, ::step].T,
                                stressy[::step, ::step].T)
                    cbar = plt.colorbar(surf, shrink=args.shrinkcb)
                    cbar.set_label(constants.FIELD_VAR_LIST['s'].name)

                    # Annotation with time and step
                    axis.text(1., 0.9, str(round(time, 0)) + ' My',
                              transform=axis.transAxes, fontsize=args.fontsize)
                    axis.text(1., 0.1, str(timestep),
                              transform=axis.transAxes, fontsize=args.fontsize)

                    # Put arrow where ridges and trenches are
                    plot_plate_limits_field(args, fig, axis, ridges, trenches)

                    args.plt.tight_layout()
                    args.plt.savefig(
                        misc.out_name(args, 'sx').format(viscosity.step) +
                        '.pdf', format='PDF')
                    args.plt.close(fig)

    if args.vzcheck:
        seuil_memz = 0
        nb_plates = []
        timedat = TimeData(args)
        slc = slice(*(i * args.par_nml['ioin']['save_file_framestep']
                      for i in args.timestep))
        time, ch2o = timedat.data[:, 1][slc], timedat.data[:, 27][slc]

        for timestep in range(*args.timestep):
            velocity = BinData(args, 'v', timestep)
            temp = BinData(args, 't', timestep)
            rprof_data = RprofData(args)
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

        if args.timeprofile:
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
