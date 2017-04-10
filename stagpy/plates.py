"""Plots position of subduction and ridge at the surface.

Date: 2016/26/01
"""
from copy import deepcopy
import functools
import shutil
import sys
import tempfile
import os.path
import numpy as np
from scipy.signal import argrelextrema
from . import constants, field, misc
from .stagyydata import StagyyData


def detect_plates_vzcheck(step, seuil_memz):
    """detect plates and check with vz and plate size"""
    v_z = step.fields['w'][0, :, :, 0]
    v_x = step.fields['v'][0, :, :, 0]
    h2o = step.fields['h'][0, :, :, 0]
    tcell = step.fields['t'][0, :, :, 0]
    n_z = step.geom.nztot
    nphi = step.geom.nptot  # -1? should be OK, ghost not included
    rcmb = max(0, step.geom.rcmb)
    radius = step.geom.r_coord + rcmb
    radiusgrid = step.geom.rgeom[:, 0] + rcmb
    dphi = 1 / nphi

    # water profile
    water_profile = np.mean(h2o, axis=0)

    # calculing tmean
    tmean = 0
    for i_r in range(len(radius)):
        for phi in range(nphi):
            tmean += (radiusgrid[i_r + 1]**2 -
                      radiusgrid[i_r] ** 2) * dphi * tcell[phi, i_r]
    tmean /= (radiusgrid[-1]**2 - rcmb**2)

    # calculing temperature on the grid and vz_mean/v_rms
    v_rms = 0
    vz_mean = 0
    tgrid = np.zeros((nphi, n_z + 1))
    for phi in range(nphi):
        tgrid[phi, 0] = 1
    for i_z in range(1, n_z):
        for phi in range(nphi):
            tgrid[phi, i_z] = (
                tcell[phi, i_z - 1] *
                (radiusgrid[i_z] - radius[i_z - 1]) + tcell[phi, i_z] *
                (-radiusgrid[i_z] + radius[i_z])) / (radius[i_z] -
                                                     radius[i_z - 1])
            v_rms += (v_z[phi, i_z]**2 + v_x[phi, i_z]**2) / (nphi * n_z)
            vz_mean += abs(v_z[phi, i_z]) / (nphi * n_z)
    v_rms = v_rms**0.5
    print(v_rms, vz_mean)

    flux_c = n_z * [0]
    for i_z in range(1, n_z - 1):
        for phi in range(nphi):
            flux_c[i_z] += (tgrid[phi, i_z] - tmean) * \
                v_z[phi, i_z] * radiusgrid[i_z] * dphi

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
            dvphi[phi] = (v_x[phi, n_z - 1] -
                          v_x[phi - 1, n_z - 1]) / ((1 + rcmb) * dphi)
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
                    vzm += (abs(v_z[phi, i_z]) +
                            abs(v_z[phi - 1, i_z]) +
                            abs(v_z[0, i_z])) / (n_z * 3)
            else:
                for i_z in range(0, n_z):
                    vzm += (abs(v_z[phi, i_z]) +
                            abs(v_z[phi - 1, i_z]) +
                            abs(v_z[phi + 1, i_z])) / (n_z * 3)

            if seuil_memz != 0:
                vz_thres = vz_mean * 0.1 + seuil_memz / 2
            else:
                vz_thres = vz_mean * 0
            if vzm < vz_thres:
                limits.remove(phi)
                k += 1
        print(limits)

        print('\n')
    return limits, nphi, dvphi, vz_thres, v_x[:, n_z - 1], water_profile


def detect_plates(args, step, vrms_surface, fids, time):
    """detect plates using derivative of horizontal velocity"""
    timestep = step.isnap
    vphi = step.fields['v'][0, :, :, 0]
    ph_coord = step.geom.p_coord
    if args.plot_age:
        agefld = step.fields['a'][0, :, :, 0]
    else:
        agefld = []

    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; should check if you are in the
        # thermal boundary layer
        indsurf = np.argmin(abs((1 - dsa) - step.geom.r_coord)) - 4
    else:
        dsa = 0.
        indsurf = -1

    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi
    # velocity derivation
    dvph2 = (np.diff(vph2[:, indsurf]) / (ph_coord[0] * 2.))

    io_surface(timestep, time, fids[6], dvph2)
    io_surface(timestep, time, fids[7], vph2[:-1, indsurf])

    # prepare stuff to find trenches and ridges
    if step.sdat.par['boundaries']['air_layer']:
        myorder_trench = 15
    else:
        myorder_trench = 10
    myorder_ridge = 20  # threshold

    # finding trenches
    pom2 = deepcopy(dvph2)
    if step.sdat.par['boundaries']['air_layer']:
        maskbigdvel = -30 * vrms_surface
    else:
        maskbigdvel = -10 * vrms_surface
    pom2[pom2 > maskbigdvel] = maskbigdvel
    argless_dv = argrelextrema(
        pom2, np.less, order=myorder_trench, mode='wrap')[0]
    trench = ph_coord[argless_dv]
    velocity_trench = vph2[argless_dv, indsurf]
    dv_trench = dvph2[argless_dv]

    # finding ridges
    pom2 = deepcopy(dvph2)
    masksmalldvel = np.amax(dvph2) * 0.2
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
        age_surface = np.ma.masked_where(agefld[:, indsurf] < 0.00001,
                                         agefld[:, indsurf])
        age_surface_dim = age_surface * vrms_surface *\
            args.ttransit / args.yearins / 1.e6
        agetrench = age_surface_dim[argless_dv]  # age at the trench
    else:
        agetrench = np.zeros(len(argless_dv))

    # writing the output into a file, all time steps are in one file
    for itrench in np.arange(len(trench)):
        fids[0].write("%7.0f %11.7f %10.6f %9.2f %9.2f \n" % (
            timestep,
            step.geom.ti_ad,
            trench[itrench],
            velocity_trench[itrench],
            agetrench[itrench]
        ))

    return trench, ridge, agetrench, dv_trench, dv_ridge


def plot_plate_limits(args, fig, axis, ridges, trenches, ymin, ymax):
    """plot lines designating ridges and trenches"""
    args.plt.figure(fig.number)
    # need for fig and args?
    for trench in trenches:
        axis.axvline(
            x=trench, ymin=ymin, ymax=ymax,
            color='red', ls='dashed', alpha=0.4)
    for ridge in ridges:
        axis.axvline(
            x=ridge, ymin=ymin, ymax=ymax,
            color='green', ls='dashed', alpha=0.4)
    axis.set_xlim(0, 2 * np.pi)
    axis.set_ylim(ymin, ymax)


def plot_plate_limits_field(args, fig, axis, rcmb, ridges, trenches):
    """plot arrows designating ridges and trenches in 2D field plots"""
    args.plt.figure(fig.number)  # need for fig and args? Could use sca.
    for trench in trenches:  # why use index and arange?
        xxd = (rcmb + 1.02) * np.cos(trench)  # arrow begin
        yyd = (rcmb + 1.02) * np.sin(trench)  # arrow begin
        xxt = (rcmb + 1.35) * np.cos(trench)  # arrow end
        yyt = (rcmb + 1.35) * np.sin(trench)  # arrow end
        axis.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                      arrowprops=dict(facecolor='red', shrink=0.05))
    for ridge in ridges:
        xxd = (rcmb + 1.02) * np.cos(ridge)
        yyd = (rcmb + 1.02) * np.sin(ridge)
        xxt = (rcmb + 1.35) * np.cos(ridge)
        yyt = (rcmb + 1.35) * np.sin(ridge)
        axis.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                      arrowprops=dict(facecolor='green', shrink=0.05))


def plot_plates(args, step, time, vrms_surface, trench, ridge, agetrench,
                topo, fids):
    """handle ploting stuff"""
    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
    else:
        dsa = 0.

    plt = args.plt
    lwd = args.linewidth
    vphi = step.fields['v'][0, :, :, 0]
    tempfld = step.fields['t'][0, :, :, 0]
    concfld = step.fields['c'][0, :, :, 0]
    timestep = step.isnap

    if args.plot_age:
        agefld = step.fields['a'][0, :, :, 0]
    if args.plot_stress:
        stressfld = step.fields['s'][0, :, :, 0]
        scale_stress = args.kappa * args.viscosity_ref / args.mantle**2

    if step.sdat.par['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        # the surface (that is considered plane here); should check if you are
        # in the thermal boundary layer
        indsurf = np.argmin(abs((1 - dsa) - step.geom.r_coord)) - 4
        # depth to detect the continents
        indcont = np.argmin(abs((1 - dsa) - np.array(step.geom.r_coord))) - 10
    else:
        indsurf = -1
        indcont = -1  # depth to detect continents

    if step.sdat.par['boundaries']['air_layer'] and\
       not step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:-1, indcont] < 3,
                          concfld[:-1, indcont] > 4),
            concfld[:-1, indcont])
    elif step.sdat.par['boundaries']['air_layer'] and\
         step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:-1, indcont] < 3,
                          concfld[:-1, indcont] > 5),
            concfld[:-1, indcont])
    elif step.sdat.par['tracersin']['tracers_weakcrust']:
        continents = np.ma.masked_where(
            concfld[:-1, indcont] < 3, concfld[:-1, indcont])
    else:
        continents = np.ma.masked_where(
            concfld[:-1, indcont] < 2, concfld[:-1, indcont])

    # masked array, only continents are true
    continentsall = continents / continents

    if args.plot_age:
        age_surface = np.ma.masked_where(
            agefld[:, indsurf] < 0.00001, agefld[:, indsurf])
        age_surface_dim =\
            age_surface * vrms_surface * args.ttransit / args.yearins / 1.e6

    ph_coord = step.geom.p_coord

    # velocity
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi
    dvph2 = (np.diff(vph2[:, indsurf]) / (ph_coord[0] * 2.))

    # plotting
    fig0, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], concfld[:-1, indsurf],
             color='g', linewidth=lwd, label='Conc')
    ax2.plot(ph_coord[:-1], tempfld[:-1, indsurf],
             color='k', linewidth=lwd, label='Temp')
    ax3.plot(ph_coord[:-1], vph2[:-1, indsurf], linewidth=lwd, label='Vel')

    ax1.fill_between(
        ph_coord[:-1], continents, 1., facecolor='#8B6914', alpha=0.2)
    ax2.fill_between(
        ph_coord[:-1], continentsall, 0., facecolor='#8B6914', alpha=0.2)

    if step.sdat.par['boundaries']['topT_mode'] == 'iso':
        tempmin = step.sdat.par['boundaries']['topT_val'] * 0.9
    else:
        tempmin = 0.0
    if step.sdat.par['boundaries']['botT_mode'] == 'iso':
        tempmax = step.sdat.par['boundaries']['botT_val'] * 0.35
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
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
             transform=ax1.transAxes, fontsize=args.fontsize)

    plot_plate_limits(args, fig0, ax3, ridge, trench,
                      args.velocitymin, args.velocitymax)

    figname = misc.out_name(args, 'sveltempconc').format(timestep) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close(fig0)

    # plotting velocity and velocity derivative
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
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

    figname = misc.out_name(args, 'sveldvel').format(timestep) + '.pdf'
    plt.savefig(figname, format='PDF')
    plt.close(fig0)

    # plotting velocity and second invariant of stress
    if args.plot_stress:
        fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], linewidth=lwd, label='Vel')
        ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                    color='black', ls='solid', alpha=0.2)
        ax1.set_ylabel("Velocity", fontsize=args.fontsize)
        ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax1.transAxes, fontsize=args.fontsize)
        ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
                 transform=ax1.transAxes, fontsize=args.fontsize)
        ax2.plot(ph_coord[:-1], stressfld[:-1, indsurf] * scale_stress / 1.e6,
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
            ph_coord[:-1], continentsall * args.dvelocitymin,
            args.dvelocitymax,
            facecolor='#8B6914', alpha=0.2)

        figname = misc.out_name(args, 'svelstress').format(timestep) + '.pdf'
        plt.savefig(figname, format='PDF')
        plt.close(fig0)

    # plotting velocity
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylim(args.velocitymin, args.velocitymax)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    plot_plate_limits(args, fig1, ax1, ridge, trench,
                      args.velocitymin, args.velocitymax)

    # plotting velocity and age at surface
    if args.plot_age:
        fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax3.plot(ph_coord[:-1], vph2[:-1, indsurf], linewidth=lwd, label='Vel')
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
    if step.sdat.par['switches']['cont_tracers']:
        for i in range(len(trench)):  # should enumerate
            # detection of the distance in between subduction and continent
            ph_coord_noendpoint = ph_coord[:-1]
            angdistance1 = abs(ph_coord_noendpoint[continentsall == 1] - trench[i])
            angdistance2 = 2. * np.pi - angdistance1
            angdistance = np.minimum(angdistance1, angdistance2)
            distancecont = min(angdistance)
            argdistancecont = np.argmin(angdistance)
            continentpos = ph_coord_noendpoint[continentsall == 1][argdistancecont]

            ph_trench_subd.append(trench[i])
            age_subd.append(agetrench[i])
            ph_cont_subd.append(continentpos)
            distance_subd.append(distancecont)
            times_subd.append(step.geom.ti_ad)

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
             topo[:, 1] * args.mantle / 1.e3,
             color='black')
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(args.topomin, args.topomax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * args.topomax, args.topomin,
        facecolor='#8B6914', alpha=0.2)
    plot_plate_limits(args, fig1, ax2, ridge, trench,
                      args.topomin, args.topomax)
    ax1.set_title(timestep, fontsize=args.fontsize)
    figname = misc.out_name(args, 'sveltopo').format(timestep) + '.pdf'
    fig1.savefig(figname, format='PDF')
    plt.close(fig1)

    if args.plot_age:
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
        figname = misc.out_name(args, 'svelage').format(timestep) + '.pdf'
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


def io_surface(timestep, time, fid, fld):
    """Output for surface files"""
    fid.write("{} {}".format(timestep, time))
    fid.writelines(["%10.2e" % item for item in fld[:]])
    fid.writelines(["\n"])


def lithospheric_stress(args, step, trench, ridge, time):
    """calculate stress in the lithosphere"""
    timestep = step.isnap
    lwd = args.linewidth
    base_lith = step.geom.rcmb + 1 - 0.105
    scale_stress = args.kappa * args.viscosity_ref / args.mantle**2
    scale_dist = args.mantle

    stressfld = step.fields['s'][0, :, :, 0]
    stressfld = np.ma.masked_where(step.geom.r_mesh[0] < base_lith, stressfld)

    # stress integration in the lithosphere
    dzm = (step.geom.r_coord[1:] - step.geom.r_coord[:-1])
    stress_lith = np.sum((stressfld[:, 1:] * dzm.T), axis=1)
    ph_coord = step.geom.p_coord  # probably doesn't need alias

    # plot stress in the lithosphere
    fig, axis = args.plt.subplots(ncols=1)
    surf = axis.pcolormesh(step.geom.x_mesh[0], step.geom.y_mesh[0],
                           stressfld * scale_stress / 1.e6,
                           cmap='gnuplot2_r',
                           rasterized=not args.pdf, shading='gouraud')
    surf.set_clim(vmin=0, vmax=300)
    cbar = args.plt.colorbar(surf, shrink=args.shrinkcb)
    cbar.set_label(constants.FIELD_VAR_LIST['s'].name)
    args.plt.axis('equal')
    args.plt.axis('off')
    # Annotation with time and step
    axis.text(1., 0.9, str(round(time, 0)) + ' My',
              transform=axis.transAxes, fontsize=args.fontsize)
    axis.text(1., 0.1, str(timestep),
              transform=axis.transAxes, fontsize=args.fontsize)
    args.plt.savefig(
        misc.out_name(args, 'lith').format(timestep) + '.pdf',
        format='PDF')
    args.plt.close(fig)

    # velocity
    vphi = step.fields['v'][0, :, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi

    # position of continents
    concfld = step.fields['c'][0, :, :, 0]
    if step.sdat.par['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        dsa = step.sdat.par['boundaries']['air_thickness']
        # depth to detect the continents
        indcont = np.argmin(abs((1 - dsa) - step.geom.r_coord)) - 10
    else:
        # depth to detect continents
        indcont = -1
    if step.sdat.par['boundaries']['air_layer'] and\
            not step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:-1, indcont] < 3,
                          concfld[:-1, indcont] > 4),
            concfld[:-1, indcont])
    elif step.sdat.par['boundaries']['air_layer'] and\
            step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:-1, indcont] < 3,
                          concfld[:-1, indcont] > 5),
            concfld[:-1, indcont])
    elif step.sdat.par['tracersin']['tracers_weakcrust']:
        continents = np.ma.masked_where(
            concfld[:-1, indcont] < 3, concfld[:-1, indcont])
    else:
        continents = np.ma.masked_where(
            concfld[:-1, indcont] < 2, concfld[:-1, indcont])

    # masked array, only continents are true
    continentsall = continents / continents

    # plot integrated stress in the lithosphere
    fig0, (ax1, ax2) = args.plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, -1], linewidth=lwd, label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity", fontsize=args.fontsize)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes, fontsize=args.fontsize)
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
             transform=ax1.transAxes, fontsize=args.fontsize)

    ax2.plot(ph_coord, stress_lith * scale_stress * scale_dist / 1.e12,
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

    figname = misc.out_name(args, 'svelslith').format(timestep) + '.pdf'
    fig0.savefig(figname, format='PDF')
    args.plt.close(fig0)


def plates_cmd(args):
    """find positions of trenches and subductions

    uses velocity field (velocity derivation)
    plots the number of plates over a designated lapse of time
    """
    sdat = StagyyData(args.path)
    if not args.vzcheck:
        fids = [0 for _ in range(8)]
        tmpf = functools.partial(tempfile.NamedTemporaryFile, mode='w',
                                 prefix='stagpy', delete=False)
        fnames = ['plate_velocity', 'distance_subd', 'continents',
                  'flux', 'topography', 'age',
                  'velderiv', 'velocity']
        istart, iend = None, None
        with tmpf() as fids[0], tmpf() as fids[1], tmpf() as fids[2],\
             tmpf() as fids[3], tmpf() as fids[4], tmpf() as fids[5],\
             tmpf() as fids[6], tmpf() as fids[7]:

            fids[0].write('#  it  time  ph_trench vel_trench age_trench\n')
            fids[1].write('#  it      time   time [My]   distance     '
                          'ph_trench     ph_cont  age_trench [My]\n')

            for step in misc.steps_gen(sdat, args):
                # could check other fields too
                if step.fields['t'] is None:
                    continue
                timestep = step.isnap
                istart = timestep if istart is None else istart
                iend = timestep
                print('Treating snapshot', timestep)

                rcmb = step.geom.rcmb
                if args.plot_stress:
                    scale_stress = args.kappa * \
                        args.viscosity_ref / args.mantle**2

                if timestep == istart:
                    # same code in rprof!
                    # calculating averaged horizontal surface velocity
                    # needed for redimensionalisation
                    ilast = sdat.rprof.index.levels[0][-1]
                    rlast = sdat.rprof.loc[ilast]
                    nprof = 0
                    uprof_averaged = rlast.loc[:, 'vhrms'] * 0
                    for step in misc.steps_gen(sdat, args):
                        if step.rprof is None:
                            continue
                        uprof_averaged += step.rprof['vhrms']
                        nprof += 1
                    uprof_averaged /= nprof
                    radius = rlast['r'].values + rcmb
                    if sdat.par['boundaries']['air_layer']:
                        dsa = sdat.par['boundaries']['air_thickness']
                        isurf = np.argmin(abs(radius - radius[-1] + dsa))
                    else:
                        isurf = -1
                    vrms_surface = uprof_averaged.iloc[isurf]
                    if sdat.par['boundaries']['air_layer']:
                        isurf = np.argmin(abs((1 - dsa) - step.geom.r_coord))
                        isurf -= 4  # why different isurf for the rest?

                # topography
                fname = sdat.filename('sc', timestep=timestep, suffix='.dat')
                topo = np.genfromtxt(str(fname))
                # rescaling topography!
                if sdat.par['boundaries']['air_layer']:
                    topo[:, 1] = topo[:, 1] / (1. - dsa)

                time = step.geom.ti_ad * vrms_surface *\
                    args.ttransit / args.yearins / 1.e6
                trenches, ridges, agetrenches, _, _ =\
                    detect_plates(args, step, vrms_surface, fids, time)
                plot_plates(args, step, time, vrms_surface, trenches, ridges,
                            agetrenches, topo, fids)

                # prepare for continent plotting
                concfld = step.fields['c'][0, :, :, 0]
                continentsfld = np.ma.masked_where(
                    concfld < 3, concfld)  # plotting continents, to-do
                continentsfld = continentsfld / continentsfld

                temp = step.fields['t'][0, :, :, 0]
                tgrad = (temp[:, isurf - 1] - temp[:, isurf]) /\
                    (step.geom.r_coord[isurf] - step.geom.r_coord[isurf - 1])

                io_surface(timestep, time, fids[2], concfld[:-1, isurf])
                io_surface(timestep, time, fids[3], tgrad)
                io_surface(timestep, time, fids[4], topo[:, 1])
                if args.plot_age:
                    io_surface(timestep, time, fids[5],
                               step.fields['a'][0, :, isurf, 0])

                # plot viscosity field with position of trenches and ridges
                fig, axis, surf, _ = field.plot_scalar(args, step, 'n')
                etamax = sdat.par['viscosity']['eta_max']
                surf.set_clim(vmin=1e-2, vmax=etamax)

                # plotting continents
                xmesh, ymesh = step.geom.x_mesh[0], step.geom.y_mesh[0]
                axis.pcolormesh(xmesh, ymesh, continentsfld,
                                rasterized=not args.pdf, cmap='cool_r',
                                vmin=0, vmax=0, shading='goaround')
                cmap2 = args.plt.cm.ocean
                cmap2.set_over('m')

                # plotting velocity vectors
                if step.geom.cartesian:
                   velx = step.fields['v'][0, :, :, 0]
                   vely = step.fields['w'][0, :, :, 0]
                else: # spherical yz
                   vphi = step.fields['v'][0, :, :, 0]
                   velr = step.fields['w'][0, :, :, 0]
                   ph_mesh = step.geom.p_mesh[0]
                   velx = -vphi * np.sin(ph_mesh) + velr * np.cos(ph_mesh)
                   vely = vphi * np.cos(ph_mesh) + velr * np.sin(ph_mesh)
                dip = step.geom.nptot // 100
                axis.quiver(xmesh[::dip, ::dip], ymesh[::dip, ::dip],
                            velx[::dip, ::dip], vely[::dip, ::dip])

                # Annotation with time and step
                axis.text(1., 0.9, str(round(time, 0)) + ' My',
                          transform=axis.transAxes, fontsize=args.fontsize)
                axis.text(1., 0.1, str(timestep),
                          transform=axis.transAxes, fontsize=args.fontsize)

                # Put arrow where ridges and trenches are
                plot_plate_limits_field(args, fig, axis, rcmb,
                                        ridges, trenches)

                # Save figure
                args.plt.tight_layout()
                args.plt.savefig(
                    misc.out_name(args, 'eta').format(timestep) + '.pdf',
                    format='PDF')

                # Zoom
                if args.zoom is not None:
                    if args.zoom > 360 or args.zoom < 0:
                        print(' *WARNING* ')
                        print('Zoom angle should be positive and '
                              'less than 360 deg')
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
                        misc.out_name(args, 'etazoom').format(timestep) +
                        '.pdf', format='PDF')
                args.plt.close(fig)

                # plot stress field with position of trenches and ridges
                if args.plot_stress:
                    fig, axis, surf, cbar = field.plot_scalar(
                        args, step, 's', scale_stress / 1.e6)
                    surf.set_clim(vmin=0, vmax=300)
                    cbar.set_label('Stress [MPa]')

                    # Annotation with time and step
                    axis.text(1., 0.9, str(round(time, 0)) + ' My',
                              transform=axis.transAxes, fontsize=args.fontsize)
                    axis.text(1., 0.1, str(timestep),
                              transform=axis.transAxes, fontsize=args.fontsize)

                    # Put arrow where ridges and trenches are
                    plot_plate_limits_field(args, fig, axis, rcmb,
                                            ridges, trenches)

                    # Save figure
                    args.plt.tight_layout()
                    args.plt.savefig(
                        misc.out_name(args, 's').format(timestep) + '.pdf',
                        format='PDF')

                    # Zoom
                    if args.zoom is not None:
                        axis.set_xlim(xzoom - ladd, xzoom + radd)
                        axis.set_ylim(yzoom - dadd, yzoom + uadd)
                        args.plt.savefig(
                            misc.out_name(args, 'szoom').format(timestep) +
                            '.pdf', format='PDF')
                    args.plt.close(fig)

                    # calculate stresses in the lithosphere
                    lithospheric_stress(args, step, trenches, ridges, time)

                # plotting the principal deviatoric stress field
                if args.plot_deviatoric_stress:
                    fig, axis, surf, _ = field.plot_scalar(args,
                                                           step, 's',
                                                           alpha=0.1)
                    # surf.set_clim(vmin=0, vmax=300)

                    # plotting continents
                    axis.pcolormesh(xmesh, ymesh, continentsfld,
                                    rasterized=not args.pdf, cmap='cool_r',
                                    vmin=0, vmax=0, shading='goaround')
                    cmap2 = args.plt.cm.ocean
                    cmap2.set_over('m')

                    # plotting principal deviatoric stress
                    dummy = step.fields['x'][0, :, :, 0]
                    sphi = step.fields['sxj'][0, :, :, 0]
                    sr = step.fields['sxk'][0, :, :, 0]
                    stressx = -sphi * np.sin(ph_mesh) + sr * np.cos(ph_mesh)
                    stressy = sphi * np.cos(ph_mesh) + sr * np.sin(ph_mesh)
                    dip = step.geom.nptot // 100
                    axis.quiver(xmesh[::dip, ::dip], ymesh[::dip, ::dip],
                                stressx[::dip, ::dip], stressy[::dip, ::dip])
                    # cbar = args.plt.colorbar(surf, shrink=args.shrinkcb)
                    # cbar.set_label(constants.FIELD_VAR_LIST['s'].name)

                    # Annotation with time and step
                    axis.text(1., 0.9, str(round(time, 0)) + ' My',
                              transform=axis.transAxes, fontsize=args.fontsize)
                    axis.text(1., 0.1, str(timestep),
                              transform=axis.transAxes, fontsize=args.fontsize)

                    # Put arrow where ridges and trenches are
                    plot_plate_limits_field(args, fig, axis, rcmb,
                                            ridges, trenches)

                    args.plt.tight_layout()
                    args.plt.savefig(
                        misc.out_name(args, 'sx').format(timestep) + '.pdf',
                        format='PDF')
                    args.plt.close(fig)

        for fname, tmp in zip(fnames, fids):
            stem = '{}_{}_{}'.format(fname, istart, iend)
            idx = 0
            fmt = '{}.dat'
            while os.path.isfile(fmt.format(stem, idx)):
                fmt = '{}_{}.dat'
                idx += 1
            shutil.move(tmp.name, fmt.format(stem, idx))
    else:  # args.vzcheck
        seuil_memz = 0
        nb_plates = []
        time = []
        ch2o = []
        istart, iend = None, None

        for step in misc.steps_gen(sdat, args):
            # could check other fields too
            if step.fields['t'] is None:
                continue
            if args.timeprofile:
                time.append(step.timeinfo.loc['t'])
                ch2o.append(step.timeinfo.loc[0])
            istart = step.isnap if istart is None else istart
            iend = step.isnap
            plt = args.plt
            limits, nphi, dvphi, seuil_memz, vphi_surf, water_profile =\
                detect_plates_vzcheck(step, seuil_memz)
            limits.sort()
            sizeplates = [limits[0] + nphi - limits[-1]]
            for lim in range(1, len(limits)):
                sizeplates.append(limits[lim] - limits[lim - 1])
            lim = len(limits) * [max(dvphi)]
            plt.figure(step.istep)
            plt.subplot(221)
            plt.axis([0, nphi,
                      np.min(vphi_surf) * 1.2, np.max(vphi_surf) * 1.2])
            plt.plot(vphi_surf)
            plt.subplot(223)
            plt.axis(
                [0, nphi,
                 np.min(dvphi) * 1.2, np.max(dvphi) * 1.2])
            plt.plot(dvphi)
            plt.scatter(limits, lim, color='red')
            plt.subplot(222)
            plt.hist(sizeplates, 10, (0, nphi / 2))
            plt.subplot(224)
            plt.plot(water_profile)
            plt.savefig('plates' + str(step.isnap) + '.pdf', format='PDF')
            plt.close(step.istep)

            nb_plates.append(len(limits))

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
            plt.savefig('plates_{}_{}.pdf'.format(istart, iend), format='PDF')
            plt.close(-1)
