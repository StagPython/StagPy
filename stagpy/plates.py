"""Plate analysis."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from . import conf, error, field, misc, phyvars
from .stagyydata import StagyyData


def detect_plates_vzcheck(step, seuil_memz):
    """Detect plates and check with vz and plate size."""
    v_z = step.fields['v3'][0, :, :, 0]
    v_x = step.fields['v2'][0, :, :, 0]
    tcell = step.fields['T'][0, :, :, 0]
    n_z = step.geom.nztot
    nphi = step.geom.nptot  # -1? should be OK, ghost not included
    rcmb = max(0, step.geom.rcmb)
    radius = step.geom.r_coord
    radiusgrid = step.geom.rgeom[:, 0] + rcmb
    dphi = 1 / nphi

    # calculing temperature on the grid and vz_mean
    vz_mean = 0
    tgrid = np.zeros((nphi, n_z + 1))
    tgrid[:, 0] = 1
    for i_z in range(1, n_z):
        for phi in range(nphi):
            tgrid[phi, i_z] = (
                tcell[phi, i_z - 1] *
                (radiusgrid[i_z] - radius[i_z - 1]) + tcell[phi, i_z] *
                (-radiusgrid[i_z] + radius[i_z])) / (radius[i_z] -
                                                     radius[i_z - 1])
            vz_mean += abs(v_z[phi, i_z]) / (nphi * n_z)

    flux_c = n_z * [0]
    for i_z in range(1, n_z - 1):
        for phi in range(nphi):
            flux_c[i_z] += (tgrid[phi, i_z] - step.timeinfo.loc['Tmean']) * \
                v_z[phi, i_z] * radiusgrid[i_z] * dphi

    # checking stagnant lid
    if all(abs(flux_c[i_z]) <= np.max(flux_c) / 50
           for i_z in range(n_z - n_z // 20, n_z)):
        raise error.StagnantLidError(step.sdat)
    else:
        # verifying horizontal plate speed and closeness of plates
        dvphi = nphi * [0]
        dvx_thres = 16 * step.timeinfo.loc['vrms']

        for phi in range(0, nphi):
            dvphi[phi] = (v_x[phi, n_z - 1] -
                          v_x[phi - 1, n_z - 1]) / ((1 + rcmb) * dphi)
        limits = []
        for phi in range(0, nphi - nphi // 33):
            mark = all(abs(dvphi[i]) <= abs(dvphi[phi])
                       for i in range(phi - nphi // 33, phi + nphi // 33))
            if mark and abs(dvphi[phi]) >= dvx_thres:
                limits.append(phi)
        for phi in range(nphi - nphi // 33 + 1, nphi):
            mark = all(abs(dvphi[i]) <= abs(dvphi[phi])
                       for i in range(phi - nphi // 33 - nphi,
                                      phi + nphi // 33 - nphi))
            if mark and abs(dvphi[phi]) >= dvx_thres:
                limits.append(phi)

        # verifying vertical speed
        k = 0
        for i in range(len(limits)):
            vzm = 0
            phi = limits[i - k]
            for i_z in range(1 if phi == nphi - 1 else 0, n_z):
                vzm += (abs(v_z[phi, i_z]) +
                        abs(v_z[phi - 1, i_z]) +
                        abs(v_z[(phi + 1) % nphi, i_z])) / (n_z * 3)

            vz_thres = vz_mean * 0.1 + seuil_memz / 2 if seuil_memz else 0
            if vzm < vz_thres:
                limits.remove(phi)
                k += 1

    return limits, nphi, dvphi, vz_thres, v_x[:, n_z - 1]


def detect_plates(step, vrms_surface, fids, time):
    """Detect plates using derivative of horizontal velocity."""
    vphi = step.fields['v2'][0, :, :, 0]
    ph_coord = step.geom.p_coord

    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; should check if you are in the
        # thermal boundary layer
        indsurf = np.argmin(
            np.abs(1 - dsa - step.geom.r_coord + step.geom.rcmb)) - 4
    else:
        indsurf = -1

    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi
    # velocity derivation
    dvph2 = (np.diff(vph2[:, indsurf]) / (ph_coord[0] * 2.))

    io_surface(step.isnap, time, fids[6], dvph2)
    io_surface(step.isnap, time, fids[7], vph2[:-1, indsurf])

    # prepare stuff to find trenches and ridges
    myorder_trench = 15 if step.sdat.par['boundaries']['air_layer'] else 10
    myorder_ridge = 20  # threshold

    # finding trenches
    pom2 = np.copy(dvph2)
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
    pom2 = np.copy(dvph2)
    masksmalldvel = np.amax(dvph2) * 0.2
    pom2[pom2 < masksmalldvel] = masksmalldvel
    arggreat_dv = argrelextrema(
        pom2, np.greater, order=myorder_ridge, mode='wrap')[0]
    ridge = ph_coord[arggreat_dv]

    # elimination of ridges that are too close to trench
    argdel = []
    if trench.any() and ridge.any():
        for i, ridge_i in enumerate(ridge):
            mdistance = np.amin(abs(trench - ridge_i))
            if mdistance < 0.016:
                argdel.append(i)
        if argdel:
            print('deleting from ridge', trench, ridge[argdel])
            ridge = np.delete(ridge, np.array(argdel))
            arggreat_dv = np.delete(arggreat_dv, np.array(argdel))

    dv_ridge = dvph2[arggreat_dv]
    if 'age' in conf.plates:
        agefld = step.fields['age'][0, :, :, 0]
        age_surface = np.ma.masked_where(agefld[:, indsurf] < 0.00001,
                                         agefld[:, indsurf])
        age_surface_dim = age_surface * vrms_surface *\
            conf.scaling.ttransit / conf.scaling.yearins / 1.e6
        agetrench = age_surface_dim[argless_dv]  # age at the trench
    else:
        agetrench = np.zeros(len(argless_dv))

    # writing the output into a file, all time steps are in one file
    for itrench in np.arange(len(trench)):
        fids[0].write("%7.0f %11.7f %10.6f %9.2f %9.2f \n" % (
            step.isnap,
            step.geom.ti_ad,
            trench[itrench],
            velocity_trench[itrench],
            agetrench[itrench]
        ))

    return trench, ridge, agetrench, dv_trench, dv_ridge


def plot_plate_limits(axis, ridges, trenches, ymin, ymax):
    """Plot lines designating ridges and trenches."""
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


def plot_plate_limits_field(axis, rcmb, ridges, trenches):
    """Plot arrows designating ridges and trenches in 2D field plots."""
    for trench in trenches:
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


def plot_plates(step, time, vrms_surface, trench, ridge, agetrench,
                topo, fids):
    """Handle plotting stuff."""
    vphi = step.fields['v2'][0, :, :, 0]
    tempfld = step.fields['T'][0, :, :, 0]
    concfld = step.fields['c'][0, :, :, 0]
    timestep = step.isnap

    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; delete "-some number"
        # to be just below
        # the surface (that is considered plane here); should check if you are
        # in the thermal boundary layer
        indsurf = np.argmin(
            np.abs(1 - dsa - step.geom.r_coord + step.geom.rcmb)) - 4
        # depth to detect the continents
        indcont = np.argmin(
            np.abs(1 - dsa - step.geom.r_coord + step.geom.rcmb)) - 10
    else:
        indsurf = -1
        indcont = -1  # depth to detect continents

    if step.sdat.par['boundaries']['air_layer'] and\
       not step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:-1, indcont] < 3,
                          concfld[:-1, indcont] > 4),
            concfld[:-1, indcont])
    elif (step.sdat.par['boundaries']['air_layer'] and
          step.sdat.par['continents']['proterozoic_belts']):
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

    ph_coord = step.geom.p_coord

    # velocity
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi
    dvph2 = (np.diff(vph2[:, indsurf]) / (ph_coord[0] * 2.))

    # plotting
    fig0, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], concfld[:-1, indsurf], color='g', label='Conc')
    ax2.plot(ph_coord[:-1], tempfld[:-1, indsurf], color='k', label='Temp')
    ax3.plot(ph_coord[:-1], vph2[:-1, indsurf], label='Vel')

    ax1.fill_between(
        ph_coord[:-1], continents, 1., facecolor='#8B6914', alpha=0.2)
    ax2.fill_between(
        ph_coord[:-1], continentsall, 0., facecolor='#8B6914', alpha=0.2)

    tempmin = step.sdat.par['boundaries']['topT_val'] * 0.9\
        if step.sdat.par['boundaries']['topT_mode'] == 'iso' else 0.0
    tempmax = step.sdat.par['boundaries']['botT_val'] * 0.35\
        if step.sdat.par['boundaries']['botT_mode'] == 'iso' else 0.8

    ax2.set_ylim(tempmin, tempmax)
    ax3.fill_between(
        ph_coord[:-1], continentsall * round(1.5 * np.amax(dvph2), 1),
        round(np.amin(dvph2) * 1.1, 1), facecolor='#8B6914', alpha=0.2)
    ax3.set_ylim(conf.plates.vmin, conf.plates.vmax)

    ax1.set_ylabel("Concentration")
    ax2.set_ylabel("Temperature")
    ax3.set_ylabel("Velocity")
    ax1.set_title(timestep)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
             transform=ax1.transAxes)

    plot_plate_limits(ax3, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)

    misc.saveplot(fig0, 'sveltempconc', timestep)

    # plotting velocity and velocity derivative
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity")
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
             transform=ax1.transAxes)
    ax2.plot(ph_coord[:-1] + ph_coord[0], dvph2, color='k', label='dv')
    ax2.set_ylabel("dv")

    plot_plate_limits(ax1, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)
    plot_plate_limits(ax2, ridge, trench, conf.plates.dvmin,
                      conf.plates.dvmax)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_title(timestep)

    ax1.fill_between(
        ph_coord[:-1], continentsall * conf.plates.vmin, conf.plates.vmax,
        facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * conf.plates.dvmin,
        conf.plates.dvmax, facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(conf.plates.dvmin, conf.plates.dvmax)

    misc.saveplot(fig0, 'sveldvel', timestep)

    # plotting velocity and second invariant of stress
    if 'str' in conf.plates.plot:
        stressfld = step.fields['sII'][0, :, :, 0]
        fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], label='Vel')
        ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                    color='black', ls='solid', alpha=0.2)
        ax1.set_ylabel("Velocity")
        ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax1.transAxes)
        ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
                 transform=ax1.transAxes)
        ax2.plot(ph_coord[:-1],
                 stressfld[:-1, indsurf] * step.sdat.scales.stress / 1.e6,
                 color='k', label='Stress')
        ax2.set_ylim(conf.plates.stressmin, conf.plates.stressmax)
        ax2.set_ylabel("Stress [MPa]")

        plot_plate_limits(ax1, ridge, trench,
                          conf.plates.vmin, conf.plates.vmax)
        plot_plate_limits(ax2, ridge, trench,
                          conf.plates.stressmin, conf.plates.stressmax)
        ax1.set_xlim(0, 2 * np.pi)
        ax1.set_title(timestep)

        ax1.fill_between(
            ph_coord[:-1], continentsall * conf.plates.vmin,
            conf.plates.vmax, facecolor='#8B6914', alpha=0.2)
        ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
        ax2.fill_between(
            ph_coord[:-1], continentsall * conf.plates.dvmin,
            conf.plates.dvmax,
            facecolor='#8B6914', alpha=0.2)

        misc.saveplot(fig0, 'svelstress', timestep)

    # plotting velocity
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, indsurf], label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
    ax1.set_ylabel("Velocity")
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    plot_plate_limits(ax1, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)

    # plotting velocity and age at surface
    if 'age' in conf.plates.plot:
        agefld = step.fields['age'][0, :, :, 0]
        age_surface = np.ma.masked_where(
            agefld[:, indsurf] < 0.00001, agefld[:, indsurf])
        age_surface_dim = (age_surface * vrms_surface * conf.scaling.ttransit /
                           conf.scaling.yearins / 1.e6)

        fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax3.plot(ph_coord[:-1], vph2[:-1, indsurf], label='Vel')
        ax3.axhline(
            y=0, xmin=0, xmax=2 * np.pi,
            color='black', ls='solid', alpha=0.2)
        ax3.set_ylim(conf.plates.vmin, conf.plates.vmax)
        ax3.set_ylabel("Velocity")
        ax3.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax3.transAxes)
        ax3.fill_between(
            ph_coord[:-1], continentsall * conf.plates.vmax,
            conf.plates.vmin, facecolor='#8B6914', alpha=0.2)
        plot_plate_limits(ax3, ridge, trench,
                          conf.plates.vmin, conf.plates.vmax)

    times_subd = []
    age_subd = []
    distance_subd = []
    ph_trench_subd = []
    ph_cont_subd = []
    if step.sdat.par['switches']['cont_tracers']:
        for i, trench_i in enumerate(trench):
            # detection of the distance in between subduction and continent
            ph_coord_noend = ph_coord[:-1]
            angdistance1 = abs(ph_coord_noend[continentsall == 1] - trench_i)
            angdistance2 = 2. * np.pi - angdistance1
            angdistance = np.minimum(angdistance1, angdistance2)
            distancecont = min(angdistance)
            argdistancecont = np.argmin(angdistance)
            continentpos = ph_coord_noend[continentsall == 1][argdistancecont]

            ph_trench_subd.append(trench_i)
            age_subd.append(agetrench[i])
            ph_cont_subd.append(continentpos)
            distance_subd.append(distancecont)
            times_subd.append(step.geom.ti_ad)

            if angdistance1[argdistancecont] < angdistance2[argdistancecont]:
                if continentpos - trench_i < 0:  # continent is on the left
                    distancecont = - distancecont
                ax1.annotate('', xy=(trench_i + distancecont, 2000),
                             xycoords='data', xytext=(trench_i, 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))
            else:  # distance over boundary
                xy_anot, xy_text = 0, 2 * np.pi
                if continentpos - trench_i < 0:
                    xy_anot, xy_text = xy_text, xy_anot
                ax1.annotate('', xy=(xy_anot, 2000),
                             xycoords='data', xytext=(trench_i, 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="-", lw="2",
                                             shrinkA=0, shrinkB=0))
                ax1.annotate('', xy=(continentpos, 2000),
                             xycoords='data', xytext=(xy_text, 2000),
                             textcoords='data',
                             arrowprops=dict(arrowstyle="->", lw="2",
                                             shrinkA=0, shrinkB=0))

    ax1.fill_between(
        ph_coord[:-1], continentsall * conf.plates.vmin,
        conf.plates.vmax, facecolor='#8B6914', alpha=0.2)
    ax2.set_ylabel("Topography [km]")
    ax2.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax2.plot(topo[:, 0],
             topo[:, 1] * step.sdat.scales.length / 1.e3,
             color='black')
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(conf.plates.topomin, conf.plates.topomax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * conf.plates.topomax,
        conf.plates.topomin, facecolor='#8B6914', alpha=0.2)
    plot_plate_limits(ax2, ridge, trench, conf.plates.topomin,
                      conf.plates.topomax)
    ax1.set_title(timestep)
    misc.saveplot(fig1, 'sveltopo', timestep)

    if 'age' in conf.plates.plot:
        ax4.set_ylabel("Seafloor age [My]")
        # in dimensions
        ax4.plot(ph_coord[:-1], age_surface_dim[:-1], color='black')
        ax4.set_xlim(0, 2 * np.pi)
        ax4.fill_between(
            ph_coord[:-1], continentsall * conf.plates.agemax,
            conf.plates.agemin, facecolor='#8B6914', alpha=0.2)
        ax4.set_ylim(conf.plates.agemin, conf.plates.agemax)
        plot_plate_limits(ax4, ridge, trench, conf.plates.agemin,
                          conf.plates.agemax)
        ax3.set_title(timestep)
        misc.saveplot(fig2, 'svelage', timestep)

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
    """Output surface files."""
    fid.write("{} {}".format(timestep, time))
    fid.writelines(["%10.2e" % item for item in fld[:]])
    fid.writelines(["\n"])


def lithospheric_stress(step, trench, ridge, time):
    """Calculate stress in the lithosphere."""
    timestep = step.isnap
    base_lith = step.geom.rcmb + 1 - 0.105

    stressfld = step.fields['sII'][0, :, :, 0]
    stressfld = np.ma.masked_where(step.geom.r_mesh[0] < base_lith, stressfld)

    # stress integration in the lithosphere
    dzm = (step.geom.r_coord[1:] - step.geom.r_coord[:-1])
    stress_lith = np.sum((stressfld[:, 1:] * dzm.T), axis=1)
    ph_coord = step.geom.p_coord  # probably doesn't need alias

    # plot stress in the lithosphere
    fig, axis, _, _ = field.plot_scalar(step, 'sII', stressfld,
                                        cmap='plasma_r', vmin=0, vmax=300)
    # Annotation with time and step
    axis.text(1., 0.9, str(round(time, 0)) + ' My', transform=axis.transAxes)
    axis.text(1., 0.1, str(timestep), transform=axis.transAxes)
    misc.saveplot(fig, 'lith', timestep)

    # velocity
    vphi = step.fields['v2'][0, :, :, 0]
    vph2 = 0.5 * (vphi + np.roll(vphi, 1, 0))  # interpolate to the same phi

    # position of continents
    concfld = step.fields['c'][0, :, :, 0]
    if step.sdat.par['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        dsa = step.sdat.par['boundaries']['air_thickness']
        # depth to detect the continents
        indcont = np.argmin(
            np.abs(1 - dsa - step.geom.r_coord + step.geom.rcmb)) - 10
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
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord[:-1], vph2[:-1, -1], label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity")
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.geom.ti_ad, 8)),
             transform=ax1.transAxes)

    intstr_scale = step.sdat.scales.stress * step.sdat.scales.length / 1.e12
    ax2.plot(ph_coord, stress_lith * intstr_scale, color='k', label='Stress')
    ax2.set_ylabel(r"Integrated stress [$TN\,m^{-1}$]")

    plot_plate_limits(ax1, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)
    plot_plate_limits(ax2, ridge, trench, conf.plates.stressmin,
                      conf.plates.lstressmax)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_title(timestep)

    ax1.fill_between(
        ph_coord[:-1], continentsall * conf.plates.vmin,
        conf.plates.vmax, facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
    ax2.fill_between(
        ph_coord[:-1], continentsall * conf.plates.stressmin,
        conf.plates.lstressmax, facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(conf.plates.stressmin, conf.plates.lstressmax)

    misc.saveplot(fig0, 'svelslith', timestep)


def set_of_vars(arg_plot):
    """Build set of needed variables.

    Args:
        arg_plot (str): string with variable names separated with ``,``.
    Returns:
        set of str: set of variables.
    """
    return set(var for var in arg_plot.split(',') if var in phyvars.PLATES)


def main_plates(sdat):
    """Plot several plates information."""
    # calculating averaged horizontal surface velocity
    # needed for redimensionalisation
    ilast = sdat.rprof.index.levels[0][-1]
    rlast = sdat.rprof.loc[ilast]
    nprof = 0
    uprof_averaged = rlast.loc[:, 'vhrms'] * 0
    for step in sdat.walk.filter(rprof=True):
        uprof_averaged += step.rprof['vhrms']
        nprof += 1
    uprof_averaged /= nprof
    radius = rlast['r'].values
    if sdat.par['boundaries']['air_layer']:
        dsa = sdat.par['boundaries']['air_thickness']
        isurf = np.argmin(abs(radius - radius[-1] + dsa))
        vrms_surface = uprof_averaged.iloc[isurf]
        isurf = np.argmin(abs((1 - dsa) - radius))
        isurf -= 4  # why different isurf for the rest?
    else:
        isurf = -1
        vrms_surface = uprof_averaged.iloc[isurf]

    with misc.InchoateFiles(8, 'plates') as fids:
        fids.fnames = ['plate_velocity', 'distance_subd', 'continents',
                       'flux', 'topography', 'age', 'velderiv', 'velocity']
        fids[0].write('#  it  time  ph_trench vel_trench age_trench\n')
        fids[1].write('#  it      time   time [My]   distance     '
                      'ph_trench     ph_cont  age_trench [My]\n')

        istart, iend = None, None
        for step in sdat.walk.filter(fields=['T']):
            # could check other fields too
            timestep = step.isnap
            istart = timestep if istart is None else istart
            iend = timestep
            print('Treating snapshot', timestep)

            rcmb = step.geom.rcmb
            # topography
            fname = sdat.filename('sc', timestep=timestep, suffix='.dat')
            topo = np.genfromtxt(str(fname))
            # rescaling topography!
            if sdat.par['boundaries']['air_layer']:
                topo[:, 1] = topo[:, 1] / (1. - dsa)

            time = step.geom.ti_ad * vrms_surface *\
                conf.scaling.ttransit / conf.scaling.yearins / 1.e6
            trenches, ridges, agetrenches, _, _ =\
                detect_plates(step, vrms_surface, fids, time)
            plot_plates(step, time, vrms_surface, trenches, ridges,
                        agetrenches, topo, fids)

            # prepare for continent plotting
            concfld = step.fields['c'][0, :, :, 0]
            continentsfld = np.ma.masked_where(
                concfld < 3, concfld)  # plotting continents, to-do
            continentsfld = continentsfld / continentsfld

            temp = step.fields['T'][0, :, :, 0]
            tgrad = (temp[:, isurf - 1] - temp[:, isurf]) /\
                (step.geom.r_coord[isurf] - step.geom.r_coord[isurf - 1])

            io_surface(timestep, time, fids[2], concfld[:-1, isurf])
            io_surface(timestep, time, fids[3], tgrad)
            io_surface(timestep, time, fids[4], topo[:, 1])
            if 'age' in conf.plates.plot:
                io_surface(timestep, time, fids[5],
                           step.fields['age'][0, :, isurf, 0])

            # plot viscosity field with position of trenches and ridges
            etamin, _ = sdat.scale(1e-2, 'Pa')
            etamax, _ = sdat.scale(sdat.par['viscosity']['eta_max'], 'Pa')
            fig, axis, _, _ = field.plot_scalar(step, 'eta',
                                                vmin=etamin, vmax=etamax)

            # plotting continents
            cbar = conf.field.colorbar
            conf.field.colorbar = False
            field.plot_scalar(step, 'c', continentsfld, axis,
                              cmap='cool_r', vmin=0, vmax=0)
            cmap2 = plt.cm.ocean
            cmap2.set_over('m')
            conf.field.colorbar = cbar

            # plotting velocity vectors
            field.plot_vec(axis, step, 'v')

            # Annotation with time and step
            axis.text(1., 0.9, str(round(time, 0)) + ' My',
                      transform=axis.transAxes)
            axis.text(1., 0.1, str(timestep),
                      transform=axis.transAxes)

            # Put arrow where ridges and trenches are
            plot_plate_limits_field(axis, rcmb, ridges, trenches)

            misc.saveplot(fig, 'eta', timestep, close=False)

            # Zoom
            if conf.plates.zoom is not None:
                if not 0 <= conf.plates.zoom <= 360:
                    raise error.InvalidZoomError(conf.plates.zoom)
                if 45 < conf.plates.zoom <= 135:
                    ladd, radd, uadd, dadd = 0.8, 0.8, 0.05, 0.1
                elif 135 < conf.plates.zoom <= 225:
                    ladd, radd, uadd, dadd = 0.05, 0.1, 0.8, 0.8
                elif 225 < conf.plates.zoom <= 315:
                    ladd, radd, uadd, dadd = 0.8, 0.8, 0.1, 0.05
                else:  # >315 or <=45
                    ladd, radd, uadd, dadd = 0.1, 0.05, 0.8, 0.8
                xzoom = (rcmb + 1) * np.cos(np.radians(conf.plates.zoom))
                yzoom = (rcmb + 1) * np.sin(np.radians(conf.plates.zoom))
                axis.set_xlim(xzoom - ladd, xzoom + radd)
                axis.set_ylim(yzoom - dadd, yzoom + uadd)
                misc.saveplot(fig, 'etazoom', timestep, close=False)
            plt.close(fig)

            # plot stress field with position of trenches and ridges
            if 'str' in conf.plates.plot:
                fig, axis, _, _ = field.plot_scalar(step, 'sII',
                                                    vmin=0, vmax=300)

                # Annotation with time and step
                axis.text(1., 0.9, str(round(time, 0)) + ' My',
                          transform=axis.transAxes)
                axis.text(1., 0.1, str(timestep),
                          transform=axis.transAxes)

                # Put arrow where ridges and trenches are
                plot_plate_limits_field(axis, rcmb, ridges, trenches)

                misc.saveplot(fig, 's', timestep, close=False)

                # Zoom
                if conf.plates.zoom is not None:
                    axis.set_xlim(xzoom - ladd, xzoom + radd)
                    axis.set_ylim(yzoom - dadd, yzoom + uadd)
                    misc.saveplot(fig, 'szoom', timestep, close=False)
                plt.close(fig)

                # calculate stresses in the lithosphere
                lithospheric_stress(step, trenches, ridges, time)

            # plotting the principal deviatoric stress field
            if 'sx' in conf.plates.plot:
                fig, axis, _, _ = field.plot_scalar(step, 'sII',
                                                    alpha=0.1)

                # plotting continents
                cbar = conf.field.colorbar
                conf.field.colorbar = False
                field.plot_scalar(step, 'c', continentsfld, axis,
                                  cmap='cool_r', vmin=0, vmax=0)
                cmap2 = plt.cm.ocean
                cmap2.set_over('m')
                conf.field.colorbar = cbar

                # plotting principal deviatoric stress
                field.plot_vec(axis, step, 'sx')

                # Annotation with time and step
                axis.text(1., 0.9, str(round(time, 0)) + ' My',
                          transform=axis.transAxes)
                axis.text(1., 0.1, str(timestep),
                          transform=axis.transAxes)

                # Put arrow where ridges and trenches are
                plot_plate_limits_field(axis, rcmb, ridges, trenches)

                misc.saveplot(fig, 'sx', timestep)

        # determine names of files
        ptn = misc.out_name('{}_{}_{}')
        stem = ptn.format(fids.fnames[0], istart, iend)
        idx = 0
        fmt = '{}.dat'
        while pathlib.Path(fmt.format(stem, idx)).is_file():
            fmt = '{}_{}.dat'
            idx += 1
        fids.fnames = [fmt.format(ptn.format(fname, istart, iend), idx)
                       for fname in fids.fnames]


def cmd():
    """Implementation of plates subcommand.

    Other Parameters:
        conf.plates
        conf.scaling
        conf.plot
        conf.core
    """
    sdat = StagyyData()
    conf.plates.plot = set_of_vars(conf.plates.plot)
    if not conf.plates.vzcheck:
        conf.scaling.dimensional = True
        conf.scaling.factors['Pa'] = 'M'
        main_plates(sdat)
    else:
        seuil_memz = 0
        nb_plates = []
        time = []
        ch2o = []
        istart, iend = None, None

        for step in sdat.walk.filter(fields=['T']):
            # could check other fields too
            if conf.plates.timeprofile:
                time.append(step.timeinfo.loc['t'])
                ch2o.append(step.timeinfo.loc[0])
            istart = step.isnap if istart is None else istart
            iend = step.isnap
            limits, nphi, dvphi, seuil_memz, vphi_surf =\
                detect_plates_vzcheck(step, seuil_memz)
            water_profile = np.mean(step.fields['wtr'][0, :, :, 0], axis=0)
            limits.sort()
            sizeplates = [limits[0] + nphi - limits[-1]]
            for lim in range(1, len(limits)):
                sizeplates.append(limits[lim] - limits[lim - 1])
            lim = len(limits) * [max(dvphi)]
            fig = plt.figure()
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
            misc.saveplot(fig, 'plates', step.isnap)

            nb_plates.append(len(limits))

        if conf.plates.timeprofile:
            for i in range(2, len(nb_plates) - 3):
                nb_plates[i] = (nb_plates[i - 2] + nb_plates[i - 1] +
                                nb_plates[i] + nb_plates[i + 1] +
                                nb_plates[i + 2]) / 5
            figt = plt.figure()
            plt.subplot(121)
            plt.axis([time[0], time[-1], 0, np.max(nb_plates)])
            plt.plot(time, nb_plates)
            plt.subplot(122)
            plt.plot(time, ch2o)
            misc.saveplot(figt, 'plates_{}_{}'.format(istart, iend))
