"""Plate analysis."""

from contextlib import ExitStack

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from . import conf, error, field, phyvars
from ._helpers import saveplot
from .stagyydata import StagyyData


def detect_plates_vzcheck(step, vz_thres_ratio=0):
    """Detect plates and check with vz and plate size."""
    v_z = step.fields['v3'].values[0, :-1, :, 0]
    v_x = step.fields['v2'].values[0, :, :, 0]
    tcell = step.fields['T'].values[0, :, :, 0]
    n_z = step.geom.nztot
    nphi = step.geom.nptot
    r_c = step.geom.r_centers
    r_w = step.geom.r_walls
    dphi = 1 / nphi

    flux_c = n_z * [0]
    for i_z in range(0, n_z):
        flux_c[i_z] = np.sum((tcell[:, i_z] - step.timeinfo.loc['Tmean']) *
                             v_z[:, i_z]) * r_w[i_z] * dphi

    # checking stagnant lid, criterion seems weird!
    if all(abs(flux_c[i_z]) <= np.max(flux_c) / 50
           for i_z in range(n_z - n_z // 20, n_z)):
        raise error.StagnantLidError(step.sdat)

    # verifying horizontal plate speed and closeness of plates
    vphi_surf = v_x[:, -1]
    dvphi = np.diff(vphi_surf) / (r_c[-1] * dphi)
    dvx_thres = 16 * step.timeinfo.loc['vrms']

    limits = [
        phi for phi in range(nphi)
        if (abs(dvphi[phi]) >= dvx_thres and
            all(abs(dvphi[i % nphi]) <= abs(dvphi[phi])
                for i in range(phi - nphi // 33, phi + nphi // 33)))
    ]

    # verifying vertical velocity
    vz_thres = 0
    if vz_thres_ratio > 0:
        vz_mean = (np.sum(step.rprofs['vzabs'].values * np.diff(r_w)) /
                   (r_w[-1] - r_w[0]))
        vz_thres = vz_mean * vz_thres_ratio
    k = 0
    for i in range(len(limits)):
        vzm = 0
        phi = limits[i - k]
        for i_z in range(1 if phi == nphi - 1 else 0, n_z):
            vzm += (abs(v_z[phi, i_z]) +
                    abs(v_z[phi - 1, i_z]) +
                    abs(v_z[(phi + 1) % nphi, i_z])) / (n_z * 3)

        if vzm < vz_thres:
            limits.remove(phi)
            k += 1

    return limits, dvphi, vphi_surf


def detect_plates(step, vrms_surface, fids, time):
    """Detect plates using derivative of horizontal velocity."""
    vphi = step.fields['v2'].values[0, :, :, 0]
    ph_coord = step.geom.p_centers

    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; should check if you are in the
        # thermal boundary layer
        indsurf = np.argmin(
            np.abs(1 - dsa - step.geom.r_centers + step.geom.rcmb)) - 4
    else:
        indsurf = -1

    # vphi at cell-center
    vph2 = 0.5 * (vphi[1:] + vphi[:-1])
    # dvphi/dphi at cell-center
    dvph2 = np.diff(vphi[:, indsurf]) / (ph_coord[1] - ph_coord[0])

    # finding trenches
    pom2 = np.copy(dvph2)
    maskbigdvel = -vrms_surface * (
        30 if step.sdat.par['boundaries']['air_layer'] else 10)
    pom2[pom2 > maskbigdvel] = maskbigdvel
    trench_span = 15 if step.sdat.par['boundaries']['air_layer'] else 10
    argless_dv = argrelextrema(
        pom2, np.less, order=trench_span, mode='wrap')[0]
    trench = ph_coord[argless_dv]
    velocity_trench = vph2[argless_dv, indsurf]
    dv_trench = dvph2[argless_dv]

    # finding ridges
    pom2 = np.copy(dvph2)
    masksmalldvel = np.amax(dvph2) * 0.2
    pom2[pom2 < masksmalldvel] = masksmalldvel
    ridge_span = 20
    arggreat_dv = argrelextrema(
        pom2, np.greater, order=ridge_span, mode='wrap')[0]
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
    if 'age' in conf.plates.plot:
        agefld = step.fields['age'].values[0, :, :, 0]
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
            step.time,
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
    vphi = step.fields['v2'].values[0, :, :, 0]
    tempfld = step.fields['T'].values[0, :, :, 0]
    concfld = step.fields['c'].values[0, :, :, 0]
    timestep = step.isnap

    if step.sdat.par['boundaries']['air_layer']:
        dsa = step.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; delete "-some number"
        # to be just below
        # the surface (that is considered plane here); should check if you are
        # in the thermal boundary layer
        indsurf = np.argmin(
            np.abs(1 - dsa - step.geom.r_centers + step.geom.rcmb)) - 4
        # depth to detect the continents
        indcont = np.argmin(
            np.abs(1 - dsa - step.geom.r_centers + step.geom.rcmb)) - 10
    else:
        indsurf = -1
        indcont = -1  # depth to detect continents

    if step.sdat.par['boundaries']['air_layer'] and\
       not step.sdat.par['continents']['proterozoic_belts']:
        continents = np.ma.masked_where(
            np.logical_or(concfld[:, indcont] < 3,
                          concfld[:, indcont] > 4),
            concfld[:, indcont])
    elif (step.sdat.par['boundaries']['air_layer'] and
          step.sdat.par['continents']['proterozoic_belts']):
        continents = np.ma.masked_where(
            np.logical_or(concfld[:, indcont] < 3,
                          concfld[:, indcont] > 5),
            concfld[:, indcont])
    elif step.sdat.par['tracersin']['tracers_weakcrust']:
        continents = np.ma.masked_where(
            concfld[:, indcont] < 3, concfld[:, indcont])
    else:
        continents = np.ma.masked_where(
            concfld[:, indcont] < 2, concfld[:, indcont])

    # masked array, only continents are true
    continentsall = continents / continents

    ph_coord = step.geom.p_centers

    # velocity derivative at cell-center
    dvph2 = np.diff(vphi[:, indsurf]) / (ph_coord[1] - ph_coord[0])

    # plotting
    fig0, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax1.plot(ph_coord, concfld[:, indsurf], color='g', label='Conc')
    ax2.plot(ph_coord, tempfld[:, indsurf], color='k', label='Temp')
    ax3.plot(step.geom.p_walls, vphi[:, indsurf], label='Vel')

    ax1.fill_between(
        ph_coord, continents, 1., facecolor='#8B6914', alpha=0.2)
    ax2.fill_between(
        ph_coord, continentsall, 0., facecolor='#8B6914', alpha=0.2)

    tempmin = step.sdat.par['boundaries']['topT_val'] * 0.9\
        if step.sdat.par['boundaries']['topT_mode'] == 'iso' else 0.0
    tempmax = step.sdat.par['boundaries']['botT_val'] * 0.35\
        if step.sdat.par['boundaries']['botT_mode'] == 'iso' else 0.8

    ax2.set_ylim(tempmin, tempmax)
    ax3.fill_between(
        ph_coord, continentsall * round(1.5 * np.amax(dvph2), 1),
        round(np.amin(dvph2) * 1.1, 1), facecolor='#8B6914', alpha=0.2)
    ax3.set_ylim(conf.plates.vmin, conf.plates.vmax)

    ax1.set_ylabel("Concentration")
    ax2.set_ylabel("Temperature")
    ax3.set_ylabel("Velocity")
    ax1.set_title(timestep)
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.time, 8)),
             transform=ax1.transAxes)

    plot_plate_limits(ax3, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)

    saveplot(fig0, 'sveltempconc', timestep)

    # plotting velocity and velocity derivative
    fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(step.geom.p_walls, vphi[:, indsurf], label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity")
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.time, 8)),
             transform=ax1.transAxes)
    ax2.plot(ph_coord, dvph2, color='k', label='dv')
    ax2.set_ylabel("dv")

    plot_plate_limits(ax1, ridge, trench, conf.plates.vmin,
                      conf.plates.vmax)
    plot_plate_limits(ax2, ridge, trench, conf.plates.dvmin,
                      conf.plates.dvmax)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_title(timestep)

    ax1.fill_between(
        ph_coord, continentsall * conf.plates.vmin, conf.plates.vmax,
        facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
    ax2.fill_between(
        ph_coord, continentsall * conf.plates.dvmin,
        conf.plates.dvmax, facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(conf.plates.dvmin, conf.plates.dvmax)

    saveplot(fig0, 'sveldvel', timestep)

    # plotting velocity and second invariant of stress
    if 'str' in conf.plates.plot:
        stressfld = step.fields['sII'].values[0, :, :, 0]
        fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(step.geom.p_walls, vphi[:, indsurf], label='Vel')
        ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                    color='black', ls='solid', alpha=0.2)
        ax1.set_ylabel("Velocity")
        ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax1.transAxes)
        ax1.text(0.01, 1.07, str(round(step.time, 8)),
                 transform=ax1.transAxes)
        ax2.plot(ph_coord,
                 stressfld[:, indsurf] * step.sdat.scales.stress / 1.e6,
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
            ph_coord, continentsall * conf.plates.vmin,
            conf.plates.vmax, facecolor='#8B6914', alpha=0.2)
        ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
        ax2.fill_between(
            ph_coord, continentsall * conf.plates.dvmin,
            conf.plates.dvmax,
            facecolor='#8B6914', alpha=0.2)

        saveplot(fig0, 'svelstress', timestep)

    # plotting velocity
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(step.geom.p_walls, vphi[:, indsurf], label='Vel')
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
        agefld = step.fields['age'].values[0, :, :, 0]
        age_surface = np.ma.masked_where(
            agefld[:, indsurf] < 0.00001, agefld[:, indsurf])
        age_surface_dim = (age_surface * vrms_surface * conf.scaling.ttransit /
                           conf.scaling.yearins / 1.e6)

        fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax3.plot(ph_coord, vphi[:, indsurf], label='Vel')
        ax3.axhline(
            y=0, xmin=0, xmax=2 * np.pi,
            color='black', ls='solid', alpha=0.2)
        ax3.set_ylim(conf.plates.vmin, conf.plates.vmax)
        ax3.set_ylabel("Velocity")
        ax3.text(0.95, 1.07, str(round(time, 0)) + ' My',
                 transform=ax3.transAxes)
        ax3.fill_between(
            ph_coord, continentsall * conf.plates.vmax,
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
            angdistance1 = abs(ph_coord[continentsall == 1] - trench_i)
            angdistance2 = 2. * np.pi - angdistance1
            angdistance = np.minimum(angdistance1, angdistance2)
            distancecont = min(angdistance)
            argdistancecont = np.argmin(angdistance)
            continentpos = ph_coord[continentsall == 1][argdistancecont]

            ph_trench_subd.append(trench_i)
            age_subd.append(agetrench[i])
            ph_cont_subd.append(continentpos)
            distance_subd.append(distancecont)
            times_subd.append(step.time)

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
        ph_coord, continentsall * conf.plates.vmin,
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
        ph_coord, continentsall * conf.plates.topomax,
        conf.plates.topomin, facecolor='#8B6914', alpha=0.2)
    plot_plate_limits(ax2, ridge, trench, conf.plates.topomin,
                      conf.plates.topomax)
    ax1.set_title(timestep)
    saveplot(fig1, 'sveltopo', timestep)

    if 'age' in conf.plates.plot:
        ax4.set_ylabel("Seafloor age [My]")
        # in dimensions
        ax4.plot(ph_coord, age_surface_dim, color='black')
        ax4.set_xlim(0, 2 * np.pi)
        ax4.fill_between(
            ph_coord, continentsall * conf.plates.agemax,
            conf.plates.agemin, facecolor='#8B6914', alpha=0.2)
        ax4.set_ylim(conf.plates.agemin, conf.plates.agemax)
        plot_plate_limits(ax4, ridge, trench, conf.plates.agemin,
                          conf.plates.agemax)
        ax3.set_title(timestep)
        saveplot(fig2, 'svelage', timestep)

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


def lithospheric_stress(step, trench, ridge, time):
    """Calculate stress in the lithosphere."""
    timestep = step.isnap
    base_lith = step.geom.rcmb + 1 - 0.105

    stressfld = step.fields['sII'].values[0, :, :, 0]
    r_centers = np.outer(np.ones(stressfld.shape[0]), step.geom.r_centers)
    stressfld = np.ma.masked_where(r_centers < base_lith, stressfld)

    # stress integration in the lithosphere
    dzm = (step.geom.r_centers[1:] - step.geom.r_centers[:-1])
    stress_lith = np.sum((stressfld[:, 1:] * dzm.T), axis=1)
    ph_coord = step.geom.p_centers  # probably doesn't need alias

    # plot stress in the lithosphere
    fig, axis, _, _ = field.plot_scalar(step, 'sII', stressfld,
                                        cmap='plasma_r', vmin=0, vmax=300)
    # Annotation with time and step
    axis.text(1., 0.9, str(round(time, 0)) + ' My', transform=axis.transAxes)
    axis.text(1., 0.1, str(timestep), transform=axis.transAxes)
    saveplot(fig, 'lith', timestep)

    # velocity
    vphi = step.fields['v2'].values[0, :, :, 0]

    # position of continents
    concfld = step.fields['c'].values[0, :, :, 0]
    if step.sdat.par['boundaries']['air_layer']:
        # we are a bit below the surface; delete "-some number"
        # to be just below
        dsa = step.sdat.par['boundaries']['air_thickness']
        # depth to detect the continents
        indcont = np.argmin(
            np.abs(1 - dsa - step.geom.r_centers + step.geom.rcmb)) - 10
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
    ax1.plot(step.geom.p_walls, vphi[:, -1], label='Vel')
    ax1.axhline(y=0, xmin=0, xmax=2 * np.pi,
                color='black', ls='solid', alpha=0.2)
    ax1.set_ylabel("Velocity")
    ax1.text(0.95, 1.07, str(round(time, 0)) + ' My',
             transform=ax1.transAxes)
    ax1.text(0.01, 1.07, str(round(step.time, 8)),
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
        ph_coord, continentsall * conf.plates.vmin,
        conf.plates.vmax, facecolor='#8b6914', alpha=0.2)
    ax1.set_ylim(conf.plates.vmin, conf.plates.vmax)
    ax2.fill_between(
        ph_coord, continentsall * conf.plates.stressmin,
        conf.plates.lstressmax, facecolor='#8b6914', alpha=0.2)
    ax2.set_ylim(conf.plates.stressmin, conf.plates.lstressmax)

    saveplot(fig0, 'svelslith', timestep)


def set_of_vars(arg_plot):
    """Build set of needed variables.

    Args:
        arg_plot (str): string with variable names separated with ``,``.
    Returns:
        set of str: set of variables.
    """
    return set(var for var in arg_plot.split(',') if var in phyvars.PLATES)


def plot_scalar_field(step, fieldname, ridges, trenches):
    """Plot scalar field with plate information."""
    fig, axis, _, _ = field.plot_scalar(step, fieldname)

    if conf.plates.continents:
        concfld = step.fields['c'].values[0, :, :, 0]
        continentsfld = np.ma.masked_where(
            concfld < 3, concfld)  # plotting continents, to-do
        continentsfld = continentsfld / continentsfld
        cbar = conf.field.colorbar
        conf.field.colorbar = False
        field.plot_scalar(step, 'c', continentsfld, axis,
                          cmap='cool_r', vmin=0, vmax=0)
        cmap2 = plt.cm.ocean
        cmap2.set_over('m')
        conf.field.colorbar = cbar

    # plotting velocity vectors
    field.plot_vec(axis, step, 'sx' if conf.plates.stress else 'v')

    # Put arrow where ridges and trenches are
    plot_plate_limits_field(axis, step.geom.rcmb, ridges, trenches)

    saveplot(fig, f'plates_{fieldname}', step.isnap,
             close=conf.plates.zoom is None)

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
        xzoom = (step.geom.rcmb + 1) * np.cos(np.radians(conf.plates.zoom))
        yzoom = (step.geom.rcmb + 1) * np.sin(np.radians(conf.plates.zoom))
        axis.set_xlim(xzoom - ladd, xzoom + radd)
        axis.set_ylim(yzoom - dadd, yzoom + uadd)
        saveplot(fig, f'plates_zoom_{fieldname}', step.isnap)


def main_plates(sdat):
    """Plot several plates information."""
    # averaged horizontal surface velocity needed for redimensionalisation
    uprof_averaged, radius, _ = sdat.walk.filter(rprofs=True)\
        .rprofs_averaged['vhrms']
    if sdat.par['boundaries']['air_layer']:
        dsa = sdat.par['boundaries']['air_thickness']
        isurf = np.argmin(abs(radius - radius[-1] + dsa))
        vrms_surface = uprof_averaged[isurf]
        isurf = np.argmin(abs((1 - dsa) - radius))
        isurf -= 4  # why different isurf for the rest?
    else:
        isurf = -1
        vrms_surface = uprof_averaged[isurf]

    # determine names of files
    fnames = ['plate_velocity', 'distance_subd']
    fnames = [f'plates_{stem}_{sdat.walk.stepstr}' for stem in fnames]
    with ExitStack() as stack:
        fids = [stack.enter_context(open(fname, 'w')) for fname in fnames]
        fids[0].write('#  it  time  ph_trench vel_trench age_trench\n')
        fids[1].write('#  it      time   time [My]   distance     '
                      'ph_trench     ph_cont  age_trench [My]\n')

        for step in sdat.walk.filter(fields=['T']):
            # could check other fields too
            timestep = step.isnap
            print('Treating snapshot', timestep)

            # topography
            fname = sdat.filename('sc', timestep=timestep, suffix='.dat')
            topo = np.genfromtxt(str(fname))
            # rescaling topography!
            if sdat.par['boundaries']['air_layer']:
                topo[:, 1] = topo[:, 1] / (1. - dsa)

            time = step.time * vrms_surface *\
                conf.scaling.ttransit / conf.scaling.yearins / 1.e6
            trenches, ridges, agetrenches, _, _ =\
                detect_plates(step, vrms_surface, fids, time)
            plot_plates(step, time, vrms_surface, trenches, ridges,
                        agetrenches, topo, fids)

            # plot scalar field with position of trenches and ridges
            plot_scalar_field(step, conf.plates.field, ridges, trenches)


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
        nb_plates = []
        time = []
        istart, iend = None, None

        for step in sdat.walk.filter(fields=['T']):
            # could check other fields too
            if conf.plates.timeprofile:
                time.append(step.timeinfo.loc['t'])
            istart = step.isnap if istart is None else istart
            iend = step.isnap
            phi = step.geom.p_centers
            limits, dvphi, vphi_surf = detect_plates_vzcheck(step)
            limits.sort()
            sizeplates = [phi[limits[0]] + 2 * np.pi - phi[limits[-1]]]
            for lim in range(1, len(limits)):
                sizeplates.append(phi[limits[lim]] - phi[limits[lim - 1]])
            phi_lim = [phi[i] for i in limits]
            dvp_lim = [dvphi[i] for i in limits]
            fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6.4, 6.4))
            axes[0].plot(step.geom.p_walls, vphi_surf)
            axes[0].set_ylabel(r"Horizontal velocity $v_\phi$")
            axes[1].plot(phi, dvphi)
            axes[1].scatter(phi_lim, dvp_lim, color='red')
            axes[1].set_ylabel(r"$dv_\phi/d\phi$")
            axes[1].set_xlabel(r"$\phi$")
            saveplot(fig, 'plate_limits', step.isnap)
            fig, axis = plt.subplots()
            axis.hist(sizeplates, 10, (0, np.pi))
            axis.set_ylabel("Number of plates")
            axis.set_xlabel(r"$\phi$-span")
            saveplot(fig, 'plate_size_distribution', step.isnap)

            nb_plates.append(len(limits))

        if conf.plates.timeprofile:
            for i in range(2, len(nb_plates) - 3):
                nb_plates[i] = (nb_plates[i - 2] + nb_plates[i - 1] +
                                nb_plates[i] + nb_plates[i + 1] +
                                nb_plates[i + 2]) / 5
            figt, axis = plt.subplots()
            axis.plot(time, nb_plates)
            axis.set_xlabel("Time")
            axis.set_ylabel("Number of plates")
            saveplot(figt, f'plates_{istart}_{iend}')
