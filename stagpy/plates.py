"""Plate analysis."""

from contextlib import suppress

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.signal import argrelextrema

from . import conf, error, field, phyvars
from ._helpers import saveplot, list_of_vars
from ._step import Field
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


def detect_plates(step):
    """Detect plates using derivative of horizontal velocity."""
    vphi = step.fields['v2'].values[0, :, :, 0]
    ph_coord = step.geom.p_centers

    indsurf = _isurf(step)

    # dvphi/dphi at cell-center
    dvph2 = np.diff(vphi[:, indsurf]) / (ph_coord[1] - ph_coord[0])

    # finding trenches
    pom2 = np.copy(dvph2)
    maskbigdvel = np.amin(dvph2) * 0.2
    pom2[pom2 > maskbigdvel] = maskbigdvel
    trench_span = 15 if step.sdat.par['boundaries']['air_layer'] else 10
    argless_dv = argrelextrema(
        pom2, np.less, order=trench_span, mode='wrap')[0]
    trench = ph_coord[argless_dv]

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
            ridge = np.delete(ridge, np.array(argdel))

    return trench, ridge, argless_dv


def plot_plate_limits(axis, ridges, trenches):
    """Plot lines designating ridges and trenches."""
    for trench in trenches:
        axis.axvline(x=trench, color='red', ls='dashed', alpha=0.4)
    for ridge in ridges:
        axis.axvline(x=ridge, color='green', ls='dashed', alpha=0.4)


def plot_plate_limits_field(axis, rcmb, ridges, trenches):
    """Plot arrows designating ridges and trenches in 2D field plots."""
    for trench in trenches:
        xxd = (rcmb + 1.02) * np.cos(trench)  # arrow begin
        yyd = (rcmb + 1.02) * np.sin(trench)  # arrow begin
        xxt = (rcmb + 1.35) * np.cos(trench)  # arrow end
        yyt = (rcmb + 1.35) * np.sin(trench)  # arrow end
        axis.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                      arrowprops=dict(facecolor='red', shrink=0.05),
                      annotation_clip=False)
    for ridge in ridges:
        xxd = (rcmb + 1.02) * np.cos(ridge)
        yyd = (rcmb + 1.02) * np.sin(ridge)
        xxt = (rcmb + 1.35) * np.cos(ridge)
        yyt = (rcmb + 1.35) * np.sin(ridge)
        axis.annotate('', xy=(xxd, yyd), xytext=(xxt, yyt),
                      arrowprops=dict(facecolor='green', shrink=0.05),
                      annotation_clip=False)


def _isurf(snap):
    """Return index of surface accounting for air layer."""
    if snap.sdat.par['boundaries']['air_layer']:
        dsa = snap.sdat.par['boundaries']['air_thickness']
        # we are a bit below the surface; delete "-some number" to be just
        # below the surface (that is considered plane here); should check if
        # in the thermal boundary layer
        isurf = np.argmin(
            np.abs(1 - dsa - snap.geom.r_centers + snap.geom.rcmb)) - 4
    else:
        isurf = -1
    return isurf


def _surf_diag(snap, name):
    """Get a surface field.

    Can be a sfield, a regular scalar field evaluated at the surface,
    or dv2 (which is dvphi/dphi).
    """
    isurf = _isurf(snap)
    with suppress(error.UnknownVarError):
        return snap.sfields[name]
    with suppress(error.UnknownVarError):
        field, meta = snap.fields[name]
        return Field(field[0, :, isurf, 0], meta)
    if name == 'dv2':
        vphi = snap.fields['v2'].values[0, :, isurf, 0]
        dvphi = np.diff(vphi) / (snap.geom.r_centers[isurf] *
                                 np.diff(snap.geom.p_walls))
        return Field(dvphi, phyvars.Varf(r"$dv_\phi/d\phi$", '1/s'))
    raise error.UnknownVarError(name)


def _continents_location(snap, at_surface=True):
    """Location of continents as a boolean array.

    If at_surface is True, it is evaluated only at the surface, otherwise it is
    evaluated in the entire domain.
    """
    if at_surface:
        if snap.sdat.par['boundaries']['air_layer']:
            icont = _isurf(snap) - 6
        else:
            icont = -1
    else:
        icont = slice(None)
    csurf = snap.fields['c'].values[0, :, icont, 0]
    if snap.sdat.par['boundaries']['air_layer'] and\
       not snap.sdat.par['continents']['proterozoic_belts']:
        return (csurf >= 3) & (csurf <= 4)
    elif (snap.sdat.par['boundaries']['air_layer'] and
          snap.sdat.par['continents']['proterozoic_belts']):
        return (csurf >= 3) & (csurf <= 5)
    elif snap.sdat.par['tracersin']['tracers_weakcrust']:
        return csurf >= 3
    return csurf >= 2


def plot_at_surface(snap, names, trenches, ridges):
    """Plot surface diagnostics."""
    for vfig in list_of_vars(names):
        fig, axes = plt.subplots(nrows=len(vfig), sharex=True,
                                 figsize=(12, 2 * len(vfig)))
        axes = [axes] if len(vfig) == 1 else axes
        fname = 'plates_surf_'
        for axis, vplt in zip(axes, vfig):
            fname += '_'.join(vplt) + '_'
            label = ''
            for name in vplt:
                data, meta = _surf_diag(snap, name)
                label = meta.description
                phi = (snap.geom.p_centers if data.size == snap.geom.nptot
                       else snap.geom.p_walls)
                axis.plot(phi, data, label=label)
                axis.set_ylim([conf.plot.vmin, conf.plot.vmax])
            if conf.plates.continents:
                continents = _continents_location(snap)
                ymin, ymax = axis.get_ylim()
                axis.fill_between(snap.geom.p_centers, ymin, ymax,
                                  where=continents, alpha=0.2,
                                  facecolor='#8B6914')
                axis.set_ylim([ymin, ymax])
            # need to have trenches and ridges detection without side-
            # effects, call it here. Memoized.
            plot_plate_limits(axis, ridges, trenches)
            if len(vplt) == 1:
                axis.set_ylabel(label)
            else:
                axis.legend()
        axes[-1].set_xlabel(r"$\phi$")
        axes[-1].set_xlim(snap.geom.p_walls[[0, -1]])
        saveplot(fig, fname, snap.isnap)


def _write_trench_diagnostics(step, vrms_surf, itrenches, fid):
    """Print out some trench diagnostics."""
    time = step.time * vrms_surf *\
        conf.scaling.ttransit / conf.scaling.yearins / 1.e6
    isurf = _isurf(step)
    trenches = step.geom.p_centers[itrenches]

    # vphi at trenches
    vphi = step.fields['v2'].values[0, :, isurf, 0]
    vphi = (vphi[1:] + vphi[:-1]) / 2
    v_trenches = vphi[itrenches]

    if 'age' in step.fields:
        agefld = step.fields['age'].values[0, :, isurf, 0]
        age_surface = np.ma.masked_where(agefld < 1.e-5, agefld)
        age_surface_dim = age_surface * vrms_surf *\
            conf.scaling.ttransit / conf.scaling.yearins / 1.e6
        agetrenches = age_surface_dim[itrenches]  # age at the trench
    else:
        agetrenches = np.zeros(len(itrenches))

    phi_cont = step.geom.p_centers[_continents_location(step)]

    distance_subd = []
    ph_cont_subd = []
    for trench_i in trenches:
        # compute distance between subduction and continent
        angdistance1 = np.abs(phi_cont - trench_i)
        angdistance2 = 2 * np.pi - angdistance1
        angdistance = np.minimum(angdistance1, angdistance2)
        i_closest = np.argmin(angdistance)

        ph_cont_subd.append(phi_cont[i_closest])
        distance_subd.append(angdistance[i_closest])

    # writing the output into a file, all time steps are in one file
    for isubd in range(len(trenches)):
        fid.write(
            "%6.0f %11.7f %11.3f %10.6f %10.6f %10.6f %10.6f %11.3f\n" % (
                step.isnap,
                step.time,
                time,
                trenches[isubd],
                v_trenches[isubd],
                distance_subd[isubd],
                ph_cont_subd[isubd],
                agetrenches[isubd]))


def plot_scalar_field(step, fieldname, ridges, trenches):
    """Plot scalar field with plate information."""
    fig, axis, _, _ = field.plot_scalar(step, fieldname)

    if conf.plates.continents:
        c_field = np.ma.masked_where(
            ~_continents_location(step, at_surface=False),
            step.fields['c'].values[0, :, :, 0])
        cbar = conf.field.colorbar
        conf.field.colorbar = False
        cmap = colors.ListedColormap(["k", "g", "m"])
        field.plot_scalar(step, 'c', c_field, axis, cmap=cmap,
                          norm=colors.BoundaryNorm([2, 3, 4, 5], cmap.N))
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


def cmd():
    """Implementation of plates subcommand.

    Other Parameters:
        conf.plates
        conf.scaling
        conf.plot
        conf.core
    """
    sdat = StagyyData()
    if not conf.plates.vzcheck:
        isurf = _isurf(next(iter(sdat.walk)))
        vrms_surf = sdat.walk.filter(rprofs=True)\
            .rprofs_averaged['vhrms'].values[isurf]

        with open(f'plates_trenches_{sdat.walk.stepstr}.dat', 'w') as fid:
            fid.write('#  istep     time   time_My   phi_trench  vel_trench  '
                      'distance     phi_cont  age_trench_My\n')

            for step in sdat.walk.filter(fields=['T']):
                # could check other fields too
                trenches, ridges, itrenches = detect_plates(step)

                _write_trench_diagnostics(step, vrms_surf, itrenches, fid)
                plot_at_surface(step, conf.plates.plot, trenches, ridges)
                plot_scalar_field(step, conf.plates.field, ridges, trenches)
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
