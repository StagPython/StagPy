"""Plate analysis."""

from contextlib import suppress
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.signal import argrelmin, argrelmax

from . import conf, error, field, phyvars
from ._helpers import saveplot, list_of_vars
from ._step import Field
from .stagyydata import StagyyData


def _vzcheck(iphis, snap, vz_thres):
    """Detect plates and check with vz and plate size."""
    # verifying vertical velocity
    vzabs = np.abs(snap.fields['v3'].values[0, ..., 0])
    argdel = []
    for i, iphi in enumerate(iphis):
        vzm = np.mean(vzabs[[iphi - 1, iphi, iphi + 1], :])
        if vzm < vz_thres:
            argdel.append(i)
    return np.delete(iphis, argdel)


@lru_cache()
def detect_plates(snap, vz_thres_ratio=0):
    """Detect plates using derivative of horizontal velocity."""
    dvphi = _surf_diag(snap, 'dv2').values

    # finding trenches
    dvphi_saturated = np.copy(dvphi)
    max_dvphi = np.amin(dvphi) * 0.2
    dvphi_saturated[dvphi > max_dvphi] = max_dvphi
    trench_span = 15 if snap.sdat.par['boundaries']['air_layer'] else 10
    itrenches = argrelmin(dvphi_saturated, order=trench_span, mode='wrap')[0]

    # finding ridges
    dvphi_saturated = np.copy(dvphi)
    min_dvphi = np.amax(dvphi) * 0.2
    dvphi_saturated[dvphi < min_dvphi] = min_dvphi
    ridge_span = 20
    iridges = argrelmax(dvphi_saturated, order=ridge_span, mode='wrap')[0]

    # elimination of ridges that are too close to a trench
    phi = snap.geom.p_centers
    phi_trenches = phi[itrenches]
    argdel = []
    if itrenches.size and iridges.size:
        for i, iridge in enumerate(iridges):
            mdistance = np.amin(np.abs(phi_trenches - phi[iridge]))
            if mdistance < 0.016:
                argdel.append(i)
        if argdel:
            iridges = np.delete(iridges, argdel)

    # additional check on vz
    if vz_thres_ratio > 0:
        r_w = snap.geom.r_walls
        vz_mean = (np.sum(snap.rprofs['vzabs'].values * np.diff(r_w)) /
                   (r_w[-1] - r_w[0]))
        vz_thres = vz_mean * vz_thres_ratio
        itrenches = _vzcheck(itrenches, snap, vz_thres)
        iridges = _vzcheck(iridges, snap, vz_thres)

    return itrenches, iridges


def _plot_plate_limits(axis, trenches, ridges):
    """Plot lines designating ridges and trenches."""
    for trench in trenches:
        axis.axvline(x=trench, color='red', ls='dashed', alpha=0.4)
    for ridge in ridges:
        axis.axvline(x=ridge, color='green', ls='dashed', alpha=0.4)


def _plot_plate_limits_field(axis, rcmb, trenches, ridges):
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


def plot_at_surface(snap, names):
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
            phi = snap.geom.p_centers
            itrenches, iridges = detect_plates(snap, conf.plates.vzratio)
            _plot_plate_limits(axis, phi[itrenches], phi[iridges])
            if len(vplt) == 1:
                axis.set_ylabel(label)
            else:
                axis.legend()
        axes[-1].set_xlabel(r"$\phi$")
        axes[-1].set_xlim(snap.geom.p_walls[[0, -1]])
        saveplot(fig, fname, snap.isnap)


def _write_trench_diagnostics(step, vrms_surf, fid):
    """Print out some trench diagnostics."""
    itrenches, _ = detect_plates(step, conf.plates.vzratio)
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

    if conf.plates.continents:
        phi_cont = step.geom.p_centers[_continents_location(step)]
    else:
        phi_cont = np.array([np.nan])

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


def plot_scalar_field(snap, fieldname):
    """Plot scalar field with plate information."""
    fig, axis, _, _ = field.plot_scalar(snap, fieldname)

    if conf.plates.continents:
        c_field = np.ma.masked_where(
            ~_continents_location(snap, at_surface=False),
            snap.fields['c'].values[0, :, :, 0])
        cbar = conf.field.colorbar
        conf.field.colorbar = False
        cmap = colors.ListedColormap(["k", "g", "m"])
        field.plot_scalar(snap, 'c', c_field, axis, cmap=cmap,
                          norm=colors.BoundaryNorm([2, 3, 4, 5], cmap.N))
        conf.field.colorbar = cbar

    # plotting velocity vectors
    field.plot_vec(axis, snap, 'sx' if conf.plates.stress else 'v')

    # Put arrow where ridges and trenches are
    phi = snap.geom.p_centers
    itrenches, iridges = detect_plates(snap, conf.plates.vzratio)
    _plot_plate_limits_field(axis, snap.geom.rcmb,
                             phi[itrenches], phi[iridges])

    saveplot(fig, f'plates_{fieldname}', snap.isnap,
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
        xzoom = (snap.geom.rcmb + 1) * np.cos(np.radians(conf.plates.zoom))
        yzoom = (snap.geom.rcmb + 1) * np.sin(np.radians(conf.plates.zoom))
        axis.set_xlim(xzoom - ladd, xzoom + radd)
        axis.set_ylim(yzoom - dadd, yzoom + uadd)
        saveplot(fig, f'plates_zoom_{fieldname}', snap.isnap)


def cmd():
    """Implementation of plates subcommand.

    Other Parameters:
        conf.plates
        conf.scaling
        conf.plot
        conf.core
    """
    sdat = StagyyData()

    isurf = _isurf(next(iter(sdat.walk)))
    vrms_surf = sdat.walk.filter(rprofs=True)\
        .rprofs_averaged['vhrms'].values[isurf]
    nb_plates = []
    time = []
    istart, iend = None, None

    with open(f'plates_trenches_{sdat.walk.stepstr}.dat', 'w') as fid:
        fid.write('#  istep     time   time_My   phi_trench  vel_trench  '
                  'distance     phi_cont  age_trench_My\n')

        for step in sdat.walk.filter(fields=['T']):
            # could check other fields too
            _write_trench_diagnostics(step, vrms_surf, fid)
            plot_at_surface(step, conf.plates.plot)
            plot_scalar_field(step, conf.plates.field)
            if conf.plates.nbplates:
                time.append(step.timeinfo.loc['t'])
                itr, ird = detect_plates(step, conf.plates.vzratio)
                nb_plates.append(itr.size + ird.size)
                istart = step.isnap if istart is None else istart
                iend = step.isnap
            if conf.plates.distribution:
                phi = step.geom.p_centers
                itr, ird = detect_plates(step, conf.plates.vzratio)
                limits = np.concatenate((itr, ird))
                limits.sort()
                sizeplates = [phi[limits[0]] + 2 * np.pi - phi[limits[-1]]]
                for lim in range(1, len(limits)):
                    sizeplates.append(phi[limits[lim]] - phi[limits[lim - 1]])
                fig, axis = plt.subplots()
                axis.hist(sizeplates, 10, (0, np.pi))
                axis.set_ylabel("Number of plates")
                axis.set_xlabel(r"$\phi$-span")
                saveplot(fig, 'plates_size_distribution', step.isnap)

        if conf.plates.nbplates:
            figt, axis = plt.subplots()
            axis.plot(time, nb_plates)
            axis.set_xlabel("Time")
            axis.set_ylabel("Number of plates")
            saveplot(figt, f'plates_{istart}_{iend}')
