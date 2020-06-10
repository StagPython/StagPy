"""Plot scalar and vector fields."""

from itertools import chain

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import conf, misc, phyvars
from .error import NotAvailableError
from .stagyydata import StagyyData


def _threed_extract(step, var):
    """Return suitable slices and mesh for 3D fields."""
    is_vector = not valid_field_var(var)
    i_x = conf.field.ix
    i_y = conf.field.iy
    i_z = conf.field.iz
    if i_x is not None or i_y is not None:
        i_z = None
    if i_x is not None or i_z is not None:
        i_y = None
    if i_x is None and i_y is None and i_z is None:
        i_x = 0
    if i_x is not None:
        xmesh, ymesh = step.geom.y_mesh[i_x, :, :], step.geom.z_mesh[i_x, :, :]
        i_y = i_z = slice(None)
        varx, vary = var + '2', var + '3'
    elif i_y is not None:
        xmesh, ymesh = step.geom.x_mesh[:, i_y, :], step.geom.z_mesh[:, i_y, :]
        i_x = i_z = slice(None)
        varx, vary = var + '1', var + '3'
    else:
        xmesh, ymesh = step.geom.x_mesh[:, :, i_z], step.geom.y_mesh[:, :, i_z]
        i_x = i_y = slice(None)
        varx, vary = var + '1', var + '2'
    if is_vector:
        data = (step.fields[varx][i_x, i_y, i_z, 0],
                step.fields[vary][i_x, i_y, i_z, 0])
    else:
        data = step.fields[var][i_x, i_y, i_z, 0]
    return (xmesh, ymesh), data


def valid_field_var(var):
    """Whether a field variable is defined.

    This function checks if a definition of the variable exists in
    :data:`phyvars.FIELD` or :data:`phyvars.FIELD_EXTRA`.

    Args:
        var (str): the variable name to be checked.
    Returns:
        bool: True is the var is defined, False otherwise.
    """
    return var in phyvars.FIELD or var in phyvars.FIELD_EXTRA


def get_meshes_fld(step, var):
    """Return scalar field along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): scalar field name.
    Returns:
        tuple of :class:`numpy.array`: xmesh, ymesh, fld
            2D arrays containing respectively the x position, y position, and
            the value of the requested field.
    """
    fld = step.fields[var]
    if step.geom.threed and step.geom.cartesian:
        (xmesh, ymesh), fld = _threed_extract(step, var)
    elif step.geom.twod_xz:
        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]
        fld = fld[:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]
        fld = fld[0, :, :, 0]
    else:  # spherical yz
        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]
        fld = fld[0, :, :, 0]
    return xmesh, ymesh, fld


def get_meshes_vec(step, var):
    """Return vector field components along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): vector field name.
    Returns:
        tuple of :class:`numpy.array`: xmesh, ymesh, fldx, fldy
            2D arrays containing respectively the x position, y position, x
            component and y component of the requested vector field.
    """
    if step.geom.threed and step.geom.cartesian:
        (xmesh, ymesh), (vec1, vec2) = _threed_extract(step, var)
    elif step.geom.twod_xz:
        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]
        vec1 = step.fields[var + '1'][:, 0, :, 0]
        vec2 = step.fields[var + '3'][:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]
        vec1 = step.fields[var + '2'][0, :, :, 0]
        vec2 = step.fields[var + '3'][0, :, :, 0]
    else:  # spherical yz
        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]
        pmesh = step.geom.p_mesh[0, :, :]
        vec_phi = step.fields[var + '2'][0, :, :, 0]
        vec_r = step.fields[var + '3'][0, :, :, 0]
        vec1 = vec_r * np.cos(pmesh) - vec_phi * np.sin(pmesh)
        vec2 = vec_phi * np.cos(pmesh) + vec_r * np.sin(pmesh)
    return xmesh, ymesh, vec1, vec2


def plot_scalar(step, var, field=None, axis=None, **extra):
    """Plot scalar field.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): the scalar field name.
        field (:class:`numpy.array`): if not None, it is plotted instead of
            step.fields[var].  This is useful to plot a masked or rescaled
            array.  Note that if conf.scaling.dimensional is True, this
            field will be scaled accordingly.
        axis (:class:`matplotlib.axes.Axes`): the axis objet where the field
            should be plotted.  If set to None, a new figure with one subplot
            is created.
        extra (dict): options that will be passed on to
            :func:`matplotlib.axes.Axes.pcolormesh`.
    Returns:
        fig, axis, surf, cbar
            handles to various :mod:`matplotlib` objects, respectively the
            figure, the axis, the surface returned by
            :func:`~matplotlib.axes.Axes.pcolormesh`, and the colorbar returned
            by :func:`matplotlib.pyplot.colorbar`.
    """
    if var in phyvars.FIELD:
        meta = phyvars.FIELD[var]
    else:
        meta = phyvars.FIELD_EXTRA[var]
        meta = phyvars.Varf(misc.baredoc(meta.description), meta.dim)
    if step.geom.threed and step.geom.spherical:
        raise NotAvailableError(
            'plot_scalar not implemented for 3D spherical geometry')

    xmesh, ymesh, fld = get_meshes_fld(step, var)
    xmin, xmax = xmesh.min(), xmesh.max()
    ymin, ymax = ymesh.min(), ymesh.max()

    if field is not None:
        fld = field
    if conf.field.perturbation:
        fld = fld - np.mean(fld, axis=0)
    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)

    fld, unit = step.sdat.scale(fld, meta.dim)

    if axis is None:
        fig, axis = plt.subplots(ncols=1)
    else:
        fig = axis.get_figure()

    if step.sdat.par['magma_oceans_in']['magma_oceans_mode']:
        rcmb = step.sdat.par['geometry']['r_cmb']
        xmax = rcmb + 1
        ymax = xmax
        xmin = -xmax
        ymin = -ymax
        rsurf = xmax if step.timeinfo['thick_tmo'] > 0 \
            else step.geom.r_mesh[0, 0, -3]
        cmb = mpat.Circle((0, 0), rcmb, color='dimgray', zorder=0)
        psurf = mpat.Circle((0, 0), rsurf, color='indianred', zorder=0)
        axis.add_patch(psurf)
        axis.add_patch(cmb)

    extra_opts = dict(
        cmap=conf.field.cmap.get(var),
        vmin=conf.plot.vmin,
        vmax=conf.plot.vmax,
        norm=mpl.colors.LogNorm() if var == 'eta' else None,
        rasterized=conf.plot.raster,
        shading='gouraud' if conf.field.interpolate else 'flat',
    )
    extra_opts.update(extra)
    surf = axis.pcolormesh(xmesh, ymesh, fld, **extra_opts)

    cbar = None
    if conf.field.colorbar:
        cax = make_axes_locatable(axis).append_axes(
            'right', size="3%", pad=0.15)
        cbar = plt.colorbar(surf, cax=cax)
        cbar.set_label(meta.description +
                       (' pert.' if conf.field.perturbation else '') +
                       (' ({})'.format(unit) if unit else ''))
    if step.geom.spherical or conf.plot.ratio is None:
        axis.set_aspect('equal')
        axis.set_axis_off()
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())

    axis.set_adjustable('box')
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    return fig, axis, surf, cbar


def plot_iso(axis, step, var, **extra):
    """Plot isocontours of scalar field.

    Args:
        axis (:class:`matplotlib.axes.Axes`): the axis handler of an
            existing matplotlib figure where the isocontours should
            be plotted.
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): the scalar field name.
        extra (dict): options that will be passed on to
            :func:`matplotlib.axes.Axes.contour`.
    """
    xmesh, ymesh, fld = get_meshes_fld(step, var)
    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)
    extra_opts = dict(linewidths=1)
    if 'cmap' not in extra and conf.field.isocolors:
        extra_opts['colors'] = conf.field.isocolors.split(',')
    elif 'colors' not in extra:
        extra_opts['cmap'] = conf.field.cmap.get(var)
    extra_opts.update(extra)
    axis.contour(xmesh, ymesh, fld, **extra_opts)


def plot_vec(axis, step, var):
    """Plot vector field.

    Args:
        axis (:class:`matplotlib.axes.Axes`): the axis handler of an
            existing matplotlib figure where the vector field should
            be plotted.
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): the vector field name.
    """
    xmesh, ymesh, vec1, vec2 = get_meshes_vec(step, var)
    dipz = step.geom.nztot // 10
    if conf.field.shift:
        vec1 = np.roll(vec1, conf.field.shift, axis=0)
        vec2 = np.roll(vec2, conf.field.shift, axis=0)
    if step.geom.spherical or conf.plot.ratio is None:
        dipx = dipz
    else:
        dipx = step.geom.nytot if step.geom.twod_yz else step.geom.nxtot
        dipx = int(dipx // 10 * conf.plot.ratio) + 1
    axis.quiver(xmesh[::dipx, ::dipz], ymesh[::dipx, ::dipz],
                vec1[::dipx, ::dipz], vec2[::dipx, ::dipz],
                linewidths=1)


def _findminmax(sdat, sovs):
    """Find min and max values of several fields."""
    minmax = {}
    for step in sdat.walk.filter(snap=True):
        for var in sovs:
            if var in step.fields:
                if var in phyvars.FIELD:
                    dim = phyvars.FIELD[var].dim
                else:
                    dim = phyvars.FIELD_EXTRA[var].dim
                field, _ = sdat.scale(step.fields[var], dim)
                if var in minmax:
                    minmax[var] = (min(minmax[var][0], np.nanmin(field)),
                                   max(minmax[var][1], np.nanmax(field)))
                else:
                    minmax[var] = np.nanmin(field), np.nanmax(field)
    return minmax


def cmd():
    """Implementation of field subcommand.

    Other Parameters:
        conf.field
        conf.core
    """
    sdat = StagyyData()
    lovs = misc.list_of_vars(conf.field.plot)
    # no more than two fields in a subplot
    lovs = [[slov[:2] for slov in plov] for plov in lovs]
    minmax = {}
    if conf.plot.cminmax:
        conf.plot.vmin = None
        conf.plot.vmax = None
        sovs = set(slov[0] for plov in lovs for slov in plov)
        minmax = _findminmax(sdat, sovs)
    for step in sdat.walk.filter(snap=True):
        for vfig in lovs:
            fig, axes = plt.subplots(ncols=len(vfig), squeeze=False,
                                     figsize=(9 * len(vfig), 6))
            for axis, var in zip(axes[0], vfig):
                if var[0] not in step.fields:
                    print("'{}' field on snap {} not found".format(var[0],
                                                                   step.isnap))
                    continue
                opts = {}
                if var[0] in minmax:
                    opts = dict(vmin=minmax[var[0]][0], vmax=minmax[var[0]][1])
                plot_scalar(step, var[0], axis=axis, **opts)
                if len(var) == 2:
                    if valid_field_var(var[1]):
                        plot_iso(axis, step, var[1])
                    elif valid_field_var(var[1] + '1'):
                        plot_vec(axis, step, var[1])
            if conf.field.timelabel:
                time, unit = sdat.scale(step.timeinfo['t'], 's')
                time = misc.scilabel(time)
                axes[0, 0].text(0.02, 1.02, '$t={}$ {}'.format(time, unit),
                                transform=axes[0, 0].transAxes)
            oname = '_'.join(chain.from_iterable(vfig))
            plt.tight_layout(w_pad=3)
            misc.saveplot(fig, oname, step.isnap)
