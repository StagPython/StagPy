"""Plot scalar and vector fields."""

from __future__ import annotations
from itertools import chain
import typing

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import conf, phyvars, _helpers
from .error import NotAvailableError
from .stagyydata import StagyyData

if typing.TYPE_CHECKING:
    from typing import Tuple, Optional, Any, Iterable, Dict
    from numpy import ndarray
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from .datatypes import Varf
    from ._step import Step


# The location is off for vertical velocities: they have an extra
# point in (x,y) instead of z in the output


def _threed_extract(
    step: Step, var: str, walls: bool = False
) -> Tuple[Tuple[ndarray, ndarray], Any]:
    """Return suitable slices and coords for 3D fields."""
    is_vector = not valid_field_var(var)
    hwalls = is_vector or walls
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
        xcoord = step.geom.y_walls if hwalls else step.geom.y_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        i_y = i_z = slice(None)
        varx, vary = var + '2', var + '3'
    elif i_y is not None:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        i_x = i_z = slice(None)
        varx, vary = var + '1', var + '3'
    else:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.y_walls if hwalls else step.geom.y_centers
        i_x = i_y = slice(None)
        varx, vary = var + '1', var + '2'
    if is_vector:
        data = (step.fields[varx].values[i_x, i_y, i_z, 0],
                step.fields[vary].values[i_x, i_y, i_z, 0])
    else:
        data = step.fields[var].values[i_x, i_y, i_z, 0]
    return (xcoord, ycoord), data


def valid_field_var(var: str) -> bool:
    """Whether a field variable is defined.

    Args:
        var: the variable name to be checked.
    Returns:
        whether the var is defined in :data:`~stagpy.phyvars.FIELD` or
        :data:`~stagpy.phyvars.FIELD_EXTRA`.
    """
    return var in phyvars.FIELD or var in phyvars.FIELD_EXTRA


def get_meshes_fld(
    step: Step, var: str, walls: bool = False
) -> Tuple[ndarray, ndarray, ndarray, Varf]:
    """Return scalar field along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: scalar field name.
        walls: consider the walls as the relevant mesh.
    Returns:
        tuple (xmesh, ymesh, fld, meta).  2D arrays containing respectively the
        x position, y position, the values and the metadata of the requested
        field.
    """
    fld, meta = step.fields[var]
    hwalls = (walls or fld.shape[0] != step.geom.nxtot or
              fld.shape[1] != step.geom.nytot)
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), fld = _threed_extract(step, var, walls)
    elif step.geom.twod_xz:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        fld = fld[:, 0, :, 0]
    else:  # twod_yz
        xcoord = step.geom.y_walls if hwalls else step.geom.y_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        if step.geom.curvilinear:
            pmesh, rmesh = np.meshgrid(xcoord, ycoord, indexing='ij')
            xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
        fld = fld[0, :, :, 0]
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing='ij')
    return xmesh, ymesh, fld, meta


def get_meshes_vec(
    step: Step, var: str
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Return vector field components along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: vector field name.
    Returns:
        tuple (xmesh, ymesh, fldx, fldy).  2D arrays containing respectively
        the x position, y position, x component and y component of the
        requested vector field.
    """
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), (vec1, vec2) = _threed_extract(step, var)
    elif step.geom.twod_xz:
        xcoord, ycoord = step.geom.x_walls, step.geom.z_centers
        vec1 = step.fields[var + '1'].values[:, 0, :, 0]
        vec2 = step.fields[var + '3'].values[:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xcoord, ycoord = step.geom.y_walls, step.geom.z_centers
        vec1 = step.fields[var + '2'].values[0, :, :, 0]
        vec2 = step.fields[var + '3'].values[0, :, :, 0]
    else:  # spherical yz
        pcoord = step.geom.p_walls
        pmesh = np.outer(pcoord, np.ones(step.geom.nrtot))
        vec_phi = step.fields[var + '2'].values[0, :, :, 0]
        vec_r = step.fields[var + '3'].values[0, :, :, 0]
        vec1 = vec_r * np.cos(pmesh) - vec_phi * np.sin(pmesh)
        vec2 = vec_phi * np.cos(pmesh) + vec_r * np.sin(pmesh)
        pcoord, rcoord = step.geom.p_walls, step.geom.r_centers
        pmesh, rmesh = np.meshgrid(pcoord, rcoord, indexing='ij')
        xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing='ij')
    return xmesh, ymesh, vec1, vec2


def plot_scalar(step: Step, var: str, field: Optional[ndarray] = None,
                axis: Optional[Axes] = None, **extra: Any
                ) -> Tuple[Figure, Axes, QuadMesh, Colorbar]:
    """Plot scalar field.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of step.fields[var].  This is
            useful to plot a masked or rescaled array.  Note that if
            conf.scaling.dimensional is True, this field will be scaled
            accordingly.
        axis: the :class:`matplotlib.axes.Axes` object where the field should
            be plotted.  If set to None, a new figure with one subplot is
            created.
        extra: options that will be passed on to
            :func:`matplotlib.axes.Axes.pcolormesh`.
    Returns:
        fig, axis, surf, cbar
            handles to various :mod:`matplotlib` objects, respectively the
            figure, the axis, the surface returned by
            :func:`~matplotlib.axes.Axes.pcolormesh`, and the colorbar returned
            by :func:`matplotlib.pyplot.colorbar`.
    """
    if step.geom.threed and step.geom.spherical:
        raise NotAvailableError(
            'plot_scalar not implemented for 3D spherical geometry')

    xmesh, ymesh, fld, meta = get_meshes_fld(step, var,
                                             walls=not conf.field.interpolate)
    # interpolate at cell centers, this should be abstracted by field objects
    # via an "at_cell_centers" method or similar
    if fld.shape[0] > max(step.geom.nxtot, step.geom.nytot):
        fld = (fld[:-1] + fld[1:]) / 2

    if conf.field.interpolate and \
       step.geom.spherical and step.geom.twod_yz:
        # add one point to close spherical annulus
        xmesh = np.concatenate((xmesh, xmesh[:1]), axis=0)
        ymesh = np.concatenate((ymesh, ymesh[:1]), axis=0)
        newline = (fld[:1] + fld[-1:]) / 2
        fld = np.concatenate((fld, newline), axis=0)
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
            else step.geom.r_walls[0, 0, -3]
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
                       (f' ({unit})' if unit else ''))
    if step.geom.spherical or conf.plot.ratio is None:
        axis.set_aspect('equal')
        axis.set_axis_off()
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())

    axis.set_adjustable('box')
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    return fig, axis, surf, cbar


def plot_iso(axis: Axes, step: Step, var: str,
             field: Optional[ndarray] = None, **extra: Any) -> None:
    """Plot isocontours of scalar field.

    Args:
        axis: the :class:`matplotlib.axes.Axes` of an existing matplotlib
            figure where the isocontours should be plotted.
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of step.fields[var].  This is
            useful to plot a masked or rescaled array.  Note that if
            conf.scaling.dimensional is True, this field will be scaled
            accordingly.
        extra: options that will be passed on to
            :func:`matplotlib.axes.Axes.contour`.
    """
    xmesh, ymesh, fld, _ = get_meshes_fld(step, var)

    if field is not None:
        fld = field

    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)
    extra_opts: Dict[str, Any] = dict(linewidths=1)
    if 'cmap' not in extra and conf.field.isocolors:
        extra_opts['colors'] = conf.field.isocolors.split(',')
    elif 'colors' not in extra:
        extra_opts['cmap'] = conf.field.cmap.get(var)
    if conf.plot.isolines:
        extra_opts['levels'] = sorted(conf.plot.isolines)
    extra_opts.update(extra)
    axis.contour(xmesh, ymesh, fld, **extra_opts)


def plot_vec(axis: Axes, step: Step, var: str) -> None:
    """Plot vector field.

    Args:
        axis: the :class:`matplotlib.axes.Axes` of an existing matplotlib
            figure where the vector field should be plotted.
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the vector field name.
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


def _findminmax(
    sdat: StagyyData, sovs: Iterable[str]
) -> Dict[str, Tuple[float, float]]:
    """Find min and max values of several fields."""
    minmax: Dict[str, Tuple[float, float]] = {}
    for step in sdat.walk.filter(snap=True):
        for var in sovs:
            if var in step.fields:
                field, meta = step.fields[var]
                field, _ = sdat.scale(field, meta.dim)
                if var in minmax:
                    minmax[var] = (min(minmax[var][0], np.nanmin(field)),
                                   max(minmax[var][1], np.nanmax(field)))
                else:
                    minmax[var] = np.nanmin(field), np.nanmax(field)
    return minmax


def cmd() -> None:
    """Implementation of field subcommand.

    Other Parameters:
        conf.field
        conf.core
    """
    sdat = StagyyData()
    lovs = _helpers.list_of_vars(conf.field.plot)
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
                                     figsize=(6 * len(vfig), 6))
            for axis, var in zip(axes[0], vfig):
                if var[0] not in step.fields:
                    print(f"{var[0]!r} field on snap {step.isnap} not found")
                    continue
                opts: Dict[str, Any] = {}
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
                time = _helpers.scilabel(time)
                axes[0, 0].text(0.02, 1.02, f'$t={time}$ {unit}',
                                transform=axes[0, 0].transAxes)
            oname = '_'.join(chain.from_iterable(vfig))
            plt.tight_layout(w_pad=3)
            _helpers.saveplot(fig, oname, step.isnap)
