"""Plot scalar and vector fields."""

from __future__ import annotations

import typing
from itertools import chain

import matplotlib as mpl
import matplotlib.patches as mpat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import _helpers, phyvars
from .config import Config
from .error import NotAvailableError
from .stagyydata import _sdat_from_conf

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, Optional, Union

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .datatypes import Varf
    from .stagyydata import StepsView
    from .step import Step


# The location is off for vertical velocities: they have an extra
# point in (x,y) instead of z in the output


def _threed_extract(
    conf: Config, step: Step, var: str, walls: bool = False
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Return suitable slices and coords for 3D fields."""
    is_vector = not valid_field_var(var)
    hwalls = is_vector or walls
    i_x: Optional[Union[int, slice]] = conf.field.ix
    i_y: Optional[Union[int, slice]] = conf.field.iy
    i_z: Optional[Union[int, slice]] = conf.field.iz
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
        varx, vary = var + "2", var + "3"
    elif i_y is not None:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        i_x = i_z = slice(None)
        varx, vary = var + "1", var + "3"
    else:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.y_walls if hwalls else step.geom.y_centers
        i_x = i_y = slice(None)
        varx, vary = var + "1", var + "2"
    data: Any
    if is_vector:
        data = (
            step.fields[varx].values[i_x, i_y, i_z, 0],
            step.fields[vary].values[i_x, i_y, i_z, 0],
        )
    else:
        data = step.fields[var].values[i_x, i_y, i_z, 0]
    return (xcoord, ycoord), data


def valid_field_var(var: str) -> bool:
    """Whether a field variable is defined.

    Args:
        var: the variable name to be checked.

    Returns:
        whether the var is defined in either [`FIELD`][stagpy.phyvars.FIELD] or
            [`FIELD_EXTRA`][stagpy.phyvars.FIELD_EXTRA].
    """
    return var in phyvars.FIELD or var in phyvars.FIELD_EXTRA


def get_meshes_fld(
    conf: Config, step: Step, var: str, walls: bool = False
) -> tuple[NDArray, NDArray, NDArray, Varf]:
    """Return scalar field along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        conf: configuration.
        step: a `Step` of a `StagyyData` instance.
        var: scalar field name.
        walls: consider the walls as the relevant mesh.

    Returns:
        xmesh: x position
        ymesh: y position
        fld: field values
        meta: metadata
    """
    fld = step.fields[var]
    hwalls = (
        walls
        or fld.values.shape[0] != step.geom.nxtot
        or fld.values.shape[1] != step.geom.nytot
    )
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), vals = _threed_extract(conf, step, var, walls)
    elif step.geom.twod_xz:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        vals = fld.values[:, 0, :, 0]
    else:  # twod_yz
        xcoord = step.geom.y_walls if hwalls else step.geom.y_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        if step.geom.curvilinear:
            pmesh, rmesh = np.meshgrid(xcoord, ycoord, indexing="ij")
            xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
        vals = fld.values[0, :, :, 0]
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xmesh, ymesh, vals, fld.meta


def get_meshes_vec(
    conf: Config, step: Step, var: str
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return vector field components along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        conf: configuration.
        step: a `Step` of a `StagyyData` instance.
        var: vector field name.

    Returns:
        xmesh: x position
        ymesh: y position
        fldx: x component
        fldy: y component
    """
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), (vec1, vec2) = _threed_extract(conf, step, var)
    elif step.geom.twod_xz:
        xcoord, ycoord = step.geom.x_walls, step.geom.z_centers
        vec1 = step.fields[var + "1"].values[:, 0, :, 0]
        vec2 = step.fields[var + "3"].values[:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xcoord, ycoord = step.geom.y_walls, step.geom.z_centers
        vec1 = step.fields[var + "2"].values[0, :, :, 0]
        vec2 = step.fields[var + "3"].values[0, :, :, 0]
    else:  # spherical yz
        pcoord = step.geom.p_walls
        pmesh = np.outer(pcoord, np.ones(step.geom.nrtot))
        vec_phi = step.fields[var + "2"].values[0, :, :, 0]
        vec_r = step.fields[var + "3"].values[0, :, :, 0]
        vec1 = vec_r * np.cos(pmesh) - vec_phi * np.sin(pmesh)
        vec2 = vec_phi * np.cos(pmesh) + vec_r * np.sin(pmesh)
        pcoord, rcoord = step.geom.p_walls, step.geom.r_centers
        pmesh, rmesh = np.meshgrid(pcoord, rcoord, indexing="ij")
        xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xmesh, ymesh, vec1, vec2


def plot_scalar(
    step: Step,
    var: str,
    field: Optional[NDArray] = None,
    axis: Optional[Axes] = None,
    conf: Optional[Config] = None,
    **extra: Any,
) -> tuple[Figure, Axes, QuadMesh, Optional[Colorbar]]:
    """Plot scalar field.

    Args:
        step: a `Step` of a `StagyyData` instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of `step.fields[var]`.  This is
            useful to plot a masked or rescaled array.
        axis: the `matplotlib.axes.Axes` object where the field should
            be plotted.  If set to None, a new figure with one subplot is
            created.
        conf: configuration.
        extra: options that will be passed on to `matplotlib.axes.Axes.pcolormesh`.

    Returns:
        fig: matplotlib figure
        axes: matplotlib axes
        surf: surface returned by `pcolormesh`
        cbar: colorbar
    """
    if conf is None:
        conf = Config.default_()
    if step.geom.threed and step.geom.spherical:
        raise NotAvailableError("plot_scalar not implemented for 3D spherical geometry")

    xmesh, ymesh, fld, meta = get_meshes_fld(conf, step, var, walls=True)
    # interpolate at cell centers, this should be abstracted by field objects
    # via an "at_cell_centers" method or similar
    if fld.shape[0] > max(step.geom.nxtot, step.geom.nytot):
        fld = (fld[:-1] + fld[1:]) / 2

    xmin, xmax = xmesh.min(), xmesh.max()
    ymin, ymax = ymesh.min(), ymesh.max()

    if field is not None:
        fld = field
    if conf.field.perturbation:
        fld = fld - np.mean(fld, axis=0)
    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)

    if axis is None:
        fig, axis = plt.subplots(ncols=1)
    else:
        fig = axis.get_figure()  # type: ignore

    if step.sdat.par.get("magma_oceans_in", "evolving_magma_oceans", False):
        rcmb = step.sdat.par.nml["geometry"]["r_cmb"]
        xmax = rcmb + 1
        ymax = xmax
        xmin = -xmax
        ymin = -ymax
        rsurf = xmax if step.timeinfo["thick_tmo"] > 0 else step.geom.r_walls[-3]
        cmb = mpat.Circle((0, 0), rcmb, color="dimgray", zorder=0)
        psurf = mpat.Circle((0, 0), rsurf, color="indianred", zorder=0)
        axis.add_patch(psurf)
        axis.add_patch(cmb)

    extra_opts = dict(
        cmap=conf.field.cmap.get(var),
        vmin=conf.plot.vmin,
        vmax=conf.plot.vmax,
        norm=mpl.colors.LogNorm() if var == "eta" else None,
        rasterized=conf.plot.raster,
        shading="flat",
    )
    extra_opts.update(extra)
    surf = axis.pcolormesh(xmesh, ymesh, fld, **extra_opts)  # type: ignore

    cbar = None
    if conf.field.colorbar:
        cax = make_axes_locatable(axis).append_axes("right", size="3%", pad=0.15)
        cbar = plt.colorbar(surf, cax=cax)
        cbar.set_label(meta.description + (" pert." if conf.field.perturbation else ""))
    if step.geom.spherical or conf.plot.ratio is None:
        axis.set_aspect("equal")
        axis.set_axis_off()
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())

    axis.set_adjustable("box")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    return fig, axis, surf, cbar


def plot_iso(
    axis: Axes,
    step: Step,
    var: str,
    field: Optional[NDArray] = None,
    conf: Optional[Config] = None,
    **extra: Any,
) -> None:
    """Plot isocontours of scalar field.

    Args:
        axis: the `matplotlib.axes.Axes` of an existing matplotlib
            figure where the isocontours should be plotted.
        step: a `Step` of a `StagyyData` instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of `step.fields[var]`.  This is
            useful to plot a masked or rescaled array.
        conf: configuration.
        extra: options that will be passed on to `Axes.contour`.
    """
    if conf is None:
        conf = Config.default_()
    xmesh, ymesh, fld, _ = get_meshes_fld(conf, step, var)

    if field is not None:
        fld = field

    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)
    extra_opts: dict[str, Any] = dict(linewidths=1)
    if "cmap" not in extra and conf.field.isocolors:
        extra_opts["colors"] = conf.field.isocolors
    elif "colors" not in extra:
        extra_opts["cmap"] = conf.field.cmap.get(var)
    if conf.plot.isolines:
        extra_opts["levels"] = sorted(conf.plot.isolines)
    extra_opts.update(extra)
    axis.contour(xmesh, ymesh, fld, **extra_opts)


def plot_vec(
    axis: Axes,
    step: Step,
    var: str,
    conf: Optional[Config] = None,
) -> None:
    """Plot vector field.

    Args:
        axis: the :class:`matplotlib.axes.Axes` of an existing matplotlib
            figure where the vector field should be plotted.
        step: a `Step` of a `StagyyData` instance.
        var: the vector field name.
        conf: configuration.
    """
    if conf is None:
        conf = Config.default_()
    xmesh, ymesh, vec1, vec2 = get_meshes_vec(conf, step, var)
    dipz = step.geom.nztot // 10
    if conf.field.shift:
        vec1 = np.roll(vec1, conf.field.shift, axis=0)
        vec2 = np.roll(vec2, conf.field.shift, axis=0)
    if step.geom.spherical or conf.plot.ratio is None:
        dipx = dipz
    else:
        dipx = step.geom.nytot if step.geom.twod_yz else step.geom.nxtot
        dipx = int(dipx // 10 * conf.plot.ratio) + 1
    axis.quiver(
        xmesh[::dipx, ::dipz],
        ymesh[::dipx, ::dipz],
        vec1[::dipx, ::dipz],
        vec2[::dipx, ::dipz],
        linewidths=1,
    )


def _findminmax(view: StepsView, sovs: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Find min and max values of several fields."""
    minmax: dict[str, tuple[float, float]] = {}
    for step in view.filter(snap=True):
        for var in sovs:
            if var in step.fields:
                vals = step.fields[var].values
                if var in minmax:
                    minmax[var] = (
                        min(minmax[var][0], np.nanmin(vals)),
                        max(minmax[var][1], np.nanmax(vals)),
                    )
                else:
                    minmax[var] = np.nanmin(vals), np.nanmax(vals)
    return minmax


def cmd(conf: Config) -> None:
    """Implementation of field subcommand."""
    sdat = _sdat_from_conf(conf.core)
    view = _helpers.walk(sdat, conf)
    # no more than two fields in a subplot
    lovs = [[slov[:2] for slov in plov] for plov in conf.field.plot]
    minmax = {}
    if conf.plot.cminmax:
        conf.plot.vmin = None
        conf.plot.vmax = None
        sovs = set(slov[0] for plov in lovs for slov in plov)
        minmax = _findminmax(view, sovs)
    for step in view.filter(snap=True):
        for vfig in lovs:
            fig, axes = plt.subplots(
                ncols=len(vfig), squeeze=False, figsize=(6 * len(vfig), 6)
            )
            for axis, var in zip(axes[0], vfig):
                if var[0] not in step.fields:
                    print(f"{var[0]!r} field on snap {step.isnap} not found")
                    continue
                opts: dict[str, Any] = {}
                if var[0] in minmax:
                    opts = dict(vmin=minmax[var[0]][0], vmax=minmax[var[0]][1])
                plot_scalar(step, var[0], axis=axis, conf=conf, **opts)
                if len(var) == 2:
                    if valid_field_var(var[1]):
                        plot_iso(axis, step, var[1], conf=conf)
                    elif valid_field_var(var[1] + "1"):
                        plot_vec(axis, step, var[1], conf=conf)
            if conf.field.timelabel:
                time = step.timeinfo["t"]
                time = _helpers.scilabel(time)
                axes[0, 0].text(
                    0.02, 1.02, f"$t={time}$", transform=axes[0, 0].transAxes
                )
            oname = "_".join(chain.from_iterable(vfig))
            plt.tight_layout(w_pad=3)
            _helpers.saveplot(conf, fig, oname, step.isnap)
