"""Plot scalar and vector fields."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import conf, misc, phyvars
from .error import NotAvailableError
from .stagyydata import StagyyData


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

    Only works properly in 2D geometry.

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
    if step.geom.twod_xz:
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

    Only works properly in 2D geometry.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): vector field name.
    Returns:
        tuple of :class:`numpy.array`: xmesh, ymesh, fldx, fldy
            2D arrays containing respectively the x position, y position, x
            component and y component of the requested vector field.
    """
    if step.geom.twod_xz:
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


def set_of_vars(arg_plot):
    """Build set of needed field variables.

    Each var is a tuple, first component is a scalar field, second component is
    either:

    - a scalar field, isocontours are added to the plot.
    - a vector field (e.g. 'v' for the (v1,v2,v3) vector), arrows are added to
      the plot.

    Args:
        arg_plot (str): string with variable names separated with
            ``,`` (figures), and ``+`` (same plot).
    Returns:
        set of str: set of needed field variables.
    """
    sovs = set(tuple((var + '+').split('+')[:2])
               for var in arg_plot.split(','))
    sovs.discard(('', ''))
    return sovs


def plot_scalar(step, var, scaling=None, **extra):
    """Plot scalar field.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): the scalar field name.
        scaling (float): if not None, the scalar field values are multiplied by
            this factor before plotting. This can be used e.g. to obtain
            dimensionful values.
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
        meta = phyvars.Varf(misc.baredoc(meta.description),
                            meta.shortname, meta.popts)
    if step.geom.threed:
        raise NotAvailableError('plot_scalar only implemented for 2D fields')

    xmesh, ymesh, fld = get_meshes_fld(step, var)

    if scaling is not None:
        fld = np.copy(fld) * scaling

    fig, axis = plt.subplots(ncols=1)
    extra_opts = {'cmap': 'RdBu_r'}
    extra_opts.update(meta.popts)
    extra_opts.update({} if var != 'eta'
                      else {'norm': mpl.colors.LogNorm()})
    extra_opts.update(extra)
    surf = axis.pcolormesh(xmesh, ymesh, fld, rasterized=conf.plot.raster,
                           shading='gouraud', **extra_opts)

    cbar = plt.colorbar(surf, shrink=conf.field.shrinkcb)
    cbar.set_label(r'${}$'.format(meta.shortname),
                   rotation='horizontal', va='center')
    if step.geom.spherical or conf.plot.ratio is None:
        plt.axis('equal')
        plt.axis('off')
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())
    axis.set_adjustable('box')
    axis.set_xlim(xmesh.min(), xmesh.max())
    axis.set_ylim(ymesh.min(), ymesh.max())
    return fig, axis, surf, cbar


def plot_iso(axis, step, var):
    """Plot isocontours of scalar field.

    Args:
        axis (:class:`matplotlib.axes.Axes`): the axis handler of an
            existing matplotlib figure where the isocontours should
            be plotted.
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): the scalar field name.
    """
    xmesh, ymesh, fld = get_meshes_fld(step, var)
    axis.contour(xmesh, ymesh, fld, linewidths=1)


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
    if step.geom.spherical or conf.plot.ratio is None:
        dipx = dipz
    else:
        dipx = step.geom.nytot if step.geom.twod_yz else step.geom.nxtot
        dipx = int(dipx // 10 * conf.plot.ratio) + 1
    axis.quiver(xmesh[::dipx, ::dipz], ymesh[::dipx, ::dipz],
                vec1[::dipx, ::dipz], vec2[::dipx, ::dipz],
                linewidths=1)


def cmd():
    """Implementation of field subcommand.

    Other Parameters:
        conf.field
        conf.core
    """
    sdat = StagyyData(conf.core.path)
    sovs = set_of_vars(conf.field.plot)
    for step in sdat.walk.filter(snap=True):
        for var in sovs:
            if step.fields[var[0]] is None:
                print("'{}' field on snap {} not found".format(var,
                                                               step.isnap))
                continue
            fig, axis, _, _ = plot_scalar(step, var[0])
            if valid_field_var(var[1]):
                plot_iso(axis, step, var[1])
            elif var[1]:
                plot_vec(axis, step, var[1])
            oname = '{}_{}'.format(*var) if var[1] else var[0]
            misc.saveplot(fig, oname, step.isnap)
