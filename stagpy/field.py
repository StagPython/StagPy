"""plot fields"""

from inspect import getdoc
import numpy as np
from . import constants, misc
from .stagyydata import StagyyData


def get_meshes_fld(step, var):
    """Return 2D meshes and field"""
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


def set_of_vars(arg_plot):
    """List of vars

    Each var is a tuple, first component is a scalar field, second component is
    either:
    - a scalar field, isocontours are added to the plot
    - a vector field (e.g. 'v' for the (v1,v2,v3) vector), arrows are added to
      the plot
    """
    sovs = set(tuple((var + '+').split('+')[:2])
               for var in arg_plot.split(','))
    sovs.discard(('', ''))
    return sovs


def plot_scalar(args, step, var, scaling=None, **extra):
    """Plot scalar field"""
    plt = args.plt

    if var in constants.FIELD_VARS:
        meta = constants.FIELD_VARS[var]
    else:
        meta = constants.FIELD_VARS_EXTRA[var]
        meta = constants.Varf(getdoc(meta.description),
                              meta.shortname, meta.popts)
    if step.geom.threed:
        raise ValueError('plot_scalar only implemented for 2D fields')

    xmesh, ymesh, fld = get_meshes_fld(step, var)

    if scaling is not None:
        fld = np.copy(fld) * scaling

    fig, axis = plt.subplots(ncols=1)
    extra_opts = {'cmap': 'jet'}
    extra_opts.update(meta.popts)
    extra_opts.update({} if var != 'n'
                      else {'norm': args.mpl.colors.LogNorm()})
    extra_opts.update(extra)
    surf = axis.pcolormesh(xmesh, ymesh, fld, rasterized=not args.pdf,
                           shading='gouraud', **extra_opts)

    cbar = plt.colorbar(surf, shrink=args.shrinkcb)
    cbar.set_label(r'${}$'.format(meta.shortname))
    plt.axis('equal')
    plt.axis('off')
    return fig, axis, surf, cbar


def plot_iso(axis, step, var):
    """Plot isocontours"""
    xmesh, ymesh, fld = get_meshes_fld(step, var)
    axis.contour(xmesh, ymesh, fld, linewidths=1)


def field_cmd(args):
    """extract and plot field data"""
    sdat = StagyyData(args.path)
    sovs = set_of_vars(args.plot)
    for step in misc.steps_gen(sdat, args):
        for var in sovs:
            if step.fields[var[0]] is None:
                print("'{}' field on snap {} not found".format(var,
                                                               step.isnap))
                continue
            fig, axis, _, _ = plot_scalar(args, step, var[0])
            if var[1]:
                plot_iso(axis, step, var[1])
            oname = '{}_{}'.format(*var) if var[1] else var[0]
            fig.savefig(
                misc.out_name(args, oname).format(step.isnap) + '.pdf',
                format='PDF', bbox_inches='tight')
