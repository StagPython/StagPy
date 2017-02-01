"""plot fields"""

import numpy as np
from . import constants, misc
from .stagyydata import StagyyData


def plot_scalar(args, step, var, scaling=None, **extra):
    """var: one of the key of constants.FIELD_VAR_LIST"""
    plt = args.plt

    fld = step.fields[var]
    if step.geom.threed:
        raise ValueError('plot_scalar only implemented for 2D fields')

    if step.geom.twod_xz:
        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]
        fld = fld[:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]
        fld = fld[0, :, :, 0]
    else:  # spherical yz
        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]
        fld = fld[0, :, :, 0]

    if scaling is not None:
        fld = np.copy(fld) * scaling

    fig, axis = plt.subplots(ncols=1)
    extra_opts = {'cmap': 'jet'}
    extra_opts.update(constants.FIELD_VAR_LIST[var].pcolor_opts)
    extra_opts.update({} if var != 'n'
                      else {'norm': args.mpl.colors.LogNorm()})
    extra_opts.update(extra)
    surf = axis.pcolormesh(xmesh, ymesh, fld, rasterized=not args.pdf,
                           shading='gouraud', **extra_opts)

    cbar = plt.colorbar(surf, shrink=args.shrinkcb)
    cbar.set_label(constants.FIELD_VAR_LIST[var].name)
    plt.axis('equal')
    plt.axis('off')
    return fig, axis, surf, cbar


def plot_stream(args, fig, axis, component1, component2):
    """use of streamplot to plot stream lines

    only works in cartesian with regular grids
    """
    x_1, v_1 = component1
    x_2, v_2 = component2
    v_tot = np.sqrt(v_1**2 + v_2**2)
    lwd = 2 * v_tot / v_tot.max()
    args.plt.figure(fig.number)
    axis.streamplot(x_1, x_2, v_1, v_2, density=0.8, color='k', linewidth=lwd)


def field_cmd(args):
    """extract and plot field data"""
    sdat = StagyyData(args.path)
    for step in misc.steps_gen(sdat, args):
        for var, meta in constants.FIELD_VAR_LIST.items():
            if misc.get_arg(args, meta.arg):
                if step.fields[var] is None:
                    print("'{}' field on snap {} not found".format(var,
                                                                   step.isnap))
                    continue
                fig, _, _, _ = plot_scalar(args, step, var)
                args.plt.figure(fig.number)
                args.plt.tight_layout()
                args.plt.savefig(
                    misc.out_name(args, var).format(step.isnap) + '.pdf',
                    format='PDF')
                args.plt.close(fig)
