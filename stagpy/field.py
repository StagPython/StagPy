"""plot fields"""

import numpy as np
from . import constants, misc
from .stagyydata import StagyyData


def plot_scalar(args, step, var):
    """var: one of the key of constants.FIELD_VAR_LIST"""
    plt = args.plt
    if var == 'l':
        raise ValueError('Stream function plotting unavailable')

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

    fig, axis = plt.subplots(ncols=1)
    if var == 'n':  # viscosity
        surf = axis.pcolormesh(xmesh, ymesh, fld,
                               norm=args.mpl.colors.LogNorm(),
                               cmap='jet_r',
                               rasterized=not args.pdf,
                               shading='gouraud')
    elif var == 'd':  # density
        surf = axis.pcolormesh(xmesh, ymesh, fld, cmap='bwr_r',
                               vmin=0.96, vmax=1.04,
                               rasterized=not args.pdf,
                               shading='gouraud')
    elif var == 's':  # second invariant of stress
        surf = axis.pcolormesh(xmesh, ymesh, fld, cmap='gnuplot2_r',
                               vmin=500, vmax=20000,
                               rasterized=not args.pdf,
                               shading='gouraud')
    elif var == 'e':  # strain rate
        surf = axis.pcolormesh(xmesh, ymesh, fld, cmap='Reds',
                               vmin=500, vmax=20000,
                               rasterized=not args.pdf,
                               shading='gouraud')
    elif var == 'r':  # topography
        plt.plot(stgdat.ph_coord[:-1], fld[:-1, 1] *
                 args.par_nml['geometry']['d_dimensional'] / 1000., '-')
        plt.xlim([np.amin(stgdat.ph_coord), np.amax(stgdat.ph_coord)])
        plt.xlabel('Distance')
        plt.ylabel('Topography [km]')
    elif var == 'a':  # age
        surf = axis.pcolormesh(xmesh, ymesh, fld, cmap='jet',
                               vmin=0.0,
                               rasterized=not args.pdf,
                               shading='gouraud')
    else:
        surf = axis.pcolormesh(xmesh, ymesh, fld, cmap='jet',
                               rasterized=not args.pdf,
                               shading='gouraud')

        if var != 'r':
            cbar = plt.colorbar(surf, shrink=args.shrinkcb)
            cbar.set_label(constants.FIELD_VAR_LIST[var].name)
            plt.axis('equal')
            plt.axis('off')
    return fig, axis, surf


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
    sdat = StagyyData(args)
    for step in misc.steps_gen(sdat, args):
        for var, meta in constants.FIELD_VAR_LIST.items():
            if misc.get_arg(args, meta.arg):
                if step.fields[var] is None:
                    print("'{}' field on snap {} not found".format(var,
                                                                   step.isnap))
                    continue
                fig, _, _ = plot_scalar(args, step, var)
                args.plt.figure(fig.number)
                args.plt.tight_layout()
                args.plt.savefig(
                    misc.out_name(args, var).format(step.isnap) + '.pdf',
                    format='PDF')
                args.plt.close(fig)
