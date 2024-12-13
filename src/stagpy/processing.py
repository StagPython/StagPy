"""Computations of phyvars that are not output by StagYY.

Time series are returned along with the time at which the variables are
evaluated. Radial profiles are returned along with the radial positions at
which the variables are evaluated.
"""

from __future__ import annotations

import typing

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .datatypes import Field, Rprof, Tseries, Varf, Varr, Vart
from .error import NotAvailableError

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .stagyydata import StagyyData
    from .step import Step


def dtime(sdat: StagyyData) -> Tseries:
    """Time increment.

    Compute dt as a function of time.

    Args:
        sdat: a `StagyyData` instance.

    Returns:
        dt and time arrays.
    """
    time = sdat.tseries.time
    return Tseries(np.diff(time), time[:-1], Vart("Time increment dt", "dt", "s"))


def dt_dt(sdat: StagyyData) -> Tseries:
    """Time derivative of temperature.

    Compute dT/dt as a function of time using an explicit Euler scheme.

    Args:
        sdat: a `StagyyData` instance.

    Returns:
        derivative of temperature and time arrays.
    """
    series = sdat.tseries["Tmean"]
    temp = series.values
    time = series.time
    dtdt = np.diff(temp) / np.diff(time)
    return Tseries(dtdt, time[:-1], Vart("Derivative of temperature", r"dT/dt", "K/s"))


def ebalance(sdat: StagyyData) -> Tseries:
    """Energy balance.

    Compute Nu_t - Nu_b + V*dT/dt as a function of time using an explicit
    Euler scheme. This should be zero if energy is conserved.

    Args:
        sdat: a `StagyyData` instance.

    Returns:
        energy balance and time arrays.
    """
    rbot, rtop = sdat.steps[-1].rprofs.bounds
    if rbot != 0:  # spherical
        coefsurf = (rtop / rbot) ** 2
        volume = rbot * ((rtop / rbot) ** 3 - 1) / 3
    else:
        coefsurf = 1.0
        volume = 1.0
    dtdt = dt_dt(sdat)
    ftop = sdat.tseries["ftop"].values * coefsurf
    fbot = sdat.tseries["fbot"].values
    radio = sdat.tseries["H_int"].values
    ebal = ftop[1:] - fbot[1:] + volume * (dtdt.values - radio[1:])
    return Tseries(ebal, dtdt.time, Vart("Energy balance", r"$\mathrm{Nu}$", "1"))


def mobility(sdat: StagyyData) -> Tseries:
    """Plates mobility.

    Compute the ratio vsurf / vrms.

    Args:
        sdat: a `StagyyData` instance.

    Returns:
        mobility and time arrays.
    """
    time = []
    mob = []
    for step in sdat.steps.filter(rprofs=True):
        time.append(step.timeinfo["t"])
        mob.append(step.rprofs["vrms"].values[-1] / step.timeinfo["vrms"])
    return Tseries(
        np.array(mob), np.array(time), Vart("Plates mobility", "Mobility", "1")
    )


def delta_r(step: Step) -> Rprof:
    """Cell thicknesses.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the thickness of the cells and radius.
    """
    edges = step.rprofs.walls
    meta = Varr("Cell thickness", "dr", "m")
    return Rprof(np.diff(edges), step.rprofs.centers, meta)


def _scale_prof(
    step: Step, rprof: NDArray[np.float64], rad: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Scale profile to take sphericity into account."""
    rbot, rtop = step.rprofs.bounds
    if rbot == 0:  # not spherical
        return rprof
    if rad is None:
        rad = step.rprofs.centers
    return rprof * (2 * rad / (rtop + rbot)) ** 2


def diff_prof(step: Step) -> Rprof:
    """Diffusion flux.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the diffusion and radius.
    """
    rbot, rtop = step.rprofs.bounds
    rad = step.rprofs.centers
    tprof = step.rprofs["Tmean"].values
    diff = (tprof[:-1] - tprof[1:]) / (rad[1:] - rad[:-1])
    # assume tbot = 1
    diff = np.insert(diff, 0, (1 - tprof[0]) / (rad[0] - rbot))
    # assume ttop = 0
    diff = np.append(diff, tprof[-1] / (rtop - rad[-1]))
    meta = Varr("Diffusion", "Heat flux", "W/m2")
    return Rprof(diff, step.rprofs.walls, meta)


def diffs_prof(step: Step) -> Rprof:
    """Scaled diffusion flux.

    This computation takes sphericity into account if necessary.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the diffusion and radius.
    """
    rpf = diff_prof(step)
    meta = Varr("Scaled diffusion", "Heat flux", "W/m2")
    return Rprof(_scale_prof(step, rpf.values, rpf.rad), rpf.rad, meta)


def advts_prof(step: Step) -> Rprof:
    """Scaled advection flux.

    This computation takes sphericity into account if necessary.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the scaled advection and radius.
    """
    return Rprof(
        _scale_prof(step, step.rprofs["advtot"].values),
        step.rprofs.centers,
        Varr("Scaled advection", "Heat flux", "W/m2"),
    )


def advds_prof(step: Step) -> Rprof:
    """Scaled downward advection flux.

    This computation takes sphericity into account if necessary.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the scaled downward advection and radius.
    """
    return Rprof(
        _scale_prof(step, step.rprofs["advdesc"].values),
        step.rprofs.centers,
        Varr("Scaled downward advection", "Heat flux", "W/m2"),
    )


def advas_prof(step: Step) -> Rprof:
    """Scaled upward advection flux.

    This computation takes sphericity into account if necessary.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the scaled upward advection and radius.
    """
    return Rprof(
        _scale_prof(step, step.rprofs["advasc"].values),
        step.rprofs.centers,
        Varr("Scaled upward advection", "Heat flux", "W/m2"),
    )


def energy_prof(step: Step) -> Rprof:
    """Total heat flux.

    This computation takes sphericity into account if necessary.

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the energy flux and radius.
    """
    diff_p = diffs_prof(step)
    adv_p = advts_prof(step)
    return Rprof(
        diff_p.values + np.append(adv_p.values, 0),
        diff_p.rad,
        Varr("Total heat flux", "Heat flux", "W/m2"),
    )


def stream_function(step: Step) -> Field:
    """Stream function (2D).

    Args:
        step: a `Step` of a `StagyyData` instance.

    Returns:
        the stream function field.
    """
    if step.geom.twod_yz:
        x_coord = step.geom.y_walls
        v_x = step.fields["v2"].values[0, :, :, 0]
        v_z = step.fields["v3"].values[0, :, :, 0]
        shape = (1, v_x.shape[0], v_x.shape[1], 1)
    elif step.geom.twod_xz and step.geom.cartesian:
        x_coord = step.geom.x_walls
        v_x = step.fields["v1"].values[:, 0, :, 0]
        v_z = step.fields["v3"].values[:, 0, :, 0]
        shape = (v_x.shape[0], 1, v_x.shape[1], 1)
    else:
        raise NotAvailableError(
            "Stream function only implemented in 2D cartesian and spherical annulus"
        )
    psi = np.zeros_like(v_x)
    if step.geom.spherical:  # YZ annulus
        # positions
        r_nc = step.rprofs.centers  # numerical centers
        r_pc = step.geom.r_centers  # physical centers
        r_nw = step.rprofs.walls[:2]  # numerical walls of first cell
        # vz at center of bottom cells
        vz0 = ((r_nw[1] - r_nc[0]) * v_z[:, 0] + (r_nc[0] - r_nw[0]) * v_z[:, 1]) / (
            r_nw[1] - r_nw[0]
        )
        psi[1:, 0] = -cumulative_trapezoid(r_pc[0] ** 2 * vz0, x=x_coord)
        # vx at center
        vxc = (v_x + np.roll(v_x, -1, axis=0)) / 2
        for i_x in range(len(x_coord)):
            psi[i_x, 1:] = psi[i_x, 0] + cumulative_trapezoid(r_pc * vxc[i_x], x=r_nc)
    else:  # assume cartesian geometry
        z_nc = step.geom.r_centers
        z_nw = step.rprofs.walls[:2]
        vz0 = ((z_nw[1] - z_nc[0]) * v_z[:, 0] + (z_nc[0] - z_nw[0]) * v_z[:, 1]) / (
            z_nw[1] - z_nw[0]
        )
        psi[1:, 0] = -cumulative_trapezoid(vz0, x=x_coord)
        # vx at center
        vxc = (v_x + np.roll(v_x, -1, axis=0)) / 2
        for i_x in range(len(x_coord)):
            psi[i_x, 1:] = psi[i_x, 0] + cumulative_trapezoid(vxc[i_x], x=z_nc)
    if step.geom.twod_xz:
        psi = -psi
    psi = np.reshape(psi, shape)
    return Field(psi, Varf("Stream function", "m2/s"))
