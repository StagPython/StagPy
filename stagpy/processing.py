"""Computations of phyvars that are not output by StagYY.

Time series are returned along with the time at which the variables are
evaluated. Radial profiles are returned along with the radial positions at
which the variables are evaluated.
"""

import numpy as np
from scipy import integrate

from .error import NotAvailableError


def dtime(sdat):
    """Time increment dt.

    Compute dt as a function of time.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: dt and time arrays.
    """
    time = sdat.tseries.time
    return time[1:] - time[:-1], time[:-1]


def dt_dt(sdat):
    """Derivative of temperature.

    Compute dT/dt as a function of time using an explicit Euler scheme.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: derivative of temperature and time
        arrays.
    """
    temp, time, _ = sdat.tseries['Tmean']
    dtdt = (temp[1:] - temp[:-1]) / (time[1:] - time[:-1])
    return dtdt, time[:-1]


def ebalance(sdat):
    """Energy balance.

    Compute Nu_t - Nu_b + V*dT/dt as a function of time using an explicit
    Euler scheme. This should be zero if energy is conserved.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: energy balance and time arrays.
    """
    rbot, rtop = sdat.steps[-1].rprofs.bounds
    if rbot != 0:  # spherical
        coefsurf = (rtop / rbot)**2
        volume = rbot * ((rtop / rbot)**3 - 1) / 3
    else:
        coefsurf = 1.
        volume = 1.
    dtdt, time = dt_dt(sdat)
    ftop = sdat.tseries['ftop'].values * coefsurf
    fbot = sdat.tseries['fbot'].values
    radio = sdat.tseries['H_int'].values
    ebal = ftop[1:] - fbot[1:] + volume * (dtdt - radio[1:])
    return ebal, time


def mobility(sdat):
    """Plates mobility.

    Compute the ratio vsurf / vrms.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: mobility and time arrays.
    """
    time = []
    mob = []
    for step in sdat.steps.filter(rprofs=True):
        time.append(step.timeinfo['t'])
        mob.append(step.rprofs['vrms'].values[-1] / step.timeinfo['vrms'])
    return np.array(mob), np.array(time)


def delta_r(step):
    """Cells thickness.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the thickness of the cells and radius.
    """
    edges = step.rprofs.walls
    return (edges[1:] - edges[:-1]), step.rprofs.centers


def _scale_prof(step, rprof, rad=None):
    """Scale profile to take sphericity into account."""
    rbot, rtop = step.rprofs.bounds
    if rbot == 0:  # not spherical
        return rprof
    if rad is None:
        rad = step.rprofs.centers
    return rprof * (2 * rad / (rtop + rbot))**2


def diff_prof(step):
    """Diffusion.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the diffusion and radius.
    """
    rbot, rtop = step.rprofs.bounds
    rad = step.rprofs.centers
    tprof = step.rprofs['Tmean'].values
    diff = (tprof[:-1] - tprof[1:]) / (rad[1:] - rad[:-1])
    # assume tbot = 1
    diff = np.insert(diff, 0, (1 - tprof[0]) / (rad[0] - rbot))
    # assume ttop = 0
    diff = np.append(diff, tprof[-1] / (rtop - rad[-1]))
    return diff, step.rprofs.walls


def diffs_prof(step):
    """Scaled diffusion.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the diffusion and radius.
    """
    diff, rad = diff_prof(step)
    return _scale_prof(step, diff, rad), rad


def advts_prof(step):
    """Scaled advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the scaled advection and radius.
    """
    return _scale_prof(step, step.rprofs['advtot'].values), step.rprofs.centers


def advds_prof(step):
    """Scaled downward advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the scaled downward advection and
            radius.
    """
    return (_scale_prof(step, step.rprofs['advdesc'].values),
            step.rprofs.centers)


def advas_prof(step):
    """Scaled upward advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the scaled upward advection and radius.
    """
    return _scale_prof(step, step.rprofs['advasc'].values), step.rprofs.centers


def energy_prof(step):
    """Energy flux.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the energy flux and radius.
    """
    diff, rad = diffs_prof(step)
    adv, _ = advts_prof(step)
    return (diff + np.append(adv, 0)), rad


def advth(step):
    """Theoretical advection.

    This compute the theoretical profile of total advection as function of
    radius.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the theoretical advection and radius.
    """
    rbot, rtop = step.rprofs.bounds
    rmean = 0.5 * (rbot + rtop)
    rad = step.rprofs.centers
    radio = step.timeinfo['H_int']
    if rbot != 0:  # spherical
        th_adv = -(rtop**3 - rad**3) / rmean**2 / 3
    else:
        th_adv = rad - rtop
    th_adv *= radio
    th_adv += step.timeinfo['Nutop']
    return th_adv, rad


def init_c_overturn(step):
    """Initial concentration.

    This compute the resulting composition profile if fractional
    crystallization of a SMO is assumed.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the composition and radius.
    """
    rbot, rtop = step.rprofs.bounds
    xieut = step.sdat.par['tracersin']['fe_eut']
    k_fe = step.sdat.par['tracersin']['k_fe']
    xi0l = step.sdat.par['tracersin']['fe_cont']
    xi0s = k_fe * xi0l
    xired = xi0l / xieut
    rsup = (rtop**3 - xired**(1 / (1 - k_fe)) *
            (rtop**3 - rbot**3))**(1 / 3)

    def initprof(rpos):
        """Theoretical initial profile."""
        if rpos < rsup:
            return xi0s * ((rtop**3 - rbot**3) /
                           (rtop**3 - rpos**3))**(1 - k_fe)
        return xieut

    rad = np.linspace(rbot, rtop, 500)
    initprof = np.vectorize(initprof)
    return initprof(rad), rad


def c_overturned(step):
    """Theoretical overturned concentration.

    This compute the resulting composition profile if fractional
    crystallization of a SMO is assumed and then a purely radial
    overturn happens.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        tuple of :class:`numpy.array`: the composition and radius.
    """
    rbot, rtop = step.rprofs.bounds
    cinit, rad = init_c_overturn(step)
    radf = (rtop**3 + rbot**3 - rad**3)**(1 / 3)
    return cinit, radf


def stream_function(step):
    """Stream function.

    Args:
        step (:class:`~stagpy._step.Step`): a step of a StagyyData instance.
    Returns:
        :class:`numpy.array`: the stream function field, with four dimensions:
        x-direction, y-direction, z-direction and block.
    """
    if step.geom.twod_yz:
        x_coord = step.geom.y_walls
        v_x = step.fields['v2'][0, :, :, 0]
        v_z = step.fields['v3'][0, :, :, 0]
        shape = (1, v_x.shape[0], v_x.shape[1], 1)
    elif step.geom.twod_xz and step.geom.cartesian:
        x_coord = step.geom.x_walls
        v_x = step.fields['v1'][:, 0, :, 0]
        v_z = step.fields['v3'][:, 0, :, 0]
        shape = (v_x.shape[0], 1, v_x.shape[1], 1)
    else:
        raise NotAvailableError('Stream function only implemented in '
                                '2D cartesian and spherical annulus')
    psi = np.zeros_like(v_x)
    if step.geom.spherical:  # YZ annulus
        # positions
        r_nc = step.rprofs.centers  # numerical centers
        r_pc = step.geom.r_centers  # physical centers
        r_nw = step.rprofs.walls[:2]  # numerical walls of first cell
        # vz at center of bottom cells
        vz0 = ((r_nw[1] - r_nc[0]) * v_z[:, 0] +
               (r_nc[0] - r_nw[0]) * v_z[:, 1]) / (r_nw[1] - r_nw[0])
        psi[1:, 0] = -integrate.cumtrapz(r_pc[0]**2 * vz0, x=x_coord)
        # vx at center
        vxc = (v_x + np.roll(v_x, -1, axis=0)) / 2
        for i_x in range(len(x_coord)):
            psi[i_x, 1:] = psi[i_x, 0] + \
                integrate.cumtrapz(r_pc * vxc[i_x], x=r_nc)
    else:  # assume cartesian geometry
        z_nc = step.geom.r_centers
        z_nw = step.rprofs.walls[:2]
        vz0 = ((z_nw[1] - z_nc[0]) * v_z[:, 0] +
               (z_nc[0] - z_nw[0]) * v_z[:, 1]) / (z_nw[1] - z_nw[0])
        psi[1:, 0] = -integrate.cumtrapz(vz0, x=x_coord)
        # vx at center
        vxc = (v_x + np.roll(v_x, -1, axis=0)) / 2
        for i_x in range(len(x_coord)):
            psi[i_x, 1:] = psi[i_x, 0] + \
                integrate.cumtrapz(vxc[i_x], x=z_nc)
    if step.geom.twod_xz:
        psi = - psi
    psi = np.reshape(psi, shape)
    return psi
