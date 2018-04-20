"""Computations of phyvars that are not output by StagYY.

Time series are returned along with the time at which the variables are
evaluated. Radial profiles are returned along with the radial positions at
which the variables are evaluated. These time and radial positions are set to
None is they are identical to the one at which StagYY output data are produced.
This is why some of the functions in this module return a tuple where the
second element is None.
"""

import numpy as np
from scipy import integrate

from . import misc
from .error import NotAvailableError


def dt_dt(sdat, tstart=None, tend=None):
    """Derivative of temperature.

    Compute dT/dt as a function of time using an explicit Euler scheme.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        tstart (float): time at which the computation should start. Use the
            beginning of the time series data if set to None.
        tend (float): time at which the computation should end. Use the
            end of the time series data if set to None.
    Returns:
        tuple of :class:`numpy.array`: derivative of temperature and time
        arrays.
    """
    tseries = sdat.tseries_between(tstart, tend)
    time = tseries['t'].values
    temp = tseries['Tmean'].values
    dtdt = (temp[1:] - temp[:-1]) / (time[1:] - time[:-1])
    return dtdt, time[:-1]


def ebalance(sdat, tstart=None, tend=None):
    """Energy balance.

    Compute Nu_t - Nu_b + V*dT/dt as a function of time using an explicit
    Euler scheme. This should be zero if energy is conserved.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        tstart (float): time at which the computation should start. Use the
            beginning of the time series data if set to None.
        tend (float): time at which the computation should end. Use the
            end of the time series data if set to None.
    Returns:
        tuple of :class:`numpy.array`: energy balance and time arrays.
    """
    tseries = sdat.tseries_between(tstart, tend)
    rbot, rtop = misc.get_rbounds(sdat.steps.last)
    if rbot != 0:  # spherical
        coefsurf = (rtop / rbot)**2
        volume = rbot * ((rtop / rbot)**3 - 1) / 3
    else:
        coefsurf = 1.
        volume = 1.
    dtdt, time = dt_dt(sdat, tstart, tend)
    ftop = tseries['ftop'].values * coefsurf
    fbot = tseries['fbot'].values
    ebal = ftop[:-1] - fbot[:-1] + volume * dtdt
    return ebal, time


def mobility(sdat, tstart=None, tend=None):
    """Plates mobility.

    Compute the ratio vsurf / vrms.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        tstart (float): time at which the computation should start. Use the
            beginning of the time series data if set to None.
        tend (float): time at which the computation should end. Use the
            end of the time series data if set to None.
    Returns:
        tuple of :class:`numpy.array`: mobility and time arrays.
    """
    tseries = sdat.tseries_between(tstart, tend)
    steps = sdat.steps[tseries.index[0]:tseries.index[-1]]
    time = []
    mob = []
    for step in steps.filter(rprof=True):
        time.append(step.timeinfo['t'])
        mob.append(step.rprof.iloc[-1].loc['vrms'] / step.timeinfo['vrms'])
    return np.array(mob), np.array(time)


def r_edges(step):
    """Cell border.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the position of the bottom and top walls
        of the cells. The two elements of the tuple are identical.
    """
    rbot, rtop = misc.get_rbounds(step)
    centers = step.rprof.loc[:, 'r'].values + rbot
    # assume walls are mid-way between T-nodes
    # could be T-nodes at center between walls
    edges = (centers[:-1] + centers[1:]) / 2
    edges = np.insert(edges, 0, rbot)
    edges = np.append(edges, rtop)
    return edges, edges


def delta_r(step):
    """Cells thickness.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array` and None: the thickness of the cells.
        The second element of the tuple is None.
    """
    edges, _ = r_edges(step)
    return (edges[1:] - edges[:-1]), None


def _scale_prof(step, rprof, rad=None):
    """Scale profile to take sphericity into account"""
    rbot, rtop = misc.get_rbounds(step)
    if rbot == 0:  # not spherical
        return rprof
    if rad is None:
        rad = step.rprof['r'].values + rbot
    return rprof * (rad / rtop)**2


def diff_prof(step):
    """Diffusion.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the diffusion and the radial position
        at which it is evaluated.
    """
    rbot, rtop = misc.get_rbounds(step)
    rad = step.rprof['r'].values + rbot
    tprof = step.rprof['Tmean'].values
    diff = (tprof[:-1] - tprof[1:]) / (rad[1:] - rad[:-1])
    # assume tbot = 1
    diff = np.insert(diff, 0, (1 - tprof[0]) / (rad[0] - rbot))
    # assume ttop = 0
    diff = np.append(diff, tprof[-1] / (rtop - rad[-1]))
    # actually computed at r_edges...
    return diff, np.append(rad, rtop)


def diffs_prof(step):
    """Scaled diffusion.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the diffusion and the radial position
        at which it is evaluated.
    """
    diff, rad = diff_prof(step)
    return _scale_prof(step, diff, rad), rad


def advts_prof(step):
    """Scaled advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array` and None: the scaled advection.
        The second element of the tuple is None.
    """
    return _scale_prof(step, step.rprof['advtot']), None


def advds_prof(step):
    """Scaled downward advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array` and None: the scaled downward advection.
        The second element of the tuple is None.
    """
    return _scale_prof(step, step.rprof['advdesc']), None


def advas_prof(step):
    """Scaled upward advection.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array` and None: the scaled upward advection.
        The second element of the tuple is None.
    """
    return _scale_prof(step, step.rprof['advasc']), None


def energy_prof(step):
    """Energy flux.

    This computation takes sphericity into account if necessary.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the energy flux and the radial position
        at which it is evaluated.
    """
    diff, rad = diffs_prof(step)
    adv, _ = advts_prof(step)
    return (diff + np.append(adv, 0)), rad


def init_c_overturn(step):
    """Initial concentration.

    This compute the resulting composition profile if fractional
    crystallization of a SMO is assumed.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the composition and the radial position
        at which it is evaluated.
    """
    rbot, rtop = misc.get_rbounds(step)
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
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of :class:`numpy.array`: the composition and the radial position
        at which it is evaluated.
    """
    rbot, rtop = misc.get_rbounds(step)
    cinit, rad = init_c_overturn(step)
    radf = (rtop**3 + rbot**3 - rad**3)**(1 / 3)
    return cinit, radf


def stream_function(step):
    """Stream function.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        :class:`numpy.array`: the stream function field, with four dimensions:
        x-direction, y-direction, z-direction and block.
    """
    if step.geom.twod_yz:
        x_coord = step.geom.y_coord
        v_x = step.fields['v2'][0, :, :, 0]
        v_z = step.fields['v3'][0, :, :, 0]
        shape = (1, v_x.shape[0], v_x.shape[1], 1)
    elif step.geom.twod_xz and step.geom.cartesian:
        x_coord = step.geom.x_coord
        v_x = step.fields['v1'][:, 0, :, 0]
        v_z = step.fields['v3'][:, 0, :, 0]
        shape = (v_x.shape[0], 1, v_x.shape[1], 1)
    else:
        raise NotAvailableError('Stream function only implemented in '
                                '2D cartesian and spherical annulus')
    psi = np.zeros_like(v_x)
    if step.geom.spherical:  # YZ annulus
        r_coord = step.geom.r_coord + step.geom.rcmb
        psi[0, :] = integrate.cumtrapz(r_coord * v_x[0, :],
                                       x=r_coord,
                                       initial=0)
        for i_z, r_pos in enumerate(r_coord):
            psi[:, i_z] = psi[0, i_z] - \
                integrate.cumtrapz(r_pos**2 * v_z[:, i_z],
                                   x=x_coord, initial=0)
    else:  # assume cartesian geometry
        psi[0, :] = integrate.cumtrapz(v_x[0, :],
                                       x=step.geom.z_coord,
                                       initial=0)
        for i_z in range(step.geom.nztot):
            psi[:, i_z] = psi[0, i_z] - integrate.cumtrapz(v_z[:, i_z],
                                                           x=x_coord,
                                                           initial=0)
    if step.geom.twod_xz:
        psi = - psi
    psi = np.reshape(psi, shape)
    return psi
