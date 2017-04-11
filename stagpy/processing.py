"""Various computations"""

import numpy as np
from . import misc


def dt_dt(sdat, tstart=0., tend=None):
    """Derivative of temperature"""
    tseries = sdat.tseries_between(tstart, tend)
    time = tseries['t'].values
    temp = tseries['Tmean'].values
    dtdt = (temp[2:] - temp[:-2]) / (time[2:] - time[:-2])
    return dtdt, time[1:-1]


def ebalance(sdat, tstart=0., tend=None):
    """Energy balance"""
    tseries = sdat.tseries_between(tstart, tend)
    rbot, rtop = misc.get_rbounds(sdat.steps.last)
    if rbot != 0:  # spherical
        coefsurf = (rtop / rbot)**2
        volume = rbot * (1 - (rtop / rbot)**3) / 3
    else:
        coefsurf = 1.
        volume = 1.
    dtdt, time = dt_dt(sdat, tstart, tend)
    ftop = tseries['ftop'].values * coefsurf
    fbot = tseries['fbot'].values
    ebal = ftop[1:-1] - fbot[1:-1] - volume * dtdt
    return ebal, time


def r_edges(step):
    """Cell border"""
    rbot, rtop = misc.get_rbounds(step)
    centers = step.rprof.loc[:, 'r'].values + rbot
    # assume walls are mid-way between T-nodes
    # could be T-nodes at center between walls
    edges = (centers[:-1] + centers[1:]) / 2
    edges = np.insert(edges, 0, rbot)
    edges = np.append(edges, rtop)
    return edges, edges


def delta_r(step):
    """Cells thickness"""
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
    """Diffusion"""
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
    """Scaled diffusion"""
    diff, rad = diff_prof(step)
    return _scale_prof(step, diff, rad), rad


def advts_prof(step):
    """Scaled advection"""
    return _scale_prof(step, step.rprof['advtot']), None


def advds_prof(step):
    """Scaled downward advection"""
    return _scale_prof(step, step.rprof['advdesc']), None


def advas_prof(step):
    """Scaled upward advection"""
    return _scale_prof(step, step.rprof['advasc']), None


def energy_prof(step):
    """Energy flux"""
    diff, rad = diffs_prof(step)
    adv, _ = advts_prof(step)
    return (diff + np.append(adv, 0)), rad


def init_c_overturn(step):
    """Initial concentration"""
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
        else:
            return xieut

    rad = np.linspace(rbot, rtop, 500)
    initprof = np.vectorize(initprof)
    return initprof(rad), rad


def c_overturned(step):
    """Theoretical overturned concentration"""
    rbot, rtop = misc.get_rbounds(step)
    cinit, rad = init_c_overturn(step)
    radf = (rtop**3 + rbot**3 - rad**3)**(1 / 3)
    return cinit, radf
