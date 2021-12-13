"""Plot reference state profiles."""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import conf, _helpers
from .phyvars import REFSTATE
from .stagyydata import StagyyData


def plot_ref(sdat: StagyyData, var: str) -> None:
    """Plot one reference state.

    Args:
        sdat: a :class:`~stagpy.stagyydata.StagyyData` instance.
        var: refstate variable, a key of :data:`~stagpy.phyvars.REFSTATE`.
    """
    fig, axis = plt.subplots()
    adbts = sdat.refstate.adiabats
    if len(adbts) > 2:
        for iad, adia in enumerate(adbts[:-1], 1):
            axis.plot(adia[var], adia['z'],
                      conf.refstate.style,
                      label=f'System {iad}')
    axis.plot(adbts[-1][var], adbts[-1]['z'],
              conf.refstate.style, color='k',
              label='Combined profile')
    if var == 'Tcond':
        axis.set_xscale('log')
    axis.set_xlabel(REFSTATE[var].description)
    axis.set_ylabel('z Position')
    if len(adbts) > 2:
        axis.legend()
    _helpers.saveplot(fig, f'refstate_{var}')


def cmd() -> None:
    """Implementation of refstate subcommand.

    Other Parameters:
        conf.core
        conf.plot
    """
    sdat = StagyyData()

    lov = conf.refstate.plot.split(',')
    if not lov:
        return
    for var in lov:
        plot_ref(sdat, var)
