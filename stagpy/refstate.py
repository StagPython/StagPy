"""Plot reference state profiles."""

import matplotlib.pyplot as plt

from . import conf, misc
from .phyvars import REFSTATE
from .stagyydata import StagyyData


_COMB_TO_SIMPLE = {
    'z': 'z',
    'T': 'Tref',
    'rho': 'rhoref',
    'expan': 'expan',
    'Cp': 'Cp',
    'tcond': 'Tcond',
    'P': 'P',
}


def plot_ref(sdat, var):
    """Plot one reference state.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): a StagyyData instance.
        var (str): refstate variable, a key of :data:`stagpy.phyvars.REFSTATE`.
    """
    fig, axis = plt.subplots()
    adbts = sdat.refstate.adiabats
    if len(adbts) > 2:
        for iad, adia in enumerate(adbts[:-1]):
            try:
                prof = adia[_COMB_TO_SIMPLE[var]]
            except KeyError:
                prof = adia[var]
            axis.plot(prof, adia['z'],
                      conf.refstate.style,
                      label='System {}'.format(iad + 1))
    axis.plot(adbts[-1][var], adbts[-1]['z'],
              conf.refstate.style, color='k',
              label='Combined profile')
    if var == 'tcond':
        axis.set_xscale('log')
    axis.set_xlabel(REFSTATE[var].description)
    axis.set_ylabel('z Position')
    if len(adbts) > 2:
        axis.legend()
    misc.saveplot(fig, 'refstate_{}'.format(var))


def cmd():
    """Implementation of refstate subcommand.

    Other Parameters:
        conf.core
        conf.plot
    """
    sdat = StagyyData(conf.core.path)
    if sdat.refstate.adiabats is None:
        return

    lov = conf.refstate.plot.split(',')
    if not lov:
        return
    for var in lov:
        plot_ref(sdat, var)
