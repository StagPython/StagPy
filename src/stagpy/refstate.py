"""Plot reference state profiles."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from . import _helpers
from .config import Config
from .phyvars import REFSTATE
from .stagyydata import StagyyData, _sdat_from_conf


def plot_ref(sdat: StagyyData, var: str, conf: Optional[Config] = None) -> None:
    """Plot one reference state.

    Args:
        sdat: a `StagyyData` instance.
        var: refstate variable, a key of :data:`~stagpy.phyvars.REFSTATE`.
        conf: configuration.
    """
    if conf is None:
        conf = Config.default_()
    fig, axis = plt.subplots()
    adbts = sdat.refstate.adiabats
    if len(adbts) > 2:
        for iad, adia in enumerate(adbts[:-1], 1):
            axis.plot(adia[var], adia["z"], conf.refstate.style, label=f"System {iad}")
    axis.plot(
        adbts[-1][var],
        adbts[-1]["z"],
        conf.refstate.style,
        color="k",
        label="Combined profile",
    )
    if var == "Tcond":
        axis.set_xscale("log")
    axis.set_xlabel(REFSTATE[var].description)
    axis.set_ylabel("z Position")
    if len(adbts) > 2:
        axis.legend()
    _helpers.saveplot(conf, fig, f"refstate_{var}")


def cmd(conf: Config) -> None:
    """Implementation of refstate subcommand."""
    sdat = _sdat_from_conf(conf.core)

    for var in conf.refstate.plot:
        plot_ref(sdat, var, conf)
