"""Parse command line arguments and update :attr:`stagpy.conf`."""

from __future__ import annotations

import importlib.resources as imlr
import typing
from inspect import isfunction
from types import MappingProxyType

import matplotlib.pyplot as plt
import matplotlib.style as mpls
from loam.cli import CLIManager, Subcmd

from . import ISOLATED
from . import __doc__ as doc_module
from . import (
    _styles,
    commands,
    conf,
    config,
    field,
    plates,
    refstate,
    rprof,
    time_series,
)
from ._helpers import baredoc
from .config import CONFIG_DIR

if typing.TYPE_CHECKING:
    from typing import Any, Callable, List, Optional


def _sub(cmd: Any, *sections: str) -> Subcmd:
    """Build Subcmd instance."""
    cmd_func = cmd if isfunction(cmd) else cmd.cmd
    return Subcmd(baredoc(cmd), *sections, func=cmd_func)


def _bare_cmd() -> None:
    """Print help message when no arguments are given."""
    print(doc_module)
    print("Run `stagpy -h` for usage")


def _load_mplstyle() -> None:
    """Try to load conf.plot.mplstyle matplotlib style."""
    for style in conf.plot.mplstyle:
        style_fname = style + ".mplstyle"
        if not ISOLATED:
            stfile = config.CONFIG_DIR / style_fname
            if stfile.is_file():
                mpls.use(str(stfile))
                continue
        # try packaged version
        if imlr.is_resource(_styles, style_fname):
            with imlr.path(_styles, style_fname) as stfile:
                mpls.use(str(stfile))
                continue
        mpls.use(style)
    if conf.plot.xkcd:
        plt.xkcd()


SUB_CMDS = MappingProxyType(
    {
        "common_": Subcmd(doc_module, "common", func=_bare_cmd),
        "field": _sub(field, "core", "plot", "scaling"),
        "rprof": _sub(rprof, "core", "plot", "scaling"),
        "time": _sub(time_series, "core", "plot", "scaling"),
        "refstate": _sub(refstate, "core", "plot"),
        "plates": _sub(plates, "core", "plot", "scaling"),
        "info": _sub(commands.info_cmd, "core", "scaling"),
        "var": _sub(commands.var_cmd),
        "version": _sub(commands.version_cmd),
        "config": _sub(commands.config_cmd),
    }
)


def parse_args(arglist: Optional[List[str]] = None) -> Callable[[], None]:
    """Parse cmd line arguments.

    Update :attr:`stagpy.conf` accordingly.

    Args:
        arglist: the list of cmd line arguments. If set to None, the arguments
            are taken from :attr:`sys.argv`.

    Returns:
        the function implementing the sub command to be executed.
    """
    climan = CLIManager(conf, **SUB_CMDS)

    bash_script = CONFIG_DIR / "bash" / "stagpy.sh"
    bash_script.parent.mkdir(parents=True, exist_ok=True)
    climan.bash_complete(bash_script, "stagpy")
    zsh_script = CONFIG_DIR / "zsh" / "_stagpy.sh"
    zsh_script.parent.mkdir(parents=True, exist_ok=True)
    climan.zsh_complete(zsh_script, "stagpy", sourceable=True)

    cmd_args = climan.parse_args(arglist)
    sub_cmd = cmd_args.loam_sub_name

    if sub_cmd is None:
        return cmd_args.func

    if conf.common.config:
        commands.config_pp(climan.sections_list(sub_cmd))

    _load_mplstyle()

    return cmd_args.func
