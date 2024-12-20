"""Parse command line arguments."""

from __future__ import annotations

import importlib.resources as imlr
import typing
from pathlib import Path
from types import MappingProxyType

import matplotlib.pyplot as plt
import matplotlib.style as mpls
from loam.cli import CLIManager, Subcmd

from . import __doc__ as doc_module
from . import (
    _styles,
    commands,
    field,
    plates,
    refstate,
    rprof,
    time_series,
)
from ._helpers import baredoc
from .config import Config

if typing.TYPE_CHECKING:
    from typing import Callable


def _sub(cmd: Callable[[Config], None], *sections: str) -> Subcmd:
    """Build Subcmd instance."""
    return Subcmd(baredoc(cmd), *sections, func=cmd)


def _bare_cmd(conf: Config) -> None:
    """Print help message when no arguments are given."""
    print(doc_module)
    print("Run `stagpy -h` for usage")


def _load_mplstyle(conf: Config) -> None:
    """Try to load conf.plot.mplstyle matplotlib style."""
    for style in conf.plot.mplstyle:
        # try packaged version
        style_file = imlr.files(_styles).joinpath(f"{style}.mplstyle")
        if style_file.is_file():
            with imlr.as_file(style_file) as stfile:
                mpls.use(str(stfile))
                continue
        mpls.use(style)
    if conf.plot.xkcd:
        plt.xkcd()


assert doc_module is not None
SUB_CMDS = MappingProxyType(
    {
        "common_": Subcmd(doc_module, "common", func=_bare_cmd),
        "field": _sub(field.cmd, "core", "plot", "scaling"),
        "rprof": _sub(rprof.cmd, "core", "plot", "scaling"),
        "time": _sub(time_series.cmd, "core", "plot", "scaling"),
        "refstate": _sub(refstate.cmd, "core", "plot"),
        "plates": _sub(plates.cmd, "core", "plot", "scaling"),
        "info": _sub(commands.info_cmd, "core", "scaling"),
        "var": _sub(commands.var_cmd),
        "version": _sub(commands.version_cmd),
        "config": _sub(commands.config_cmd),
    }
)


def parse_args(
    conf: Config, arglist: list[str] | None = None
) -> Callable[[Config], None]:
    """Parse cmd line arguments.

    Args:
        conf: configuration.
        arglist: the list of cmd line arguments. If set to None, the arguments
            are taken from :attr:`sys.argv`.

    Returns:
        the function implementing the sub command to be executed.
    """

    def compl_cmd(conf: Config) -> None:
        if conf.completions.zsh:
            filepath = Path("_stagpy.sh")
            print(f"writing zsh completion file {filepath}")
            climan.zsh_complete(filepath, "stagpy", sourceable=True)
        elif conf.completions.bash:
            filepath = Path("stagpy.sh")
            print(f"writing bash completion file {filepath}")
            climan.bash_complete(filepath, "stagpy")
        else:
            print("please choose a shell, `--help` for available options")

    climan = CLIManager(
        conf,
        **SUB_CMDS,
        completions=Subcmd("generate completion scripts", func=compl_cmd),
    )

    cmd_args = climan.parse_args(arglist)
    sub_cmd = cmd_args.loam_sub_name

    if sub_cmd is None:
        return cmd_args.func

    if conf.common.config:
        commands.config_pp(climan.sections_list(sub_cmd), conf)

    _load_mplstyle(conf)

    return cmd_args.func
