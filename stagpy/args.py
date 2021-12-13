"""Parse command line arguments and update :attr:`stagpy.conf`."""

from __future__ import annotations
from inspect import isfunction
from types import MappingProxyType
import typing

from loam.tools import set_conf_str, create_complete_files
from loam.cli import Subcmd, CLIManager

from . import __doc__ as doc_module
from . import conf, PARSING_OUT, load_mplstyle
from . import commands, field, rprof, time_series, refstate, plates
from ._helpers import baredoc
from .config import CONFIG_DIR

if typing.TYPE_CHECKING:
    from typing import Any, Optional, List, Callable


def _sub(cmd: Any, *sections: str) -> Subcmd:
    """Build Subcmd instance."""
    cmd_func = cmd if isfunction(cmd) else cmd.cmd
    return Subcmd(baredoc(cmd), *sections, func=cmd_func)


def _bare_cmd() -> None:
    """Print help message when no arguments are given."""
    print(doc_module)
    print('Run `stagpy -h` for usage')


SUB_CMDS = MappingProxyType({
    'common_': Subcmd(doc_module, 'common', func=_bare_cmd),
    'field': _sub(field, 'core', 'plot', 'scaling'),
    'rprof': _sub(rprof, 'core', 'plot', 'scaling'),
    'time': _sub(time_series, 'core', 'plot', 'scaling'),
    'refstate': _sub(refstate, 'core', 'plot'),
    'plates': _sub(plates, 'core', 'plot', 'scaling'),
    'info': _sub(commands.info_cmd, 'core', 'scaling'),
    'var': _sub(commands.var_cmd),
    'version': _sub(commands.version_cmd),
    'config': _sub(commands.config_cmd),
})


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

    create_complete_files(climan, CONFIG_DIR, 'stagpy', zsh_sourceable=True)

    cmd_args = climan.parse_args(arglist)
    sub_cmd = cmd_args.loam_sub_name

    if sub_cmd is None:
        return cmd_args.func

    if sub_cmd != 'config':
        commands.report_parsing_problems(PARSING_OUT)

    if conf.common.set:
        set_conf_str(conf, conf.common.set)

    if conf.common.config:
        commands.config_pp(climan.sections_list(sub_cmd))

    load_mplstyle()

    return cmd_args.func
