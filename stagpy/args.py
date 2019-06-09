"""Parse command line arguments and update :attr:`stagpy.conf`."""

from collections import OrderedDict
from inspect import isfunction

from loam.tools import set_conf_str, create_complete_files
from loam.cli import Subcmd, CLIManager

from . import conf, PARSING_OUT, load_mplstyle
from . import commands, field, rprof, time_series, refstate, plates
from .misc import baredoc
from .config import CONFIG_DIR


def _sub(cmd, *sections):
    """Build Subcmd instance."""
    cmd_func = cmd if isfunction(cmd) else cmd.cmd
    return Subcmd(baredoc(cmd), *sections, func=cmd_func)


SUB_CMDS = OrderedDict((
    ('common_', Subcmd('read and process StagYY binary data', 'common',
                       func=lambda: print('stagpy -h for usage'))),
    ('field', _sub(field, 'core', 'plot', 'scaling')),
    ('rprof', _sub(rprof, 'core', 'plot', 'scaling')),
    ('time', _sub(time_series, 'core', 'plot', 'scaling')),
    ('refstate', _sub(refstate, 'core', 'plot')),
    ('plates', _sub(plates, 'core', 'plot', 'scaling')),
    ('info', _sub(commands.info_cmd, 'core')),
    ('var', _sub(commands.var_cmd)),
    ('version', _sub(commands.version_cmd)),
    ('config', _sub(commands.config_cmd)),
))


def parse_args(arglist=None):
    """Parse cmd line arguments.

    Update :attr:`stagpy.conf` accordingly.

    Args:
        arglist (list of str): the list of cmd line arguments. If set to
            None, the arguments are taken from :attr:`sys.argv`.

    Returns:
        function: the function implementing the sub command to be executed.
    """
    climan = CLIManager(conf, **SUB_CMDS)

    create_complete_files(climan, CONFIG_DIR, 'stagpy', 'stagpy-git',
                          zsh_sourceable=True)

    cmd_args, all_subs = climan.parse_args(arglist)
    sub_cmd = cmd_args.loam_sub_name

    if sub_cmd is None:
        return cmd_args.func

    if sub_cmd != 'config':
        commands.report_parsing_problems(PARSING_OUT)

    if conf.common.set:
        set_conf_str(conf, conf.common.set)

    if conf.common.config:
        commands.config_pp(all_subs)

    load_mplstyle()

    return cmd_args.func
