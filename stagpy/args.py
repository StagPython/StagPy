"""Parse command line arguments and update :attr:`stagpy.conf`."""

from collections import OrderedDict
from inspect import isfunction

from loam.tools import set_conf_str, create_complete_files
from loam.cli import Subcmd, CLIManager

from . import conf, PARSING_OUT
from . import commands, field, rprof, time_series, plates
from .misc import baredoc
from .config import CONFIG_DIR


def _sub(cmd, *sections):
    """Build Subcmd instance."""
    cmd_func = cmd if isfunction(cmd) else cmd.cmd
    return Subcmd(baredoc(cmd), *sections, func=cmd_func)


SUB_CMDS = OrderedDict((
    ('common_', Subcmd('read and process StagYY binary data', 'common',
                       func=lambda: print('stagpy -h for usage'))),
    ('field', _sub(field, 'core', 'plot')),
    ('rprof', _sub(rprof, 'core', 'plot')),
    ('time', _sub(time_series, 'core', 'plot')),
    ('plates', _sub(plates, 'core', 'plot', 'scaling')),
    ('info', _sub(commands.info_cmd, 'core')),
    ('var', _sub(commands.var_cmd)),
    ('version', _sub(commands.version_cmd)),
    ('config', _sub(commands.config_cmd)),
))


def _steps_to_slices():
    """parse timesteps and snapshots arguments and return slices"""
    if not (conf.core.timesteps or conf.core.snapshots):
        # default to the last snap
        conf.core.timesteps = None
        conf.core.snapshots = slice(-1, None, None)
        return
    elif conf.core.snapshots:
        # snapshots take precedence over timesteps
        # if both are defined
        conf.core.timesteps = None
        steps = conf.core.snapshots
    else:
        conf.core.snapshots = None
        steps = conf.core.timesteps
    steps = steps.split(':')
    steps[0] = int(steps[0]) if steps[0] else None
    if len(steps) == 1:
        steps.append(steps[0] + 1)
    steps[1] = int(steps[1]) if steps[1] else None
    if len(steps) != 3:
        steps = steps[0:2] + [1]
    steps[2] = int(steps[2]) if steps[2] else None
    steps = slice(*steps)
    if conf.core.snapshots is not None:
        conf.core.snapshots = steps
    else:
        conf.core.timesteps = steps


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

    try:
        _steps_to_slices()
    except AttributeError:
        pass
    return cmd_args.func
