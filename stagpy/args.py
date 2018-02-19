"""Parse command line arguments and update :attr:`stagpy.conf`."""

from collections import OrderedDict
from inspect import isfunction

from loam.tools import Subcmd, set_conf_str

from . import conf, PARSING_OUT
from . import commands, field, rprof, time_series, plates
from .misc import baredoc
from .config import CONFIG_DIR


def _sub(extra, func):
    """Build Subcmd using doc of func."""
    return Subcmd(extra, dict(func=func), baredoc(func))


SUB_CMDS = OrderedDict((
    (None, Subcmd(['common'],
                  dict(func=lambda: print('stagpy -h for usage')),
                  'read and process StagYY binary data')),
    ('field', _sub(['core', 'plot'], field)),
    ('rprof', _sub(['core', 'plot'], rprof)),
    ('time', _sub(['core', 'plot'], time_series)),
    ('plates', _sub(['core', 'plot', 'scaling'], plates)),
    ('info', _sub(['core'], commands.info_cmd)),
    ('var', _sub([], commands.var_cmd)),
    ('version', _sub([], commands.version_cmd)),
    ('config', _sub([], commands.config_cmd)),
))


def _steps_to_slices():
    """parse timesteps and snapshots arguments and return slices"""
    if conf.core.timesteps is None and conf.core.snapshots is None:
        # default to the last snap
        conf.core.snapshots = slice(-1, None, None)
        return
    elif conf.core.snapshots is not None:
        # snapshots take precedence over timesteps
        # if both are defined
        conf.core.timesteps = None
        steps = conf.core.snapshots
    else:
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


def _update_func(cmd_args):
    """Extract command func if necessary"""
    if not isfunction(cmd_args.func):
        cmd_args.func = cmd_args.func.cmd


def parse_args(arglist=None):
    """Parse cmd line arguments.

    Update :attr:`stagpy.conf` accordingly.

    Args:
        arglist (list of str): the list of cmd line arguments. If set to
            None, the arguments are taken from :attr:`sys.argv`.

    Returns:
        function: the function implementing the sub command to be executed.
    """
    conf.sub_cmds_ = SUB_CMDS
    conf.build_parser_()

    zsh_dir = CONFIG_DIR / 'zsh'
    if not zsh_dir.is_dir():
        zsh_dir.mkdir(parents=True)
    conf.zsh_complete_(zsh_dir / '_stagpy.sh', 'stagpy', 'stagpy-git',
                       sourceable=True)

    bash_dir = CONFIG_DIR / 'bash'
    if not bash_dir.is_dir():
        bash_dir.mkdir(parents=True)
    conf.bash_complete_(bash_dir / 'stagpy.sh', 'stagpy', 'stagpy-git')

    cmd_args, all_subs = conf.parse_args_(arglist)
    sub_cmd = cmd_args.loam_sub_name

    if sub_cmd is None:
        return cmd_args.func

    if sub_cmd != 'config':
        commands.report_parsing_problems(PARSING_OUT)

    _update_func(cmd_args)

    if conf.common.set:
        set_conf_str(conf, conf.common.set)

    if conf.common.config:
        commands.config_pp(all_subs)

    try:
        _steps_to_slices()
    except AttributeError:
        pass
    return cmd_args.func
