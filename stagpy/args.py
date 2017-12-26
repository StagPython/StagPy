"""Parse command line arguments and update conf"""

from collections import OrderedDict, namedtuple
from inspect import getdoc
import argparse
import argcomplete
from . import commands, conf
from .config import CONF_DEF

Sub = namedtuple('Sub', ['use_core', 'func'])
SUB_CMDS = OrderedDict((
    ('field', Sub(True, commands.field_cmd)),
    ('rprof', Sub(True, commands.rprof_cmd)),
    ('time', Sub(True, commands.time_cmd)),
    ('plates', Sub(True, commands.plates_cmd)),
    ('info', Sub(True, commands.info_cmd)),
    ('var', Sub(False, commands.var_cmd)),
    ('version', Sub(False, commands.version_cmd)),
    ('config', Sub(False, commands.config_cmd)),
))


class Toggle(argparse.Action):

    """argparse Action to store True/False to a +/-arg"""

    def __call__(self, parser, namespace, values, option_string=None):
        """set args attribute with True/False"""
        setattr(namespace, self.dest, bool('-+'.index(option_string[0])))


def add_args(subconf, parser, entries):
    """Add arguments to a parser"""
    for arg, meta in entries.items():
        if not meta.cmd_arg:
            continue
        if isinstance(meta.default, bool):
            meta.kwargs.update(action=Toggle, nargs=0)
            names = ['-{}'.format(arg), '+{}'.format(arg)]
            if meta.shortname is not None:
                names.append('-{}'.format(meta.shortname))
                names.append('+{}'.format(meta.shortname))
        else:
            if meta.default is not None:
                meta.kwargs.setdefault('type', type(meta.default))
            names = ['--{}'.format(arg)]
            if meta.shortname is not None:
                names.append('-{}'.format(meta.shortname))
        meta.kwargs.update(help=meta.help_string)
        parser.add_argument(*names, **meta.kwargs)
    parser.set_defaults(**{a: subconf[a] for a in entries})
    return parser


def _build_parser():
    """Return complete parser"""
    main_parser = argparse.ArgumentParser(
        description='read and process StagYY binary data')
    main_parser.set_defaults(func=lambda _: print('stagpy -h for usage'))
    subparsers = main_parser.add_subparsers()

    core_parser = argparse.ArgumentParser(add_help=False, prefix_chars='-+')
    for sub in CONF_DEF:
        if sub not in SUB_CMDS:
            core_parser = add_args(conf[sub], core_parser, CONF_DEF[sub])

    for sub_cmd, meta in SUB_CMDS.items():
        kwargs = {'prefix_chars': '+-', 'help': getdoc(meta.func)}
        if meta.use_core:
            kwargs.update(parents=[core_parser])
        dummy_parser = subparsers.add_parser(sub_cmd, **kwargs)
        dummy_parser = add_args(conf[sub_cmd], dummy_parser, CONF_DEF[sub_cmd])
        dummy_parser.set_defaults(func=meta.func)

    return main_parser


def _steps_to_slices(args):
    """parse timesteps and snapshots arguments and return slices"""
    if args.timesteps is None and args.snapshots is None:
        # default to the last snap
        args.snapshots = slice(-1, None, None)
        return None
    elif args.snapshots is not None:
        # snapshots take precedence over timesteps
        # if both are defined
        args.timesteps = None
        steps = args.snapshots
    else:
        steps = args.timesteps
    steps = steps.split(':')
    steps[0] = int(steps[0]) if steps[0] else None
    if len(steps) == 1:
        steps.append(steps[0] + 1)
    steps[1] = int(steps[1]) if steps[1] else None
    if len(steps) != 3:
        steps = steps[0:2] + [1]
    steps[2] = int(steps[2]) if steps[2] else None
    steps = slice(*steps)
    if args.snapshots is not None:
        args.snapshots = steps
    else:
        args.timesteps = steps


def _update_subconf(cmd_args, sub):
    """Set subconfig options accordingly to cmd line args"""
    for opt, meta in CONF_DEF[sub].items():
        if not meta.cmd_arg:
            continue
        conf[sub][opt] = getattr(cmd_args, opt)


def parse_args():
    """Parse cmd line arguments"""
    main_parser = _build_parser()
    argcomplete.autocomplete(main_parser)
    cmd_args = main_parser.parse_args()

    if cmd_args.func is not commands.config_cmd:
        conf.report_parsing_problems()

    # determine sub command
    sub_cmd = None
    for sub, meta in SUB_CMDS.items():
        if cmd_args.func is meta.func:
            sub_cmd = sub
            break

    # core options
    if SUB_CMDS[sub_cmd].use_core:
        for sub in CONF_DEF:
            if sub not in SUB_CMDS:
                _update_subconf(cmd_args, sub)

    # options specific to the subcommand
    _update_subconf(cmd_args, sub_cmd)

    try:
        _steps_to_slices(cmd_args)
    except AttributeError:
        pass
    return cmd_args
