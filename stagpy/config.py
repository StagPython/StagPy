"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict, namedtuple
from inspect import getdoc
from subprocess import call
import argcomplete
import argparse
import configparser
import shlex
from . import commands
from .constants import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / 'config'

Conf = namedtuple('ConfigEntry',
                  ['default', 'cmd_arg', 'shortname', 'kwargs',
                   'conf_arg', 'help_string'])

CONF_DEF = OrderedDict()

CONF_DEF['core'] = OrderedDict((
    ('path', Conf('./', True, 'p', {},
                  True, 'StagYY run directory')),
    ('outname', Conf('stagpy', True, 'n', {},
                     True, 'StagPy generic output file name')),
    ('timesteps', Conf(None, True, 't',
                       {'nargs': '?', 'const': ':', 'type': str},
                       False, 'timesteps slice')),
    ('snapshots', Conf(None, True, 's',
                       {'nargs': '?', 'const': ':', 'type': str},
                       False, 'snapshots slice')),
    ('xkcd', Conf(False, True, None, {},
                  True, 'use the xkcd style')),
    ('pdf', Conf(False, True, None, {},
                 True, 'produce non-rasterized pdf (slow!)')),
    ('fontsize', Conf(16, False, None, {},
                      True, 'font size')),
    ('linewidth', Conf(2, False, None, {},
                       True, 'line width')),
    ('matplotback', Conf('agg', False, None, {},
                         True, 'graphical backend')),
    ('useseaborn', Conf(True, False, None, {},
                        True, 'use or not seaborn')),
))

CONF_DEF['scaling'] = OrderedDict((
    ('yearins', Conf(3.154e7, False, None, {},
                     True, 'Year in seconds')),
    ('ttransit', Conf(1.78e15, False, None, {},
                      True, 'Transit time in My')),
    ('kappa', Conf(1.0e-6, False, None, {},
                   True, 'Earth mantle thermal diffusivity m2/s')),
    ('mantle', Conf(2890.0e3, False, None, {},
                    True, 'Thickness of Earth mantle m')),
    ('viscosity_ref', Conf(5.86e22, False, None, {},
                           True, 'Reference viscosity Pa s')),
))

CONF_DEF['plotting'] = OrderedDict((
    ('topomin', Conf(-40, False, None, {},
                     True, 'Min range for topography plots')),
    ('topomax', Conf(100, False, None, {},
                     True, 'Max range for topography plots')),
    ('agemin', Conf(-50, False, None, {},
                    True, 'Min range for age plots')),
    ('agemax', Conf(500, False, None, {},
                    True, 'Max range for age plots')),
    ('velocitymin', Conf(-5000, False, None, {},
                         True, 'Min range for velocity plots')),
    ('velocitymax', Conf(5000, False, None, {},
                         True, 'Max range for velocity plots')),
    ('dvelocitymin', Conf(-250000, False, None, {},
                          True, 'Min range for velocity derivative plots')),
    ('dvelocitymax', Conf(150000, False, None, {},
                          True, 'Max range for velocity derivative plots')),
    ('stressmin', Conf(0, False, None, {},
                       True, 'Min range for stress plots')),
    ('stressmax', Conf(800, False, None, {},
                       True, 'Max range for stress plots')),
    ('lstressmax', Conf(50, False, None, {},
                        True, 'Max range for lithospheric stress plots')),
))

CONF_DEF['field'] = OrderedDict((
    ('plot',
        Conf('T+stream', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, ('specify which variables to plot, '
                    'run stagpy var for a list of variables'))),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
))

CONF_DEF['rprof'] = OrderedDict((
    ('plot',
        Conf('Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'specify which variables to plot')),
    ('average',
        Conf(False, True, 'a', {},
             True, 'Plot temporal average')),
    ('grid',
        Conf(False, True, 'g', {},
             True, 'Plot grid')),
))

CONF_DEF['time'] = OrderedDict((
    ('plot',
        Conf('Nutop,ebalance,Nubot.Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'specify which variables to plot')),
    ('compstat',
        Conf(False, True, None, {},
             True, 'compute steady state statistics')),
    ('tstart',
        Conf(None, True, None, {'type': float},
             False, 'specify beginning for the time series')),
    ('tend',
        Conf(None, True, None, {'type': float},
             False, 'specify end time for the time series')),
))

CONF_DEF['plates'] = OrderedDict((
    ('plot',
        Conf(None, True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             False, ('specify which variable to plot, '
                     'run stagpy var for a list of variables'))),
    ('plot_composition',
        Conf(True, False, None, {},
             True, 'composition scalar field')),
    ('plot_viscosity',
        Conf(True, False, None, {},
             True, 'viscosity scalar field')),
    ('plot_topography',
        Conf(True, False, None, {},
             True, 'topography scalar field')),
    ('plot_age',
        Conf(False, False, None, {},
             True, 'age scalar field')),
    ('plot_stress',
        Conf(False, False, None, {},
             True, 'second invariant of stress scalar field')),
    ('plot_deviatoric_stress',
        Conf(False, False, None, {},
             True, 'principal deviatoric stress')),
    ('plot_strainrate',
        Conf(False, False, None, {},
             True, 'strain rate scalar field')),
    ('vzcheck',
        Conf(False, True, None, {},
             True, 'activate Colin\'s version with vz checking')),
    ('timeprofile',
        Conf(False, True, None, {},
             True, 'plots nb of plates in function of time')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
    ('zoom',
        Conf(None, True, None, {'type': float},
             False, 'Zoom around surface')),
))

CONF_DEF['info'] = OrderedDict((
))

CONF_DEF['var'] = OrderedDict((
))

CONF_DEF['version'] = OrderedDict((
))

CONF_DEF['config'] = OrderedDict((
    ('create',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'create new config file')),
    ('update',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'add missing entries to existing config file')),
    ('edit',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'open config file in a text editor')),
    ('editor',
        Conf('vim', False, None, {},
             True, 'text editor')),
))


def config_cmd(args):
    """Configuration handling"""
    if not (args.create or args.update or args.edit):
        args.update = True
    if args.create or args.update:
        create_config()
    if args.edit:
        call(shlex.split('{} {}'.format(args.editor, CONFIG_FILE)))

Sub = namedtuple('Sub', ['use_core', 'func'])
SUB_CMDS = OrderedDict((
    ('field', Sub(True, commands.field_cmd)),
    ('rprof', Sub(True, commands.rprof_cmd)),
    ('time', Sub(True, commands.time_cmd)),
    ('plates', Sub(True, commands.plates_cmd)),
    ('info', Sub(True, commands.info_cmd)),
    ('var', Sub(False, commands.var_cmd)),
    ('version', Sub(False, commands.version_cmd)),
    ('config', Sub(False, config_cmd)),
))


def _set_conf_default(sub, opt, dflt):
    """Set default value of option"""
    CONF_DEF[sub][opt] = CONF_DEF[sub][opt]._replace(default=dflt)


def _read_section(config_parser, sub_cmd):
    """read section of config parser

    read section corresponding to the sub command sub_cmd
    and set default values to the read values
    """
    config_content = []
    missing_opts = []
    for opt, meta_opt in CONF_DEF[sub_cmd].items():
        if not config_parser.has_option(sub_cmd, opt):
            if meta_opt.conf_arg:
                missing_opts.append(opt)
            continue
        if isinstance(meta_opt.default, bool):
            dflt = config_parser.getboolean(sub_cmd, opt)
        elif isinstance(meta_opt.default, float):
            dflt = config_parser.getfloat(sub_cmd, opt)
        elif isinstance(meta_opt.default, int):
            dflt = config_parser.getint(sub_cmd, opt)
        else:
            dflt = config_parser.get(sub_cmd, opt)
        config_content.append((sub_cmd, opt, dflt))
    return config_content, missing_opts


def _report_missing_config(config_out):
    """Print list of missing entries in config file"""
    _, missing_opts, missing_sections = config_out
    need_update = False
    for sub_cmd, missing in missing_opts.items():
        if missing:
            print('WARNING! Missing options in {} section of config file:'.
                  format(sub_cmd), *missing, sep='\n', end='\n\n')
            need_update = True
    if missing_sections:
        print('WARNING! Missing sections in config file:', *missing_sections,
              sep='\n', end='\n\n')
        need_update = True
    if need_update:
        print('Run stagpy config --update to update config file', end='\n\n')


def create_config():
    """Create config file"""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)
    config_parser = configparser.ConfigParser()
    for sub_cmd, entries in CONF_DEF.items():
        config_parser.add_section(sub_cmd)
        for opt, opt_meta in entries.items():
            if opt_meta.conf_arg:
                config_parser.set(sub_cmd, opt, str(opt_meta.default))
    with CONFIG_FILE.open('w') as out_stream:
        config_parser.write(out_stream)


def read_config():
    """Read config file and set default values"""
    if not CONFIG_FILE.is_file():
        return
    config_parser = configparser.ConfigParser()
    config_parser.read(str(CONFIG_FILE))
    config_content = []
    missing_sections = []
    missing_opts = {}
    for sub_cmd in CONF_DEF:
        if not config_parser.has_section(sub_cmd):
            missing_sections.append(sub_cmd)
            continue
        content, missing_opts[sub_cmd] = _read_section(config_parser, sub_cmd)
        config_content.extend(content)
    return config_content, missing_opts, missing_sections


class Toggle(argparse.Action):

    """argparse Action to store True/False to a +/-arg"""

    def __call__(self, parser, namespace, values, option_string=None):
        """set args attribute with True/False"""
        setattr(namespace, self.dest, bool('-+'.index(option_string[0])))


def add_args(parser, entries):
    """Add arguments to a parser"""
    for arg, conf in entries.items():
        if not conf.cmd_arg:
            continue
        if isinstance(conf.default, bool):
            conf.kwargs.update(action=Toggle, nargs=0)
            names = ['-{}'.format(arg), '+{}'.format(arg)]
            if conf.shortname is not None:
                names.append('-{}'.format(conf.shortname))
                names.append('+{}'.format(conf.shortname))
        else:
            if conf.default is not None:
                conf.kwargs.setdefault('type', type(conf.default))
            names = ['--{}'.format(arg)]
            if conf.shortname is not None:
                names.append('-{}'.format(conf.shortname))
        conf.kwargs.update(help=conf.help_string)
        parser.add_argument(*names, **conf.kwargs)
    parser.set_defaults(**{a: c.default for a, c in entries.items()})
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
            core_parser = add_args(core_parser, CONF_DEF[sub])

    for sub_cmd, meta in SUB_CMDS.items():
        kwargs = {'prefix_chars': '+-', 'help': getdoc(meta.func)}
        if meta.use_core:
            kwargs.update(parents=[core_parser])
        dummy_parser = subparsers.add_parser(sub_cmd, **kwargs)
        dummy_parser = add_args(dummy_parser, CONF_DEF[sub_cmd])
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


def parse_args():
    """Parse cmd line arguments"""
    main_parser = _build_parser()
    argcomplete.autocomplete(main_parser)
    args = main_parser.parse_args()

    if args.func is not config_cmd:
        args.create = False
        args.edit = False
        args.update = False

    try:
        config_out = read_config()
    except configparser.Error:
        config_out = None

    if not args.create or args.edit:
        if config_out is None:
            print('Unable to read config file {}!'.format(CONFIG_FILE),
                  'Run stagpy config --create to obtain a new config file.',
                  '=' * 26, sep='\n')
        elif config_out and not args.update:
            _report_missing_config(config_out)
            for sub, opt, dflt in config_out[0]:
                _set_conf_default(sub, opt, dflt)
                main_parser = _build_parser()
                args = main_parser.parse_args()

    try:
        _steps_to_slices(args)
    except AttributeError:
        pass
    return args
