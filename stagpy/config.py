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
import pathlib
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


def _build_parser(conf):
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


def parse_args(conf):
    """Parse cmd line arguments"""
    main_parser = _build_parser(conf)
    argcomplete.autocomplete(main_parser)
    args = main_parser.parse_args()

    if args.func is not config_cmd:
        args.create = False
        args.edit = False
        args.update = False
        conf.report_parsing_problems()

    try:
        _steps_to_slices(args)
    except AttributeError:
        pass
    return args


class _SubConfig:

    """Hold options for a single subcommand"""

    def __init__(self, parent, name, entries):
        self.parent = parent
        self.name = name
        for opt, meta in entries.items():
            self[opt] = meta.default

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def read_section(self):
        """Read section of config parser

        read section corresponding to the sub command
        and set options accordingly
        """
        missing_opts = []
        config_parser = self.parent.config_parser
        for opt, meta_opt in CONF_DEF[self.name].items():
            if not meta_opt.conf_arg:
                continue
            if not config_parser.has_option(self.name, opt):
                missing_opts.append(opt)
                continue
            if isinstance(meta_opt.default, bool):
                dflt = config_parser.getboolean(self.name, opt)
            elif isinstance(meta_opt.default, float):
                dflt = config_parser.getfloat(self.name, opt)
            elif isinstance(meta_opt.default, int):
                dflt = config_parser.getint(self.name, opt)
            else:
                dflt = config_parser.get(self.name, opt)
            self[opt] = dflt
        return missing_opts


class StagpyConfiguration:

    """Hold StagPy configuration options"""

    def __init__(self, config_file=CONFIG_FILE):
        """Config is set with default values and updated with config_file"""
        for sub, entries in CONF_DEF.items():
            self[sub] = _SubConfig(self, sub, entries)
        self.config_parser = configparser.ConfigParser()
        if config_file is not None:
            self.config_file = pathlib.Path(config_file)
            self._missing_parsing = self.read_config()
        else:
            self.config_file = None
            self._missing_parsing = {}, []

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def read_config(self):
        """Read config file and set config values accordingly"""
        if not self.config_file.is_file():
            return None, None
        try:
            self.config_parser.read(str(self.config_file))
        except configparser.Error:
            return None, None
        missing_sections = []
        missing_opts = {}
        for sub in CONF_DEF:
            if not self.config_parser.has_section(sub):
                missing_sections.append(sub)
                continue
            missing_opts[sub] = self[sub].read_section()
        return missing_opts, missing_sections

    def report_parsing_problems(self):
        """Output message about parsing problems"""
        missing_opts, missing_sections = self._missing_parsing
        need_update = False
        if missing_opts is None or missing_sections is None:
            print('Unable to read config file {}!'.format(CONFIG_FILE),
                  'Run stagpy config --create to obtain a new config file.',
                  '=' * 26, sep='\n')
            return
        for sub_cmd, missing in missing_opts.items():
            if missing:
                print('WARNING! Missing options in {} section of config file:'.
                      format(sub_cmd), *missing, sep='\n', end='\n\n')
                need_update = True
        if missing_sections:
            print('WARNING! Missing sections in config file:',
                  *missing_sections, sep='\n', end='\n\n')
            need_update = True
        if need_update:
            print('Run stagpy config --update to update config file',
                  end='\n\n')
