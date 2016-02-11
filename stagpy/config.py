"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict, namedtuple
from os import mkdir
from subprocess import call
import argcomplete
import argparse
import configparser
import os.path
import shlex
from . import commands, parfile
from .constants import CONFIG_DIR

CONFIG_FILE = os.path.join(CONFIG_DIR, 'config')

Conf = namedtuple('ConfigEntry',
                  ['default', 'cmd_arg', 'shortname', 'kwargs',
                   'conf_arg', 'help_string'])
CORE = OrderedDict((
    ('path', Conf('./', True, 'p', {},
                  True, 'StagYY run directory')),
    ('name', Conf('test', True, None, {},
                  True, 'StagYY generic output file name')),
    ('outname', Conf('stagpy', True, 'n', {},
                     True, 'StagPy generic output file name')),
    ('geometry', Conf('annulus', True, 'g', {'choices': ['annulus']},
                      True, 'geometry of the domain')),
    ('timestep', Conf('100', True, 's', {},
                      True, 'timestep slice')),
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
FIELD = OrderedDict((
    ('plot',
        Conf(None, True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             False, 'specify which variable to plot')),
    ('plot_temperature',
        Conf(True, False, None, {},
             True, 'temperature scalar field')),
    ('plot_xvelo',
        Conf(False, False, None, {},
             True, 'x velocity scalar field')),
    ('plot_yvelo',
        Conf(False, False, None, {},
             True, 'y velocity scalar field')),
    ('plot_zvelo',
        Conf(False, False, None, {},
             True, 'z velocity scalar field')),
    ('plot_pressure',
        Conf(True, False, None, {},
             True, 'pressure scalar field')),
    ('plot_stream',
        Conf(True, False, None, {},
             True, 'stream function scalar field')),
    ('plot_composition',
        Conf(False, False, None, {},
             True, 'composition scalar field')),
    ('plot_viscosity',
        Conf(False, False, None, {},
             True, 'viscosity scalar field')),
    ('plot_density',
        Conf(False, False, None, {},
             True, 'density scalar field')),
    ('plot_water',
        Conf(False, False, None, {},
             True, 'water concentration scalar field')),
    ('plot_age',
        Conf(False, False, None, {},
             True, 'age scalar field')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
))
RPROF = OrderedDict((
    ('plot',
        Conf(None, True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             False, 'specify which variable to plot')),
    ('plot_grid',
        Conf(True, False, None, {},
             True, 'plot grid')),
    ('plot_temperature',
        Conf(True, False, None, {},
             True, 'plot temperature')),
    ('plot_minmaxtemp',
        Conf(False, False, None, {},
             True, 'plot min and max temperature')),
    ('plot_velocity',
        Conf(True, False, None, {},
             True, 'plot velocity')),
    ('plot_minmaxvelo',
        Conf(False, False, None, {},
             True, 'plot min and max velocity')),
    ('plot_viscosity',
        Conf(False, False, None, {},
             True, 'plot viscosity')),
    ('plot_minmaxvisco',
        Conf(False, False, None, {},
             True, 'plot min and max viscosity')),
    ('plot_advection',
        Conf(True, False, None, {},
             True, 'plot heat advction')),
    ('plot_energy',
        Conf(True, False, None, {},
             True, 'plot energy')),
    ('plot_concentration',
        Conf(True, False, None, {},
             True, 'plot concentration')),
    ('plot_minmaxcon',
        Conf(False, False, None, {},
             True, 'plot min and max concentration')),
    ('plot_conctheo',
        Conf(True, False, None, {},
             True, 'plot concentration theo')),
    ('plot_overturn_init',
        Conf(True, False, None, {},
             True, 'plot overturn init')),
    ('plot_difference',
        Conf(True, False, None, {},
             True, 'plot difference between T and C profs and overturned \
                version of their initial values')),
))
TIME = OrderedDict((
    ('compstat',
        Conf(True, True, None, {},
             True, 'compute steady state statistics')),
    ('annottmin',
        Conf(False, True, None, {},
             True, 'put an arrow at tminc and tmint')),
    ('tmint',
        Conf(0., True, None, {},
             False, 'specify tmint')),
    ('tminc',
        Conf(0., True, None, {},
             False, 'specify tminc')),
))
PLATES = OrderedDict((
    ('vzcheck',
        Conf(False, True, None, {},
             True, 'activate Colin\'s version with vz checking')),
    ('timeprofile',
        Conf(False, True, None, {},
             True, 'plots nb of plates in function of time')),
    ('dsa',
        Conf(0.05, False, None, {},
             True, 'thickness of sticky air')),
))
VAR = OrderedDict((
))
CONFIG = OrderedDict((
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
    """sub command config"""
    if args.create or args.update:
        create_config()
    if args.edit:
        call(shlex.split(args.editor + ' ' + CONFIG_FILE))

Sub = namedtuple('Sub', ['conf_dict', 'use_core', 'func', 'help_string'])
SUB_CMDS = OrderedDict((
    ('field', Sub(FIELD, True, commands.field_cmd,
                  'plot scalar fields')),
    ('rprof', Sub(RPROF, True, commands.rprof_cmd,
                  'plot radial profiles')),
    ('time', Sub(TIME, True, commands.time_cmd,
                 'plot temporal series')),
    ('plates', Sub(PLATES, True, commands.plates_cmd,
                   'plate analysis')),
    ('var', Sub(VAR, False, commands.var_cmd,
                'print the list of variables')),
    ('config', Sub(CONFIG, False, config_cmd,
                   'configuration handling')),
))
DummySub = namedtuple('DummySub', ['conf_dict'])
DUMMY_CMDS = OrderedDict((
    ('core', DummySub(CORE)),
))
DUMMY_CMDS.update(SUB_CMDS)


def _set_conf_default(conf_dict, opt, dflt):
    """set default value of option in conf_dict"""
    conf_dict[opt] = conf_dict[opt]._replace(default=dflt)


def _read_section(config_parser, sub_cmd, meta):
    """read section of config parser

    read section corresponding to the sub command sub_cmd
    and set meta.conf_dict default values to the read values
    """
    missing_opts = []
    for opt, meta_opt in meta.conf_dict.items():
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
        _set_conf_default(meta.conf_dict, opt, dflt)
    return missing_opts


def create_config():
    """Create config file"""
    config_parser = configparser.ConfigParser()
    for sub_cmd, meta in DUMMY_CMDS.items():
        config_parser.add_section(sub_cmd)
        for opt, opt_meta in meta.conf_dict.items():
            if opt_meta.conf_arg:
                config_parser.set(sub_cmd, opt, str(opt_meta.default))
    with open(CONFIG_FILE, 'w') as out_stream:
        config_parser.write(out_stream)


def read_config(args):
    """Read config file and set conf_dict as needed"""
    if not os.path.isfile(CONFIG_FILE):
        if not args.update:
            print('Config file {} not found.'.format(CONFIG_FILE))
            print('Run stagpy config --create')
        return
    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_FILE)
    missing_sections = []
    for sub_cmd, meta in DUMMY_CMDS.items():
        if not config_parser.has_section(sub_cmd):
            missing_sections.append(sub_cmd)
            continue
        missing_opts = _read_section(config_parser, sub_cmd, meta)
        if missing_opts and not args.update:
            print('WARNING! Missing options in {} section of config file:'.
                  format(sub_cmd))
            print(*missing_opts)
            print()
    if missing_sections and not args.update:
        print('WARNING! Missing sections in config file:')
        print(*missing_sections)
        print()
    if (missing_sections or missing_opts) and not args.update:
        print('Run stagpy config --update to update config file')
        print()


class Toggle(argparse.Action):

    """argparse Action to store True/False to a +/-arg"""

    def __call__(self, parser, namespace, values, option_string=None):
        """set args attribute with True/False"""
        setattr(namespace, self.dest, bool('-+'.index(option_string[0])))


def add_args(parser, conf_dict):
    """Add arguments to a parser"""
    for arg, conf in conf_dict.items():
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
    parser.set_defaults(**{a: c.default for a, c in conf_dict.items()})
    return parser


def parse_args():
    """Parse cmd line arguments"""
    # get path from config file before
    if not os.path.isdir(CONFIG_DIR):
        mkdir(CONFIG_DIR)
    dummy_parser = argparse.ArgumentParser(add_help=False)
    _, remainder = dummy_parser.parse_known_args()
    keep_cmd_path = '-p' in remainder or '--path' in remainder
    dummy_parser = add_args(dummy_parser, {'path': CORE['path']})
    args, remainder = dummy_parser.parse_known_args()
    cmd_path = args.path
    _set_conf_default(CORE, 'path', args.path)
    if 'config' in remainder:
        dummy_sub = dummy_parser.add_subparsers()
        dummy_conf = dummy_sub.add_parser('config', add_help=False)
        dummy_conf = add_args(dummy_conf, CONFIG)
        args, _ = dummy_parser.parse_known_args()
    else:
        args.create = False
        args.edit = False
        args.update = False
    if not (args.create or args.edit):
        try:
            read_config(args)
        except:
            print('ERROR while reading config file')
            print('Run stagpy config --create to obtain a new config file')
            print('=' * 26)
            raise
    if keep_cmd_path:
        args.path = cmd_path
        _set_conf_default(CORE, 'path', args.path)
    par_nml = parfile.readpar(args)

    main_parser = argparse.ArgumentParser(
        description='read and process StagYY binary data')
    main_parser = add_args(main_parser, {'path': CORE['path']})
    main_parser.set_defaults(func=lambda _: print('stagpy -h for usage'))
    subparsers = main_parser.add_subparsers()

    core_parser = argparse.ArgumentParser(add_help=False, prefix_chars='-+')
    core_parser = add_args(core_parser, CORE)
    core_parser.set_defaults(name=par_nml['ioin']['output_file_stem'])

    for sub_cmd, meta in SUB_CMDS.items():
        kwargs = {'prefix_chars': '+-', 'help': meta.help_string}
        if meta.use_core:
            kwargs.update(parents=[core_parser])
        dummy_parser = subparsers.add_parser(sub_cmd, **kwargs)
        dummy_parser = add_args(dummy_parser, meta.conf_dict)
        dummy_parser.set_defaults(func=meta.func)

    argcomplete.autocomplete(main_parser)
    args = main_parser.parse_args()
    args.par_nml = par_nml
    return args
