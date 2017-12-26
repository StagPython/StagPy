"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict, namedtuple
from os.path import expanduser
import configparser
import pathlib

HOME_DIR = pathlib.Path(expanduser('~'))
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'
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


class _SubConfig:

    """Hold options for a single subcommand"""

    def __init__(self, parent, name, entries):
        self._parent = parent
        self._name = name
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
        config_parser = self._parent.config_parser
        for opt, meta_opt in CONF_DEF[self._name].items():
            if not meta_opt.conf_arg:
                continue
            if not config_parser.has_option(self._name, opt):
                missing_opts.append(opt)
                continue
            if isinstance(meta_opt.default, bool):
                dflt = config_parser.getboolean(self._name, opt)
            elif isinstance(meta_opt.default, float):
                dflt = config_parser.getfloat(self._name, opt)
            elif isinstance(meta_opt.default, int):
                dflt = config_parser.getint(self._name, opt)
            else:
                dflt = config_parser.get(self._name, opt)
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
