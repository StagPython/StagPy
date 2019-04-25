"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict
import pathlib

from loam.manager import ConfOpt as Conf
from loam.tools import switch_opt, config_conf_section, set_conf_opt

HOME_DIR = pathlib.Path.home()
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'
CONFIG_FILE = CONFIG_DIR / 'config.toml'
CONFIG_LOCAL = pathlib.Path('.stagpy.toml')

CONF_DEF = OrderedDict()

CONF_DEF['common'] = OrderedDict((
    ('config', Conf(None, True, None, {'action': 'store_true'},
                    False, 'print config options')),
    ('set', set_conf_opt()),
))

CONF_DEF['core'] = OrderedDict((
    ('path', Conf('./', True, 'p', {},
                  True, 'path of StagYY run directory or par file', '_files')),
    ('outname', Conf('stagpy', True, 'n', {},
                     True, 'StagPy output file name prefix')),
    ('timesteps', Conf(None, True, 't',
                       {'nargs': '?', 'const': '', 'type': str},
                       False, 'timesteps slice')),
    ('snapshots', Conf(None, True, 's',
                       {'nargs': '?', 'const': '', 'type': str},
                       False, 'snapshots slice')),
))

CONF_DEF['plot'] = OrderedDict((
    ('ratio', Conf(None, True, None,
                   {'nargs': '?', 'const': 0.6, 'type': float},
                   False, 'Force aspect ratio of field plot')),
    ('raster', switch_opt(True, None, 'rasterize field plots')),
    ('format', Conf('pdf', True, None, {},
                    True, 'figure format (pdf, eps, svg, png)')),
    ('vmin', Conf(None, True, None, {'type': float},
                  False, 'minimal value on plot')),
    ('vmax', Conf(None, True, None, {'type': float},
                  False, 'maximal value on plot')),
    ('cminmax', switch_opt(False, 'C', 'constant min max across plots')),
    ('mplstyle', Conf('stagpy-paper', True, None,
                      {'nargs': '?', 'const': '', 'type': str},
                      True, 'matplotlib style')),
    ('xkcd', Conf(False, False, None, {},
                  True, 'use the xkcd style')),
))

CONF_DEF['scaling'] = OrderedDict((
    ('yearins', Conf(3.154e7, False, None, {},
                     True, 'year in seconds')),
    ('ttransit', Conf(1.78e15, False, None, {},
                      True, 'transit time in My')),
    ('dimensional', switch_opt(False, None, 'use dimensional units')),
    ('time_in_y', switch_opt(True, None, 'dimensionful time is in year')),
    ('vel_in_cmpy', switch_opt(True, None,
                               'dimensionful velocity is in cm/year')),
    ('factors', Conf({'s': 'M',
                      'm': 'k',
                      'Pa': 'G'},
                     False, None, {}, True, 'custom factors')),
))

CONF_DEF['field'] = OrderedDict((
    ('plot',
        Conf('T+stream', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, 'variables to plot (see stagpy var)')),
    ('perturbation', switch_opt(False, None,
                                'plot departure from average profile')),
    ('shift', Conf(None, True, None, {'type': int},
                   False, 'Shift plot horizontally')),
    ('interpolate', switch_opt(True, None, 'apply Gouraud shading')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
    ('cmap',
        Conf({'T': 'RdBu_r',
              'eta': 'viridis_r',
              'rho': 'RdBu',
              'sII': 'plasma_r',
              'edot': 'Reds'},
             False, None, {}, True, 'custom colormaps')),
))

CONF_DEF['rprof'] = OrderedDict((
    ('plot',
        Conf('Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'variables to plot (see stagpy var)')),
    ('style',
        Conf('-', True, None, {},
             True, 'matplotlib line style')),
    ('average', switch_opt(False, 'a', 'plot temporal average')),
    ('grid', switch_opt(False, 'g', 'plot grid')),
    ('depth', switch_opt(False, 'd', 'depth as vertical axis')),
))

CONF_DEF['time'] = OrderedDict((
    ('plot',
        Conf('Nutop,ebalance,Nubot.Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'variables to plot (see stagpy var)')),
    ('style',
        Conf('-', True, None, {},
             True, 'matplotlib line style')),
    ('compstat', switch_opt(False, None, 'compute steady state statistics')),
    ('tstart',
        Conf(None, True, None, {'type': float},
             False, 'beginning time')),
    ('tend',
        Conf(None, True, None, {'type': float},
             False, 'end time')),
    ('fraction',
        Conf(None, True, None, {'type': float},
             False, 'ending fraction of series to process')),
))

CONF_DEF['plates'] = OrderedDict((
    ('plot',
        Conf('c,eta,sc', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, 'variables to plot (see stagpy var)')),
    ('vzcheck', switch_opt(False, None,
                           'activate Colin\'s version with vz checking')),
    ('timeprofile', switch_opt(False, None,
                               'nb of plates as function of time')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
    ('zoom',
        Conf(None, True, None, {'type': float},
             False, 'zoom around surface')),
    ('topomin', Conf(-40, False, None, {},
                     True, 'min topography in plots')),
    ('topomax', Conf(100, False, None, {},
                     True, 'max topography in plots')),
    ('agemin', Conf(-50, False, None, {},
                    True, 'min age in plots')),
    ('agemax', Conf(500, False, None, {},
                    True, 'max age in plots')),
    ('vmin', Conf(-5000, False, None, {},
                  True, 'min velocity in plots')),
    ('vmax', Conf(5000, False, None, {},
                  True, 'max velocity in plots')),
    ('dvmin', Conf(-250000, False, None, {},
                   True, 'min velocity derivative in plots')),
    ('dvmax', Conf(150000, False, None, {},
                   True, 'max velocity derivative in plots')),
    ('stressmin', Conf(0, False, None, {},
                       True, 'min stress in plots')),
    ('stressmax', Conf(800, False, None, {},
                       True, 'max stress in plots')),
    ('lstressmax', Conf(50, False, None, {},
                        True, 'max lithospheric stress in plots')),
))

CONF_DEF['var'] = OrderedDict((
    ('field', Conf(None, True, None, {'action': 'store_true'},
                   False, 'print field variables')),
    ('sfield', Conf(None, True, None, {'action': 'store_true'},
                    False, 'print surface field variables')),
    ('rprof', Conf(None, True, None, {'action': 'store_true'},
                   False, 'print rprof variables')),
    ('time', Conf(None, True, None, {'action': 'store_true'},
                  False, 'print time variables')),
    ('plates', Conf(None, True, None, {'action': 'store_true'},
                    False, 'print plates variables')),
))

CONF_DEF['config'] = config_conf_section()
