"""Define configuration variables for StagPy.

See :mod:`stagpy.args` for additional definitions related to the command line
interface.
"""

from collections import OrderedDict
import pathlib

from loam.manager import ConfOpt as Conf
from loam.tools import switch_opt, config_conf_section, set_conf_opt


def _actual_index(arg):
    """Turn a string in a integer or slice."""
    if ':' in arg:
        idxs = arg.split(':')
        if len(idxs) > 3:
            raise ValueError('{} is an invalid slice'.format(arg))
        idxs[0] = int(idxs[0]) if idxs[0] else None
        idxs[1] = int(idxs[1]) if idxs[1] else None
        if len(idxs) == 3:
            idxs[2] = int(idxs[2]) if idxs[2] else None
        else:
            idxs = idxs[0:2] + [1]
        return slice(*idxs)
    return int(arg)


def _index_collection(arg):
    """Build an index collection from a command line input."""
    return [_actual_index(item) for item in arg.split(',') if item]


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
    ('shortname', switch_opt(False, None,
                             'StagPy output file name is only prefix')),
    ('timesteps', Conf(None, True, 't',
                       {'nargs': '?', 'const': '', 'type': _index_collection},
                       False, 'timesteps slice')),
    ('snapshots', Conf(None, True, 's',
                       {'nargs': '?', 'const': '', 'type': _index_collection},
                       False, 'snapshots slice')),
))

CONF_DEF['plot'] = OrderedDict((
    ('ratio', Conf(None, True, None,
                   {'nargs': '?', 'const': 0.6, 'type': float},
                   False, 'force aspect ratio of field plot')),
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
        Conf('T,stream', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, 'variables to plot (see stagpy var)')),
    ('perturbation', switch_opt(False, None,
                                'plot departure from average profile')),
    ('shift', Conf(None, True, None, {'type': int},
                   False, 'shift plot horizontally')),
    ('timelabel', switch_opt(False, None, 'add label with time')),
    ('interpolate', switch_opt(True, None, 'apply Gouraud shading')),
    ('colorbar', switch_opt(True, None, 'add color bar to plot')),
    ('ix', Conf(None, True, None, {'type': int},
                False, 'x-index of slice for 3D fields')),
    ('iy', Conf(None, True, None, {'type': int},
                False, 'y-index of slice for 3D fields')),
    ('iz', Conf(None, True, None, {'type': int},
                False, 'z-index of slice for 3D fields')),
    ('isocolors', Conf('', True, None, {}, True,
                       'comma-separated list of colors for isolines')),
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
    ('marktimes',
        Conf('', True, 'M', {},
             False, 'list of times where to put a mark')),
    ('marksteps',
        Conf('', True, 'T', {'type': _index_collection},
             False, 'list of steps where to put a mark')),
    ('marksnaps',
        Conf('', True, 'S', {'type': _index_collection},
             False, 'list of snaps where to put a mark')),
))

CONF_DEF['refstate'] = OrderedDict((
    ('plot',
        Conf('T', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'variables to plot (see stagpy var)')),
    ('style',
        Conf('-', True, None, {},
             True, 'matplotlib line style')),
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

CONF_DEF['info'] = OrderedDict((
    ('output', Conf('t,Tmean,vrms,Nutop,Nubot', True, 'o', {},
                    True, 'time series to print')),
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
    ('refstate', Conf(None, True, None, {'action': 'store_true'},
                      False, 'print refstate variables')),
    ('plates', Conf(None, True, None, {'action': 'store_true'},
                    False, 'print plates variables')),
))

CONF_DEF['config'] = config_conf_section()
