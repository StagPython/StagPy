"""Define configuration variables for StagPy.

See :mod:`stagpy.args` for additional definitions related to the command line
interface.
"""

from __future__ import annotations
import pathlib

from loam.manager import ConfOpt as Conf
from loam import tools
from loam.tools import switch_opt, command_flag
import loam.types


_index_collection = loam.types.list_of(loam.types.slice_or_int_parser)
_float_list = loam.types.list_of(float)

HOME_DIR = pathlib.Path.home()
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'
CONFIG_FILE = CONFIG_DIR / 'config.toml'
CONFIG_LOCAL = pathlib.Path('.stagpy.toml')

CONF_DEF = {}

CONF_DEF['common'] = dict(
    config=command_flag(None, 'print config options'),
    set=tools.set_conf_opt(),
)

CONF_DEF['core'] = dict(
    path=Conf('./', True, 'p', {},
              True, 'path of StagYY run directory or par file', '_files'),
    outname=Conf('stagpy', True, 'n', {}, True, 'output file name prefix'),
    shortname=switch_opt(False, None, 'output file name is only prefix'),
    timesteps=Conf(None, True, 't',
                   {'nargs': '?', 'const': '', 'type': _index_collection},
                   False, 'timesteps slice'),
    snapshots=Conf(None, True, 's',
                   {'nargs': '?', 'const': '', 'type': _index_collection},
                   False, 'snapshots slice'),
)

CONF_DEF['plot'] = dict(
    ratio=Conf(None, True, None, {'nargs': '?', 'const': 0.6, 'type': float},
               False, 'force aspect ratio of field plot'),
    raster=switch_opt(True, None, 'rasterize field plots'),
    format=Conf('pdf', True, None, {},
                True, 'figure format (pdf, eps, svg, png)'),
    vmin=Conf(None, True, None, {'type': float},
              False, 'minimal value on plot'),
    vmax=Conf(None, True, None, {'type': float},
              False, 'maximal value on plot'),
    cminmax=switch_opt(False, 'C', 'constant min max across plots'),
    isolines=Conf(None, True, None, {'type': _float_list},
                  False, 'arbitrary isoline value, comma separated'),
    mplstyle=Conf('stagpy-paper', True, None,
                  {'nargs': '?', 'const': '', 'type': str},
                  True, 'matplotlib style'),
    xkcd=Conf(False, False, None, {}, True, 'use the xkcd style'),
)

CONF_DEF['scaling'] = dict(
    yearins=Conf(3.154e7, False, None, {}, True, 'year in seconds'),
    ttransit=Conf(1.78e15, False, None, {}, True, 'transit time in My'),
    dimensional=switch_opt(False, None, 'use dimensional units'),
    time_in_y=switch_opt(True, None, 'dimensional time is in year'),
    vel_in_cmpy=switch_opt(True, None, 'dimensional velocity is in cm/year'),
    factors=Conf({'s': 'M', 'm': 'k', 'Pa': 'G'},
                 False, None, {}, True, 'custom factors'),
)

CONF_DEF['field'] = dict(
    plot=Conf('T,stream', True, 'o', {'nargs': '?', 'const': '', 'type': str},
              True, 'variables to plot (see stagpy var)'),
    perturbation=switch_opt(False, None,
                            'plot departure from average profile'),
    shift=Conf(None, True, None, {'type': int},
               False, 'shift plot horizontally'),
    timelabel=switch_opt(False, None, 'add label with time'),
    interpolate=switch_opt(False, None, 'apply Gouraud shading'),
    colorbar=switch_opt(True, None, 'add color bar to plot'),
    ix=Conf(None, True, None, {'type': int},
            False, 'x-index of slice for 3D fields'),
    iy=Conf(None, True, None, {'type': int},
            False, 'y-index of slice for 3D fields'),
    iz=Conf(None, True, None, {'type': int},
            False, 'z-index of slice for 3D fields'),
    isocolors=Conf('', True, None, {}, True,
                   'comma-separated list of colors for isolines'),
    cmap=Conf({'T': 'RdBu_r',
               'eta': 'viridis_r',
               'rho': 'RdBu',
               'sII': 'plasma_r',
               'edot': 'Reds'},
              False, None, {}, True, 'custom colormaps'),
)

CONF_DEF['rprof'] = dict(
    plot=Conf('Tmean', True, 'o', {'nargs': '?', 'const': ''},
              True, 'variables to plot (see stagpy var)'),
    style=Conf('-', True, None, {}, True, 'matplotlib line style'),
    average=switch_opt(False, 'a', 'plot temporal average'),
    grid=switch_opt(False, 'g', 'plot grid'),
    depth=switch_opt(False, 'd', 'depth as vertical axis'),
)

CONF_DEF['time'] = dict(
    plot=Conf('Nutop,ebalance,Nubot.Tmean', True, 'o',
              {'nargs': '?', 'const': ''},
              True, 'variables to plot (see stagpy var)'),
    style=Conf('-', True, None, {}, True, 'matplotlib line style'),
    compstat=Conf('', True, None, {'nargs': '?', 'const': ''},
                  False, 'compute mean and rms of listed variables'),
    tstart=Conf(None, True, None, {'type': float}, False, 'beginning time'),
    tend=Conf(None, True, None, {'type': float}, False, 'end time'),
    fraction=Conf(None, True, None, {'type': float},
                  False, 'ending fraction of series to process'),
    marktimes=Conf('', True, 'M', {'type': _float_list},
                   False, 'list of times where to put a mark'),
    marksteps=Conf('', True, 'T', {'type': _index_collection},
                   False, 'list of steps where to put a mark'),
    marksnaps=Conf('', True, 'S', {'type': _index_collection},
                   False, 'list of snaps where to put a mark'),
)

CONF_DEF['refstate'] = dict(
    plot=Conf('T', True, 'o', {'nargs': '?', 'const': ''},
              True, 'variables to plot (see stagpy var)'),
    style=Conf('-', True, None, {}, True, 'matplotlib line style'),
)

CONF_DEF['plates'] = dict(
    plot=Conf('c.T.v2-v2.dv2-v2.topo_top', True, 'o',
              {'nargs': '?', 'const': ''}, True,
              'variables to plot, can be a surface field, field, or dv2'),
    field=Conf('eta', True, None, {},
               True, 'field variable to plot with plates info'),
    stress=switch_opt(
        False, None,
        'Plot deviatoric stress instead of velocity on field plots'),
    continents=switch_opt(True, None, 'Whether to shade continents on plots'),
    vzratio=Conf(0., True, None, {}, True,
                 'Ratio of mean vzabs used as threshold for plates limits'),
    nbplates=switch_opt(False, None,
                        'Plot number of plates as function of time'),
    distribution=switch_opt(False, None, 'Plot plate size distribution'),
    zoom=Conf(None, True, None, {'type': float}, False, 'zoom around surface'),
)

CONF_DEF['info'] = dict(
    output=Conf('t,Tmean,vrms,Nutop,Nubot', True, 'o', {},
                True, 'time series to print'),
)

CONF_DEF['var'] = dict(
    field=command_flag(None, 'print field variables'),
    sfield=command_flag(None, 'print surface field variables'),
    rprof=command_flag(None, 'print rprof variables'),
    time=command_flag(None, 'print time variables'),
    refstate=command_flag(None, 'print refstate variables'),
)

CONF_DEF['config'] = tools.config_conf_section()
