"""defines some constants"""

import os.path
from collections import OrderedDict, namedtuple

CONFIG_DIR = os.path.expanduser('~/.config/stagpy')

Varf = namedtuple('Varf', ['par', 'name', 'arg', 'pcolor_opts'])
FIELD_VAR_LIST = OrderedDict((
    ('t', Varf('t', 'Temperature', 'plot_temperature', {})),
    ('c', Varf('c', 'Composition', 'plot_composition', {})),
    ('n', Varf('eta', 'Viscosity', 'plot_viscosity',
               {'cmap': 'jet_r'})),
    ('d', Varf('rho', 'Density', 'plot_density',
               {'cmap': 'bwr_r', 'vmin': 0.96, 'vmax': 1.04})),
    ('h', Varf('wtr', 'Water', 'plot_water', {})),
    ('a', Varf('age', 'Age', 'plot_age',
               {'vmin': 0})),
    ('i', Varf('nrc', 'ID of continents', 'plot_continents', {})),
    ('s', Varf('str', 'Stress (second invariant)', 'plot_stress',
               {'cmap': 'gnuplot2_r', 'vmin': 500, 'vmax': 20000})),
    ('x', Varf('sx', 'Principal deviatoric stress', 'plot_deviatoric_stress',
               {})),
    ('e', Varf('ed', 'Strain rate', 'plot_strainrate',
               {'cmap': 'Reds', 'vmin': 500, 'vmax': 20000})),
    ('u', Varf('vp', 'x Velocity', 'plot_xvelo', {})),
    ('v', Varf('vp', 'y Velocity', 'plot_yvelo', {})),
    ('w', Varf('vp', 'z Velocity', 'plot_zvelo', {})),
    ('p', Varf('vp', 'Pressure', 'plot_pressure', {})),
    ('l', Varf('vp', 'Stream function', 'plot_stream', {})),
))

Varr = namedtuple('Varr', ['name', 'arg', 'min_max', 'prof_idx'])
RPROF_VAR_LIST = OrderedDict((
    ('t', Varr('Temperature', 'plot_temperature', 'plot_minmaxtemp', 1)),
    ('v', Varr('Vertical velocity', 'plot_velocity', 'plot_minmaxvelo', 7)),
    ('u', Varr('Horizontal velocity', 'plot_velocity', 'plot_minmaxvelo', 10)),
    ('n', Varr('Viscosity', 'plot_viscosity', 'plot_minmaxvisco', 13)),
    ('c', Varr('Concentration', 'plot_concentration', 'plot_minmaxcon', 36)),
    ('g', Varr('Grid', 'plot_grid', None, None)),
    ('z', Varr('Grid km', 'plot_grid_units', None, None)),
    ('a', Varr('Advection', 'plot_advection', None, None)),
    ('e', Varr('Energy', 'plot_energy', None, None)),
    ('h', Varr('Concentration Theo', 'plot_conctheo', None, None)),
    ('i', Varr('Init overturn', 'plot_overturn_init', None, None)),
    ('d', Varr('Difference', 'plot_difference', None, None)),
))

Varp = namedtuple('Varp', ['par', 'name', 'arg'])
PLATES_VAR_LIST = OrderedDict((
    ('c', Varp('c', 'Composition', 'plot_composition')),
    ('n', Varp('eta', 'Viscosity', 'plot_viscosity')),
    ('r', Varp('cs', 'Topography', 'plot_topography')),
    ('a', Varp('age', 'Age', 'plot_age')),
    ('s', Varp('str', 'Stress', 'plot_stress')),
    ('x', Varp('sx', 'Principal deviatoric stress', 'plot_deviatoric_stress')),
    ('e', Varp('ed', 'Strain rate', 'plot_strainrate')),
))
