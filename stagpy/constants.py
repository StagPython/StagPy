"""defines some constants"""

import os.path
from collections import OrderedDict, namedtuple

CONFIG_DIR = os.path.expanduser('~/.config/stagpy')

Varf = namedtuple('Varf', ['par', 'name', 'arg'])
FIELD_VAR_LIST = OrderedDict((
    ('t', Varf('t', 'Temperature', 'plot_temperature')),
    ('c', Varf('c', 'Composition', 'plot_composition')),
    ('n', Varf('eta', 'Viscosity', 'plot_viscosity')),
    ('d', Varf('rho', 'Density', 'plot_density')),
    ('h', Varf('wtr', 'Water', 'plot_water')),
    ('a', Varf('age', 'Age', 'plot_age')),
    ('u', Varf('vp', 'x Velocity', 'plot_xvelo')),
    ('v', Varf('vp', 'y Velocity', 'plot_yvelo')),
    ('w', Varf('vp', 'z Velocity', 'plot_zvelo')),
    ('p', Varf('vp', 'Pressure', 'plot_pressure')),
    ('s', Varf('vp', 'Stream function', 'plot_stream')),
))

Varr = namedtuple('Varr', ['name', 'arg', 'min_max', 'prof_idx'])
RPROF_VAR_LIST = OrderedDict((
    ('t', Varr('Temperature', 'plot_temperature', 'plot_minmaxtemp', 1)),
    ('v', Varr('Vertical velocity', 'plot_velocity', 'plot_minmaxvelo', 7)),
    ('n', Varr('Viscosity', 'plot_viscosity', 'plot_minmaxvisco', 13)),
    ('c', Varr('Concentration', 'plot_concentration', 'plot_minmaxcon', 36)),
    ('g', Varr('Grid', 'plot_grid', None, None)),
    ('a', Varr('Advection', 'plot_advection', None, None)),
    ('e', Varr('Energy', 'plot_energy', None, None)),
    ('h', Varr('Concentration Theo', 'plot_conctheo', None, None)),
    ('i', Varr('Init overturn', 'plot_overturn_init', None, None)),
    ('d', Varr('Difference', 'plot_difference', None, None)),
))
