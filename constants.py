"""defines some constants"""

from collections import OrderedDict, namedtuple
import misc
import processing

Varf = namedtuple('Varf', ['par', 'name', 'arg', 'func'])
FIELD_VAR_LIST = OrderedDict((
    ('t', Varf('t', 'Temperature', 'plot_temperature',
        misc.takefield(0))),
    ('c', Varf('c', 'Composition', 'plot_composition',
        misc.takefield(0))),
    ('n', Varf('eta', 'Viscosity', 'plot_viscosity',
        misc.takefield(0))),
    ('d', Varf('rho', 'Density', 'plot_density',
        misc.takefield(0))),
    ('p', Varf('vp', 'Pressure', 'plot_pressure',
        misc.takefield(3))),
    ('s', Varf('vp', 'Stream function', 'plot_stream',
        processing.calc_stream))
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
