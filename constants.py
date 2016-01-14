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

Varr = namedtuple('Varr', ['name', 'arg', 'min_max'])
RPROF_VAR_LIST = OrderedDict((
    ('t', Varr('Temperature', 'plot_temperature', 'plot_minmaxtemp')),
    ('v', Varr('Velocity', 'plot_velocity', 'plot_minmaxvelo')),
    ('n', Varr('Viscosity', 'plot_viscosity', 'plot_minmaxvisco')),
    ('c', Varr('Concentration', 'plot_concentration', 'plot_minmaxcon')),
    ('g', Varr('Grid', 'plot_grid', None)),
    ('a', Varr('Advection', 'plot_advection', None)),
    ('e', Varr('Energy', 'plot_energy', None)),
    ('h', Varr('Concentration Theo', 'plot_conctheo', None)),
    ('i', Varr('Init overturn', 'plot_overturn_init', None)),
    ('d', Varr('Difference', 'plot_difference', None)),
    ))
