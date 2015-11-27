"""defines some constants"""

from collections import OrderedDict, namedtuple

import misc
import processing

Var = namedtuple('Var', ['par', 'name', 'func'])

VAR_LIST = OrderedDict((
    ('t', Var('t', 'Temperature', misc.takefield(0))),
    ('c', Var('c', 'Composition', misc.takefield(0))),
    ('v', Var('eta', 'Viscosity', misc.takefield(0))),
    ('d', Var('rho', 'Density', misc.takefield(0))),
    ('p', Var('vp', 'Pressure', misc.takefield(3))),
    ('s', Var('vp', 'Stream function', processing.calc_stream))
    ))

DEFAULT_CONFIG = OrderedDict((
    ('geometry', 'annulus'),
    ('path', './'),
    ('name', 'test'),
    ('timestep', '100'),
    ('plot', 'tps'),
    ('dsa', 0.05),
    ('shrinkcb', 0.5),
    ('linewidth',2),
    ('fontsize', 16)
    ))
