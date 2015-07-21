"""defines some constants"""

from collections import OrderedDict, namedtuple

import misc
import processing

Var = namedtuple('Var', ['par', 'name', 'func'])

varlist = OrderedDict((
    ('t', Var('t', 'temperature', misc.takefield(0))),
    ('p', Var('vp', 'pressure', misc.takefield(3))),
    ('s', Var('vp', 'stream function', processing.calc_stream))
    ))

default_config = OrderedDict((
    ('geometry', 'annulus'),
    ('path', './'),
    ('name', 'test'),
    ('timestep', '100'),
    ('plot', 'tps'),
    ('dsa', 0.1),
    ('shrinkcb', 0.5)
    ))
