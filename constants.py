"""defines some constants"""

from collections import OrderedDict, namedtuple
from misc import takefield, calc_stream

Var = namedtuple('Var', ['name', 'func'])

varlist = OrderedDict((
    ('t', Var('temperature', takefield(0))),
    ('p', Var('pressure', takefield(3))),
    ('s', Var('stream function', calc_stream))
    ))

default_config = OrderedDict((
    ('geometry', 'annulus'),
    ('path', './'),
    ('name', 'test'),
    ('timestep', 100),
    ('plot', 'tps'),
    ('dsa', 0.1),
    ('shrinkcb', 0.5)
    ))
