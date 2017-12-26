"""define current version"""

from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound
import importlib
import signal
import sys
from . import config


def sigint_handler(*_):
    """SIGINT handler"""
    print('\nSo long, and thanks for all the fish.')
    sys.exit()


def _load_mpl():
    """Load matplotlib and set some configuration"""
    mpl = importlib.import_module('matplotlib')
    if conf.core.matplotback:
        mpl.use(conf.core.matplotback)
    plt = importlib.import_module('matplotlib.pyplot')
    if conf.core.useseaborn:
        importlib.import_module('seaborn')
    if conf.core.xkcd:
        plt.xkcd()


_PREV_INT = signal.signal(signal.SIGINT, sigint_handler)

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stagpy').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'

conf = config.StagpyConfiguration()

_load_mpl()

signal.signal(signal.SIGINT, _PREV_INT)
