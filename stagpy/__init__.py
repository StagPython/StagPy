"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at
http://stagpy.readthedocs.io/en/stable/
"""

from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound
import importlib
import signal
import sys
from . import config


def sigint_handler(*_):
    """Handler of SIGINT signal.

    It is set when you use StagPy as a command line tool to handle gracefully
    keyboard interruption.
    """
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


def init_config(config_file=config.CONFIG_FILE):
    """Initialize configuration :data:`stagpy.conf`.

    It is automatically called whenever the :mod:`stagpy` module is
    imported. You can use this function if you want to reset the StagPy
    configuration.

    Args:
        config_file (pathlike): the path of a config file. Set this parameter
            to None if you do not want to use any config file.
    """
    global conf  # pylint:disable=global-variable-undefined,invalid-name
    conf = config.StagpyConfiguration(config_file)


_PREV_INT = signal.signal(signal.SIGINT, sigint_handler)

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stagpy').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'

init_config()
_load_mpl()

signal.signal(signal.SIGINT, _PREV_INT)
