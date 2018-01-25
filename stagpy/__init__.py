"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at
http://stagpy.readthedocs.io/en/stable/

If the environment variable STAGPY_NO_CONFIG is set to 'True', StagPy does not
attempt to read any configuration file.
"""

from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound
import importlib
import os
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
    if conf.plot.matplotback:
        mpl.use(conf.plot.matplotback)
    plt = importlib.import_module('matplotlib.pyplot')
    if conf.plot.useseaborn:
        sns = importlib.import_module('seaborn')
        sns.set()
    if conf.plot.xkcd:
        plt.xkcd()


_PREV_INT = signal.signal(signal.SIGINT, sigint_handler)

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stagpy').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'

_CONF_FILE = config.CONFIG_FILE\
    if os.getenv('STAGPY_NO_CONFIG') != 'True' else None
# pylint: disable=invalid-name
conf = config.StagpyConfiguration(_CONF_FILE)
# pylint: enable=invalid-name

_load_mpl()

signal.signal(signal.SIGINT, _PREV_INT)
