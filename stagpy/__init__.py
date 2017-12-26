"""define current version"""

from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound
import importlib
from . import config

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stagpy').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'

conf = config.StagpyConfiguration()

mpl = importlib.import_module('matplotlib')
if conf.core.matplotback:
    mpl.use(conf.core.matplotback)
plt = importlib.import_module('matplotlib.pyplot')
if conf.core.useseaborn:
    importlib.import_module('seaborn')
if conf.core.xkcd:
    plt.xkcd()
