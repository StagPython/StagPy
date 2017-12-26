"""define current version"""

from setuptools_scm import get_version
from pkg_resources import get_distribution, DistributionNotFound
import importlib

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    __version__ = get_distribution('stagpy').version
except (DistributionNotFound, ValueError):
    __version__ = 'unknown'

config = importlib.import_module('stagpy.config')
conf = config.StagpyConfiguration()
