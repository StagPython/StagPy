"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at
https://stagpy.readthedocs.io/en/stable/

If the environment variable STAGPY_ISOLATED is set to a truthy value, StagPy
does not attempt to read any configuration file (including mplstyle).

When using the CLI interface, if the environment variable STAGPY_DEBUG is set
to a truthy value, warnings are issued normally and StagpyError are raised.
Otherwise, warnings are ignored and only a short form of encountered
StagpyErrors is printed.

Truthy values for environment variables are 'true', 't', 'yes', 'y', 'on', '1',
and uppercase versions of those.
"""

from __future__ import annotations
import importlib.resources as imlr
import os
import shutil
import signal
import sys
import typing

from setuptools_scm import get_version
from loam.manager import ConfigurationManager

from . import config, _styles

if typing.TYPE_CHECKING:
    from typing import NoReturn, Any, Iterator


def _env(var: str) -> bool:
    """Return whether var is set to True."""
    val = os.getenv(var, default='').lower()
    return val in ('true', 't', 'yes', 'y', 'on', '1')


DEBUG = _env('STAGPY_DEBUG')
ISOLATED = _env('STAGPY_ISOLATED')


def sigint_handler(*_: Any) -> NoReturn:
    """Handler of SIGINT signal.

    It is set when you use StagPy as a command line tool to handle gracefully
    keyboard interruption.
    """
    print('\nSo long, and thanks for all the fish.')
    sys.exit()


def _iter_styles() -> Iterator[str]:
    for resource in imlr.contents(_styles):
        if resource.endswith(".mplstyle"):
            yield resource


def _check_config() -> None:
    """Create config files as necessary."""
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    verfile = config.CONFIG_DIR / '.version'
    uptodate = verfile.is_file() and verfile.read_text() == __version__
    if not uptodate:
        verfile.write_text(__version__)
    if not (uptodate and config.CONFIG_FILE.is_file()):
        conf.create_config_(update=True)
    for stfile in _iter_styles():
        stfile_conf = config.CONFIG_DIR / stfile
        if not (uptodate and stfile_conf.is_file()):
            with imlr.path(_styles, stfile) as stfile_local:
                shutil.copy(str(stfile_local), str(stfile_conf))


def load_mplstyle() -> None:
    """Try to load conf.plot.mplstyle matplotlib style."""
    import matplotlib.style as mpls
    if conf.plot.mplstyle:
        for style in conf.plot.mplstyle.split():
            style_fname = style + ".mplstyle"
            if not ISOLATED:
                stfile = config.CONFIG_DIR / style_fname
                if stfile.is_file():
                    mpls.use(str(stfile))
                    continue
            # try packaged version
            if imlr.is_resource(_styles, style_fname):
                with imlr.path(_styles, style_fname) as stfile:
                    mpls.use(str(stfile))
                    continue
            mpls.use(style)
    if conf.plot.xkcd:
        import matplotlib.pyplot as plt
        plt.xkcd()


if DEBUG:
    print('StagPy runs in DEBUG mode because the environment variable',
          'STAGPY_DEBUG is set to "True"', sep='\n', end='\n\n')
else:
    _PREV_INT = signal.signal(signal.SIGINT, sigint_handler)

try:
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown"

_CONF_FILES = ([config.CONFIG_FILE, config.CONFIG_LOCAL]
               if not ISOLATED else [])
conf = ConfigurationManager.from_dict_(config.CONF_DEF)
conf.set_config_files_(*_CONF_FILES)
if not ISOLATED:
    _check_config()
PARSING_OUT = conf.read_configs_()

load_mplstyle()

if not DEBUG:
    signal.signal(signal.SIGINT, _PREV_INT)
