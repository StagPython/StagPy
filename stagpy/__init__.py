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

from . import _styles, config

if typing.TYPE_CHECKING:
    from typing import Any, Iterator, NoReturn


def _env(var: str) -> bool:
    """Return whether var is set to True."""
    val = os.getenv(var, default="").lower()
    return val in ("true", "t", "yes", "y", "on", "1")


DEBUG = _env("STAGPY_DEBUG")
ISOLATED = _env("STAGPY_ISOLATED")


def sigint_handler(*_: Any) -> NoReturn:
    """Handler of SIGINT signal.

    It is set when you use StagPy as a command line tool to handle gracefully
    keyboard interruption.
    """
    print("\nSo long, and thanks for all the fish.")
    sys.exit()


def _iter_styles() -> Iterator[str]:
    for resource in imlr.contents(_styles):
        if resource.endswith(".mplstyle"):
            yield resource


def _check_config() -> None:
    """Create config files as necessary."""
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    verfile = config.CONFIG_DIR / ".version"
    uptodate = verfile.is_file() and verfile.read_text() == __version__
    if not uptodate:
        verfile.write_text(__version__)
    if not (uptodate and config.CONFIG_FILE.is_file()):
        conf.to_file_(config.CONFIG_FILE)
    for stfile in _iter_styles():
        stfile_conf = config.CONFIG_DIR / stfile
        if not (uptodate and stfile_conf.is_file()):
            with imlr.path(_styles, stfile) as stfile_local:
                shutil.copy(str(stfile_local), str(stfile_conf))


if DEBUG:
    print(
        "StagPy runs in DEBUG mode because the environment variable",
        'STAGPY_DEBUG is set to "True"',
        sep="\n",
        end="\n\n",
    )
else:
    _PREV_INT = signal.signal(signal.SIGINT, sigint_handler)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

conf = config.Config.default_()
if not ISOLATED:
    _check_config()
    conf.update_from_file_(config.CONFIG_FILE)
    if config.CONFIG_LOCAL.is_file():
        conf.update_from_file_(config.CONFIG_LOCAL)

if not DEBUG:
    signal.signal(signal.SIGINT, _PREV_INT)
