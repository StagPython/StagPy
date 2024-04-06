"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at
https://stagpy.readthedocs.io/en/stable/

When using the CLI interface, if the environment variable STAGPY_DEBUG is set
to a truthy value, warnings are issued normally and StagpyError are raised.
Otherwise, warnings are ignored and only a short form of encountered
StagpyErrors is printed.

Truthy values for environment variables are 'true', 't', 'yes', 'y', 'on', '1',
and uppercase versions of those.
"""

from __future__ import annotations

import os
import signal
import sys
import typing

from . import config

if typing.TYPE_CHECKING:
    from typing import Any, NoReturn


def _env(var: str) -> bool:
    """Return whether var is set to True."""
    val = os.getenv(var, default="").lower()
    return val in ("true", "t", "yes", "y", "on", "1")


DEBUG = _env("STAGPY_DEBUG")


def sigint_handler(*_: Any) -> NoReturn:
    """Handler of SIGINT signal.

    It is set when you use StagPy as a command line tool to handle gracefully
    keyboard interruption.
    """
    print("\nSo long, and thanks for all the fish.")
    sys.exit()


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
if config.CONFIG_LOCAL.is_file():
    conf.update_from_file_(config.CONFIG_LOCAL)

if not DEBUG:
    signal.signal(signal.SIGINT, _PREV_INT)
