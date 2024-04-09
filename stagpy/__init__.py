"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at
https://stagpy.readthedocs.io/en/stable/

When using the CLI interface, warnings are ignored and only a short form of
encountered StagpyErrors is printed. Set the environment variable STAGPY_DEBUG
to issue warnings normally and raise StagpyError.
"""

from __future__ import annotations

import os
import signal
import sys
import typing

if typing.TYPE_CHECKING:
    from typing import Any, NoReturn


DEBUG = os.getenv("STAGPY_DEBUG") is not None


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

if not DEBUG:
    signal.signal(signal.SIGINT, _PREV_INT)
