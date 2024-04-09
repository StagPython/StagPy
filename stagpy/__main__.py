"""The stagpy module is callable."""

from __future__ import annotations

import os
import signal
import sys
import warnings
from typing import NoReturn


def sigint_handler(*_: object) -> NoReturn:
    """Handler of SIGINT signal.

    It is set when you use StagPy as a command line tool to handle gracefully
    keyboard interruption.
    """
    print("\nSo long, and thanks for all the fish.")
    sys.exit()


def main() -> None:
    """Implement StagPy entry point."""
    debug = os.getenv("STAGPY_DEBUG") is not None
    if debug:
        print(
            "env variable 'STAGPY_DEBUG' is set: StagPy runs in DEBUG mode",
            end="\n\n",
        )
    else:
        signal.signal(signal.SIGINT, sigint_handler)
        warnings.simplefilter("ignore")

    from . import args, config, error

    conf = config.Config.default_()
    if config.CONFIG_LOCAL.is_file():
        conf.update_from_file_(config.CONFIG_LOCAL)

    try:
        args.parse_args(conf)(conf)
    except error.StagpyError as err:
        if debug:
            raise
        errtype = type(err).__name__
        print(
            "Oops! StagPy encountered the following problem while "
            "processing your request.",
            "Please check the path to your "
            "simulation and the command line arguments.",
            "",
            f"{errtype}: {err}",
            sep="\n",
            file=sys.stderr,
        )
        sys.exit()


if __name__ == "__main__":
    main()
