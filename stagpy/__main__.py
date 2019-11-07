"""The stagpy module is callable."""

import importlib
import signal
import sys
import warnings

from . import DEBUG, sigint_handler


def main():
    """Implement StagPy entry point."""
    if not DEBUG:
        signal.signal(signal.SIGINT, sigint_handler)
        warnings.simplefilter('ignore')
    args = importlib.import_module('stagpy.args')
    error = importlib.import_module('stagpy.error')
    try:
        args.parse_args()()
    except error.StagpyError as err:
        if DEBUG:
            raise
        print('Oops! StagPy encountered the following problem while '
              'processing your request.',
              'Please check the path to your simulation and the command line '
              'arguments.', '',
              '{}: {}'.format(err.__class__.__name__, err),
              sep='\n', file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
