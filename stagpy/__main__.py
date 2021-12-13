"""The stagpy module is callable."""

import signal
import sys
import warnings

from . import DEBUG, sigint_handler


def main() -> None:
    """Implement StagPy entry point."""
    if not DEBUG:
        signal.signal(signal.SIGINT, sigint_handler)
        warnings.simplefilter('ignore')
    from . import args, error
    try:
        args.parse_args()()
    except error.StagpyError as err:
        if DEBUG:
            raise
        errtype = type(err).__name__
        print('Oops! StagPy encountered the following problem while '
              'processing your request.', 'Please check the path to your '
              'simulation and the command line arguments.', '',
              f'{errtype}: {err}', sep='\n', file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
