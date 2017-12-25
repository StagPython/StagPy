# PYTHON_ARGCOMPLETE_OK
"""The stagpy module is callable"""

import importlib
import signal
import sys


def sigint_handler(*_):
    """SIGINT handler"""
    print('\nSo long, and thanks for all the fish.')
    sys.exit()


def main():
    """StagPy entry point"""
    signal.signal(signal.SIGINT, sigint_handler)
    config = importlib.import_module('stagpy.config')
    error = importlib.import_module('stagpy.error')
    try:
        args = config.parse_args()
        args.func(args)
    except error.StagpyError as err:
        print('Oops! StagPy encountered the following problem while '
              'processing your request.',
              'Please check the path to your simulation and the command line '
              'arguments.', '',
              '{}: {}'.format(err.__class__.__name__, err),
              sep='\n')
        sys.exit()

if __name__ == '__main__':
    main()
