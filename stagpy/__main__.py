# PYTHON_ARGCOMPLETE_OK
"""The stagpy module is callable"""

import importlib
import signal
import sys
from . import sigint_handler


def main():
    """StagPy entry point"""
    signal.signal(signal.SIGINT, sigint_handler)
    args = importlib.import_module('stagpy.args')
    error = importlib.import_module('stagpy.error')
    try:
        args.parse_args()()
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
