# PYTHON_ARGCOMPLETE_OK
"""The stagpy module is callable"""

import importlib
import signal
import sys
import warnings
from . import conf, sigint_handler


def main():
    """StagPy entry point"""
    prev_int = signal.signal(signal.SIGINT, sigint_handler)
    warnings.simplefilter('ignore')
    args = importlib.import_module('stagpy.args')
    error = importlib.import_module('stagpy.error')
    try:
        func = args.parse_args()
        if conf.common.debug:
            signal.signal(signal.SIGINT, prev_int)
            warnings.simplefilter('default')
        func()
    except error.StagpyError as err:
        if conf.common.debug:
            raise
        print('Oops! StagPy encountered the following problem while '
              'processing your request.',
              'Please check the path to your simulation and the command line '
              'arguments.', '',
              '{}: {}'.format(err.__class__.__name__, err),
              sep='\n')
        sys.exit()

if __name__ == '__main__':
    main()
