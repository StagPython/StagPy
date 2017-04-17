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
    args = config.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
