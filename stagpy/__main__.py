# PYTHON_ARGCOMPLETE_OK
"""The stagpy module is callable"""

import signal
import sys
from . import config


def sigint_handler(*_):
    """SIGINT handler"""
    print('\nSo long, and thanks for all the fish.')
    sys.exit()


def main():
    """StagPy entry point"""
    signal.signal(signal.SIGINT, sigint_handler)
    args = config.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
